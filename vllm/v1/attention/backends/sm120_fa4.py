# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional paged FlashAttention 4 backend for consumer Blackwell.

The kernel lives in the independent ``sm120-paged-attention`` distribution.
vLLM retains ownership of KV-cache writes, metadata, output buffers, backend
selection, and fallback.
"""

from __future__ import annotations

import copy
import importlib.util
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import KVCacheSpec


def _kernel_package_available() -> bool:
    return importlib.util.find_spec("sm120_paged_attention") is not None


def _deterministic_kernel_options(
    *, max_seqlen_q: int, batch_invariant: bool
) -> tuple[int, tuple[int, int] | None]:
    """Return the qualified SM120 reduction policy for this query shape."""
    tile_mn = (64, 64) if batch_invariant and max_seqlen_q > 1 else None
    return (1 if batch_invariant else 0, tile_mn)


class Sm120FA4Backend(AttentionBackend):
    """Paged BF16/FP16 causal attention using the extracted SM120 kernel."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name() -> str:
        return "SM120_FA4"

    @staticmethod
    def get_impl_cls() -> type[Sm120FA4Impl]:
        return Sm120FA4Impl

    @staticmethod
    def get_builder_cls() -> type[Sm120FA4MetadataBuilder]:
        return Sm120FA4MetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 96, 128, 256]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return (capability.major, capability.minor) in {(12, 0), (12, 1)}

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        del (
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )
        if has_sink:
            return "SM120_FA4 does not yet support attention sinks"
        if not _kernel_package_available():
            return (
                "Install the optional sm120-paged-attention kernel package "
                "to use SM120_FA4"
            )
        return None


@dataclass
class Sm120FA4Metadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True


class Sm120FA4MetadataBuilder(AttentionMetadataBuilder[Sm120FA4Metadata]):
    supports_update_block_table = True

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        del vllm_config, kv_cache_spec
        # The paged kernel writes only to caller-owned output and KV buffers.
        # Its launch plan is shape-specialized and safe to replay in vLLM's
        # target-model graphs. Uno proposal execution remains eager and is not
        # captured through this capability.
        return AttentionCGSupport.ALWAYS

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Sm120FA4Metadata:
        del common_prefix_len, fast_build
        common = common_attn_metadata
        return Sm120FA4Metadata(
            num_actual_tokens=common.num_actual_tokens,
            max_query_len=common.max_query_len,
            query_start_loc=common.query_start_loc,
            max_seq_len=common.max_seq_len,
            seq_lens=common.seq_lens,
            block_table=common.block_table_tensor,
            slot_mapping=common.slot_mapping,
            causal=common.causal,
        )

    def update_block_table(
        self,
        metadata: Sm120FA4Metadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> Sm120FA4Metadata:
        updated = copy.copy(metadata)
        updated.block_table = blk_table
        updated.slot_mapping = slot_mapping
        return updated


class Sm120FA4Impl(AttentionImpl[Sm120FA4Metadata]):
    can_return_lse_for_decode = False
    supports_dcp = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if not _kernel_package_available():
            raise ImportError("SM120_FA4 requires the sm120-paged-attention package.")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("SM120_FA4 supports decoder attention only.")
        if alibi_slopes is not None:
            raise NotImplementedError("SM120_FA4 does not support ALiBi.")
        if sinks is not None:
            raise NotImplementedError("SM120_FA4 does not support attention sinks.")
        if logits_soft_cap not in (None, 0):
            raise NotImplementedError("SM120_FA4 does not support logits soft cap.")
        if kv_cache_dtype not in {"auto", "float16", "bfloat16"}:
            raise NotImplementedError(
                "SM120_FA4 supports only auto, float16, and bfloat16 KV cache."
            )
        if num_heads % num_kv_heads:
            raise ValueError("SM120_FA4 requires Q heads divisible by KV heads.")
        if self.total_cp_world_size > 1:
            raise NotImplementedError("SM120_FA4 does not yet support DCP or PCP.")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.window = (-1, -1) if sliding_window is None else (sliding_window - 1, 0)
        self.supports_quant_query_input = False

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Sm120FA4Metadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("SM120_FA4 does not fuse output quantization.")
        if attn_metadata is None:
            return output.fill_(0)

        from sm120_paged_attention import flash_attn_with_kvcache

        num_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        query_view = query[:num_tokens]
        output_view = output[:num_tokens].view(
            num_tokens, self.num_heads, self.head_size
        )
        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        num_reqs = attn_metadata.seq_lens.shape[0]
        if max_seqlen_q > 0 and num_tokens == num_reqs * max_seqlen_q:
            # Match the qualified SM120 block-decode contract. Shape-based
            # dispatch avoids a device synchronization and retains the same
            # paged KV metadata.
            query_view = query_view.view(
                num_reqs, max_seqlen_q, self.num_heads, self.head_size
            )
            output_view = output_view.view(
                num_reqs, max_seqlen_q, self.num_heads, self.head_size
            )
            cu_seqlens_q = None
        num_splits, tile_mn = _deterministic_kernel_options(
            max_seqlen_q=max_seqlen_q,
            batch_invariant=envs.VLLM_BATCH_INVARIANT,
        )
        flash_attn_with_kvcache(
            q=query_view,
            k_cache=key_cache,
            v_cache=value_cache,
            out=output_view,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cache_seqlens=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            page_table=attn_metadata.block_table,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            window_size=self.window,
            num_splits=num_splits,
            # SM120's tuned single-sequence prefill may choose a different
            # K-reduction tile than a multi-sequence prefill. Both are
            # numerically valid, but the different reduction order writes
            # different historical KV states and violates vLLM's batch-
            # invariant contract. Freeze the qualified M64N64 prefill tile;
            # decode keeps its separately qualified fast path.
            tile_mn=tile_mn,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        from vllm._custom_ops import reshape_and_cache_flash

        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
