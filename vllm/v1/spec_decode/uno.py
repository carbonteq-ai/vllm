# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uno Psi-Spec proposer backed by the target model and a gated LoRA.

The two-pass draft shape follows IFM's Uno reference implementation: the
already sampled target token is the causal seed, future positions contain
noise, and the Uno adapter is active only for those noise positions. Target
verification and rejection sampling remain owned by vLLM.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


_UNO_ADAPTER_ID = 2_147_483_000


class UnoProposer(SpecDecodeBaseProposer):
    """Run one parallel Uno draft with shared target weights and KV cache."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: "GPUModelRunner",
    ) -> None:
        spec = vllm_config.speculative_config
        assert spec is not None and spec.use_uno()
        if runner.lora_config is None:
            raise ValueError("Uno speculative decoding requires --enable-lora")
        if runner.lora_config.max_loras < 2:
            raise ValueError(
                "Uno speculative decoding requires --max-loras 2 or greater "
                "so startup profiling can coexist with the pinned Uno adapter"
            )
        self.runner = runner
        self.uno_adapter_request = LoRARequest(
            lora_name="__vllm_uno_draft__",
            lora_int_id=_UNO_ADAPTER_ID,
            lora_path=spec.uno_adapter or "",
        )
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )

    @override
    def _init_parallel_drafting_params(self) -> None:
        mask_token_id = self.speculative_config.uno_mask_token_id
        assert mask_token_id is not None
        self.parallel_drafting_token_id = mask_token_id
        self.parallel_drafting_hidden_state_tensor = None

    @override
    def load_model(self, target_model: nn.Module) -> None:
        """Share target parameters and register the draft-only Uno adapter."""
        self.model = target_model
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        self._draft_attn_layer_names = {
            name
            for name, layer in all_attn_layers.items()
            if layer.get_kv_cache_spec(self.vllm_config) is not None
        }
        if not self._draft_attn_layer_names:
            raise ValueError("Uno requires at least one target attention layer")
        self.runner.add_lora(self.uno_adapter_request)
        self.runner.pin_lora(self.uno_adapter_request.lora_int_id)

    @override
    def model_returns_tuple(self) -> bool:
        return False

    @staticmethod
    def _build_lora_mappings(
        noise_mask: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        adapter_id: int = _UNO_ADAPTER_ID,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return output and input mappings for position-gated Uno LoRA."""
        token_noise = noise_mask.detach().to(device="cpu", dtype=torch.bool).tolist()
        sample_noise = (
            noise_mask[token_indices_to_sample.long()]
            .detach()
            .to(device="cpu", dtype=torch.bool)
            .tolist()
        )
        token_mapping = tuple(adapter_id if enabled else 0 for enabled in token_noise)
        prompt_mapping = tuple(adapter_id if enabled else 0 for enabled in sample_noise)
        return prompt_mapping, token_mapping

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        if np.any(
            self.runner.input_batch.request_lora_mapping[
                : self.runner.input_batch.num_reqs
            ]
            > 0
        ):
            raise ValueError(
                "Uno with a request/policy LoRA is not qualified yet; use a "
                "full-weight target policy"
            )

        _, expanded_sample_indices, expanded_metadata = super().set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
        assert self.is_masked_token_mask is not None
        sample_rows = expanded_sample_indices.long()
        num_tokens = sample_rows.numel()

        # The shared target model has already written the scheduled target
        # tokens into its KV cache. Uno must therefore run only the newly
        # target-sampled seed followed by the diffusion-noise rows. Replaying
        # the target query here duplicates the prefix and changes q(draft).
        compact_input_ids = self.input_ids.index_select(0, sample_rows)
        compact_positions = self.positions.index_select(0, sample_rows)
        compact_slot_mapping = expanded_metadata.slot_mapping.index_select(
            0, sample_rows
        )
        noise_mask = self.is_masked_token_mask.index_select(0, sample_rows)
        self.input_ids[:num_tokens].copy_(compact_input_ids)
        self.positions[:num_tokens].copy_(compact_positions)
        self.is_masked_token_mask[:num_tokens].copy_(noise_mask)

        batch_size = cad.batch_size()
        query_start_loc = self.arange[: batch_size + 1] * self.num_speculative_tokens
        metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            seq_lens=expanded_metadata.seq_lens,
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * self.num_speculative_tokens
            ),
            seq_lens_cpu_upper_bound=expanded_metadata.seq_lens_cpu_upper_bound,
            num_reqs=expanded_metadata.num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=self.num_speculative_tokens,
            max_seq_len=expanded_metadata.max_seq_len,
            block_table_tensor=expanded_metadata.block_table_tensor,
            slot_mapping=compact_slot_mapping,
            causal=True,
        )
        sample_indices = torch.arange(num_tokens, dtype=torch.int32, device=self.device)

        if self.speculative_config.uno_noise_mode == "random_uniform":
            count = int(noise_mask.sum().item())
            mask_token_id = self.speculative_config.uno_mask_token_id
            assert mask_token_id is not None
            self.input_ids[:num_tokens][noise_mask] = torch.randint(
                1,
                mask_token_id,
                (count,),
                dtype=self.input_ids.dtype,
                device=self.input_ids.device,
            )

        prompt_mapping, token_mapping = self._build_lora_mappings(
            noise_mask, sample_indices
        )
        self.runner._set_active_loras(
            prompt_mapping,
            token_mapping,
            {self.uno_adapter_request},
        )
        return num_tokens, sample_indices, metadata
