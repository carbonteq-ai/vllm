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
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


UNO_DRAFT_ADAPTER_ID = 2_147_483_000


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
            lora_int_id=UNO_DRAFT_ADAPTER_ID,
            lora_path=spec.uno_adapter or "",
        )
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        self._uno_policy_lora_ids = torch.zeros(
            self.max_batch_size, dtype=torch.int64, device=device
        )
        width = self.num_speculative_tokens
        self._uno_query_start_loc = (
            torch.arange(self.max_batch_size + 1, dtype=torch.int32, device=device)
            * width
        )
        self._uno_query_start_loc_cpu = (
            torch.arange(
                self.max_batch_size + 1,
                dtype=torch.int32,
                device="cpu",
                pin_memory=PIN_MEMORY,
            )
            * width
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
        self.runner.reserve_lora(self.uno_adapter_request.lora_int_id)
        self.runner.enable_system_lora_overlay(
            self.uno_adapter_request.lora_int_id
        )

    @override
    def _system_lora_mask(self, num_input_tokens: int) -> torch.Tensor | None:
        assert self.is_masked_token_mask is not None
        return self.is_masked_token_mask[:num_input_tokens]

    @override
    def model_returns_tuple(self) -> bool:
        return False

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
        request_lora_ids = self.runner.input_batch.request_lora_mapping[
            : self.runner.input_batch.num_reqs
        ]
        composes_request_lora = self.speculative_config.uno_composes_request_lora
        if np.any(request_lora_ids > 0) and not composes_request_lora:
            raise ValueError(
                "Uno request LoRA requires uno_composes_request_lora=true and "
                "a policy-plus-Uno composite adapter"
            )
        if composes_request_lora and np.any(request_lora_ids <= 0):
            raise ValueError(
                "Uno composite mode requires one request/policy LoRA for every sequence"
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
        query_start_loc = self._uno_query_start_loc[: batch_size + 1]
        metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            seq_lens=expanded_metadata.seq_lens,
            query_start_loc_cpu=self._uno_query_start_loc_cpu[: batch_size + 1],
            seq_lens_cpu_upper_bound=expanded_metadata.seq_lens_cpu_upper_bound,
            num_reqs=expanded_metadata.num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=self.num_speculative_tokens,
            max_seq_len=expanded_metadata.max_seq_len,
            block_table_tensor=expanded_metadata.block_table_tensor,
            slot_mapping=compact_slot_mapping,
            causal=True,
        )
        sample_indices = self.arange[:num_tokens]

        if self.speculative_config.uno_noise_mode == "random_uniform":
            # Every request has one clean seed and width-1 noise rows. Deriving
            # this known shape avoids a device reduction followed by .item(),
            # which otherwise synchronizes CPU and GPU once per proposal.
            width = self.num_speculative_tokens
            count = batch_size * (width - 1)
            mask_token_id = self.speculative_config.uno_mask_token_id
            assert mask_token_id is not None
            noise = torch.randint(
                1,
                mask_token_id,
                (count,),
                dtype=self.input_ids.dtype,
                device=self.input_ids.device,
            )
            self.input_ids[:num_tokens].view(batch_size, width)[:, 1:].copy_(
                noise.view(batch_size, width - 1)
            )

        active_loras = {self.uno_adapter_request}
        if composes_request_lora:
            self._uno_policy_lora_ids[:batch_size].copy_(
                torch.as_tensor(
                    request_lora_ids,
                    dtype=torch.int64,
                    device=noise_mask.device,
                )
            )
            policy_ids = self._uno_policy_lora_ids[:batch_size]
            expanded_policy_ids = policy_ids.repeat_interleave(
                self.num_speculative_tokens
            )
            token_mapping = tuple(expanded_policy_ids.cpu().tolist())
            prompt_mapping = tuple(
                expanded_policy_ids[sample_indices.long()].cpu().tolist()
            )
            for policy_id in np.unique(request_lora_ids):
                request = self.runner.input_batch.lora_id_to_lora_request.get(
                    int(policy_id)
                )
                if request is None:
                    raise ValueError(
                        f"Uno composite mode cannot resolve policy LoRA {policy_id}"
                    )
                active_loras.add(request)
        else:
            # Uno is applied as a position-gated system overlay. Punica routing
            # remains the independent request/policy channel.
            prompt_mapping = (0,) * num_tokens
            token_mapping = (0,) * num_tokens
        self.runner._set_active_loras(
            prompt_mapping,
            token_mapping,
            active_loras,
            metadata_bank="uno",
        )
        return num_tokens, sample_indices, metadata
