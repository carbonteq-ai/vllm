# SPDX-License-Identifier: Apache-2.0
"""LFM2 targets and LFM2 DSpark drafters (CarbonTeq)."""

import pytest

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.model_executor.models.lfm2 import Lfm2ForCausalLM, Lfm2Model

TARGET = "LiquidAI/LFM2.5-2.6B"
TARGET_REVISION = "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"
DRAFT = "LiquidAI/LFM2.5-2.6B-DSpark"
DRAFT_REVISION = "458cedab07d0f7b2b05700c77e1aa463d43d6f04"


def test_lfm2_exposes_the_eagle3_aux_hidden_state_interface():
    assert supports_eagle3(Lfm2ForCausalLM)
    assert hasattr(Lfm2Model, "_maybe_add_hidden_state")
    assert hasattr(Lfm2Model, "_set_aux_hidden_state_layers")


@pytest.mark.skip_global_cleanup
def test_lfm2_dspark_drafter_maps_to_qwen3_dspark_with_interleaved_rope():
    target = ModelConfig(TARGET, revision=TARGET_REVISION, max_model_len=4096)
    speculative = SpeculativeConfig(
        method="dspark",
        model=DRAFT,
        revision=DRAFT_REVISION,
        num_speculative_tokens=9,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
    )
    draft_hf_config = speculative.draft_model_config.hf_config
    assert draft_hf_config.architectures == ["Qwen3DSparkModel"]
    # The LFM2 drafter uses interleaved (non-neox) RoPE.
    assert draft_hf_config.is_neox_style is False
    # Its confidence head reads hidden + Markov features (2048 + 256 inputs).
    assert draft_hf_config.confidence_head_with_markov is True
    # The drafter reads target layers 2, 9, 17, 21 and 27, i.e. the outputs
    # captured at indices 3, 10, 18, 22 and 28.
    assert draft_hf_config.dflash_config["target_layer_ids"] == [2, 9, 17, 21, 27]


@pytest.mark.skip_global_cleanup
def test_dspark_keeps_its_exact_trailing_prefix_cache_block():
    """DSpark KV at position i depends only on the target state at i."""
    target = ModelConfig(TARGET, revision=TARGET_REVISION, max_model_len=4096)
    speculative = SpeculativeConfig(
        method="dspark",
        model=DRAFT,
        revision=DRAFT_REVISION,
        num_speculative_tokens=9,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
    )
    assert speculative.use_eagle()
    assert not speculative.use_eagle_block_drop()
