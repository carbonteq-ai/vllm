# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from vllm.config.speculative import SpeculativeConfig


def _target_configs() -> tuple[Mock, Mock]:
    model = Mock()
    parallel = Mock()
    parallel.tensor_parallel_size = 1
    parallel.pipeline_parallel_size = 1
    return model, parallel


def test_uno_reuses_target_model_and_reserves_noise_slots() -> None:
    model, parallel = _target_configs()

    config = SpeculativeConfig(
        method="uno",
        num_speculative_tokens=7,
        uno_adapter="IFM/K2-Horizon-7B-Uno",
        uno_adapter_revision="ec92bbd768f4a404319625204544782e3377bcd7",
        uno_mask_token_id=250624,
        target_model_config=model,
        target_parallel_config=parallel,
    )

    assert config.use_uno()
    assert config.model is None
    assert config.draft_model_config is model
    assert config.draft_parallel_config is parallel
    assert config.max_num_new_slots_for_drafting == 7
    assert config.enforce_eager is True




def test_uno_proposer_remains_eager_when_false_is_requested() -> None:
    model, parallel = _target_configs()

    config = SpeculativeConfig(
        method="uno",
        num_speculative_tokens=7,
        uno_adapter="IFM/K2-Horizon-7B-Uno",
        uno_mask_token_id=250624,
        enforce_eager=False,
        target_model_config=model,
        target_parallel_config=parallel,
    )

    assert config.enforce_eager is True


def test_uno_rejects_out_of_vocab_mask_before_gpu_submission() -> None:
    model, parallel = _target_configs()
    model.get_vocab_size.return_value = 250624

    with pytest.raises(ValidationError, match="inside the target embedding vocabulary"):
        SpeculativeConfig(
            method="uno",
            num_speculative_tokens=7,
            uno_adapter="IFM/K2-Horizon-7B-Uno",
            uno_mask_token_id=250624,
            uno_noise_mode="mask",
            target_model_config=model,
            target_parallel_config=parallel,
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({}, "requires a non-empty uno_adapter"),
        (
            {"model": "IFM/K2-Horizon-7B-Uno"},
            "pass uno_adapter instead of speculative_config.model",
        ),
    ],
)
def test_uno_rejects_ambiguous_adapter_configuration(
    kwargs: dict[str, str], match: str
) -> None:
    model, parallel = _target_configs()

    with pytest.raises(ValidationError, match=match):
        SpeculativeConfig(
            method="uno",
            num_speculative_tokens=7,
            target_model_config=model,
            target_parallel_config=parallel,
            uno_mask_token_id=250624,
            **kwargs,
        )


def test_uno_fields_are_rejected_for_other_methods() -> None:
    model, parallel = _target_configs()

    with pytest.raises(ValidationError, match="require method='uno'"):
        SpeculativeConfig(
            method="ngram",
            model="ngram",
            num_speculative_tokens=7,
            uno_adapter="IFM/K2-Horizon-7B-Uno",
            uno_mask_token_id=250624,
            target_model_config=model,
            target_parallel_config=parallel,
        )


@pytest.mark.parametrize(
    "field",
    ["tensor_parallel_size", "pipeline_parallel_size"],
)
def test_uno_initially_rejects_multi_gpu_parallelism(field: str) -> None:
    model, parallel = _target_configs()
    setattr(parallel, field, 2)

    with pytest.raises(ValidationError, match=f"{field.replace('_', ' ')} 1"):
        SpeculativeConfig(
            method="uno",
            num_speculative_tokens=7,
            uno_adapter="/models/k2-uno",
            uno_mask_token_id=250624,
            target_model_config=model,
            target_parallel_config=parallel,
        )
