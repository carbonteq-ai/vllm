# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.uno import UNO_DRAFT_ADAPTER_ID, UnoProposer


def test_uno_lora_mapping_gates_only_noise_positions() -> None:
    noise_mask = torch.tensor([False, True, True, False, True])
    sample_indices = torch.tensor([0, 1, 4], dtype=torch.int32)

    prompt_mapping, token_mapping = UnoProposer._build_lora_mappings(
        noise_mask,
        sample_indices,
        adapter_id=17,
    )

    assert token_mapping == (0, 17, 17, 0, 17)
    assert prompt_mapping == (0, 17, 17)


def test_uno_composite_mapping_keeps_policy_on_seed_rows() -> None:
    noise_mask = torch.tensor([False, True, True, False, True, True])
    sample_indices = torch.arange(6, dtype=torch.int32)
    policy_ids = torch.tensor([11, 12])

    prompt_mapping, token_mapping = UnoProposer._build_composite_lora_mappings(
        noise_mask,
        sample_indices,
        policy_ids,
        num_speculative_tokens=3,
    )

    expected = (
        11,
        UNO_DRAFT_ADAPTER_ID,
        UNO_DRAFT_ADAPTER_ID,
        12,
        UNO_DRAFT_ADAPTER_ID,
        UNO_DRAFT_ADAPTER_ID,
    )
    assert token_mapping == expected
    assert prompt_mapping == expected
