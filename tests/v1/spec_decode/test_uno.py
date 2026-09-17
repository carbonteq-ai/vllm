# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.uno import UnoProposer


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
