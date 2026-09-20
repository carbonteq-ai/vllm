# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.uno import UnoProposer


def test_uno_system_overlay_mask_is_limited_to_active_rows() -> None:
    proposer = object.__new__(UnoProposer)
    proposer.is_masked_token_mask = torch.tensor(
        [False, True, True, False, True, True]
    )

    active = proposer._system_lora_mask(3)

    assert active is not None
    assert torch.equal(active, torch.tensor([False, True, True]))


def test_uno_system_overlay_mask_reuses_persistent_storage() -> None:
    proposer = object.__new__(UnoProposer)
    proposer.is_masked_token_mask = torch.tensor([False, True, True, False])

    active = proposer._system_lora_mask(3)

    assert active is not None
    assert active.untyped_storage().data_ptr() == (
        proposer.is_masked_token_mask.untyped_storage().data_ptr()
    )
