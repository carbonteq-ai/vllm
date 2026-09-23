# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Under batch invariance, GDN prefill chunks end on kernel chunk boundaries."""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.scheduler import (
    _GDN_INVARIANT_PREFILL_ALIGNMENT,
    Scheduler,
)


def _split(computed, prompt, num_new, outputs=0):
    scheduler = SimpleNamespace(
        batch_invariant_prefill_alignment=_GDN_INVARIANT_PREFILL_ALIGNMENT
    )
    request = SimpleNamespace(
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
        num_tokens=prompt + outputs,
    )
    return Scheduler._batch_invariant_prefill_split(scheduler, request, num_new)


def test_alignment_matches_fla_chunk_size():
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

    assert _GDN_INVARIANT_PREFILL_ALIGNMENT == FLA_CHUNK_SIZE


@pytest.mark.parametrize(
    "computed,num_new,expected",
    [
        (0, 1533, 1472),  # budget minus decode tokens: round down to 64
        (0, 1536, 1536),  # already aligned
        (1472, 700, 640),  # later chunk ends on a boundary too
        (0, 40, 0),  # cannot reach a boundary: wait a step
    ],
)
def test_partial_chunks_end_on_boundaries(computed, num_new, expected):
    got = _split(computed, 3000, num_new)
    assert got == expected
    assert (computed + got) % _GDN_INVARIANT_PREFILL_ALIGNMENT == 0 or got == 0


def test_final_chunk_and_decode_are_exempt():
    assert _split(2944, 3000, 56) == 56  # finishes the prompt
    assert _split(2900, 3000, 500) == 500  # budget covers the rest
    assert _split(3004, 3000, 1, outputs=5) == 1  # decode step
