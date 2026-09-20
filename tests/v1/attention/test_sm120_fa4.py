# SPDX-License-Identifier: Apache-2.0

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.sm120_fa4 import (
    Sm120FA4MetadataBuilder,
    _deterministic_kernel_options,
)


def test_target_attention_supports_cudagraph_replay() -> None:
    assert (
        Sm120FA4MetadataBuilder.get_cudagraph_support(None, None)
        == AttentionCGSupport.ALWAYS
    )


def test_deterministic_prefill_uses_fixed_reduction_plan() -> None:
    assert _deterministic_kernel_options(
        max_seqlen_q=256,
        batch_invariant=True,
    ) == (1, (64, 64))


def test_deterministic_decode_keeps_decode_tile_policy() -> None:
    assert _deterministic_kernel_options(
        max_seqlen_q=1,
        batch_invariant=True,
    ) == (1, None)


def test_fast_mode_keeps_kernel_autotuning() -> None:
    assert _deterministic_kernel_options(
        max_seqlen_q=256,
        batch_invariant=False,
    ) == (0, None)
