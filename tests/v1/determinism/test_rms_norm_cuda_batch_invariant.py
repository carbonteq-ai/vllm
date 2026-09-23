# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariant RMSNorm: a row's bits must not depend on its batch."""

import os

# The CUDA kernel reads VLLM_BATCH_INVARIANT once, on its first launch.
os.environ["VLLM_BATCH_INVARIANT"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm.model_executor.determinism import batch_invariant as bi  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="RMSNorm kernels need CUDA"
)

# One launch over many rows (>= 256, where the non-invariant launch shrinks its
# block) against small launches. Last-bit differences in the fp32 reduction
# survive bf16 rounding in only ~1% of rows, so the test needs many rows: with
# invariance off, 470 of 32,768 rows differ.
ROWS = 16384
CHUNK = 128


def _norm_in_chunks(make_input, w):
    return torch.cat(
        [
            bi.rms_norm_batch_invariant(make_input(i, i + CHUNK), w, 1e-6)
            for i in range(0, ROWS, CHUNK)
        ]
    )


@pytest.mark.parametrize("hidden", [896, 2560, 3840, 5120])
def test_contiguous_rows_are_batch_invariant(hidden):
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(ROWS, hidden, generator=g, device="cuda").bfloat16()
    w = torch.randn(hidden, generator=g, device="cuda").bfloat16()
    assert bi._cuda_rms_norm_is_batch_invariant(x, w)
    full = bi.rms_norm_batch_invariant(x, w, 1e-6)
    chunked = _norm_in_chunks(lambda lo, hi: x[lo:hi].clone(), w)
    assert torch.equal(full, chunked)


@pytest.mark.parametrize("num_heads,head_dim", [(16, 256), (8, 128), (16, 512)])
def test_strided_head_views_are_batch_invariant(num_heads, head_dim):
    # q/k norm on a view into the fused QKV output: [tokens, heads, head_dim]
    # with a row stride wider than one head block.
    g = torch.Generator(device="cuda").manual_seed(1)
    width = 3 * num_heads * head_dim
    w = torch.randn(head_dim, generator=g, device="cuda").bfloat16()
    qkv = torch.randn(ROWS // num_heads, width, generator=g, device="cuda").bfloat16()

    def q_view(t):
        return t[:, num_heads * head_dim : 2 * num_heads * head_dim].view(
            -1, num_heads, head_dim
        )

    view = q_view(qkv)
    assert not view.is_contiguous()
    assert bi._cuda_rms_norm_is_batch_invariant(view, w)
    full = bi.rms_norm_batch_invariant(view, w, 1e-6)
    tokens = qkv.shape[0]
    chunked = torch.cat(
        [
            bi.rms_norm_batch_invariant(q_view(qkv[i : i + 8].clone()), w, 1e-6)
            for i in range(0, tokens, 8)
        ]
    )
    assert torch.equal(full, chunked)


def test_misaligned_rows_fall_back_to_triton():
    x = torch.randn(8, 4099, device="cuda").bfloat16()[:, 1:4097]
    assert not bi._cuda_rms_norm_is_batch_invariant(x, None)
    torch.testing.assert_close(
        bi.rms_norm_batch_invariant(x, None, 1e-6),
        bi.rms_norm_batch_invariant(x.contiguous(), None, 1e-6),
        atol=2e-2,
        rtol=2e-2,
    )


def test_cuda_path_matches_triton_numerics():
    x = torch.randn(64, 3840, device="cuda").bfloat16()
    w = torch.randn(3840, device="cuda").bfloat16()
    cuda_out = bi.rms_norm_batch_invariant(x, w, 1e-6)
    triton_out = torch.empty_like(x)
    bi._rms_norm_kernel[(64,)](
        x, w, triton_out, x.stride(0), triton_out.stride(0), 3840, 1e-6,
        BLOCK_SIZE=1024, HAS_WEIGHT=True,
    )  # fmt: skip
    torch.testing.assert_close(cuda_out, triton_out, atol=2e-2, rtol=2e-2)
