# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariant split-KV in the Triton unified attention kernel.

Under ``VLLM_BATCH_INVARIANT`` both the 2D and the 3D (split-KV) kernels
reduce fixed KV segments in the same order, so a query token must produce the
same bits whichever kernel runs it, whatever else is in the batch, and
whether it arrives as a decode step or inside a prefill chunk.
"""

import pytest
import torch

import vllm.v1.attention.ops.triton_unified_attention as tua
from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton attention needs CUDA"
)

BLOCK_SIZE = 16
SEGMENT_LEN = 64  # two 32-token tiles, so short sequences span many segments
MAX_LEN = 1024
NUM_POOL_SEQS = 3


class _Pool:
    """KV cache and per-position queries shared by every call in a test."""

    def __init__(self, num_q_heads, num_kv_heads, head_size, use_sinks):
        g = torch.Generator(device="cuda").manual_seed(0)
        blocks_per_seq = MAX_LEN // BLOCK_SIZE
        num_blocks = blocks_per_seq * NUM_POOL_SEQS
        shape = (num_blocks, BLOCK_SIZE, num_kv_heads, head_size)
        self.k = torch.randn(shape, generator=g, device="cuda").to(torch.bfloat16)
        self.v = torch.randn(shape, generator=g, device="cuda").to(torch.bfloat16)
        perm = torch.randperm(num_blocks, generator=g, device="cuda")
        self.block_table = perm.view(NUM_POOL_SEQS, blocks_per_seq).to(torch.int32)
        self.q = torch.randn(
            (NUM_POOL_SEQS, MAX_LEN, num_q_heads, head_size),
            generator=g,
            device="cuda",
        ).to(torch.bfloat16)
        self.sinks = (
            torch.randn(num_q_heads, generator=g, device="cuda") if use_sinks else None
        )
        self.num_q_heads = num_q_heads
        self.head_size = head_size

    def run(self, requests, *, allow_3d, sliding_window, softcap):
        """requests: list of (pool_seq, context_len, query_len)."""
        q = torch.cat(
            [self.q[s, c : c + n] for s, c, n in requests], dim=0
        ).contiguous()
        query_lens = [n for _, _, n in requests]
        cu = torch.tensor(
            [0] + torch.tensor(query_lens).cumsum(0).tolist(),
            dtype=torch.int32,
            device="cuda",
        )
        seq_lens = torch.tensor(
            [c + n for _, c, n in requests], dtype=torch.int32, device="cuda"
        )
        block_table = self.block_table[[s for s, _, _ in requests]]
        out = torch.empty_like(q)
        num_segments = next_power_of_2(MAX_LEN // SEGMENT_LEN)
        padded = next_power_of_2(self.head_size)
        rows = max(8, len(requests))
        tua.unified_attention(
            q=q,
            k=self.k,
            v=self.v,
            out=out,
            cu_seqlens_q=cu,
            max_seqlen_q=max(query_lens),
            seqused_k=seq_lens,
            max_seqlen_k=int(seq_lens.max()),
            softmax_scale=self.head_size**-0.5,
            causal=True,
            window_size=(sliding_window - 1, 0) if sliding_window else (-1, -1),
            block_table=block_table,
            softcap=softcap,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            sinks=self.sinks,
            seq_threshold_3D=rows if allow_3d else 0,
            num_par_softmax_segments=num_segments,
            softmax_segm_output=torch.full(
                (rows, self.num_q_heads, num_segments, padded),
                float("nan"),
                device="cuda",
            ),
            softmax_segm_max=torch.full(
                (rows, self.num_q_heads, num_segments), float("nan"), device="cuda"
            ),
            softmax_segm_expsum=torch.full(
                (rows, self.num_q_heads, num_segments), float("nan"), device="cuda"
            ),
            invariant_segment_len=SEGMENT_LEN,
        )
        pieces = out.split(query_lens)
        return {
            (s, c + i): pieces[k][i]
            for k, (s, c, n) in enumerate(requests)
            for i in range(n)
        }


def _assert_same(a, b, what):
    for key in a.keys() & b.keys():
        assert torch.equal(a[key], b[key]), f"{what}: token {key} differs"


@pytest.fixture
def invariant(monkeypatch):
    monkeypatch.setattr(tua, "is_batch_invariant", True)


@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_size",
    [(16, 8, 256), (16, 1, 512), (8, 2, 256), (8, 8, 128)],
)
@pytest.mark.parametrize("sliding_window", [None, 100])
@pytest.mark.parametrize("use_sinks", [False, True])
@pytest.mark.parametrize("softcap", [0.0, 30.0])
def test_token_bits_independent_of_kernel_batch_and_chunking(
    invariant, num_q_heads, num_kv_heads, head_size, sliding_window, use_sinks, softcap
):
    pool = _Pool(num_q_heads, num_kv_heads, head_size, use_sinks)
    kw = dict(sliding_window=sliding_window, softcap=softcap)
    target = 0
    lengths = [1, 31, 64, 65, 200, 517, 1000]

    # Decode alone: one sequence, pure decode, so the split-KV 3D kernel runs.
    alone = {}
    for n in lengths:
        alone |= pool.run([(target, n - 1, 1)], allow_3d=True, **kw)

    # Same decode steps inside a larger decode batch forced onto the 2D kernel.
    batched = {}
    for n in lengths:
        batched |= pool.run(
            [(1, 400, 1), (target, n - 1, 1), (2, 77, 1)], allow_3d=False, **kw
        )
    _assert_same(alone, batched, "3D alone vs 2D batched decode")

    # Decode next to a prefill: mixed batch, 2D kernel.
    for n in lengths:
        mixed = pool.run([(1, 0, 300), (target, n - 1, 1)], allow_3d=True, **kw)
        _assert_same(alone, mixed, "decode alone vs decode beside a prefill")

    # The same tokens computed inside prefill chunks, whole and split unevenly.
    whole = pool.run([(target, 0, 1000)], allow_3d=True, **kw)
    _assert_same(alone, whole, "decode vs one-shot prefill")
    chunked = {}
    for c, n in [(0, 37), (37, 300), (337, 1), (338, 662)]:
        chunked |= pool.run([(target, c, n), (2, 10, 5)], allow_3d=True, **kw)
    _assert_same(whole, chunked, "one-shot vs chunked prefill")
    assert len(chunked) >= 1000


@pytest.mark.parametrize("num_q_heads,num_kv_heads,head_size", [(16, 8, 256)])
def test_fixed_segments_match_default_numerics(
    monkeypatch, num_q_heads, num_kv_heads, head_size
):
    pool = _Pool(num_q_heads, num_kv_heads, head_size, use_sinks=False)
    requests = [(0, 999, 1), (1, 300, 1), (2, 0, 200)]
    kw = dict(sliding_window=None, softcap=0.0)
    monkeypatch.setattr(tua, "is_batch_invariant", False)
    reference = pool.run(requests, allow_3d=True, **kw)
    monkeypatch.setattr(tua, "is_batch_invariant", True)
    fixed = pool.run(requests, allow_3d=True, **kw)
    for key, value in reference.items():
        torch.testing.assert_close(fixed[key], value, atol=2e-2, rtol=2e-2)


def test_segment_len_grows_with_max_model_len():
    assert tua.batch_invariant_segment_len(4096) == tua.INVARIANT_MIN_SEGMENT_LEN
    for max_model_len in (32768, 131072, 262144):
        seg_len = tua.batch_invariant_segment_len(max_model_len)
        assert seg_len * tua.INVARIANT_MAX_SEGMENTS >= max_model_len
        assert seg_len % 32 == 0
