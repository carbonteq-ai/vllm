# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-aware prefix-cache eviction (CarbonTeq)."""

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.kv_session_tracker import SessionBlockTracker
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

BLOCK = 16


@pytest.fixture(autouse=True)
def _hash():
    init_none_hash(sha256)


def _manager(num_blocks: int = 16) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            )
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=BLOCK,
        scheduler_block_size=BLOCK,
    )


def _request(request_id: str, tokens: list[int], session_id: str | None) -> Request:
    params = SamplingParams(max_tokens=1)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
        session_id=session_id,
    )


def _run(manager: KVCacheManager, request: Request) -> list[int]:
    """Prefill ``request`` fully, free it, and return its block ids."""
    hit, num_hit, _ = manager.get_computed_blocks(request)
    blocks = manager.allocate_slots(request, request.num_tokens - num_hit, num_hit, hit)
    assert blocks is not None
    ids = list(manager.get_block_ids(request.request_id)[0])
    request.num_computed_tokens = request.num_tokens
    manager.free(request)
    return ids


def _tokens(block_values: list[int]) -> list[int]:
    return [value for value in block_values for _ in range(BLOCK)]


def _eviction_order(manager: KVCacheManager) -> list[int]:
    queue = manager.block_pool.free_block_queue
    return [block.block_id for block in queue.get_all_free_blocks()]


def test_released_session_is_evicted_first_but_shared_prompt_is_kept():
    manager = _manager()
    # Two rollouts of one task share a two-block prompt.
    a = _run(manager, _request("a", _tokens([1, 2, 3, 4]), "rollout-a"))
    b = _run(manager, _request("b", _tokens([1, 2, 5, 6]), "rollout-b"))
    assert a[:2] == b[:2]
    free_before = manager.block_pool.get_num_free_blocks()

    assert manager.release_session("rollout-a") == 2

    # A's own blocks lead the eviction order, tail first; the shared prompt,
    # still held by the live rollout B, keeps its LRU place.
    assert _eviction_order(manager)[:2] == [a[3], a[2]]
    assert manager.block_pool.get_num_free_blocks() == free_before
    # Released blocks stay cached until they are reused.
    hit, num_hit, _ = manager.get_computed_blocks(
        _request("a-late", _tokens([1, 2, 3, 4, 7]), None)
    )
    assert num_hit == 4 * BLOCK

    # Once B ends too, the shared prompt goes as well.
    assert manager.release_session("rollout-b") == 4
    assert set(_eviction_order(manager)[:4]) == {b[0], b[1], b[2], b[3]}


def test_a_later_turn_replaces_the_session_holdings():
    manager = _manager()
    first = _run(manager, _request("t0", _tokens([1, 2]), "s"))
    second = _run(manager, _request("t1", _tokens([1, 2, 3]), "s"))
    assert second[:2] == first
    assert manager.release_session("s") == 3
    assert _eviction_order(manager)[:3] == [second[2], second[1], second[0]]
    assert manager.release_session("s") == 0


def test_unknown_session_and_reset_are_harmless():
    manager = _manager()
    _run(manager, _request("a", _tokens([1, 2]), "s"))
    assert manager.release_session("missing") == 0
    assert manager.reset_prefix_cache()
    assert len(manager.sessions) == 0
    assert manager.release_session("s") == 0


def test_requests_without_a_session_are_not_tracked():
    manager = _manager()
    _run(manager, _request("a", _tokens([1, 2]), None))
    assert len(manager.sessions) == 0


def test_a_block_recached_for_other_tokens_is_not_released():
    manager = _manager(num_blocks=4)  # three usable blocks
    _run(manager, _request("a", _tokens([1, 2]), "s"))
    # Evict both of the session's blocks by caching three other blocks.
    _run(manager, _request("b", _tokens([7, 8, 9]), None))
    assert manager.release_session("s") == 0


def test_tracker_forgets_the_oldest_sessions_beyond_its_bound():
    tracker = SessionBlockTracker(max_sessions=2)
    for session in ("a", "b", "c"):
        tracker.record(session, [])
    assert len(tracker) == 2
    assert tracker.release("a") == []
