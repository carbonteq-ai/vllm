# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-aware prefix-cache eviction (CarbonTeq).

A multi-turn client (an agent episode, an RL rollout) sends each turn with the
same session id. Between turns the session's cached prefix sits in the free
block queue, ordered only by recency, so a session waiting on a slow tool call
is evicted before the blocks of sessions that have already finished.

This tracker remembers, per live session, the cached blocks its latest request
left behind. When the client says a session has ended, the blocks that no
other live session still holds move to the front of the free queue, so they
are evicted before any live session's prefix.
"""

from collections import OrderedDict

from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock

_Holding = tuple[KVCacheBlock, BlockHashWithGroupId]


class SessionBlockTracker:
    """Map live sessions to the cached blocks their latest request left.

    Blocks are identified by object and block hash together, so a block that
    was evicted and re-cached for other tokens is never mistaken for the one a
    session held.
    """

    def __init__(self, max_sessions: int = 65536) -> None:
        # Tracking is only an eviction hint, so the oldest sessions are
        # forgotten beyond this bound rather than growing without limit when
        # clients never release their sessions.
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, list[_Holding]] = OrderedDict()
        self._holders: dict[tuple[int, BlockHashWithGroupId], int] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    def record(self, session_id: str, blocks: list[KVCacheBlock]) -> None:
        """Replace the session's holdings with the cached ``blocks``.

        A turn's prompt contains the previous turns, so the latest request's
        blocks cover everything the session can still reuse.
        """
        holdings = [
            (block, block.block_hash)
            for block in blocks
            if not block.is_null and block.block_hash is not None
        ]
        for block, block_hash in holdings:
            key = (id(block), block_hash)
            self._holders[key] = self._holders.get(key, 0) + 1
        self._drop(self._sessions.pop(session_id, ()))
        self._sessions[session_id] = holdings
        while len(self._sessions) > self.max_sessions:
            _, oldest = self._sessions.popitem(last=False)
            self._drop(oldest)

    def release(self, session_id: str) -> list[KVCacheBlock]:
        """Forget the session and return its blocks nobody else holds.

        The result is tail first, the order in which they should be evicted,
        and holds only blocks that are still cached for the same tokens and
        are not in use by a running request.
        """
        holdings = self._sessions.pop(session_id, ())
        self._drop(holdings)
        return [
            block
            for block, block_hash in reversed(holdings)
            if block.ref_cnt == 0
            and block.block_hash == block_hash
            and (id(block), block_hash) not in self._holders
        ]

    def clear(self) -> None:
        self._sessions.clear()
        self._holders.clear()

    def _drop(self, holdings) -> None:
        for block, block_hash in holdings:
            key = (id(block), block_hash)
            remaining = self._holders[key] - 1
            if remaining:
                self._holders[key] = remaining
            else:
                del self._holders[key]
