"""Process-local LRU + TTL cache for search successes (Phase 8b PRD §6.7)."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Sequence

from app.tools.search.providers import SearchOutcome


def _normalize_query(query: str) -> str:
    return " ".join(query.split()).lower()


class SearchCache:
    """Process-local LRU + TTL cache; only successful outcomes are stored."""

    def __init__(self, max_entries: int = 256, ttl_s: float = 900.0) -> None:
        self._max_entries = max_entries
        self._ttl_s = ttl_s
        self._entries: OrderedDict[
            str, tuple[float, SearchOutcome, tuple[str, ...]]
        ] = OrderedDict()

    @staticmethod
    def _key(provider: str, query: str) -> str:
        return f"{provider}:{_normalize_query(query)}"

    def get(
        self, provider: str, query: str
    ) -> tuple[SearchOutcome, tuple[str, ...]] | None:
        if self._ttl_s <= 0:
            return None
        key = self._key(provider, query)
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, outcome, attempts = entry
        if time.monotonic() - stored_at > self._ttl_s:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return outcome, attempts

    def put(
        self,
        outcome: SearchOutcome,
        query: str,
        *,
        attempts: Sequence[str] = (),
    ) -> None:
        # Never cache blocks or transport failures: both are moment-in-time
        # signals. A legitimate empty SERP (not failed) is cached normally.
        if outcome.failed or self._ttl_s <= 0:
            return
        key = self._key(outcome.provider, query)
        self._entries[key] = (time.monotonic(), outcome, tuple(attempts))
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
