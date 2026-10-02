"""Global pacing gate for search providers (Phase 8b PRD §6.6)."""

from __future__ import annotations

import asyncio
import random
import time


class SearchGate:
    """One pacing gate per process: serializes provider calls to an interval.

    Agent tool loops fire 3-5 searches back-to-back; bursts are the most
    reliable challenge trigger observed. `jitter_s=None` derives 20% of the
    interval, capped at 0.6 s.
    """

    def __init__(
        self, min_interval_s: float = 3.0, jitter_s: float | None = None
    ) -> None:
        self._min_interval_s = min_interval_s
        self._jitter_s = (
            min(0.6, min_interval_s * 0.2) if jitter_s is None else jitter_s
        )
        self._lock = asyncio.Lock()
        self._last: float | None = None

    async def acquire(self) -> None:
        async with self._lock:
            interval = self._min_interval_s + random.uniform(0, self._jitter_s)
            now = time.monotonic()
            if self._last is not None:
                wait = self._last + interval - now
                if wait > 0:
                    await asyncio.sleep(wait)
            self._last = time.monotonic()
