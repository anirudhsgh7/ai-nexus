"""Graceful shutdown: stream sentinel + lifespan ordering (Phase 10 PRD §6.1).

The defect this guards: with an SSE client attached, uvicorn's graceful
shutdown waited for the connection, delaying lifespan `finally` (and therefore
`cancel_active_task()`) arbitrarily — the Phase 8 two-writer overlap. The fix
has two layers: `RunManager.close_streams()` wakes every subscriber with
`STREAM_CLOSED`, and `scripts/serve.py` bounds how long uvicorn may wait for
remaining connections (`shutdown_grace_s`).
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.main import create_app
from app.runs import (
    STREAM_CLOSED,
    RunEventType,
    RunManager,
    RunStatus,
)

# ------------------------------------------------------ RunManager contract


async def test_close_streams_pushes_sentinel_to_every_subscriber() -> None:
    runs = RunManager()
    one = runs.create("one")
    queue_a, _ = runs.subscribe(one.id)
    queue_b, _ = runs.subscribe(one.id)
    one.status = RunStatus.COMPLETED  # free the single-active slot
    two = runs.create("two")
    queue_c, _ = runs.subscribe(two.id)

    closed = runs.close_streams()
    assert closed == 3
    assert queue_a.get_nowait() is STREAM_CLOSED
    assert queue_b.get_nowait() is STREAM_CLOSED
    assert queue_c.get_nowait() is STREAM_CLOSED
    # subscribers were cleared: fan-out after close reaches nobody
    runs.append_event(one.id, RunEventType.RUN_COMPLETED, duration_ms=1.0)
    with pytest.raises(asyncio.QueueEmpty):
        queue_a.get_nowait()


def test_close_streams_idempotent() -> None:
    runs = RunManager()
    run = runs.create("x")
    runs.subscribe(run.id)
    assert runs.close_streams() == 1
    assert runs.close_streams() == 0
    assert runs.close_streams() == 0


def test_close_streams_with_no_subscribers_is_noop() -> None:
    assert RunManager().close_streams() == 0


# ----------------------------------------------------- lifespan end-to-end

async def test_lifespan_closes_streams_then_cancels_active_run() -> None:
    app = create_app()
    queue_holder: dict[str, object] = {}
    task_holder: dict[str, asyncio.Task[None]] = {}

    async with app.router.lifespan_context(app):
        runs: RunManager = app.state.runs
        run = runs.create("long running")  # status=RUNNING -> active()
        queue, _ = runs.subscribe(run.id)

        async def forever() -> None:
            await asyncio.sleep(60)

        task = asyncio.create_task(forever())
        runs.register_task(run.id, task)
        queue_holder["queue"] = queue
        task_holder["task"] = task

    # lifespan `finally` has now run: streams closed, then the task cancelled
    assert queue_holder["queue"].get_nowait() is STREAM_CLOSED  # type: ignore[union-attr]
    assert task_holder["task"].done()
    assert task_holder["task"].cancelled()


# ------------------------------------------------- SSE end-to-end (sentinel)

async def test_sentinel_ends_live_sse_stream_cleanly() -> None:
    app = create_app()
    async with app.router.lifespan_context(app):
        runs: RunManager = app.state.runs
        run = runs.create("live sse")
        runs.append_event(run.id, RunEventType.RUN_STARTED, task=run.task)

        async def close_later() -> None:
            await asyncio.sleep(0.05)
            assert runs.close_streams() >= 1

        closer = asyncio.create_task(close_later())
        transport = httpx.ASGITransport(app=app)
        lines: list[str] = []
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Without the sentinel this stream never ends (15 s keep-alives),
            # so the timeout is the test's failure detector.
            async with asyncio.timeout(2):
                async with client.stream(
                    "GET", f"/api/runs/{run.id}/events"
                ) as response:
                    assert response.status_code == 200
                    async for line in response.aiter_lines():
                        lines.append(line)
        await closer

    text = "\n".join(lines)
    assert "event: run_started" in text          # replay still delivered
    assert "run_failed" not in text              # no fabricated terminal event
    assert ": keep-alive" not in text            # closed promptly, not via timeout
