"""RunManager contract (PRD §6.1): records, events, fan-out, lifecycle."""

import asyncio
import contextlib

import pytest

from app.runs import (
    TERMINAL_EVENT_TYPES,
    ErrorInfo,
    RunActiveError,
    RunEvent,
    RunEventType,
    RunManager,
    RunStatus,
    StepKind,
    StepRecord,
    StepStatus,
)
from app.schemas import AgentMessage, AgentRole, MessageType


def _manager(max_runs: int = 50) -> RunManager:
    return RunManager(max_runs=max_runs)


def _complete(run) -> None:
    run.status = RunStatus.COMPLETED


# ------------------------------------------------------------------ create/get

def test_create_returns_running_record():
    runs = _manager()
    run = runs.create("Should we build X?")
    assert len(run.id) == 32
    assert run.task == "Should we build X?"
    assert run.status is RunStatus.RUNNING
    assert run.created_at.tzinfo is not None
    assert run.started_at is None and run.finished_at is None
    assert run.steps == [] and run.events == []
    assert run.duration_ms is None
    assert runs.get(run.id) is run


def test_create_strips_task():
    runs = _manager()
    assert runs.create("  hello  ").task == "hello"


def test_create_rejects_blank_and_oversized():
    runs = _manager()
    with pytest.raises(ValueError):
        runs.create("   ")
    with pytest.raises(ValueError):
        runs.create("x" * (RunManager.MAX_TASK_CHARS + 1))


def test_single_active_run_enforced():
    runs = _manager()
    first = runs.create("one")
    with pytest.raises(RunActiveError) as exc_info:
        runs.create("two")
    assert exc_info.value.active_run_id == first.id
    _complete(first)
    assert runs.active() is None
    second = runs.create("two")
    assert second.status is RunStatus.RUNNING


def test_get_unknown_returns_none():
    assert _manager().get("nope") is None


# ------------------------------------------------------------------ list/evict

def test_list_newest_first_and_limited():
    runs = _manager()
    a = runs.create("a")
    _complete(a)
    b = runs.create("b")
    _complete(b)
    listed = runs.list()
    assert [r.id for r in listed] == [b.id, a.id]
    assert runs.list(limit=1) == [b]


def test_eviction_caps_runs_and_never_evicts_running():
    runs = _manager(max_runs=3)
    ids = []
    for i in range(3):
        run = runs.create(f"task {i}")
        _complete(run)
        ids.append(run.id)
    assert len(runs.list(limit=50)) == 3

    running = runs.create("active")
    assert runs.get(ids[0]) is None, "oldest completed run evicted"
    assert runs.get(running.id) is running
    assert len(runs.list(limit=50)) == 3


def test_eviction_stops_when_all_running_impossible():
    # only one run can be RUNNING, so eviction always finds candidates; this
    # guards the loop's safety break via a manager whose cap is below zero-ish
    runs = _manager(max_runs=1)
    run = runs.create("a")
    _complete(run)
    runs.create("b")
    assert len(runs.list(limit=50)) == 1


# ------------------------------------------------------------------ events

def test_append_event_assigns_monotonic_seq_and_history():
    runs = _manager()
    run = runs.create("t")
    e1 = runs.append_event(run.id, RunEventType.RUN_STARTED, task=run.task)
    e2 = runs.append_event(run.id, RunEventType.STEP_STARTED, step=1,
                           kind=StepKind.PLAN, agent=AgentRole.MANAGER)
    assert (e1.seq, e2.seq) == (1, 2)
    assert run.events == [e1, e2]
    assert e1.ts.tzinfo is not None


def test_append_event_unknown_run_keyerror():
    with pytest.raises(KeyError):
        _manager().append_event("nope", RunEventType.RUN_STARTED)


def test_event_json_round_trip_and_extra_forbidden():
    runs = _manager()
    run = runs.create("t")
    message = AgentMessage(from_agent=AgentRole.MANAGER, type=MessageType.PLAN,
                           content="plan")
    event = runs.append_event(run.id, RunEventType.STEP_COMPLETED, step=1,
                              kind=StepKind.PLAN, agent=AgentRole.MANAGER,
                              message=message, duration_ms=12.5)
    again = RunEvent.model_validate_json(event.model_dump_json())
    assert again == event
    with pytest.raises(Exception):
        RunEvent(seq=1, type=RunEventType.RUN_STARTED, run_id="x", mystery=1)


def test_terminal_event_types_constant():
    assert TERMINAL_EVENT_TYPES == {RunEventType.RUN_COMPLETED, RunEventType.RUN_FAILED}


# ------------------------------------------------------------------ subscribe

async def test_subscribe_backlog_and_live_fanout():
    runs = _manager()
    run = runs.create("t")
    runs.append_event(run.id, RunEventType.RUN_STARTED, task=run.task)

    queue, backlog = runs.subscribe(run.id)
    assert [e.seq for e in backlog] == [1]

    runs.append_event(run.id, RunEventType.RUN_COMPLETED, duration_ms=1.0)
    live = queue.get_nowait()
    assert live.type is RunEventType.RUN_COMPLETED


async def test_subscribe_since_filters_backlog():
    runs = _manager()
    run = runs.create("t")
    for _ in range(3):
        runs.append_event(run.id, RunEventType.STEP_STARTED, step=1,
                          kind=StepKind.PLAN, agent=AgentRole.MANAGER)
    _, backlog = runs.subscribe(run.id, since=2)
    assert [e.seq for e in backlog] == [3]


async def test_unsubscribe_stops_fanout():
    runs = _manager()
    run = runs.create("t")
    queue, _ = runs.subscribe(run.id)
    runs.unsubscribe(run.id, queue)
    runs.append_event(run.id, RunEventType.RUN_STARTED, task=run.task)
    assert queue.empty()
    runs.unsubscribe(run.id, queue)  # idempotent


# ------------------------------------------------------------------ tasks

async def test_task_registration_and_cancel_active():
    runs = _manager()
    run = runs.create("t")

    async def _never():
        await asyncio.sleep(60)

    task = asyncio.create_task(_never())
    runs.register_task(run.id, task)
    assert runs.get_task(run.id) is task

    cancelled = runs.cancel_active_task()
    assert cancelled is task
    assert task.cancelled() or task.cancelling() > 0
    with contextlib.suppress(asyncio.CancelledError):
        await task


def test_cancel_active_task_returns_none_when_idle():
    assert _manager().cancel_active_task() is None


# ------------------------------------------------------------------ misc

def test_duration_ms_requires_both_timestamps():
    from datetime import UTC, datetime

    runs = _manager()
    run = runs.create("t")
    assert run.duration_ms is None
    run.started_at = datetime(2026, 1, 1, tzinfo=UTC)
    assert run.duration_ms is None
    run.finished_at = datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC)
    assert run.duration_ms == 10000.0


def test_step_record_defaults():
    from datetime import UTC, datetime

    step = StepRecord(index=1, kind=StepKind.PLAN, agent=AgentRole.MANAGER,
                      status=StepStatus.RUNNING, started_at=datetime.now(UTC))
    assert step.duration_ms is None
    assert step.message is None
    assert step.skipped is False
    assert step.error is None
    assert ErrorInfo(type="X", message="m").hint == ""
