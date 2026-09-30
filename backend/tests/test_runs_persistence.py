"""RunManager + RunStore semantics: write-through, reload, eviction, recovery (PRD §9).

The restart test drives the REAL orchestrator over the ITERATIVE scripted
chain (plan → research ∥ ideate → critique → decide → revise → critique →
decide → synthesize), then opens a fresh RunManager on the same file — the
offline proof of PLAN.md's exit criterion.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.agents import build_registry
from app.db import PersistenceError, RunStore
from app.orchestrator import Orchestrator
from app.runs import (
    RoundSnapshot,
    RunEventType,
    RunManager,
    RunRecord,
    RunStatus,
)
from app.schemas import (
    AgentRole,
    DecisionAction,
    ManagerDecision,
)
from tests.fakes import FakeProvider

TASK = "Should we build X?"


def _claims(content: str, claims: list[dict]) -> str:
    return json.dumps({"content": content, "claims": claims})


def _verdicts(content: str, verdicts: list[dict]) -> str:
    return json.dumps({"content": content, "verdicts": verdicts})


ITERATIVE_PAYLOADS = [
    _claims("Plan prose.", []),
    _claims("Research prose.", [
        {"statement": "Finding one", "status": "unverified", "evidence": []},
        {"statement": "Finding two", "status": "fact",
         "evidence": [{"source": "report"}]},
    ]),
    _claims("Ideation prose.", [
        {"statement": "Idea A", "status": "hypothesis", "evidence": []},
    ]),
    _verdicts("Critique prose.", [
        {"claim_id": "c1", "verdict": "refuted", "objection": "no source",
         "evidence": []},
        {"claim_id": "c2", "verdict": "supported", "objection": "report holds",
         "evidence": [{"source": "report"}]},
        {"claim_id": "c3", "verdict": "unverifiable", "objection": "no data",
         "evidence": []},
    ]),
    json.dumps({"action": "call_agent", "target": "researcher",
                "instruction": "Find a published source.",
                "reason": "evidence gap remains", "confidence": 0.8}),
    _claims("Revision prose.", [
        {"statement": "Revised finding A", "status": "fact",
         "evidence": [{"source": "audit report"}]},
        {"statement": "Revised finding B", "status": "unverified",
         "evidence": [{"source": "press release"}]},
    ]),
    _verdicts("Critique prose round 2.", [
        {"claim_id": "c4", "verdict": "supported", "objection": "audit confirms",
         "evidence": [{"source": "audit report"}]},
        {"claim_id": "c5", "verdict": "unverifiable", "objection": "single source",
         "evidence": []},
    ]),
    json.dumps({"action": "finish", "reason": "no useful work remains",
                "confidence": 0.7}),
    _claims("Final answer.", []),
]


async def _execute_iterative(path: Path) -> tuple[RunManager, RunRecord]:
    provider = FakeProvider()
    for payload in ITERATIVE_PAYLOADS:
        provider.queue_result(FakeProvider.make_result(payload))
    manager = RunManager(store=RunStore(path))
    run = manager.create(TASK)
    orchestrator = Orchestrator(
        build_registry(provider), manager, max_rounds=3
    )
    await orchestrator.execute(run.id)
    assert run.status is RunStatus.COMPLETED
    assert len(run.steps) == 9, "ITERATIVE chain shape"
    return manager, run


def assert_restart_equivalent(loaded: RunRecord, original: RunRecord) -> None:
    """Deep equality with timestamp tolerance (PRD §6.6 note).

    `run.started_at/finished_at` and each `StepRecord.started_at` are assigned
    in memory, then the matching event is stamped microseconds later — the DB
    copy comes from the event. Everything else must match exactly.
    """
    assert loaded.id == original.id

    def _same_moment(a: datetime | None, b: datetime | None) -> bool:
        return a is not None and b is not None and abs(
            (a - b).total_seconds()
        ) < 1

    assert _same_moment(loaded.started_at, original.started_at)
    assert _same_moment(loaded.finished_at, original.finished_at)
    assert len(loaded.steps) == len(original.steps)
    for loaded_step, original_step in zip(loaded.steps, original.steps):
        assert _same_moment(loaded_step.started_at, original_step.started_at)
        loaded_step.started_at = original_step.started_at
    loaded.started_at = original.started_at
    loaded.finished_at = original.finished_at
    assert loaded == original


# ------------------------------------------------------------------- basics


def test_create_persists_and_reloads(tmp_path: Path):
    path = tmp_path / "a.db"
    manager = RunManager(store=RunStore(path))
    run = manager.create("persist me")
    _complete(manager, run)

    restarted = RunManager(store=RunStore(path))
    loaded = restarted.get(run.id)
    assert loaded is not None
    assert loaded.task == "persist me"
    assert loaded.status is RunStatus.COMPLETED
    assert restarted.get("missing") is None


def test_unfinished_run_is_recovered_on_reload(tmp_path: Path):
    """A run left `running` in DB (crash/restart) reloads as failed, never as
    a phantom active run."""
    path = tmp_path / "b.db"
    manager = RunManager(store=RunStore(path))
    run = manager.create("never finished")

    reloaded = RunManager(store=RunStore(path)).get(run.id)
    assert reloaded.status is RunStatus.FAILED
    assert reloaded.error.type == "ServerRestart"


def test_disabled_store_is_phase6_behavior(tmp_path: Path):
    manager = RunManager()
    manager.create("ephemeral")
    assert manager.get("missing") is None
    with pytest.raises(KeyError):
        manager.subscribe("missing")   # pre-Phase-7 contract preserved


# ------------------------------------------------------- orchestrator restart


async def test_iterative_run_survives_restart(tmp_path: Path):
    path = tmp_path / "restart.db"
    _, run = await _execute_iterative(path)

    restarted = RunManager(store=RunStore(path))   # "restart the app"
    loaded = restarted.get(run.id)
    assert loaded is not None
    assert_restart_equivalent(loaded, run)
    assert len(loaded.rounds) == 2
    assert loaded.rounds[0].decision is not None
    assert loaded.rounds[0].decision.action is DecisionAction.CALL_AGENT
    assert loaded.rounds[1].decision.action is DecisionAction.FINISH
    assert loaded.final_message.content == "Final answer."


async def test_restarted_list_includes_persisted_run(tmp_path: Path):
    path = tmp_path / "restart.db"
    _, run = await _execute_iterative(path)
    restarted = RunManager(store=RunStore(path))
    assert [r.id for r in restarted.list()] == [run.id]
    # second get hits the cache, not the disk
    assert restarted.get(run.id) is restarted.get(run.id)


async def test_subscribe_db_only_replays_terminal_backlog(tmp_path: Path):
    path = tmp_path / "restart.db"
    _, run = await _execute_iterative(path)
    restarted = RunManager(store=RunStore(path))
    queue, backlog = restarted.subscribe(run.id, since=5)
    assert [e.seq for e in backlog] == list(range(6, len(run.events) + 1))
    assert backlog[-1].type is RunEventType.RUN_COMPLETED
    assert queue.empty(), "DB-only runs get no live subscription"
    # in-memory run still fans out normally
    live_queue, _ = restarted.subscribe(run.id, since=0)
    restarted.unsubscribe(run.id, live_queue)


# ------------------------------------------------------------------ eviction


def _complete(manager: RunManager, run: RunRecord) -> None:
    run.status = RunStatus.COMPLETED
    run.finished_at = datetime.now(UTC)
    manager.append_event(run.id, RunEventType.RUN_COMPLETED, message=None)


def test_evicted_run_still_retrievable(tmp_path: Path):
    path = tmp_path / "evict.db"
    manager = RunManager(max_runs=1, store=RunStore(path))
    first = manager.create("first")
    _complete(manager, first)
    second = manager.create("second")

    assert first.id not in manager._runs, "memory eviction happened"
    loaded = manager.get(first.id)
    assert loaded is not None and loaded.task == "first"
    assert loaded.status is RunStatus.COMPLETED


def test_list_merges_memory_over_db(tmp_path: Path):
    path = tmp_path / "merge.db"
    manager = RunManager(max_runs=1, store=RunStore(path))
    first = manager.create("first")
    _complete(manager, first)
    second = manager.create("second")

    listed = manager.list()
    assert [r.id for r in listed] == [second.id, first.id]
    # memory wins for the live run (same object, not a stub)
    assert listed[0] is second
    assert listed[1] is not first, "evicted run comes back as a DB summary"


# ------------------------------------------------------------------ recovery


def test_recovery_runs_at_manager_construction(tmp_path: Path, caplog):
    path = tmp_path / "crash.db"
    manager = RunManager(store=RunStore(path))
    run = manager.create("interrupted")
    manager.append_event(run.id, RunEventType.RUN_STARTED, task=run.task)
    # simulate crash: nothing else; new process opens the same file

    with caplog.at_level("WARNING", logger="ai_nexus.runs"):
        restarted = RunManager(store=RunStore(path))
    assert any(
        "persistence_recovered" in record.getMessage()
        and run.id in record.getMessage()
        for record in caplog.records
    )
    recovered = restarted.get(run.id)
    assert recovered.status is RunStatus.FAILED
    assert recovered.error.type == "ServerRestart"
    assert recovered.events[-1].type is RunEventType.RUN_FAILED
    assert restarted.active() is None, "no phantom active run after restart"


# -------------------------------------------------------------------- rounds


def test_rounds_and_decisions_survive_restart(tmp_path: Path):
    path = tmp_path / "rounds.db"
    manager = RunManager(store=RunStore(path))
    run = manager.create("rounds")
    snapshot = RoundSnapshot(
        round_number=1, claims=[], origins={}, verdicts=[],
        worker_content={}, skeptic_content="s", supported_count=1,
        unresolved_count=1,
    )
    manager.record_round(run.id, snapshot)
    manager.record_round_decision(run.id, ManagerDecision(
        action=DecisionAction.CALL_AGENT, target=AgentRole.RESEARCHER,
        instruction="more evidence", reason="gap", confidence=0.6,
    ))
    snapshot.decision = ManagerDecision(
        action=DecisionAction.CALL_AGENT, target=AgentRole.RESEARCHER,
        instruction="more evidence", reason="gap", confidence=0.6,
    )

    restarted = RunManager(store=RunStore(path))
    loaded = restarted.get(run.id)
    assert loaded.rounds == [snapshot]


def test_record_round_decision_without_rounds_is_noop(tmp_path: Path):
    path = tmp_path / "noop.db"
    manager = RunManager(store=RunStore(path))
    run = manager.create("t")
    manager.record_round_decision(run.id, ManagerDecision(
        action=DecisionAction.FINISH, reason="x", confidence=0.1,
    ))   # must not raise; nothing to record

    loaded = RunManager(store=RunStore(path)).get(run.id)
    assert loaded.rounds == []


# ---------------------------------------------------------- failure semantics


def test_fans_out_before_persist_failure(tmp_path: Path):
    store = RunStore(tmp_path / "fail.db")
    manager = RunManager(store=store)
    run = manager.create("doomed")
    queue, _ = manager.subscribe(run.id)

    def _boom(event):
        raise PersistenceError("disk full")

    store.append_event = _boom
    with pytest.raises(PersistenceError, match="disk full"):
        manager.append_event(run.id, RunEventType.RUN_STARTED, task=run.task)
    assert not queue.empty(), "live subscribers see events before persistence"


async def test_orchestrator_write_failure_marks_run_failed(tmp_path: Path):
    store = RunStore(tmp_path / "fail.db")
    manager = RunManager(store=store)
    run = manager.create("doomed")

    def _boom(event):
        raise PersistenceError("disk full")

    store.append_event = _boom
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result('{"content":"p","claims":[]}'))
    orchestrator = Orchestrator(build_registry(provider), manager, max_rounds=1)

    with pytest.raises(PersistenceError):
        await orchestrator.execute(run.id)

    failed = manager.get(run.id)
    assert failed.status is RunStatus.FAILED
    assert failed.error.type == "PersistenceError"
    assert "disk full" in failed.error.message
