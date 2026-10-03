"""RunStore contract: schema, projections, deep-equal round-trip, recovery (PRD §9).

`_drive_full_run` mirrors what the orchestrator does step-for-step (memory
mutation + store call at the same points); the round-trip test then asserts a
fresh `load_run` deep-equals the in-memory record.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import Settings
from app.db import (
    MIGRATIONS,
    SCHEMA_VERSION,
    PersistenceError,
    RunStore,
    build_run_store,
)
from app.llm.base import ToolCall
from app.runs import (
    ErrorInfo,
    RoundSnapshot,
    RunEvent,
    RunEventType,
    RunRecord,
    RunStatus,
    StepKind,
    StepRecord,
    StepStatus,
)
from app.schemas import (
    AccountabilityReport,
    AccountabilityStatus,
    AgentMessage,
    AgentRole,
    Claim,
    ClaimProvenance,
    ClaimStatus,
    ClaimVerdict,
    ClaimVerification,
    DecisionAction,
    Evidence,
    ManagerDecision,
    MessageType,
    ToolResult,
    Verdict,
    VerificationReport,
    VerificationStatus,
)

T0 = datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC)


def _at(seconds: int) -> datetime:
    return datetime(2026, 9, 30, 12, 0, seconds, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "t.db")


# ---------------------------------------------------------------- schema/init


def test_migration_creates_all_tables(store: RunStore, tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "t.db")
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert {
        "runs", "steps", "messages", "claims", "claim_evidence", "verdicts",
        "verdict_evidence", "tool_calls", "rounds", "events",
    } <= names
    assert version == SCHEMA_VERSION == 2


def test_migration_v1_upgrades_to_v2(tmp_path: Path):
    """A populated pre-Phase-11 DB gains the audit columns with defaults."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(MIGRATIONS[0])  # v1 DDL only
    conn.execute("PRAGMA user_version = 1")
    conn.execute(
        "INSERT INTO runs(id, task, status, created_at, final_message_id)"
        " VALUES (?,?,?,?,?)",
        ("oldrun", "legacy run", "completed", T0.isoformat(), "m1"),
    )
    conn.execute(
        "INSERT INTO messages(id, run_id, from_agent, type, content, created_at)"
        " VALUES (?,?,?,?,?,?)",
        ("m1", "oldrun", "manager", "synthesis", "legacy answer", T0.isoformat()),
    )
    conn.commit()
    conn.close()

    store = RunStore(path)  # triggers v1 -> v2
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
    conn.close()
    assert {"retries", "verification_json", "accountability_json"} <= columns

    # defaults on legacy rows: no retries, no reports
    loaded = store.load_run("oldrun")
    assert loaded is not None
    message = loaded.final_message
    assert message is not None
    assert message.retries == 0
    assert message.verification is None
    assert message.accountability is None

    # re-open is idempotent (user_version already current)
    RunStore(path)


def test_audit_reports_and_retries_round_trip(store: RunStore):
    """Phase 11: verification/accountability/retries persist byte-for-byte."""
    from app.schemas import (
        AccountabilityFlag,
        AccountabilityFlagKind,
        AccountabilityReport,
        AccountabilityStatus,
        ClaimProvenance,
        ClaimVerification,
        ClaimVerdict,
        Evidence,
        FlagSeverity,
        VerificationReport,
        VerificationStatus,
    )

    run = RunRecord(
        id="auditrun", task="audit me", status=RunStatus.RUNNING, created_at=T0,
    )
    store.save_run(run)
    run.started_at = _at(1)
    store.append_event(RunEvent(
        seq=1, type=RunEventType.RUN_STARTED, run_id=run.id, ts=_at(1),
        task=run.task,
    ))

    verification = VerificationReport(claims=[
        ClaimVerification(
            claim_id="c1", verification_status=VerificationStatus.VERIFIED,
            evidence_checked=["growth_report.txt"],
            supporting_evidence=[Evidence(source="growth_report.txt", quote="23%")],
            source_references=["growth_report.txt"],
            explanation="opened the file",
            confidence=0.9,
        ),
    ])
    accountability = AccountabilityReport(
        trace_completeness=True,
        final_claim_provenance=[
            ClaimProvenance(claim_id="c1", origin=AgentRole.RESEARCHER,
                            verdict=ClaimVerdict.SUPPORTED, evidence_count=1),
        ],
        flags=[
            AccountabilityFlag(
                kind=AccountabilityFlagKind.RETRY_ACTIVITY,
                severity=FlagSeverity.INFO, refs=["1"],
                explanation="one structured retry",
            ),
        ],
        overall_status=AccountabilityStatus.WARNINGS,
        summary="clean except one retry",
    )

    verify = StepRecord(
        index=1, kind=StepKind.VERIFY, agent=AgentRole.VERIFIER,
        status=StepStatus.RUNNING, started_at=_at(2),
    )
    run.steps.append(verify)
    store.append_event(RunEvent(
        seq=2, type=RunEventType.STEP_STARTED, run_id=run.id, ts=_at(2),
        step=1, kind=StepKind.VERIFY, agent=AgentRole.VERIFIER,
    ))
    verify.status = StepStatus.COMPLETED
    verify.duration_ms = 5.0
    verify.message = _msg(
        from_agent=AgentRole.VERIFIER, type=MessageType.VERIFICATION,
        content="verified", verification=verification, retries=1,
    )
    store.append_event(RunEvent(
        seq=3, type=RunEventType.STEP_COMPLETED, run_id=run.id, ts=_at(3),
        step=1, kind=StepKind.VERIFY, agent=AgentRole.VERIFIER,
        duration_ms=5.0, message=verify.message,
    ))

    audit = StepRecord(
        index=2, kind=StepKind.AUDIT, agent=AgentRole.ACCOUNTABILITY,
        status=StepStatus.RUNNING, started_at=_at(4),
    )
    run.steps.append(audit)
    store.append_event(RunEvent(
        seq=4, type=RunEventType.STEP_STARTED, run_id=run.id, ts=_at(4),
        step=2, kind=StepKind.AUDIT, agent=AgentRole.ACCOUNTABILITY,
    ))
    audit.status = StepStatus.COMPLETED
    audit.duration_ms = 4.0
    audit.message = _msg(
        from_agent=AgentRole.ACCOUNTABILITY, type=MessageType.ACCOUNTABILITY,
        content="audit", accountability=accountability,
    )
    store.append_event(RunEvent(
        seq=5, type=RunEventType.STEP_COMPLETED, run_id=run.id, ts=_at(5),
        step=2, kind=StepKind.AUDIT, agent=AgentRole.ACCOUNTABILITY,
        duration_ms=4.0, message=audit.message,
    ))

    loaded = RunStore(store._path).load_run("auditrun")
    assert loaded is not None
    loaded_verify = loaded.steps[0].message
    assert loaded_verify is not None
    assert loaded_verify.verification == verification
    assert loaded_verify.retries == 1
    loaded_audit = loaded.steps[1].message
    assert loaded_audit is not None
    assert loaded_audit.accountability == accountability
    assert loaded_audit.retries == 0


def test_reinit_is_idempotent(tmp_path: Path):
    path = tmp_path / "t.db"
    RunStore(path)
    RunStore(path)  # second open must not re-run migrations


def test_newer_schema_rejected(tmp_path: Path):
    path = tmp_path / "t.db"
    RunStore(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(PersistenceError, match="newer than this build"):
        RunStore(path)


def test_corrupt_file_rejected(tmp_path: Path):
    path = tmp_path / "notadb.db"
    path.write_text("this is not a database")
    with pytest.raises(PersistenceError):
        RunStore(path)


def test_uncreatable_parent_rejected(tmp_path: Path):
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    with pytest.raises(PersistenceError) as exc:
        RunStore(blocker / "sub" / "t.db")
    assert "AI_NEXUS_DB_PATH" in exc.value.hint


def test_build_run_store_toggle(tmp_path: Path):
    assert build_run_store(Settings(db_path="")) is None
    store = build_run_store(
        Settings(db_path=str(tmp_path / "x.db"), db_retention_runs=7)
    )
    assert isinstance(store, RunStore)
    assert store._retention_runs == 7


def test_migration_sql_is_versioned():
    assert len(MIGRATIONS) == SCHEMA_VERSION


# ------------------------------------------------------- full round-trip drive


def _msg(**overrides) -> AgentMessage:
    base = {"created_at": T0}
    base.update(overrides)
    return AgentMessage(**base)


def _drive_full_run(store: RunStore, run_id: str = "run1") -> RunRecord:
    """Create -> run 2 rounds -> complete, exactly like the orchestrator."""
    run = RunRecord(
        id=run_id, task="verify the 40% claim", status=RunStatus.RUNNING,
        created_at=T0,
    )
    store.save_run(run)

    def emit(event: RunEvent) -> None:
        run.events.append(event)      # RunManager does this before persisting
        store.append_event(event)

    # --- RUN_STARTED + PLAN ------------------------------------------------
    run.started_at = _at(1)
    emit(RunEvent(seq=1, type=RunEventType.RUN_STARTED, run_id=run_id, ts=_at(1),
                  task=run.task))

    plan = StepRecord(index=1, kind=StepKind.PLAN, agent=AgentRole.MANAGER,
                      status=StepStatus.RUNNING, started_at=_at(2))
    run.steps.append(plan)
    emit(RunEvent(seq=2, type=RunEventType.STEP_STARTED, run_id=run_id, ts=_at(2),
                  step=1, kind=StepKind.PLAN, agent=AgentRole.MANAGER))
    plan.status = StepStatus.COMPLETED
    plan.duration_ms = 120.5
    plan.message = _msg(from_agent=AgentRole.MANAGER, type=MessageType.PLAN,
                        content="a plan", claims=[
                            Claim(id="c1", statement="plan step one",
                                  status=ClaimStatus.ASSUMPTION),
                        ])
    emit(RunEvent(seq=3, type=RunEventType.STEP_COMPLETED, run_id=run_id, ts=_at(3),
                  step=1, kind=StepKind.PLAN, agent=AgentRole.MANAGER,
                  duration_ms=120.5, message=plan.message))

    # --- RESEARCH with claims + evidence ----------------------------------
    research = StepRecord(index=2, kind=StepKind.RESEARCH,
                          agent=AgentRole.RESEARCHER, status=StepStatus.RUNNING,
                          started_at=_at(4), round=1)
    run.steps.append(research)
    emit(RunEvent(seq=4, type=RunEventType.STEP_STARTED, run_id=run_id, ts=_at(4),
                  step=2, kind=StepKind.RESEARCH, agent=AgentRole.RESEARCHER,
                  round=1))
    research.status = StepStatus.COMPLETED
    research.duration_ms = 1500.0
    research.message = _msg(
        from_agent=AgentRole.RESEARCHER, type=MessageType.FINDING,
        content="findings",
        claims=[
            Claim(id="c1", statement="growth was 23% in 2025",
                  status=ClaimStatus.FACT, confidence=0.9,
                  evidence=[Evidence(source="growth_report.txt",
                                     quote="Annual growth was 23%")]),
            Claim(id="c2", statement="the report is unaudited",
                  status=ClaimStatus.ASSUMPTION, confidence=None, evidence=[]),
        ],
    )
    emit(RunEvent(seq=5, type=RunEventType.STEP_COMPLETED, run_id=run_id,
                  ts=_at(5), step=2, kind=StepKind.RESEARCH,
                  agent=AgentRole.RESEARCHER, round=1, duration_ms=1500.0,
                  message=research.message))

    # --- CRITIQUE with verdicts + evidence --------------------------------
    critique = StepRecord(index=3, kind=StepKind.CRITIQUE,
                          agent=AgentRole.SKEPTIC, status=StepStatus.RUNNING,
                          started_at=_at(6), round=1)
    run.steps.append(critique)
    emit(RunEvent(seq=6, type=RunEventType.STEP_STARTED, run_id=run_id, ts=_at(6),
                  step=3, kind=StepKind.CRITIQUE, agent=AgentRole.SKEPTIC,
                  round=1))
    critique.status = StepStatus.COMPLETED
    critique.duration_ms = 900.25
    critique.message = _msg(
        from_agent=AgentRole.SKEPTIC, type=MessageType.CRITIQUE,
        content="critique",
        verdicts=[
            Verdict(claim_id="c1", verdict=ClaimVerdict.SUPPORTED,
                    objection="source cited",
                    evidence=[Evidence(source="growth_report.txt")]),
            Verdict(claim_id="c2", verdict=ClaimVerdict.UNVERIFIABLE,
                    objection="no audit trail", evidence=[]),
        ],
    )
    emit(RunEvent(seq=7, type=RunEventType.STEP_COMPLETED, run_id=run_id,
                  ts=_at(7), step=3, kind=StepKind.CRITIQUE,
                  agent=AgentRole.SKEPTIC, round=1, duration_ms=900.25,
                  message=critique.message))

    # --- round 1 snapshot + decision ---------------------------------------
    round1 = RoundSnapshot(
        round_number=1,
        claims=list(research.message.claims),
        origins={"c1": AgentRole.RESEARCHER, "c2": AgentRole.RESEARCHER},
        verdicts=list(critique.message.verdicts),
        worker_content={AgentRole.RESEARCHER: "findings", AgentRole.IDEATOR: ""},
        skeptic_content="critique",
        supported_count=1, unresolved_count=1,
    )
    run.rounds.append(round1)
    store.record_round(run_id, round1)
    decision = ManagerDecision(
        action=DecisionAction.CALL_AGENT, target=AgentRole.RESEARCHER,
        instruction="find an audit source", reason="c2 unresolved", confidence=0.7,
    )
    round1.decision = decision
    store.record_round_decision(run_id, 1, decision)

    # --- DECIDE step (decision message) + REVISE with tool trace ----------
    decide = StepRecord(index=4, kind=StepKind.DECIDE, agent=AgentRole.MANAGER,
                        status=StepStatus.RUNNING, started_at=_at(8), round=1)
    run.steps.append(decide)
    emit(RunEvent(seq=8, type=RunEventType.STEP_STARTED, run_id=run_id, ts=_at(8),
                  step=4, kind=StepKind.DECIDE, agent=AgentRole.MANAGER, round=1))
    decide.status = StepStatus.COMPLETED
    decide.duration_ms = 50.0
    decide.message = _msg(from_agent=AgentRole.MANAGER,
                          type=MessageType.DECISION, content="",
                          decision=decision)
    emit(RunEvent(seq=9, type=RunEventType.STEP_COMPLETED, run_id=run_id,
                  ts=_at(9), step=4, kind=StepKind.DECIDE,
                  agent=AgentRole.MANAGER, round=1, duration_ms=50.0,
                  message=decide.message))

    revise = StepRecord(index=5, kind=StepKind.REVISE,
                        agent=AgentRole.RESEARCHER, status=StepStatus.RUNNING,
                        started_at=_at(10), round=2)
    run.steps.append(revise)
    emit(RunEvent(seq=10, type=RunEventType.STEP_STARTED, run_id=run_id,
                  ts=_at(10), step=5, kind=StepKind.REVISE,
                  agent=AgentRole.RESEARCHER, round=2))
    revise.status = StepStatus.COMPLETED
    revise.duration_ms = 2000.0
    revise.message = _msg(
        from_agent=AgentRole.RESEARCHER, type=MessageType.REVISION,
        content="revised",
        claims=[Claim(id="c1", statement="audit found: growth was 23%",
                      status=ClaimStatus.FACT,
                      evidence=[Evidence(source="audit.pdf")])],
        tool_calls=[ToolCall(name="file_search",
                             arguments={"query": "audit 2025"}, id="call_9")],
        tool_results=[ToolResult(
            name="file_search",
            content='{"ok":true,"tool":"file_search","matches":[]}',
            error=None, duration_ms=3.5,
        )],
    )
    emit(RunEvent(seq=11, type=RunEventType.STEP_COMPLETED, run_id=run_id,
                  ts=_at(11), step=5, kind=StepKind.REVISE,
                  agent=AgentRole.RESEARCHER, round=2, duration_ms=2000.0,
                  message=revise.message))

    # --- skipped step (forced finish) + empty-claims message ---------------
    skipped = StepRecord(index=6, kind=StepKind.DECIDE, agent=AgentRole.MANAGER,
                         status=StepStatus.SKIPPED, started_at=_at(12),
                         skipped=True, duration_ms=0.0, round=2)
    run.steps.append(skipped)
    emit(RunEvent(seq=12, type=RunEventType.STEP_COMPLETED, run_id=run_id,
                  ts=_at(12), step=6, kind=StepKind.DECIDE,
                  agent=AgentRole.MANAGER, round=2, skipped=True,
                  duration_ms=0.0))

    synth = StepRecord(index=7, kind=StepKind.SYNTHESIZE,
                       agent=AgentRole.MANAGER, status=StepStatus.RUNNING,
                       started_at=_at(13))
    run.steps.append(synth)
    emit(RunEvent(seq=13, type=RunEventType.STEP_STARTED, run_id=run_id,
                  ts=_at(13), step=7, kind=StepKind.SYNTHESIZE,
                  agent=AgentRole.MANAGER))
    synth.status = StepStatus.COMPLETED
    synth.duration_ms = 400.0
    synth.message = _msg(from_agent=AgentRole.MANAGER,
                         type=MessageType.SYNTHESIS, content="final answer",
                         claims=[])          # empty list, NOT None
    emit(RunEvent(seq=14, type=RunEventType.STEP_COMPLETED, run_id=run_id,
                  ts=_at(14), step=7, kind=StepKind.SYNTHESIZE,
                  agent=AgentRole.MANAGER, duration_ms=400.0,
                  message=synth.message))

    # --- round 2 snapshot (no decision yet) --------------------------------
    round2 = RoundSnapshot(
        round_number=2,
        claims=[claim.model_copy() for claim in revise.message.claims],
        origins={"c1": AgentRole.RESEARCHER},
        verdicts=[Verdict(claim_id="c1", verdict=ClaimVerdict.SUPPORTED,
                          objection="audit cited",
                          evidence=[Evidence(source="audit.pdf")])],
        worker_content={AgentRole.RESEARCHER: "revised", AgentRole.IDEATOR: ""},
        skeptic_content="critique",
        supported_count=2, unresolved_count=0,
    )
    run.rounds.append(round2)
    store.record_round(run_id, round2)

    # --- VERIFY (Phase 11): report + retries persist ------------------------
    verify = StepRecord(index=8, kind=StepKind.VERIFY,
                         agent=AgentRole.VERIFIER, status=StepStatus.RUNNING,
                         started_at=_at(14))
    run.steps.append(verify)
    emit(RunEvent(seq=15, type=RunEventType.STEP_STARTED, run_id=run_id,
                  ts=_at(14), step=8, kind=StepKind.VERIFY,
                  agent=AgentRole.VERIFIER))
    verify.status = StepStatus.COMPLETED
    verify.duration_ms = 750.0
    verify.message = _msg(
        from_agent=AgentRole.VERIFIER, type=MessageType.VERIFICATION,
        content="checked the audit",
        verification=VerificationReport(claims=[
            ClaimVerification(
                claim_id="c1",
                verification_status=VerificationStatus.VERIFIED,
                evidence_checked=["audit.pdf"],
                supporting_evidence=[Evidence(source="audit.pdf",
                                              quote="23%")],
                contradicting_evidence=[],
                source_references=["audit.pdf"],
                explanation="the audit states 23%",
                confidence=0.9,
            ),
        ]),
        retries=1,
    )
    emit(RunEvent(seq=16, type=RunEventType.STEP_COMPLETED, run_id=run_id,
                  ts=_at(15), step=8, kind=StepKind.VERIFY,
                  agent=AgentRole.VERIFIER, duration_ms=750.0,
                  message=verify.message))

    # --- AUDIT (Phase 11): enforced report persists -------------------------
    audit = StepRecord(index=9, kind=StepKind.AUDIT,
                        agent=AgentRole.ACCOUNTABILITY,
                        status=StepStatus.RUNNING, started_at=_at(16))
    run.steps.append(audit)
    emit(RunEvent(seq=17, type=RunEventType.STEP_STARTED, run_id=run_id,
                  ts=_at(16), step=9, kind=StepKind.AUDIT,
                  agent=AgentRole.ACCOUNTABILITY))
    audit.status = StepStatus.COMPLETED
    audit.duration_ms = 300.0
    audit.message = _msg(
        from_agent=AgentRole.ACCOUNTABILITY, type=MessageType.ACCOUNTABILITY,
        content="trace audit",
        accountability=AccountabilityReport(
            trace_completeness=True,
            final_claim_provenance=[
                ClaimProvenance(claim_id="c1", origin=AgentRole.RESEARCHER,
                                verdict=ClaimVerdict.SUPPORTED,
                                evidence_count=1),
            ],
            flags=[],
            overall_status=AccountabilityStatus.CLEAN,
            summary="no process gaps found",
        ),
    )
    emit(RunEvent(seq=18, type=RunEventType.STEP_COMPLETED, run_id=run_id,
                  ts=_at(17), step=9, kind=StepKind.AUDIT,
                  agent=AgentRole.ACCOUNTABILITY, duration_ms=300.0,
                  message=audit.message))

    # --- RUN_COMPLETED ------------------------------------------------------
    run.final_message = synth.message
    run.status = RunStatus.COMPLETED
    run.finished_at = _at(18)
    emit(RunEvent(seq=19, type=RunEventType.RUN_COMPLETED, run_id=run_id,
                  ts=_at(18), message=run.final_message, duration_ms=15000.0))
    return run


def test_full_record_round_trip(store: RunStore):
    run = _drive_full_run(store)
    loaded = store.load_run(run.id)
    assert loaded is not None
    assert loaded == run


def test_round_trip_across_fresh_store_instance(tmp_path: Path):
    path = tmp_path / "restart.db"
    run = _drive_full_run(RunStore(path))
    # "restart": brand-new store over the same file
    loaded = RunStore(path).load_run(run.id)
    assert loaded == run


def test_round_trip_failed_run(store: RunStore):
    run = RunRecord(id="bad", task="doomed", status=RunStatus.RUNNING,
                    created_at=T0)
    store.save_run(run)
    run.started_at = _at(1)
    event1 = RunEvent(seq=1, type=RunEventType.RUN_STARTED, run_id="bad",
                      ts=_at(1), task=run.task)
    run.events.append(event1)
    store.append_event(event1)
    step = StepRecord(index=1, kind=StepKind.RESEARCH,
                      agent=AgentRole.RESEARCHER, status=StepStatus.RUNNING,
                      started_at=_at(2), round=1)
    run.steps.append(step)
    started = RunEvent(seq=2, type=RunEventType.STEP_STARTED, run_id="bad",
                       ts=_at(2), step=1, kind=StepKind.RESEARCH,
                       agent=AgentRole.RESEARCHER, round=1)
    run.events.append(started)
    store.append_event(started)
    step.status = StepStatus.FAILED
    step.error = ErrorInfo(type="RequestTimeoutError", message="timed out",
                           hint="raise timeout")
    run.status = RunStatus.FAILED
    run.error = ErrorInfo(type="RequestTimeoutError", message="timed out",
                          hint="raise timeout")
    run.finished_at = _at(3)
    fail = RunEvent(seq=3, type=RunEventType.RUN_FAILED, run_id="bad", ts=_at(3),
                    step=1, kind=StepKind.RESEARCH, agent=AgentRole.RESEARCHER,
                    round=1, error=run.error)
    run.events.append(fail)
    store.append_event(fail)

    loaded = store.load_run("bad")
    assert loaded == run
    assert loaded.status is RunStatus.FAILED
    assert loaded.steps[0].status is StepStatus.FAILED


def test_empty_claims_list_survives_vs_none(store: RunStore):
    run = _drive_full_run(store, "nulls")
    loaded = store.load_run("nulls")
    synth_msg = next(
        step.message for step in loaded.steps if step.kind is StepKind.SYNTHESIZE
    )
    assert synth_msg.claims == []            # preserved as empty list
    assert synth_msg.verdicts is None        # preserved as None
    assert loaded.steps[0].message.verdicts is None
    assert loaded.steps[1].message.claims is not None


def test_audit_reports_and_retries_survive_full_trace(store: RunStore):
    """The realistic full trace round-trips both Phase 11 reports (§11.7)."""
    run = _drive_full_run(store, "audited")
    loaded = store.load_run("audited")
    verify = next(s for s in loaded.steps if s.kind is StepKind.VERIFY)
    assert verify.message is not None
    assert verify.message.verification == run.steps[-2].message.verification
    assert verify.message.retries == 1
    audit = next(s for s in loaded.steps if s.kind is StepKind.AUDIT)
    assert audit.message is not None
    assert audit.message.accountability == run.steps[-1].message.accountability
    assert audit.message.accountability.overall_status is (
        AccountabilityStatus.CLEAN
    )


def test_shared_final_and_step_message_stored_once(store: RunStore):
    _drive_full_run(store)
    conn = sqlite3.connect(store._path)
    count = conn.execute("SELECT COUNT(*) FROM messages WHERE run_id='run1'")\
        .fetchone()[0]
    conn.close()
    # 9 steps, of which one is message-less (skipped decide) and the final
    # message is the synthesize message again -> 8 distinct rows
    assert count == 8


# ---------------------------------------------------------------- event reads


def test_load_events_since(store: RunStore):
    _drive_full_run(store)
    all_events = store.load_events("run1")
    assert [e.seq for e in all_events] == list(range(1, 20))
    tail = store.load_events("run1", since_seq=18)
    assert [e.seq for e in tail] == [19]
    assert tail[-1].type is RunEventType.RUN_COMPLETED
    assert tail[-1].message.content == "final answer"


def test_list_summaries_are_stubs(store: RunStore):
    _drive_full_run(store, "older")
    store.save_run(RunRecord(id="newer", task="t2", status=RunStatus.RUNNING,
                             created_at=datetime(2026, 9, 30, 13, 0, tzinfo=UTC)))
    summaries = store.list_summaries(10)
    assert [s.id for s in summaries] == ["newer", "older"]  # created_at desc
    stub = summaries[0]
    assert stub.status is RunStatus.RUNNING
    assert stub.steps == [] and stub.events == [] and stub.rounds == []
    assert stub.final_message is None and stub.error is None
    assert stub.duration_ms is None


def test_load_unknown_run_returns_none(store: RunStore):
    assert store.load_run("nope") is None


# ------------------------------------------------------------------ projections


def test_step_started_then_completed_updates_row(store: RunStore):
    store.save_run(RunRecord(id="r", task="t", status=RunStatus.RUNNING,
                             created_at=T0))
    store.append_event(RunEvent(seq=1, type=RunEventType.STEP_STARTED,
                                run_id="r", ts=_at(1), step=1,
                                kind=StepKind.RESEARCH,
                                agent=AgentRole.RESEARCHER, round=1))
    msg = _msg(from_agent=AgentRole.RESEARCHER, type=MessageType.FINDING,
               content="done")
    store.append_event(RunEvent(seq=2, type=RunEventType.STEP_COMPLETED,
                                run_id="r", ts=_at(5), step=1,
                                kind=StepKind.RESEARCH,
                                agent=AgentRole.RESEARCHER, round=1,
                                duration_ms=4000.0, message=msg))
    loaded = store.load_run("r")
    step = loaded.steps[0]
    assert step.status is StepStatus.COMPLETED
    assert step.duration_ms == 4000.0
    assert step.started_at == _at(1), "STEP_STARTED timestamp must survive"
    assert step.message.content == "done"


def test_run_failed_marks_running_steps_failed(store: RunStore):
    store.save_run(RunRecord(id="r", task="t", status=RunStatus.RUNNING,
                             created_at=T0))
    store.append_event(RunEvent(seq=1, type=RunEventType.STEP_STARTED,
                                run_id="r", ts=_at(1), step=1,
                                kind=StepKind.CRITIQUE, agent=AgentRole.SKEPTIC))
    error = ErrorInfo(type="ServerShutdown", message="cancelled")
    store.append_event(RunEvent(seq=2, type=RunEventType.RUN_FAILED, run_id="r",
                                ts=_at(2), error=error))
    loaded = store.load_run("r")
    assert loaded.status is RunStatus.FAILED
    assert loaded.steps[0].status is StepStatus.FAILED
    assert loaded.error.type == "ServerShutdown"


def test_cascade_delete_removes_children(store: RunStore):
    _drive_full_run(store)
    conn = sqlite3.connect(store._path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("DELETE FROM runs WHERE id = 'run1'")
    conn.commit()
    for table in ("steps", "messages", "claims", "verdicts", "tool_calls",
                  "rounds", "events"):
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table}"  # noqa: S608 - fixed table names
        ).fetchone()[0]
        assert count == 0, table
    conn.close()


def test_load_corrupt_snapshot_raises(store: RunStore):
    store.save_run(RunRecord(id="r", task="t", status=RunStatus.RUNNING,
                             created_at=T0))
    conn = sqlite3.connect(store._path)
    conn.execute(
        "INSERT INTO rounds(run_id, round_number, supported_count,"
        " unresolved_count, decision_json, snapshot_json)"
        " VALUES ('r', 1, 0, 0, NULL, 'not json')"
    )
    conn.commit()
    conn.close()
    with pytest.raises(PersistenceError):
        store.load_run("r")


def test_event_projection_requires_step_identity(store: RunStore):
    store.save_run(RunRecord(id="r", task="t", status=RunStatus.RUNNING,
                             created_at=T0))
    with pytest.raises(PersistenceError, match="step event missing"):
        store.append_event(RunEvent(seq=1, type=RunEventType.STEP_STARTED,
                                    run_id="r", ts=_at(1)))


# ------------------------------------------------------------- round recording


def test_record_round_and_decision(store: RunStore):
    store.save_run(RunRecord(id="r", task="t", status=RunStatus.RUNNING,
                             created_at=T0))
    snapshot = RoundSnapshot(
        round_number=1, claims=[], origins={}, verdicts=[],
        worker_content={}, skeptic_content="s", supported_count=0,
        unresolved_count=2,
    )
    store.record_round("r", snapshot)
    decision = ManagerDecision(action=DecisionAction.FINISH,
                               reason="done", confidence=0.9)
    store.record_round_decision("r", 1, decision)
    loaded = store.load_run("r")
    assert loaded.rounds[0].decision == decision
    assert loaded.rounds[0].supported_count == 0


def test_record_round_unknown_run_fails(store: RunStore):
    with pytest.raises(PersistenceError):
        store.record_round("ghost", RoundSnapshot(
            round_number=1, claims=[], origins={}, verdicts=[],
            worker_content={}, skeptic_content="", supported_count=0,
            unresolved_count=0,
        ))


def test_record_decision_unknown_round_fails(store: RunStore):
    store.save_run(RunRecord(id="r", task="t", status=RunStatus.RUNNING,
                             created_at=T0))
    with pytest.raises(PersistenceError, match="not found for decision"):
        store.record_round_decision("r", 5, ManagerDecision(
            action=DecisionAction.FINISH, reason="x", confidence=0.1,
        ))


# ------------------------------------------------------------------- recovery


def test_recover_interrupted_marks_failed_with_terminal_event(store: RunStore):
    store.save_run(RunRecord(id="stuck", task="t", status=RunStatus.RUNNING,
                             created_at=T0))
    store.append_event(RunEvent(seq=1, type=RunEventType.RUN_STARTED,
                                run_id="stuck", ts=_at(1), task="t"))
    store.append_event(RunEvent(seq=2, type=RunEventType.STEP_STARTED,
                                run_id="stuck", ts=_at(2), step=2,
                                kind=StepKind.CRITIQUE, agent=AgentRole.SKEPTIC,
                                round=1))
    # a completed run must never be touched
    store.save_run(RunRecord(id="fine", task="ok",
                             status=RunStatus.COMPLETED, created_at=_at(50),
                             started_at=_at(51), finished_at=_at(52)))

    recovered = store.recover_interrupted()
    assert recovered == ["stuck"]
    assert store.recover_interrupted() == [], "recovery is not repeatable"

    run = store.load_run("stuck")
    assert run.status is RunStatus.FAILED
    assert run.error.type == "ServerRestart"
    assert run.finished_at is not None
    assert run.steps[0].status is StepStatus.FAILED
    last = run.events[-1]
    assert last.seq == 3
    assert last.type is RunEventType.RUN_FAILED
    assert last.step == 2 and last.round == 1
    assert store.load_run("fine").status is RunStatus.COMPLETED


# ---------------------------------------------------------------------- prune


def _make_run(store: RunStore, run_id: str, created_offset: int) -> None:
    store.save_run(RunRecord(id=run_id, task=f"t-{run_id}",
                             status=RunStatus.COMPLETED,
                             created_at=_at(created_offset),
                             started_at=_at(created_offset),
                             finished_at=_at(created_offset)))


def test_prune_keeps_newest_and_cascades(store: RunStore):
    for index in range(4):
        _make_run(store, f"r{index}", index)
    deleted = store.prune(2)
    assert deleted == 2
    assert {s.id for s in store.list_summaries(10)} == {"r2", "r3"}
    conn = sqlite3.connect(store._path)
    orphans = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close()
    assert orphans == 0


def test_prune_zero_keeps_everything(store: RunStore):
    for index in range(3):
        _make_run(store, f"r{index}", index)
    assert store.prune(0) == 0
    assert len(store.list_summaries(10)) == 3


def test_prune_never_touches_running(store: RunStore):
    store.save_run(RunRecord(id="live", task="t", status=RunStatus.RUNNING,
                             created_at=T0))
    _make_run(store, "old", 1)
    # keep=1: the single finished run is within budget; the running run is
    # excluded from the prune predicate entirely
    assert store.prune(1) == 0
    assert {s.id for s in store.list_summaries(10)} == {"live", "old"}


def test_retention_applies_at_init(tmp_path: Path):
    path = tmp_path / "ret.db"
    first = RunStore(path, retention_runs=10)
    for index in range(5):
        _make_run(first, f"r{index}", index)
    RunStore(path, retention_runs=2)
    remaining = {s.id for s in RunStore(path).list_summaries(10)}
    assert remaining == {"r3", "r4"}


# --------------------------------------------------------------------- guards


def test_save_run_duplicate_id_fails_typed(store: RunStore):
    run = RunRecord(id="dup", task="t", status=RunStatus.RUNNING, created_at=T0)
    store.save_run(run)
    with pytest.raises(PersistenceError):
        store.save_run(run)


def test_append_event_unknown_run_fails_typed(store: RunStore):
    with pytest.raises(PersistenceError):
        store.append_event(RunEvent(seq=1, type=RunEventType.RUN_STARTED,
                                    run_id="ghost", ts=T0, task="t"))
