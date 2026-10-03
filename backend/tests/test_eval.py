"""Eval scoring predicates, aggregate math, and usage capture (Phase 10 §6.5).

Every predicate is exercised against hand-built `RunRecord`s (the offline
policy): a green live eval must rest on checks that never need a model.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from app.eval import (
    PROBLEMS,
    ProblemResult,
    ProblemSpec,
    RecordingProvider,
    UsageTotals,
    build_report,
    refutation_rate,
    rounds_per_problem,
    score_problem,
    summarize,
    verdict_counts,
)
from app.llm.base import ChatMessage, ChatRole, ToolCall
from app.runs import (
    RoundSnapshot,
    RunManager,
    RunStatus,
    StepKind,
    StepRecord,
    StepStatus,
)
from app.schemas import (
    AgentMessage,
    AgentRole,
    Claim,
    ClaimStatus,
    ClaimVerdict,
    Evidence,
    MessageType,
    Verdict,
)
from tests.fakes import FakeProvider

P1, P2, P3 = PROBLEMS


# ------------------------------------------------------------------ helpers


def _run(*, status: RunStatus = RunStatus.COMPLETED, duration: bool = True) -> RunRecord:
    run = RunManager().create("eval task")
    run.status = status
    if duration:
        run.started_at = datetime.now(UTC) - timedelta(seconds=30)
        run.finished_at = datetime.now(UTC)
    return run


def _final(run: RunRecord, content: str) -> None:
    run.final_message = AgentMessage(
        from_agent=AgentRole.MANAGER, type=MessageType.SYNTHESIS, content=content
    )


def _snapshot(
    number: int,
    claims: list[Claim],
    verdicts: list[Verdict],
    *,
    supported: int = 0,
    unresolved: int = 0,
) -> RoundSnapshot:
    return RoundSnapshot(
        round_number=number,
        claims=claims,
        origins={claim.id: AgentRole.RESEARCHER for claim in claims},
        verdicts=verdicts,
        worker_content={},
        skeptic_content="",
        supported_count=supported,
        unresolved_count=unresolved,
    )


def _tool_step(run: RunRecord, name: str, index: int = 1) -> None:
    run.steps.append(
        StepRecord(
            index=index, kind=StepKind.RESEARCH, agent=AgentRole.RESEARCHER,
            status=StepStatus.COMPLETED, started_at=datetime.now(UTC),
            message=AgentMessage(
                from_agent=AgentRole.RESEARCHER, type=MessageType.FINDING,
                content="searched",
                tool_calls=[ToolCall(name=name, arguments={"query": "q"})],
            ),
        )
    )


def _names(score) -> set[str]:
    return {check.name for check in score.mandatory}


def _by_name(checks, name: str):
    return next(check for check in checks if check.name == name)


def _attach_audits(
    run,
    *,
    overall: object = None,
    drop_from_verification: str | None = None,
) -> None:
    """Append scripted VERIFY/AUDIT steps whose reports cover the selected round.

    Mirrors what the real pipeline persists: verification covers every final
    claim exactly once; accountability provenance echoes the selected round.
    """
    from app.orchestrator import select_best_round
    from app.schemas import (
        AccountabilityReport,
        AccountabilityStatus,
        ClaimProvenance,
        ClaimVerification,
        VerificationReport,
        VerificationStatus,
    )

    selected = select_best_round(run.rounds) if run.rounds else None
    verdict_by_id = {v.claim_id: v for v in selected.verdicts} if selected else {}
    entries = []
    provenance = []
    if selected is not None:
        for claim in selected.claims:
            if drop_from_verification is not None and claim.id == drop_from_verification:
                continue
            entries.append(ClaimVerification(
                claim_id=claim.id,
                verification_status=VerificationStatus.UNVERIFIABLE,
                explanation="no authoritative source in this context",
                confidence=0.4,
            ))
            verdict = verdict_by_id.get(claim.id)
            provenance.append(ClaimProvenance(
                claim_id=claim.id,
                origin=selected.origins[claim.id],
                verdict=verdict.verdict if verdict is not None else None,
                evidence_count=len(claim.evidence),
            ))
    now = datetime.now(UTC)
    run.steps.append(StepRecord(
        index=len(run.steps) + 1, kind=StepKind.VERIFY,
        agent=AgentRole.VERIFIER, status=StepStatus.COMPLETED,
        started_at=now, duration_ms=5.0,
        message=AgentMessage(
            from_agent=AgentRole.VERIFIER, type=MessageType.VERIFICATION,
            content="checked the provided records",
            verification=VerificationReport(claims=entries),
        ),
    ))
    run.steps.append(StepRecord(
        index=len(run.steps) + 2, kind=StepKind.AUDIT,
        agent=AgentRole.ACCOUNTABILITY, status=StepStatus.COMPLETED,
        started_at=now, duration_ms=4.0,
        message=AgentMessage(
            from_agent=AgentRole.ACCOUNTABILITY, type=MessageType.ACCOUNTABILITY,
            content="process audit",
            accountability=AccountabilityReport(
                trace_completeness=True,
                final_claim_provenance=provenance,
                flags=[],
                overall_status=(overall or AccountabilityStatus.CLEAN),
                summary="no process gaps found",
            ),
        ),
    ))


# ---------------------------------------------------------------------- P1

def test_p1_full_pass():
    claim = Claim(
        id="c1",
        statement="The 2025 annual growth rate was 40%.",
        status=ClaimStatus.UNVERIFIED,
        evidence=[Evidence(source="company blog")],
    )
    verdict = Verdict(
        claim_id="c1",
        verdict=ClaimVerdict.REFUTED,
        objection="not audited",
        evidence=[Evidence(source="growth_report.txt", quote="23%")],
    )
    run = _run()
    run.rounds.append(_snapshot(1, [claim], [verdict], supported=0, unresolved=1))
    _tool_step(run, "file_search")
    _final(run, "The report says 23%, not 40%.")
    _attach_audits(run)

    score = score_problem(P1, run)
    assert score.passed
    assert {c.name for c in score.mandatory if c.passed} == _names(score)
    assert _by_name(score.advisory, "final_mentions_23").passed
    assert _by_name(score.advisory, "no_fact_lacking_evidence").passed


def test_p1_missing_file_tool_fails_mandatory():
    claim = Claim(id="c1", statement="growth was 40%", status=ClaimStatus.UNVERIFIED)
    verdict = Verdict(
        claim_id="c1", verdict=ClaimVerdict.REFUTED, objection="no source",
        evidence=[Evidence(source="growth_report.txt")],
    )
    run = _run()
    run.rounds.append(_snapshot(1, [claim], [verdict]))
    _final(run, "23%")
    score = score_problem(P1, run)
    assert not score.passed
    assert not _by_name(score.mandatory, "file_tool_used").passed
    assert "tools=[]" in _by_name(score.mandatory, "file_tool_used").detail


def test_p1_refutation_without_file_citation_fails():
    claim = Claim(id="c1", statement="growth was 40%", status=ClaimStatus.UNVERIFIED)
    verdict = Verdict(
        claim_id="c1", verdict=ClaimVerdict.REFUTED, objection="no source",
        evidence=[Evidence(source="some other doc")],
    )
    run = _run()
    run.rounds.append(_snapshot(1, [claim], [verdict]))
    _tool_step(run, "file_reader")
    _final(run, "23%")
    score = score_problem(P1, run)
    assert not score.passed
    assert _by_name(score.mandatory, "falsehood_disproved").passed
    assert not _by_name(score.mandatory, "correction_cites_growth_report").passed
    assert "no claim/verdict evidence cites growth_report" in (
        _by_name(score.mandatory, "correction_cites_growth_report").detail
    )


def test_p1_disproof_claim_shape_passes():
    """The live-run shape (PRD §18 #2): the researcher disproves the figure
    up front; the skeptic supports the disproof against the document."""
    claim = Claim(
        id="c2",
        statement="The claimed annual growth rate of 40% for 2025 is incorrect.",
        status=ClaimStatus.FACT,
        evidence=[Evidence(source="file_search(growth_report.txt)")],
    )
    verdict = Verdict(
        claim_id="c2", verdict=ClaimVerdict.SUPPORTED,
        objection="The document clearly states the correct figure.",
        evidence=[Evidence(source="file_search(growth_report.txt)")],
    )
    run = _run()
    run.rounds.append(_snapshot(1, [claim], [verdict]))
    _tool_step(run, "file_search")
    _final(run, "The correct figure according to growth_report.txt is 23%.")
    _attach_audits(run)
    score = score_problem(P1, run)
    assert score.passed
    assert _by_name(score.mandatory, "falsehood_disproved").passed
    assert "disproof claim c2" in _by_name(
        score.mandatory, "falsehood_disproved"
    ).detail
    assert _by_name(score.mandatory, "correction_cites_growth_report").passed
    assert _by_name(score.advisory, "final_mentions_23").passed


def test_p1_failed_run_fails_and_never_raises():
    run = _run(status=RunStatus.FAILED, duration=False)
    score = score_problem(P1, run)  # empty: rounds, steps, final
    assert not score.passed
    assert not _by_name(score.mandatory, "run_completed").passed
    assert not _by_name(score.mandatory, "final_message_nonempty").passed


# ---------------------------------------------------------------------- P2

def test_p2_honesty_required_when_unresolved():
    claim = Claim(id="c1", statement="Salary will be huge", status=ClaimStatus.HYPOTHESIS)
    verdict = Verdict(claim_id="c1", verdict=ClaimVerdict.UNVERIFIABLE,
                      objection="no data exists")
    run = _run()
    run.rounds.append(_snapshot(1, [claim], [verdict], unresolved=1))
    _final(run, "Around 200k USD — this is only a rough estimate.")
    _attach_audits(run)
    honest = score_problem(P2, run)
    assert _by_name(honest.mandatory, "honesty_cue_when_unresolved").passed
    assert honest.passed

    _final(run, "The average salary will be 310000 USD in 2030.")
    confident = score_problem(P2, run)
    assert not confident.passed
    detail = _by_name(confident.mandatory, "honesty_cue_when_unresolved").detail
    assert "unresolved=1" in detail


def test_p2_honesty_cue_accepts_live_wording():
    """PRD §18 #2: the first live run hedged with 'remains speculative'."""
    claim = Claim(id="c1", statement="s", status=ClaimStatus.HYPOTHESIS)
    verdict = Verdict(claim_id="c1", verdict=ClaimVerdict.UNVERIFIABLE,
                      objection="no data exists")
    run = _run()
    run.rounds.append(_snapshot(1, [claim], [verdict], unresolved=1))
    _final(
        run,
        "Predicting a precise average salary for an AI engineer in 2030 "
        "remains speculative; the claims lack specific data points.",
    )
    _attach_audits(run)
    score = score_problem(P2, run)
    assert score.passed
    assert _by_name(score.mandatory, "honesty_cue_when_unresolved").passed


def test_p2_nothing_unresolved_needs_no_cue():
    claim = Claim(id="c1", statement="s", status=ClaimStatus.FACT,
                  evidence=[Evidence(source="x")])
    verdict = Verdict(claim_id="c1", verdict=ClaimVerdict.SUPPORTED,
                      objection="holds", evidence=[])
    run = _run()
    run.rounds.append(_snapshot(1, [claim], [verdict], supported=1))
    _final(run, "Confident answer with no caveats.")
    _attach_audits(run)
    score = score_problem(P2, run)
    assert score.passed
    assert _by_name(score.mandatory, "honesty_cue_when_unresolved").detail == (
        "nothing unresolved"
    )


def test_p2_bare_number_is_advisory_only():
    claim = Claim(id="c1", statement="s", status=ClaimStatus.HYPOTHESIS)
    verdict = Verdict(claim_id="c1", verdict=ClaimVerdict.UNVERIFIABLE,
                      objection="no data exists")
    run = _run()
    run.rounds.append(_snapshot(1, [claim], [verdict], unresolved=1))
    _final(
        run,
        "There is no data for 2030. The salary will be 250000 dollars in 2030.",
    )
    _attach_audits(run)
    score = score_problem(P2, run)
    # honesty cue appears somewhere -> mandatory green; the bare figure in
    # another sentence is advisory noise that the report records
    assert score.passed
    assert not _by_name(score.advisory, "no_bare_salary_number").passed
    assert not _by_name(score.advisory, "iterated").passed  # 1 round only


# ---------------------------------------------------------------------- P3

def _p3_steps(run: RunRecord, kinds: list[StepKind]) -> None:
    for index, kind in enumerate(kinds, start=1):
        agent = AgentRole.MANAGER if kind in {StepKind.PLAN, StepKind.SYNTHESIZE} else (
            AgentRole.SKEPTIC if kind is StepKind.CRITIQUE else AgentRole.RESEARCHER
        )
        run.steps.append(
            StepRecord(
                index=index, kind=kind, agent=agent,
                status=StepStatus.COMPLETED, started_at=datetime.now(UTC),
                duration_ms=10.0,
            )
        )


def test_p3_ideal_trace_passes_all_checks():
    run = _run()
    _p3_steps(run, [StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE,
                    StepKind.CRITIQUE, StepKind.SYNTHESIZE])
    run.rounds.append(_snapshot(1, [], []))
    _final(run, "Yes — write them down.")
    _attach_audits(run)
    score = score_problem(P3, run)
    assert score.passed
    assert _by_name(score.mandatory, "no_failed_steps").passed
    assert _by_name(score.mandatory, "duration_recorded").passed
    assert _by_name(score.mandatory, "rounds_within_budget").passed
    assert _by_name(score.advisory, "exact_trace").passed
    assert _by_name(score.advisory, "single_round").passed
    assert _by_name(score.advisory, "wall_under_15min").passed


def test_p3_extra_decide_step_is_advisory_model_judgment():
    """A live manager may spend a decision + revision on a simple task; that
    is model judgment (PRD §18 amendment #1), so it is advisory — but the run
    must still complete inside the cost budget."""
    run = _run()
    _p3_steps(run, [StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE,
                    StepKind.CRITIQUE, StepKind.DECIDE, StepKind.SYNTHESIZE])
    run.rounds.append(_snapshot(1, [], []))
    _final(run, "Yes.")
    _attach_audits(run)
    score = score_problem(P3, run)
    assert score.passed
    assert not _by_name(score.advisory, "exact_trace").passed
    assert "decide" in _by_name(score.advisory, "exact_trace").detail


def test_p3_third_round_breaks_the_cost_ceiling():
    run = _run()
    _p3_steps(run, [StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE,
                    StepKind.CRITIQUE, StepKind.SYNTHESIZE])
    for number in (1, 2, 3):
        run.rounds.append(_snapshot(number, [], []))
    _final(run, "Yes.")
    score = score_problem(P3, run)
    assert not score.passed
    ceiling = _by_name(score.mandatory, "rounds_within_budget")
    assert not ceiling.passed
    assert "rounds=3" in ceiling.detail
    assert not _by_name(score.advisory, "single_round").passed


def test_p3_failed_step_fails_mandatory():
    run = _run()
    run.steps.append(
        StepRecord(
            index=1, kind=StepKind.PLAN, agent=AgentRole.MANAGER,
            status=StepStatus.FAILED, started_at=datetime.now(UTC),
        )
    )
    run.rounds.append(_snapshot(1, [], []))
    _final(run, "Yes.")
    score = score_problem(P3, run)
    assert not score.passed
    assert not _by_name(score.mandatory, "no_failed_steps").passed


# ------------------------------------------------------- aggregates & math

def _run_with_verdicts(*verdict_kinds: ClaimVerdict) -> RunRecord:
    claims, verdicts = [], []
    for index, kind in enumerate(verdict_kinds, start=1):
        claim_id = f"c{index}"
        claims.append(Claim(id=claim_id, statement=f"claim {index}",
                            status=ClaimStatus.UNVERIFIED))
        verdicts.append(Verdict(claim_id=claim_id, verdict=kind, objection="o"))
    run = _run()
    run.rounds.append(_snapshot(1, claims, verdicts))
    return run


def test_refutation_rate_math():
    assert refutation_rate([]) == 0.0
    run = _run_with_verdicts(
        ClaimVerdict.REFUTED, ClaimVerdict.REFUTED,
        ClaimVerdict.SUPPORTED, ClaimVerdict.UNVERIFIABLE,
    )
    assert refutation_rate([run]) == pytest.approx(0.5)
    counts = verdict_counts(run)
    assert counts == {"supported": 1, "refuted": 2, "unverifiable": 1, "total": 4}


def test_rounds_per_problem_keys_by_input_id():
    run = _run()
    run.rounds.append(_snapshot(1, [], []))
    assert rounds_per_problem([("p1", run), ("p3", _run())]) == {"p1": 1, "p3": 0}


def test_summarize_and_report_shape():
    claim = Claim(id="c1", statement="40%", status=ClaimStatus.UNVERIFIED)
    verdict = Verdict(claim_id="c1", verdict=ClaimVerdict.REFUTED, objection="x",
                      evidence=[Evidence(source="growth_report.txt")])
    run = _run()
    run.rounds.append(_snapshot(1, [claim], [verdict], unresolved=1))
    _tool_step(run, "file_search")
    _final(run, "23%")
    _attach_audits(run)
    score = score_problem(P1, run)
    result = ProblemResult(
        problem_id="p1", title=P1.title, task=P1.task, run_id=run.id,
        status="completed", attempts=1, rounds=len(run.rounds),
        steps=len(run.steps), verdicts=verdict_counts(run),
        tokens=UsageTotals(prompt_tokens=100, completion_tokens=50,
                           total_tokens=150, calls=3),
        wall_ms=run.duration_ms, mandatory=score.mandatory,
        advisory=score.advisory, passed=score.passed,
    )
    report = build_report(
        [result], app_version="0.1.0", model="qwen2.5:14b-instruct",
        num_ctx=8192, params={"max_rounds": 3},
    )
    payload = json.loads(json.dumps(report.as_dict()))  # round-trippable
    assert payload["model"] == "qwen2.5:14b-instruct"
    assert payload["summary"]["refutation_rate"] == 1.0
    assert payload["summary"]["rounds_per_problem"] == {"p1": 1}
    assert payload["summary"]["tokens_total"] == 150
    assert payload["summary"]["all_passed"] is True
    assert payload["problems"][0]["human_scores"] == {
        "correctness": None, "evidence_use": None, "honesty": None, "notes": "",
    }
    assert payload["problems"][0]["mandatory"][0] == {
        "name": "run_completed", "passed": True, "detail": "completed",
    }


def test_summarize_all_passed_false_when_any_fails():
    failed = _run(status=RunStatus.FAILED, duration=False)
    score = score_problem(P1, failed)
    result = ProblemResult(
        problem_id="p1", title="t", task="t", run_id=failed.id, status="failed",
        attempts=2, rounds=0, steps=0, verdicts=verdict_counts(failed),
        tokens=UsageTotals(), wall_ms=None, mandatory=score.mandatory,
        advisory=score.advisory, passed=score.passed,
    )
    summary = summarize([result])
    assert summary["all_passed"] is False
    assert summary["mandatory_passed"] < summary["mandatory_total"]


# ------------------------------------------------------- RecordingProvider

async def test_recording_provider_accumulates_and_delegates():
    inner = FakeProvider()
    inner.queue_result(FakeProvider.make_result("one", prompt_tokens=10,
                                                completion_tokens=5))
    inner.queue_result(FakeProvider.make_result("two", prompt_tokens=7,
                                                completion_tokens=3))
    provider = RecordingProvider(inner)

    messages = [ChatMessage(role=ChatRole.USER, content="hi")]
    first = await provider.chat(messages, temperature=0.4, max_tokens=64)
    second = await provider.chat(messages)

    assert first.content == "one" and second.content == "two"
    assert provider.totals.prompt_tokens == 17
    assert provider.totals.completion_tokens == 8
    assert provider.totals.total_tokens == 25
    assert provider.totals.calls == 2
    # kwargs pass through untouched
    assert inner.chat_calls[0]["kwargs"]["temperature"] == 0.4
    assert inner.chat_calls[0]["kwargs"]["max_tokens"] == 64
    assert provider.name == "fake"


async def test_recording_provider_delegates_lifecycle():
    inner = FakeProvider()
    provider = RecordingProvider(inner)
    assert await provider.list_models() == []
    assert (await provider.health()).reachable is True
    await provider.aclose()
    assert provider.totals.calls == 0


# --------------------------------------------------------- specs are sane

def test_problem_specs_locked():
    assert [problem.id for problem in PROBLEMS] == ["p1", "p2", "p3"]
    assert all(not problem.web_search for problem in PROBLEMS)
    assert PROBLEMS[0].task.startswith("Verify whether the claimed")
    assert PROBLEMS[2].max_rounds == 3
    assert not ProblemSpec(id="p1", title="x", task="x").seeding_required


# ------------------------------------------ Phase 11 audit predicates (eval)


def _p2_run_with_claim() -> "object":
    run = _run()
    claim = Claim(id="c1", statement="Salary in 2030", status=ClaimStatus.HYPOTHESIS)
    verdict = Verdict(claim_id="c1", verdict=ClaimVerdict.UNVERIFIABLE,
                      objection="no data")
    run.rounds.append(_snapshot(1, [claim], [verdict], unresolved=1))
    _final(run, "Rough estimate only.")
    return run


def test_missing_audit_steps_fail_mandatory():
    run = _p2_run_with_claim()  # no verify/audit steps attached
    score = score_problem(P2, run)
    assert not score.passed
    assert not _by_name(score.mandatory, "verification_report_present").passed
    assert _by_name(
        score.mandatory, "verification_report_present"
    ).detail == "verify step missing"
    assert not _by_name(score.mandatory, "accountability_report_present").passed
    assert not _by_name(score.mandatory, "accountability_no_violations").passed


def test_accountability_violations_fail_mandatory():
    from app.schemas import AccountabilityStatus

    run = _p2_run_with_claim()
    _attach_audits(run, overall=AccountabilityStatus.VIOLATIONS)
    score = score_problem(P2, run)
    check = _by_name(score.mandatory, "accountability_no_violations")
    assert not check.passed
    assert check.detail == "violations"
    assert not score.passed
    # everything else still green: the violation is the only failure
    assert _by_name(score.mandatory, "verification_report_present").passed
    assert _by_name(score.mandatory, "honesty_cue_when_unresolved").passed


def test_verification_coverage_mismatch_fails_mandatory():
    run = _p2_run_with_claim()
    _attach_audits(run, drop_from_verification="c1")
    score = score_problem(P2, run)
    check = _by_name(score.mandatory, "verification_report_present")
    assert not check.passed
    assert "missing=['c1']" in check.detail
    assert not score.passed


def test_audit_advisory_readouts():
    run = _p2_run_with_claim()
    _attach_audits(run)
    score = score_problem(P2, run)
    # scripted verification entries are unverifiable -> advisory reports it
    fully = _by_name(score.advisory, "verification_fully_verified")
    assert not fully.passed
    assert "c1" in fully.detail
    assert _by_name(score.advisory, "accountability_clean").passed
    # advisory never gates
    assert score.passed


def _p1_23_run() -> "object":
    """P1 fixture: the corrected figure plus the refuted falsehood, so every
    mandatory check passes and only the advisory under test can differ."""
    fixed = Claim(
        id="c1",
        statement="The correct 2025 growth rate is 23% per growth_report.txt.",
        status=ClaimStatus.UNVERIFIED,
        evidence=[Evidence(source="growth_report.txt", quote="23%")],
    )
    false = Claim(
        id="c2",
        statement="The claimed annual growth rate of 40% for 2025 is correct.",
        status=ClaimStatus.UNVERIFIED,
        evidence=[Evidence(source="company blog")],
    )
    verdicts = [
        Verdict(
            claim_id="c1",
            verdict=ClaimVerdict.SUPPORTED,
            objection="matches the document",
            evidence=[Evidence(source="growth_report.txt", quote="23%")],
        ),
        Verdict(
            claim_id="c2",
            verdict=ClaimVerdict.REFUTED,
            objection="growth_report.txt says 23%, not 40%",
            evidence=[Evidence(source="growth_report.txt", quote="23%")],
        ),
    ]
    run = _run()
    run.rounds.append(
        _snapshot(1, [fixed, false], verdicts, supported=1, unresolved=1)
    )
    _tool_step(run, "file_search")
    _final(run, "The report says 23%, not 40%.")
    _attach_audits(run)
    return run


def _set_verification_status(run, claim_id: str, status: object) -> None:
    """Rewrite one scripted verification entry's status (audit steps are the
    last two appended by `_attach_audits`)."""
    from app.schemas import ClaimVerification, VerificationReport

    for step in reversed(run.steps):
        if step.kind is StepKind.VERIFY and step.message is not None:
            report = step.message.verification
            assert report is not None
            entries = [
                ClaimVerification(
                    claim_id=entry.claim_id,
                    verification_status=(
                        status if entry.claim_id == claim_id
                        else entry.verification_status
                    ),
                    evidence_checked=entry.evidence_checked,
                    supporting_evidence=entry.supporting_evidence,
                    contradicting_evidence=entry.contradicting_evidence,
                    source_references=entry.source_references,
                    explanation=entry.explanation,
                    confidence=entry.confidence,
                )
                for entry in report.claims
            ]
            step.message = step.message.model_copy(
                update={"verification": VerificationReport(claims=entries)}
            )
            return
    raise AssertionError("verify step missing")


def test_p1_verifier_confirms_correction_when_verified():
    from app.schemas import VerificationStatus

    run = _p1_23_run()
    _set_verification_status(run, "c1", VerificationStatus.VERIFIED)
    score = score_problem(P1, run)
    check = _by_name(score.advisory, "verifier_confirms_correction")
    assert check.passed


def test_p1_verifier_confirms_correction_accepts_partial():
    from app.schemas import VerificationStatus

    run = _p1_23_run()
    _set_verification_status(run, "c1", VerificationStatus.PARTIALLY_VERIFIED)
    score = score_problem(P1, run)
    check = _by_name(score.advisory, "verifier_confirms_correction")
    assert check.passed


def test_p1_verifier_confirms_correction_fails_when_unverified():
    from app.schemas import VerificationStatus

    run = _p1_23_run()  # scripted status is UNVERIFIABLE
    score = score_problem(P1, run)
    check = _by_name(score.advisory, "verifier_confirms_correction")
    assert not check.passed
    assert check.detail == "unverifiable"
    assert not _by_name(score.advisory, "verification_fully_verified").passed
    # advisory never gates the run
    assert score.passed


def test_p1_verifier_confirms_correction_fails_when_contradicted():
    from app.schemas import VerificationStatus

    run = _p1_23_run()
    _set_verification_status(run, "c1", VerificationStatus.CONTRADICTED)
    score = score_problem(P1, run)
    check = _by_name(score.advisory, "verifier_confirms_correction")
    assert not check.passed
    assert check.detail == "contradicted"


def test_p1_verifier_confirms_correction_fails_when_no_23_claim():
    """No final claim mentions the corrected figure -> honest advisory fail."""
    from app.schemas import VerificationStatus

    run = _p1_23_run()
    _set_verification_status(run, "c1", VerificationStatus.VERIFIED)
    # swap the selected claim for one whose statement contains no "23"
    snapshot = run.rounds[0]
    snapshot.claims = [
        Claim(
            id="c1",
            statement="The growth rate was corrected upward.",
            status=ClaimStatus.UNVERIFIED,
            evidence=[Evidence(source="growth_report.txt")],
        )
    ]
    score = score_problem(P1, run)
    check = _by_name(score.advisory, "verifier_confirms_correction")
    assert not check.passed
    assert "no final claim mentions the corrected figure" in check.detail


def test_p1_verifier_confirms_correction_without_audit_data():
    """A run that never reached VERIFY reports the advisory honestly."""
    run = _p1_23_run()
    run.steps = [
        step for step in run.steps if step.kind not in (StepKind.VERIFY, StepKind.AUDIT)
    ]
    score = score_problem(P1, run)
    check = _by_name(score.advisory, "verifier_confirms_correction")
    assert not check.passed
    assert check.detail == "no audit data"
    # mandatory audit presence fails too (the run is incomplete)
    assert not _by_name(score.mandatory, "verification_report_present").passed
