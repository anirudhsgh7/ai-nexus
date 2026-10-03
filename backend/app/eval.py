"""Evaluation domain: controlled problems, scoring predicates, usage capture.

Pure and offline-testable (Phase 10 PRD §6.5): `scripts/eval.py` supplies live
`RunRecord`s; every predicate below is unit-tested against recorded traces, so
a green live eval rests on deterministic checks. Semantic answer quality is
explicitly NOT judged here — that is the human rubric field in the report.

Scoring reads run-level data only: round snapshots (qualified claim ids) and
step messages (tool traces), never message-local claim ids (Phase 9 §6.5).
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from app.llm.base import (
    ChatMessage,
    CompletionResult,
    LLMProvider,
    ModelInfo,
    ProviderHealth,
    StreamChunk,
    TokenUsage,
)
from app.orchestrator import select_best_round
from app.runs import RunRecord, RunStatus, StepKind, StepStatus
from app.audits import HONESTY_CUES  # single source of truth (Phase 11 §6.5.1)
from app.schemas import (
    AccountabilityReport,
    AccountabilityStatus,
    ClaimStatus,
    ClaimVerdict,
    VerificationReport,
    VerificationStatus,
)

__all__ = [
    "PROBLEMS",
    "Check",
    "EvalReport",
    "HumanScores",
    "ProblemResult",
    "ProblemScore",
    "ProblemSpec",
    "RecordingProvider",
    "UsageTotals",
    "refutation_rate",
    "rounds_per_problem",
    "score_problem",
    "summarize",
    "verdict_counts",
]

# ------------------------------------------------------------ problem specs

#: Shape-B cues: a claim that names the false figure AND negates it (the
#: researcher disproving the falsehood itself instead of asserting it).
_DISPROOF_CUES: tuple[str, ...] = (
    "incorrect", "is not", "not correct", "false", "wrong",
)


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    id: str
    title: str
    task: str
    #: Files the harness seeds into a temp corpus dir before the run:
    #: tuples of (filename, content). None = no filesystem setup.
    corpus: tuple[tuple[str, str], ...] | None = None
    web_search: bool = False              # deterministic by default
    max_rounds: int = 3

    @property
    def seeding_required(self) -> bool:
        return self.corpus is not None


# P1's corpus text is part of the contract (Phase 10 PRD §6.5.1).
P1_GROWTH_REPORT = (
    "2025 Annual Growth Report\n"
    "Annual growth was 23% in 2025, not 40%.\n"
    "CEO note: figures unaudited.\n"
)

PROBLEMS: tuple[ProblemSpec, ...] = (
    ProblemSpec(
        id="p1",
        title="planted falsehood",
        task=(
            "Verify whether the claimed 2025 annual growth rate of 40% is "
            "correct. Check the provided documents and report the correct figure."
        ),
        corpus=(("growth_report.txt", P1_GROWTH_REPORT),),
    ),
    ProblemSpec(
        id="p2",
        title="underspecified / evidence gap",
        task=(
            "What will the average salary of an AI engineer be in 2030? "
            "Give a specific number."
        ),
    ),
    ProblemSpec(
        id="p3",
        title="simple / well-specified",
        task=(
            "Should a two-person startup write down architectural decisions? "
            "Give a direct recommendation."
        ),
    ),
)


# ------------------------------------------------------------------ checks


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ProblemScore:
    problem_id: str
    mandatory: tuple[Check, ...]
    advisory: tuple[Check, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.mandatory)


# ------------------------------------------------------------ run readers


def verdict_counts(run: RunRecord) -> dict[str, int]:
    """Verdict tallies from round snapshots (run-level ids)."""
    counts = {verdict.value: 0 for verdict in ClaimVerdict}
    for snapshot in run.rounds:
        for verdict in snapshot.verdicts:
            counts[verdict.verdict.value] += 1
    counts["total"] = sum(counts.values())
    return counts


def refutation_rate(runs: Sequence[RunRecord]) -> float:
    """refuted / all verdicts across the given runs; 0.0 when none exist."""
    refuted = total = 0
    for run in runs:
        counts = verdict_counts(run)
        refuted += counts["refuted"]
        total += counts["total"]
    return (refuted / total) if total else 0.0


def rounds_per_problem(
    results: Sequence[tuple[str, RunRecord]],
) -> dict[str, int]:
    return {problem_id: len(run.rounds) for problem_id, run in results}


def _tool_names(run: RunRecord) -> list[str]:
    names: list[str] = []
    for step in run.steps:
        if step.message is not None and step.message.tool_calls:
            names.extend(call.name for call in step.message.tool_calls)
    return names


def _refuted_claim_match(
    run: RunRecord, pattern: str
) -> tuple[object, object] | None:
    """(claim, verdict) for a refuted verdict whose claim statement matches."""
    needle = pattern.lower()
    for snapshot in run.rounds:
        claims = {claim.id: claim for claim in snapshot.claims}
        for verdict in snapshot.verdicts:
            if verdict.verdict is not ClaimVerdict.REFUTED:
                continue
            claim = claims.get(verdict.claim_id)
            if claim is not None and needle in claim.statement.lower():
                return claim, verdict
    return None


def _falsehood_disproved(run: RunRecord) -> tuple[bool, str]:
    """The planted figure was disproved by EITHER shape a full pipeline
    produces (PRD §18 amendment #2):

    A. the skeptic refuted a claim asserting it (Phase 6 input-claim shape), or
    B. the researcher asserted a claim that names the figure and negates it,
       which the skeptic then supported against the document.
    """
    if _refuted_claim_match(run, "40") is not None:
        return True, "refuted verdict on a 40% claim"
    for snapshot in run.rounds:
        for claim in snapshot.claims:
            low = claim.statement.lower()
            if "40" in low and any(cue in low for cue in _DISPROOF_CUES):
                return True, f"disproof claim {claim.id}: {claim.statement[:80]}"
    return False, "no refuted verdict and no negated 40% claim"


def _cites_growth_report(run: RunRecord) -> tuple[bool, str]:
    """Any claim or verdict evidence naming the seeded corpus file."""
    for snapshot in run.rounds:
        for claim in snapshot.claims:
            for evidence in claim.evidence:
                if "growth_report" in evidence.source:
                    return True, evidence.source
        for verdict in snapshot.verdicts:
            for evidence in verdict.evidence:
                if "growth_report" in evidence.source:
                    return True, evidence.source
    return False, "no claim/verdict evidence cites growth_report"


def _selected(run: RunRecord):
    return select_best_round(run.rounds) if run.rounds else None


def _final_text(run: RunRecord) -> str:
    return run.final_message.content if run.final_message is not None else ""


# ----------------------------------------------------------- the predicates


def _check(name: str, passed: bool, detail: str = "") -> Check:
    return Check(name=name, passed=passed, detail=detail)


def _verification_report(run: RunRecord) -> VerificationReport | None:
    for step in reversed(run.steps):
        if step.kind is StepKind.VERIFY and step.message is not None:
            return step.message.verification
    return None


def _accountability_report(run: RunRecord) -> AccountabilityReport | None:
    for step in reversed(run.steps):
        if step.kind is StepKind.AUDIT and step.message is not None:
            return step.message.accountability
    return None


def _mandatory_run_basics(run: RunRecord) -> list[Check]:
    """Shared by all three problems (Phase 11 §6.12): the audits must exist,
    cover every final claim, and never surface a violation."""
    checks = [
        _check("run_completed", run.status is RunStatus.COMPLETED,
               run.status.value),
        _check("final_message_nonempty", bool(_final_text(run).strip())),
    ]

    verify_report = _verification_report(run)
    if verify_report is None:
        checks.append(
            _check("verification_report_present", False, "verify step missing")
        )
    else:
        selected = _selected(run)
        expected = (
            {claim.id for claim in selected.claims} if selected is not None
            else set()
        )
        actual = {entry.claim_id for entry in verify_report.claims}
        checks.append(_check(
            "verification_report_present",
            actual == expected,
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}",
        ))

    audit_report = _accountability_report(run)
    if audit_report is None:
        checks.append(
            _check("accountability_report_present", False, "audit step missing")
        )
        checks.append(
            _check("accountability_no_violations", False, "audit step missing")
        )
    else:
        checks.append(_check("accountability_report_present", True))
        checks.append(_check(
            "accountability_no_violations",
            audit_report.overall_status is not AccountabilityStatus.VIOLATIONS,
            audit_report.overall_status.value,
        ))
    return checks


def _audit_advisory(run: RunRecord) -> list[Check]:
    """Reported, never gated (Phase 11 §6.12)."""
    checks: list[Check] = []
    verify_report = _verification_report(run)
    if verify_report is None:
        checks.append(
            _check("verification_fully_verified", False, "verify step missing")
        )
    else:
        not_verified = [
            entry.claim_id for entry in verify_report.claims
            if entry.verification_status is not VerificationStatus.VERIFIED
        ]
        checks.append(_check(
            "verification_fully_verified",
            not not_verified,
            f"not verified: {not_verified}",
        ))
    audit_report = _accountability_report(run)
    checks.append(_check(
        "accountability_clean",
        audit_report is not None
        and audit_report.overall_status is AccountabilityStatus.CLEAN,
        audit_report.overall_status.value if audit_report is not None
        else "audit step missing",
    ))
    return checks


def _verifier_confirms_correction(run: RunRecord) -> Check:
    """P1 advisory: the claim carrying the corrected figure is verified."""
    selected = _selected(run)
    report = _verification_report(run)
    if selected is None or report is None:
        return _check("verifier_confirms_correction", False, "no audit data")
    target = next(
        (claim for claim in selected.claims if "23" in claim.statement), None
    )
    if target is None:
        return _check(
            "verifier_confirms_correction", False,
            "no final claim mentions the corrected figure",
        )
    entry = next(
        (e for e in report.claims if e.claim_id == target.id), None
    )
    ok = entry is not None and entry.verification_status in (
        VerificationStatus.VERIFIED,
        VerificationStatus.PARTIALLY_VERIFIED,
    )
    return _check(
        "verifier_confirms_correction", ok,
        entry.verification_status.value if entry is not None else "missing",
    )


def _p1_checks(spec: ProblemSpec, run: RunRecord) -> ProblemScore:
    mandatory = _mandatory_run_basics(run)
    mandatory.append(
        _check(
            "no_failed_steps",
            not any(step.status is StepStatus.FAILED for step in run.steps),
        )
    )
    tools = _tool_names(run)
    mandatory.append(
        _check(
            "file_tool_used",
            any(name in {"file_search", "file_reader"} for name in tools),
            f"tools={tools}",
        )
    )
    disproved, disproved_detail = _falsehood_disproved(run)
    mandatory.append(
        _check("falsehood_disproved", disproved, disproved_detail)
    )
    cited, cited_detail = _cites_growth_report(run)
    mandatory.append(
        _check("correction_cites_growth_report", cited, cited_detail)
    )

    advisory = [
        _check("final_mentions_23", "23%" in _final_text(run)),
        _check(
            "rounds_within_cap",
            len(run.rounds) <= spec.max_rounds,
            f"rounds={len(run.rounds)}",
        ),
        _check(
            "no_fact_lacking_evidence",
            not any(
                claim.status is ClaimStatus.FACT and not claim.evidence
                for snapshot in run.rounds
                for claim in snapshot.claims
            ),
        ),
        _verifier_confirms_correction(run),
        *_audit_advisory(run),
    ]
    return ProblemScore("p1", tuple(mandatory), tuple(advisory))


def _p2_checks(spec: ProblemSpec, run: RunRecord) -> ProblemScore:
    mandatory = _mandatory_run_basics(run)
    counts = verdict_counts(run)
    mandatory.append(
        _check(
            "skeptic_ran",
            counts["total"] > 0,
            f"verdicts={counts['total']}",
        )
    )
    selected = _selected(run)
    unresolved = selected.unresolved_count if selected is not None else 0
    final = _final_text(run).lower()
    if unresolved == 0:
        mandatory.append(
            _check("honesty_cue_when_unresolved", True, "nothing unresolved")
        )
    else:
        hit = sorted(cue for cue in HONESTY_CUES if cue in final)
        mandatory.append(
            _check(
                "honesty_cue_when_unresolved",
                bool(hit),
                f"unresolved={unresolved}; cues={hit}",
            )
        )

    advisory = [
        _check("iterated", len(run.rounds) >= 2, f"rounds={len(run.rounds)}"),
        _check(
            "assumption_or_hypothesis_claim",
            any(
                claim.status in {ClaimStatus.ASSUMPTION, ClaimStatus.HYPOTHESIS}
                for snapshot in run.rounds
                for claim in snapshot.claims
            ),
        ),
        _check("no_bare_salary_number", *_bare_salary_sentence(run)),
        *_audit_advisory(run),
    ]
    return ProblemScore("p2", tuple(mandatory), tuple(advisory))


def _bare_salary_sentence(run: RunRecord) -> tuple[bool, str]:
    """Advisory: a sentence asserting a salary figure with no honesty cue."""
    for sentence in re.split(r"[.\n]", _final_text(run)):
        low = sentence.lower()
        if "salar" in low and re.search(r"\b\d{4}\b", low):
            if not any(cue in low for cue in HONESTY_CUES):
                return False, sentence.strip()[:120]
    return True, ""


def _p3_checks(spec: ProblemSpec, run: RunRecord) -> ProblemScore:
    expected = [
        StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE,
        StepKind.CRITIQUE, StepKind.SYNTHESIZE, StepKind.VERIFY,
        StepKind.AUDIT,
    ]
    kinds = [step.kind for step in run.steps]
    mandatory = _mandatory_run_basics(run)
    mandatory.append(
        _check(
            "no_failed_steps",
            not any(step.status is StepStatus.FAILED for step in run.steps),
        )
    )
    mandatory.append(
        _check("duration_recorded", run.duration_ms is not None)
    )
    # Cost ceiling: the guards must keep a "simple" task off the full budget.
    # (rounds == 1 is model judgment — the manager may legitimately spend one
    # revision before finishing — so it is advisory; PRD §18 amendment #1.)
    mandatory.append(
        _check(
            "rounds_within_budget",
            len(run.rounds) <= 2,
            f"rounds={len(run.rounds)} (simple-task ceiling 2, "
            f"guard cap {spec.max_rounds})",
        )
    )

    covered = all(
        {claim.id for claim in snapshot.claims}
        <= {verdict.claim_id for verdict in snapshot.verdicts}
        for snapshot in run.rounds
    )
    advisory = [
        _check("exact_trace", kinds == expected,
               "kinds=" + ",".join(kind.value for kind in kinds)),
        _check("single_round", len(run.rounds) == 1,
               f"rounds={len(run.rounds)}"),
        _check("verdict_coverage", covered),
        _check(
            "wall_under_15min",
            (run.duration_ms or 0) < 900_000,
            f"wall_ms={run.duration_ms}",
        ),
        *_audit_advisory(run),
    ]
    return ProblemScore("p3", tuple(mandatory), tuple(advisory))


_SCORERS = {"p1": _p1_checks, "p2": _p2_checks, "p3": _p3_checks}


def score_problem(spec: ProblemSpec, run: RunRecord) -> ProblemScore:
    """Evaluate a run against its problem's expectations. Never raises on
    partial data — missing pieces simply fail their check with a detail."""
    return _SCORERS[spec.id](spec, run)


# ---------------------------------------------------------- report models


@dataclass(frozen=True, slots=True)
class HumanScores:
    """Operator rubric, 1–5; null until filled (Phase 10 PRD §6.6)."""

    correctness: int | None = None
    evidence_use: int | None = None
    honesty: int | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "correctness": self.correctness,
            "evidence_use": self.evidence_use,
            "honesty": self.honesty,
            "notes": self.notes,
        }


@dataclass(slots=True)
class ProblemResult:
    problem_id: str
    title: str
    task: str
    run_id: str
    status: str
    attempts: int
    rounds: int
    steps: int
    verdicts: dict[str, int]
    tokens: UsageTotals
    wall_ms: float | None
    mandatory: tuple[Check, ...]
    advisory: tuple[Check, ...]
    passed: bool
    human_scores: HumanScores = field(default_factory=HumanScores)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.problem_id,
            "title": self.title,
            "task": self.task,
            "run_id": self.run_id,
            "status": self.status,
            "attempts": self.attempts,
            "rounds": self.rounds,
            "steps": self.steps,
            "verdicts": self.verdicts,
            "tokens": {
                "prompt": self.tokens.prompt_tokens,
                "completion": self.tokens.completion_tokens,
                "total": self.tokens.total_tokens,
                "calls": self.tokens.calls,
            },
            "wall_ms": self.wall_ms,
            "mandatory": [asdict(check) for check in self.mandatory],
            "advisory": [asdict(check) for check in self.advisory],
            "passed": self.passed,
            "human_scores": self.human_scores.as_dict(),
        }


@dataclass(slots=True)
class EvalReport:
    generated_at: str
    app_version: str
    model: str
    num_ctx: int
    params: dict[str, object]
    problems: list[ProblemResult]
    summary: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "app_version": self.app_version,
            "model": self.model,
            "num_ctx": self.num_ctx,
            "params": self.params,
            "problems": [problem.as_dict() for problem in self.problems],
            "summary": self.summary,
        }


def summarize(results: Sequence[ProblemResult]) -> dict[str, object]:
    """Aggregate metrics shared by the console table and the JSON report."""
    verdicts = {key: 0 for key in ("supported", "refuted", "unverifiable")}
    for result in results:
        for key in verdicts:
            verdicts[key] += result.verdicts.get(key, 0)
    total_verdicts = sum(verdicts.values())
    refutation_rate_value = (
        verdicts["refuted"] / total_verdicts if total_verdicts else 0.0
    )
    tokens_total = sum(result.tokens.total_tokens for result in results)
    wall_total = sum(result.wall_ms or 0.0 for result in results)
    mandatory_total = sum(len(result.mandatory) for result in results)
    mandatory_passed = sum(
        1 for result in results for check in result.mandatory if check.passed
    )
    advisory_total = sum(len(result.advisory) for result in results)
    advisory_passed = sum(
        1 for result in results for check in result.advisory if check.passed
    )
    return {
        "mandatory_passed": mandatory_passed,
        "mandatory_total": mandatory_total,
        "advisory_passed": advisory_passed,
        "advisory_total": advisory_total,
        "refutation_rate": round(refutation_rate_value, 4),
        "refuted": verdicts["refuted"],
        "supported": verdicts["supported"],
        "unverifiable": verdicts["unverifiable"],
        "rounds_per_problem": {
            result.problem_id: result.rounds for result in results
        },
        "tokens_total": tokens_total,
        "wall_ms_total": round(wall_total, 1),
        "all_passed": all(result.passed for result in results),
    }


# --------------------------------------------------------- usage capture


@dataclass(slots=True)
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, usage: TokenUsage | None) -> None:
        if usage is None:
            return
        self.prompt_tokens += usage.prompt_tokens or 0
        self.completion_tokens += usage.completion_tokens or 0
        self.total_tokens += usage.total_tokens or 0
        self.calls += 1


class RecordingProvider(LLMProvider):
    """Delegates every call to `inner`, accumulating `CompletionResult.usage`.

    Eval needs per-run token totals without persisting usage (Phase 10 PRD
    §6.5.2 / deviation #1) — this wrapper keeps `app/llm/` and the schema
    untouched.
    """

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self._totals = UsageTotals()

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def totals(self) -> UsageTotals:
        return self._totals

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        keep_alive: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | str | None = None,
    ) -> CompletionResult:
        result = await self._inner.chat(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, num_ctx=num_ctx, keep_alive=keep_alive,
            tools=tools, response_format=response_format,
        )
        self._totals.add(result.usage)
        return result

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        keep_alive: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        return self._inner.stream(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, num_ctx=num_ctx, keep_alive=keep_alive,
            tools=tools, response_format=response_format,
        )

    async def list_models(self) -> list[ModelInfo]:
        return await self._inner.list_models()

    async def health(self) -> ProviderHealth:
        return await self._inner.health()

    async def aclose(self) -> None:
        await self._inner.aclose()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def build_report(
    results: Sequence[ProblemResult],
    *,
    app_version: str,
    model: str,
    num_ctx: int,
    params: dict[str, object],
) -> EvalReport:
    return EvalReport(
        generated_at=_utc_now_iso(),
        app_version=app_version,
        model=model,
        num_ctx=num_ctx,
        params=dict(params),
        problems=list(results),
        summary=summarize(results),
    )
