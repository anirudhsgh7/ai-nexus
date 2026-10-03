"""Mechanical audit layer (Phase 11 PRD §6.5): facts, validators, enforcement.

Two independent mechanisms keep the audit agents honest:

* `validate_verification` — the Verifier's status rules, applied inside
  `Agent.run`'s structured retry-once loop (a `verified` status without the
  agent's own evidence is corrected, not trusted).
* `compute_audit_facts` + `enforce_accountability` — process facts computed
  in code and merged INTO the Accountability agent's report by the
  orchestrator before anything is persisted, so a model claiming `clean` over
  a computed warning/violation cannot suppress it.

Pure: imports `app.runs` data + `app.schemas` only — no agents, no
orchestrator (orchestrator imports *this*). Never raises on partial data
inside the validators; facts assume orchestrator-produced snapshots (loud
failure on corruption is intentional).

`HONESTY_CUES` lives here as the single definition shared with `app.eval`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.runs import RoundSnapshot, StepKind, StepRecord, StepStatus
from app.schemas import (
    AccountabilityFlag,
    AccountabilityFlagKind,
    AccountabilityReport,
    AccountabilityStatus,
    AgentRole,
    Claim,
    ClaimProvenance,
    ClaimVerdict,
    ClaimVerification,
    DecisionAction,
    FlagSeverity,
    VerificationReport,
    VerificationStatus,
)

__all__ = [
    "HONESTY_CUES",
    "AuditFacts",
    "FactFlag",
    "compute_audit_facts",
    "enforce_accountability",
    "validate_verification",
]

#: Case-insensitive cues that count as "the answer admits it does not know".
#: Single source of truth (Phase 10 eval imports it); extended from live
#: evidence during Phase 10 (PRD §18 #7 there).
HONESTY_CUES: frozenset[str] = frozenset(
    {"uncertain", "unresolved", "unverifiable", "assumption", "estimate",
     "no data", "cannot", "speculative", "lack specific data"}
)

#: Steps that must exist in a healthy trace when the AUDIT step's input is
#: built. The AUDIT step itself cannot be required — it has not run yet.
_REQUIRED_KINDS = (
    StepKind.PLAN,
    StepKind.RESEARCH,
    StepKind.IDEATE,
    StepKind.CRITIQUE,
    StepKind.SYNTHESIZE,
    StepKind.VERIFY,
)

_MIN_CONFIDENCE = 0.7


# ------------------------------------------------------------- fact model


@dataclass(frozen=True, slots=True)
class FactFlag:
    """A code-computed finding the report is required to carry."""

    kind: AccountabilityFlagKind
    severity: FlagSeverity
    refs: tuple[str, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class AuditFacts:
    trace_complete: bool
    trace_missing: tuple[str, ...]
    provenance: tuple[ClaimProvenance, ...]
    required_flags: tuple[FactFlag, ...]


def _select_snapshot(rounds: Sequence[RoundSnapshot]) -> RoundSnapshot | None:
    """Selected-round pick — mirrors `orchestrator.select_best_round`
    (Phase 5 §6.6: net evidence score, later round on ties).

    Duplicated rather than imported to avoid the audits↔orchestrator cycle;
    `test_audits.py` asserts equality with the canonical helper over a matrix.
    """
    if not rounds:
        return None
    return max(
        rounds, key=lambda r: (r.supported_count - r.unresolved_count, r.round_number)
    )


def _is_unsupported(claim_id: str, verdict_by_id: dict[str, ClaimVerdict]) -> bool:
    verdict = verdict_by_id.get(claim_id)
    return verdict is None or verdict is not ClaimVerdict.SUPPORTED


def compute_audit_facts(
    rounds: Sequence[RoundSnapshot],
    steps: Sequence[StepRecord],
    final_message_content: str | None,
) -> AuditFacts:
    """Deterministic process/provenance facts for the accountability context.

    Never raises on partial traces: missing pieces simply produce facts (the
    `trace_incompleteness` fact flags them).
    """
    selected = _select_snapshot(rounds)
    flags: list[FactFlag] = []

    # ---- trace completeness ------------------------------------------------
    kinds_present = {step.kind for step in steps}
    missing = [kind.value for kind in _REQUIRED_KINDS if kind not in kinds_present]
    failed = [str(step.index) for step in steps if step.status is StepStatus.FAILED]
    trace_complete = not missing and not failed and final_message_content is not None
    if failed:
        flags.append(FactFlag(
            kind=AccountabilityFlagKind.TRACE_INCOMPLETENESS,
            severity=FlagSeverity.VIOLATION,
            refs=tuple(failed),
            detail="failed steps present in a completed trace",
        ))
    elif missing:
        flags.append(FactFlag(
            kind=AccountabilityFlagKind.TRACE_INCOMPLETENESS,
            severity=FlagSeverity.WARNING,
            refs=tuple(missing),
            detail="required step kinds missing",
        ))

    # ---- provenance + claim-level facts ------------------------------------
    provenance: list[ClaimProvenance] = []
    unsupported: list[str] = []
    evidence_gaps: list[str] = []
    confidence_mismatches: list[str] = []
    verdict_by_id: dict[str, ClaimVerdict] = {}
    if selected is not None:
        verdict_by_id = {
            claim_id: verdict.verdict
            for claim_id, verdict in
            ((v.claim_id, v) for v in selected.verdicts)
        }
        for claim in selected.claims:
            provenance.append(ClaimProvenance(
                claim_id=claim.id,
                origin=selected.origins[claim.id],
                verdict=verdict_by_id.get(claim.id),
                evidence_count=len(claim.evidence),
            ))
            if _is_unsupported(claim.id, verdict_by_id):
                unsupported.append(claim.id)
            if claim.status.value == "fact" and not claim.evidence:
                evidence_gaps.append(claim.id)
            if (
                not claim.evidence
                and verdict_by_id.get(claim.id) is ClaimVerdict.SUPPORTED
            ):
                evidence_gaps.append(claim.id)
            if (
                claim.confidence is not None and claim.confidence >= _MIN_CONFIDENCE
            ) and not claim.evidence:
                confidence_mismatches.append(claim.id)
            elif (
                claim.confidence is not None and claim.confidence >= _MIN_CONFIDENCE
            ) and _is_unsupported(claim.id, verdict_by_id):
                confidence_mismatches.append(claim.id)

    if unsupported:
        flags.append(FactFlag(
            kind=AccountabilityFlagKind.UNSUPPORTED_FINAL_CLAIM,
            severity=FlagSeverity.WARNING,
            refs=tuple(unsupported),
            detail="final claim has no supported verdict",
        ))

    # ---- the answer must not hide the unresolved remainder ------------------
    text = (final_message_content or "").lower()
    claims_are_named = any(
        re.search(rf"\b{re.escape(cid)}\b", text) for cid in unsupported
    )
    honesty_present = any(cue in text for cue in HONESTY_CUES)
    if unsupported and not claims_are_named and not honesty_present:
        flags.append(FactFlag(
            kind=AccountabilityFlagKind.UNRESOLVED_CLAIM_SUPPRESSED,
            severity=FlagSeverity.WARNING,
            refs=tuple(unsupported),
            detail="final answer neither names the unresolved claims nor admits uncertainty",
        ))

    # ---- decision consistency ------------------------------------------------
    decision_issues = _decision_issues(steps)
    if decision_issues:
        flags.append(FactFlag(
            kind=AccountabilityFlagKind.DECISION_INCONSISTENCY,
            severity=FlagSeverity.VIOLATION,
            refs=tuple(decision_issues),
            detail="decision chain does not match the guarded iteration contract",
        ))

    # ---- evidence provenance ------------------------------------------------
    if evidence_gaps:
        flags.append(FactFlag(
            kind=AccountabilityFlagKind.EVIDENCE_PROVENANCE_GAP,
            severity=FlagSeverity.WARNING,
            refs=tuple(dict.fromkeys(evidence_gaps)),
            detail="fact or supported claim carries no evidence",
        ))

    # ---- tool-use consistency ------------------------------------------------
    tool_issues = [
        str(step.index)
        for step in steps
        if step.message is not None
        and step.message.tool_calls is not None
        and (
            step.message.tool_results is None
            or len(step.message.tool_results) != len(step.message.tool_calls)
        )
    ]
    if tool_issues:
        flags.append(FactFlag(
            kind=AccountabilityFlagKind.TOOL_USE_INCONSISTENCY,
            severity=FlagSeverity.VIOLATION,
            refs=tuple(tool_issues),
            detail="tool_calls without parallel tool_results",
        ))

    # ---- premature stop (guard fired while claims remain unresolved) ----------
    guarded = [str(step.index) for step in steps if step.skipped and step.kind is StepKind.DECIDE]
    if guarded and unsupported:
        round_refs = tuple(
            sorted({str(step.round) for step in steps
                    if step.skipped and step.kind is StepKind.DECIDE and step.round})
        )
        flags.append(FactFlag(
            kind=AccountabilityFlagKind.PREMATURE_STOP,
            severity=FlagSeverity.WARNING,
            refs=round_refs or tuple(guarded),
            detail="guard stopped the run while unresolved claims remained",
        ))

    # ---- confidence vs evidence/support -------------------------------------
    if confidence_mismatches:
        flags.append(FactFlag(
            kind=AccountabilityFlagKind.CONFIDENCE_EVIDENCE_MISMATCH,
            severity=FlagSeverity.WARNING,
            refs=tuple(dict.fromkeys(confidence_mismatches)),
            detail="high-confidence claim lacks evidence or support",
        ))

    # ---- retry activity --------------------------------------------------------
    retry_steps = [
        str(step.index)
        for step in steps
        if step.message is not None and step.message.retries > 0
    ]
    if retry_steps:
        flags.append(FactFlag(
            kind=AccountabilityFlagKind.RETRY_ACTIVITY,
            severity=FlagSeverity.INFO,
            refs=tuple(retry_steps),
            detail="structured-output correction retry occurred",
        ))

    return AuditFacts(
        trace_complete=trace_complete,
        trace_missing=tuple(missing),
        provenance=tuple(provenance),
        required_flags=tuple(flags),
    )


def _decision_issues(steps: Sequence[StepRecord]) -> list[str]:
    """Flag indexes of decisions that break the guarded iteration contract.

    * a non-skipped DECIDE without a parseable decision
    * CALL_AGENT not followed by its target's REVISE step and not terminated
      by a guard (guards legitimately stop before revision)
    * any DECIDE after a non-skipped FINISH
    * a guard-skipped DECIDE that lacks a synthetic FINISH decision
    """
    issues: list[str] = []
    ordered = sorted(steps, key=lambda s: s.index)
    for step in ordered:
        if step.kind is not StepKind.DECIDE:
            continue
        later_decides = [
            s for s in ordered if s.kind is StepKind.DECIDE and s.index > step.index
        ]
        if step.skipped:
            message = step.message
            decision = message.decision if message is not None else None
            if decision is None or decision.action is not DecisionAction.FINISH:
                issues.append(str(step.index))
            continue
        message = step.message
        if message is None or message.decision is None:
            issues.append(str(step.index))
            continue
        decision = message.decision
        if decision.action is DecisionAction.CALL_AGENT:
            honored = any(
                later.kind is StepKind.REVISE
                and later.agent is decision.target
                and step.round is not None
                and later.round == step.round + 1
                for later in ordered
                if later.index > step.index
            )
            guarded = any(
                later.skipped and later.kind is StepKind.DECIDE
                for later in ordered
                if later.index > step.index
            )
            if not honored and not guarded:
                issues.append(str(step.index))
        elif decision.action is DecisionAction.FINISH and later_decides:
            issues.append(str(step.index))
    return issues


# ------------------------------------------------- verification validation


def validate_verification(
    report: VerificationReport | None,
    final_claims: Sequence[Claim],
    tool_calls_executed: int,
    tools_available: bool,
) -> list[str]:
    """Status rules for the Verifier (PRD §6.5.2). Empty list = clean."""
    if report is None:
        return ["verification report missing"]
    problems: list[str] = []
    expected = {claim.id for claim in final_claims}
    seen: set[str] = set()
    entries: list[ClaimVerification] = list(report.claims)

    for entry in entries:
        if entry.claim_id in seen:
            problems.append(f"claim {entry.claim_id} verified more than once")
        seen.add(entry.claim_id)
        if entry.claim_id not in expected:
            problems.append(f"verification for unknown claim {entry.claim_id}")

    for claim_id in sorted(expected - seen):
        problems.append(f"claim {claim_id} was not verified")

    any_independent_status = False
    for entry in entries:
        status = entry.verification_status
        if status in (
            VerificationStatus.VERIFIED,
            VerificationStatus.PARTIALLY_VERIFIED,
        ):
            any_independent_status = True
        if status is VerificationStatus.VERIFIED:
            if not entry.supporting_evidence:
                problems.append(
                    f"claim {entry.claim_id} verified without supporting evidence"
                )
            if entry.contradicting_evidence:
                problems.append(
                    f"claim {entry.claim_id} verified despite contradicting evidence"
                )
            if not entry.source_references:
                problems.append(
                    f"claim {entry.claim_id} verified without source references"
                )
            if not entry.evidence_checked:
                problems.append(
                    f"claim {entry.claim_id} verified without recording checked evidence"
                )
        elif status is VerificationStatus.PARTIALLY_VERIFIED:
            if not entry.evidence_checked:
                problems.append(
                    f"claim {entry.claim_id} verified without recording checked evidence"
                )
            if not entry.supporting_evidence and not entry.contradicting_evidence:
                problems.append(
                    f"claim {entry.claim_id} partially verified without evidence"
                )
        elif status is VerificationStatus.CONTRADICTED:
            if not entry.contradicting_evidence:
                problems.append(
                    f"claim {entry.claim_id} contradicted without contradicting evidence"
                )
        # UNVERIFIABLE: nothing checkable — unconstrained by design

    if tools_available and tool_calls_executed == 0 and any_independent_status:
        problems.append(
            "verified claims require at least one independent tool check"
        )
    return problems


# --------------------------------------------- accountability enforcement


def _derive_status(flags: Sequence[AccountabilityFlag]) -> AccountabilityStatus:
    if any(flag.severity is FlagSeverity.VIOLATION for flag in flags):
        return AccountabilityStatus.VIOLATIONS
    if any(flag.severity is FlagSeverity.WARNING for flag in flags):
        return AccountabilityStatus.WARNINGS
    return AccountabilityStatus.CLEAN


def enforce_accountability(
    report: AccountabilityReport, facts: AuditFacts
) -> AccountabilityReport:
    """Merge code-canonical facts into the model's report (PRD §6.5.4).

    * provenance/trace come from `facts` (the model cannot alter them);
    * every required fact flag is present with the fact's severity
      (a matching model flag keeps its explanation but takes the fact's
      severity, so it can never be understated);
    * `overall_status` is derived from the merged flags.
    """
    flags: list[AccountabilityFlag] = list(report.flags)
    for fact in facts.required_flags:
        match_index = next(
            (
                index
                for index, flag in enumerate(flags)
                if flag.kind is fact.kind and set(flag.refs) == set(fact.refs)
            ),
            None,
        )
        if match_index is None:
            flags.append(AccountabilityFlag(
                kind=fact.kind,
                severity=fact.severity,
                refs=list(fact.refs),
                explanation=fact.detail,
            ))
        else:
            matched = flags[match_index]
            if matched.severity is not fact.severity:
                flags[match_index] = matched.model_copy(
                    update={"severity": fact.severity}
                )
    return report.model_copy(update={
        "trace_completeness": facts.trace_complete,
        "final_claim_provenance": list(facts.provenance),
        "flags": flags,
        "overall_status": _derive_status(flags),
    })
