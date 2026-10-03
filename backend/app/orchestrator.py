"""Iterative orchestration: Manager routing, revision rounds, deterministic stops.

Phase 5 replaces Phase 4's fixed pipeline with a bounded loop. Three locks:

- The Manager is a *router*: it emits grammar-constrained `ManagerDecision`
  JSON executed by Python, never prose, and it only ever sees the bounded
  registry summary — never worker prose.
- Stop conditions are deterministic: zero unresolved finishes with no decision
  call at all; MAX_ROUNDS, repeated-decision, no-progress, and step caps force
  finish visibly (synthetic skipped DECIDE step + `guard_triggered` log).
- Synthesis uses the **best round** (net evidence score), not the last.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.agents import AgentRegistry
from app.agents.structured import OutputKind, normalize_decision
from app.audits import AuditFacts, compute_audit_facts, enforce_accountability
from app.runs import (
    ErrorInfo,
    RoundSnapshot,
    RunEventType,
    RunManager,
    RunRecord,
    RunStatus,
    StepKind,
    StepRecord,
    StepStatus,
)
from app.schemas import (
    AgentMessage,
    AgentRole,
    Claim,
    ClaimVerdict,
    DecisionAction,
    ManagerDecision,
    MessageType,
    VerificationStatus,
    Verdict,
)

logger = logging.getLogger("ai_nexus.orchestrator")

__all__ = [
    "MAX_TOTAL_STEPS",
    "ClaimOrigin",
    "Orchestrator",
    "PoolState",
    "QualifiedClaim",
    "ROUND_ONE_STEPS",
    "qualify_claims",
    "render_decision_summary",
    "render_evidence_board",
    "render_iteration_history",
    "render_accountability_context",
    "render_verification_context",
    "render_revision_context",
    "render_synthesis_context",
    "resolve_claim",
    "select_best_round",
]

ROUND_ONE_STEPS: tuple[tuple[StepKind, AgentRole], ...] = (
    (StepKind.RESEARCH, AgentRole.RESEARCHER),
    (StepKind.IDEATE, AgentRole.IDEATOR),
)

MAX_TOTAL_STEPS = 20          # Phase 11: audit headroom above MAX_ROUNDS;
                              # the loop-top guard still bounds iteration only
MAX_SUMMARY_ENTRIES = 10      # decision-context entry cap (PRD §6.7)
CLIP_CHARS = 200              # per-field clipping in the decision context

_SKIPPED_CRITIQUE_NOTE = "(skipped: no claims were produced to evaluate)"
_WORKERS: tuple[AgentRole, ...] = (AgentRole.RESEARCHER, AgentRole.IDEATOR)


# ---------------------------------------------------------------- claim mapping


@dataclass(frozen=True, slots=True)
class ClaimOrigin:
    agent: AgentRole
    original_id: str


@dataclass(frozen=True, slots=True)
class QualifiedClaim:
    claim: Claim
    origin: ClaimOrigin


def qualify_claims(
    batches: Sequence[tuple[AgentRole, Sequence[Claim]]],
    *,
    start: int = 1,
) -> list[QualifiedClaim]:
    """Deterministic run-level re-identification c{start}..c{start+N-1}.

    Claim ordering and origins are preserved so verdicts referencing run-level
    ids map back to (agent, original id).
    """
    qualified: list[QualifiedClaim] = []
    counter = start - 1
    for agent, claims in batches:
        for claim in claims:
            counter += 1
            qualified.append(
                QualifiedClaim(
                    claim=Claim(
                        id=f"c{counter}",
                        statement=claim.statement,
                        status=claim.status,
                        confidence=claim.confidence,
                        evidence=list(claim.evidence),
                    ),
                    origin=ClaimOrigin(agent=agent, original_id=claim.id),
                )
            )
    return qualified


def resolve_claim(
    qualified: Sequence[QualifiedClaim], claim_id: str
) -> QualifiedClaim | None:
    for item in qualified:
        if item.claim.id == claim_id:
            return item
    return None


def _normalize_statement(statement: str) -> str:
    return " ".join(statement.split()).lower()


def _normalize_instruction(instruction: str) -> str:
    return " ".join(instruction.split()).lower()


# ---------------------------------------------------------------- pool state


class PoolState:
    """Active claim pool with verdicts and continuing run-level IDs (PRD §6.5)."""

    def __init__(self) -> None:
        self.claims: list[QualifiedClaim] = []
        self.verdicts: dict[str, Verdict] = {}
        self.counter = 0

    def _is_supported(self, item: QualifiedClaim) -> bool:
        verdict = self.verdicts.get(item.claim.id)
        return verdict is not None and verdict.verdict is ClaimVerdict.SUPPORTED

    def _append(
        self, agent: AgentRole, claims: Sequence[Claim] | None
    ) -> list[QualifiedClaim]:
        """Add claims with fresh IDs; dedupe by normalized statement (PRD §6.5-3)."""
        active_statements = {
            _normalize_statement(item.claim.statement) for item in self.claims
        }
        added: list[QualifiedClaim] = []
        for claim in claims or []:
            normalized = _normalize_statement(claim.statement)
            if normalized in active_statements:
                logger.debug(
                    "pool_dedupe agent=%s statement=%.80s", agent.value, claim.statement
                )
                continue
            active_statements.add(normalized)
            self.counter += 1
            added.append(
                QualifiedClaim(
                    claim=Claim(
                        id=f"c{self.counter}",
                        statement=claim.statement,
                        status=claim.status,
                        confidence=claim.confidence,
                        evidence=list(claim.evidence),
                    ),
                    origin=ClaimOrigin(agent=agent, original_id=claim.id),
                )
            )
        self.claims.extend(added)
        return added

    def add(
        self, agent: AgentRole, claims: Sequence[Claim] | None
    ) -> list[QualifiedClaim]:
        """Round-1 intake (also dedupes cross-agent duplicates)."""
        return self._append(agent, claims)

    def apply_revision(
        self, agent: AgentRole, revision_claims: Sequence[Claim] | None
    ) -> tuple[list[QualifiedClaim], list[QualifiedClaim]]:
        """Revision rule: keep the agent's SUPPORTED claims, drop its unresolved
        ones (and their verdicts), append deduplicated new claims (PRD §6.5)."""
        dropped = [
            item for item in self.claims
            if item.origin.agent is agent and not self._is_supported(item)
        ]
        if dropped:
            self.claims = [
                item for item in self.claims
                if not (item.origin.agent is agent and not self._is_supported(item))
            ]
            for item in dropped:
                self.verdicts.pop(item.claim.id, None)
        added = self._append(agent, revision_claims)
        return added, dropped

    def by_agent(self, agent: AgentRole) -> list[QualifiedClaim]:
        return [item for item in self.claims if item.origin.agent is agent]

    def active_claims(self) -> list[Claim]:
        return [item.claim for item in self.claims]

    def record_verdicts(self, verdicts: Sequence[Verdict] | None) -> None:
        for verdict in verdicts or []:
            self.verdicts[verdict.claim_id] = verdict

    def verdicts_for(self, items: Sequence[QualifiedClaim]) -> list[Verdict]:
        return [
            self.verdicts[item.claim.id]
            for item in items
            if item.claim.id in self.verdicts
        ]

    def applicable_verdicts(self) -> list[Verdict]:
        return self.verdicts_for(self.claims)

    def unresolved(self) -> list[QualifiedClaim]:
        return [item for item in self.claims if not self._is_supported(item)]

    def supported_count(self) -> int:
        return sum(1 for item in self.claims if self._is_supported(item))


# ---------------------------------------------------------------- rendering


def _render_evidence_lines(claim: Claim) -> list[str]:
    if not claim.evidence:
        return ["Evidence: none"]
    lines = ["Evidence:"]
    for item in claim.evidence:
        lines.append(f"- source={item.source}")
        if item.quote is not None:
            lines.append(f'  quote="{item.quote}"')
    return lines


def render_evidence_board(
    qualified: Sequence[QualifiedClaim], verdicts: Sequence[Verdict]
) -> str:
    """Golden claim+verdict view shared by revision contexts and synthesis."""
    by_id = {v.claim_id: v for v in verdicts}
    blocks: list[str] = []
    for item in qualified:
        claim = item.claim
        confidence = claim.confidence if claim.confidence is not None else "none"
        lines = [
            f"[{claim.id}] ({item.origin.agent.value}) "
            f"status={claim.status.value} confidence={confidence}",
            f"Claim: {claim.statement}",
            *_render_evidence_lines(claim),
        ]
        verdict = by_id.get(claim.id)
        if verdict is None:
            lines.append("Verdict: none")
        else:
            lines.append(f"Verdict: {verdict.verdict.value} — {verdict.objection}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_synthesis_context(
    researcher_content: str,
    ideator_content: str,
    skeptic_content: str | None,
    board: str,
) -> str:
    """Canonical Manager synthesis context (Phase 4 §6.4, unchanged)."""
    skeptic_body = (
        skeptic_content if skeptic_content is not None else _SKIPPED_CRITIQUE_NOTE
    )
    board_body = board if board else "(none)"
    instruction = (
        "SYNTHESIS INSTRUCTION:\n"
        "Answer the user's task directly and substantively using the materials "
        "below. Write the answer itself; never describe what an answer should "
        "contain or what analysis would be needed. Do not restate the task. "
        "Use the findings, evidence, verdicts, and disagreements. Produce a "
        "thorough, well-organized answer that includes concrete details from "
        "the materials (named roles, skills, sources, and remaining "
        "disagreements), and end with a clear recommendation or conclusion. "
        "When evidence is thin or unverified, still give the best direct "
        "answer possible from the provided claims and clearly label which "
        "parts are unverified; do not make the missing evidence the whole "
        "answer."
    )
    return (
        f"{instruction}\n\n"
        f"RESEARCHER FINDINGS:\n{researcher_content}\n\n"
        f"IDEATOR OPTIONS:\n{ideator_content}\n\n"
        f"SKEPTIC CRITIQUE:\n{skeptic_body}\n\n"
        f"EVALUATED CLAIMS:\n{board_body}"
    )


def _clip(text: str, limit: int = CLIP_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def render_decision_summary(
    round_number: int,
    max_rounds: int,
    active: Sequence[QualifiedClaim],
    verdicts: dict[str, Verdict],
    previous_decision: ManagerDecision | None = None,
) -> str:
    """Bounded registry summary — the ONLY thing the Manager sees to route.

    Contains claim records and verdicts; never worker prose (PRD §6.7).
    """
    blocks = [f"ROUND {round_number} OF {max_rounds}"]

    if previous_decision is not None:
        target = (
            previous_decision.target.value
            if previous_decision.target
            else "agent"
        )
        blocks.append(
            f"PREVIOUS INSTRUCTION:\n{target}: "
            f"{_clip(previous_decision.instruction)}"
        )

    status_lines = ["AGENT STATUS:"]
    for agent in _WORKERS:
        owned = [item for item in active if item.origin.agent is agent]
        supported = sum(
            1 for item in owned
            if item.claim.id in verdicts
            and verdicts[item.claim.id].verdict is ClaimVerdict.SUPPORTED
        )
        status_lines.append(
            f"- {agent.value}: {len(owned)} claims "
            f"({supported} supported, {len(owned) - supported} unresolved)"
        )
    blocks.append("\n".join(status_lines))

    unresolved = [
        item for item in active
        if item.claim.id not in verdicts
        or verdicts[item.claim.id].verdict is not ClaimVerdict.SUPPORTED
    ]
    if not unresolved:
        blocks.append("UNRESOLVED CLAIMS:\n(none)")
    else:
        entry_lines = ["UNRESOLVED CLAIMS:"]
        shown = unresolved[:MAX_SUMMARY_ENTRIES]
        for item in shown:
            verdict = verdicts.get(item.claim.id)
            verdict_value = verdict.verdict.value if verdict else "none"
            objection = _clip(verdict.objection) if verdict else "none"
            entry_lines.append(
                f"[{item.claim.id}] ({item.origin.agent.value}) "
                f"verdict={verdict_value}"
            )
            entry_lines.append(f"  Claim: {_clip(item.claim.statement)}")
            entry_lines.append(f"  Objection: {objection}")
        if len(unresolved) > MAX_SUMMARY_ENTRIES:
            entry_lines.append(
                f"... and {len(unresolved) - MAX_SUMMARY_ENTRIES} more unresolved claims"
            )
        blocks.append("\n".join(entry_lines))

    return "\n\n".join(blocks)


def render_revision_context(instruction: str, board: str) -> str:
    """Target worker's context: manager instruction + own claims only (PRD §6.7)."""
    return (
        f"MANAGER INSTRUCTION:\n{instruction}\n\n"
        f"YOUR CURRENT CLAIMS AND THEIR VERDICTS:\n{board or '(none)'}"
    )


def render_iteration_history(
    snapshots: Sequence[RoundSnapshot], selected: RoundSnapshot
) -> str:
    """Minority-report section appended to synthesis: rounds + remaining unresolved."""
    lines = ["ITERATION HISTORY:"]
    for snapshot in snapshots:
        lines.append(
            f"- round {snapshot.round_number}: {snapshot.supported_count} supported, "
            f"{snapshot.unresolved_count} unresolved"
        )
    score = selected.supported_count - selected.unresolved_count
    lines.append(
        f"Selected round: {selected.round_number} (net evidence score {score})"
    )

    applicable = {v.claim_id: v for v in selected.verdicts}
    remaining = [
        claim for claim in selected.claims
        if claim.id not in applicable
        or applicable[claim.id].verdict is not ClaimVerdict.SUPPORTED
    ]
    lines.append("")
    lines.append("REMAINING UNRESOLVED:")
    if not remaining:
        lines.append("(none)")
    else:
        for claim in remaining:
            verdict = applicable.get(claim.id)
            verdict_value = verdict.verdict.value if verdict else "none"
            origin = selected.origins.get(claim.id)
            origin_value = origin.value if origin else "?"
            lines.append(f"[{claim.id}] ({origin_value}) verdict={verdict_value}")
    return "\n".join(lines)


def select_best_round(snapshots: Sequence[RoundSnapshot]) -> RoundSnapshot:
    """Net evidence score first (supported - unresolved), later round on ties.

    Empty rounds score 0 and cannot beat a net-positive round (PRD §6.6).
    """
    return max(
        snapshots,
        key=lambda snap: (snap.supported_count - snap.unresolved_count,
                          snap.round_number),
    )


# ------------------------------------------------- Phase 11 audit renderers


_VERIFICATION_VERDICTS_HEADER = (
    "PRIOR VERDICTS (context only — never grounds for verification):"
)


def render_verification_context(
    final_content: str,
    claims: Sequence[Claim],
    verdicts: Sequence[Verdict],
) -> str:
    """Verifier input (Phase 11 PRD §6.6).

    The final answer under audit plus prior verdicts — strictly as context.
    Claims themselves arrive through the canonical `CLAIMS TO EVALUATE`
    block (`render_claims`), so evidence rendering is not duplicated.
    """
    header = f"FINAL ANSWER:\n{final_content or '(empty)'}"
    if not claims:
        return f"{header}\n\n{_VERIFICATION_VERDICTS_HEADER} (none)"
    verdict_by_id = {verdict.claim_id: verdict for verdict in verdicts}
    lines = []
    for claim in claims:
        verdict = verdict_by_id.get(claim.id)
        if verdict is None:
            lines.append(f"[{claim.id}] none")
        else:
            lines.append(
                f"[{claim.id}] {verdict.verdict.value} — {verdict.objection}"
            )
    return f"{header}\n\n{_VERIFICATION_VERDICTS_HEADER}\n" + "\n".join(lines)


_UNRESOLVED_CAP = 10


def render_accountability_context(
    run: RunRecord,
    facts: AuditFacts,
    verification: Any,
) -> str:
    """Accountability input (Phase 11 PRD §6.6): deterministic trace facts,
    code-computed provenance, the verification summary, the final answer, and
    the FACT ENTRIES the report must echo in its flags."""
    steps_line = " ".join(
        f"{step.kind.value}={step.status.value}" for step in run.steps
    )

    decision_parts: list[str] = []
    guard_parts: list[str] = []
    failed_steps: list[str] = []
    retry_steps: list[str] = []
    retries_total = 0
    for step in run.steps:
        if step.status is StepStatus.FAILED:
            failed_steps.append(str(step.index))
        if step.message is not None and step.message.retries > 0:
            retries_total += step.message.retries
            retry_steps.append(str(step.index))
        if step.kind is not StepKind.DECIDE or step.message is None:
            continue
        decision = step.message.decision
        if decision is None:
            continue
        rnd = f"r{step.round}" if step.round is not None else "r?"
        if step.skipped:
            decision_parts.append(f"{rnd} guard_finish({decision.reason})")
            guard_parts.append(f"round {step.round} {decision.reason}")
        else:
            target = f"→{decision.target.value}" if decision.target else ""
            decision_parts.append(
                f"{rnd} {decision.action.value}{target} "
                f"confidence={decision.confidence:.2f}"
            )

    selected = select_best_round(run.rounds) if run.rounds else None
    if selected is None:
        selected_line = "selected_round: none"
        final_claims_line = "final_claims: 0"
        provenance_lines = ["(none)"]
        unresolved_lines = ["(none)"]
    else:
        selected_line = (
            f"selected_round: {selected.round_number} "
            f"supported={selected.supported_count} "
            f"unresolved={selected.unresolved_count}"
        )
        final_claims_line = f"final_claims: {len(selected.claims)}"
        provenance_lines = [
            f"[{p.claim_id}] origin={p.origin.value} "
            f"verdict={p.verdict.value if p.verdict is not None else 'none'} "
            f"evidence={p.evidence_count}"
            for p in facts.provenance
        ] or ["(none)"]
        verdict_by_id = {v.claim_id: v for v in selected.verdicts}
        unresolved: list[str] = []
        for claim in selected.claims:
            verdict = verdict_by_id.get(claim.id)
            if verdict is not None and verdict.verdict is ClaimVerdict.SUPPORTED:
                continue
            origin = selected.origins.get(claim.id)
            label = verdict.verdict.value if verdict is not None else "none"
            objection = verdict.objection if verdict is not None else "no verdict recorded"
            unresolved.append(
                f"[{claim.id}] ({origin.value if origin else '?'}) "
                f"verdict={label} — {objection}"
            )
        if not unresolved:
            unresolved_lines = ["(none)"]
        elif len(unresolved) <= _UNRESOLVED_CAP:
            unresolved_lines = unresolved
        else:
            unresolved_lines = [
                *unresolved[:_UNRESOLVED_CAP],
                f"... and {len(unresolved) - _UNRESOLVED_CAP} more unresolved claims",
            ]

    counts = {status: 0 for status in VerificationStatus}
    if verification is not None:
        for entry in verification.claims:
            counts[entry.verification_status] += 1
    verification_line = (
        f"verified={counts[VerificationStatus.VERIFIED]} "
        f"partially_verified={counts[VerificationStatus.PARTIALLY_VERIFIED]} "
        f"contradicted={counts[VerificationStatus.CONTRADICTED]} "
        f"unverifiable={counts[VerificationStatus.UNVERIFIABLE]}"
    )

    final_answer = ""
    for step in run.steps:
        if step.kind is StepKind.SYNTHESIZE and step.message is not None:
            final_answer = step.message.content
            break

    fact_lines = [
        (
            f"- kind={fact.kind.value} severity={fact.severity.value} "
            f"refs=[{','.join(fact.refs)}] detail={fact.detail}"
        )
        for fact in facts.required_flags
    ] or ["(none)"]

    return "\n".join([
        "RUN TRACE FACTS:",
        f"steps: {steps_line or '(none)'}",
        f"decisions: {' | '.join(decision_parts) or 'none'}",
        f"guards: {'; '.join(guard_parts) or 'none'}",
        f"failed_steps: {','.join(failed_steps) or 'none'}",
        f"retries: total={retries_total} steps=[{','.join(retry_steps)}]",
        selected_line,
        final_claims_line,
        "",
        "FINAL CLAIM PROVENANCE:",
        *provenance_lines,
        "",
        "UNRESOLVED AT SYNTHESIS:",
        *unresolved_lines,
        "",
        "VERIFICATION SUMMARY:",
        verification_line,
        "",
        "FINAL ANSWER:",
        final_answer,
        "",
        "FACT ENTRIES (every entry MUST appear in your flags with the same kind/severity/refs):",
        *fact_lines,
    ])


# ---------------------------------------------------------------- orchestrator


def _error_info(exc: Exception) -> ErrorInfo:
    return ErrorInfo(
        type=type(exc).__name__,
        message=str(exc),
        hint=getattr(exc, "hint", "") or "",
    )


def _same_decision(a: ManagerDecision, b: ManagerDecision) -> bool:
    return (
        a.action is b.action
        and a.target is b.target
        and _normalize_instruction(a.instruction)
        == _normalize_instruction(b.instruction)
    )


class Orchestrator:
    """Executes the iterative pipeline and publishes run events (PRD §6.6)."""

    def __init__(
        self, registry: AgentRegistry, store: RunManager, *,
        max_rounds: int | None = None,
    ) -> None:
        if max_rounds is None:
            # Resolved from Settings here so the frozen lifespan wiring stays
            # untouched while AI_NEXUS_MAX_ROUNDS still governs API runs.
            from app.config import get_settings

            max_rounds = get_settings().max_rounds
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self._registry = registry
        self._store = store
        self._max_rounds = max_rounds

    async def execute(self, run_id: str) -> None:
        run = self._store.get(run_id)
        if run is None:
            return
        manager = self._registry.get(AgentRole.MANAGER)
        researcher = self._registry.get(AgentRole.RESEARCHER)
        ideator = self._registry.get(AgentRole.IDEATOR)
        skeptic = self._registry.get(AgentRole.SKEPTIC)

        pool = PoolState()
        worker_content: dict[AgentRole, str] = {}
        skeptic_content: str | None = None
        previous_decision: ManagerDecision | None = None

        try:
            run.started_at = datetime.now(UTC)
            run.status = RunStatus.RUNNING
            task_text = run.task
            self._store.append_event(
                run_id, RunEventType.RUN_STARTED, task=task_text
            )

            plan = await self._step(
                run, StepKind.PLAN, AgentRole.MANAGER,
                lambda: manager.run(task_text, run_id=run_id), round_=None,
            )

            # ---- round 1 (fixed participation, Phase 4 compatibility) ----
            research = await self._step(
                run, StepKind.RESEARCH, AgentRole.RESEARCHER,
                lambda: researcher.run(
                    task_text, context=plan.content, run_id=run_id
                ),
                round_=1,
            )
            pool.add(AgentRole.RESEARCHER, research.claims)
            worker_content[AgentRole.RESEARCHER] = research.content

            ideation = await self._step(
                run, StepKind.IDEATE, AgentRole.IDEATOR,
                lambda: ideator.run(
                    task_text, context=plan.content, run_id=run_id
                ),
                round_=1,
            )
            pool.add(AgentRole.IDEATOR, ideation.claims)
            worker_content[AgentRole.IDEATOR] = ideation.content

            if pool.claims:
                critique = await self._step(
                    run, StepKind.CRITIQUE, AgentRole.SKEPTIC,
                    lambda: skeptic.run(
                        task_text, claims=pool.active_claims(), run_id=run_id
                    ),
                    round_=1,
                )
                pool.record_verdicts(critique.verdicts)
                skeptic_content = critique.content
            else:
                self._mark_skipped(run, StepKind.CRITIQUE, AgentRole.SKEPTIC,
                                   round_=1)

            rounds = 1
            self._store.record_round(
                run.id,
                self._snapshot(rounds, pool, worker_content, skeptic_content),
            )

            # ---- iteration loop ----
            while True:
                if not pool.unresolved():
                    break  # deterministic finish; no decision call (PRD §6.6)
                if rounds >= self._max_rounds:
                    self._forced_finish(
                        run, f"round cap reached (max_rounds={self._max_rounds})",
                        rounds,
                    )
                    break
                if len(run.steps) >= MAX_TOTAL_STEPS:
                    self._forced_finish(run, "step cap reached", rounds)
                    break

                summary = render_decision_summary(
                    rounds, self._max_rounds, pool.claims, pool.verdicts,
                    previous_decision,
                )
                decision_msg = await self._step(
                    run, StepKind.DECIDE, AgentRole.MANAGER,
                    lambda: manager.run(
                        task_text,
                        context=summary,
                        message_type=MessageType.DECISION,
                        output_kind=OutputKind.DECISION,
                        run_id=run_id,
                    ),
                    round_=rounds,
                )
                if decision_msg.decision is None:  # defensive; parse guarantees
                    self._forced_finish(run, "decision missing", rounds)
                    break
                decision = normalize_decision(decision_msg.decision)
                self._store.record_round_decision(run.id, decision)

                if decision.action is DecisionAction.FINISH:
                    break
                if previous_decision is not None and _same_decision(
                    previous_decision, decision
                ):
                    self._forced_finish(run, "repeated decision", rounds)
                    break
                previous_decision = decision

                target = decision.target
                assert target in _WORKERS  # enforced by ManagerDecision validator
                own_board = render_evidence_board(
                    pool.by_agent(target), pool.verdicts_for(pool.by_agent(target))
                )
                revision_context = render_revision_context(
                    decision.instruction, own_board
                )
                next_round = rounds + 1
                revision = await self._step(
                    run, StepKind.REVISE, target,
                    lambda: self._registry.get(target).run(
                        task_text,
                        context=revision_context,
                        message_type=MessageType.REVISION,
                        run_id=run_id,
                    ),
                    round_=next_round,
                )
                added, dropped = pool.apply_revision(target, revision.claims)
                worker_content[target] = revision.content

                dropped_statements = {
                    _normalize_statement(item.claim.statement) for item in dropped
                }
                new_statements = {
                    _normalize_statement(item.claim.statement) for item in added
                }
                made_progress = bool(
                    new_statements - dropped_statements
                ) or bool(dropped and not added)
                if not made_progress:
                    # The synthetic decide closes the round the revise opened;
                    # stamp it `next_round` so step grouping stays monotonic
                    # (rounds, revise, decide all carry the same round).
                    self._forced_finish(
                        run, "revision produced no progress", next_round
                    )
                    break

                if added:
                    critique = await self._step(
                        run, StepKind.CRITIQUE, AgentRole.SKEPTIC,
                        lambda: skeptic.run(
                            task_text, claims=[q.claim for q in added],
                            run_id=run_id,
                        ),
                        round_=next_round,
                    )
                    pool.record_verdicts(critique.verdicts)
                    skeptic_content = critique.content
                rounds = next_round
                self._store.record_round(
                    run.id,
                    self._snapshot(rounds, pool, worker_content, skeptic_content),
                )

            # ---- synthesis from the best round ----
            best = select_best_round(run.rounds)
            best_board = render_evidence_board(
                [
                    QualifiedClaim(
                        claim=claim,
                        origin=ClaimOrigin(
                            agent=best.origins[claim.id], original_id=claim.id
                        ),
                    )
                    for claim in best.claims
                ],
                best.verdicts,
            )
            context = (
                render_synthesis_context(
                    best.worker_content.get(AgentRole.RESEARCHER, ""),
                    best.worker_content.get(AgentRole.IDEATOR, ""),
                    best.skeptic_content or None,
                    best_board,
                )
                + "\n\n"
                + render_iteration_history(run.rounds, best)
            )
            final = await self._step(
                run, StepKind.SYNTHESIZE, AgentRole.MANAGER,
                lambda: manager.run(
                    task_text, context=context, message_type=MessageType.SYNTHESIS,
                    run_id=run_id,
                ),
                round_=None,
            )

            # ---- Phase 11: independent verification of the final answer ----
            verifier = self._registry.get(AgentRole.VERIFIER)
            verify_context = render_verification_context(
                final.content, best.claims, best.verdicts
            )
            verification_msg = await self._step(
                run, StepKind.VERIFY, AgentRole.VERIFIER,
                lambda: verifier.run(
                    task_text, context=verify_context, claims=list(best.claims),
                    message_type=MessageType.VERIFICATION, run_id=run_id,
                ),
                round_=None,
            )

            # ---- Phase 11: process audit over the completed trace -----------
            # Facts are computed BEFORE the audit step exists (its own step
            # record is not part of the trace being audited).
            facts = compute_audit_facts(run.rounds, run.steps, final.content)
            audit_context = render_accountability_context(
                run, facts, verification_msg.verification
            )
            accountability = self._registry.get(AgentRole.ACCOUNTABILITY)
            await self._step(
                run, StepKind.AUDIT, AgentRole.ACCOUNTABILITY,
                lambda: accountability.run(
                    task_text, context=audit_context,
                    message_type=MessageType.ACCOUNTABILITY, run_id=run_id,
                ),
                round_=None,
                transform=lambda message: self._enforce_audit(
                    run, message, facts
                ),
            )

            run.final_message = final
            run.status = RunStatus.COMPLETED
            run.finished_at = datetime.now(UTC)
            self._store.append_event(
                run_id,
                RunEventType.RUN_COMPLETED,
                message=final,
                duration_ms=run.duration_ms,
            )
        except asyncio.CancelledError:
            run.status = RunStatus.FAILED
            run.finished_at = datetime.now(UTC)
            run.error = ErrorInfo(
                type="ServerShutdown",
                message="run cancelled during server shutdown",
            )
            if run.steps and run.steps[-1].status is StepStatus.RUNNING:
                run.steps[-1].status = StepStatus.FAILED
                run.steps[-1].error = run.error
            try:
                self._store.append_event(
                    run_id,
                    RunEventType.RUN_FAILED,
                    error=run.error,
                    duration_ms=run.duration_ms,
                )
            except Exception:  # pragma: no cover - subscribers may be gone
                pass
            raise
        except Exception as exc:
            logger.exception("run %s failed", run_id)
            run.status = RunStatus.FAILED
            run.finished_at = datetime.now(UTC)
            run.error = _error_info(exc)
            failed = next(
                (s for s in reversed(run.steps) if s.status is StepStatus.FAILED),
                None,
            )
            self._store.append_event(
                run_id,
                RunEventType.RUN_FAILED,
                step=failed.index if failed else None,
                kind=failed.kind if failed else None,
                agent=failed.agent if failed else None,
                round=failed.round if failed else None,
                error=run.error,
                duration_ms=run.duration_ms,
            )

    # ------------------------------------------------------------ step helpers

    async def _step(
        self,
        run: RunRecord,
        kind: StepKind,
        agent: AgentRole,
        call: Callable[[], Coroutine[Any, Any, AgentMessage]],
        *,
        round_: int | None,
        transform: Callable[[AgentMessage], AgentMessage] | None = None,
    ) -> AgentMessage:
        index = len(run.steps) + 1
        record = StepRecord(
            index=index,
            kind=kind,
            agent=agent,
            status=StepStatus.RUNNING,
            started_at=datetime.now(UTC),
            round=round_,
        )
        run.steps.append(record)
        self._store.append_event(
            run.id, RunEventType.STEP_STARTED, step=index, kind=kind,
            agent=agent, round=round_,
        )
        started = time.monotonic()
        try:
            message = await call()
            # Applied BEFORE persistence/publication so the stored message,
            # the step record, and the SSE event are always identical
            # (Phase 11 enforcement transform).
            if transform is not None:
                message = transform(message)
        except Exception as exc:
            record.status = StepStatus.FAILED
            record.duration_ms = round((time.monotonic() - started) * 1000, 1)
            record.error = _error_info(exc)
            raise
        record.status = StepStatus.COMPLETED
        record.duration_ms = round((time.monotonic() - started) * 1000, 1)
        record.message = message
        self._store.append_event(
            run.id, RunEventType.STEP_COMPLETED, step=index, kind=kind,
            agent=agent, round=round_, duration_ms=record.duration_ms,
            message=message,
        )
        return message

    def _enforce_audit(
        self, run: RunRecord, message: AgentMessage, facts: AuditFacts
    ) -> AgentMessage:
        """Merge code-canonical facts into the audit report (PRD §6.5.4).

        Runs as the AUDIT step's transform, so the enforced report is what the
        step record, the STEP_COMPLETED event, and persistence all receive.
        A missing report is a programming error and fails the step loudly.
        """
        report = message.accountability
        if report is None:
            raise ValueError("accountability step returned no report")
        enforced = enforce_accountability(report, facts)
        if enforced != report:
            logger.info(
                "audit_enforced run=%s added_flags=%d trace=%s status=%s",
                run.id,
                len(enforced.flags) - len(report.flags),
                enforced.trace_completeness,
                enforced.overall_status.value,
            )
        return message.model_copy(update={"accountability": enforced})

    def _mark_skipped(
        self, run: RunRecord, kind: StepKind, agent: AgentRole, *,
        round_: int | None,
    ) -> None:
        index = len(run.steps) + 1
        run.steps.append(
            StepRecord(
                index=index, kind=kind, agent=agent,
                status=StepStatus.SKIPPED,
                started_at=datetime.now(UTC),
                duration_ms=0.0, skipped=True, round=round_,
            )
        )
        self._store.append_event(
            run.id, RunEventType.STEP_COMPLETED, step=index, kind=kind,
            agent=agent, round=round_, duration_ms=0.0, skipped=True,
        )

    def _forced_finish(self, run: RunRecord, reason: str, round_: int) -> None:
        """Visible guard stop: synthetic skipped DECIDE step (PRD §6.6)."""
        logger.warning(
            "guard_triggered run=%s guard=%s round=%s", run.id, reason, round_
        )
        decision = ManagerDecision(
            action=DecisionAction.FINISH, reason=reason, confidence=0.0
        )
        message = AgentMessage(
            from_agent=AgentRole.MANAGER,
            type=MessageType.DECISION,
            content="",
            decision=decision,
        )
        index = len(run.steps) + 1
        run.steps.append(
            StepRecord(
                index=index, kind=StepKind.DECIDE, agent=AgentRole.MANAGER,
                status=StepStatus.SKIPPED,
                started_at=datetime.now(UTC),
                duration_ms=0.0, message=message, skipped=True, round=round_,
            )
        )
        self._store.append_event(
            run.id, RunEventType.STEP_COMPLETED, step=index,
            kind=StepKind.DECIDE, agent=AgentRole.MANAGER, round=round_,
            duration_ms=0.0, skipped=True, message=message,
        )

    @staticmethod
    def _snapshot(
        round_number: int,
        pool: PoolState,
        worker_content: dict[AgentRole, str],
        skeptic_content: str | None,
    ) -> RoundSnapshot:
        return RoundSnapshot(
            round_number=round_number,
            claims=pool.active_claims(),
            origins={
                item.claim.id: item.origin.agent for item in pool.claims
            },
            verdicts=pool.applicable_verdicts(),
            worker_content=dict(worker_content),
            skeptic_content=skeptic_content or "",
            supported_count=pool.supported_count(),
            unresolved_count=len(pool.unresolved()),
        )
