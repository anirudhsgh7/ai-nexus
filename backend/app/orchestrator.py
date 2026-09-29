"""Fixed 5-step pipeline: plan -> research || ideate -> critique -> synthesize.

Phase 4 is deliberately linear. The independence guarantee is structural:
Researcher and Ideator receive the same plan context and neither receives the
other's output; the Skeptic receives only re-identified claim records. Phase 5
replaces the step selection while keeping step mechanics, events, and errors.
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
from app.runs import (
    ErrorInfo,
    RunEventType,
    RunManager,
    RunRecord,
    RunStatus,
    StepKind,
    StepRecord,
    StepStatus,
)
from app.schemas import AgentMessage, AgentRole, Claim, MessageType, Verdict

logger = logging.getLogger("ai_nexus.orchestrator")

__all__ = [
    "PIPELINE",
    "ClaimOrigin",
    "Orchestrator",
    "QualifiedClaim",
    "qualify_claims",
    "render_evidence_board",
    "render_synthesis_context",
    "resolve_claim",
]

PIPELINE: tuple[tuple[StepKind, AgentRole], ...] = (
    (StepKind.PLAN, AgentRole.MANAGER),
    (StepKind.RESEARCH, AgentRole.RESEARCHER),
    (StepKind.IDEATE, AgentRole.IDEATOR),
    (StepKind.CRITIQUE, AgentRole.SKEPTIC),
    (StepKind.SYNTHESIZE, AgentRole.MANAGER),
)

_SKIPPED_CRITIQUE_NOTE = "(skipped: no claims were produced to evaluate)"


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
) -> list[QualifiedClaim]:
    """Deterministic run-level re-identification c1..cK in batch order.

    Claims are copied with new ids; origins keep (agent, original_id) so
    verdicts referencing run-level ids can be mapped back for display.
    """
    qualified: list[QualifiedClaim] = []
    counter = 0
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
    """Golden claim+verdict view shared by synthesis (and Phase 5 replanning)."""
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
    """Canonical Manager synthesis context (PRD §6.4)."""
    skeptic_body = (
        skeptic_content if skeptic_content is not None else _SKIPPED_CRITIQUE_NOTE
    )
    board_body = board if board else "(none)"
    return (
        f"RESEARCHER FINDINGS:\n{researcher_content}\n\n"
        f"IDEATOR OPTIONS:\n{ideator_content}\n\n"
        f"SKEPTIC CRITIQUE:\n{skeptic_body}\n\n"
        f"EVALUATED CLAIMS:\n{board_body}"
    )


# ---------------------------------------------------------------- orchestrator


def _error_info(exc: Exception) -> ErrorInfo:
    return ErrorInfo(
        type=type(exc).__name__,
        message=str(exc),
        hint=getattr(exc, "hint", "") or "",
    )


class Orchestrator:
    """Executes the fixed pipeline and publishes run events (PRD §6.5)."""

    def __init__(self, registry: AgentRegistry, store: RunManager) -> None:
        self._registry = registry
        self._store = store

    async def execute(self, run_id: str) -> None:
        run = self._store.get(run_id)
        if run is None:
            return
        manager = self._registry.get(AgentRole.MANAGER)
        researcher = self._registry.get(AgentRole.RESEARCHER)
        ideator = self._registry.get(AgentRole.IDEATOR)
        skeptic = self._registry.get(AgentRole.SKEPTIC)

        try:
            run.started_at = datetime.now(UTC)
            run.status = RunStatus.RUNNING
            task_text = run.task
            self._store.append_event(
                run_id, RunEventType.RUN_STARTED, task=task_text
            )

            plan = await self._step(
                run, StepKind.PLAN, AgentRole.MANAGER,
                lambda: manager.run(task_text),
            )
            research = await self._step(
                run, StepKind.RESEARCH, AgentRole.RESEARCHER,
                lambda: researcher.run(task_text, context=plan.content),
            )
            ideation = await self._step(
                run, StepKind.IDEATE, AgentRole.IDEATOR,
                lambda: ideator.run(task_text, context=plan.content),
            )

            qualified = qualify_claims(
                [
                    (AgentRole.RESEARCHER, research.claims or []),
                    (AgentRole.IDEATOR, ideation.claims or []),
                ]
            )
            critique: AgentMessage | None
            if qualified:
                critique = await self._step(
                    run, StepKind.CRITIQUE, AgentRole.SKEPTIC,
                    lambda: skeptic.run(
                        task_text, claims=[q.claim for q in qualified]
                    ),
                )
            else:
                critique = None
                self._mark_skipped(run, StepKind.CRITIQUE, AgentRole.SKEPTIC)

            board = render_evidence_board(
                qualified, critique.verdicts or [] if critique else []
            )
            context = render_synthesis_context(
                research.content,
                ideation.content,
                critique.content if critique else None,
                board,
            )
            final = await self._step(
                run, StepKind.SYNTHESIZE, AgentRole.MANAGER,
                lambda: manager.run(
                    task_text, context=context, message_type=MessageType.SYNTHESIS
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
    ) -> AgentMessage:
        index = len(run.steps) + 1
        record = StepRecord(
            index=index,
            kind=kind,
            agent=agent,
            status=StepStatus.RUNNING,
            started_at=datetime.now(UTC),
        )
        run.steps.append(record)
        self._store.append_event(
            run.id, RunEventType.STEP_STARTED, step=index, kind=kind, agent=agent
        )
        started = time.monotonic()
        try:
            message = await call()
        except Exception as exc:
            record.status = StepStatus.FAILED
            record.duration_ms = round((time.monotonic() - started) * 1000, 1)
            record.error = _error_info(exc)
            raise
        record.status = StepStatus.COMPLETED
        record.duration_ms = round((time.monotonic() - started) * 1000, 1)
        record.message = message
        self._store.append_event(
            run.id,
            RunEventType.STEP_COMPLETED,
            step=index,
            kind=kind,
            agent=agent,
            duration_ms=record.duration_ms,
            message=message,
        )
        return message

    def _mark_skipped(self, run: RunRecord, kind: StepKind, agent: AgentRole) -> None:
        index = len(run.steps) + 1
        run.steps.append(
            StepRecord(
                index=index,
                kind=kind,
                agent=agent,
                status=StepStatus.SKIPPED,
                started_at=datetime.now(UTC),
                duration_ms=0.0,
                skipped=True,
            )
        )
        self._store.append_event(
            run.id,
            RunEventType.STEP_COMPLETED,
            step=index,
            kind=kind,
            agent=agent,
            duration_ms=0.0,
            skipped=True,
        )
