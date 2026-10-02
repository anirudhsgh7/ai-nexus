"""Run state, event history, subscriber fan-out, and optional persistence.

`RunManager` is the single owner of run records, step records, and the SSE
event stream (history + live subscriber queues). Phase 7 adds durable storage
*behind* this interface: constructed with a `store` (app.db.RunStore), every
mutation writes through and every read falls back to disk on a memory miss;
constructed without one, behavior is byte-identical to Phase 6.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Final
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import AgentMessage, AgentRole, Claim, ManagerDecision, Verdict

if TYPE_CHECKING:  # pragma: no cover - annotation only, no runtime cycle
    from app.db import RunStore

logger = logging.getLogger("ai_nexus.runs")

__all__ = [
    "ErrorInfo",
    "RoundSnapshot",
    "RunActiveError",
    "RunEvent",
    "RunEventType",
    "RunManager",
    "RunRecord",
    "RunStatus",
    "STREAM_CLOSED",
    "StepKind",
    "StepRecord",
    "StepStatus",
    "TERMINAL_EVENT_TYPES",
]

#: Queue sentinel pushed by `RunManager.close_streams()` during shutdown so
#: SSE handlers return immediately instead of idling a keep-alive window
#: (Phase 10 PRD §6.1). Identity-compared, never serialized.
STREAM_CLOSED: Final[object] = object()


def _now() -> datetime:
    return datetime.now(UTC)


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StepKind(str, Enum):
    PLAN = "plan"
    RESEARCH = "research"
    IDEATE = "ideate"
    CRITIQUE = "critique"
    DECIDE = "decide"          # Phase 5
    REVISE = "revise"          # Phase 5
    SYNTHESIZE = "synthesize"


class StepStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunEventType(str, Enum):
    RUN_STARTED = "run_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


TERMINAL_EVENT_TYPES = frozenset({RunEventType.RUN_COMPLETED, RunEventType.RUN_FAILED})


class ErrorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    message: str
    hint: str = ""


class RunEvent(BaseModel):
    """Wire object for SSE and NDJSON consumers."""

    model_config = ConfigDict(extra="forbid")

    seq: int = Field(ge=1)
    type: RunEventType
    run_id: str
    ts: datetime = Field(default_factory=_now)
    step: int | None = Field(default=None, ge=1)
    kind: StepKind | None = None
    agent: AgentRole | None = None
    round: int | None = Field(default=None, ge=1)
    task: str | None = None
    message: AgentMessage | None = None
    duration_ms: float | None = None
    skipped: bool | None = None
    error: ErrorInfo | None = None


@dataclass(slots=True)
class StepRecord:
    index: int
    kind: StepKind
    agent: AgentRole
    status: StepStatus
    started_at: datetime
    duration_ms: float | None = None
    message: AgentMessage | None = None
    skipped: bool = False
    error: ErrorInfo | None = None
    round: int | None = None


@dataclass(slots=True)
class RoundSnapshot:
    """Evidence state at the end of one critique round (Phase 5 PRD §6.4)."""

    round_number: int
    claims: list[Claim]
    origins: dict[str, AgentRole]
    verdicts: list[Verdict]
    worker_content: dict[AgentRole, str]
    skeptic_content: str
    supported_count: int
    unresolved_count: int
    decision: ManagerDecision | None = None


@dataclass(slots=True)
class RunRecord:
    id: str
    task: str
    status: RunStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    steps: list[StepRecord] = field(default_factory=list)
    final_message: AgentMessage | None = None
    error: ErrorInfo | None = None
    events: list[RunEvent] = field(default_factory=list)
    rounds: list[RoundSnapshot] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return round((self.finished_at - self.started_at).total_seconds() * 1000, 1)


class RunActiveError(Exception):
    def __init__(self, active_run_id: str) -> None:
        super().__init__("a run is already active")
        self.active_run_id = active_run_id


class RunManager:
    """Records + event history + live fan-out, in memory (Phase 7 persists)."""

    MAX_RUNS = 50
    MAX_TASK_CHARS = 4000

    def __init__(
        self, max_runs: int = MAX_RUNS, *, store: "RunStore | None" = None
    ) -> None:
        self._max_runs = max_runs
        self._store = store
        self._runs: dict[str, RunRecord] = {}
        self._subscribers: dict[str, set[asyncio.Queue[RunEvent]]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        if store is not None:
            # startup recovery: interrupted runs become visible failed runs
            for recovered_id in store.recover_interrupted():
                logger.warning("persistence_recovered run=%s", recovered_id)

    # ------------------------------------------------------------- lifecycle

    def create(self, task: str) -> RunRecord:
        stripped = task.strip()
        if not stripped:
            raise ValueError("task must be non-empty")
        if len(task) > self.MAX_TASK_CHARS:
            raise ValueError(f"task exceeds {self.MAX_TASK_CHARS} characters")
        active = self.active()
        if active is not None:
            raise RunActiveError(active.id)

        run = RunRecord(
            id=uuid4().hex,
            task=stripped,
            status=RunStatus.RUNNING,
            created_at=_now(),
        )
        if self._store is not None:
            self._store.save_run(run)   # durable before visible
        self._runs[run.id] = run
        self._evict_oldest_completed()
        return run

    def get(self, run_id: str) -> RunRecord | None:
        run = self._runs.get(run_id)
        if run is not None or self._store is None:
            return run
        loaded = self._store.load_run(run_id)
        if loaded is None:
            return None
        # full loads may be cached (stubs from list() never are); never
        # evict the run we are serving right now
        self._runs[loaded.id] = loaded
        self._evict_oldest_completed(exclude=loaded.id)
        return loaded

    def list(self, limit: int = 20) -> list[RunRecord]:
        if self._store is None:
            clamped = max(1, min(limit, self._max_runs))
            newest = sorted(
                self._runs.values(), key=lambda r: r.created_at, reverse=True
            )
            return newest[:clamped]
        # history may exceed the memory cache: clamp the caller's limit only
        effective = max(1, limit)
        merged = dict(self._runs)
        for stub in self._store.list_summaries(effective):
            merged.setdefault(stub.id, stub)   # memory wins for live runs
        newest = sorted(
            merged.values(), key=lambda r: (r.created_at, r.id), reverse=True
        )
        return newest[:effective]

    def active(self) -> RunRecord | None:
        for run in self._runs.values():
            if run.status is RunStatus.RUNNING:
                return run
        return None

    def _evict_oldest_completed(self, exclude: str | None = None) -> None:
        while len(self._runs) > self._max_runs:
            candidates = [
                r for r in self._runs.values()
                if r.status is not RunStatus.RUNNING and r.id != exclude
            ]
            if not candidates:
                break
            oldest = min(candidates, key=lambda r: r.created_at)
            self._runs.pop(oldest.id, None)
            self._subscribers.pop(oldest.id, None)
            self._tasks.pop(oldest.id, None)

    # ----------------------------------------------------------------- events

    def append_event(
        self, run_id: str, event_type: RunEventType, **fields: object
    ) -> RunEvent:
        run = self._runs[run_id]  # KeyError for unknown run
        seq = run.events[-1].seq + 1 if run.events else 1
        event = RunEvent(seq=seq, type=event_type, run_id=run_id, **fields)
        run.events.append(event)
        for queue in self._subscribers.get(run_id, ()):
            queue.put_nowait(event)
        if self._store is not None:
            # persist AFTER fan-out: live subscribers never stall on disk; a
            # write failure still raises and fails the run visibly (PRD §6.10)
            self._store.append_event(event)
        return event

    def subscribe(
        self, run_id: str, since: int = 0
    ) -> tuple[asyncio.Queue[RunEvent], list[RunEvent]]:
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        run = self._runs.get(run_id)
        if run is not None:
            backlog = [e for e in run.events if e.seq > since]
            self._subscribers.setdefault(run_id, set()).add(queue)
            return queue, backlog
        if self._store is None:
            raise KeyError(run_id)  # pre-Phase-7 contract
        # DB-only run: replay stored events; a persisted run is terminal by
        # construction, so no live subscription is registered
        backlog = self._store.load_events(run_id, since)
        return queue, backlog

    # --------------------------------------------------------------- rounds

    def record_round(self, run_id: str, snapshot: RoundSnapshot) -> None:
        run = self._runs[run_id]
        run.rounds.append(snapshot)
        if self._store is not None:
            self._store.record_round(run_id, snapshot)

    def record_round_decision(
        self, run_id: str, decision: ManagerDecision
    ) -> None:
        run = self._runs[run_id]
        if not run.rounds:
            return
        run.rounds[-1].decision = decision
        if self._store is not None:
            self._store.record_round_decision(
                run_id, run.rounds[-1].round_number, decision
            )

    def unsubscribe(self, run_id: str, queue: asyncio.Queue[RunEvent]) -> None:
        subscribers = self._subscribers.get(run_id)
        if not subscribers:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(run_id, None)

    def close_streams(self) -> int:
        """End every live SSE stream immediately; returns the queue count.

        Called at the top of lifespan shutdown so stream handlers return
        before uvicorn cancels them mid-frame (Phase 10 PRD §6.1). Idempotent:
        a second call sees no subscribers and returns 0. The SSE generator
        treats `STREAM_CLOSED` as end-of-stream — no event frame, no terminal
        event (the run's own failure event is emitted by the cancelled
        orchestrator task).
        """
        closed = 0
        for queues in self._subscribers.values():
            for queue in queues:
                queue.put_nowait(STREAM_CLOSED)  # type: ignore[arg-type]
                closed += 1
        self._subscribers.clear()
        return closed

    # ------------------------------------------------------------------ tasks

    def register_task(self, run_id: str, task: asyncio.Task) -> None:
        self._tasks[run_id] = task

    def get_task(self, run_id: str) -> asyncio.Task | None:
        return self._tasks.get(run_id)

    def cancel_active_task(self) -> asyncio.Task | None:
        active = self.active()
        if active is None:
            return None
        task = self._tasks.get(active.id)
        if task is not None and not task.done():
            task.cancel()
        return task
