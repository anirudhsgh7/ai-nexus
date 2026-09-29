"""In-memory run state, event history, and subscriber fan-out.

`RunManager` is the single owner of run records, step records, and the SSE
event stream (history + live subscriber queues). Phase 7 replaces only the
storage behind this interface; the API and orchestrator keep their contracts.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import AgentMessage, AgentRole

__all__ = [
    "ErrorInfo",
    "RunActiveError",
    "RunEvent",
    "RunEventType",
    "RunManager",
    "RunRecord",
    "RunStatus",
    "StepKind",
    "StepRecord",
    "StepStatus",
    "TERMINAL_EVENT_TYPES",
]


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

    def __init__(self, max_runs: int = MAX_RUNS) -> None:
        self._max_runs = max_runs
        self._runs: dict[str, RunRecord] = {}
        self._subscribers: dict[str, set[asyncio.Queue[RunEvent]]] = {}
        self._tasks: dict[str, asyncio.Task] = {}

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
        self._runs[run.id] = run
        self._evict_oldest_completed()
        return run

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def list(self, limit: int = 20) -> list[RunRecord]:
        clamped = max(1, min(limit, self._max_runs))
        newest = sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)
        return newest[:clamped]

    def active(self) -> RunRecord | None:
        for run in self._runs.values():
            if run.status is RunStatus.RUNNING:
                return run
        return None

    def _evict_oldest_completed(self) -> None:
        while len(self._runs) > self._max_runs:
            candidates = [
                r for r in self._runs.values() if r.status is not RunStatus.RUNNING
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
        return event

    def subscribe(
        self, run_id: str, since: int = 0
    ) -> tuple[asyncio.Queue[RunEvent], list[RunEvent]]:
        run = self._runs[run_id]  # KeyError for unknown run
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        backlog = [e for e in run.events if e.seq > since]
        self._subscribers.setdefault(run_id, set()).add(queue)
        return queue, backlog

    def unsubscribe(self, run_id: str, queue: asyncio.Queue[RunEvent]) -> None:
        subscribers = self._subscribers.get(run_id)
        if not subscribers:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(run_id, None)

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
