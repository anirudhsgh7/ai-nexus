"""Run lifecycle API: create, inspect, list, and stream events (SSE)."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.orchestrator import Orchestrator
from app.runs import (
    TERMINAL_EVENT_TYPES,
    RunActiveError,
    RunEvent,
    RunManager,
    RunRecord,
    StepRecord,
)

router = APIRouter(tags=["runs"])

SSE_KEEPALIVE_S = 15.0


class RunCreateRequest(BaseModel):
    task: str = Field(min_length=1, max_length=RunManager.MAX_TASK_CHARS)

    @field_validator("task")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task must be non-empty")
        return value


def _error_response(
    status: int, error_type: str, message: str, hint: str = "", **extra: Any
) -> JSONResponse:
    error: dict[str, Any] = {"type": error_type, "message": message, "hint": hint}
    error.update(extra)
    return JSONResponse(status_code=status, content={"error": error})


def _run_not_found(run_id: str) -> JSONResponse:
    return _error_response(404, "RunNotFoundError", f"unknown run: {run_id}")


def _decision_summary(decision: Any) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "action": decision.action.value,
        "target": decision.target.value if decision.target else None,
        "reason": decision.reason,
    }


def _step_payload(step: StepRecord) -> dict[str, Any]:
    return {
        "index": step.index,
        "kind": step.kind.value,
        "agent": step.agent.value,
        "status": step.status.value,
        "round": step.round,
        "duration_ms": step.duration_ms,
        "skipped": step.skipped,
        "message": step.message.model_dump(mode="json") if step.message else None,
        "error": step.error.model_dump() if step.error else None,
    }


def _run_payload(run: RunRecord) -> dict[str, Any]:
    return {
        "run_id": run.id,
        "task": run.task,
        "status": run.status.value,
        "created_at": run.created_at.isoformat(),
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "duration_ms": run.duration_ms,
        "steps": [_step_payload(s) for s in run.steps],
        "rounds": [
            {
                "round": snap.round_number,
                "supported": snap.supported_count,
                "unresolved": snap.unresolved_count,
                "decision": _decision_summary(snap.decision),
            }
            for snap in run.rounds
        ],
        "final_message": (
            run.final_message.model_dump(mode="json") if run.final_message else None
        ),
        "error": run.error.model_dump() if run.error else None,
    }


def _summary_payload(run: RunRecord) -> dict[str, Any]:
    return {
        "run_id": run.id,
        "status": run.status.value,
        "task_preview": run.task[:120],
        "created_at": run.created_at.isoformat(),
        "duration_ms": run.duration_ms,
    }


@router.post("/runs", status_code=202)
async def create_run(request: Request, body: RunCreateRequest) -> Any:
    runs: RunManager = request.app.state.runs
    orchestrator: Orchestrator = request.app.state.orchestrator
    try:
        run = runs.create(body.task)
    except RunActiveError as exc:
        return _error_response(
            409,
            "RunActiveError",
            "a run is already active",
            f"wait for run {exc.active_run_id} to finish",
            active_run_id=exc.active_run_id,
        )
    task = asyncio.create_task(orchestrator.execute(run.id))
    runs.register_task(run.id, task)
    return {
        "run_id": run.id,
        "status": run.status.value,
        "task": run.task,
        "created_at": run.created_at.isoformat(),
    }


@router.get("/runs")
async def list_runs(request: Request, limit: int = 20) -> Any:
    runs: RunManager = request.app.state.runs
    return [_summary_payload(r) for r in runs.list(limit=limit)]


@router.get("/runs/{run_id}")
async def get_run(request: Request, run_id: str) -> Any:
    run = request.app.state.runs.get(run_id)
    if run is None:
        return _run_not_found(run_id)
    return _run_payload(run)


def _parse_since(request: Request) -> int:
    raw = request.headers.get("last-event-id")
    if raw is None:
        raw = request.query_params.get("since", "0")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(value, 0)


def _sse_frame(event: RunEvent) -> str:
    return (
        f"id: {event.seq}\n"
        f"event: {event.type.value}\n"
        f"data: {event.model_dump_json()}\n\n"
    )


@router.get("/runs/{run_id}/events")
async def stream_events(request: Request, run_id: str) -> Any:
    runs: RunManager = request.app.state.runs
    if runs.get(run_id) is None:
        return _run_not_found(run_id)
    since = _parse_since(request)

    async def generate() -> AsyncIterator[str]:
        queue, backlog = runs.subscribe(run_id, since)
        try:
            for event in backlog:
                yield _sse_frame(event)
                if event.type in TERMINAL_EVENT_TYPES:
                    return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), SSE_KEEPALIVE_S)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield _sse_frame(event)
                if event.type in TERMINAL_EVENT_TYPES:
                    return
        finally:
            runs.unsubscribe(run_id, queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
