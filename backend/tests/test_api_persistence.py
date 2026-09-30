"""Two-app restart simulation through the real FastAPI app (PRD §9).

App A completes a run (FakeOrchestrator), app A closes, app B — a brand-new
`create_app()` over the same per-test SQLite file — must serve the full trace,
list it, and replay byte-identical SSE frames. This is the offline proof of
PLAN.md's exit criterion ("restart the app, reload a past session with its
full trace").
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.api.runs import _sse_frame
from app.config import clear_settings_cache
from app.db import PersistenceError
from app.main import create_app
from app.runs import RunEventType
from tests.test_api_runs import FakeOrchestrator

TASK = "persist this run"


def _complete_run_in_app() -> tuple[list[str], str]:
    """App A: create + complete a run; returns (live SSE frames, run_id)."""
    app = create_app()
    with TestClient(app) as client:
        app.state.orchestrator = FakeOrchestrator(app.state.runs)
        response = client.post("/api/runs", json={"task": TASK})
        assert response.status_code == 202
        run_id = response.json()["run_id"]

        deadline = time.monotonic() + 5.0
        body = client.get(f"/api/runs/{run_id}").json()
        while body["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.01)
            body = client.get(f"/api/runs/{run_id}").json()
        assert body["status"] == "completed"

        record = app.state.runs.get(run_id)
        frames = [_sse_frame(event) for event in record.events]
    return frames, run_id


def test_restart_serves_full_trace_and_replays_events():
    live_frames, run_id = _complete_run_in_app()

    # ---- "restart": a brand-new app over the same database ----
    app = create_app()
    with TestClient(app) as client:
        body = client.get(f"/api/runs/{run_id}").json()
        assert body["status"] == "completed"
        assert body["task"] == TASK
        assert body["duration_ms"] is not None
        steps = body["steps"]
        assert [step["kind"] for step in steps] == ["plan"]
        assert steps[0]["status"] == "completed"
        assert steps[0]["message"]["content"] == "plan"
        assert steps[0]["message"]["claims"] is None
        assert body["final_message"]["content"] == "plan"
        assert body["error"] is None

        # history list includes the persisted run
        listed = client.get("/api/runs").json()
        assert [item["run_id"] for item in listed] == [run_id]
        assert listed[0]["status"] == "completed"

        # full event replay, byte-identical to what live subscribers saw
        with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
            replayed = "".join(stream.iter_text())

    assert replayed == "".join(live_frames)
    assert "run_completed" in replayed


def test_restart_unknown_run_still_404():
    _complete_run_in_app()
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/api/runs/definitely-not-a-run").status_code == 404


def test_disabled_persistence_is_memory_only(monkeypatch):
    _, run_id = _complete_run_in_app()

    monkeypatch.setenv("AI_NEXUS_DB_PATH", "")
    clear_settings_cache()
    app = create_app()
    with TestClient(app) as client:
        # the previous run's history is simply gone: Phase 6 behavior
        assert client.get("/api/runs").json() == []
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 404
        # and a run created here still works end-to-end in memory
        app.state.orchestrator = FakeOrchestrator(app.state.runs)
        created = client.post("/api/runs", json={"task": "memory only"})
        run_id = created.json()["run_id"]
        deadline = time.monotonic() + 5.0
        body = client.get(f"/api/runs/{run_id}").json()
        while body["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.01)
            body = client.get(f"/api/runs/{run_id}").json()
        assert body["status"] == "completed"


def test_persistence_error_maps_to_503():
    _complete_run_in_app()
    app = create_app()
    with TestClient(app) as client:
        def _boom(run_id):
            raise PersistenceError(
                "load_run failed: disk gone", hint="check the disk"
            )

        app.state.runs._store.load_run = _boom
        response = client.get("/api/runs/not-in-memory")
        assert response.status_code == 503
        error = response.json()["error"]
        assert error["type"] == "PersistenceError"
        assert "disk gone" in error["message"]
        assert error["hint"] == "check the disk"


def test_restart_recovers_interrupted_run_as_failed():
    """App A dies mid-run (running row left behind) -> app B shows it failed."""
    app_a = create_app()
    with TestClient(app_a):
        runs = app_a.state.runs
        run = runs.create("interrupted by restart")
        runs.append_event(run.id, RunEventType.RUN_STARTED, task=run.task)
        # no terminal event: simulate process death by closing app A
    # process is gone; same DB, new process
    app_b = create_app()
    with TestClient(app_b) as client_b:
        body = client_b.get(f"/api/runs/{run.id}").json()
        assert body["status"] == "failed"
        assert body["error"]["type"] == "ServerRestart"
        assert body["error"]["hint"]
