"""Runs API contract: POST/GET/list/SSE with a fake orchestrator (PRD §6.6)."""

import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import runs as runs_api
from app.main import create_app
from app.runs import (
    RunEventType,
    RunManager,
    RunStatus,
    StepKind,
    StepRecord,
    StepStatus,
)
from app.schemas import AgentMessage, AgentRole, MessageType

TASK = "Should we build X?"


class FakeOrchestrator:
    """Publishes one completed plan step and finishes; no LLM calls."""

    def __init__(self, runs: RunManager) -> None:
        self.runs = runs

    async def execute(self, run_id: str) -> None:
        run = self.runs.get(run_id)
        assert run is not None
        from datetime import UTC, datetime

        run.started_at = datetime.now(UTC)
        run.status = RunStatus.RUNNING
        self.runs.append_event(run_id, RunEventType.RUN_STARTED, task=run.task)

        message = AgentMessage(
            from_agent=AgentRole.MANAGER, type=MessageType.PLAN, content="plan"
        )
        step = StepRecord(
            index=1, kind=StepKind.PLAN, agent=AgentRole.MANAGER,
            status=StepStatus.RUNNING, started_at=datetime.now(UTC),
        )
        run.steps.append(step)
        self.runs.append_event(
            run_id, RunEventType.STEP_STARTED, step=1,
            kind=StepKind.PLAN, agent=AgentRole.MANAGER,
        )
        step.status = StepStatus.COMPLETED
        step.duration_ms = 1.0
        step.message = message
        self.runs.append_event(
            run_id, RunEventType.STEP_COMPLETED, step=1,
            kind=StepKind.PLAN, agent=AgentRole.MANAGER,
            duration_ms=1.0, message=message,
        )
        run.final_message = message
        run.status = RunStatus.COMPLETED
        run.finished_at = datetime.now(UTC)
        self.runs.append_event(
            run_id, RunEventType.RUN_COMPLETED, message=message, duration_ms=1.0
        )


@pytest.fixture
def api():
    app = create_app()
    with TestClient(app) as client:
        app.state.orchestrator = FakeOrchestrator(app.state.runs)
        yield client, app


def _wait_terminal(client: TestClient, run_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/runs/{run_id}").json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(0.01)
    raise AssertionError("run did not reach a terminal state")


# ------------------------------------------------------------------ POST

def test_post_creates_run_and_completes(api):
    client, _ = api
    response = client.post("/api/runs", json={"task": TASK})
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "running"
    assert body["task"] == TASK
    assert len(body["run_id"]) == 32

    final = _wait_terminal(client, body["run_id"])
    assert final["status"] == "completed"
    assert final["duration_ms"] is not None
    assert final["steps"][0]["kind"] == "plan"
    assert final["steps"][0]["round"] is None
    assert final["steps"][0]["message"]["content"] == "plan"
    assert final["final_message"]["type"] == "plan"
    assert final["error"] is None
    # Phase 5: per-round summaries present (fake orchestrator creates none)
    assert final["rounds"] == []


def test_run_payload_round_summaries_round_trip():
    """GET shape for rounds: counts + FULL decision (Phase 9 PRD §6.8/§6.1)."""
    from datetime import UTC, datetime

    from app.api.runs import _run_payload
    from app.runs import RoundSnapshot, RunStatus
    from app.schemas import DecisionAction, ManagerDecision

    run = appless_run()
    run.rounds.append(
        RoundSnapshot(
            round_number=1, claims=[], origins={}, verdicts=[],
            worker_content={}, skeptic_content="", supported_count=3,
            unresolved_count=2,
            decision=ManagerDecision(
                action=DecisionAction.CALL_AGENT, target=AgentRole.RESEARCHER,
                instruction="Find a source.", reason="gap", confidence=0.8,
            ),
        )
    )
    payload = _run_payload(run)
    assert payload["rounds"] == [
        {
            "round": 1,
            "supported": 3,
            "unresolved": 2,
            "decision": {
                "action": "call_agent",
                "target": "researcher",
                "instruction": "Find a source.",
                "reason": "gap",
                "confidence": 0.8,
            },
            "claims": [],
            "origins": {},
            "verdicts": [],
        }
    ]
    # not completed yet -> nothing was selected
    assert payload["selected_round"] is None


def test_round_payload_evidence_round_trip():
    """Snapshot claims/origins/verdicts survive serialization verbatim."""
    from app.api.runs import _run_payload
    from app.runs import RoundSnapshot
    from app.schemas import (
        Claim,
        ClaimStatus,
        ClaimVerdict,
        Evidence,
        Verdict,
    )

    run = appless_run()
    run.rounds.append(
        RoundSnapshot(
            round_number=1,
            claims=[
                Claim(
                    id="c1",
                    statement="Growth was 40% in 2025.",
                    status=ClaimStatus.UNVERIFIED,
                    confidence=0.55,
                    evidence=[
                        Evidence(source="company blog", quote="revenue up 40%")
                    ],
                ),
                Claim(
                    id="c2",
                    statement="Rivals grew slower.",
                    status=ClaimStatus.ASSUMPTION,
                ),
            ],
            origins={"c1": AgentRole.RESEARCHER, "c2": AgentRole.IDEATOR},
            verdicts=[
                Verdict(
                    claim_id="c1",
                    verdict=ClaimVerdict.REFUTED,
                    objection="marketing post, not audited",
                    evidence=[Evidence(source="audit report", quote="23%")],
                ),
                Verdict(
                    claim_id="c2",
                    verdict=ClaimVerdict.UNVERIFIABLE,
                    objection="no comparison data",
                ),
            ],
            worker_content={},
            skeptic_content="",
            supported_count=0,
            unresolved_count=2,
        )
    )
    payload = _run_payload(run)
    round_payload = payload["rounds"][0]
    assert round_payload["claims"][0] == {
        "id": "c1",
        "statement": "Growth was 40% in 2025.",
        "status": "unverified",
        "confidence": 0.55,
        "evidence": [{"source": "company blog", "quote": "revenue up 40%"}],
    }
    assert round_payload["claims"][1]["evidence"] == []
    assert round_payload["origins"] == {"c1": "researcher", "c2": "ideator"}
    assert round_payload["verdicts"][0] == {
        "claim_id": "c1",
        "verdict": "refuted",
        "objection": "marketing post, not audited",
        "evidence": [{"source": "audit report", "quote": "23%"}],
    }
    assert round_payload["verdicts"][1]["evidence"] == []
    assert payload["selected_round"] is None


def test_selected_round_uses_best_round_rule():
    """selected_round is select_best_round's pick — ties go to the later round."""
    from app.api.runs import _run_payload
    from app.runs import RoundSnapshot, RunStatus

    def _snap(number: int, supported: int, unresolved: int) -> RoundSnapshot:
        return RoundSnapshot(
            round_number=number, claims=[], origins={}, verdicts=[],
            worker_content={}, skeptic_content="",
            supported_count=supported, unresolved_count=unresolved,
        )

    run = appless_run()
    run.rounds.extend([_snap(1, 3, 2), _snap(2, 6, 2)])
    # running -> null (nothing selected yet)
    assert _run_payload(run)["selected_round"] is None
    # completed -> the net-score winner (6-2 > 3-2)
    run.status = RunStatus.COMPLETED
    assert _run_payload(run)["selected_round"] == 2
    # tie -> later round
    tied = appless_run()
    tied.rounds.extend([_snap(1, 3, 2), _snap(2, 3, 2)])
    tied.status = RunStatus.COMPLETED
    assert _run_payload(tied)["selected_round"] == 2
    # failed run -> null even with snapshots
    run.status = RunStatus.FAILED
    assert _run_payload(run)["selected_round"] is None


def appless_run():
    from app.runs import RunManager, RunStatus

    manager = RunManager()
    return manager.create("payload test")


def test_step_payload_carries_tool_trace():
    """Phase 6: tool activity rides the step message into GET payloads (PRD §11.9)."""
    from datetime import UTC, datetime

    from app.api.runs import _run_payload
    from app.llm.base import ToolCall
    from app.schemas import ToolResult

    run = appless_run()
    run.steps.append(
        StepRecord(
            index=1, kind=StepKind.CRITIQUE, agent=AgentRole.SKEPTIC,
            status=StepStatus.COMPLETED, started_at=datetime.now(UTC),
            duration_ms=5.0,
            message=AgentMessage(
                from_agent=AgentRole.SKEPTIC, type=MessageType.CRITIQUE,
                content="refuted",
                tool_calls=[
                    ToolCall(
                        name="file_search",
                        arguments={"query": "growth"},
                        id="call1",
                    )
                ],
                tool_results=[
                    ToolResult(
                        name="file_search",
                        content='{"ok":true,"tool":"file_search","matches":[]}',
                        error=None,
                        duration_ms=1.2,
                    )
                ],
            ),
        )
    )
    message = _run_payload(run)["steps"][0]["message"]
    assert message["tool_calls"][0]["name"] == "file_search"
    assert message["tool_calls"][0]["arguments"] == {"query": "growth"}
    assert message["tool_results"][0]["name"] == "file_search"
    assert message["tool_results"][0]["error"] is None
    assert message["tool_results"][0]["duration_ms"] == 1.2


def test_post_while_active_returns_409(api):
    client, app = api
    blocker = app.state.runs.create("blocker")
    response = client.post("/api/runs", json={"task": "another"})
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["type"] == "RunActiveError"
    assert error["active_run_id"] == blocker.id
    assert blocker.id in error["hint"]


def test_post_invalid_task_422(api):
    client, _ = api
    assert client.post("/api/runs", json={"task": "   "}).status_code == 422
    assert client.post("/api/runs", json={"task": "x" * 4001}).status_code == 422
    assert client.post("/api/runs", json={}).status_code == 422


# ------------------------------------------------------------------ GET

def test_get_unknown_run_404(api):
    client, _ = api
    response = client.get("/api/runs/nope")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "RunNotFoundError"


def test_list_summaries_newest_first(api):
    client, app = api
    first = app.state.runs.create("first task")
    first.status = RunStatus.COMPLETED
    first.finished_at = first.created_at
    second = app.state.runs.create("second task")
    second.status = RunStatus.COMPLETED
    second.finished_at = second.created_at

    response = client.get("/api/runs")
    assert response.status_code == 200
    ids = [item["run_id"] for item in response.json()]
    assert ids == [second.id, first.id]
    assert response.json()[0]["task_preview"] == "second task"


# ------------------------------------------------------------------ SSE

def _append_terminal_events(app, runs: RunManager) -> str:
    run = runs.create("sse task")
    message = AgentMessage(
        from_agent=AgentRole.MANAGER, type=MessageType.PLAN, content="final"
    )
    runs.append_event(run.id, RunEventType.RUN_STARTED, task=run.task)
    runs.append_event(
        run.id, RunEventType.STEP_STARTED, step=1,
        kind=StepKind.PLAN, agent=AgentRole.MANAGER,
    )
    runs.append_event(
        run.id, RunEventType.RUN_COMPLETED, message=message, duration_ms=9.0
    )
    run.status = RunStatus.COMPLETED
    return run.id


def test_sse_replay_format_and_close(api):
    client, app = api
    run_id = _append_terminal_events(app, app.state.runs)

    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())

    assert "id: 1\nevent: run_started" in text
    assert "id: 2\nevent: step_started" in text
    assert "id: 3\nevent: run_completed" in text
    assert '"seq":3' in text.replace(" ", "")
    assert text.index("id: 1") < text.index("id: 2") < text.index("id: 3")


def test_sse_since_and_last_event_id(api):
    client, app = api
    run_id = _append_terminal_events(app, app.state.runs)

    with client.stream("GET", f"/api/runs/{run_id}/events?since=1") as response:
        text = "".join(response.iter_text())
    assert "id: 1\n" not in text
    assert "id: 2\n" in text and "id: 3\n" in text

    with client.stream(
        "GET", f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "2"}
    ) as response:
        text = "".join(response.iter_text())
    assert "id: 2\n" not in text
    assert "id: 3\n" in text


def test_sse_unknown_run_returns_404_json(api):
    client, _ = api
    response = client.get("/api/runs/nope/events")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "RunNotFoundError"


async def test_sse_keepalive_emitted_before_terminal(monkeypatch):
    monkeypatch.setattr(runs_api, "SSE_KEEPALIVE_S", 0.05)
    app = create_app()
    async with app.router.lifespan_context(app):
        runs: RunManager = app.state.runs
        run = runs.create("ka task")
        runs.append_event(run.id, RunEventType.RUN_STARTED, task=run.task)

        async def finish_later() -> None:
            await asyncio.sleep(0.2)
            runs.append_event(
                run.id, RunEventType.RUN_COMPLETED, duration_ms=1.0
            )

        finisher = asyncio.create_task(finish_later())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            saw_keepalive = False
            async with client.stream("GET", f"/api/runs/{run.id}/events") as response:
                async for line in response.aiter_lines():
                    if line == ": keep-alive":
                        saw_keepalive = True
        await finisher
        assert saw_keepalive


def test_health_endpoint_still_works_after_wiring(api):
    client, _ = api
    assert client.get("/api/health").status_code in {200, 503}
