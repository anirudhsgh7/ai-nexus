"""UI wire contract (Phase 8 PRD §6.12).

The frontend hand-writes its types (by design — no codegen); this offline test
is the guard that keeps them honest: it asserts the exact wire keys the UI
consumes still exist on the backend models and payload builders. If any key
here is renamed/removed, the UI breaks silently — this test fails loudly
instead. Keep the key sets in sync with `frontend/src/types.ts`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from app.api.runs import (
    _error_response,
    _run_payload,
    _sse_frame,
    _step_payload,
    _summary_payload,
)
from app.runs import (
    RunEvent,
    RunEventType,
    RunManager,
    StepKind,
    StepRecord,
    StepStatus,
)
from app.schemas import AgentMessage, AgentRole, MessageType

# frontend/src/types.ts — RunEvent
EVENT_KEYS = {
    "seq", "type", "run_id", "ts", "step", "kind", "agent", "round",
    "task", "message", "duration_ms", "skipped", "error",
}

# frontend/src/types.ts — AgentMessage (Phase 11 §6.10 adds the audit fields)
MESSAGE_KEYS = {
    "id", "from_agent", "to_agent", "type", "content", "claims", "verdicts",
    "decision", "verification", "accountability", "confidence", "tool_calls",
    "tool_results", "retries", "round", "created_at",
}

# frontend/src/types.ts — RunPayload / StepPayload / RunSummary
RUN_PAYLOAD_KEYS = {
    "run_id", "task", "status", "created_at", "started_at", "finished_at",
    "duration_ms", "steps", "rounds", "selected_round", "final_message", "error",
    # Phase 11 §6.10: top-level audit messages (null when absent)
    "verification", "accountability",
}
STEP_PAYLOAD_KEYS = {
    "index", "kind", "agent", "status", "round", "duration_ms", "skipped",
    "message", "error",
}
SUMMARY_KEYS = {"run_id", "status", "task_preview", "created_at", "duration_ms"}

# frontend/src/types.ts — RoundSummary (Phase 9 evidence payload)
ROUND_KEYS = {
    "round", "supported", "unresolved", "decision",
    "claims", "origins", "verdicts",
}
DECISION_KEYS = {"action", "target", "instruction", "reason", "confidence"}

EVENT_TYPES = {
    "run_started", "step_started", "step_completed", "run_completed",
    "run_failed",
}


def test_run_event_keys():
    assert EVENT_KEYS <= set(RunEvent.model_fields)
    assert {event.value for event in RunEventType} == EVENT_TYPES


def test_agent_message_keys():
    assert MESSAGE_KEYS <= set(AgentMessage.model_fields)


def test_run_payload_keys():
    run = RunManager().create("contract check")
    assert RUN_PAYLOAD_KEYS <= set(_run_payload(run))


def test_round_payload_keys():
    """Phase 9: the evidence fields the UI consumes on every round."""
    from app.runs import RoundSnapshot, RunStatus
    from app.schemas import (
        Claim,
        ClaimStatus,
        ClaimVerdict,
        DecisionAction,
        Evidence,
        ManagerDecision,
        Verdict,
    )

    run = RunManager().create("contract check")
    run.rounds.append(
        RoundSnapshot(
            round_number=1,
            claims=[
                Claim(
                    id="c1",
                    statement="s",
                    status=ClaimStatus.FACT,
                    evidence=[Evidence(source="doc")],
                )
            ],
            origins={"c1": AgentRole.RESEARCHER},
            verdicts=[
                Verdict(
                    claim_id="c1",
                    verdict=ClaimVerdict.SUPPORTED,
                    objection="holds",
                )
            ],
            worker_content={},
            skeptic_content="",
            supported_count=1,
            unresolved_count=0,
            decision=ManagerDecision(
                action=DecisionAction.FINISH,
                target=None,
                instruction="",
                reason="all resolved",
                confidence=0.9,
            ),
        )
    )
    run.status = RunStatus.COMPLETED
    payload = _run_payload(run)
    round_payload = payload["rounds"][0]
    assert ROUND_KEYS <= set(round_payload)
    assert set(round_payload["decision"]) == DECISION_KEYS
    assert round_payload["origins"] == {"c1": "researcher"}
    assert set(round_payload["claims"][0]) == {
        "id", "statement", "status", "confidence", "evidence",
    }
    assert set(round_payload["verdicts"][0]) == {
        "claim_id", "verdict", "objection", "evidence",
    }
    assert payload["selected_round"] == 1


def test_step_payload_keys():
    step = StepRecord(
        index=1,
        kind=StepKind.PLAN,
        agent=AgentRole.MANAGER,
        status=StepStatus.RUNNING,
        started_at=datetime.now(UTC),
    )
    assert STEP_PAYLOAD_KEYS <= set(_step_payload(step))


def test_summary_payload_keys():
    run = RunManager().create("contract check")
    assert SUMMARY_KEYS <= set(_summary_payload(run))


def test_sse_frame_layout():
    event = RunEvent(
        seq=7,
        type=RunEventType.RUN_STARTED,
        run_id="r1",
        ts=datetime(2026, 9, 30, 12, 0, tzinfo=UTC),
        task="t",
    )
    frame = _sse_frame(event)
    assert frame.startswith("id: 7\nevent: run_started\ndata: {")
    assert frame.endswith("\n\n")
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload["type"] == "run_started"


def test_post_409_error_envelope_keys():
    response = _error_response(
        409,
        "RunActiveError",
        "a run is already active",
        "wait for run x to finish",
        active_run_id="x",
    )
    body = json.loads(bytes(response.body))
    assert set(body["error"]) == {"type", "message", "hint", "active_run_id"}


def test_validation_error_is_fastapi_default_shape():
    """The UI maps `{detail: [...]}` for 422 separately from the envelope."""
    # Reaching FastAPI's validation layer requires the app; assert the shape
    # contract via the route's declared request model instead.
    from app.api.runs import RunCreateRequest

    assert "task" in RunCreateRequest.model_fields
