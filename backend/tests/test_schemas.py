"""Envelope contract tests (PRD §6.1): round-trip, defaults, validation."""

from datetime import UTC

import pytest
from pydantic import ValidationError

from app.llm.base import ToolCall
from app.schemas import (
    AgentMessage,
    AgentRole,
    Claim,
    ClaimStatus,
    ClaimVerdict,
    DecisionAction,
    Evidence,
    ManagerDecision,
    MessageType,
    Verdict,
)


def _msg(**overrides) -> AgentMessage:
    base = {"from_agent": AgentRole.RESEARCHER, "type": MessageType.FINDING, "content": "x"}
    base.update(overrides)
    return AgentMessage(**base)


# ------------------------------------------------------------------ round-trip

def test_round_trip_json():
    m = _msg()
    again = AgentMessage.model_validate_json(m.model_dump_json())
    assert again == m


def test_round_trip_with_claims_and_tool_calls():
    m = _msg(
        claims=[Claim(id="c1", statement="X grew 40%", confidence=0.55)],
        tool_calls=[ToolCall(name="web_search", arguments={"query": "x"}, id="call_1")],
        confidence=0.7,
        round=2,
        to_agent=AgentRole.SKEPTIC,
    )
    payload = m.model_dump_json()
    again = AgentMessage.model_validate_json(payload)
    assert again == m
    assert again.tool_calls[0].name == "web_search"
    assert again.tool_calls[0].arguments == {"query": "x"}
    assert again.claims[0].statement == "X grew 40%"


def test_wire_field_names():
    import json

    data = json.loads(_msg().model_dump_json())
    assert set(data) == {
        "id", "from_agent", "to_agent", "type", "content", "claims",
        "verdicts", "decision", "confidence", "tool_calls", "round",
        "created_at",
    }
    assert data["from_agent"] == "researcher"
    assert data["type"] == "finding"


# ------------------------------------------------------------------ defaults

def test_defaults():
    m = _msg()
    assert m.to_agent is None
    assert m.claims is None
    assert m.verdicts is None
    assert m.confidence is None
    assert m.tool_calls is None
    assert m.round is None
    assert m.content == "x"
    assert m.created_at.tzinfo is not None
    assert m.created_at.utcoffset() == UTC.utcoffset(None)
    assert len(m.id) == 32  # uuid4().hex


def test_ids_unique():
    assert _msg().id != _msg().id


# ------------------------------------------------------------------ validation

def test_confidence_bounds():
    with pytest.raises(ValidationError):
        _msg(confidence=1.01)
    with pytest.raises(ValidationError):
        _msg(confidence=-0.01)
    _msg(confidence=0.0)
    _msg(confidence=1.0)


def test_round_ge_one():
    with pytest.raises(ValidationError):
        _msg(round=0)
    with pytest.raises(ValidationError):
        _msg(round=-1)


def test_invalid_role_and_type_rejected():
    with pytest.raises(ValidationError):
        _msg(from_agent="verifier")
    with pytest.raises(ValidationError):
        _msg(type="summary")


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        _msg(mystery_field=1)


def test_naive_created_at_rejected():
    from datetime import datetime

    with pytest.raises(ValidationError):
        _msg(created_at=datetime(2026, 1, 1))


def test_frozen_immutability():
    m = _msg()
    with pytest.raises(ValidationError):
        m.content = "changed"  # type: ignore[misc]


def test_claim_validation():
    with pytest.raises(ValidationError):
        Claim(id="", statement="x")
    with pytest.raises(ValidationError):
        Claim(id="c1", statement="")
    with pytest.raises(ValidationError):
        Claim(id="c1", statement="x", confidence=2.0)
    with pytest.raises(ValidationError):
        Claim(id="c1", statement="x", unexpected=True)


# ------------------------------------------------- Phase 3 extensions

def test_claim_phase3_defaults():
    c = Claim(id="c1", statement="X grew 40%")
    assert c.status is ClaimStatus.UNVERIFIED
    assert c.evidence == []
    assert c.confidence is None


def test_claim_with_status_and_evidence_round_trip():
    c = Claim(
        id="c1",
        statement="X grew 40%",
        status=ClaimStatus.FACT,
        confidence=0.8,
        evidence=[Evidence(source="annual report", quote="revenue up 40%")],
    )
    again = Claim.model_validate_json(c.model_dump_json())
    assert again == c
    assert again.status is ClaimStatus.FACT
    assert again.evidence[0].source == "annual report"


def test_evidence_requires_source():
    with pytest.raises(ValidationError):
        Evidence(source="")
    with pytest.raises(ValidationError):
        Evidence(source="   ")
    with pytest.raises(ValidationError):
        Evidence(source="src", extra_field=True)  # extra=forbid
    Evidence(source="src")  # quote optional


def test_claim_status_enum_rejects_unknown():
    with pytest.raises(ValidationError):
        Claim(id="c1", statement="x", status="verified")


def test_verdict_validation():
    v = Verdict(claim_id="c1", verdict=ClaimVerdict.REFUTED, objection="no source")
    assert v.evidence == []
    with pytest.raises(ValidationError):
        Verdict(claim_id="", verdict=ClaimVerdict.REFUTED, objection="o")
    with pytest.raises(ValidationError):
        Verdict(claim_id="c1", verdict="verified", objection="o")
    with pytest.raises(ValidationError):
        Verdict(claim_id="c1", verdict=ClaimVerdict.REFUTED, objection="   ")
    with pytest.raises(ValidationError):
        Verdict(claim_id="c1", verdict=ClaimVerdict.SUPPORTED, objection="o", nope=1)


def test_verdict_round_trip():
    v = Verdict(
        claim_id="c1",
        verdict=ClaimVerdict.UNVERIFIABLE,
        objection="no evidence available",
        evidence=[Evidence(source="search attempted")],
    )
    assert Verdict.model_validate_json(v.model_dump_json()) == v


def test_message_with_verdicts_round_trip():
    m = _msg(
        verdicts=[
            Verdict(claim_id="c1", verdict=ClaimVerdict.SUPPORTED, objection="holds"),
        ],
        claims=[Claim(id="c1", statement="x")],
    )
    again = AgentMessage.model_validate_json(m.model_dump_json())
    assert again == m
    assert again.verdicts[0].verdict is ClaimVerdict.SUPPORTED


# ------------------------------------------------- Phase 5 decision extensions

def test_decision_call_agent_validation():
    d = ManagerDecision(
        action=DecisionAction.CALL_AGENT,
        target=AgentRole.RESEARCHER,
        instruction="Find a source for the 40% claim.",
        reason="Evidence gap blocks support.",
        confidence=0.8,
    )
    assert d.action is DecisionAction.CALL_AGENT
    with pytest.raises(ValidationError):
        ManagerDecision(action=DecisionAction.CALL_AGENT, target=AgentRole.SKEPTIC,
                        instruction="x", reason="r", confidence=0.5)
    with pytest.raises(ValidationError):
        ManagerDecision(action=DecisionAction.CALL_AGENT, target=AgentRole.RESEARCHER,
                        instruction="   ", reason="r", confidence=0.5)
    with pytest.raises(ValidationError):
        ManagerDecision(action=DecisionAction.CALL_AGENT, reason="r", confidence=0.5)


def test_decision_finish_needs_no_target():
    d = ManagerDecision(action=DecisionAction.FINISH, reason="All set.", confidence=0.7)
    assert d.target is None and d.instruction == ""
    with pytest.raises(ValidationError):
        ManagerDecision(action=DecisionAction.FINISH, reason="   ", confidence=0.5)
    with pytest.raises(ValidationError):
        ManagerDecision(action=DecisionAction.FINISH, reason="r", confidence=1.5)


def test_decision_round_trip_on_message():
    decision = ManagerDecision(action=DecisionAction.FINISH, reason="done", confidence=0.9)
    m = _msg(type=MessageType.DECISION, decision=decision, content="")
    again = AgentMessage.model_validate_json(m.model_dump_json())
    assert again == m
    assert again.decision == decision
    assert again.content == ""


def test_message_type_members():
    assert MessageType.DECISION.value == "decision"
    assert MessageType.REVISION.value == "revision"
