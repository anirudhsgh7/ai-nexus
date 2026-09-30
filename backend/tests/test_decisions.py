"""Decision output contract (Phase 5 PRD §6.1–6.3): schema, parse, normalize, run."""

import json

import pytest
from pydantic import ValidationError

from app.agents.base import Agent, AgentConfig
from app.agents.structured import (
    DECISION_SCHEMA,
    DEFAULT_STRUCTURED_MAX_TOKENS,
    OutputKind,
    directive_for,
    normalize_decision,
    parse_decision,
    schema_for,
)
from app.llm.base import ChatRole
from app.schemas import (
    AgentRole,
    DecisionAction,
    ManagerDecision,
    MessageType,
)
from tests.fakes import FakeProvider


def _manager_config(**overrides) -> AgentConfig:
    base = dict(
        role=AgentRole.MANAGER,
        display_name="Manager",
        instructions="You coordinate the team.",
        temperature=0.0,
        output_type=MessageType.PLAN,
        output_kind=OutputKind.CLAIMS,  # config stays CLAIMS; decision via override
    )
    base.update(overrides)
    return AgentConfig(**base)


def _decision_json(**overrides) -> str:
    payload = {
        "action": "call_agent",
        "target": "researcher",
        "instruction": "Find a published source for the 40% growth claim.",
        "reason": "Claim c2 is the only evidence gap blocking support.",
        "confidence": 0.8,
    }
    payload.update(overrides)
    return json.dumps(payload)


def _finish_json(**overrides) -> str:
    payload = {"action": "finish", "reason": "All claims are supported.", "confidence": 0.9}
    payload.update(overrides)
    return json.dumps(payload)


# ---------------------------------------------------------------- schema/directive

def test_decision_schema_shape():
    assert schema_for(OutputKind.DECISION) is DECISION_SCHEMA
    assert DECISION_SCHEMA["required"] == ["action", "reason", "confidence"]
    assert DECISION_SCHEMA["properties"]["action"]["enum"] == [
        "call_agent", "finish",
    ]
    assert DECISION_SCHEMA["properties"]["target"]["enum"] == ["researcher", "ideator"]
    assert "content" not in DECISION_SCHEMA["properties"]


def test_decision_directive():
    directive = directive_for(OutputKind.DECISION)
    assert directive.startswith("OUTPUT FORMAT (mandatory):")
    assert json.dumps(DECISION_SCHEMA, separators=(",", ":")) in directive
    assert "routing decision only" in directive
    assert 'action="call_agent"' in directive
    assert 'action="finish"' in directive
    assert "prose answer" not in directive  # no content-field sentence


def test_decision_directive_stops_failing_searches():
    """A blocked tool must route to finish, not to a repeated failing search."""
    directive = directive_for(OutputKind.DECISION)
    assert "cannot be resolved with the available tools" in directive
    assert "provider or network errors" in directive


def test_decision_directive_differs_from_claims():
    assert directive_for(OutputKind.DECISION) != directive_for(OutputKind.CLAIMS)


# ---------------------------------------------------------------- parse

def test_parse_call_agent():
    result = parse_decision(_decision_json())
    assert result.content == ""
    assert result.claims is None and result.verdicts is None
    d = result.decision
    assert d.action is DecisionAction.CALL_AGENT
    assert d.target is AgentRole.RESEARCHER
    assert d.instruction.startswith("Find a published source")
    assert d.confidence == 0.8


def test_parse_finish_without_target():
    result = parse_decision(_finish_json())
    d = result.decision
    assert d.action is DecisionAction.FINISH
    assert d.target is None and d.instruction == ""


def test_parse_incoherent_call_agent_raises():
    with pytest.raises(ValidationError):
        parse_decision(_decision_json(target=None))
    with pytest.raises(ValidationError):
        parse_decision(_decision_json(target="skeptic"))
    with pytest.raises(ValidationError):
        parse_decision(_decision_json(instruction="   "))


def test_parse_finish_with_stray_target_tolerated():
    result = parse_decision(_finish_json(target="researcher", instruction="x"))
    assert result.decision.target is AgentRole.RESEARCHER  # normalized later


def test_parse_extra_fields_ignored_and_fenced_recovers():
    raw = _decision_json()
    data = json.loads(raw)
    data["model_invented_field"] = 1
    assert parse_decision(json.dumps(data)).decision is not None
    fenced = "```json\n" + raw + "\n```"
    assert parse_decision(fenced).decision is not None


def test_parse_garbage_raises():
    with pytest.raises((ValueError, ValidationError)):
        parse_decision("not json at all")


def test_parse_blank_reason_raises():
    with pytest.raises(ValidationError):
        parse_decision(_decision_json(reason="   "))


# ---------------------------------------------------------------- normalize

def test_normalize_finish_scrubs_extras():
    d = ManagerDecision(
        action=DecisionAction.FINISH,
        target=AgentRole.RESEARCHER,
        instruction="leftover",
        reason="done",
        confidence=0.5,
    )
    n = normalize_decision(d)
    assert n.target is None and n.instruction == ""
    assert n.action is DecisionAction.FINISH and n.reason == "done"


def test_normalize_passes_call_agent_through():
    d = parse_decision(_decision_json()).decision
    assert normalize_decision(d) is d


# ---------------------------------------------------------------- agent.run override

async def test_decision_through_run_uses_override_not_config_kind():
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result(_finish_json()))
    agent = Agent(_manager_config(), provider)

    message = await agent.run(
        "task",
        context="ROUND 1 OF 3\n\nUNRESOLVED CLAIMS: ...",
        message_type=MessageType.DECISION,
        output_kind=OutputKind.DECISION,
    )

    call = provider.chat_calls[0]
    assert call["kwargs"]["response_format"] is DECISION_SCHEMA
    assert call["kwargs"]["max_tokens"] == DEFAULT_STRUCTURED_MAX_TOKENS
    assert call["kwargs"]["temperature"] == 0.0

    system = call["messages"][0].content
    assert system.startswith("You coordinate the team.")
    assert "routing decision only" in system

    user = call["messages"][1].content
    assert "ROUND 1 OF 3" in user and "UNRESOLVED CLAIMS" in user

    assert message.type is MessageType.DECISION
    assert message.decision is not None
    assert message.decision.action is DecisionAction.FINISH
    assert message.content == ""
    assert message.claims is None and message.verdicts is None
    assert message.tool_calls is None


async def test_config_kind_untouched_by_override():
    """A plain run after a decision run still uses the config kind."""
    provider = FakeProvider()
    provider.queue_result(
        FakeProvider.make_result(
            json.dumps({"content": "Plan prose.", "claims": []})
        )
    )
    agent = Agent(_manager_config(), provider)
    message = await agent.run("task")  # no override -> config CLAIMS
    assert provider.chat_calls[0]["kwargs"]["response_format"] is not DECISION_SCHEMA
    assert message.claims == []
    assert message.decision is None


async def test_invalid_decision_retries_once():
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result(_decision_json(target="skeptic")))
    provider.queue_result(FakeProvider.make_result(_finish_json()))
    agent = Agent(_manager_config(), provider)

    message = await agent.run(
        "task", message_type=MessageType.DECISION, output_kind=OutputKind.DECISION
    )
    assert len(provider.chat_calls) == 2
    assert "target researcher or ideator" in provider.chat_calls[1]["messages"][-1].content
    assert message.decision.action is DecisionAction.FINISH
