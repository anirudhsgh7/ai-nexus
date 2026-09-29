"""Structured run pipeline (PRD §6.5): kwargs, parsing, retry-once, errors."""

import json
import logging

import pytest

from app.agents.base import Agent, AgentConfig
from app.agents.errors import StructuredOutputError
from app.agents.structured import (
    CLAIMS_SCHEMA,
    DEFAULT_STRUCTURED_MAX_TOKENS,
    VERDICTS_SCHEMA,
    OutputKind,
    directive_for,
)
from app.llm.base import ProviderUnavailableError, ToolCall
from app.schemas import AgentRole, Claim, ClaimStatus, ClaimVerdict, MessageType
from tests.fakes import FakeProvider


def _claims_config(**overrides) -> AgentConfig:
    base = dict(
        role=AgentRole.RESEARCHER,
        display_name="Researcher",
        instructions="You investigate evidence and assumptions.",
        temperature=0.2,
        output_type=MessageType.FINDING,
        output_kind=OutputKind.CLAIMS,
    )
    base.update(overrides)
    return AgentConfig(**base)


def _verdicts_config(**overrides) -> AgentConfig:
    base = dict(
        role=AgentRole.SKEPTIC,
        display_name="Skeptic",
        instructions="You challenge claims.",
        temperature=0.0,
        output_type=MessageType.CRITIQUE,
        output_kind=OutputKind.VERDICTS,
    )
    base.update(overrides)
    return AgentConfig(**base)


def _claims_json(**overrides) -> str:
    payload = {
        "content": "Prose findings here.",
        "claims": [
            {"statement": "X grew 40%.", "status": "unverified", "confidence": 0.5,
             "evidence": []},
            {"statement": "Assumed baseline.", "status": "assumption",
             "evidence": []},
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _verdicts_json(claim_ids=("c1",), verdict="unverifiable") -> str:
    return json.dumps({
        "content": "Checked.",
        "verdicts": [
            {"claim_id": cid, "verdict": verdict, "objection": "no source",
             "evidence": []}
            for cid in claim_ids
        ],
    })


async def _run(config: AgentConfig, provider: FakeProvider, **kwargs):
    return await Agent(config, provider).run(**kwargs)


# ------------------------------------------------------------ kwargs / framing

async def test_structured_kwargs_and_system_prompt():
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result(_claims_json()))
    agent = Agent(_claims_config(), provider)
    await agent.run("What happened?")

    call = provider.chat_calls[0]
    kwargs = call["kwargs"]
    assert kwargs["response_format"] is CLAIMS_SCHEMA
    assert kwargs["max_tokens"] == DEFAULT_STRUCTURED_MAX_TOKENS
    assert kwargs["temperature"] == 0.2  # role temperature preserved
    assert kwargs["model"] is None

    system = call["messages"][0].content
    assert system.startswith("You investigate evidence and assumptions.")
    assert directive_for(OutputKind.CLAIMS) in system

    user = call["messages"][1].content
    assert user == "TASK:\nWhat happened?"


async def test_config_max_tokens_overrides_cap():
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result(_claims_json()))
    await _run(_claims_config(max_tokens=512), provider, task="t")
    assert provider.chat_calls[0]["kwargs"]["max_tokens"] == 512


async def test_plain_path_regression_no_response_format():
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result("plain prose"))
    config = AgentConfig(
        role=AgentRole.RESEARCHER, display_name="R", instructions="i",
        temperature=0.2, output_type=MessageType.FINDING,
    )
    message = await _run(config, provider, task="t")
    kwargs = provider.chat_calls[0]["kwargs"]
    assert "response_format" not in kwargs
    assert kwargs == {"model": None, "temperature": 0.2, "max_tokens": None}
    assert message.content == "plain prose"
    assert message.claims is None and message.verdicts is None


# ------------------------------------------------------------ population

async def test_claims_populated_with_ids():
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result(_claims_json()))
    message = await _run(_claims_config(), provider, task="t")

    assert [c.id for c in message.claims] == ["c1", "c2"]
    assert message.claims[0].status is ClaimStatus.UNVERIFIED
    assert message.claims[0].confidence == 0.5
    assert message.verdicts is None
    assert message.confidence is None  # per-claim confidence is authoritative
    assert message.content == "Prose findings here."


async def test_verdicts_populated_and_claims_rendered():
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result(_verdicts_json()))
    input_claims = [Claim(id="c1", statement="X grew 40%.", status=ClaimStatus.UNVERIFIED)]
    message = await _run(
        _verdicts_config(), provider, task="Evaluate.", claims=input_claims
    )

    assert message.verdicts is not None
    assert message.verdicts[0].claim_id == "c1"
    assert message.verdicts[0].verdict is ClaimVerdict.UNVERIFIABLE
    assert message.claims is None

    user = provider.chat_calls[0]["messages"][1].content
    assert "CLAIMS TO EVALUATE:\n[c1] status=unverified confidence=none" in user
    assert "Claim: X grew 40%." in user


async def test_verdicts_without_input_claims_fails_before_provider():
    provider = FakeProvider()
    with pytest.raises(ValueError, match="requires claims"):
        await _run(_verdicts_config(), provider, task="t")
    assert provider.chat_calls == []


# ------------------------------------------------------------ retry-once

async def test_retry_then_success(caplog):
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result("this is not json at all"))
    provider.queue_result(FakeProvider.make_result(_claims_json()))

    with caplog.at_level(logging.INFO, logger="ai_nexus.agents.base"):
        message = await _run(_claims_config(), provider, task="t")

    assert len(provider.chat_calls) == 2
    assert [c.id for c in message.claims] == ["c1", "c2"]

    retry_msgs = provider.chat_calls[1]["messages"]
    assert retry_msgs[-2].role.value == "assistant"  # failed output echoed
    assert retry_msgs[-1].role.value == "user"
    assert "rejected:" in retry_msgs[-1].content
    assert "No markdown, no commentary." in retry_msgs[-1].content

    events = [r.getMessage() for r in caplog.records]
    assert any("agent_structured_retry" in e for e in events)
    end = next(e for e in events if "agent_run_end" in e)
    assert "retries=1" in end


async def test_unknown_claim_id_violation_triggers_retry():
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result(_verdicts_json(("c9",))))
    provider.queue_result(FakeProvider.make_result(_verdicts_json(("c1",))))
    input_claims = [Claim(id="c1", statement="X", status=ClaimStatus.UNVERIFIED)]

    message = await _run(_verdicts_config(), provider, task="t", claims=input_claims)
    assert len(provider.chat_calls) == 2
    correction = provider.chat_calls[1]["messages"][-1].content
    assert "verdict for unknown claim c9" in correction
    assert message.verdicts[0].claim_id == "c1"


async def test_missing_verdict_violation_triggers_retry():
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result(_verdicts_json(("c1",))))
    provider.queue_result(FakeProvider.make_result(_verdicts_json(("c1", "c2"))))
    input_claims = [
        Claim(id="c1", statement="X", status=ClaimStatus.UNVERIFIED),
        Claim(id="c2", statement="Y", status=ClaimStatus.UNVERIFIED),
    ]
    await _run(_verdicts_config(), provider, task="t", claims=input_claims)
    assert len(provider.chat_calls) == 2
    assert "claim c2 was not evaluated" in provider.chat_calls[1]["messages"][-1].content


async def test_fact_without_evidence_triggers_retry():
    bad = json.dumps({"content": "x", "claims": [
        {"statement": "S", "status": "fact", "evidence": []}]})
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result(bad))
    provider.queue_result(FakeProvider.make_result(_claims_json()))
    message = await _run(_claims_config(), provider, task="t")
    assert len(provider.chat_calls) == 2
    assert "status=fact but no evidence" in provider.chat_calls[1]["messages"][-1].content
    assert message.claims is not None


async def test_double_failure_raises_structured_error(caplog):
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result("garbage one"))
    provider.queue_result(FakeProvider.make_result("garbage two"))

    with caplog.at_level(logging.INFO, logger="ai_nexus.agents.base"):
        with pytest.raises(StructuredOutputError) as exc_info:
            await _run(_claims_config(), provider, task="t")

    error = exc_info.value
    assert error.attempts == 2
    assert "garbage two" in error.raw_snippet
    assert error.last_error
    assert "ollama" not in error.hint  # hint is about max_tokens/model, not transport

    events = [r.getMessage() for r in caplog.records]
    assert any("agent_run_error" in e for e in events)
    assert not any("agent_run_end" in e for e in events)
    assert len(provider.chat_calls) == 2  # bounded: exactly one retry


async def test_fenced_json_first_attempt_no_retry():
    fenced = "```json\n" + _claims_json() + "\n```"
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result(fenced))
    message = await _run(_claims_config(), provider, task="t")
    assert len(provider.chat_calls) == 1
    assert [c.id for c in message.claims] == ["c1", "c2"]


# ------------------------------------------------------------ empty + transport

async def test_structured_empty_output_raises_empty_error():
    empty = json.dumps({"content": "", "claims": []})
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result(empty))
    from app.agents.errors import EmptyAgentResponseError

    with pytest.raises(EmptyAgentResponseError):
        await _run(_claims_config(), provider, task="t")


async def test_llm_error_propagates_without_retry(caplog):
    provider = FakeProvider()
    provider.queue_result(ProviderUnavailableError())

    with caplog.at_level(logging.INFO, logger="ai_nexus.agents.base"):
        with pytest.raises(ProviderUnavailableError):
            await _run(_claims_config(), provider, task="t")

    assert len(provider.chat_calls) == 1  # transport errors are not retried
    events = [r.getMessage() for r in caplog.records]
    assert any("agent_run_error" in e for e in events)
    assert not any("agent_structured_retry" in e for e in events)


async def test_tool_calls_not_transported_in_structured_messages():
    # structured mode parses JSON; provider tool_calls (if any) are not envelope fields
    provider = FakeProvider()
    result = FakeProvider.make_result(_claims_json())
    result.tool_calls = [ToolCall(name="x", arguments={})]
    provider.queue_result(result)
    message = await _run(_claims_config(), provider, task="t")
    assert message.tool_calls is None
