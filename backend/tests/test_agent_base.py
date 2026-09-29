"""Agent.run() contract tests (PRD §6.3): assembly, kwargs, errors, logging."""

import logging

import pytest

from app.agents.base import Agent, AgentConfig, format_user_content
from app.agents.errors import EmptyAgentResponseError
from app.llm.base import ChatRole, ProviderUnavailableError, ToolCall
from app.schemas import AgentRole, MessageType
from tests.fakes import FakeProvider


def _config(**overrides) -> AgentConfig:
    base = dict(
        role=AgentRole.RESEARCHER,
        display_name="Researcher",
        instructions="You investigate evidence and assumptions.",
        temperature=0.2,
        output_type=MessageType.FINDING,
    )
    base.update(overrides)
    return AgentConfig(**base)


# ------------------------------------------------------------ user framing

def test_format_task_only():
    assert format_user_content("Do the thing") == "TASK:\nDo the thing"


def test_format_task_and_context():
    out = format_user_content("Do the thing", "Some findings")
    assert out == "TASK:\nDo the thing\n\nCONTEXT:\nSome findings"


def test_format_blank_context_omitted():
    assert format_user_content("x", "   ") == "TASK:\nx"
    assert format_user_content("x", None) == "TASK:\nx"


# ------------------------------------------------------------ config validation

def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        _config(display_name="  ")
    with pytest.raises(ValueError):
        _config(instructions="   ")
    with pytest.raises(ValueError):
        _config(instructions="x" * 3001)
    with pytest.raises(ValueError):
        _config(temperature=2.5)
    with pytest.raises(ValueError):
        _config(max_tokens=0)
    with pytest.raises(ValueError):
        _config(prompt_version=0)


# ------------------------------------------------------------ run()

async def test_message_assembly_exactly_two_messages():
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result("findings"))
    agent = Agent(_config(), provider)

    await agent.run("What happened?")

    call = provider.chat_calls[0]
    messages = call["messages"]
    assert len(messages) == 2
    assert messages[0].role is ChatRole.SYSTEM
    assert messages[0].content == agent.config.instructions
    assert messages[1].role is ChatRole.USER
    assert messages[1].content == "TASK:\nWhat happened?"


async def test_context_included_in_user_message():
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result("ok"))
    await Agent(_config(), provider).run("task", context="prior findings")
    user = provider.chat_calls[0]["messages"][1]
    assert "CONTEXT:\nprior findings" in user.content


async def test_kwargs_equal_config():
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result("ok"))
    await Agent(_config(model="llama3.2:3b", temperature=0.2, max_tokens=64), provider).run("t")
    kwargs = provider.chat_calls[0]["kwargs"]
    assert kwargs == {"model": "llama3.2:3b", "temperature": 0.2, "max_tokens": 64}


async def test_returns_message_with_role_defaults():
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result("the content"))
    message = await Agent(_config(), provider).run("task")

    assert message.from_agent is AgentRole.RESEARCHER
    assert message.type is MessageType.FINDING
    assert message.to_agent is None
    assert message.content == "the content"
    assert message.claims is None and message.confidence is None
    assert message.tool_calls is None
    assert message.round is None
    assert message.created_at.tzinfo is not None
    assert len(message.id) == 32


async def test_type_and_recipient_overrides():
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result("ok"))
    manager_cfg = _config(
        role=AgentRole.MANAGER, display_name="Manager",
        instructions="plan things", output_type=MessageType.PLAN, temperature=0.0,
    )
    message = await Agent(manager_cfg, provider).run(
        "task", message_type=MessageType.SYNTHESIS, to_agent=AgentRole.MANAGER
    )
    assert message.type is MessageType.SYNTHESIS
    assert message.to_agent is AgentRole.MANAGER


async def test_tool_calls_passthrough_even_with_empty_content():
    provider = FakeProvider()
    result = FakeProvider.make_result("")
    result.tool_calls = [ToolCall(name="web_search", arguments={"query": "x"})]
    provider.queue_result(result)
    message = await Agent(_config(), provider).run("task")
    assert message.tool_calls is not None
    assert message.tool_calls[0].name == "web_search"


async def test_empty_response_raises():
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result("   "))
    with pytest.raises(EmptyAgentResponseError) as exc_info:
        await Agent(_config(), provider).run("task")
    assert exc_info.value.agent_role is AgentRole.RESEARCHER


async def test_llm_error_propagates_unchanged():
    provider = FakeProvider()
    provider.queue_result(ProviderUnavailableError())
    with pytest.raises(ProviderUnavailableError):
        await Agent(_config(), provider).run("task")


def test_empty_task_rejected_before_any_provider_call():
    async def _run():
        provider = FakeProvider()
        with pytest.raises(ValueError):
            await Agent(_config(), provider).run("   ")
        assert provider.chat_calls == []

    import asyncio
    asyncio.run(_run())


# ------------------------------------------------------------ logging events

async def test_one_start_and_one_end_log(caplog):
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result("ok"))
    with caplog.at_level(logging.INFO, logger="ai_nexus.agents.base"):
        await Agent(_config(), provider).run("task", context="ctx")
    starts = [r for r in caplog.records if "agent_run_start" in r.getMessage()]
    ends = [r for r in caplog.records if "agent_run_end" in r.getMessage()]
    errors = [r for r in caplog.records if "agent_run_error" in r.getMessage()]
    assert len(starts) == 1 and len(ends) == 1 and not errors
    assert "agent=researcher" in starts[0].getMessage()
    assert "prompt_version=1" in starts[0].getMessage()
    assert "message_type=finding" in ends[0].getMessage()
    assert "completion_tokens=5" in ends[0].getMessage()


async def test_error_log_on_empty_response(caplog):
    provider = FakeProvider()
    provider.queue_result(FakeProvider.make_result(""))
    with caplog.at_level(logging.INFO, logger="ai_nexus.agents.base"):
        with pytest.raises(EmptyAgentResponseError):
            await Agent(_config(), provider).run("task")
    assert any("agent_run_error" in r.getMessage() for r in caplog.records)


async def test_llm_error_emits_error_event_not_end(caplog):
    provider = FakeProvider()
    provider.queue_result(ProviderUnavailableError())
    with caplog.at_level(logging.INFO, logger="ai_nexus.agents.base"):
        with pytest.raises(ProviderUnavailableError):
            await Agent(_config(), provider).run("task")
    messages = [r.getMessage() for r in caplog.records]
    assert any("agent_run_error" in m for m in messages)
    assert not any("agent_run_end" in m for m in messages)
