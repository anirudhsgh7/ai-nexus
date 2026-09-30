"""Agent tool-loop traces: two-phase generation, guards, envelopes (PRD §6.5).

FakeProvider consumes queued results in exact request order — each test here
documents the full request sequence it scripts (Phase A turns, then Phase B).
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.agents import Agent, AgentConfig
from app.agents.structured import CLAIMS_SCHEMA, OutputKind
from app.agents.tool_loop import FINAL_ANSWER_NUDGE, STRUCTURED_NUDGE
from app.llm.base import ChatRole, ToolCall
from app.schemas import AgentRole, MessageType
from app.tools import Tool, ToolContext
from tests.fakes import FakeProvider

CLAIMS_JSON = json.dumps(
    {
        "content": "found evidence",
        "claims": [
            {
                "statement": "growth was 23% in 2025",
                "status": "fact",
                "evidence": [{"source": "growth_report.txt"}],
            }
        ],
    }
)

EMPTY_CLAIMS_JSON = json.dumps({"content": "", "claims": []})


class _SearchArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1)


class FakeSearchTool(Tool):
    name: ClassVar[str] = "fake_search"
    description: ClassVar[str] = "Search the fake corpus."
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    args_model: ClassVar[type[BaseModel]] = _SearchArgs

    def __init__(self, *, payload_chars: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.executions: list[ToolContext] = []
        self._payload_chars = payload_chars

    async def execute(self, args: BaseModel, context: ToolContext) -> dict[str, Any]:
        assert isinstance(args, _SearchArgs)
        self.executions.append(context)
        return {"matches": ["x" * self._payload_chars]}


def _claims_config() -> AgentConfig:
    return AgentConfig(
        role=AgentRole.RESEARCHER,
        display_name="Researcher",
        instructions="Research carefully.",
        temperature=0.2,
        output_type=MessageType.FINDING,
        output_kind=OutputKind.CLAIMS,
    )


def _plain_config() -> AgentConfig:
    return AgentConfig(
        role=AgentRole.RESEARCHER,
        display_name="Researcher",
        instructions="Research carefully.",
        temperature=0.2,
        output_type=MessageType.FINDING,
        output_kind=OutputKind.PLAIN,
    )


def _result(content: str = "", tool_calls: list[ToolCall] | None = None):
    result = FakeProvider.make_result(content)
    if tool_calls is not None:
        result.tool_calls = tool_calls
    return result


def _call(name: str = "fake_search", **arguments: Any) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


def _tool_messages(messages) -> list:
    return [m for m in messages if m.role is ChatRole.TOOL]


# ------------------------------------------------------- two-phase, no calls


async def test_structured_two_phase_without_tool_calls():
    provider = FakeProvider()
    provider.queue_result(_result("I should look for evidence."))
    provider.queue_result(_result(CLAIMS_JSON))

    message = await Agent(_claims_config(), provider, tools=[FakeSearchTool()]).run(
        "task", run_id="r1"
    )

    assert len(provider.chat_calls) == 2
    phase_a, phase_b = provider.chat_calls

    assert phase_a["kwargs"]["response_format"] is None, "Phase A must be unconstrained"
    specs = phase_a["kwargs"]["tools"]
    assert [s["function"]["name"] for s in specs] == ["fake_search"]
    assert "OUTPUT FORMAT" not in phase_a["messages"][0].content, (
        "Phase A system prompt stays directive-free or tool use collapses"
    )

    assert phase_b["kwargs"]["response_format"] == CLAIMS_SCHEMA
    assert "tools" not in phase_b["kwargs"], "Phase B must not offer tools"
    assert "OUTPUT FORMAT" in phase_b["messages"][0].content
    assert phase_b["messages"][-1].role is ChatRole.USER
    assert phase_b["messages"][-1].content == STRUCTURED_NUDGE

    assert message.content == "found evidence"
    assert message.tool_calls is None and message.tool_results is None


async def test_structured_phase_b_history_preserves_task_turn():
    provider = FakeProvider()
    provider.queue_result(_result("searching"))
    provider.queue_result(_result(CLAIMS_JSON))

    await Agent(_claims_config(), provider, tools=[FakeSearchTool()]).run(
        "task", context="prior plan"
    )

    phase_b_messages = provider.chat_calls[1]["messages"]
    assert phase_b_messages[1].role is ChatRole.USER
    assert "prior plan" in phase_b_messages[1].content


# --------------------------------------------------------------- tool turns


async def test_one_tool_call_full_trace():
    tool = FakeSearchTool()
    provider = FakeProvider()
    provider.queue_result(_result("", tool_calls=[_call(query="growth")]))
    provider.queue_result(_result("The search found a report."))
    provider.queue_result(_result(CLAIMS_JSON))

    message = await Agent(_claims_config(), provider, tools=[tool]).run(
        "task", run_id="run-42"
    )

    assert len(provider.chat_calls) == 3
    assert len(tool.executions) == 1
    assert tool.executions[0].run_id == "run-42"
    assert tool.executions[0].agent is AgentRole.RESEARCHER

    history = provider.chat_calls[2]["messages"]
    assistant = history[2]
    assert assistant.role is ChatRole.ASSISTANT
    assert assistant.tool_calls == [
        {"function": {"name": "fake_search", "arguments": {"query": "growth"}}}
    ]
    tool_msg = history[3]
    assert tool_msg.role is ChatRole.TOOL
    assert tool_msg.tool_name == "fake_search"
    assert json.loads(tool_msg.content)["ok"] is True

    assert message.tool_calls is not None
    assert [c.name for c in message.tool_calls] == ["fake_search"]
    assert message.tool_results is not None
    assert len(message.tool_results) == len(message.tool_calls)
    assert message.tool_results[0].error is None
    assert message.tool_results[0].duration_ms is not None


async def test_unknown_tool_fed_back_and_counted():
    tool = FakeSearchTool()
    provider = FakeProvider()
    provider.queue_result(_result("", tool_calls=[_call(name="ghost")]))
    provider.queue_result(_result("no such tool, moving on"))
    provider.queue_result(_result(CLAIMS_JSON))

    message = await Agent(_claims_config(), provider, tools=[tool]).run("task")

    assert len(provider.chat_calls) == 3
    assert tool.executions == [], "unknown tools never execute"
    assert message.tool_results is not None
    assert message.tool_results[0].error == "unknown_tool"
    tool_msg = _tool_messages(provider.chat_calls[2]["messages"])[-1]
    payload = json.loads(tool_msg.content)
    assert payload["error"] == "unknown_tool"
    assert "fake_search" in payload["message"]


# ------------------------------------------------------------- guard: repeat


async def test_repeated_call_aborts_gathering():
    tool = FakeSearchTool()
    provider = FakeProvider()
    provider.queue_result(_result("", tool_calls=[_call(query="growth")]))
    provider.queue_result(_result("", tool_calls=[_call(query="growth")]))
    provider.queue_result(_result(CLAIMS_JSON))

    message = await Agent(_claims_config(), provider, tools=[tool]).run("task")

    assert len(provider.chat_calls) == 3, "abort ends gathering: no 4th Phase A turn"
    assert len(tool.executions) == 1, "identical call executed exactly once"
    tool_msgs = _tool_messages(provider.chat_calls[2]["messages"])
    assert len(tool_msgs) == 2
    assert json.loads(tool_msgs[1].content)["error"] == "repeated_call"
    assert message.tool_calls is not None and len(message.tool_calls) == 1


# ------------------------------------------------------------- guard: step cap


async def test_step_cap_limits_executions(monkeypatch):
    monkeypatch.setenv("AI_NEXUS_TOOL_MAX_STEPS", "2")
    from app.config import clear_settings_cache

    clear_settings_cache()

    tool = FakeSearchTool()
    provider = FakeProvider()
    provider.queue_result(_result("", tool_calls=[_call(query="a")]))
    provider.queue_result(_result("", tool_calls=[_call(query="b")]))
    provider.queue_result(_result("", tool_calls=[_call(query="c")]))
    provider.queue_result(_result(CLAIMS_JSON))

    message = await Agent(_claims_config(), provider, tools=[tool]).run("task")

    assert len(tool.executions) == 2, "exactly max_steps executions"
    assert len(provider.chat_calls) == 4, "cap turn happens, then Phase B"
    tool_msgs = _tool_messages(provider.chat_calls[3]["messages"])
    assert len(tool_msgs) == 3
    assert json.loads(tool_msgs[2].content)["error"] == "step_limit"
    assert message.tool_results is not None
    assert len(message.tool_results) == 2


# -------------------------------------------------------------- guard: budget


async def test_result_budget_stops_further_execution(monkeypatch):
    monkeypatch.setenv("AI_NEXUS_TOOL_RESULT_MAX_CHARS", "1000")
    monkeypatch.setenv("AI_NEXUS_TOOL_RESULTS_BUDGET_CHARS", "1000")
    from app.config import clear_settings_cache

    clear_settings_cache()

    tool = FakeSearchTool(payload_chars=1500)
    provider = FakeProvider()
    provider.queue_result(_result("", tool_calls=[_call(query="a")]))
    provider.queue_result(_result("", tool_calls=[_call(query="b")]))
    provider.queue_result(_result(CLAIMS_JSON))

    message = await Agent(_claims_config(), provider, tools=[tool]).run("task")

    assert len(tool.executions) == 1, "budget exhausted after first fat result"
    tool_msgs = _tool_messages(provider.chat_calls[2]["messages"])
    assert json.loads(tool_msgs[1].content)["error"] == "result_budget_exhausted"
    assert message.tool_results is not None
    assert len(message.tool_results) == 1


# ------------------------------------------------------------------ plain path


async def test_plain_tool_path_returns_last_content():
    tool = FakeSearchTool()
    provider = FakeProvider()
    provider.queue_result(_result("", tool_calls=[_call(query="growth")]))
    provider.queue_result(_result("final prose answer"))

    message = await Agent(_plain_config(), provider, tools=[tool]).run("task")

    assert len(provider.chat_calls) == 2
    assert message.content == "final prose answer"
    assert message.tool_calls is not None
    assert message.tool_results is not None


async def test_plain_blank_content_triggers_nudge_call():
    tool = FakeSearchTool()
    provider = FakeProvider()
    provider.queue_result(_result("", tool_calls=[_call(query="growth")]))
    provider.queue_result(_result(""))
    provider.queue_result(_result("answered after nudge"))

    message = await Agent(_plain_config(), provider, tools=[tool]).run("task")

    assert len(provider.chat_calls) == 3
    nudge_turn = provider.chat_calls[2]
    assert nudge_turn["messages"][-1].content == FINAL_ANSWER_NUDGE
    assert "tools" not in nudge_turn["kwargs"], "final answer call offers no tools"
    assert message.content == "answered after nudge"


# ------------------------------------------------------- structured emptiness


async def test_structured_tool_activity_is_not_output():
    tool = FakeSearchTool()
    provider = FakeProvider()
    provider.queue_result(_result("", tool_calls=[_call(query="growth")]))
    provider.queue_result(_result(""))
    provider.queue_result(_result(EMPTY_CLAIMS_JSON))

    from app.agents.errors import EmptyAgentResponseError

    with pytest.raises(EmptyAgentResponseError):
        await Agent(_claims_config(), provider, tools=[tool]).run("task")

    assert len(provider.chat_calls) == 3
    assert len(tool.executions) == 1


# ------------------------------------------------------------- legacy guard


async def test_no_tools_legacy_requests_unchanged():
    provider = FakeProvider()
    provider.queue_result(_result(CLAIMS_JSON))

    message = await Agent(_claims_config(), provider).run("task")

    kwargs = provider.chat_calls[0]["kwargs"]
    assert "tools" not in kwargs, "legacy path must not advertise tools"
    assert kwargs["response_format"] == CLAIMS_SCHEMA
    assert message.tool_calls is None and message.tool_results is None
    assert len(provider.chat_calls) == 1


async def test_verdicts_guard_uses_resolved_kind():
    provider = FakeProvider()
    agent = Agent(_claims_config(), provider, tools=[FakeSearchTool()])
    with pytest.raises(ValueError, match="verdicts output requires claims"):
        await agent.run("task", output_kind=OutputKind.VERDICTS)
    assert provider.chat_calls == []


# ------------------------------------------------------------------- logging


async def test_tool_loop_and_agent_log_fields(caplog):
    tool = FakeSearchTool()
    provider = FakeProvider()
    provider.queue_result(_result("", tool_calls=[_call(query="growth")]))
    provider.queue_result(_result("done"))
    provider.queue_result(_result(CLAIMS_JSON))

    with caplog.at_level(logging.INFO):
        await Agent(_claims_config(), provider, tools=[tool]).run("task")

    events = [r.getMessage() for r in caplog.records]
    assert any(
        "tool_loop_end" in e and "agent=researcher" in e and "steps=1" in e
        for e in events
    )
    assert any("tool_call_start" in e and "tool=fake_search" in e for e in events)
    assert any("tool_call_end" in e and "ok=True" in e for e in events)
    start = next(e for e in events if "agent_run_start" in e)
    assert "tools_count=1" in start
    end = next(e for e in events if "agent_run_end" in e)
    assert "tool_steps=1" in end
    assert "tool_call_count=1" in end
