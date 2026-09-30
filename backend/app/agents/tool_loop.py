"""Bounded tool gather loop (Phase 6 PRD §6.5.2).

Phase A of a tool-enabled agent call: an *unconstrained* conversation (tools
in the request, no `format=` grammar — the live probe proved the two cannot
coexist) in which the model may call tools until it stops or a guard trips.
Grammar-constrained extraction happens afterwards in `Agent` (Phase B).

Guarantees: executions are capped, result bytes are budgeted, an identical
repeat call aborts gathering, an abort ends gathering deterministically, and
no exception escapes into the orchestrator (tool failures come back as
envelopes). Provider/transport errors DO propagate — same policy as Phase 5.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app import logging_config as log
from app.llm.base import (
    ChatMessage,
    ChatRole,
    CompletionResult,
    LLMProvider,
    ToolCall,
    TokenUsage,
)
from app.schemas import ToolResult
from app.tools.base import Tool, ToolContext, error_envelope

logger = logging.getLogger("ai_nexus.agents.tool_loop")

__all__ = [
    "FINAL_ANSWER_NUDGE",
    "STRUCTURED_NUDGE",
    "ToolLoopOutcome",
    "gather_with_tools",
]

FINAL_ANSWER_NUDGE = (
    "You have gathered enough information. Produce your final answer now, "
    "without calling any tools."
)
STRUCTURED_NUDGE = "Produce the required structured output now."

# aborted-reason (ToolLoopOutcome.aborted) -> envelope error code
_ABORT_CODES = {
    "step_limit": "step_limit",
    "result_budget": "result_budget_exhausted",
    "repeated_call": "repeated_call",
}
_ABORT_MESSAGES = {
    "step_limit": "Tool step limit reached. Produce your answer now.",
    "result_budget": "Tool result budget reached. Produce your answer now.",
    "repeated_call": "This exact call was just made. Use the previous result instead.",
}


@dataclass(frozen=True, slots=True)
class ToolLoopOutcome:
    """Everything the agent needs after gathering: history + audit trail."""

    messages: list[ChatMessage]   # full history incl. tool turns
    calls: list[ToolCall]         # executed calls, in order
    outcomes: list[ToolResult]    # parallel to calls
    last_content: str             # last assistant prose (plain path)
    last_usage: TokenUsage        # usage of the final provider turn
    steps: int                    # == len(calls)
    aborted: str | None           # None | repeated_call | step_limit | result_budget


def _wire(tc: ToolCall) -> dict:
    """Provider-shaped echo of a tool call for the assistant history turn."""
    function: dict = {"name": tc.name, "arguments": tc.arguments}
    payload: dict = {"function": function}
    if tc.id is not None:
        payload["id"] = tc.id
    return payload


def _canonical(arguments: dict) -> str:
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"))


def _feed(history: list[ChatMessage], tool_name: str, content: str) -> None:
    history.append(
        ChatMessage(role=ChatRole.TOOL, content=content, tool_name=tool_name)
    )


async def gather_with_tools(
    provider: LLMProvider,
    *,
    messages: list[ChatMessage],
    tools: tuple[Tool, ...],
    context: ToolContext,
    model: str | None,
    temperature: float,
    max_tokens: int | None,
    max_steps: int,
    result_budget_chars: int,
) -> ToolLoopOutcome:
    """Run the bounded loop; never raises for tool-level failures."""
    history = list(messages)
    calls: list[ToolCall] = []
    outcomes: list[ToolResult] = []
    consumed = 0
    prev_signature: str | None = None
    aborted: str | None = None
    last_content = ""
    last_usage = TokenUsage()
    by_name = {tool.name: tool for tool in tools}
    specs = [tool.spec() for tool in tools]

    while True:
        completion: CompletionResult = await provider.chat(
            history,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=specs,
            response_format=None,          # NO grammar on tool turns
        )
        last_content = completion.content
        last_usage = completion.usage
        if not completion.tool_calls:
            break

        history.append(
            ChatMessage(
                role=ChatRole.ASSISTANT,
                content=completion.content,
                tool_calls=[_wire(tc) for tc in completion.tool_calls],
            )
        )
        for tc in completion.tool_calls:
            if aborted is not None:
                _feed(
                    history, tc.name,
                    error_envelope(
                        tc.name, _ABORT_CODES[aborted], _ABORT_MESSAGES[aborted]
                    ),
                )
                continue
            if len(calls) >= max_steps:
                aborted = "step_limit"
                _feed(
                    history, tc.name,
                    error_envelope(
                        tc.name, "step_limit",
                        f"Tool step limit reached ({max_steps}). "
                        "Produce your answer now.",
                    ),
                )
                continue
            if consumed >= result_budget_chars:
                aborted = "result_budget"
                _feed(
                    history, tc.name,
                    error_envelope(
                        tc.name, "result_budget_exhausted",
                        "Tool result budget reached. Produce your answer now.",
                    ),
                )
                continue

            signature = f"{tc.name}:{_canonical(tc.arguments)}"
            if signature == prev_signature:
                aborted = "repeated_call"
                _feed(
                    history, tc.name,
                    error_envelope(
                        tc.name, "repeated_call", _ABORT_MESSAGES["repeated_call"]
                    ),
                )
                continue

            tool = by_name.get(tc.name)
            if tool is None:
                content = error_envelope(
                    tc.name, "unknown_tool",
                    "No such tool. Available: "
                    f"{', '.join(sorted(by_name))}.",
                )
                _feed(history, tc.name, content)
                calls.append(tc)
                outcomes.append(
                    ToolResult(name=tc.name, content=content, error="unknown_tool")
                )
                prev_signature = signature
                continue                                   # counts as a step

            tool_result = await tool.call(tc.arguments, context)
            _feed(history, tc.name, tool_result.content)
            calls.append(tc)
            outcomes.append(tool_result)
            consumed += len(tool_result.content)
            prev_signature = signature

        if aborted is not None:
            # DETERMINISTIC: an abort ends gathering; no further provider
            # turns, so a model that keeps emitting calls cannot extend the loop.
            break

    log.tool_loop_end(
        logger,
        agent=context.agent.value,
        steps=len(calls),
        aborted=aborted or "none",
    )
    return ToolLoopOutcome(
        messages=history,
        calls=calls,
        outcomes=outcomes,
        last_content=last_content,
        last_usage=last_usage,
        steps=len(calls),
        aborted=aborted,
    )
