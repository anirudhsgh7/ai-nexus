"""AgentConfig and Agent: the single, config-driven agent abstraction.

Four roles differ by instructions/goals/capabilities (declarative config),
never by code. No orchestration lives here — this module only invokes one
agent once and returns a structured message.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ValidationError

from app import logging_config as log
from app.agents.errors import EmptyAgentResponseError, StructuredOutputError
from app.agents.structured import (
    DEFAULT_STRUCTURED_MAX_TOKENS,
    OutputKind,
    StructuredResult,
    correction_message,
    directive_for,
    parse_accountability,
    parse_claims,
    parse_decision,
    parse_verification,
    parse_verdicts,
    schema_for,
)
from app.agents.tool_loop import (
    FINAL_ANSWER_NUDGE,
    STRUCTURED_NUDGE,
    ToolLoopOutcome,
    gather_with_tools,
)
from app.audits import validate_verification
from app.claims import render_claims, validate_claims, validate_verdicts
from app.config import get_settings
from app.llm.base import ChatMessage, ChatRole, LLMError, LLMProvider, TokenUsage
from app.schemas import AgentMessage, AgentRole, Claim, MessageType
from app.tools import Tool, ToolContext

logger = logging.getLogger("ai_nexus.agents.base")

MAX_INSTRUCTION_CHARS = 3000
_RAW_TRUNCATION = 4000
_SNIPPET_TRUNCATION = 300
_RAW_ERROR_CHARS = 200

__all__ = [
    "Agent",
    "AgentConfig",
    "MAX_INSTRUCTION_CHARS",
    "format_user_content",
]


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Static role definition. Validated at import time, not request time."""

    role: AgentRole
    display_name: str
    instructions: str
    temperature: float
    output_type: MessageType
    output_kind: OutputKind = OutputKind.PLAIN
    capabilities: frozenset[str] = field(default_factory=frozenset)
    model: str | None = None
    max_tokens: int | None = None
    prompt_version: int = 1

    def __post_init__(self) -> None:
        if not self.display_name.strip():
            raise ValueError("display_name must be non-empty")
        instructions = self.instructions.strip()
        if not instructions:
            raise ValueError(f"instructions for {self.role.value} must be non-empty")
        if len(instructions) > MAX_INSTRUCTION_CHARS:
            raise ValueError(
                f"instructions for {self.role.value} exceed {MAX_INSTRUCTION_CHARS} chars "
                f"({len(instructions)})"
            )
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0.0, 2.0], got {self.temperature}")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.prompt_version < 1:
            raise ValueError("prompt_version must be >= 1")


def format_user_content(
    task: str,
    context: str | None = None,
    claims: Sequence[Claim] | None = None,
) -> str:
    """The single tested framing point for user content (PRD §6.3/§6.4).

    Blocks in fixed order TASK -> CONTEXT -> CLAIMS TO EVALUATE, joined by a
    single blank line; empty/blank inputs are omitted.
    """
    blocks = [f"TASK:\n{task}"]
    if context is not None and context.strip():
        blocks.append(f"CONTEXT:\n{context}")
    if claims:
        rendered = render_claims(claims)
        if rendered:
            blocks.append(f"CLAIMS TO EVALUATE:\n{rendered}")
    return "\n\n".join(blocks)


class Agent:
    """One agent: config + provider (+ tools, Phase 6).

    Stateless between run() calls. With no tools resolved, every code path is
    byte-identical to Phase 5; with tools, a bounded gather loop (PRD §6.5.2)
    precedes the structured/plain answer.
    """

    def __init__(
        self, config: AgentConfig, provider: LLMProvider,
        tools: Sequence[Tool] = (),
    ) -> None:
        self._config = config
        self._provider = provider
        self._tools = tuple(sorted(tools, key=lambda tool: tool.name))
        self._tool_specs = [tool.spec() for tool in self._tools]

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def role(self) -> AgentRole:
        return self._config.role

    @property
    def display_name(self) -> str:
        return self._config.display_name

    async def run(
        self,
        task: str,
        *,
        context: str | None = None,
        claims: Sequence[Claim] | None = None,
        message_type: MessageType | None = None,
        to_agent: AgentRole | None = None,
        output_kind: OutputKind | None = None,
        run_id: str | None = None,
    ) -> AgentMessage:
        if not task.strip():
            raise ValueError("task must be non-empty")
        cfg = self._config
        kind = output_kind if output_kind is not None else cfg.output_kind
        resolved_type = message_type or cfg.output_type
        input_claims = list(claims) if claims else []

        if kind is OutputKind.VERDICTS and not input_claims:
            raise ValueError("verdicts output requires claims to evaluate")

        tools_enabled = bool(self._tools)
        directive = "" if kind is OutputKind.PLAIN else directive_for(kind)

        # Legacy structured calls carry the directive in the system prompt up
        # front. Tool-enabled calls keep Phase A prompt clean (the probe showed
        # a format-hungry prompt suppresses tool use) and add the directive at
        # Phase B instead.
        if directive and not tools_enabled:
            system_prompt = f"{cfg.instructions}\n\n{directive}"
        else:
            system_prompt = cfg.instructions
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=system_prompt),
            ChatMessage(
                role=ChatRole.USER,
                content=format_user_content(task, context, input_claims or None),
            ),
        ]

        if cfg.max_tokens is not None:
            effective_max_tokens = cfg.max_tokens
        elif kind is OutputKind.PLAIN:
            effective_max_tokens = None
        else:
            effective_max_tokens = DEFAULT_STRUCTURED_MAX_TOKENS

        log.agent_run_start(
            logger,
            agent=cfg.role.value,
            prompt_version=cfg.prompt_version,
            model=cfg.model or "default",
            temperature=cfg.temperature,
            task_chars=len(task),
            context_chars=len(context) if context else 0,
            output_kind=kind.value,
            input_claims_count=len(input_claims),
            tools_count=len(self._tools),
        )
        started = time.monotonic()

        outcome: ToolLoopOutcome | None = None
        parsed: StructuredResult | None = None
        usage = TokenUsage()
        retries = 0
        plain_content = ""
        plain_tool_calls = None

        try:
            if tools_enabled:
                settings = get_settings()
                outcome = await gather_with_tools(
                    self._provider,
                    messages=messages,
                    tools=self._tools,
                    context=ToolContext(agent=cfg.role, run_id=run_id),
                    model=cfg.model,
                    temperature=cfg.temperature,
                    max_tokens=effective_max_tokens,
                    max_steps=settings.tool_max_steps,
                    result_budget_chars=settings.tool_results_budget_chars,
                )
                if kind is OutputKind.PLAIN:
                    plain_content = outcome.last_content
                    usage = outcome.last_usage
                    if not plain_content.strip():
                        # one final no-tools turn so plain agents can still answer
                        final = await self._provider.chat(
                            [
                                *outcome.messages,
                                ChatMessage(
                                    role=ChatRole.USER,
                                    content=FINAL_ANSWER_NUDGE,
                                ),
                            ],
                            model=cfg.model,
                            temperature=cfg.temperature,
                            max_tokens=effective_max_tokens,
                        )
                        plain_content = final.content
                        usage = final.usage
                else:
                    phase_b_messages = [
                        ChatMessage(
                            role=ChatRole.SYSTEM,
                            content=f"{cfg.instructions}\n\n{directive}",
                        ),
                        *outcome.messages[1:],
                        ChatMessage(role=ChatRole.USER, content=STRUCTURED_NUDGE),
                    ]
                    result, parsed, retries = await self._structured_attempts(
                        phase_b_messages, kind, input_claims,
                        effective_max_tokens,
                        tool_calls_executed=len(outcome.calls),
                        tools_available=True,
                    )
                    usage = result.usage
            elif kind is OutputKind.PLAIN:
                result, parsed, retries = await self._plain_call(
                    messages, effective_max_tokens
                )
                usage = result.usage
                plain_content = result.content
                plain_tool_calls = result.tool_calls or None
            else:
                result, parsed, retries = await self._structured_attempts(
                    messages, kind, input_claims, effective_max_tokens
                )
                usage = result.usage
        except LLMError as exc:
            log.agent_run_error(
                logger, agent=cfg.role.value,
                error_type=type(exc).__name__, message=str(exc),
            )
            raise
        except StructuredOutputError as exc:
            log.agent_run_error(
                logger, agent=cfg.role.value,
                error_type="StructuredOutputError", message=exc.last_error,
            )
            raise
        content = parsed.content if parsed is not None else plain_content
        out_claims = parsed.claims if parsed is not None else None
        out_verdicts = parsed.verdicts if parsed is not None else None
        out_decision = parsed.decision if parsed is not None else None
        out_verification = parsed.verification if parsed is not None else None
        out_accountability = parsed.accountability if parsed is not None else None
        if outcome is not None:
            tool_calls = outcome.calls or None
            tool_results = outcome.outcomes or None
            tool_steps = outcome.steps
        else:
            tool_calls = plain_tool_calls if kind is OutputKind.PLAIN else None
            tool_results = None
            tool_steps = 0

        # Empty rule: for PLAIN, tool activity counts as output (a tool-using
        # agent that produced only calls still delivered something). For
        # structured kinds only structured fields count — activity is not output.
        # A report (even with an empty claim list) is structured output.
        if kind is OutputKind.PLAIN:
            has_output = bool(content.strip()) or bool(tool_calls)
        else:
            has_output = (
                bool(content.strip())
                or bool(out_claims)
                or bool(out_verdicts)
                or out_decision is not None
                or out_verification is not None
                or out_accountability is not None
            )
        if not has_output:
            log.agent_run_error(
                logger,
                agent=cfg.role.value,
                error_type="EmptyAgentResponseError",
                message="no content, no tool calls",
            )
            raise EmptyAgentResponseError(cfg.role)

        message = AgentMessage(
            id=uuid4().hex,
            from_agent=cfg.role,
            to_agent=to_agent,
            type=resolved_type,
            content=content,
            claims=out_claims,
            verdicts=out_verdicts,
            decision=out_decision,
            verification=out_verification,
            accountability=out_accountability,
            confidence=None,
            tool_calls=tool_calls,
            tool_results=tool_results,
            retries=retries,
            round=None,
            created_at=datetime.now(UTC),
        )
        log.agent_run_end(
            logger,
            agent=cfg.role.value,
            message_id=message.id,
            message_type=message.type.value,
            content_chars=len(content),
            tool_call_count=len(tool_calls) if tool_calls else 0,
            tool_steps=tool_steps,
            claims_count=len(out_claims) if out_claims is not None else 0,
            verdicts_count=len(out_verdicts) if out_verdicts is not None else 0,
            retries=retries,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            wall_ms=round((time.monotonic() - started) * 1000, 1),
        )
        return message

    async def _plain_call(
        self, messages: list[ChatMessage], max_tokens: int | None
    ) -> tuple:
        """Phase 2 behavior, byte-identical kwargs (no response_format)."""
        cfg = self._config
        result = await self._provider.chat(
            messages,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=max_tokens,
        )
        return result, None, 0

    async def _structured_attempts(
        self,
        messages: list[ChatMessage],
        kind: OutputKind,
        input_claims: list[Claim],
        max_tokens: int | None,
        *,
        tool_calls_executed: int = 0,
        tools_available: bool = False,
    ) -> tuple:
        """One grammar-constrained call + at most one corrected retry (PRD §6.5.2).

        `tool_calls_executed`/`tools_available` feed the Verifier's independent-
        check rule (Phase 11 §6.5.2); other kinds ignore them.
        """
        cfg = self._config
        schema = schema_for(kind)
        current = list(messages)

        for attempt in (1, 2):
            result = await self._provider.chat(
                current,
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=max_tokens,
                response_format=schema,
            )
            problems: list[str] = []
            parsed: StructuredResult | None = None
            try:
                if kind is OutputKind.CLAIMS:
                    parsed = parse_claims(result.content)
                    problems = validate_claims(parsed.claims or [])
                elif kind is OutputKind.VERDICTS:
                    parsed = parse_verdicts(result.content)
                    problems = validate_verdicts(input_claims, parsed.verdicts or [])
                elif kind is OutputKind.VERIFICATION:
                    parsed = parse_verification(result.content)
                    problems = validate_verification(
                        parsed.verification,
                        input_claims,
                        tool_calls_executed,
                        tools_available,
                    )
                elif kind is OutputKind.ACCOUNTABILITY:
                    # pydantic validators run inside the parser; the mechanical
                    # facts are enforced by the orchestrator (PRD §6.5.4)
                    parsed = parse_accountability(result.content)
                    problems = []
                else:  # DECISION: coherence enforced by ManagerDecision validators
                    parsed = parse_decision(result.content)
                    problems = []
            except (ValueError, ValidationError) as exc:
                parsed = None
                problems = [f"{type(exc).__name__}: {str(exc)[:_RAW_ERROR_CHARS]}"]

            if parsed is not None and not problems:
                return result, parsed, attempt - 1

            if attempt == 1:
                log.agent_structured_retry(
                    logger,
                    agent=cfg.role.value,
                    attempt=2,
                    error_count=len(problems),
                )
                current = [
                    *current,
                    ChatMessage(
                        role=ChatRole.ASSISTANT,
                        content=result.content[:_RAW_TRUNCATION],
                    ),
                    ChatMessage(role=ChatRole.USER, content=correction_message(problems)),
                ]
                continue

            raise StructuredOutputError(
                cfg.role,
                attempts=2,
                last_error="; ".join(problems),
                raw_snippet=result.content[:_SNIPPET_TRUNCATION],
            )
        raise AssertionError("unreachable: structured attempt loop exhausted")
