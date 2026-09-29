"""AgentConfig and Agent: the single, config-driven agent abstraction.

Four roles differ by instructions/goals/capabilities (declarative config),
never by code. No orchestration lives here — this module only invokes one
agent once and returns a structured message.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from app import logging_config as log
from app.agents.errors import EmptyAgentResponseError
from app.llm.base import ChatMessage, ChatRole, LLMError, LLMProvider
from app.schemas import AgentMessage, AgentRole, MessageType

logger = logging.getLogger("ai_nexus.agents.base")

MAX_INSTRUCTION_CHARS = 3000


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Static role definition. Validated at import time, not request time."""

    role: AgentRole
    display_name: str
    instructions: str
    temperature: float
    output_type: MessageType
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


def format_user_content(task: str, context: str | None = None) -> str:
    """The single tested framing point for user content (PRD §6.3)."""
    content = f"TASK:\n{task}"
    if context is not None and context.strip():
        content += f"\n\nCONTEXT:\n{context}"
    return content


class Agent:
    """One agent: config + provider. Stateless between run() calls."""

    def __init__(self, config: AgentConfig, provider: LLMProvider) -> None:
        self._config = config
        self._provider = provider

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
        message_type: MessageType | None = None,
        to_agent: AgentRole | None = None,
    ) -> AgentMessage:
        if not task.strip():
            raise ValueError("task must be non-empty")
        cfg = self._config
        resolved_type = message_type or cfg.output_type

        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=cfg.instructions),
            ChatMessage(role=ChatRole.USER, content=format_user_content(task, context)),
        ]

        log.agent_run_start(
            logger,
            agent=cfg.role.value,
            prompt_version=cfg.prompt_version,
            model=cfg.model or "default",
            temperature=cfg.temperature,
            task_chars=len(task),
            context_chars=len(context) if context else 0,
        )
        started = time.monotonic()
        try:
            result = await self._provider.chat(
                messages,
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
        except LLMError as exc:
            log.agent_run_error(
                logger, agent=cfg.role.value,
                error_type=type(exc).__name__, message=str(exc),
            )
            raise

        if not result.content.strip() and not result.tool_calls:
            log.agent_run_error(
                logger, agent=cfg.role.value,
                error_type="EmptyAgentResponseError", message="no content, no tool calls",
            )
            raise EmptyAgentResponseError(cfg.role)

        message = AgentMessage(
            id=uuid4().hex,
            from_agent=cfg.role,
            to_agent=to_agent,
            type=resolved_type,
            content=result.content,
            claims=None,
            confidence=None,
            tool_calls=result.tool_calls or None,
            round=None,
            created_at=datetime.now(UTC),
        )
        log.agent_run_end(
            logger,
            agent=cfg.role.value,
            message_id=message.id,
            message_type=message.type.value,
            content_chars=len(result.content),
            tool_call_count=len(result.tool_calls),
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            wall_ms=round((time.monotonic() - started) * 1000, 1),
        )
        return message
