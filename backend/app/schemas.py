"""Shared domain types for inter-agent communication.

`app.schemas` may depend only on value objects from `app.llm.base`, never on
provider implementations. This is the unified envelope agents exchange (spec §10).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.llm.base import ToolCall

__all__ = [
    "AgentMessage",
    "AgentRole",
    "Claim",
    "MessageType",
]


class AgentRole(str, Enum):
    """The four V1 roles. Future agents (Verifier, Accountability) extend this."""

    MANAGER = "manager"
    RESEARCHER = "researcher"
    IDEATOR = "ideator"
    SKEPTIC = "skeptic"


class MessageType(str, Enum):
    """Envelope types. DECISION/REVISION are deferred to Phase 5."""

    PLAN = "plan"
    FINDING = "finding"
    IDEA = "idea"
    CRITIQUE = "critique"
    QUESTION = "question"
    SYNTHESIS = "synthesis"


class Claim(BaseModel):
    """Minimal claim; Phase 3 extends with evidence[] and status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class AgentMessage(BaseModel):
    """The unified agent-to-agent message envelope.

    `confidence`, `claims`, and `round` are reserved fields: Phase 2 always
    sends them as None; Phases 3 and 5 populate them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex)
    from_agent: AgentRole
    to_agent: AgentRole | None = None
    type: MessageType
    content: str = ""
    claims: list[Claim] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    tool_calls: list[ToolCall] | None = None
    round: int | None = Field(default=None, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, __context: object) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware (use UTC)")
