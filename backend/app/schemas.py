"""Shared domain types for inter-agent communication.

`app.schemas` may depend only on value objects from `app.llm.base`, never on
provider implementations. This is the unified envelope agents exchange (spec §10).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.llm.base import ToolCall

__all__ = [
    "AgentMessage",
    "AgentRole",
    "Claim",
    "ClaimStatus",
    "ClaimVerdict",
    "DecisionAction",
    "Evidence",
    "ManagerDecision",
    "MessageType",
    "Verdict",
]


class AgentRole(str, Enum):
    """The four V1 roles. Future agents (Verifier, Accountability) extend this."""

    MANAGER = "manager"
    RESEARCHER = "researcher"
    IDEATOR = "ideator"
    SKEPTIC = "skeptic"


class MessageType(str, Enum):
    """Envelope types."""

    PLAN = "plan"
    FINDING = "finding"
    IDEA = "idea"
    CRITIQUE = "critique"
    QUESTION = "question"
    SYNTHESIS = "synthesis"
    DECISION = "decision"      # Phase 5: manager routing decisions
    REVISION = "revision"      # Phase 5: iterative worker revision turns


class ClaimStatus(str, Enum):
    """Epistemic status of a claim (spec §11). FACT requires cited evidence."""

    FACT = "fact"
    ASSUMPTION = "assumption"
    HYPOTHESIS = "hypothesis"
    OPINION = "opinion"
    INFERENCE = "inference"
    UNVERIFIED = "unverified"


class ClaimVerdict(str, Enum):
    """Skeptic verdict vocabulary (approved plan; Verifier's vocabulary differs)."""

    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNVERIFIABLE = "unverifiable"


class Evidence(BaseModel):
    """A single piece of cited support. `source` is mandatory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    quote: str | None = None

    @model_validator(mode="after")
    def _source_not_blank(self) -> "Evidence":
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        return self


class Claim(BaseModel):
    """An ID-addressable claim with epistemic status and evidence.

    `id` is assigned by our code (c1..cN), never by the model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)


class Verdict(BaseModel):
    """Skeptic evaluation of exactly one supplied claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: str = Field(min_length=1)
    verdict: ClaimVerdict
    objection: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _objection_not_blank(self) -> "Verdict":
        if not self.objection.strip():
            raise ValueError("objection must be non-empty")
        return self


class DecisionAction(str, Enum):
    """Manager routing actions (Phase 5 PRD §6.1/§13: iterate consolidated into call_agent)."""

    CALL_AGENT = "call_agent"
    FINISH = "finish"


class ManagerDecision(BaseModel):
    """Constrained routing decision; executed by the orchestrator, never prose."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: DecisionAction
    target: AgentRole | None = None
    instruction: str = ""
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _coherent(self) -> "ManagerDecision":
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.action is DecisionAction.CALL_AGENT:
            if self.target not in {AgentRole.RESEARCHER, AgentRole.IDEATOR}:
                raise ValueError("call_agent requires target researcher or ideator")
            if not self.instruction.strip():
                raise ValueError("call_agent requires a non-empty instruction")
        return self


class AgentMessage(BaseModel):
    """The unified agent-to-agent message envelope.

    Exactly one of `claims`, `verdicts`, `decision` is non-None on a structured
    message. `confidence` is reserved: per-claim confidence is authoritative.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex)
    from_agent: AgentRole
    to_agent: AgentRole | None = None
    type: MessageType
    content: str = ""
    claims: list[Claim] | None = None
    verdicts: list[Verdict] | None = None
    decision: ManagerDecision | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    tool_calls: list[ToolCall] | None = None
    round: int | None = Field(default=None, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, __context: object) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware (use UTC)")
