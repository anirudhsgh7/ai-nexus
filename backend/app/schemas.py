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
    "AccountabilityFlag",
    "AccountabilityFlagKind",
    "AccountabilityReport",
    "AccountabilityStatus",
    "AgentMessage",
    "AgentRole",
    "Claim",
    "ClaimProvenance",
    "ClaimStatus",
    "ClaimVerification",
    "ClaimVerdict",
    "DecisionAction",
    "Evidence",
    "FlagSeverity",
    "ManagerDecision",
    "MessageType",
    "ToolResult",
    "VerificationReport",
    "VerificationStatus",
    "Verdict",
]


class AgentRole(str, Enum):
    """Agent roles. Phase 11 added the two audit roles (PRD §6.8)."""

    MANAGER = "manager"
    RESEARCHER = "researcher"
    IDEATOR = "ideator"
    SKEPTIC = "skeptic"
    VERIFIER = "verifier"            # Phase 11
    ACCOUNTABILITY = "accountability"  # Phase 11


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
    VERIFICATION = "verification"    # Phase 11: final-answer audit
    ACCOUNTABILITY = "accountability"  # Phase 11: trace/process audit


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


class ToolResult(BaseModel):
    """Outcome of one executed tool call (Phase 6 PRD §6.1).

    `content` is the exact envelope string sent back to the model; `error` is
    a machine code when the call failed, `None` on success. Parallel to
    `AgentMessage.tool_calls` by index.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    content: str
    error: str | None = None
    duration_ms: float | None = Field(default=None, ge=0.0)


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


class VerificationStatus(str, Enum):
    """Verifier's four-valued vocabulary (Phase 11 PRD §6.3).

    Deliberately distinct from `ClaimVerdict`: a prior `supported` verdict is
    context for this audit, never grounds for `VERIFIED`.
    """

    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"
    PARTIALLY_VERIFIED = "partially_verified"


class ClaimVerification(BaseModel):
    """The Verifier's evaluation of exactly one final claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: str = Field(min_length=1)
    verification_status: VerificationStatus
    evidence_checked: list[str] = Field(default_factory=list)
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    contradicting_evidence: list[Evidence] = Field(default_factory=list)
    source_references: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _explanation_not_blank(self) -> "ClaimVerification":
        if not self.explanation.strip():
            raise ValueError("explanation must be non-empty")
        return self


class VerificationReport(BaseModel):
    """One entry per supplied final claim (coverage enforced by code)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claims: list[ClaimVerification] = Field(default_factory=list)


class AccountabilityFlagKind(str, Enum):
    """Flag families the process audit can raise (PRD §6.3)."""

    TRACE_INCOMPLETENESS = "trace_incompleteness"
    UNSUPPORTED_FINAL_CLAIM = "unsupported_final_claim"
    UNRESOLVED_CLAIM_SUPPRESSED = "unresolved_claim_suppressed"
    DECISION_INCONSISTENCY = "decision_inconsistency"
    EVIDENCE_PROVENANCE_GAP = "evidence_provenance_gap"
    TOOL_USE_INCONSISTENCY = "tool_use_inconsistency"
    PEER_PROSE_EXPOSURE = "peer_prose_exposure"  # only if detectable
    PREMATURE_STOP = "premature_stop"
    CONFIDENCE_EVIDENCE_MISMATCH = "confidence_evidence_mismatch"
    RETRY_ACTIVITY = "retry_activity"


class FlagSeverity(str, Enum):
    """Severity ladder; `overall_status` derives from the merged flags."""

    INFO = "info"
    WARNING = "warning"
    VIOLATION = "violation"


class AccountabilityFlag(BaseModel):
    """One process/provenance finding. Mechanical entries are code-canonical
    (merged by `enforce_accountability`); semantic entries are the model's."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: AccountabilityFlagKind
    severity: FlagSeverity
    refs: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _explanation_not_blank(self) -> "AccountabilityFlag":
        if not self.explanation.strip():
            raise ValueError("explanation must be non-empty")
        return self


class AccountabilityStatus(str, Enum):
    """Aggregate audit outcome (derived, never asserted against the flags)."""

    CLEAN = "clean"
    WARNINGS = "warnings"
    VIOLATIONS = "violations"


class ClaimProvenance(BaseModel):
    """Where a final claim came from and how it fared."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: str = Field(min_length=1)
    origin: AgentRole
    verdict: ClaimVerdict | None = None
    evidence_count: int = Field(default=0, ge=0)


class AccountabilityReport(BaseModel):
    """Trace/provenance/process audit (Phase 11 PRD §6.3) — never a quality
    judgment. The mechanical fields are overwritten with code-computed facts
    by `enforce_accountability` before anything is persisted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_completeness: bool
    final_claim_provenance: list[ClaimProvenance] = Field(default_factory=list)
    flags: list[AccountabilityFlag] = Field(default_factory=list)
    overall_status: AccountabilityStatus
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def _summary_not_blank(self) -> "AccountabilityReport":
        if not self.summary.strip():
            raise ValueError("summary must be non-empty")
        return self


class AgentMessage(BaseModel):
    """The unified agent-to-agent message envelope.

    Exactly one of `claims`, `verdicts`, `decision`, `verification`,
    `accountability` is non-None on a structured message. `confidence` is
    reserved: per-claim confidence is authoritative. `retries` records the
    structured-output correction retries of this call (Phase 11 makes retry
    activity auditable).
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
    verification: VerificationReport | None = None
    accountability: AccountabilityReport | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    retries: int = Field(default=0, ge=0)
    round: int | None = Field(default=None, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, __context: object) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware (use UTC)")
