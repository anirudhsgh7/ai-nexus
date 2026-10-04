"""Structured output contracts: JSON schemas, directives, extraction, parsing.

Grammar-constrained decoding (Ollama `format=`) makes malformed JSON rare;
`extract_json` + parse/validation give the Agent's retry-once loop something
precise to correct. Model output DTOs intentionally ignore unknown fields so
a stray model-invented `id` can never reach the domain (IDs are ours: c1..cN).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas import (
    AccountabilityFlag,
    AccountabilityFlagKind,
    AccountabilityReport,
    AccountabilityStatus,
    AgentRole,
    Claim,
    ClaimProvenance,
    ClaimStatus,
    ClaimVerdict,
    ClaimVerification,
    DecisionAction,
    Evidence,
    FlagSeverity,
    ManagerDecision,
    VerificationReport,
    VerificationStatus,
    Verdict,
)

__all__ = [
    "ACCOUNTABILITY_SCHEMA",
    "DECISION_SCHEMA",
    "DEFAULT_STRUCTURED_MAX_TOKENS",
    "OutputKind",
    "StructuredResult",
    "VERIFICATION_SCHEMA",
    "correction_message",
    "directive_for",
    "extract_json",
    "normalize_decision",
    "parse_accountability",
    "parse_claims",
    "parse_decision",
    "parse_verification",
    "parse_verdicts",
    "schema_for",
]

DEFAULT_STRUCTURED_MAX_TOKENS = 2048

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class OutputKind(str, Enum):
    PLAIN = "plain"
    CLAIMS = "claims"
    VERDICTS = "verdicts"
    DECISION = "decision"
    VERIFICATION = "verification"      # Phase 11
    ACCOUNTABILITY = "accountability"  # Phase 11


def _evidence_subschema() -> dict:
    # minLength is load-bearing: qwen2.5 intermittently emits "" for these
    # fields, which the domain models reject (string_too_short) and the
    # grammar cannot express if the constraint is absent. Live-verified that
    # Ollama's schema->grammar honors nested minLength (Phase 11 §18).
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "minLength": 1},
                "quote": {"type": "string"},
            },
            "required": ["source"],
        },
    }


CLAIMS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string", "minLength": 1},
                    "status": {
                        "type": "string",
                        "enum": [s.value for s in ClaimStatus],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": _evidence_subschema(),
                },
                "required": ["statement", "status", "evidence"],
            },
        },
    },
    "required": ["content", "claims"],
}

VERDICTS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string", "minLength": 1},
                    "verdict": {
                        "type": "string",
                        "enum": [v.value for v in ClaimVerdict],
                    },
                    "objection": {"type": "string", "minLength": 1},
                    "evidence": _evidence_subschema(),
                },
                "required": ["claim_id", "verdict", "objection", "evidence"],
            },
        },
    },
    "required": ["content", "verdicts"],
}

_CLAIMS_SENTENCE = (
    'In "claims", list every factual claim, assumption, or hypothesis you rely on. '
    "Use status=fact only when you cite evidence."
)
_VERDICTS_SENTENCE = (
    'In "verdicts", evaluate every claim under "CLAIMS TO EVALUATE" with exactly '
    "one entry per claim. Use verdict=supported only when the claim's cited "
    "evidence supports it, and a supported verdict requires evidence in either "
    "the claim or the verdict entry — without evidence in either, use "
    "unverifiable instead of supported. Every entry's objection must be a "
    "non-empty sentence; for supported claims state what the evidence shows, "
    "for unverifiable claims name the missing evidence."
)

DECISION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": [a.value for a in DecisionAction]},
        "target": {"type": "string", "enum": ["researcher", "ideator"]},
        # instruction stays unbounded: FINISH decisions legitimately carry ""
        "instruction": {"type": "string"},
        "reason": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["action", "reason", "confidence"],
}

_DIRECTIVE_HEADER = (
    "OUTPUT FORMAT (mandatory):\n"
    "Respond with a single JSON object and nothing else. No markdown, no commentary.\n"
    "The object must match this JSON schema exactly:\n"
)

_DECISION_BODY = """\
Ignore any earlier instructions about prose or sections; this call is a routing decision only.
Use action="call_agent" with target "researcher" or "ideator" and a concrete instruction when one more targeted work step is likely to resolve unresolved claims. The instruction must address the Skeptic's specific objection for one or more unresolved claims (for example: "use web_search to find a 2025 source for claim c2", or "revise claim c4 with evidence that answers the stated objection"). Do not repeat a previous instruction and do not issue generic instructions like "research more".
Use action="finish" when no unresolved claims remain, when no remaining work step is likely to help, or when the round cap has been reached and an honest incomplete synthesis is required.
If the remaining unresolved claims cannot be resolved with the available tools (for example, prior tool calls returned provider or network errors and no files cover them), do not repeat a failing search: use action="finish" so the synthesis can answer with clearly labeled uncertainty.
"""

# ------------------------------------------------------------ Phase 11

VERIFICATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string", "minLength": 1},
                    "verification_status": {
                        "type": "string",
                        "enum": [s.value for s in VerificationStatus],
                    },
                    "evidence_checked": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "supporting_evidence": _evidence_subschema(),
                    "contradicting_evidence": _evidence_subschema(),
                    "source_references": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "explanation": {"type": "string", "minLength": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "claim_id", "verification_status", "evidence_checked",
                    "supporting_evidence", "contradicting_evidence",
                    "source_references", "explanation", "confidence",
                ],
            },
        },
    },
    "required": ["content", "claims"],
}

_VERIFICATION_SENTENCE = (
    'In "claims", evaluate every claim under "CLAIMS TO EVALUATE" with exactly '
    "one entry per claim id. Verify against evidence you check yourself; a prior "
    "verdict is context, never proof. Use verified only with your own supporting "
    "evidence and source references, contradicted only with contradicting "
    "evidence, partially_verified when support is incomplete, unverifiable when "
    "nothing authoritative can be checked."
)

ACCOUNTABILITY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "trace_completeness": {"type": "boolean"},
        "final_claim_provenance": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string", "minLength": 1},
                    "origin": {
                        "type": "string",
                        "enum": ["manager", "researcher", "ideator", "skeptic"],
                    },
                    "verdict": {
                        "type": "string",
                        "enum": [v.value for v in ClaimVerdict],
                    },
                    "evidence_count": {"type": "integer"},
                },
                "required": ["claim_id", "origin", "evidence_count"],
            },
        },
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [k.value for k in AccountabilityFlagKind],
                    },
                    "severity": {
                        "type": "string",
                        "enum": [s.value for s in FlagSeverity],
                    },
                    "refs": {"type": "array", "items": {"type": "string"}},
                    "explanation": {"type": "string", "minLength": 1},
                },
                "required": ["kind", "severity", "refs", "explanation"],
            },
        },
        "overall_status": {
            "type": "string",
            "enum": [s.value for s in AccountabilityStatus],
        },
        "summary": {"type": "string", "minLength": 1},
    },
    "required": [
        "content", "trace_completeness", "final_claim_provenance",
        "flags", "overall_status", "summary",
    ],
}

_ACCOUNTABILITY_SENTENCE = (
    'Audit the trace. "final_claim_provenance" must echo the provided facts '
    'exactly. Every FACT entry listed in the context must appear in "flags" with '
    'the same kind/severity/refs. "overall_status" is violations if any flag is '
    "a violation, warnings if any is a warning, else clean. Do not re-research "
    "claims and do not judge answer quality."
)


def schema_for(kind: OutputKind) -> dict | None:
    if kind is OutputKind.CLAIMS:
        return CLAIMS_SCHEMA
    if kind is OutputKind.VERDICTS:
        return VERDICTS_SCHEMA
    if kind is OutputKind.DECISION:
        return DECISION_SCHEMA
    if kind is OutputKind.VERIFICATION:
        return VERIFICATION_SCHEMA
    if kind is OutputKind.ACCOUNTABILITY:
        return ACCOUNTABILITY_SCHEMA
    return None


_SENTENCES: dict[OutputKind, str] = {
    OutputKind.CLAIMS: _CLAIMS_SENTENCE,
    OutputKind.VERDICTS: _VERDICTS_SENTENCE,
    OutputKind.VERIFICATION: _VERIFICATION_SENTENCE,
    OutputKind.ACCOUNTABILITY: _ACCOUNTABILITY_SENTENCE,
}


def directive_for(kind: OutputKind) -> str:
    if kind is OutputKind.PLAIN:
        return ""
    schema = schema_for(kind)
    if kind is OutputKind.DECISION:
        body = _DECISION_BODY
    else:
        sentence = _SENTENCES[kind]
        body = f'The "content" field holds your full prose answer.\n{sentence}'
    return (
        f"{_DIRECTIVE_HEADER}{json.dumps(schema, separators=(',', ':'))}\n\n{body}"
    )


# Known violations paired with the concrete fix, so the one allowed retry is
# actionable instead of a bare restatement. Live-verified: a temp-0 manager
# that re-emitted `status=fact` with no evidence twice in a row complied
# (fact -> assumption) as soon as the correction named the fix (Phase 11 §18).
_FIX_HINTS: tuple[tuple[str, str], ...] = (
    (
        "status=fact but no evidence",
        "add an evidence entry with a source, or use "
        "status=unverified/assumption/hypothesis instead of fact",
    ),
    (
        "marked supported without evidence",
        "add an evidence entry with a source to either the claim or the verdict entry, or use "
        "verdict=unverifiable instead of supported",
    ),
    ("has no objection", "write a non-empty objection sentence for that verdict"),
    (
        "was not evaluated",
        "add exactly one verdict entry for every claim id under CLAIMS TO EVALUATE",
    ),
    (
        "verified without recording checked evidence",
        "list every source you actually checked in evidence_checked",
    ),
    (
        "verified without supporting evidence",
        "add your supporting evidence, or use unverifiable",
    ),
    (
        "verified without source references",
        "include the source references you actually used",
    ),
    (
        "partially verified without evidence",
        "list your supporting evidence, or use unverifiable if you checked nothing",
    ),
    (
        "contradicted without contradicting evidence",
        "add the contradicting evidence, or use unverifiable",
    ),
    (
        "verdict for unknown claim",
        "copy the exact claim id as printed (e.g. c1, c2); do not add brackets or other characters",
    ),
    (
        "verified claims require at least one independent tool check",
        "downgrade verified and partially_verified entries to unverifiable since no tool check was performed",
    ),
)


def correction_message(problems: list[str]) -> str:
    annotated = []
    for problem in problems:
        fix = next(
            (hint for needle, hint in _FIX_HINTS if needle in problem), None
        )
        annotated.append(f"{problem} -> fix: {fix}" if fix else problem)
    joined = "; ".join(annotated)
    return (
        f"Your previous response was rejected: {joined}.\n"
        "Respond again with ONLY a single valid JSON object matching the required "
        "schema.\nNo markdown, no commentary."
    )


def extract_json(raw: str) -> str:
    """Four-branch tolerant extraction (PRD §6.5.1); raises on failure."""
    text = raw.strip()
    if not text:
        raise ValueError("empty response")

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    fence = _FENCE_RE.search(text)
    if fence:
        body = fence.group(1).strip()
        try:
            json.loads(body)
            return body
        except json.JSONDecodeError:
            pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        json.loads(candidate)  # raises JSONDecodeError for the retry path
        return candidate

    raise ValueError("no JSON object found in response")


# ----------------------------------------------------------- parse DTOs
# extra fields are ignored by default: a model-invented "id" must never
# reach the domain (claim IDs are assigned by our code, PRD §6.5 step 7).


class _EvidenceDTO(BaseModel):
    source: str
    quote: str | None = None


class _ClaimDTO(BaseModel):
    statement: str
    status: ClaimStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[_EvidenceDTO] = Field(default_factory=list)


class _ClaimsOutput(BaseModel):
    content: str
    claims: list[_ClaimDTO]


class _VerdictDTO(BaseModel):
    claim_id: str
    verdict: ClaimVerdict
    objection: str
    evidence: list[_EvidenceDTO] = Field(default_factory=list)


class _VerdictsOutput(BaseModel):
    content: str
    verdicts: list[_VerdictDTO]


class _DecisionDTO(BaseModel):
    action: DecisionAction
    target: AgentRole | None = None
    instruction: str = ""
    reason: str
    confidence: float


class _VerificationDTO(BaseModel):
    claim_id: str
    verification_status: VerificationStatus
    evidence_checked: list[str] = Field(default_factory=list)
    supporting_evidence: list[_EvidenceDTO] = Field(default_factory=list)
    contradicting_evidence: list[_EvidenceDTO] = Field(default_factory=list)
    source_references: list[str] = Field(default_factory=list)
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class _VerificationOutput(BaseModel):
    content: str
    claims: list[_VerificationDTO]


class _ProvenanceDTO(BaseModel):
    claim_id: str
    origin: AgentRole
    verdict: ClaimVerdict | None = None
    evidence_count: int = Field(default=0, ge=0)


class _FlagDTO(BaseModel):
    kind: AccountabilityFlagKind
    severity: FlagSeverity
    refs: list[str] = Field(default_factory=list)
    explanation: str


class _AccountabilityOutput(BaseModel):
    content: str
    trace_completeness: bool
    final_claim_provenance: list[_ProvenanceDTO] = Field(default_factory=list)
    flags: list[_FlagDTO] = Field(default_factory=list)
    overall_status: AccountabilityStatus
    summary: str


@dataclass(frozen=True, slots=True)
class StructuredResult:
    content: str
    claims: list[Claim] | None
    verdicts: list[Verdict] | None
    decision: ManagerDecision | None = None
    verification: VerificationReport | None = None
    accountability: AccountabilityReport | None = None


def _domain_evidence(items: list[_EvidenceDTO]) -> list[Evidence]:
    return [Evidence(source=e.source, quote=e.quote) for e in items]


def parse_claims(raw: str) -> StructuredResult:
    """Extract -> parse -> assign IDs c1..cN in output order."""
    data = json.loads(extract_json(raw))
    out = _ClaimsOutput.model_validate(data)
    claims = [
        Claim(
            id=f"c{index}",
            statement=item.statement,
            status=item.status,
            confidence=item.confidence,
            evidence=_domain_evidence(item.evidence),
        )
        for index, item in enumerate(out.claims, start=1)
    ]
    return StructuredResult(content=out.content, claims=claims, verdicts=None)


def parse_verdicts(raw: str) -> StructuredResult:
    data = json.loads(extract_json(raw))
    out = _VerdictsOutput.model_validate(data)
    verdicts = [
        Verdict(
            claim_id=item.claim_id,
            verdict=item.verdict,
            objection=item.objection,
            evidence=_domain_evidence(item.evidence),
        )
        for item in out.verdicts
    ]
    return StructuredResult(content=out.content, claims=None, verdicts=verdicts)


def parse_decision(raw: str) -> StructuredResult:
    """Extract -> DTO (extras ignored) -> ManagerDecision (validators run).

    Incoherent decisions (call_agent without a worker target/instruction)
    raise ValidationError, feeding the Agent's retry-once correction loop.
    """
    data = json.loads(extract_json(raw))
    out = _DecisionDTO.model_validate(data)
    decision = ManagerDecision(
        action=out.action,
        target=out.target,
        instruction=out.instruction,
        reason=out.reason,
        confidence=out.confidence,
    )
    return StructuredResult(content="", claims=None, verdicts=None, decision=decision)


def normalize_decision(decision: ManagerDecision) -> ManagerDecision:
    """FINISH decisions tolerate (and scrub) stray target/instruction (PRD §6.1)."""
    if decision.action is DecisionAction.FINISH and (
        decision.target is not None or decision.instruction
    ):
        return decision.model_copy(update={"target": None, "instruction": ""})
    return decision


def parse_verification(raw: str) -> StructuredResult:
    """Extract -> DTO (extras ignored) -> `VerificationReport`.

    Claim ids pass through untouched; coverage and status rules are enforced
    by `app.audits.validate_verification` (retry-once correction path).
    """
    data = json.loads(extract_json(raw))
    out = _VerificationOutput.model_validate(data)
    report = VerificationReport(
        claims=[
            ClaimVerification(
                claim_id=item.claim_id,
                verification_status=item.verification_status,
                evidence_checked=item.evidence_checked,
                supporting_evidence=_domain_evidence(item.supporting_evidence),
                contradicting_evidence=_domain_evidence(
                    item.contradicting_evidence
                ),
                source_references=item.source_references,
                explanation=item.explanation,
                confidence=item.confidence,
            )
            for item in out.claims
        ]
    )
    return StructuredResult(
        content=out.content, claims=None, verdicts=None, verification=report
    )


def parse_accountability(raw: str) -> StructuredResult:
    """Extract -> DTO -> `AccountabilityReport` (pydantic validators run).

    Mechanical completeness is NOT trusted to the model: the orchestrator
    merges code-computed facts via `enforce_accountability` before persisting.
    """
    data = json.loads(extract_json(raw))
    out = _AccountabilityOutput.model_validate(data)
    report = AccountabilityReport(
        trace_completeness=out.trace_completeness,
        final_claim_provenance=[
            ClaimProvenance(
                claim_id=item.claim_id,
                origin=item.origin,
                verdict=item.verdict,
                evidence_count=item.evidence_count,
            )
            for item in out.final_claim_provenance
        ],
        flags=[
            AccountabilityFlag(
                kind=item.kind,
                severity=item.severity,
                refs=item.refs,
                explanation=item.explanation,
            )
            for item in out.flags
        ],
        overall_status=out.overall_status,
        summary=out.summary,
    )
    return StructuredResult(
        content=out.content, claims=None, verdicts=None, accountability=report
    )
