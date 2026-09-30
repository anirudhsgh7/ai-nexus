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
    AgentRole,
    Claim,
    ClaimStatus,
    ClaimVerdict,
    DecisionAction,
    Evidence,
    ManagerDecision,
    Verdict,
)

__all__ = [
    "DECISION_SCHEMA",
    "DEFAULT_STRUCTURED_MAX_TOKENS",
    "OutputKind",
    "StructuredResult",
    "correction_message",
    "directive_for",
    "extract_json",
    "normalize_decision",
    "parse_claims",
    "parse_decision",
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


def _evidence_subschema() -> dict:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
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
                    "statement": {"type": "string"},
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
                    "claim_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": [v.value for v in ClaimVerdict],
                    },
                    "objection": {"type": "string"},
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
    "evidence supports it."
)

DECISION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": [a.value for a in DecisionAction]},
        "target": {"type": "string", "enum": ["researcher", "ideator"]},
        "instruction": {"type": "string"},
        "reason": {"type": "string"},
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


def schema_for(kind: OutputKind) -> dict | None:
    if kind is OutputKind.CLAIMS:
        return CLAIMS_SCHEMA
    if kind is OutputKind.VERDICTS:
        return VERDICTS_SCHEMA
    if kind is OutputKind.DECISION:
        return DECISION_SCHEMA
    return None


def directive_for(kind: OutputKind) -> str:
    if kind is OutputKind.PLAIN:
        return ""
    schema = schema_for(kind)
    if kind is OutputKind.DECISION:
        body = _DECISION_BODY
    else:
        sentence = _CLAIMS_SENTENCE if kind is OutputKind.CLAIMS else _VERDICTS_SENTENCE
        body = f'The "content" field holds your full prose answer.\n{sentence}'
    return (
        f"{_DIRECTIVE_HEADER}{json.dumps(schema, separators=(',', ':'))}\n\n{body}"
    )


def correction_message(problems: list[str]) -> str:
    joined = "; ".join(problems)
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


@dataclass(frozen=True, slots=True)
class StructuredResult:
    content: str
    claims: list[Claim] | None
    verdicts: list[Verdict] | None
    decision: ManagerDecision | None = None


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
