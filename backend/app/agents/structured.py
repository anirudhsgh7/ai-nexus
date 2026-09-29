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

from app.schemas import Claim, ClaimStatus, ClaimVerdict, Evidence, Verdict

__all__ = [
    "DEFAULT_STRUCTURED_MAX_TOKENS",
    "OutputKind",
    "StructuredResult",
    "correction_message",
    "directive_for",
    "extract_json",
    "parse_claims",
    "parse_verdicts",
    "schema_for",
]

DEFAULT_STRUCTURED_MAX_TOKENS = 2048

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class OutputKind(str, Enum):
    PLAIN = "plain"
    CLAIMS = "claims"
    VERDICTS = "verdicts"


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

_DIRECTIVE_HEADER = (
    "OUTPUT FORMAT (mandatory):\n"
    "Respond with a single JSON object and nothing else. No markdown, no commentary.\n"
    "The object must match this JSON schema exactly:\n"
)


def schema_for(kind: OutputKind) -> dict | None:
    if kind is OutputKind.CLAIMS:
        return CLAIMS_SCHEMA
    if kind is OutputKind.VERDICTS:
        return VERDICTS_SCHEMA
    return None


def directive_for(kind: OutputKind) -> str:
    if kind not in (OutputKind.CLAIMS, OutputKind.VERDICTS):
        return ""
    schema = schema_for(kind)
    sentence = _CLAIMS_SENTENCE if kind is OutputKind.CLAIMS else _VERDICTS_SENTENCE
    return (
        f"{_DIRECTIVE_HEADER}{json.dumps(schema, separators=(',', ':'))}\n\n"
        f'The "content" field holds your full prose answer.\n{sentence}'
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


@dataclass(frozen=True, slots=True)
class StructuredResult:
    content: str
    claims: list[Claim] | None
    verdicts: list[Verdict] | None


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
