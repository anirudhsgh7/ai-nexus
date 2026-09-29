"""Structured output contracts (PRD §6.3–6.5): schemas, extraction, parsing."""

import json

import pytest
from pydantic import ValidationError

from app.agents.structured import (
    CLAIMS_SCHEMA,
    DEFAULT_STRUCTURED_MAX_TOKENS,
    VERDICTS_SCHEMA,
    OutputKind,
    correction_message,
    directive_for,
    extract_json,
    parse_claims,
    parse_verdicts,
    schema_for,
)
from app.schemas import ClaimStatus, ClaimVerdict


# ---------------------------------------------------------------- schemas

def test_schema_for_kinds():
    assert schema_for(OutputKind.PLAIN) is None
    assert schema_for(OutputKind.CLAIMS) is CLAIMS_SCHEMA
    assert schema_for(OutputKind.VERDICTS) is VERDICTS_SCHEMA


def test_claims_schema_shape():
    assert CLAIMS_SCHEMA["required"] == ["content", "claims"]
    item = CLAIMS_SCHEMA["properties"]["claims"]["items"]
    assert item["required"] == ["statement", "status", "evidence"]
    assert item["properties"]["status"]["enum"] == [s.value for s in ClaimStatus]
    assert item["properties"]["confidence"] == {"type": "number", "minimum": 0, "maximum": 1}
    evidence = item["properties"]["evidence"]
    assert evidence["items"]["required"] == ["source"]
    assert "id" not in item["properties"], "IDs must not be model-provided"


def test_verdicts_schema_shape():
    assert VERDICTS_SCHEMA["required"] == ["content", "verdicts"]
    item = VERDICTS_SCHEMA["properties"]["verdicts"]["items"]
    assert item["required"] == ["claim_id", "verdict", "objection", "evidence"]
    assert item["properties"]["verdict"]["enum"] == [v.value for v in ClaimVerdict]


def test_token_cap_constant():
    assert DEFAULT_STRUCTURED_MAX_TOKENS == 2048


# ---------------------------------------------------------------- directives

def test_directive_plain_is_empty():
    assert directive_for(OutputKind.PLAIN) == ""


def test_directive_contains_schema_and_sentence():
    directive = directive_for(OutputKind.CLAIMS)
    assert directive.startswith("OUTPUT FORMAT (mandatory):")
    assert "No markdown, no commentary." in directive
    compact = json.dumps(CLAIMS_SCHEMA, separators=(",", ":"))
    assert compact in directive
    assert 'In "claims"' in directive
    assert "content" in directive


def test_verdicts_directive():
    directive = directive_for(OutputKind.VERDICTS)
    compact = json.dumps(VERDICTS_SCHEMA, separators=(",", ":"))
    assert compact in directive
    assert 'In "verdicts"' in directive
    assert "CLAIMS TO EVALUATE" in directive


# ---------------------------------------------------------------- extraction

def test_extract_direct_json():
    assert extract_json('{"a": 1}') == '{"a": 1}'
    assert extract_json('  {"a":1}  ') == '{"a":1}'


def test_extract_fenced_json():
    raw = '```json\n{"a": 1}\n```'
    assert json.loads(extract_json(raw)) == {"a": 1}


def test_extract_fenced_bare():
    raw = '```\n{"a": 1}\n```'
    assert json.loads(extract_json(raw)) == {"a": 1}


def test_extract_embedded_in_prose():
    raw = 'Sure! Here is the result: {"a": 1} hope that helps.'
    assert json.loads(extract_json(raw)) == {"a": 1}


def test_extract_garbage_raises():
    with pytest.raises(ValueError):
        extract_json("no json here at all")
    with pytest.raises(ValueError):
        extract_json("")
    with pytest.raises(ValueError):
        extract_json("{broken")


def test_extract_broken_candidate_raises_json_error():
    import json as _json

    with pytest.raises(_json.JSONDecodeError):
        extract_json("text {not valid json} more")


# ---------------------------------------------------------------- parse_claims

def _claims_payload(**overrides) -> str:
    payload = {
        "content": "Prose answer.",
        "claims": [
            {
                "statement": "X grew 40% in 2025.",
                "status": "unverified",
                "confidence": 0.55,
                "evidence": [{"source": "blog", "quote": "up 40%"}],
            },
            {"statement": "Assumed context.", "status": "assumption", "evidence": []},
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_parse_claims_assigns_ids_in_order():
    result = parse_claims(_claims_payload())
    assert result.claims is not None
    assert [c.id for c in result.claims] == ["c1", "c2"]
    assert result.verdicts is None
    assert result.content == "Prose answer."
    assert result.claims[0].status is ClaimStatus.UNVERIFIED
    assert result.claims[0].confidence == 0.55
    assert result.claims[0].evidence[0].source == "blog"
    assert result.claims[0].evidence[0].quote == "up 40%"
    assert result.claims[1].status is ClaimStatus.ASSUMPTION
    assert result.claims[1].evidence == []


def test_parse_claims_empty_array_accepted():
    result = parse_claims(json.dumps({"content": "nothing", "claims": []}))
    assert result.claims == []


def test_parse_claims_model_invented_id_ignored():
    payload = json.dumps({
        "content": "x",
        "claims": [{"id": "hallucinated-99", "statement": "s", "status": "fact",
                    "evidence": [{"source": "s"}]}],
    })
    result = parse_claims(payload)
    assert result.claims[0].id == "c1"  # ours, not the model's


def test_parse_claims_unknown_enum_rejected():
    payload = json.dumps({"content": "x", "claims": [
        {"statement": "s", "status": "verified", "evidence": []}]})
    with pytest.raises(ValidationError):
        parse_claims(payload)


def test_parse_claims_confidence_out_of_range_rejected():
    payload = json.dumps({"content": "x", "claims": [
        {"statement": "s", "status": "fact", "confidence": 85,
         "evidence": [{"source": "s"}]}]})
    with pytest.raises(ValidationError):
        parse_claims(payload)


def test_parse_claims_wrapped_in_prose_recovers():
    raw = 'Here you go:\n```json\n' + _claims_payload() + "\n```"
    result = parse_claims(raw)
    assert [c.id for c in result.claims] == ["c1", "c2"]


# ---------------------------------------------------------------- parse_verdicts

def _verdicts_payload(**overrides) -> str:
    payload = {
        "content": "Checked both claims.",
        "verdicts": [
            {"claim_id": "c1", "verdict": "supported", "objection": "evidence holds",
             "evidence": [{"source": "report"}]},
            {"claim_id": "c2", "verdict": "unverifiable", "objection": "no source exists",
             "evidence": []},
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_parse_verdicts():
    result = parse_verdicts(_verdicts_payload())
    assert result.claims is None
    assert result.verdicts is not None
    assert [v.claim_id for v in result.verdicts] == ["c1", "c2"]
    assert result.verdicts[0].verdict is ClaimVerdict.SUPPORTED
    assert result.verdicts[0].evidence[0].source == "report"
    assert result.verdicts[1].verdict is ClaimVerdict.UNVERIFIABLE
    assert result.verdicts[1].evidence == []


def test_parse_verdicts_unknown_enum_rejected():
    payload = json.dumps({"content": "x", "verdicts": [
        {"claim_id": "c1", "verdict": "verified", "objection": "o", "evidence": []}]})
    with pytest.raises(ValidationError):
        parse_verdicts(payload)


def test_parse_verdicts_missing_required_rejected():
    payload = json.dumps({"content": "x", "verdicts": [
        {"claim_id": "c1", "verdict": "refuted", "evidence": []}]})  # no objection
    with pytest.raises(ValidationError):
        parse_verdicts(payload)


# ---------------------------------------------------------------- correction

def test_correction_message_format():
    msg = correction_message(["claim c2 was not evaluated", "bad json"])
    assert msg == (
        "Your previous response was rejected: claim c2 was not evaluated; bad json.\n"
        "Respond again with ONLY a single valid JSON object matching the required "
        "schema.\nNo markdown, no commentary."
    )
