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
    # the never-blank objection rule lives with the schema it governs
    assert "objection must be a non-empty" in directive


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
        "Your previous response was rejected: "
        "claim c2 was not evaluated -> fix: add exactly one verdict entry for "
        "every claim id under CLAIMS TO EVALUATE; bad json.\n"
        "Respond again with ONLY a single valid JSON object matching the required "
        "schema.\nNo markdown, no commentary."
    )


def test_correction_message_fix_hints_for_known_violations():
    """Each recurring violation names the concrete fix (Phase 11 §18) — the
    one allowed retry must be actionable, not a restatement."""
    cases = {
        "claim c1 has status=fact but no evidence":
            "use status=unverified/assumption/hypothesis instead of fact",
        "claim c1 marked supported without evidence":
            "verdict=unverifiable instead of supported",
        "verdict for c1 has no objection":
            "write a non-empty objection sentence",
        "claim c3 verified without recording checked evidence":
            "list every source you actually checked in evidence_checked",
        "claim c3 partially verified without evidence":
            "use unverifiable if you checked nothing",
    }
    for problem, needle in cases.items():
        msg = correction_message([problem])
        assert f"{problem} -> fix: " in msg, problem
        assert needle in msg, problem
    # unknown problems pass through without a fabricated fix
    unrecognised = "some brand-new violation"
    assert f"{unrecognised}." in correction_message([unrecognised])
    assert "-> fix:" not in correction_message([unrecognised])


# ------------------------------------------------- Phase 11 audit output kinds

def test_audit_schema_for_kinds():
    from app.agents.structured import ACCOUNTABILITY_SCHEMA, VERIFICATION_SCHEMA

    assert schema_for(OutputKind.VERIFICATION) is VERIFICATION_SCHEMA
    assert schema_for(OutputKind.ACCOUNTABILITY) is ACCOUNTABILITY_SCHEMA


def test_verification_schema_shape():
    from app.agents.structured import VERIFICATION_SCHEMA
    from app.schemas import VerificationStatus

    assert VERIFICATION_SCHEMA["required"] == ["content", "claims"]
    item = VERIFICATION_SCHEMA["properties"]["claims"]["items"]
    assert set(item["required"]) == {
        "claim_id", "verification_status", "evidence_checked",
        "supporting_evidence", "contradicting_evidence",
        "source_references", "explanation", "confidence",
    }
    assert item["properties"]["verification_status"]["enum"] == [
        s.value for s in VerificationStatus
    ]
    assert item["properties"]["confidence"] == {
        "type": "number", "minimum": 0, "maximum": 1,
    }
    assert item["properties"]["supporting_evidence"]["items"]["required"] == ["source"]


def test_accountability_schema_shape():
    from app.agents.structured import ACCOUNTABILITY_SCHEMA
    from app.schemas import (
        AccountabilityFlagKind,
        AccountabilityStatus,
        FlagSeverity,
    )

    assert ACCOUNTABILITY_SCHEMA["required"] == [
        "content", "trace_completeness", "final_claim_provenance",
        "flags", "overall_status", "summary",
    ]
    prov = ACCOUNTABILITY_SCHEMA["properties"]["final_claim_provenance"]["items"]
    assert prov["required"] == ["claim_id", "origin", "evidence_count"]
    assert prov["properties"]["origin"]["enum"] == [
        "manager", "researcher", "ideator", "skeptic",
    ]
    flag = ACCOUNTABILITY_SCHEMA["properties"]["flags"]["items"]
    assert flag["properties"]["kind"]["enum"] == [k.value for k in AccountabilityFlagKind]
    assert flag["properties"]["severity"]["enum"] == [s.value for s in FlagSeverity]
    assert ACCOUNTABILITY_SCHEMA["properties"]["overall_status"]["enum"] == [
        s.value for s in AccountabilityStatus
    ]


def test_verification_directive():
    directive = directive_for(OutputKind.VERIFICATION)
    from app.agents.structured import VERIFICATION_SCHEMA

    assert json.dumps(VERIFICATION_SCHEMA, separators=(",", ":")) in directive
    assert "prior verdict is context, never proof" in directive
    assert "one entry per claim id" in directive


def test_accountability_directive():
    directive = directive_for(OutputKind.ACCOUNTABILITY)
    from app.agents.structured import ACCOUNTABILITY_SCHEMA

    assert json.dumps(ACCOUNTABILITY_SCHEMA, separators=(",", ":")) in directive
    assert "must appear in" in directive
    assert "do not judge answer quality" in directive


def test_parse_verification():
    from app.agents.structured import parse_verification
    from app.schemas import VerificationStatus

    raw = json.dumps({
        "content": "checked the file",
        "claims": [
            {
                "claim_id": "c1",
                "verification_status": "verified",
                "evidence_checked": ["growth_report.txt"],
                "supporting_evidence": [
                    {"source": "growth_report.txt", "quote": "23%"}
                ],
                "contradicting_evidence": [],
                "source_references": ["growth_report.txt"],
                "explanation": "File states 23%.",
                "confidence": 0.9,
            },
            {
                "claim_id": "c2",
                "verification_status": "unverifiable",
                "evidence_checked": [],
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "source_references": [],
                "explanation": "No source exists.",
                "confidence": 0.3,
            },
        ],
    })
    result = parse_verification(raw)
    assert result.claims is None and result.verdicts is None
    assert result.verification is not None
    assert [c.claim_id for c in result.verification.claims] == ["c1", "c2"]
    assert (
        result.verification.claims[0].verification_status
        is VerificationStatus.VERIFIED
    )
    assert result.verification.claims[0].supporting_evidence[0].quote == "23%"
    assert result.verification.claims[1].confidence == 0.3


def test_parse_verification_unknown_enum_rejected():
    from app.agents.structured import parse_verification

    raw = json.dumps({
        "content": "x",
        "claims": [{
            "claim_id": "c1", "verification_status": "probably_true",
            "evidence_checked": [], "supporting_evidence": [],
            "contradicting_evidence": [], "source_references": [],
            "explanation": "x", "confidence": 0.5,
        }],
    })
    with pytest.raises(ValidationError):
        parse_verification(raw)


def test_parse_verification_missing_required_rejected():
    from app.agents.structured import parse_verification

    raw = json.dumps({
        "content": "x",
        "claims": [{"claim_id": "c1", "verification_status": "verified"}],
    })
    with pytest.raises(ValidationError):
        parse_verification(raw)


def test_parse_accountability():
    from app.agents.structured import parse_accountability
    from app.schemas import (
        AccountabilityFlagKind,
        AccountabilityStatus,
        FlagSeverity,
    )

    raw = json.dumps({
        "content": "trace audit",
        "trace_completeness": True,
        "final_claim_provenance": [
            {"claim_id": "c1", "origin": "researcher",
             "verdict": "supported", "evidence_count": 2},
            {"claim_id": "c2", "origin": "ideator", "evidence_count": 0},
        ],
        "flags": [{
            "kind": "retry_activity", "severity": "info",
            "refs": ["4"], "explanation": "one structured retry",
        }],
        "overall_status": "clean",
        "summary": "no gaps",
    })
    result = parse_accountability(raw)
    assert result.accountability is not None
    report = result.accountability
    assert report.trace_completeness is True
    assert report.overall_status is AccountabilityStatus.CLEAN
    assert report.final_claim_provenance[0].origin.value == "researcher"
    assert report.final_claim_provenance[1].verdict is None
    assert report.flags[0].kind is AccountabilityFlagKind.RETRY_ACTIVITY
    assert report.flags[0].severity is FlagSeverity.INFO


def test_parse_accountability_unknown_enum_rejected():
    from app.agents.structured import parse_accountability

    raw = json.dumps({
        "content": "x", "trace_completeness": True,
        "final_claim_provenance": [],
        "flags": [{"kind": "made_up_kind", "severity": "info",
                   "refs": [], "explanation": "x"}],
        "overall_status": "clean", "summary": "x",
    })
    with pytest.raises(ValidationError):
        parse_accountability(raw)


def test_parse_verification_ignores_unknown_fields():
    """Extra keys the model invents are tolerated (DTO extra=ignore, PRD §9.1)."""
    from app.agents.structured import parse_verification
    from app.schemas import VerificationStatus

    raw = json.dumps({
        "content": "checked",
        "model_note": "I am confident",
        "claims": [{
            "claim_id": "c1",
            "verification_status": "unverifiable",
            "evidence_checked": [], "supporting_evidence": [],
            "contradicting_evidence": [], "source_references": [],
            "explanation": "nothing checkable", "confidence": 0.4,
            "hallucinated_field": {"nested": True},
        }],
        "top_level_invented": [1, 2, 3],
    })
    result = parse_verification(raw)
    assert result.verification is not None
    entry = result.verification.claims[0]
    assert entry.verification_status is VerificationStatus.UNVERIFIABLE
    assert not hasattr(entry, "hallucinated_field")


def test_parse_accountability_ignores_unknown_fields():
    """Invented keys at every level pass through without failing the parse."""
    from app.agents.structured import parse_accountability
    from app.schemas import AccountabilityStatus

    raw = json.dumps({
        "content": "audit",
        "invented_top": "value",
        "trace_completeness": True,
        "final_claim_provenance": [{
            "claim_id": "c1", "origin": "researcher",
            "evidence_count": 1, "invented_prov": 7,
        }],
        "flags": [{
            "kind": "retry_activity", "severity": "info",
            "refs": ["3"], "explanation": "one retry",
            "invented_flag": True,
        }],
        "overall_status": "clean",
        "summary": "ok",
        "extra_report_block": {"deep": [1]},
    })
    result = parse_accountability(raw)
    assert result.accountability is not None
    assert result.accountability.overall_status is AccountabilityStatus.CLEAN
    assert result.accountability.flags[0].refs == ["3"]
