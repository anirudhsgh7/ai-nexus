"""Claims helpers: golden rendering + mechanical epistemic rules (PRD §6.2)."""

from app.claims import (
    render_claims,
    unresolved_claim_ids,
    validate_claims,
    validate_verdicts,
)
from app.schemas import Claim, ClaimStatus, ClaimVerdict, Evidence, Verdict


def _claim(cid: str = "c1", statement: str = "X grew 40% in 2025.", **kw) -> Claim:
    return Claim(id=cid, statement=statement, **kw)


def _verdict(cid: str, verdict: ClaimVerdict, objection: str = "why", **kw) -> Verdict:
    return Verdict(claim_id=cid, verdict=verdict, objection=objection, **kw)


# ---------------------------------------------------------------- render

def test_render_empty_sequence():
    assert render_claims([]) == ""


def test_render_single_claim_no_evidence_golden():
    claim = _claim(confidence=0.55)
    assert render_claims([claim]) == (
        "[c1] status=unverified confidence=0.55\n"
        "Claim: X grew 40% in 2025.\n"
        "Evidence: none"
    )


def test_render_evidence_with_and_without_quote_golden():
    claim = _claim(
        status=ClaimStatus.FACT,
        evidence=[
            Evidence(source="annual report", quote="revenue up 40%"),
            Evidence(source="analyst call"),
        ],
    )
    assert render_claims([claim]) == (
        "[c1] status=fact confidence=none\n"
        "Claim: X grew 40% in 2025.\n"
        "Evidence:\n"
        '- source=annual report\n'
        '  quote="revenue up 40%"\n'
        "- source=analyst call"
    )


def test_render_two_claims_blank_line_between_no_trailing():
    out = render_claims([_claim("c1", "first"), _claim("c2", "second")])
    assert out == (
        "[c1] status=unverified confidence=none\n"
        "Claim: first\n"
        "Evidence: none\n"
        "\n"
        "[c2] status=unverified confidence=none\n"
        "Claim: second\n"
        "Evidence: none"
    )
    assert not out.endswith("\n")


def test_render_confidence_zero_included():
    assert "confidence=0.0" in render_claims([_claim(confidence=0.0)])


# ---------------------------------------------------------------- validate_claims

def test_validate_claims_clean():
    assert validate_claims([
        _claim(status=ClaimStatus.ASSUMPTION),
        _claim("c2", status=ClaimStatus.FACT, evidence=[Evidence(source="s")]),
    ]) == []


def test_validate_claims_fact_without_evidence():
    violations = validate_claims([_claim(status=ClaimStatus.FACT)])
    assert violations == ["claim c1 has status=fact but no evidence"]


def test_validate_claims_duplicate_ids():
    violations = validate_claims([_claim("c1", "a"), _claim("c1", "b")])
    assert violations == ["duplicate claim id c1"]


def test_validate_claims_unverified_needs_no_evidence():
    assert validate_claims([_claim(status=ClaimStatus.UNVERIFIED)]) == []


# ---------------------------------------------------------------- validate_verdicts

def test_verdicts_clean_case():
    claims = [
        _claim("c1", evidence=[Evidence(source="s")]),
        _claim("c2"),
    ]
    verdicts = [
        _verdict("c1", ClaimVerdict.SUPPORTED),
        _verdict("c2", ClaimVerdict.UNVERIFIABLE, "no source"),
    ]
    assert validate_verdicts(claims, verdicts) == []


def test_verdict_unknown_claim():
    claims = [_claim("c1")]
    verdicts = [_verdict("c1", ClaimVerdict.REFUTED), _verdict("c9", ClaimVerdict.REFUTED)]
    assert "verdict for unknown claim c9" in validate_verdicts(claims, verdicts)


def test_verdict_missing_claim():
    claims = [_claim("c1"), _claim("c2")]
    verdicts = [_verdict("c1", ClaimVerdict.REFUTED)]
    assert validate_verdicts(claims, verdicts) == ["claim c2 was not evaluated"]


def test_verdict_duplicate_evaluation():
    claims = [_claim("c1")]
    verdicts = [
        _verdict("c1", ClaimVerdict.REFUTED),
        _verdict("c1", ClaimVerdict.SUPPORTED),
    ]
    assert validate_verdicts(claims, verdicts) == ["claim c1 evaluated more than once"]


def test_verdict_supported_without_evidence():
    claims = [_claim("c1")]  # no evidence
    verdicts = [_verdict("c1", ClaimVerdict.SUPPORTED, "looks fine")]
    assert validate_verdicts(claims, verdicts) == [
        "claim c1 marked supported without evidence"
    ]


def test_verdict_refuted_without_evidence_is_fine():
    claims = [_claim("c1")]
    verdicts = [_verdict("c1", ClaimVerdict.REFUTED)]
    assert validate_verdicts(claims, verdicts) == []


def test_verdicts_without_claims():
    verdicts = [_verdict("c1", ClaimVerdict.REFUTED)]
    assert validate_verdicts([], verdicts) == [
        "verdicts supplied with no claims to evaluate"
    ]


def test_defensive_blank_objection():
    # Pydantic rejects blank objections at construction; the helper still
    # checks (PRD §6.2) for callers passing schema-bypassed objects.
    crafted = Verdict.model_construct(
        claim_id="c1", verdict=ClaimVerdict.REFUTED, objection="  ", evidence=[]
    )
    claims = [_claim("c1")]
    assert validate_verdicts(claims, [crafted]) == ["verdict for c1 has no objection"]


# ---------------------------------------------------------------- unresolved

def test_unresolved_matrix():
    claims = [_claim("c1"), _claim("c2"), _claim("c3"), _claim("c4")]
    verdicts = [
        _verdict("c1", ClaimVerdict.SUPPORTED),
        _verdict("c2", ClaimVerdict.REFUTED),
        _verdict("c3", ClaimVerdict.UNVERIFIABLE),
        # c4: no verdict
    ]
    assert unresolved_claim_ids(claims, verdicts) == ["c2", "c3", "c4"]


def test_unresolved_empty_inputs():
    assert unresolved_claim_ids([], []) == []
    assert unresolved_claim_ids([_claim("c1")], []) == ["c1"]


def test_unresolved_only_supported_resolves():
    claims = [_claim("c1")]
    assert unresolved_claim_ids(claims, [_verdict("c1", ClaimVerdict.SUPPORTED)]) == []
