"""Pure helpers over claim/verdict structures: rendering and mechanical rules.

No I/O, no agent imports — `app.claims` depends only on `app.schemas`.
These functions encode epistemic rules in code instead of trusting prompts
(spec §11; research finding: mechanical checks beat prompt instructions).
"""

from __future__ import annotations

from collections.abc import Sequence

from app.schemas import Claim, ClaimStatus, ClaimVerdict, Verdict

__all__ = [
    "render_claims",
    "unresolved_claim_ids",
    "validate_claims",
    "validate_verdicts",
]


def render_claims(claims: Sequence[Claim]) -> str:
    """Deterministic prompt rendering of claim records (golden-tested).

    The Skeptic must receive records, not persuasive prose. Empty sequence
    renders the empty string; blocks are separated by one blank line with no
    trailing blank line.
    """
    blocks = []
    for claim in claims:
        lines = [
            f"[{claim.id}] status={claim.status.value} "
            f"confidence={claim.confidence if claim.confidence is not None else 'none'}",
            f"Claim: {claim.statement}",
        ]
        if claim.evidence:
            lines.append("Evidence:")
            for item in claim.evidence:
                lines.append(f"- source={item.source}")
                if item.quote is not None:
                    lines.append(f'  quote="{item.quote}"')
        else:
            lines.append("Evidence: none")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def validate_claims(claims: Sequence[Claim]) -> list[str]:
    """Mechanical rules on a claim sequence. Empty result = clean."""
    violations: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        if claim.id in seen:
            violations.append(f"duplicate claim id {claim.id}")
        seen.add(claim.id)
        if claim.status is ClaimStatus.FACT and not claim.evidence:
            violations.append(f"claim {claim.id} has status=fact but no evidence")
    return violations


def validate_verdicts(
    claims: Sequence[Claim], verdicts: Sequence[Verdict]
) -> list[str]:
    """Mechanical adversarial contract: coverage, references, evidence rules."""
    if verdicts and not claims:
        return ["verdicts supplied with no claims to evaluate"]

    violations: list[str] = []
    claim_map = {c.id: c for c in claims}
    seen: set[str] = set()

    for verdict in verdicts:
        if verdict.claim_id in seen:
            violations.append(f"claim {verdict.claim_id} evaluated more than once")
            continue
        seen.add(verdict.claim_id)
        if not verdict.objection.strip():
            violations.append(f"verdict for {verdict.claim_id} has no objection")
        claim = claim_map.get(verdict.claim_id)
        if claim is None:
            violations.append(f"verdict for unknown claim {verdict.claim_id}")
            continue
        # Check either claim.evidence OR verdict.evidence (both are valid)
        has_evidence = bool(claim.evidence) or bool(verdict.evidence)
        if verdict.verdict is ClaimVerdict.SUPPORTED and not has_evidence:
            violations.append(f"claim {verdict.claim_id} marked supported without evidence")

    for claim in claims:
        if claim.id not in seen:
            violations.append(f"claim {claim.id} was not evaluated")
    return violations


def unresolved_claim_ids(
    claims: Sequence[Claim], verdicts: Sequence[Verdict]
) -> list[str]:
    """Claims not resolved by SUPPORTED: no verdict, refuted, or unverifiable.

    Preserves claim order. This is the primitive for Phase 5's stop signal.
    """
    by_id = {v.claim_id: v for v in verdicts}
    return [
        claim.id
        for claim in claims
        if claim.id not in by_id
        or by_id[claim.id].verdict is not ClaimVerdict.SUPPORTED
    ]
