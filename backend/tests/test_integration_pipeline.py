"""One full live pipeline run (Phase 5 PRD §9). Contract-level only.

Runtime ≈ 5–10 min on the M4/14B. Iteration depends on what the live Manager
decides (not asserted); the deterministic iteration proof is the scripted
unit tests in test_orchestrator.py.
"""

import httpx
import pytest

from app.agents import build_registry
from app.config import Settings
from app.llm import OllamaProvider
from app.orchestrator import Orchestrator
from app.runs import RunEventType, RunManager, RunStatus, StepKind, StepStatus
from app.schemas import MessageType

pytestmark = pytest.mark.integration

OLLAMA_URL = "http://localhost:11434"
TASK = "Verify the exact 2026 revenue figure reported for Acme Corp."


def _ollama_reachable() -> bool:
    try:
        httpx.get(f"{OLLAMA_URL}/api/version", timeout=2.0).raise_for_status()
        return True
    except Exception:
        return False


if not _ollama_reachable():
    pytest.skip("Ollama is not running; skipping integration suite", allow_module_level=True)


async def test_full_iterative_pipeline_completes():
    provider = OllamaProvider(Settings())
    store = RunManager()
    run = store.create(TASK)
    try:
        orchestrator = Orchestrator(build_registry(provider), store, max_rounds=3)
        await orchestrator.execute(run.id)
    finally:
        await provider.aclose()

    assert run.status is RunStatus.COMPLETED, (
        f"run failed: {run.error.type if run.error else '?'}: "
        f"{run.error.message if run.error else ''}"
    )
    assert run.final_message is not None
    assert run.final_message.type is MessageType.SYNTHESIS
    assert run.final_message.content.strip()
    assert run.duration_ms is not None

    # ---- structural trace (Phase 5 PRD §9 + Phase 11 audits) ----
    kinds = [s.kind for s in run.steps]
    assert kinds[:4] == [
        StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE, StepKind.CRITIQUE,
    ]
    # every completed run now ends synthesize -> verify -> audit
    assert kinds[-3:] == [StepKind.SYNTHESIZE, StepKind.VERIFY, StepKind.AUDIT]
    # both audit steps must have actually run (never skipped) — §9.2
    assert run.steps[-2].status is StepStatus.COMPLETED
    assert run.steps[-1].status is StepStatus.COMPLETED
    assert set(kinds) <= {
        StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE, StepKind.CRITIQUE,
        StepKind.DECIDE, StepKind.REVISE, StepKind.SYNTHESIZE,
        StepKind.VERIFY, StepKind.AUDIT,
    }
    for step in run.steps:
        assert step.status in {StepStatus.COMPLETED, StepStatus.SKIPPED}
        if step.status is StepStatus.COMPLETED:
            assert step.message is not None
            if step.kind is StepKind.DECIDE:
                # decisions carry no prose — content is legitimately empty
                assert step.message.decision is not None
            elif step.kind is StepKind.SYNTHESIZE:
                assert step.message.content.strip()
            else:
                # a completed work step must carry substance somewhere:
                # prose, claims, verdicts, or an audit report (empty-only
                # output would have raised EmptyAgentResponseError in Agent.run)
                assert (
                    step.message.content.strip()
                    or step.message.claims
                    or step.message.verdicts
                    or step.message.verification
                    or step.message.accountability
                ), f"{step.kind.value} step {step.index} had no substance"

    # ---- Phase 11: audits ran, cover the final claims, stay consistent ----
    from app.orchestrator import select_best_round

    selected = select_best_round(run.rounds)
    verify_message = run.steps[-2].message
    assert verify_message is not None
    assert verify_message.verification is not None
    assert sorted(
        entry.claim_id for entry in verify_message.verification.claims
    ) == sorted(claim.id for claim in selected.claims)
    audit_message = run.steps[-1].message
    assert audit_message is not None
    assert audit_message.accountability is not None
    # enforcement makes provenance code-canonical: it matches the selected
    # round exactly regardless of what the model produced
    assert [
        provenance.claim_id
        for provenance in audit_message.accountability.final_claim_provenance
    ] == [claim.id for claim in selected.claims]
    assert audit_message.accountability.trace_completeness is True
    # a healthy run must not audit itself into violations (§9.2)
    from app.schemas import AccountabilityStatus

    assert audit_message.accountability.overall_status is not (
        AccountabilityStatus.VIOLATIONS
    )

    # round numbering: round-tagged steps strictly increase; plan/synth untagged
    assert run.steps[0].round is None and run.steps[-1].round is None
    round_tags = [s.round for s in run.steps if s.round is not None]
    assert all(b >= a for a, b in zip(round_tags, round_tags[1:])), round_tags
    assert max(round_tags or [1]) <= 3

    # every round-2+ critique is preceded by a revise step
    for i, step in enumerate(run.steps):
        if step.kind is StepKind.CRITIQUE and step.round and step.round > 1:
            assert run.steps[i - 1].kind is StepKind.REVISE

    # ---- rounds state ----
    assert len(run.rounds) >= 1
    assert [r.round_number for r in run.rounds] == list(
        range(1, len(run.rounds) + 1)
    )
    for snapshot in run.rounds:
        assert snapshot.supported_count + snapshot.unresolved_count == len(
            snapshot.claims
        )

    # ---- well-formed decisions on decide steps ----
    for step in run.steps:
        if step.kind is StepKind.DECIDE and step.message is not None:
            decision = step.message.decision
            assert decision is not None
            assert decision.reason.strip()

    # ---- if iteration happened, first critique must have covered the
    # round-1 pool exactly (pool is the canonical qualified+deduped set;
    # deriving it again from raw messages would duplicate orchestrator logic
    # and diverge on cross-agent duplicate statements) ----
    critique_steps = [s for s in run.steps if s.kind is StepKind.CRITIQUE]
    assert critique_steps, "round-1 critique must exist"
    if critique_steps[0].message is not None:
        pool_ids = [c.id for c in run.rounds[0].claims]
        verdict_ids = [
            v.claim_id for v in critique_steps[0].message.verdicts or []
        ]
        assert sorted(verdict_ids) == sorted(pool_ids), (
            f"first critique verdicts must cover the round-1 pool exactly "
            f"(pool={pool_ids}, verdicts={verdict_ids})"
        )

    assert run.events[-1].type is RunEventType.RUN_COMPLETED
    assert run.events[-1].message is not None
