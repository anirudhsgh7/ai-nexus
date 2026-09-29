"""One full live pipeline run (PRD §7). Contract-level assertions only.

Runtime ≈ 5–10 min on the M4/14B; this is the only new live test in Phase 4.
"""

import httpx
import pytest

from app.agents import build_registry
from app.config import Settings
from app.llm import OllamaProvider
from app.orchestrator import PIPELINE, Orchestrator, qualify_claims
from app.runs import RunEventType, RunManager, RunStatus, StepKind, StepStatus
from app.schemas import AgentRole, MessageType

pytestmark = pytest.mark.integration

OLLAMA_URL = "http://localhost:11434"
TASK = "In one sentence: should a tiny team write down its decisions?"


def _ollama_reachable() -> bool:
    try:
        httpx.get(f"{OLLAMA_URL}/api/version", timeout=2.0).raise_for_status()
        return True
    except Exception:
        return False


if not _ollama_reachable():
    pytest.skip("Ollama is not running; skipping integration suite", allow_module_level=True)


async def test_full_pipeline_completes_with_structured_trace():
    provider = OllamaProvider(Settings())
    store = RunManager()
    run = store.create(TASK)
    try:
        orchestrator = Orchestrator(build_registry(provider), store)
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

    assert [s.kind for s in run.steps] == [k for k, _ in PIPELINE]
    for step in run.steps:
        assert step.status in {StepStatus.COMPLETED, StepStatus.SKIPPED}
        if step.status is StepStatus.COMPLETED and step.kind is not StepKind.SYNTHESIZE:
            assert step.message is not None and step.message.content.strip()

    research = next(s for s in run.steps if s.kind is StepKind.RESEARCH)
    ideation = next(s for s in run.steps if s.kind is StepKind.IDEATE)
    assert research.message is not None and ideation.message is not None

    critique = next(s for s in run.steps if s.kind is StepKind.CRITIQUE)
    if critique.skipped:
        assert critique.message is None
    else:
        qualified = qualify_claims([
            (AgentRole.RESEARCHER, research.message.claims or []),
            (AgentRole.IDEATOR, ideation.message.claims or []),
        ])
        assert critique.message is not None
        verdict_ids = [v.claim_id for v in critique.message.verdicts or []]
        assert sorted(verdict_ids) == sorted(q.claim.id for q in qualified), (
            "skeptic must evaluate every qualified claim exactly once "
            f"(qualified={[q.claim.id for q in qualified]}, got={verdict_ids})"
        )

    assert run.events[-1].type is RunEventType.RUN_COMPLETED
    assert run.events[-1].message is not None
