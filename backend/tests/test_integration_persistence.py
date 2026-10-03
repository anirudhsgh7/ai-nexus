"""Live persistence: a real Ollama run survives a store-level restart (PRD §9).

Structure-only assertions per repo policy. The run is capped at one round to
stay within the integration time budget.
"""

from __future__ import annotations

import httpx
import pytest

from app.agents import build_registry
from app.config import Settings
from app.db import RunStore
from app.llm import get_provider
from app.orchestrator import Orchestrator
from app.runs import RunEventType, RunManager, RunStatus

pytestmark = pytest.mark.integration

OLLAMA_URL = "http://localhost:11434"

TASK = "What is 2 + 2? Answer briefly and list any assumptions as claims."


def _ollama_reachable() -> bool:
    try:
        httpx.get(f"{OLLAMA_URL}/api/version", timeout=2.0).raise_for_status()
        return True
    except Exception:
        return False


if not _ollama_reachable():
    pytest.skip("Ollama is not running; skipping integration suite", allow_module_level=True)


async def test_live_run_survives_restart(tmp_path):
    path = tmp_path / "live.db"
    settings = Settings()
    provider = get_provider(settings)
    manager = RunManager(store=RunStore(path))
    run = manager.create(TASK)
    try:
        orchestrator = Orchestrator(
            build_registry(provider), manager, max_rounds=1
        )
        await orchestrator.execute(run.id)
    finally:
        await provider.aclose()

    assert run.status is RunStatus.COMPLETED
    step_kinds = [step.kind.value for step in run.steps]
    assert step_kinds[0] == "plan"
    # Phase 11: completed runs always end synthesize -> verify -> audit
    assert step_kinds[-3:] == ["synthesize", "verify", "audit"]
    assert run.final_message is not None
    assert run.final_message.content.strip()
    assert run.rounds, "round 1 snapshot must be recorded"

    # ---- the restart: a brand-new manager over the same database file ----
    restarted = RunManager(store=RunStore(path))
    loaded = restarted.get(run.id)

    assert loaded is not None
    assert loaded.status is RunStatus.COMPLETED
    assert loaded.task == TASK
    assert [step.kind.value for step in loaded.steps] == step_kinds
    assert len(loaded.events) == len(run.events)
    assert [e.seq for e in loaded.events] == [e.seq for e in run.events]
    assert loaded.events[-1].type is RunEventType.RUN_COMPLETED
    assert loaded.final_message.content == run.final_message.content
    assert len(loaded.rounds) == len(run.rounds)
    assert loaded.rounds[0].supported_count == run.rounds[0].supported_count
    assert loaded.rounds[0].unresolved_count == run.rounds[0].unresolved_count
    assert loaded.duration_ms is not None
