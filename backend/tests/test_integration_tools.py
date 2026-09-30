"""Live tool tests (Phase 6 PRD §9): tool trace structure + planted refutation.

Most integration assertions are structure-only per repo policy. The planted
scenario carries one intentional *semantic* assertion (`refuted`): the corpus
is deterministic and the phase exit criterion IS that semantic (PRD §9).
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.agents import build_registry
from app.config import Settings, get_settings
from app.llm import get_provider
from app.schemas import (
    AgentRole,
    Claim,
    ClaimStatus,
    ClaimVerdict,
)
from app.tools import build_tool_registry

pytestmark = pytest.mark.integration

OLLAMA_URL = "http://localhost:11434"

TASK = "Is the claimed 2025 annual growth rate of 40% correct? Check the documents."
PLANTED_CLAIM = "The 2025 annual growth rate was 40%."


def _ollama_reachable() -> bool:
    try:
        httpx.get(f"{OLLAMA_URL}/api/version", timeout=2.0).raise_for_status()
        return True
    except Exception:
        return False


if not _ollama_reachable():
    pytest.skip("Ollama is not running; skipping integration suite", allow_module_level=True)


@pytest.fixture
def refute_corpus(tmp_path):
    (tmp_path / "growth_report.txt").write_text(
        "2025 Annual Growth Report\n"
        "Annual growth was 23% in 2025, not 40%.\n"
        "CEO note: figures unaudited.\n"
    )
    return tmp_path


async def test_live_skeptic_uses_file_tool_and_refutes(refute_corpus):
    settings = Settings(tool_files_root=str(refute_corpus))
    provider = get_provider(settings)
    try:
        registry = build_registry(provider, tools=build_tool_registry(settings))
        skeptic = registry.get(AgentRole.SKEPTIC)
        message = await skeptic.run(
            TASK,
            claims=[
                Claim(
                    id="c1", statement=PLANTED_CLAIM,
                    status=ClaimStatus.UNVERIFIED,
                )
            ],
            run_id="integration-refute",
        )
    finally:
        await provider.aclose()

    # structure: the skeptic went to the corpus itself
    assert message.tool_results, "skeptic must execute at least one tool"
    executed = {r.name for r in message.tool_results}
    assert executed & {"file_search", "file_reader"}, f"unexpected tools: {executed}"
    assert message.tool_calls is not None
    assert len(message.tool_calls) == len(message.tool_results)

    # structure: verdicts cover the input claim
    assert message.verdicts, "skeptic must return verdicts"
    verdict = next((v for v in message.verdicts if v.claim_id == "c1"), None)
    assert verdict is not None, "c1 was not evaluated"

    # THE planted-scenario assertion (sanctioned semantic check, PRD §9)
    assert verdict.verdict is ClaimVerdict.REFUTED, (
        f"expected refuted, got {verdict.verdict.value}: {verdict.objection}"
    )
    assert any(
        "growth_report" in evidence.source for evidence in verdict.evidence
    ), "refutation must cite the corpus file as evidence"


async def test_live_web_search_returns_results():
    if not get_settings().tool_web_search_enabled:
        pytest.skip("AI_NEXUS_TOOL_WEB_SEARCH_ENABLED is not set")

    provider = get_provider(Settings())
    try:
        registry = build_registry(provider, tools=build_tool_registry(get_settings()))
        researcher = registry.get(AgentRole.RESEARCHER)
        message = await researcher.run(
            "What is the current population of Canada? Use your tools.",
        )
    finally:
        await provider.aclose()

    assert message.tool_results, "web search tool should have been used"
    result = message.tool_results[0]
    if result.error is not None:
        # provider_blocked/network errors are graceful outcomes, not failures
        assert result.error in {"provider_blocked", "network_error"}
        return
    payload = json.loads(result.content)
    assert payload["ok"] is True
    assert isinstance(payload["results"], list)
    for entry in payload["results"]:
        assert entry["title"] and entry["url"]
        assert "snippet" in entry
