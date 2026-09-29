"""Contract-level integration tests: all four role configs run on live Ollama.

Deliberately asserts NO semantic content (PRD §12) — prompt overfitting is a
real risk on local models; role quality is reviewed manually via the demo.
"""

import httpx
import pytest

from app.agents import DEFAULT_AGENT_CONFIGS, build_registry
from app.config import Settings
from app.llm import LLMError, OllamaProvider
from app.schemas import AgentRole, MessageType

pytestmark = pytest.mark.integration

OLLAMA_URL = "http://localhost:11434"

TASK = "In one short paragraph: should a tiny team track its work in writing?"


def _ollama_reachable() -> bool:
    try:
        httpx.get(f"{OLLAMA_URL}/api/version", timeout=2.0).raise_for_status()
        return True
    except Exception:
        return False


if not _ollama_reachable():
    pytest.skip("Ollama is not running; skipping integration suite", allow_module_level=True)

EXPECTED_TYPES = {
    AgentRole.MANAGER: MessageType.PLAN,
    AgentRole.RESEARCHER: MessageType.FINDING,
    AgentRole.IDEATOR: MessageType.IDEA,
    AgentRole.SKEPTIC: MessageType.CRITIQUE,
}


@pytest.fixture
async def live_registry():
    provider = OllamaProvider(Settings())
    try:
        yield build_registry(provider)
    finally:
        await provider.aclose()


@pytest.mark.parametrize("role", list(AgentRole))
async def test_each_agent_runs_and_returns_contract_message(live_registry, role):
    agent = live_registry.get(role)
    try:
        message = await agent.run(TASK)
    except LLMError as exc:
        pytest.fail(f"{role.value} run failed: {type(exc).__name__}: {exc}")

    assert message.from_agent is role
    assert message.type is EXPECTED_TYPES[role]
    assert message.to_agent is None
    assert message.content.strip(), f"{role.value} returned empty content"
    assert message.claims is None
    assert message.confidence is None
    assert message.round is None
    assert message.created_at.tzinfo is not None


async def test_registry_default_configs_match_agents(live_registry):
    assert len(live_registry) == len(DEFAULT_AGENT_CONFIGS) == 4
    for config in DEFAULT_AGENT_CONFIGS:
        agent = live_registry.get(config.role)
        assert agent.config is config
