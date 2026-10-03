"""Contract-level integration tests: structured outputs on live Ollama.

Deliberately asserts NO semantic choice (which verdict the model made) —
prompt overfitting is a real risk; role quality is reviewed via the demo.
Structural coverage (ids, enums, objection presence) IS asserted: the
retry-once validation loop makes it deterministic, not luck (PRD §9).
"""

import httpx
import pytest

from app.agents import DEFAULT_AGENT_CONFIGS, build_registry
from app.agents.errors import StructuredOutputError
from app.config import Settings
from app.llm import LLMError, OllamaProvider
from app.schemas import (
    AgentRole,
    Claim,
    ClaimStatus,
    ClaimVerdict,
    Evidence,
    MessageType,
)

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

CRAFTED_CLAIMS = [
    Claim(
        id="c1",
        statement="Water boils at 100 degrees celsius at standard sea-level pressure.",
        status=ClaimStatus.UNVERIFIED,
        confidence=0.9,
        evidence=[Evidence(source="general chemistry textbook convention")],
    ),
    Claim(
        id="c2",
        statement="The team in question uses AI-assisted code review today.",
        status=ClaimStatus.UNVERIFIED,
        confidence=0.4,
        evidence=[],
    ),
]


@pytest.fixture
async def live_registry():
    provider = OllamaProvider(Settings())
    try:
        yield build_registry(provider)
    finally:
        await provider.aclose()


def _common_contract(message, role):
    assert message.from_agent is role
    assert message.type is EXPECTED_TYPES[role]
    assert message.to_agent is None
    assert message.content.strip(), f"{role.value} returned empty content"
    assert message.confidence is None
    assert message.round is None
    assert message.created_at.tzinfo is not None


@pytest.mark.parametrize(
    "role", [AgentRole.MANAGER, AgentRole.RESEARCHER, AgentRole.IDEATOR]
)
async def test_claims_kind_agents_emit_structured_messages(live_registry, role):
    agent = live_registry.get(role)
    try:
        message = await agent.run(TASK)
    except (LLMError, StructuredOutputError) as exc:
        pytest.fail(f"{role.value} run failed: {type(exc).__name__}: {exc}")

    _common_contract(message, role)
    assert message.claims is not None, f"{role.value} must emit a claims list"
    assert message.verdicts is None
    assert all(c.id for c in message.claims)
    if role is AgentRole.RESEARCHER:
        assert len(message.claims) >= 1, "researcher must surface at least one claim"


async def test_skeptic_verdicts_reference_supplied_claims(live_registry):
    agent = live_registry.get(AgentRole.SKEPTIC)
    try:
        message = await agent.run(
            "Evaluate each claim under CLAIMS TO EVALUATE.",
            claims=CRAFTED_CLAIMS,
        )
    except StructuredOutputError as exc:
        pytest.fail(f"skeptic structured output never converged: {exc.last_error}")
    except LLMError as exc:
        pytest.fail(f"skeptic run failed: {type(exc).__name__}: {exc}")

    _common_contract(message, AgentRole.SKEPTIC)
    assert message.verdicts is not None
    assert message.claims is None

    supplied_ids = [c.id for c in CRAFTED_CLAIMS]
    verdict_ids = [v.claim_id for v in message.verdicts]
    assert sorted(verdict_ids) == sorted(supplied_ids), (
        "every supplied claim must be addressed exactly once "
        f"(supplied={supplied_ids}, got={verdict_ids})"
    )
    assert len(verdict_ids) == len(set(verdict_ids)), "duplicate evaluations"
    for verdict in message.verdicts:
        assert verdict.verdict in set(ClaimVerdict)
        assert verdict.objection.strip(), f"{verdict.claim_id} has empty objection"
        assert verdict.claim_id in supplied_ids


async def test_registry_default_configs_match_agents(live_registry):
    # Phase 11: six default configs (four workers + verifier + accountability)
    assert len(live_registry) == len(DEFAULT_AGENT_CONFIGS) == 6
    for config in DEFAULT_AGENT_CONFIGS:
        agent = live_registry.get(config.role)
        assert agent.config is config
