"""Registry contract (PRD §6.5), including the future-agent extension proof.

Extension recipe (amended — see PHASE_2_PRD.md §6.5): a new agent needs one
enum member in app/schemas.py, one config module, and one registry entry.
These tests prove the registry half works through public API only.
"""

import pytest

from app.agents import (
    Agent,
    AgentConfig,
    AgentRegistry,
    AgentNotRegisteredError,
    DEFAULT_AGENT_CONFIGS,
    build_registry,
)
from app.schemas import AgentRole, MessageType
from tests.fakes import FakeProvider


def test_build_registry_canonical_order():
    registry = build_registry(FakeProvider())
    assert len(registry) == 6
    assert registry.roles == [
        AgentRole.MANAGER,
        AgentRole.RESEARCHER,
        AgentRole.IDEATOR,
        AgentRole.SKEPTIC,
        AgentRole.VERIFIER,
        AgentRole.ACCOUNTABILITY,
    ]


def test_get_every_role():
    registry = build_registry(FakeProvider())
    for role in AgentRole:
        agent = registry.get(role)
        assert agent.role is role


def test_unregistered_role_raises_with_registered_list():
    configs = [c for c in DEFAULT_AGENT_CONFIGS if c.role is not AgentRole.SKEPTIC]
    registry = build_registry(FakeProvider(), configs=configs)
    with pytest.raises(AgentNotRegisteredError) as exc_info:
        registry.get(AgentRole.SKEPTIC)
    assert "skeptic" in str(exc_info.value)
    assert "registered: manager, researcher, ideator" in str(exc_info.value)


def test_duplicate_role_raises_value_error():
    registry = build_registry(FakeProvider())
    with pytest.raises(ValueError, match="duplicate agent role"):
        AgentRegistry([*registry, next(iter(registry))])


def test_contains_iter_len():
    registry = build_registry(FakeProvider())
    assert AgentRole.SKEPTIC in registry
    assert "skeptic" in registry  # str-enum keys match plain strings too
    assert "nonexistent" not in registry
    assert {a.role for a in registry} == set(AgentRole)


def test_out_of_band_config_injects_via_public_api():
    """Extension proof: a config that is NOT in DEFAULT_AGENT_CONFIGS is
    accepted and resolved without touching registry code."""
    foreign = AgentConfig(
        role=AgentRole.SKEPTIC,
        display_name="Skeptic-2",
        instructions="Independently verify: challenge unsupported claims.",
        temperature=0.0,
        output_type=MessageType.CRITIQUE,
    )
    configs = [*DEFAULT_AGENT_CONFIGS[:3], foreign]
    registry = build_registry(FakeProvider(), configs=configs)
    assert len(registry) == 4
    assert registry.get(AgentRole.SKEPTIC).display_name == "Skeptic-2"
    # default path unchanged
    assert build_registry(FakeProvider()).get(AgentRole.SKEPTIC).display_name == "Skeptic"


def test_agents_share_injected_provider():
    provider = FakeProvider()
    registry = build_registry(provider)
    for agent in registry:
        assert isinstance(agent, Agent)
        assert agent._provider is provider


def test_default_configs_untouched():
    assert len(DEFAULT_AGENT_CONFIGS) == 6
    assert [c.role for c in DEFAULT_AGENT_CONFIGS] == list(AgentRole)


def test_build_registry_subset():
    only_manager = [c for c in DEFAULT_AGENT_CONFIGS if c.role is AgentRole.MANAGER]
    registry = build_registry(FakeProvider(), configs=only_manager)
    assert len(registry) == 1
    assert registry.roles == [AgentRole.MANAGER]
