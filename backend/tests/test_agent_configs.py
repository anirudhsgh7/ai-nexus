"""Role config lint (PRD §6.4): exactly four, table-locked, directive keywords."""

import pytest

from app.agents import ideator, manager, researcher, skeptic
from app.agents.base import MAX_INSTRUCTION_CHARS
from app.schemas import AgentRole, MessageType

CONFIGS = {
    AgentRole.MANAGER: manager.CONFIG,
    AgentRole.RESEARCHER: researcher.CONFIG,
    AgentRole.IDEATOR: ideator.CONFIG,
    AgentRole.SKEPTIC: skeptic.CONFIG,
}

# Case-insensitive substring sets; any listed variant must appear.
DIRECTIVE_KEYWORDS = {
    AgentRole.MANAGER: ["objective", "subproblem", "agent", "task"],
    AgentRole.RESEARCHER: ["evidence", "assumption", "perspectiv", "uncertain"],
    AgentRole.IDEATOR: ["alternative", "assumption", "reason", "converg"],
    AgentRole.SKEPTIC: ["challenge", "unsupported", "contradict", "missing", "independ", "disprov"],
}

TEMPERATURES = {
    AgentRole.MANAGER: 0.0,
    AgentRole.RESEARCHER: 0.2,
    AgentRole.IDEATOR: 0.4,
    AgentRole.SKEPTIC: 0.0,
}

OUTPUT_TYPES = {
    AgentRole.MANAGER: MessageType.PLAN,
    AgentRole.RESEARCHER: MessageType.FINDING,
    AgentRole.IDEATOR: MessageType.IDEA,
    AgentRole.SKEPTIC: MessageType.CRITIQUE,
}


def test_exactly_four_unique_roles():
    assert set(CONFIGS) == set(AgentRole)
    assert len({c.role for c in CONFIGS.values()}) == 4


def test_temperature_table():
    for role, expected in TEMPERATURES.items():
        assert CONFIGS[role].temperature == expected, role


def test_output_type_mapping():
    for role, expected in OUTPUT_TYPES.items():
        assert CONFIGS[role].output_type is expected, role


def test_capabilities_empty_in_phase2():
    for role, cfg in CONFIGS.items():
        assert cfg.capabilities == frozenset(), role


def test_directive_keywords_present():
    for role, cfg in CONFIGS.items():
        instructions = cfg.instructions.lower()
        for keyword in DIRECTIVE_KEYWORDS[role]:
            assert keyword in instructions, f"{role.value}: missing directive keyword {keyword!r}"


def test_no_format_mandates():
    """Phase 3 owns output formats; Phase 2 prompts must not mandate them."""
    for role, cfg in CONFIGS.items():
        assert "json" not in cfg.instructions.lower(), role
        assert "yaml" not in cfg.instructions.lower(), role


def test_role_identity_stated():
    for role, cfg in CONFIGS.items():
        assert cfg.display_name.lower() in cfg.instructions.lower(), role


def test_must_not_list_present():
    for role, cfg in CONFIGS.items():
        assert "do not" in cfg.instructions.lower(), role


def test_instruction_length_budget():
    for role, cfg in CONFIGS.items():
        assert 100 <= len(cfg.instructions) <= MAX_INSTRUCTION_CHARS, role


def test_prompt_version_and_display_names():
    assert [CONFIGS[r].prompt_version for r in AgentRole] == [1, 1, 1, 1]
    assert {c.display_name for c in CONFIGS.values()} == {
        "Manager", "Researcher", "Ideator", "Skeptic",
    }


@pytest.mark.parametrize("role", list(AgentRole))
def test_role_defaults_to_primary_model(role):
    assert CONFIGS[role].model is None
    assert CONFIGS[role].max_tokens is None
