"""Role config lint (PRD §6.4/Phase 11 §6.8): six roles, table-locked, directive keywords."""

import pytest

from app.agents import accountability, ideator, manager, researcher, skeptic, verifier
from app.agents.base import MAX_INSTRUCTION_CHARS
from app.agents.structured import OutputKind
from app.schemas import AgentRole, MessageType

CONFIGS = {
    AgentRole.MANAGER: manager.CONFIG,
    AgentRole.RESEARCHER: researcher.CONFIG,
    AgentRole.IDEATOR: ideator.CONFIG,
    AgentRole.SKEPTIC: skeptic.CONFIG,
    AgentRole.VERIFIER: verifier.CONFIG,
    AgentRole.ACCOUNTABILITY: accountability.CONFIG,
}

# Case-insensitive substring sets; any listed variant must appear.
DIRECTIVE_KEYWORDS = {
    AgentRole.MANAGER: ["objective", "subproblem", "agent", "task"],
    AgentRole.RESEARCHER: ["evidence", "assumption", "perspectiv", "uncertain"],
    AgentRole.IDEATOR: ["alternative", "assumption", "reason", "converg"],
    AgentRole.SKEPTIC: ["challenge", "unsupported", "contradict", "missing", "independ", "disprov"],
    AgentRole.VERIFIER: ["verify", "independent", "source", "contradict", "evidence"],
    AgentRole.ACCOUNTABILITY: ["trace", "provenance", "flag", "unresolved", "decision"],
}

TEMPERATURES = {
    AgentRole.MANAGER: 0.0,
    AgentRole.RESEARCHER: 0.2,
    AgentRole.IDEATOR: 0.4,
    AgentRole.SKEPTIC: 0.0,
    AgentRole.VERIFIER: 0.0,
    AgentRole.ACCOUNTABILITY: 0.0,
}

OUTPUT_TYPES = {
    AgentRole.MANAGER: MessageType.PLAN,
    AgentRole.RESEARCHER: MessageType.FINDING,
    AgentRole.IDEATOR: MessageType.IDEA,
    AgentRole.SKEPTIC: MessageType.CRITIQUE,
    AgentRole.VERIFIER: MessageType.VERIFICATION,
    AgentRole.ACCOUNTABILITY: MessageType.ACCOUNTABILITY,
}

OUTPUT_KINDS = {
    AgentRole.MANAGER: OutputKind.CLAIMS,
    AgentRole.RESEARCHER: OutputKind.CLAIMS,
    AgentRole.IDEATOR: OutputKind.CLAIMS,
    AgentRole.SKEPTIC: OutputKind.VERDICTS,
    AgentRole.VERIFIER: OutputKind.VERIFICATION,
    AgentRole.ACCOUNTABILITY: OutputKind.ACCOUNTABILITY,
}


def test_exactly_six_unique_roles():
    assert set(CONFIGS) == set(AgentRole)
    assert len({c.role for c in CONFIGS.values()}) == 6


def test_temperature_table():
    for role, expected in TEMPERATURES.items():
        assert CONFIGS[role].temperature == expected, role


def test_output_type_mapping():
    for role, expected in OUTPUT_TYPES.items():
        assert CONFIGS[role].output_type is expected, role


def test_output_kind_table():
    """All six agents are structured-output producers (PRD §6.6 + Phase 11 §6.8)."""
    for role, expected in OUTPUT_KINDS.items():
        assert CONFIGS[role].output_kind is expected, role
    assert all(c.output_kind is not OutputKind.PLAIN for c in CONFIGS.values())


def test_capability_table():
    """Phase 6 PRD §6.7 + Phase 11 §6.8: workers get tools, router/audits differ."""
    expected_workers = frozenset({"file_search", "file_reader", "web_search", "memory"})
    assert CONFIGS[AgentRole.MANAGER].capabilities == frozenset()
    for role in (AgentRole.RESEARCHER, AgentRole.IDEATOR, AgentRole.SKEPTIC):
        assert CONFIGS[role].capabilities == expected_workers, role
    # Verifier checks sources (no memory workspace); Accountability is trace-only
    assert CONFIGS[AgentRole.VERIFIER].capabilities == frozenset(
        {"file_search", "file_reader", "web_search"}
    )
    assert CONFIGS[AgentRole.ACCOUNTABILITY].capabilities == frozenset()


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
    # Phase 6 touched the three worker prompts; the fallback hardening bumped
    # Manager, Researcher, and Ideator again. Phase 11 adds two fresh roles;
    # live-eval hardening bumped Skeptic (never-blank objection) and
    # Verifier (recorded-evidence rule) to 3 and 2.
    assert [CONFIGS[r].prompt_version for r in AgentRole] == [2, 3, 3, 3, 2, 1]
    assert {c.display_name for c in CONFIGS.values()} == {
        "Manager", "Researcher", "Ideator", "Skeptic",
        "Verifier", "Accountability",
    }


def test_tool_failure_fallback_is_declared():
    """Agents must answer with labeled priors when tools return nothing."""
    researcher = CONFIGS[AgentRole.RESEARCHER].instructions.lower()
    assert "model prior knowledge" in researcher
    assert "never fact" in researcher
    ideator = CONFIGS[AgentRole.IDEATOR].instructions.lower()
    assert "tools fail" in ideator
    assert "hypothesis" in ideator


@pytest.mark.parametrize("role", list(AgentRole))
def test_role_defaults_to_primary_model(role):
    assert CONFIGS[role].model is None
    assert CONFIGS[role].max_tokens is None
