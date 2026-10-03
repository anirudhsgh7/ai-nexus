"""Agent lookup. Extension recipe for a future agent (spec §3):

0. Add one member to AgentRole in app/schemas.py (a new role is a new enum
   member — type-checked roles require this edit outside app/agents/).
1. Add one module under app/agents/ defining `CONFIG = AgentConfig(...)`.
2. Append it to `DEFAULT_AGENT_CONFIGS`.
3. Change nothing else — tests, scripts, and the orchestrator resolve agents
   through this registry only.

Phase 6: capabilities declared on a config resolve against the injected
`ToolRegistry`. With no registry (`tools=None`) agents get no tools — the
legacy path every pre-Phase-6 caller relies on.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from app.agents import accountability, ideator, manager, researcher, skeptic, verifier
from app.agents.base import Agent, AgentConfig
from app.agents.errors import AgentNotRegisteredError
from app.llm.base import LLMProvider
from app.schemas import AgentRole
from app.tools.registry import ToolRegistry

DEFAULT_AGENT_CONFIGS: tuple[AgentConfig, ...] = (
    manager.CONFIG,
    researcher.CONFIG,
    ideator.CONFIG,
    skeptic.CONFIG,
    verifier.CONFIG,
    accountability.CONFIG,
)


class AgentRegistry:
    def __init__(self, agents: Iterable[Agent]) -> None:
        self._agents: dict[AgentRole, Agent] = {}
        for agent in agents:
            if agent.role in self._agents:
                raise ValueError(f"duplicate agent role: {agent.role.value}")
            self._agents[agent.role] = agent

    def get(self, role: AgentRole) -> Agent:
        try:
            return self._agents[role]
        except KeyError:
            raise AgentNotRegisteredError(role, self.roles) from None

    @property
    def roles(self) -> list[AgentRole]:
        return list(self._agents)

    def __contains__(self, role: object) -> bool:
        return role in self._agents

    def __iter__(self) -> Iterator[Agent]:
        return iter(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)


def build_registry(
    provider: LLMProvider,
    configs: Sequence[AgentConfig] = DEFAULT_AGENT_CONFIGS,
    tools: ToolRegistry | None = None,
) -> AgentRegistry:
    """Construct a registry. No import-time instances; provider always injected.

    `tools=None` (default) resolves every capability set to the empty tuple —
    pre-Phase-6 callers keep byte-identical behavior.
    """

    def factory(config: AgentConfig) -> Agent:
        resolved = (
            tools.resolve(config.capabilities, agent=config.role)
            if tools is not None
            else ()
        )
        return Agent(config, provider, resolved)

    return AgentRegistry(factory(config) for config in configs)
