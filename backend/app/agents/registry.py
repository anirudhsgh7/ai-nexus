"""Agent lookup. Extension recipe for a future agent (spec §3):

0. Add one member to AgentRole in app/schemas.py (a new role is a new enum
   member — type-checked roles require this edit outside app/agents/).
1. Add one module under app/agents/ defining `CONFIG = AgentConfig(...)`.
2. Append it to `DEFAULT_AGENT_CONFIGS`.
3. Change nothing else — tests, scripts, and the orchestrator resolve agents
   through this registry only.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from app.agents import ideator, manager, researcher, skeptic
from app.agents.base import Agent, AgentConfig
from app.agents.errors import AgentNotRegisteredError
from app.llm.base import LLMProvider
from app.schemas import AgentRole

DEFAULT_AGENT_CONFIGS: tuple[AgentConfig, ...] = (
    manager.CONFIG,
    researcher.CONFIG,
    ideator.CONFIG,
    skeptic.CONFIG,
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
) -> AgentRegistry:
    """Construct a registry. No import-time instances; provider always injected."""
    return AgentRegistry(Agent(config, provider) for config in configs)
