from __future__ import annotations

from app.schemas import AgentRole


class AgentError(Exception):
    """Base for agent-layer errors (LLM errors are never wrapped)."""

    def __init__(self, message: str, *, agent_role: AgentRole | None = None) -> None:
        super().__init__(message)
        self.agent_role = agent_role


class EmptyAgentResponseError(AgentError):
    """Model returned neither content nor tool calls."""

    def __init__(self, role: AgentRole) -> None:
        super().__init__(
            f"Agent {role.value} produced an empty response",
            agent_role=role,
        )
        self.hint = "Retry or lower temperature; check Ollama logs."


class AgentNotRegisteredError(AgentError):
    """Registry lookup for a role that was never registered."""

    def __init__(self, role: AgentRole, registered: list[AgentRole]) -> None:
        available = ", ".join(r.value for r in registered) or "none"
        super().__init__(
            f"No agent registered for role {role.value!r} (registered: {available})",
            agent_role=role,
        )
        self.registered = registered
