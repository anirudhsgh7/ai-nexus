"""Tool-layer error taxonomy (Phase 6 PRD §6.2).

`ToolError` is a *domain* failure: it is converted to an error envelope and
fed back to the model, never raised out of `Tool.call`.
"""

from __future__ import annotations

__all__ = ["ToolError", "ToolNotRegisteredError"]


class ToolError(Exception):
    """A tool refused or failed to perform the requested operation."""

    def __init__(self, code: str, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


class ToolNotRegisteredError(ToolError):
    """Lookup of a tool name that the registry does not know."""

    def __init__(self, name: str, available: list[str]) -> None:
        listed = ", ".join(available) or "none"
        super().__init__(
            "unknown_tool", f"no such tool: {name} (available: {listed})"
        )
        self.name = name
        self.available = available
