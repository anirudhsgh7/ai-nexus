"""Tool lookup and capability resolution (Phase 6 PRD §6.3).

Capability sets live on `AgentConfig`; this module resolves them against the
registry of tools the environment actually provides. A capability that names
a known-but-disabled tool (e.g. `web_search` with the flag off) degrades
gracefully; a capability that names nothing we have ever heard of is a config
defect and fails fast.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from app.config import Settings
from app.schemas import AgentRole
from app.tools.base import Tool
from app.tools.errors import ToolNotRegisteredError
from app.tools.file_reader import FileReaderTool
from app.tools.file_search import FileSearchTool
from app.tools.memory import MemoryTool
from app.tools.web_search import WebSearchTool

logger = logging.getLogger("ai_nexus.tools.registry")

__all__ = [
    "KNOWN_CAPABILITIES",
    "ToolRegistry",
    "build_tool_registry",
]

KNOWN_CAPABILITIES = frozenset(
    {"file_search", "file_reader", "memory", "web_search"}
)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotRegisteredError(name, self.names) from None

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def resolve(
        self, capabilities: Iterable[str], *, agent: AgentRole | None = None
    ) -> list[Tool]:
        """Capabilities → tools, sorted by name (deterministic spec order)."""
        resolved: list[Tool] = []
        for capability in capabilities:
            if capability not in KNOWN_CAPABILITIES:
                known = ", ".join(sorted(KNOWN_CAPABILITIES))
                raise ValueError(
                    f"unknown capability {capability!r} (known: {known})"
                )
            tool = self._tools.get(capability)
            if tool is None:
                logger.info(
                    "capability_unavailable agent=%s capability=%s",
                    agent.value if agent else "?",
                    capability,
                )
            else:
                resolved.append(tool)
        return sorted(resolved, key=lambda tool: tool.name)

    def __contains__(self, item: object) -> bool:
        return item in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


def build_tool_registry(settings: Settings) -> ToolRegistry:
    """Register the tools this environment allows (PRD §6.3)."""
    tools: list[Tool] = []
    root_raw = (settings.tool_files_root or "").strip()
    if root_raw:
        root = Path(root_raw).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(
                "AI_NEXUS_TOOL_FILES_ROOT is not an existing directory: "
                f"{settings.tool_files_root}"
            )
        common = dict(
            timeout_s=settings.tool_timeout_s,
            result_max_chars=settings.tool_result_max_chars,
        )
        tools.append(FileSearchTool(root=root, **common))
        tools.append(FileReaderTool(root=root, **common))
    tools.append(
        MemoryTool(
            timeout_s=settings.tool_timeout_s,
            result_max_chars=settings.tool_result_max_chars,
        )
    )
    if settings.tool_web_search_enabled:
        tools.append(
            WebSearchTool(
                max_results=settings.tool_web_search_max_results,
                timeout_s=settings.tool_timeout_s,
                result_max_chars=settings.tool_result_max_chars,
            )
        )
    return ToolRegistry(tools)
