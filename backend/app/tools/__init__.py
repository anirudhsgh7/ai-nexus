"""Tool package: contract, registry, and the built-in tools (Phase 6)."""

from app.tools.base import (
    Tool,
    ToolContext,
    error_envelope,
    ok_envelope,
    resolve_within_root,
)
from app.tools.errors import ToolError, ToolNotRegisteredError
from app.tools.file_reader import FileReaderTool
from app.tools.file_search import FileSearchTool
from app.tools.memory import MemoryStore, MemoryTool
from app.tools.registry import KNOWN_CAPABILITIES, ToolRegistry, build_tool_registry
from app.tools.web_search import WebSearchTool

__all__ = [
    "KNOWN_CAPABILITIES",
    "FileReaderTool",
    "FileSearchTool",
    "MemoryStore",
    "MemoryTool",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolNotRegisteredError",
    "ToolRegistry",
    "WebSearchTool",
    "build_tool_registry",
    "error_envelope",
    "ok_envelope",
    "resolve_within_root",
]
