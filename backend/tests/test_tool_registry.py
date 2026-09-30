"""Tool registry: capability resolution, fail-fast, build matrix (PRD §6.3)."""

from __future__ import annotations

import logging

import pytest

from app.config import Settings
from app.schemas import AgentRole
from app.tools import (
    KNOWN_CAPABILITIES,
    MemoryTool,
    ToolNotRegisteredError,
    ToolRegistry,
    build_tool_registry,
)

WORKER_CAPS = frozenset({"file_search", "file_reader", "web_search", "memory"})


def test_known_capabilities_locked():
    assert KNOWN_CAPABILITIES == {
        "file_search", "file_reader", "memory", "web_search",
    }


def test_duplicate_tool_name_rejected():
    with pytest.raises(ValueError, match="duplicate tool name"):
        ToolRegistry([MemoryTool(), MemoryTool()])


def test_get_unknown_raises_tool_not_registered():
    registry = ToolRegistry([MemoryTool()])
    with pytest.raises(ToolNotRegisteredError) as exc:
        registry.get("web_search")
    assert exc.value.code == "unknown_tool"
    assert "memory" in exc.value.message


def test_names_sorted():
    registry = build_tool_registry(
        Settings(tool_files_root=".", tool_web_search_enabled=True)
    )
    assert registry.names == ["file_reader", "file_search", "memory", "web_search"]


def test_resolve_sorted_deterministic():
    registry = build_tool_registry(
        Settings(tool_files_root=".", tool_web_search_enabled=True)
    )
    tools = registry.resolve(WORKER_CAPS, agent=AgentRole.SKEPTIC)
    assert [t.name for t in tools] == [
        "file_reader", "file_search", "memory", "web_search",
    ]


def test_resolve_unknown_capability_fails_fast():
    registry = ToolRegistry([MemoryTool()])
    with pytest.raises(ValueError, match="unknown capability 'vetto'"):
        registry.resolve({"vetto"})


def test_resolve_unavailable_capability_skipped(caplog):
    registry = ToolRegistry([MemoryTool()])  # no web_search registered
    with caplog.at_level(logging.INFO, logger="ai_nexus.tools.registry"):
        tools = registry.resolve(WORKER_CAPS, agent=AgentRole.IDEATOR)
    assert [t.name for t in tools] == ["memory"]
    assert any(
        "capability_unavailable" in r.getMessage()
        and "capability=web_search" in r.getMessage()
        for r in caplog.records
    )


def test_manager_capabilities_resolve_empty():
    registry = build_tool_registry(
        Settings(tool_files_root=".", tool_web_search_enabled=True)
    )
    assert registry.resolve(frozenset(), agent=AgentRole.MANAGER) == []


def test_build_default_registers_only_memory():
    assert build_tool_registry(Settings()).names == ["memory"]


def test_build_with_files_root(tmp_path):
    registry = build_tool_registry(Settings(tool_files_root=str(tmp_path)))
    assert registry.names == ["file_reader", "file_search", "memory"]


def test_build_invalid_root_fails_fast():
    with pytest.raises(ValueError, match="AI_NEXUS_TOOL_FILES_ROOT"):
        build_tool_registry(Settings(tool_files_root="/definitely/not/a/dir"))


def test_build_web_flag_toggle():
    assert "web_search" not in build_tool_registry(Settings()).names
    assert "web_search" in build_tool_registry(
        Settings(tool_web_search_enabled=True)
    ).names


def test_memory_tool_always_registered():
    assert "memory" in build_tool_registry(Settings(tool_files_root="")).names
