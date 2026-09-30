"""Memory tool: namespaced scratchpad semantics (PRD §6.6.3)."""

from __future__ import annotations

import json

from app.schemas import AgentRole
from app.tools import MemoryStore, MemoryTool, ToolContext

R1_RESEARCH = ToolContext(agent=AgentRole.RESEARCHER, run_id="run1")
R1_SKEPTIC = ToolContext(agent=AgentRole.SKEPTIC, run_id="run1")
R2_RESEARCH = ToolContext(agent=AgentRole.RESEARCHER, run_id="run2")


async def test_write_read_round_trip():
    tool = MemoryTool()
    wrote = await tool.call(
        {"action": "write", "key": "finding1", "content": "growth is 23%"},
        R1_RESEARCH,
    )
    assert wrote.error is None
    assert json.loads(wrote.content) == {
        "ok": True, "tool": "memory", "action": "write",
        "key": "finding1", "chars": len("growth is 23%"),
    }
    read = await tool.call({"action": "read", "key": "finding1"}, R1_RESEARCH)
    payload = json.loads(read.content)
    assert payload["found"] is True
    assert payload["content"] == "growth is 23%"


async def test_read_missing_key_is_not_found():
    payload = json.loads(
        (await MemoryTool().call({"action": "read", "key": "ghost"}, R1_RESEARCH)).content
    )
    assert payload == {
        "ok": True, "tool": "memory", "action": "read",
        "key": "ghost", "found": False,
    }


async def test_list_keys_sorted():
    tool = MemoryTool()
    for key in ("beta", "alpha"):
        await tool.call(
            {"action": "write", "key": key, "content": "x"}, R1_RESEARCH
        )
    payload = json.loads(
        (await tool.call({"action": "list"}, R1_RESEARCH)).content
    )
    assert payload["keys"] == ["alpha", "beta"]


async def test_namespaces_isolated_by_agent_and_run():
    tool = MemoryTool()
    await tool.call(
        {"action": "write", "key": "note", "content": "researcher private"}, R1_RESEARCH
    )
    same_key_other_agent = json.loads(
        (await tool.call({"action": "read", "key": "note"}, R1_SKEPTIC)).content
    )
    assert same_key_other_agent["found"] is False
    same_key_other_run = json.loads(
        (await tool.call({"action": "read", "key": "note"}, R2_RESEARCH)).content
    )
    assert same_key_other_run["found"] is False


async def test_invalid_key_rejected():
    result = await MemoryTool().call(
        {"action": "write", "key": "bad key!", "content": "x"}, R1_RESEARCH
    )
    assert result.error == "invalid_key"


async def test_oversize_content_rejected():
    result = await MemoryTool().call(
        {"action": "write", "key": "k", "content": "y" * 2001}, R1_RESEARCH
    )
    assert result.error == "invalid_arguments"


async def test_write_requires_content_and_read_rejects_content():
    assert (
        await MemoryTool().call({"action": "write", "key": "k"}, R1_RESEARCH)
    ).error == "invalid_arguments"
    assert (
        await MemoryTool().call(
            {"action": "read", "key": "k", "content": "x"}, R1_RESEARCH
        )
    ).error == "invalid_arguments"
    assert (
        await MemoryTool().call({"action": "list", "key": "k"}, R1_RESEARCH)
    ).error == "invalid_arguments"


async def test_memory_full_after_cap():
    tool = MemoryTool(store=MemoryStore(max_keys=3))
    for index in range(3):
        result = await tool.call(
            {"action": "write", "key": f"k{index}", "content": "x"}, R1_RESEARCH
        )
        assert result.error is None
    overflow = await tool.call(
        {"action": "write", "key": "k3", "content": "x"}, R1_RESEARCH
    )
    assert overflow.error == "memory_full"
    # updating an existing key still works at capacity
    update = await tool.call(
        {"action": "write", "key": "k0", "content": "updated"}, R1_RESEARCH
    )
    assert update.error is None


def test_namespace_eviction_bound():
    store = MemoryStore(max_namespaces=2)
    store.write("run1:researcher", "k", "one")
    store.write("run2:researcher", "k", "two")
    store.write("run3:researcher", "k", "three")
    assert store.read("run1:researcher", "k") is None
    assert store.read("run3:researcher", "k") == "three"
