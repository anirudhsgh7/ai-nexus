"""`Tool.call` contract: envelopes, error taxonomy, timeout, truncation (PRD §6.2)."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.schemas import AgentRole
from app.tools import (
    Tool,
    ToolContext,
    error_envelope,
    ok_envelope,
    resolve_within_root,
)
from app.tools.errors import ToolError

CTX = ToolContext(agent=AgentRole.RESEARCHER, run_id="r1")


class _Args(BaseModel):
    model_config = ConfigDict(extra="ignore")

    value: str = "x"


class _RequiredArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    needed: str = Field(min_length=1)


class _EchoTool(Tool):
    name: ClassVar[str] = "echo"
    description: ClassVar[str] = "Echo the value back."
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
    }
    args_model: ClassVar[type[BaseModel]] = _Args

    async def execute(self, args: BaseModel, context: ToolContext) -> dict[str, Any]:
        assert isinstance(args, _Args)
        return {"echo": args.value}


class _RejectTool(_EchoTool):
    name: ClassVar[str] = "reject"
    args_model: ClassVar[type[BaseModel]] = _RequiredArgs


class _DomainTool(_EchoTool):
    name: ClassVar[str] = "domain"

    async def execute(self, args: BaseModel, context: ToolContext) -> dict[str, Any]:
        raise ToolError("not_found", "no such thing")


class _BoomTool(_EchoTool):
    name: ClassVar[str] = "boom"

    async def execute(self, args: BaseModel, context: ToolContext) -> dict[str, Any]:
        raise RuntimeError("kaboom")


class _SlowTool(_EchoTool):
    name: ClassVar[str] = "slow"

    async def execute(self, args: BaseModel, context: ToolContext) -> dict[str, Any]:
        await asyncio.sleep(0.5)
        return {}


# ------------------------------------------------------------------ envelopes


def test_ok_envelope_shape():
    data = json.loads(ok_envelope("file_search", {"query": "q", "matches": []}))
    assert data == {
        "ok": True,
        "tool": "file_search",
        "query": "q",
        "matches": [],
    }


def test_error_envelope_shape():
    data = json.loads(error_envelope("web_search", "network_error", "boom"))
    assert data == {
        "ok": False,
        "tool": "web_search",
        "error": "network_error",
        "message": "boom",
    }


# --------------------------------------------------------------- Tool.call


async def test_success_returns_ok_result():
    result = await _EchoTool().call({"value": "hi"}, CTX)
    assert result.name == "echo"
    assert result.error is None
    assert json.loads(result.content) == {"ok": True, "tool": "echo", "echo": "hi"}
    assert result.duration_ms is not None and result.duration_ms >= 0


async def test_invalid_arguments_envelope():
    result = await _RejectTool().call({}, CTX)
    assert result.error == "invalid_arguments"
    payload = json.loads(result.content)
    assert payload["ok"] is False
    assert payload["error"] == "invalid_arguments"
    assert payload["message"]


async def test_domain_error_passthrough():
    result = await _DomainTool().call({"value": "x"}, CTX)
    assert result.error == "not_found"
    payload = json.loads(result.content)
    assert payload["message"] == "no such thing"


async def test_unexpected_exception_becomes_internal_error():
    result = await _BoomTool().call({"value": "x"}, CTX)
    assert result.error == "internal_error"
    payload = json.loads(result.content)
    assert payload["error"] == "internal_error"
    assert payload["message"] == "RuntimeError"


async def test_timeout_envelope():
    result = await _SlowTool(timeout_s=0.01).call({"value": "x"}, CTX)
    assert result.error == "timeout"
    payload = json.loads(result.content)
    assert "timed out" in payload["message"]


async def test_result_overflow_stays_valid_json():
    big = "y" * 500
    result = await _EchoTool(result_max_chars=300).call({"value": big}, CTX)
    assert result.error is None
    payload = json.loads(result.content)  # never a clipped, malformed envelope
    assert payload["truncated"] is True
    assert "exceeded 300 characters" in payload["message"]
    assert len(result.content) <= 300
    assert "preview" in payload


async def test_call_never_raises_on_junk_arguments():
    result = await _EchoTool().call({"value": ["not", "a", "string"]}, CTX)
    assert result.error == "invalid_arguments"


async def test_tool_call_logging(caplog):
    with caplog.at_level(logging.INFO, logger="ai_nexus.tools"):
        await _EchoTool().call({"value": "v"}, CTX)
    events = [r.getMessage() for r in caplog.records]
    assert any("tool_call_start" in e and "tool=echo" in e for e in events)
    assert any("tool_call_end" in e and "ok=True" in e for e in events)


def test_spec_shape():
    spec = _EchoTool().spec()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "echo"
    assert spec["function"]["description"]
    assert spec["function"]["parameters"]["type"] == "object"


# ------------------------------------------------------------- path guard


def test_resolve_ok(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    target = tmp_path / "sub" / "f.txt"
    target.write_text("x")
    assert resolve_within_root(tmp_path, "sub/f.txt") == target.resolve()


def test_resolve_rejects_absolute(tmp_path: Path):
    with pytest.raises(ToolError) as exc:
        resolve_within_root(tmp_path, "/etc/passwd")
    assert exc.value.code == "path_outside_root"


def test_resolve_rejects_dotdot_escape(tmp_path: Path):
    with pytest.raises(ToolError) as exc:
        resolve_within_root(tmp_path, "../outside.txt")
    assert exc.value.code == "path_outside_root"


def test_resolve_rejects_symlink_escape(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (root / "link").symlink_to(outside)
    with pytest.raises(ToolError) as exc:
        resolve_within_root(root, "link/secret.txt")
    assert exc.value.code == "path_outside_root"
