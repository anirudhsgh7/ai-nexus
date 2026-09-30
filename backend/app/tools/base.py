"""Tool contract: ABC, bounded execution wrapper, path confinement, envelopes.

The agent tool loop only ever enters through `Tool.call`, which never raises:
every outcome (success, domain error, timeout, tool bug, bad arguments) comes
back as a `ToolResult` whose `content` is the exact envelope string fed to the
model (Phase 6 PRD §6.2).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from app import logging_config as log
from app.schemas import AgentRole, ToolResult
from app.tools.errors import ToolError

logger = logging.getLogger("ai_nexus.tools")

ARG_ERROR_CLIP = 300
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_RESULT_MAX_CHARS = 2000

__all__ = [
    "Tool",
    "ToolContext",
    "available_entries",
    "error_envelope",
    "ok_envelope",
    "resolve_within_root",
]


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Who is calling and in which run — tool state is namespaced by both."""

    agent: AgentRole
    run_id: str | None = None


def resolve_within_root(root: Path, relative: str) -> Path:
    """Resolve `relative` under `root`.

    Raises `ToolError('path_outside_root')` for absolute paths and for anything
    whose resolved target (after `..` collapse and symlink following) falls
    outside the root.
    """
    if Path(relative).is_absolute():
        raise ToolError(
            "path_outside_root", f"absolute paths are not allowed: {relative}"
        )
    resolved_root = root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ToolError(
            "path_outside_root", f"path escapes the corpus root: {relative}"
        )
    return candidate


def available_entries(root: Path) -> str:
    """Guidance string for error envelopes: what IS in the directory.

    Models hallucinate paths; naming the real entries lets them correct
    course on the next tool step instead of burning the step cap.
    """
    try:
        entries = sorted(
            p.name for p in root.iterdir() if not p.name.startswith(".")
        )
    except OSError:
        return "(unreadable)"
    listed = ", ".join(entries[:20])
    if len(entries) > 20:
        listed += ", ..."
    return listed or "(empty)"


def ok_envelope(name: str, payload: dict[str, Any]) -> str:
    body: dict[str, Any] = {"ok": True, "tool": name}
    body.update(payload)
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False)


def error_envelope(
    name: str, code: str, message: str, **extra: Any
) -> str:
    body: dict[str, Any] = {
        "ok": False,
        "tool": name,
        "error": code,
        "message": message,
    }
    body.update(extra)
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False)


class Tool(ABC):
    """One registry-owned tool. Subclasses declare metadata as class vars."""

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict]
    args_model: ClassVar[type[BaseModel]]

    def __init__(
        self, *, timeout_s: float = DEFAULT_TIMEOUT_S,
        result_max_chars: int = DEFAULT_RESULT_MAX_CHARS,
    ) -> None:
        self._timeout_s = timeout_s
        self._result_max_chars = result_max_chars

    @property
    def result_max_chars(self) -> int:
        return self._result_max_chars

    def spec(self) -> dict[str, Any]:
        """Ollama/OpenAI-style function spec for the request `tools` field."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def execute(self, args: BaseModel, context: ToolContext) -> dict[str, Any]:
        """Do the work; return the envelope payload (without ok/tool keys).

        Raise `ToolError(code, message)` for domain failures.
        """

    def _overflow_envelope(self, payload: dict[str, Any]) -> str:
        """Always-valid-JSON fallback when the payload exceeds the cap.

        Raw string slicing would hand the model malformed JSON; a preview
        envelope stays parseable and says what happened. The preview is
        shrunk until the *serialized* result fits (JSON escaping grows it),
        which is why this measures actual output length instead of estimating.
        """
        full = ok_envelope(self.name, payload)
        message = f"result exceeded {self._result_max_chars} characters"
        preview = full
        while True:
            candidate = json.dumps(
                {
                    "ok": True,
                    "tool": self.name,
                    "truncated": True,
                    "message": message,
                    "preview": preview,
                },
                separators=(",", ":"),
                ensure_ascii=False,
            )
            if len(candidate) <= self._result_max_chars or not preview:
                return candidate
            shrink = int(len(preview) * 0.75)
            preview = preview[: max(shrink - 16, 0)]

    async def call(self, arguments: dict, context: ToolContext) -> ToolResult:
        """Bounded, exception-free entry point used by the agent loop."""
        started = time.monotonic()
        log.tool_call_start(
            logger,
            tool=self.name,
            agent=context.agent.value,
            args_preview=json.dumps(arguments, ensure_ascii=False, default=str)[:120],
        )
        error: str | None = None
        try:
            args = self.args_model.model_validate(arguments)
        except ValidationError as exc:
            error = "invalid_arguments"
            message = " ".join(str(exc).split())[:ARG_ERROR_CLIP]
            content = error_envelope(self.name, error, message)
        else:
            try:
                payload = await asyncio.wait_for(
                    self.execute(args, context), timeout=self._timeout_s
                )
            except ToolError as exc:
                error = exc.code
                content = error_envelope(self.name, exc.code, exc.message)
            except TimeoutError:
                error = "timeout"
                content = error_envelope(
                    self.name, "timeout", f"tool timed out after {self._timeout_s}s"
                )
            except Exception as exc:  # noqa: BLE001 - tool bugs never kill runs
                logger.exception("tool %s raised", self.name)
                error = "internal_error"
                content = error_envelope(
                    self.name, "internal_error", type(exc).__name__
                )
            else:
                content = ok_envelope(self.name, payload)
                if len(content) > self._result_max_chars:
                    content = self._overflow_envelope(payload)
        duration_ms = round((time.monotonic() - started) * 1000, 1)
        log.tool_call_end(
            logger,
            tool=self.name,
            agent=context.agent.value,
            ok=error is None,
            error=error or "",
            duration_ms=duration_ms,
            content_chars=len(content),
        )
        return ToolResult(
            name=self.name, content=content, error=error, duration_ms=duration_ms
        )
