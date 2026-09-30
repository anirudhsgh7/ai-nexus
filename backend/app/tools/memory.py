"""`memory`: bounded per-(run, agent) scratchpad (Phase 6 PRD §6.6.3).

The namespace is `run_id:agent` so one agent can carry notes across its own
revision rounds while no agent (or operator) can read another agent's notes —
a shared store would be an undocumented peer-prose channel and would break
the anti-conformity guarantee.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.tools.base import Tool, ToolContext
from app.tools.errors import ToolError

__all__ = ["MemoryStore", "MemoryTool"]

KEY_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
MAX_KEYS = 20
MAX_VALUE_CHARS = 2000
MAX_NAMESPACES = 200


class MemoryStore:
    """Process-local, bounded key-value store namespaced by run and agent."""

    def __init__(
        self, *, max_namespaces: int = MAX_NAMESPACES,
        max_keys: int = MAX_KEYS,
    ) -> None:
        self._max_namespaces = max_namespaces
        self._max_keys = max_keys
        self._namespaces: OrderedDict[str, dict[str, str]] = OrderedDict()

    def _namespace(self, namespace: str) -> dict[str, str]:
        if namespace not in self._namespaces:
            self._namespaces[namespace] = {}
            while len(self._namespaces) > self._max_namespaces:
                self._namespaces.popitem(last=False)
        else:
            self._namespaces.move_to_end(namespace)
        return self._namespaces[namespace]

    @staticmethod
    def _check_key(key: str) -> None:
        if not KEY_RE.match(key):
            raise ToolError(
                "invalid_key",
                f"key must match {KEY_RE.pattern} (got {key!r})",
            )

    @staticmethod
    def _check_content(content: str) -> None:
        if not content.strip():
            raise ToolError("invalid_arguments", "content must be non-empty")
        if len(content) > MAX_VALUE_CHARS:
            raise ToolError(
                "invalid_arguments",
                f"content exceeds {MAX_VALUE_CHARS} chars",
            )

    def write(self, namespace: str, key: str, content: str) -> None:
        self._check_key(key)
        self._check_content(content)
        store = self._namespace(namespace)
        if key not in store and len(store) >= self._max_keys:
            raise ToolError(
                "memory_full",
                f"namespace {namespace} already holds {self._max_keys} keys",
            )
        store[key] = content

    def read(self, namespace: str, key: str) -> str | None:
        self._check_key(key)
        return self._namespace(namespace).get(key)

    def keys(self, namespace: str) -> list[str]:
        return sorted(self._namespace(namespace))


class _MemoryArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: Literal["read", "write", "list"]
    key: str | None = Field(default=None, max_length=64)
    content: str | None = Field(default=None, max_length=MAX_VALUE_CHARS)

    @model_validator(mode="after")
    def _action_fields(self) -> "_MemoryArgs":
        if self.action == "write":
            if self.key is None:
                raise ValueError("action=write requires key")
            if self.content is None:
                raise ValueError("action=write requires content")
        elif self.action == "read":
            if self.key is None:
                raise ValueError("action=read requires key")
            if self.content is not None:
                raise ValueError("action=read does not take content")
        elif self.key is not None or self.content is not None:
            raise ValueError("action=list takes no key or content")
        return self


class MemoryTool(Tool):
    name: ClassVar[str] = "memory"
    description: ClassVar[str] = (
        "Your private scratchpad for this run: save notes with action=write "
        "(key + content), fetch one note with action=read (key), or list your "
        "note keys with action=list. Notes persist across your own steps of "
        "the current run and are visible only to you."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["read", "write", "list"]},
            "key": {
                "type": "string",
                "description": "Note name: letters, digits, '_', '.', '-' (max 64).",
            },
            "content": {
                "type": "string",
                "description": "Note text to save (action=write only).",
            },
        },
        "required": ["action"],
    }
    args_model: ClassVar[type[BaseModel]] = _MemoryArgs

    def __init__(
        self, *, store: MemoryStore | None = None, timeout_s: float = 20.0,
        result_max_chars: int = 2000,
    ) -> None:
        super().__init__(timeout_s=timeout_s, result_max_chars=result_max_chars)
        self._store = store or MemoryStore()

    async def execute(self, args: BaseModel, context: ToolContext) -> dict[str, Any]:
        assert isinstance(args, _MemoryArgs)  # enforced by Tool.call
        namespace = f"{context.run_id or 'adhoc'}:{context.agent.value}"
        if args.action == "write":
            assert args.key is not None and args.content is not None
            self._store.write(namespace, args.key, args.content)
            return {"action": "write", "key": args.key, "chars": len(args.content)}
        if args.action == "read":
            assert args.key is not None
            stored = self._store.read(namespace, args.key)
            payload: dict[str, Any] = {"action": "read", "key": args.key}
            if stored is None:
                payload["found"] = False
            else:
                payload["found"] = True
                payload["content"] = stored
            return payload
        return {"action": "list", "keys": self._store.keys(namespace)}
