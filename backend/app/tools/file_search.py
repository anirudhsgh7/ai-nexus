"""`file_search`: deterministic token-AND search over a corpus directory (PRD §6.6.1)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.tools.base import (
    Tool,
    ToolContext,
    available_entries,
    resolve_within_root,
)
from app.tools.errors import ToolError

__all__ = ["FileSearchTool"]

MAX_FILE_BYTES = 1_000_000
SNIPPET_CHARS = 200
_SNIFF_BYTES = 4096


class _FileSearchArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1, max_length=200)
    max_results: int = Field(default=10, ge=1, le=50)
    path: str = Field(default="", max_length=512)

    @model_validator(mode="after")
    def _query_not_blank(self) -> "_FileSearchArgs":
        if not self.query.strip():
            raise ValueError("query must contain non-whitespace characters")
        return self


def _iter_files(base: Path, root: Path) -> Iterator[Path]:
    """Deterministic path-ordered files, hidden entries skipped, escapes filtered."""
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(base).parts):
            continue
        try:
            if not path.resolve().is_relative_to(root):
                continue  # symlink escaping the corpus root
        except OSError:
            continue
        yield path


def _readable_lines(path: Path) -> list[str] | None:
    """None when the file should be skipped (oversize/binary/unreadable)."""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        with path.open("rb") as handle:
            head = handle.read(_SNIFF_BYTES)
        if b"\x00" in head:
            return None
        return path.read_text(errors="replace").splitlines()
    except OSError:
        return None


class FileSearchTool(Tool):
    name: ClassVar[str] = "file_search"
    description: ClassVar[str] = (
        "Search the local document corpus for a query. A line matches when it "
        "contains every word of the query (case-insensitive); if nothing "
        "matches, lines containing any query word are returned instead (those "
        "results carry partial=true). Returns matching lines with file paths "
        "and line numbers."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Words to search for; all must appear on a line.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": "Maximum number of matching lines to return (default 10).",
            },
            "path": {
                "type": "string",
                "description": (
                    "Optional subdirectory of the corpus to search; leave empty "
                    "to search everything."
                ),
            },
        },
        "required": ["query"],
    }
    args_model: ClassVar[type[BaseModel]] = _FileSearchArgs

    def __init__(
        self, *, root: Path, timeout_s: float = 20.0,
        result_max_chars: int = 2000,
    ) -> None:
        super().__init__(timeout_s=timeout_s, result_max_chars=result_max_chars)
        self._root = root

    def _scan(
        self, base: Path, resolved_root: Path, tokens: list[str], limit: int,
        *,
        strict: bool,
    ) -> tuple[list[dict[str, Any]], bool]:
        matches: list[dict[str, Any]] = []
        truncated = False
        for path in _iter_files(base, resolved_root):
            lines = _readable_lines(path)
            if lines is None:
                continue
            relative = str(path.relative_to(resolved_root))
            for lineno, line in enumerate(lines, start=1):
                lowered = line.lower()
                if strict:
                    hit = all(token in lowered for token in tokens)
                else:
                    hit = any(token in lowered for token in tokens)
                if not hit:
                    continue
                snippet = line.strip()
                if len(snippet) > SNIPPET_CHARS:
                    snippet = snippet[:SNIPPET_CHARS]
                matches.append(
                    {"file": relative, "line": lineno, "snippet": snippet}
                )
                if len(matches) > limit:
                    truncated = True
                    break
            if truncated:
                break
        return matches[:limit], truncated

    async def execute(self, args: BaseModel, context: ToolContext) -> dict[str, Any]:
        assert isinstance(args, _FileSearchArgs)  # enforced by Tool.call
        resolved_root = self._root.resolve()
        base = resolve_within_root(self._root, args.path) if args.path else resolved_root
        if not base.is_dir():
            if args.path:
                # Guide the model: hallucinated paths are the common failure.
                raise ToolError(
                    "path_not_found",
                    f"no such directory in the corpus: {args.path}. "
                    f"Top-level entries: {available_entries(resolved_root)}",
                )
            raise ToolError("corpus_unavailable", "corpus root is not a directory")

        tokens = args.query.lower().split()
        matches, truncated = self._scan(
            base, resolved_root, tokens, args.max_results, strict=True
        )
        payload: dict[str, Any] = {
            "query": args.query,
            "matches": matches,
            "truncated": truncated,
        }
        if not matches:
            matches, truncated = self._scan(
                base, resolved_root, tokens, args.max_results, strict=False
            )
            if matches:
                payload = {
                    "query": args.query,
                    "matches": matches,
                    "truncated": truncated,
                    "partial": True,
                }
        return payload
