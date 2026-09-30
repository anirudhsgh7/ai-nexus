"""`file_reader`: read a line-numbered window of one corpus file (PRD §6.6.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.tools.base import (
    Tool,
    ToolContext,
    available_entries,
    ok_envelope,
    resolve_within_root,
)
from app.tools.errors import ToolError

__all__ = ["FileReaderTool"]

MAX_FILE_BYTES = 1_000_000
MAX_READ_LINES = 200
_SNIFF_BYTES = 4096


class _FileReaderArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = Field(min_length=1, max_length=512)
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _window_ordered(self) -> "_FileReaderArgs":
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class FileReaderTool(Tool):
    name: ClassVar[str] = "file_reader"
    description: ClassVar[str] = (
        "Read a file from the local document corpus. Returns line-numbered text "
        "for at most 200 consecutive lines starting at start_line."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path of the file inside the corpus.",
            },
            "start_line": {
                "type": "integer",
                "minimum": 1,
                "description": "First line to return (default 1).",
            },
            "end_line": {
                "type": "integer",
                "minimum": 1,
                "description": "Last line to return (default: start_line + 199).",
            },
        },
        "required": ["path"],
    }
    args_model: ClassVar[type[BaseModel]] = _FileReaderArgs

    def __init__(
        self, *, root: Path, timeout_s: float = 20.0,
        result_max_chars: int = 2000,
    ) -> None:
        super().__init__(timeout_s=timeout_s, result_max_chars=result_max_chars)
        self._root = root

    async def execute(self, args: BaseModel, context: ToolContext) -> dict[str, Any]:
        assert isinstance(args, _FileReaderArgs)  # enforced by Tool.call
        target = resolve_within_root(self._root, args.path)
        if not target.exists():
            raise ToolError(
                "not_found",
                f"no such file in corpus: {args.path}. "
                f"Top-level entries: {available_entries(self._root)}",
            )
        if target.is_dir():
            raise ToolError("is_directory", f"not a file: {args.path}")
        try:
            if target.stat().st_size > MAX_FILE_BYTES:
                raise ToolError(
                    "file_too_large",
                    f"{args.path} exceeds {MAX_FILE_BYTES} bytes",
                )
            with target.open("rb") as handle:
                head = handle.read(_SNIFF_BYTES)
            if b"\x00" in head:
                raise ToolError("binary_file", f"binary file not readable: {args.path}")
            lines = target.read_text(errors="replace").splitlines()
        except OSError as exc:
            raise ToolError("not_found", f"cannot read {args.path}: {exc}") from exc

        total = len(lines)
        start = args.start_line
        end = args.end_line if args.end_line is not None else start + MAX_READ_LINES - 1
        end = min(end, start + MAX_READ_LINES - 1, total)
        selected = list(range(start, end + 1))
        # Drop lines from the end until the whole envelope fits the model-facing
        # cap, so the model always receives well-formed JSON (never a clipped one).
        while selected and len(ok_envelope(self.name, self._payload(
                args, lines, selected, total))) > self.result_max_chars:
            selected.pop()
        return self._payload(args, lines, selected, total)

    def _payload(
        self, args: _FileReaderArgs, lines: list[str], selected: list[int],
        total: int,
    ) -> dict[str, Any]:
        last = selected[-1] if selected else args.start_line - 1
        window = [f"{i}: {lines[i - 1]}" for i in selected]
        return {
            "path": args.path,
            "start_line": args.start_line,
            "end_line": max(min(last, total), 0),
            "total_lines": total,
            "content": "\n".join(window),
            "truncated": last < total,
        }
