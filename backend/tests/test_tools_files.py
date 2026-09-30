"""File tools: deterministic search/read semantics and safety rails (PRD §6.6.1/2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas import AgentRole
from app.tools import ToolContext
from app.tools.file_reader import FileReaderTool, MAX_READ_LINES
from app.tools.file_search import SNIPPET_CHARS, FileSearchTool

CTX = ToolContext(agent=AgentRole.SKEPTIC, run_id="run1")

GROWTH = (
    "2025 Annual Growth Report\n"
    "Annual growth was 23% in 2025, not 40%.\n"
    "CEO note: figures unaudited.\n"
)
PRICING = (
    "The cheaper option is Plan B.\n"
    "Growth plans are optional upgrades.\n"
)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "growth_report.txt").write_text(GROWTH)
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "pricing.txt").write_text(PRICING)
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.txt").write_text("Hidden growth is 99%.\n")
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01binary growth")
    return tmp_path


# ---------------------------------------------------------------- file_search


async def test_search_token_and_matching(corpus: Path):
    result = await FileSearchTool(root=corpus).call(
        {"query": "Growth 2025"}, CTX
    )
    assert result.error is None
    payload = json.loads(result.content)
    assert payload["ok"] is True
    # both tokens on report lines; not pricing (no "2025"), not hidden/binary
    assert payload["matches"] == [
        {"file": "growth_report.txt", "line": 1, "snippet": "2025 Annual Growth Report"},
        {
            "file": "growth_report.txt",
            "line": 2,
            "snippet": "Annual growth was 23% in 2025, not 40%.",
        },
    ]
    assert payload["truncated"] is False


async def test_search_case_insensitive(corpus: Path):
    payload = json.loads(
        (await FileSearchTool(root=corpus).call({"query": "GROWTH"}, CTX)).content
    )
    files = {m["file"] for m in payload["matches"]}
    assert files == {"growth_report.txt", "notes/pricing.txt"}


async def test_search_max_results_and_truncated(corpus: Path):
    payload = json.loads(
        (
            await FileSearchTool(root=corpus).call(
                {"query": "growth", "max_results": 1}, CTX
            )
        ).content
    )
    assert len(payload["matches"]) == 1
    assert payload["truncated"] is True


async def test_search_subdirectory(corpus: Path):
    payload = json.loads(
        (
            await FileSearchTool(root=corpus).call(
                {"query": "option", "path": "notes"}, CTX
            )
        ).content
    )
    # substring semantics: "option" matches "option" and "optional"
    assert [m["file"] for m in payload["matches"]] == [
        "notes/pricing.txt", "notes/pricing.txt",
    ]
    assert [m["line"] for m in payload["matches"]] == [1, 2]


async def test_search_skips_hidden_and_binary(corpus: Path):
    payload = json.loads(
        (await FileSearchTool(root=corpus).call({"query": "99%"}, CTX)).content
    )
    assert payload["matches"] == []
    payload = json.loads(
        (await FileSearchTool(root=corpus).call({"query": "binary"}, CTX)).content
    )
    assert payload["matches"] == []


async def test_search_outside_root_rejected(corpus: Path):
    result = await FileSearchTool(root=corpus).call(
        {"query": "x", "path": ".."}, CTX
    )
    assert result.error == "path_outside_root"


async def test_search_unknown_subdirectory_guides_model(corpus: Path):
    """Hallucinated paths must teach the model what exists (live failure mode)."""
    result = await FileSearchTool(root=corpus).call(
        {"query": "growth", "path": "data/financial_reports"}, CTX
    )
    assert result.error == "path_not_found"
    payload = json.loads(result.content)
    assert "growth_report.txt" in payload["message"]
    assert "notes" in payload["message"]


async def test_search_relaxed_fallback_marks_partial(corpus: Path):
    """Strict token-AND finds nothing for 'growth rate' -> any-token fallback."""
    payload = json.loads(
        (await FileSearchTool(root=corpus).call({"query": "growth rate"}, CTX)).content
    )
    assert payload["partial"] is True
    assert payload["matches"], "relaxed pass must return lines with any token"
    assert {m["file"] for m in payload["matches"]} <= {
        "growth_report.txt", "notes/pricing.txt",
    }


async def test_search_strict_hit_has_no_partial_flag(corpus: Path):
    payload = json.loads(
        (await FileSearchTool(root=corpus).call({"query": "growth 2025"}, CTX)).content
    )
    assert "partial" not in payload


async def test_search_blank_query_rejected(corpus: Path):
    result = await FileSearchTool(root=corpus).call({"query": "   "}, CTX)
    assert result.error == "invalid_arguments"


async def test_search_snippet_clipped(tmp_path: Path):
    long_line = "growth " + "x" * 300
    (tmp_path / "long.txt").write_text(long_line + "\n")
    payload = json.loads(
        (await FileSearchTool(root=tmp_path).call({"query": "growth"}, CTX)).content
    )
    assert len(payload["matches"][0]["snippet"]) == SNIPPET_CHARS


async def test_search_deterministic_order(tmp_path: Path):
    (tmp_path / "b.txt").write_text("growth here\n")
    (tmp_path / "a.txt").write_text("growth here\n")
    (tmp_path / "c.txt").write_text("growth here\n")
    payload = json.loads(
        (await FileSearchTool(root=tmp_path).call({"query": "growth"}, CTX)).content
    )
    assert [m["file"] for m in payload["matches"]] == ["a.txt", "b.txt", "c.txt"]


# ---------------------------------------------------------------- file_reader


async def test_reader_full_file(corpus: Path):
    result = await FileReaderTool(root=corpus).call(
        {"path": "growth_report.txt"}, CTX
    )
    assert result.error is None
    payload = json.loads(result.content)
    assert payload["start_line"] == 1
    assert payload["end_line"] == 3
    assert payload["total_lines"] == 3
    assert payload["content"] == (
        "1: 2025 Annual Growth Report\n"
        "2: Annual growth was 23% in 2025, not 40%.\n"
        "3: CEO note: figures unaudited."
    )
    assert payload["truncated"] is False


async def test_reader_window(corpus: Path):
    payload = json.loads(
        (
            await FileReaderTool(root=corpus).call(
                {"path": "growth_report.txt", "start_line": 2, "end_line": 2}, CTX
            )
        ).content
    )
    assert payload["start_line"] == 2
    assert payload["end_line"] == 2
    assert payload["content"] == "2: Annual growth was 23% in 2025, not 40%."


async def test_reader_window_clamped_to_200_lines(tmp_path: Path):
    (tmp_path / "many.txt").write_text("\n".join(f"line {i}" for i in range(1, 401)))
    payload = json.loads(
        (
            await FileReaderTool(root=tmp_path, result_max_chars=100_000).call(
                {"path": "many.txt", "start_line": 10}, CTX
            )
        ).content
    )
    assert payload["end_line"] - payload["start_line"] + 1 == MAX_READ_LINES
    assert payload["total_lines"] == 400
    assert payload["truncated"] is True


async def test_reader_budget_trims_but_keeps_valid_json(tmp_path: Path):
    (tmp_path / "many.txt").write_text("\n".join(f"line {i}" for i in range(1, 401)))
    tool = FileReaderTool(root=tmp_path)  # default 2000-char cap
    result = await tool.call({"path": "many.txt"}, CTX)
    assert result.error is None
    payload = json.loads(result.content)  # well-formed JSON, not a clipped string
    assert len(result.content) <= tool.result_max_chars
    assert payload["truncated"] is True
    assert payload["start_line"] == 1
    assert payload["end_line"] < 400
    assert payload["content"].splitlines()[-1] == (
        f"{payload['end_line']}: line {payload['end_line']}"
    )


async def test_reader_beyond_eof_empty_window(corpus: Path):
    payload = json.loads(
        (
            await FileReaderTool(root=corpus).call(
                {"path": "growth_report.txt", "start_line": 50}, CTX
            )
        ).content
    )
    assert payload["content"] == ""
    assert payload["total_lines"] == 3


async def test_reader_end_before_start_rejected(corpus: Path):
    result = await FileReaderTool(root=corpus).call(
        {"path": "growth_report.txt", "start_line": 3, "end_line": 1}, CTX
    )
    assert result.error == "invalid_arguments"


async def test_reader_missing_file(corpus: Path):
    result = await FileReaderTool(root=corpus).call({"path": "nope.txt"}, CTX)
    assert result.error == "not_found"
    assert "growth_report.txt" in json.loads(result.content)["message"]


async def test_reader_directory_rejected(corpus: Path):
    result = await FileReaderTool(root=corpus).call({"path": "notes"}, CTX)
    assert result.error == "is_directory"


async def test_reader_binary_rejected(corpus: Path):
    result = await FileReaderTool(root=corpus).call({"path": "bin.dat"}, CTX)
    assert result.error == "binary_file"


async def test_reader_outside_root_rejected(corpus: Path):
    result = await FileReaderTool(root=corpus).call(
        {"path": "../outside.txt"}, CTX
    )
    assert result.error == "path_outside_root"
