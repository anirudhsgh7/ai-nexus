#!/usr/bin/env python3
"""Phase 6 exit-criterion gate: the Skeptic refutes a planted false claim with a tool.

Scenario (PLAN.md §Phase 6): a corpus document says the 2025 annual growth
rate was 23%; the planted claim asserts 40%. The Skeptic must independently
run a tool and return a `refuted` verdict citing the document.

Usage (from backend/):
  python scripts/tool_refutation_demo.py [--keep]

Exit codes:
  0  PASS — refuted AND at least one tool was executed
  1  FAIL — the refutation or the tool use was not observed
  2  setup/run error (Ollama down, bad config, ...)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import logging_config  # noqa: E402
from app.agents import build_registry  # noqa: E402
from app.config import Settings  # noqa: E402
from app.llm import get_provider  # noqa: E402
from app.schemas import (  # noqa: E402
    AgentRole,
    Claim,
    ClaimStatus,
    ClaimVerdict,
)
from app.tools import build_tool_registry  # noqa: E402

TASK = "Is the claimed 2025 annual growth rate of 40% correct? Check the documents."
PLANTED_CLAIM = "The 2025 annual growth rate was 40%."
TRUTH_LINE = "Annual growth was 23% in 2025, not 40%."


def _make_corpus() -> Path:
    corpus = Path(tempfile.mkdtemp(prefix="ai_nexus_refute_"))
    (corpus / "growth_report.txt").write_text(
        "2025 Annual Growth Report\n"
        f"{TRUTH_LINE}\n"
        "CEO note: figures unaudited.\n"
    )
    return corpus


async def _run_demo(corpus: Path) -> tuple[bool, bool, object]:
    """Returns (refuted, used_tool, message)."""
    settings = Settings(tool_files_root=str(corpus))
    logging_config.configure(settings.log_level)
    provider = get_provider(settings)
    try:
        registry = build_registry(provider, tools=build_tool_registry(settings))
        skeptic = registry.get(AgentRole.SKEPTIC)
        message = await skeptic.run(
            TASK,
            claims=[
                Claim(
                    id="c1",
                    statement=PLANTED_CLAIM,
                    status=ClaimStatus.UNVERIFIED,
                )
            ],
            run_id="refutation-demo",
        )
    finally:
        await provider.aclose()

    print("=" * 70)
    print("TOOL TRACE")
    print("=" * 70)
    if not message.tool_results:
        print("(no tools were executed)")
    for call, result in zip(message.tool_calls or [], message.tool_results or []):
        args = json.dumps(call.arguments, ensure_ascii=False)
        status = "ok" if result.error is None else f"error={result.error}"
        ms = f"{result.duration_ms:.1f}ms" if result.duration_ms is not None else ""
        print(f"  tool: {call.name}({args}) -> {status} {ms}")
        print(f"        {result.content[:200]}")

    print("=" * 70)
    print("VERDICTS")
    print("=" * 70)
    for verdict in message.verdicts or []:
        print(f"  [{verdict.claim_id}] {verdict.verdict.value} — {verdict.objection}")
        for evidence in verdict.evidence:
            print(f"      evidence: {evidence.source} | {evidence.quote}")

    verdict = next(
        (v for v in (message.verdicts or []) if v.claim_id == "c1"), None
    )
    refuted = verdict is not None and verdict.verdict is ClaimVerdict.REFUTED
    used_tool = bool(message.tool_results)
    return refuted, used_tool, verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep", action="store_true", help="keep the temp corpus for inspection"
    )
    args = parser.parse_args()

    corpus = _make_corpus()
    try:
        try:
            refuted, used_tool, verdict = asyncio.run(_run_demo(corpus))
        except Exception as exc:  # noqa: BLE001 - operator-facing gate
            print(f"run error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    finally:
        if args.keep:
            print(f"corpus kept at: {corpus}")
        else:
            shutil.rmtree(corpus, ignore_errors=True)

    print("=" * 70)
    if refuted and used_tool:
        print("PASS — skeptic refuted the planted 40% claim with tool evidence")
        return 0
    if not used_tool:
        print("FAIL — no tool was executed", file=sys.stderr)
    if not refuted:
        observed = verdict.verdict.value if verdict is not None else "no verdict"
        print(f"FAIL — planted claim verdict was '{observed}', not 'refuted'",
              file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
