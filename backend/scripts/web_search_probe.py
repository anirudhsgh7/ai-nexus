#!/usr/bin/env python3
"""Phase 8b exit-criterion gate: measure web_search reliability.

Runs curated queries through the real `WebSearchTool` (same config path the
pipeline uses — no agent, no LLM) and reports per-provider hit rates.

Usage (from backend/):
  AI_NEXUS_TOOL_WEB_SEARCH_ENABLED=true .venv/bin/python \
      scripts/web_search_probe.py [--queries 3] [--delay 2] [--json]

Exit codes:
  0  overall hit rate >= 66% (the phase gate)
  1  overall hit rate below 66%
  2  setup error (flag off, bad config, provider construction failed)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import logging_config  # noqa: E402
from app.config import Settings  # noqa: E402
from app.schemas import AgentRole  # noqa: E402
from app.tools import ToolContext, build_tool_registry  # noqa: E402

QUERIES = [
    "machine learning engineer skills 2025",
    "SQL vs NoSQL databases",
    "kubernetes vs docker",
]

HIT_RATE_GATE = 0.66


def _recover_overflow(
    tool, settings: Settings, query: str, envelope: dict
) -> dict:
    """A large success becomes a `preview` envelope (Phase 6 `Tool.call` cap).

    The preview still carries the full payload inside its JSON string, but the
    top-level keys are gone. For honest per-provider reporting the probe reads
    the tool's cache after a successful call; probe-only introspection, no
    runtime path depends on it.
    """
    cache = getattr(tool, "_cache", None)
    if cache is not None:
        for name in settings.tool_web_search_providers:
            hit = cache.get(name, query)
            if hit is None:
                continue
            outcome, attempts = hit
            return {
                "ok": True,
                "provider": outcome.provider,
                "results": [dict(result) for result in outcome.results],
                "truncated": outcome.truncated,
                "attempts": list(attempts),
                "cached": False,
                "overflow": True,
            }
    return envelope


async def _probe(settings: Settings, queries: list[str], delay: float) -> list[dict]:
    registry = build_tool_registry(settings)
    tool = registry.get("web_search")
    context = ToolContext(agent=AgentRole.RESEARCHER, run_id="web-search-probe")
    rows: list[dict] = []
    for index, query in enumerate(queries):
        result = await tool.call({"query": query}, context)
        try:
            payload = json.loads(result.content)
        except ValueError:
            payload = {"ok": False, "error": "unparseable_envelope"}
        if payload.get("ok") and "provider" not in payload:
            payload = _recover_overflow(tool, settings, query, payload)
        row = {
            "query": query,
            "ok": bool(payload.get("ok")),
            "provider": payload.get("provider"),
            "results": len(payload.get("results", [])),
            "truncated": payload.get("truncated"),
            "attempts": payload.get("attempts", []),
            "cached": payload.get("cached"),
            "overflow": bool(payload.get("overflow")),
            "error": payload.get("error"),
            "duration_ms": result.duration_ms,
        }
        rows.append(row)
        if index < len(queries) - 1 and delay > 0:
            await asyncio.sleep(delay)
    return rows


def _summarize(settings: Settings, rows: list[dict]) -> dict:
    provider_stats = {
        name: {"successes": 0, "attempts": 0}
        for name in settings.tool_web_search_providers
    }
    for row in rows:
        if row["ok"]:
            providers = row["attempts"] or [row["provider"]]
            for name in providers:
                if name in provider_stats:
                    provider_stats[name]["attempts"] += 1
            winner = row["provider"]
            if winner in provider_stats:
                provider_stats[winner]["successes"] += 1
        else:
            # Failure envelopes carry the provider breakdown in their message
            # (frozen Tool.call has no structured extras), so count every
            # configured provider as an attempt for the failed query.
            for name in settings.tool_web_search_providers:
                provider_stats[name]["attempts"] += 1
    successes = sum(1 for row in rows if row["ok"])
    total = len(rows)
    return {
        "queries": total,
        "successes": successes,
        "hit_rate": (successes / total) if total else 0.0,
        "providers": provider_stats,
    }


def _print_human(rows: list[dict], summary: dict) -> None:
    print("=" * 78)
    for row in rows:
        status = "ok " if row["ok"] else f"error={row['error']}"
        provider = row["provider"] or "-"
        latency = f"{row['duration_ms']:.0f}ms" if row["duration_ms"] else "-"
        suffix = " (overflow preview)" if row["overflow"] else ""
        print(
            f"{row['query'][:44]:<46} {status:<32} provider={provider:<5} "
            f"results={row['results']:<2} attempts={','.join(row['attempts']) or '-':<10} "
            f"{latency}{suffix}"
        )
    print("=" * 78)
    print("PER-PROVIDER")
    for name, stats in summary["providers"].items():
        print(
            f"  {name}: {stats['successes']}/{stats['attempts']} successful calls"
        )
    print(
        f"OVERALL: {summary['successes']}/{summary['queries']} queries answered "
        f"({summary['hit_rate'] * 100:.0f}%)"
    )
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=3, help="number of queries")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between calls")
    parser.add_argument("--json", action="store_true", help="NDJSON output")
    args = parser.parse_args()

    settings = Settings()
    logging_config.configure(settings.log_level)
    if not settings.tool_web_search_enabled:
        print(
            "setup error: web_search is disabled. Run with "
            "AI_NEXUS_TOOL_WEB_SEARCH_ENABLED=true",
            file=sys.stderr,
        )
        return 2
    try:
        queries = (QUERIES * ((args.queries // len(QUERIES)) + 1))[: args.queries]
        rows = asyncio.run(_probe(settings, queries, args.delay))
    except Exception as exc:  # noqa: BLE001 - operator-facing gate
        print(f"setup error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    summary = _summarize(settings, rows)
    if args.json:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        print(json.dumps({"summary": summary}, ensure_ascii=False))
    else:
        _print_human(rows, summary)

    if summary["hit_rate"] >= HIT_RATE_GATE:
        print(
            f"PASS — overall hit rate {summary['hit_rate'] * 100:.0f}% "
            f">= {HIT_RATE_GATE * 100:.0f}%"
        )
        return 0
    print(
        f"FAIL — overall hit rate {summary['hit_rate'] * 100:.0f}% "
        f"< {HIT_RATE_GATE * 100:.0f}%",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
