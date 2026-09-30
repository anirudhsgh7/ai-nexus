#!/usr/bin/env python3
"""Run the full iterative pipeline in one command (Phases 4-6).

Usage (from backend/):
  python scripts/run_pipeline.py "Should a two-person startup write down decisions?"
  python scripts/run_pipeline.py "..." --verbose     # + claims/verdicts/tool calls
  python scripts/run_pipeline.py "..." --json        # NDJSON RunEvents

Tools are configured via AI_NEXUS_TOOL_* (see .env.example): file tools need
AI_NEXUS_TOOL_FILES_ROOT; web_search needs AI_NEXUS_TOOL_WEB_SEARCH_ENABLED=true.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import logging_config  # noqa: E402
from app.agents import build_registry  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm import get_provider  # noqa: E402
from app.orchestrator import Orchestrator  # noqa: E402
from app.runs import RunEvent, RunEventType, RunManager  # noqa: E402
from app.tools import build_tool_registry  # noqa: E402

def _fmt_step(event: RunEvent) -> str:
    message = event.message
    if event.skipped and message is None:
        detail = "skipped"
    elif event.kind is not None and event.kind.value == "decide" and message is not None:
        decision = message.decision
        if event.skipped and decision is not None:
            detail = f"forced finish — {decision.reason}"
        elif decision is not None:
            detail = decision.action.value + (
                f" → {decision.target.value}" if decision.target else ""
            )
        else:
            detail = ""
    elif message is None:
        detail = "no output"
    elif message.verdicts is not None:
        detail = f"{len(message.verdicts)} verdicts"
    elif message.claims is not None:
        detail = f"{len(message.claims)} claims"
    else:
        detail = ""
    if message is not None and message.tool_results:
        detail = f"tools={len(message.tool_results)} {detail}".strip()
    duration = (
        f"{event.duration_ms / 1000:6.1f}s"
        if event.duration_ms is not None
        else "   n/a"
    )
    name = (event.agent.value if event.agent else "?").capitalize()
    kind = event.kind.value if event.kind else "?"
    round_tag = f"r{event.round} " if event.round is not None else ""
    return f"[{event.step}] {name:<11}{kind:<11}{round_tag:<4}{duration}   {detail}"


def _print_details(event: RunEvent) -> None:
    message = event.message
    if message is None:
        return
    for call, result in zip(message.tool_calls or [], message.tool_results or []):
        args = json.dumps(call.arguments, ensure_ascii=False)
        if len(args) > 80:
            args = args[:80] + "..."
        status = "ok" if result.error is None else f"error={result.error}"
        ms = f"{result.duration_ms:.1f}ms" if result.duration_ms is not None else ""
        print(f"         tool: {call.name}({args}) → {status} {ms}")
    if message.decision is not None:
        print(f"         reason: {message.decision.reason}")
        if message.decision.instruction:
            print(f"         instruction: {message.decision.instruction}")
        return
    origin = message.from_agent.value
    for claim in message.claims or []:
        print(f"         [{claim.id}] ({origin}) {claim.status.value} — {claim.statement}")
    for verdict in message.verdicts or []:
        print(f"         [{verdict.claim_id}] {verdict.verdict.value} — {verdict.objection}")


async def run(args: argparse.Namespace) -> int:
    try:
        settings = get_settings()
        logging_config.configure(settings.log_level)
        provider = get_provider(settings)
        tool_registry = build_tool_registry(settings)
        registry = build_registry(provider, tools=tool_registry)
        store = RunManager()
        orchestrator = Orchestrator(registry, store, max_rounds=settings.max_rounds)
    except Exception as exc:
        print(f"setup error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        run_record = store.create(args.task)
        queue, _ = store.subscribe(run_record.id)
        task = asyncio.create_task(orchestrator.execute(run_record.id))
        store.register_task(run_record.id, task)

        while True:
            event = await queue.get()
            if args.json:
                print(event.model_dump_json(), flush=True)
            elif event.type is RunEventType.STEP_COMPLETED:
                print(_fmt_step(event), flush=True)
                if args.verbose:
                    _print_details(event)
            elif event.type is RunEventType.RUN_FAILED:
                error = event.error
                print(
                    f"\nRUN FAILED at step {event.step} ({event.kind.value if event.kind else '?'}): "
                    f"{error.type}: {error.message}",
                    file=sys.stderr,
                )
                if error.hint:
                    print(f"HINT: {error.hint}", file=sys.stderr)
                return 1
            elif event.type is RunEventType.RUN_COMPLETED:
                if not args.json:
                    final = event.message
                    print("=" * 60)
                    print("FINAL ANSWER")
                    print("=" * 60)
                    print((final.content if final else "").strip())
                return 0
    finally:
        await provider.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="the problem the team works on")
    parser.add_argument("--verbose", action="store_true",
                        help="print claims/verdicts under each step")
    parser.add_argument("--json", action="store_true",
                        help="emit NDJSON RunEvents (no human output)")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
