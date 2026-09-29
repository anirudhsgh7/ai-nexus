#!/usr/bin/env python3
"""Run the four AI Nexus agents independently on one task (Phase 2 proof).

NO ORCHESTRATION: every agent gets the same task and no other agent's output.
This demonstrates independent reasoning (spec §5), not a pipeline.

Phase 3 note: the Skeptic's output kind is verdicts, which requires input
claims. Since there is no orchestrator yet, the demo derives ONE premise
claim from the task itself (it is never another agent's output). Phase 4
replaces this with real researcher claims.

Usage (from backend/):
  python scripts/agents_demo.py "Should a small team adopt AI-assisted code review?"
  python scripts/agents_demo.py "..." --agents researcher skeptic --json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import logging_config  # noqa: E402
from app.agents import build_registry  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm import LLMError, get_provider  # noqa: E402
from app.schemas import AgentRole, Claim, ClaimStatus  # noqa: E402

ROLE_ORDER = [AgentRole.MANAGER, AgentRole.RESEARCHER, AgentRole.IDEATOR, AgentRole.SKEPTIC]


def _premise_claim(task: str) -> Claim:
    """Deterministic demo input for the Skeptic; not derived from any agent."""
    return Claim(
        id="c1",
        statement=f"The task is answerable as stated: {task}",
        status=ClaimStatus.UNVERIFIED,
        confidence=None,
        evidence=[],
    )


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    logging_config.configure(settings.log_level)
    provider = get_provider(settings)
    try:
        registry = build_registry(provider)
        roles = (
            [AgentRole(r) for r in args.agents]
            if args.agents
            else ROLE_ORDER
        )
        print("No orchestration in Phase 2: agents ran independently.\n", file=sys.stderr)

        for role in roles:
            agent = registry.get(role)
            run_kwargs = {}
            if role is AgentRole.SKEPTIC:
                run_kwargs["claims"] = [_premise_claim(args.task)]
                print(
                    "NOTE: no orchestrator yet — the Skeptic evaluates a single "
                    "task-premise claim (Phase 4 wires in researcher claims).\n",
                    file=sys.stderr,
                )
            started = time.monotonic()
            try:
                message = await agent.run(args.task, **run_kwargs)
            except LLMError as exc:
                print(f"[{agent.display_name}] ERROR {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                if exc.hint:
                    print(f"[{agent.display_name}] HINT : {exc.hint}", file=sys.stderr)
                return 1
            except Exception as exc:
                print(f"[{agent.display_name}] ERROR {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                return 1
            wall_s = time.monotonic() - started

            if args.json:
                print(message.model_dump_json())
            else:
                if message.verdicts is not None:
                    detail = f"{len(message.verdicts)} verdicts"
                elif message.claims is not None:
                    detail = f"{len(message.claims)} claims"
                else:
                    detail = "no structure"
                print("=" * 70)
                print(f"{agent.display_name.upper()}  ({message.type.value}, "
                      f"{wall_s:.1f}s, {len(message.content)} chars, {detail})")
                print("=" * 70)
                print(message.content.strip())
                print()
        return 0
    finally:
        await provider.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="the problem all agents work on")
    parser.add_argument(
        "--agents", nargs="+", choices=[r.value for r in ROLE_ORDER],
        help="subset of agents (default: all four)",
    )
    parser.add_argument("--json", action="store_true",
                        help="emit one AgentMessage JSON line per agent")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
