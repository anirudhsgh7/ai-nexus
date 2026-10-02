#!/usr/bin/env python3
"""Phase 10 eval harness: run the three controlled problems, score, report.

Runs each problem through the REAL orchestrator + Ollama (in-process — the same
code path as run_pipeline.py, no API), scores it with the offline-tested
predicates in `app/eval.py`, aggregates refutation rate / rounds / tokens /
wall time, and writes `eval_results.json` (gitignored). Operator rubric scores
live in the report's `human_scores` fields (fill them for the phase gate).

Usage (from backend/):
  .venv/bin/python scripts/eval.py [--problems p1 p2 p3] [--out eval_results.json]
                                   [--db data/eval.db] [--json] [--reruns 1]
  .venv/bin/python scripts/eval.py --rescore eval_results.json [--out ...]

Exit codes:
  0  every mandatory check passed on every problem
  1  at least one mandatory check failed (advisory results never gate)
  2  setup error (Ollama unreachable, model missing, unwritable output,
     rescore input unreadable/unknown run)

`--rescore` re-runs only the SCORING over runs already persisted in `--db`
(no model calls): useful after predicate changes and for Phase 11 rubric
iterations. Tokens/wall/attempts are preserved from the input report.

Notes:
  * Per-request timeout defaults to 900 s (export AI_NEXUS_REQUEST_TIMEOUT_S
    to override) — tool-heavy 14B structured calls exceed the 300 s default.
  * A failed problem is re-run once (`--reruns`, default 1), visibly recorded
    as `attempts: 2`; a second failure is final.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import logging_config  # noqa: E402
from app.agents import build_registry  # noqa: E402
from app.config import Settings  # noqa: E402
from app.db import build_run_store  # noqa: E402
from app.eval import (  # noqa: E402
    PROBLEMS,
    HumanScores,
    ProblemResult,
    ProblemSpec,
    RecordingProvider,
    UsageTotals,
    build_report,
    score_problem,
    verdict_counts,
)
from app.llm import get_provider  # noqa: E402
from app.orchestrator import Orchestrator  # noqa: E402
from app.runs import RunManager, RunRecord, RunStatus  # noqa: E402
from app.tools import build_tool_registry  # noqa: E402


def _settings_for(spec: ProblemSpec, db_path: str, corpus_dir: str | None) -> Settings:
    return Settings(
        db_path=db_path,
        tool_files_root=corpus_dir or "",
        tool_web_search_enabled=spec.web_search,
        max_rounds=spec.max_rounds,
        # tool-heavy 14B calls exceed the 300 s default; env still wins when set
        request_timeout_s=float(os.environ.get("AI_NEXUS_REQUEST_TIMEOUT_S", "900")),
    )


async def _preflight() -> tuple[Settings, object]:
    """Exit-2 class checks: provider reachable, primary model present."""
    settings = Settings()
    logging_config.configure(settings.log_level)
    inner = get_provider(settings)
    health = await inner.health()
    if not health.reachable:
        await inner.aclose()
        print(
            f"setup error: Ollama unreachable at {settings.ollama_base_url} "
            f"({health.error}). Start it with: ollama serve",
            file=sys.stderr,
        )
        raise SystemExit(2)
    names = [model.name for model in health.models]
    if settings.primary_model not in names:
        await inner.aclose()
        print(
            f"setup error: model {settings.primary_model} not present. "
            f"Pull it with: ollama pull {settings.primary_model}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return settings, inner


async def _run_problem(
    spec: ProblemSpec,
    db_path: str,
    reruns: int,
) -> tuple[ProblemResult, RunRecord]:
    corpus_dir: str | None = None
    if spec.seeding_required:
        corpus_dir = tempfile.mkdtemp(prefix=f"eval_{spec.id}_")
        assert spec.corpus is not None
        for name, content in spec.corpus:
            (Path(corpus_dir) / name).write_text(content)

    try:
        settings = _settings_for(spec, db_path, corpus_dir)
        attempts_allowed = 1 + max(reruns, 0)
        result: ProblemResult | None = None
        run: RunRecord | None = None
        for attempt in range(1, attempts_allowed + 1):
            provider = RecordingProvider(get_provider(settings))
            try:
                registry = build_registry(
                    provider, tools=build_tool_registry(settings)
                )
                runs = RunManager(store=build_run_store(settings))
                orchestrator = Orchestrator(registry, runs, max_rounds=spec.max_rounds)
                run = runs.create(spec.task)
                started = time.monotonic()
                await orchestrator.execute(run.id)
                wall_ms = (time.monotonic() - started) * 1000.0
                tokens = provider.totals
            finally:
                await provider.aclose()

            if run.status is not RunStatus.COMPLETED and attempt < attempts_allowed:
                reason = run.error.message if run.error else "?"
                print(
                    f"[{spec.id}] attempt {attempt} failed ({reason}) "
                    f"— re-running once…"
                )
                continue

            score = score_problem(spec, run)
            result = ProblemResult(
                problem_id=spec.id,
                title=spec.title,
                task=spec.task,
                run_id=run.id,
                status=run.status.value,
                attempts=attempt,
                rounds=len(run.rounds),
                steps=len(run.steps),
                verdicts=verdict_counts(run),
                tokens=tokens,
                wall_ms=wall_ms,
                mandatory=score.mandatory,
                advisory=score.advisory,
                passed=score.passed,
            )
            break
        assert result is not None and run is not None
        return result, run
    finally:
        if corpus_dir is not None:
            shutil.rmtree(corpus_dir, ignore_errors=True)


def _print_row(result: ProblemResult) -> None:
    verdicts = result.verdicts
    tokens = result.tokens
    print(
        f"{result.problem_id:<4} {result.status:<10} "
        f"rounds={result.rounds} steps={result.steps} "
        f"v(s/r/u)={verdicts['supported']}/{verdicts['refuted']}/"
        f"{verdicts['unverifiable']} "
        f"tok={tokens.total_tokens} wall={round((result.wall_ms or 0) / 1000)}s "
        f"mandatory={sum(c.passed for c in result.mandatory)}/"
        f"{len(result.mandatory)} "
        f"advisory={sum(c.passed for c in result.advisory)}/"
        f"{len(result.advisory)}"
    )
    for check in result.mandatory:
        if not check.passed:
            print(f"     FAIL mandatory {check.name}: {check.detail}")
    for check in result.advisory:
        if not check.passed:
            print(f"     note advisory {check.name}: {check.detail}")


def _print_summary(summary: dict[str, object]) -> None:
    print("=" * 70)
    print(
        f"MANDATORY {summary['mandatory_passed']}/{summary['mandatory_total']}  "
        f"ADVISORY {summary['advisory_passed']}/{summary['advisory_total']}"
    )
    print(
        f"refutation_rate={summary['refutation_rate']} "
        f"(refuted={summary['refuted']} supported={summary['supported']} "
        f"unverifiable={summary['unverifiable']})"
    )
    print(f"rounds_per_problem={summary['rounds_per_problem']}")
    print(
        f"tokens_total={summary['tokens_total']} "
        f"wall_total={round((summary['wall_ms_total'] or 0) / 1000)}s"
    )
    print(f"all_passed={summary['all_passed']}")
    print("=" * 70)
    print(
        "Next: fill human_scores (1–5 correctness/evidence/honesty + notes) in "
        "the report — that rubric is the Phase 11 baseline."
    )


def _tokens_from_report(entry: dict) -> UsageTotals:
    tokens = entry.get("tokens") or {}
    return UsageTotals(
        prompt_tokens=int(tokens.get("prompt", 0)),
        completion_tokens=int(tokens.get("completion", 0)),
        total_tokens=int(tokens.get("total", 0)),
        calls=int(tokens.get("calls", 0)),
    )


def _rescore(args: argparse.Namespace) -> int:
    """Re-score persisted runs from a prior report (no LLM calls)."""
    source = Path(args.rescore)
    if not source.exists():
        print(f"setup error: report not found: {source}", file=sys.stderr)
        return 2
    previous = json.loads(source.read_text())
    store = build_run_store(Settings(db_path=args.db))
    if store is None:
        print(
            f"setup error: --db {args.db} disabled; cannot rescore",
            file=sys.stderr,
        )
        return 2
    runs = RunManager(store=store)

    results: list[ProblemResult] = []
    for entry in previous["problems"]:
        spec = next((p for p in PROBLEMS if p.id == entry["id"]), None)
        if spec is None:
            print(f"setup error: unknown problem id {entry['id']!r}", file=sys.stderr)
            return 2
        run = runs.get(entry["run_id"])
        if run is None:
            print(
                f"setup error: run {entry['run_id']} not found in {args.db}",
                file=sys.stderr,
            )
            return 2
        score = score_problem(spec, run)
        result = ProblemResult(
            problem_id=spec.id,
            title=spec.title,
            task=spec.task,
            run_id=run.id,
            status=run.status.value,
            attempts=int(entry.get("attempts", 1)),
            rounds=len(run.rounds),
            steps=len(run.steps),
            verdicts=verdict_counts(run),
            tokens=_tokens_from_report(entry),
            wall_ms=entry.get("wall_ms"),
            mandatory=score.mandatory,
            advisory=score.advisory,
            passed=score.passed,
        )
        # The operator rubric is never rebuilt: rescoring must not erase it.
        scores = entry.get("human_scores") or {}
        if any(value is not None for key, value in scores.items() if key != "notes"):
            result.human_scores = HumanScores(
                correctness=scores.get("correctness"),
                evidence_use=scores.get("evidence_use"),
                honesty=scores.get("honesty"),
                notes=scores.get("notes", ""),
            )
        results.append(result)

    report = build_report(
        results,
        app_version=previous.get("app_version", ""),
        model=previous.get("model", ""),
        num_ctx=int(previous.get("num_ctx", 0)),
        params=dict(previous.get("params") or {}),
    )
    Path(args.out).write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        for result in results:
            _print_row(result)
        _print_summary(report.summary)
        print(f"rescored report written: {Path(args.out).resolve()}")
    return 0 if report.summary["all_passed"] else 1


async def _amain(args: argparse.Namespace) -> int:
    base, inner = await _preflight()
    problems = [p for p in PROBLEMS if p.id in set(args.problems)]

    results: list[ProblemResult] = []
    for spec in problems:
        print(f"--- {spec.id}: {spec.title} ---")
        result, _run = await _run_problem(spec, args.db, args.reruns)
        _print_row(result)
        results.append(result)
    await inner.aclose()

    report = build_report(
        results,
        app_version=base.app_version,
        model=base.primary_model,
        num_ctx=base.num_ctx,
        params={"max_rounds": base.max_rounds, "web_search": False},
    )
    out_path = Path(args.out)
    out_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n")

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        _print_summary(report.summary)
        print(f"report written: {out_path.resolve()}")
    return 0 if report.summary["all_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems", nargs="+", choices=[p.id for p in PROBLEMS],
        default=[p.id for p in PROBLEMS],
    )
    parser.add_argument("--out", default="eval_results.json")
    parser.add_argument("--db", default="data/eval.db")
    parser.add_argument("--json", action="store_true", help="report JSON to stdout")
    parser.add_argument(
        "--reruns", type=int, default=1,
        help="re-runs after a failed problem (default 1; recorded as attempts)",
    )
    parser.add_argument(
        "--rescore", metavar="REPORT",
        help="re-score runs from a prior eval_results.json (no model calls)",
    )
    args = parser.parse_args()
    if args.rescore:
        return _rescore(args)
    try:
        return asyncio.run(_amain(args))
    except SystemExit as exc:  # preflight's exit-2 path
        return int(exc.code or 2)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
