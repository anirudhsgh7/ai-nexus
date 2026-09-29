#!/usr/bin/env python3
"""Benchmark local models and recommend primary_model (PRD §6.8).

Usage (from backend/): python scripts/benchmark.py [--runs 3] [--models ...]

Decision rule: a model passes if median tok/s >= 5.0 AND warm TTFT <= 15.0s
AND structured-output validity = 100%. Models are tested in preference order
(first listed wins); the recommendation is the first pass, else the last tested.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import logging_config  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm import (  # noqa: E402
    ChatMessage,
    ChatRole,
    LLMError,
    LLMProvider,
    get_provider,
)

DEFAULT_MODELS = ["qwen2.5:14b-instruct", "qwen2.5:7b-instruct"]
MIN_TOK_PER_S = 5.0
MAX_WARM_TTFT_S = 15.0

P2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "confidence"],
}


def prompts() -> list[tuple[str, str, int, float, dict | str | None]]:
    """(id, text, max_tokens, temperature, response_format)"""
    return [
        ("P1_short_factual", "What is the capital of Australia? One word only.", 50, 0.0, None),
        ("P2_structured_json", "State that the capital of Australia is Canberra, with a confidence score.", 200, 0.0, P2_SCHEMA),
        ("P3_multi_step_reasoning", "Explain step by step why the sky appears blue during the day and red at sunset. Be thorough.", 300, 0.4, None),
    ]


def _msgs(text: str) -> list[ChatMessage]:
    return [ChatMessage(role=ChatRole.USER, content=text)]


async def _first_call_walls(provider: LLMProvider, model: str) -> float:
    started = time.monotonic()
    await provider.chat(_msgs("Reply with the single word: ok."), model=model, max_tokens=10)
    return time.monotonic() - started


async def _ttft(provider: LLMProvider, model: str) -> float:
    started = time.monotonic()
    async for chunk in provider.stream(_msgs("Count from 1 to 10."), model=model, max_tokens=40):
        if chunk.delta:
            return time.monotonic() - started
    return time.monotonic() - started


def _structured_valid(content: str) -> bool:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return False
    return isinstance(data, dict) and all(k in data for k in P2_SCHEMA["required"])


async def bench_model(provider: LLMProvider, model: str, runs: int) -> dict[str, Any]:
    print(f"\n=== {model} ===")
    result: dict[str, Any] = {"model": model, "installed": True}

    cold_s = await _first_call_walls(provider, model)
    print(f"  cold load+first completion : {cold_s:.2f}s")
    result["cold_first_call_s"] = round(cold_s, 2)

    ttft_s = await _ttft(provider, model)
    print(f"  warm TTFT (stream)         : {ttft_s:.2f}s")
    result["warm_ttft_s"] = round(ttft_s, 2)

    prompt_results = []
    for pid, text, max_tokens, temperature, fmt in prompts():
        samples: list[dict[str, Any]] = []
        for i in range(runs):
            try:
                out = await provider.chat(
                    _msgs(text), model=model, max_tokens=max_tokens,
                    temperature=temperature, response_format=fmt,
                )
            except LLMError as exc:
                print(f"    {pid} run {i + 1}: ERROR {exc}")
                samples.append({"error": str(exc)})
                continue
            eval_s = (out.usage.eval_duration_ms or 0) / 1000
            tok_s = (out.usage.completion_tokens / eval_s) if eval_s > 0 else None
            entry = {
                "completion_tokens": out.usage.completion_tokens,
                "prompt_tokens": out.usage.prompt_tokens,
                "eval_duration_ms": out.usage.eval_duration_ms,
                "tokens_per_s": round(tok_s, 2) if tok_s else None,
                "wall_s": None,
            }
            if fmt is not None:
                entry["structured_valid"] = _structured_valid(out.content)
            samples.append(entry)
            print(f"    {pid} run {i + 1}: {entry['tokens_per_s']} tok/s, "
                  f"valid={entry.get('structured_valid', '-')}")
        prompt_results.append({"prompt": pid, "runs": samples})

    result["prompts"] = prompt_results

    tok_values = [
        s["tokens_per_s"]
        for p in prompt_results
        for s in p["runs"]
        if s.get("tokens_per_s")
    ]
    med = statistics.median(tok_values) if tok_values else 0.0
    result["median_tokens_per_s"] = round(med, 2)

    p2 = next(p for p in prompt_results if p["prompt"] == "P2_structured_json")
    valid_runs = [s for s in p2["runs"] if "structured_valid" in s]
    validity = (
        100.0 * sum(1 for s in valid_runs if s["structured_valid"]) / len(valid_runs)
        if valid_runs else 0.0
    )
    result["structured_validity_pct"] = validity

    result["checks"] = {
        "tok_per_s": med >= MIN_TOK_PER_S,
        "ttft": ttft_s <= MAX_WARM_TTFT_S,
        "structured_validity": validity == 100.0,
    }
    result["passes"] = all(result["checks"].values())
    print(f"  median {med} tok/s | TTFT {ttft_s:.1f}s | structured {validity}% "
          f"-> {'PASS' if result['passes'] else 'FAIL'}")
    return result


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    logging_config.configure(settings.log_level)
    provider = get_provider(settings)
    try:
        try:
            installed = {m.name for m in await provider.list_models()}
        except LLMError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            if exc.hint:
                print(f"HINT : {exc.hint}", file=sys.stderr)
            return 1

        results = []
        for model in args.models:
            if model not in installed:
                print(f"SKIPPED {model}: not installed. Pull with: ollama pull {model}")
                results.append({"model": model, "installed": False, "passes": False})
                continue
            try:
                results.append(await bench_model(provider, model, args.runs))
            except LLMError as exc:
                print(f"ERROR benchmarking {model}: {exc}")
                results.append({"model": model, "installed": True, "passes": False, "error": str(exc)})

        recommendation = None
        for res in results:
            if res.get("installed") and res.get("passes"):
                recommendation = res["model"]
                break
        if recommendation is None:
            tested = [r for r in results if r.get("installed")]
            recommendation = tested[-1]["model"] if tested else None

        report = {
            "hardware": {"platform": sys.platform},
            "criteria": {
                "min_median_tokens_per_s": MIN_TOK_PER_S,
                "max_warm_ttft_s": MAX_WARM_TTFT_S,
                "structured_validity_pct": 100.0,
            },
            "results": results,
            "recommendation": recommendation,
        }
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nRECOMMENDATION: primary_model={recommendation}")
        print(f"Results written to {args.out}")
        if recommendation is None:
            return 1
        if not next(r for r in results if r["model"] == recommendation).get("passes"):
            print("WARNING: no model passed all criteria; recommendation is the last tested model.")
        return 0
    finally:
        await provider.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "benchmark_results.json"))
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
