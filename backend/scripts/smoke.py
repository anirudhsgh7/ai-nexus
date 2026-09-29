#!/usr/bin/env python3
"""One-shot LLM round trip against live Ollama.

Usage (from backend/): python scripts/smoke.py "your prompt"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import logging_config  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm import ChatMessage, ChatRole, LLMError, get_provider  # noqa: E402


async def run(prompt: str, model: str | None) -> int:
    settings = get_settings()
    logging_config.configure(settings.log_level)
    provider = get_provider(settings)
    try:
        messages = [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content="You are a precise assistant. Answer directly.",
            ),
            ChatMessage(role=ChatRole.USER, content=prompt),
        ]
        started = time.monotonic()
        result = await provider.chat(messages, model=model)
        wall_ms = round((time.monotonic() - started) * 1000, 1)

        print(f"provider : {provider.name}")
        print(f"model    : {result.model}")
        print(f"finish   : {result.finish_reason}")
        print(f"content  : {result.content.strip()}")
        print(
            f"tokens   : prompt={result.usage.prompt_tokens} "
            f"completion={result.usage.completion_tokens} total={result.usage.total_tokens}"
        )
        print(f"wall_ms  : {wall_ms}")
        return 0
    except LLMError as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        if exc.hint:
            print(f"HINT : {exc.hint}", file=sys.stderr)
        return 1
    finally:
        await provider.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="prompt to send")
    parser.add_argument("--model", default=None, help="override configured model")
    args = parser.parse_args()
    return asyncio.run(run(args.prompt, args.model))


if __name__ == "__main__":
    raise SystemExit(main())
