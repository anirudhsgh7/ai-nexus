#!/usr/bin/env python3
"""Recommended API runner: uvicorn with a bounded graceful shutdown.

Plain `uvicorn app.main:app` waits *indefinitely* for open connections — and
an attached SSE stream never ends while a run is live — which delays lifespan
shutdown (and the run cancellation inside it) until the client disconnects.
This wrapper passes `timeout_graceful_shutdown` so uvicorn force-closes
lingering connections after `AI_NEXUS_SHUTDOWN_GRACE_S` (default 5 s) and the
app shuts down promptly: streams closed, run marked `ServerShutdown`, provider
released (Phase 10 PRD §6.1).

Usage (from backend/):
  .venv/bin/python scripts/serve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402

from app.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        timeout_graceful_shutdown=settings.shutdown_grace_s,
    )


if __name__ == "__main__":
    main()
