from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Request, Response

from app.llm import LLMProvider, ProviderHealth

logger = logging.getLogger("ai_nexus.api.health")

router = APIRouter(tags=["health"])


def build_health_payload(
    settings: Any,
    provider_name: str,
    health: ProviderHealth,
    *,
    started: float,
) -> tuple[int, dict[str, Any]]:
    """Pure function so status semantics are unit-testable without a live server."""
    models = [
        {
            "name": m.name,
            "size_bytes": m.size_bytes,
            "parameter_size": m.parameter_size,
            "quantization": m.quantization,
            "context_length": m.context_length,
            "capabilities": m.capabilities,
        }
        for m in health.models
    ]
    names = {m.name for m in health.models}
    primary_ok = settings.primary_model in names
    fallback_ok = settings.fallback_model in names

    if not health.reachable:
        status, status_code = "unavailable", 503
        hint = "Is Ollama running? Start it with: ollama serve"
    elif not primary_ok:
        status, status_code = "degraded", 200
        hint = f"Pull the primary model: ollama pull {settings.primary_model}"
    else:
        status, status_code = "ok", 200
        hint = ""

    payload: dict[str, Any] = {
        "status": status,
        "app": {"name": settings.app_name, "version": settings.app_version},
        "provider": {
            "name": provider_name,
            "base_url": settings.ollama_base_url,
            "reachable": health.reachable,
            "latency_ms": health.latency_ms,
            "server_version": health.server_version,
            "models": models,
            **({"error": health.error} if health.error else {}),
        },
        "config": {
            "primary_model": settings.primary_model,
            "fallback_model": settings.fallback_model,
            "num_ctx": settings.num_ctx,
            "max_concurrent_generations": settings.max_concurrent_generations,
        },
        "checks": {
            "primary_model_available": primary_ok,
            "fallback_model_available": fallback_ok,
        },
        "took_ms": round((time.monotonic() - started) * 1000, 2),
        **({"hint": hint} if hint else {}),
    }
    return status_code, payload


@router.get("/health")
async def get_health(request: Request, fresh: bool = False) -> Response:
    settings = request.app.state.settings
    provider: LLMProvider = request.app.state.provider

    cache: tuple[float, int, dict[str, Any]] | None = getattr(
        request.app.state, "health_cache", None
    )
    now = time.monotonic()
    if not fresh and cache is not None and now - cache[0] <= settings.health_cache_ttl_s:
        return Response(
            content=json.dumps(cache[2]).encode(),
            status_code=cache[1],
            media_type="application/json",
        )

    started = time.monotonic()
    try:
        health = await provider.health()
    except Exception:
        logger.exception("health check raised unexpectedly")
        health = ProviderHealth(reachable=False, error="internal error during health check")

    status_code, payload = build_health_payload(
        settings, provider.name, health, started=started
    )
    request.app.state.health_cache = (time.monotonic(), status_code, payload)
    return Response(
        content=json.dumps(payload).encode(),
        status_code=status_code,
        media_type="application/json",
    )
