from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import logging_config
from app.agents import build_registry
from app.api import health, runs as runs_api
from app.config import get_settings
from app.llm import (
    LLMError,
    LLMProvider,
    ModelNotFoundError,
    ProviderUnavailableError,
    RequestTimeoutError,
    get_provider,
)
from app.orchestrator import Orchestrator
from app.runs import RunManager
from app.tools import build_tool_registry

logger = logging.getLogger("ai_nexus.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging_config.configure(settings.log_level)
    provider: LLMProvider = get_provider(settings)
    runs = RunManager()
    tool_registry = build_tool_registry(settings)
    app.state.provider = provider
    app.state.settings = settings
    app.state.health_cache = None
    app.state.tool_registry = tool_registry
    app.state.registry = build_registry(provider, tools=tool_registry)
    app.state.runs = runs
    app.state.orchestrator = Orchestrator(app.state.registry, runs)
    logger.info(
        "app_start version=%s base_url=%s model=%s num_ctx=%s tools=%s",
        settings.app_version,
        settings.ollama_base_url,
        settings.primary_model,
        settings.num_ctx,
        ",".join(tool_registry.names) or "none",
    )
    try:
        yield
    finally:
        active = runs.cancel_active_task()
        if active is not None:
            with suppress(asyncio.CancelledError):
                await active
        await provider.aclose()
        logger.info("app_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    fastapi_app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    fastapi_app.include_router(health.router, prefix="/api")
    fastapi_app.include_router(runs_api.router, prefix="/api")

    @fastapi_app.exception_handler(LLMError)
    async def _llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        status = 502
        if isinstance(exc, ProviderUnavailableError):
            status = 503
        elif isinstance(exc, ModelNotFoundError):
            status = 424
        elif isinstance(exc, RequestTimeoutError):
            status = 504
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "hint": exc.hint,
                }
            },
        )

    return fastapi_app


app = create_app()
