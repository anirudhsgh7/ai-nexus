from __future__ import annotations

import logging
from typing import Any

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"


def configure(level: str = "INFO") -> None:
    """Configure root logging. Called once from app lifespan."""
    logging.basicConfig(level=level.upper(), format=_FORMAT, force=True)
    for noisy in ("httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _kv(**fields: Any) -> str:
    return " ".join(f"{k}={v}" for k, v in fields.items())


def llm_request_start(
    logger: logging.Logger, *, provider: str, model: str, num_ctx: int,
    keep_alive: str, message_count: int, stream: bool, temperature: float,
) -> None:
    logger.info(
        "llm_request_start %s",
        _kv(provider=provider, model=model, num_ctx=num_ctx, keep_alive=keep_alive,
            message_count=message_count, stream=stream, temperature=temperature),
    )


def llm_request_end(logger: logging.Logger, **fields: Any) -> None:
    logger.info("llm_request_end %s", _kv(**fields))


def llm_request_error(logger: logging.Logger, **fields: Any) -> None:
    logger.error("llm_request_error %s", _kv(**fields))


def context_pressure(logger: logging.Logger, **fields: Any) -> None:
    logger.warning("context_pressure %s", _kv(**fields))


def context_truncation_suspected(logger: logging.Logger, **fields: Any) -> None:
    logger.warning("context_truncation_suspected %s", _kv(**fields))


def provider_unavailable(logger: logging.Logger, **fields: Any) -> None:
    logger.warning("provider_unavailable %s", _kv(**fields))


def upstream_error(logger: logging.Logger, **fields: Any) -> None:
    logger.error("upstream_error %s", _kv(**fields))


def agent_run_start(logger: logging.Logger, **fields: Any) -> None:
    logger.info("agent_run_start %s", _kv(**fields))


def agent_run_end(logger: logging.Logger, **fields: Any) -> None:
    logger.info("agent_run_end %s", _kv(**fields))


def agent_run_error(logger: logging.Logger, **fields: Any) -> None:
    logger.error("agent_run_error %s", _kv(**fields))


def agent_structured_retry(logger: logging.Logger, **fields: Any) -> None:
    logger.warning("agent_structured_retry %s", _kv(**fields))
