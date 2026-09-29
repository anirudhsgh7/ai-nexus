"""Provider factory: the only place that names a concrete provider class."""

from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.ollama import OllamaProvider

_PROVIDERS: dict[str, type[OllamaProvider]] = {"ollama": OllamaProvider}


def get_provider(settings: Settings) -> LLMProvider:
    provider_name = "ollama"  # Phase 1: only provider; config-driven selection comes later
    try:
        provider_cls = _PROVIDERS[provider_name]
    except KeyError:
        raise ValueError(f"unknown LLM provider: {provider_name!r}") from None
    return provider_cls(settings)
