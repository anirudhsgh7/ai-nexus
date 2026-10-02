"""`WebSearchTool`: gate → cache → provider cascade → envelope composition."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import Tool, ToolContext
from app.tools.errors import ToolError
from app.tools.search.cache import SearchCache
from app.tools.search.constants import KNOWN_SEARCH_PROVIDERS
from app.tools.search.gate import SearchGate
from app.tools.search.providers import (
    AsyncClient,
    BingProvider,
    DuckDuckGoProvider,
    SearchOutcome,
)

_PROVIDER_CLASSES: dict[str, type] = {
    "ddg": DuckDuckGoProvider,
    "bing": BingProvider,
}

_GUIDANCE = (
    "Use file tools or clearly labeled model prior knowledge instead; "
    "do not claim external verification."
)


class _WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1, max_length=300)


class WebSearchTool(Tool):
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = (
        "Search the web and return the top results as title, url and snippet. "
        "Snippets are summaries, not full pages."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (max 300 characters).",
            },
        },
        "required": ["query"],
    }
    args_model: ClassVar[type[BaseModel]] = _WebSearchArgs

    def __init__(
        self,
        *,
        max_results: int = 5,
        timeout_s: float = 20.0,
        result_max_chars: int = 2000,
        providers: Sequence[str] = ("ddg", "bing"),
        retries: int = 2,
        backoff_base_s: float = 2.0,
        region: str = "us-en",
        min_interval_s: float = 3.0,
        cache_ttl_s: float = 900.0,
        gate: SearchGate | None = None,
        cache: SearchCache | None = None,
        ddg_client: AsyncClient | None = None,
        bing_client: AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(timeout_s=timeout_s, result_max_chars=result_max_chars)
        names = tuple(
            dict.fromkeys(
                str(name).strip().lower() for name in providers if str(name).strip()
            )
        )
        unknown = [name for name in names if name not in KNOWN_SEARCH_PROVIDERS]
        if unknown:
            known = ", ".join(sorted(KNOWN_SEARCH_PROVIDERS))
            raise ValueError(
                f"unknown web search provider(s): {', '.join(unknown)} "
                f"(known: {known})"
            )
        if not names:
            raise ValueError("at least one web search provider is required")
        self._max_results = max_results
        self._gate = gate if gate is not None else SearchGate(min_interval_s)
        self._cache = (
            cache if cache is not None else SearchCache(ttl_s=cache_ttl_s)
        )
        attempts = max(retries, 0) + 1
        request_timeout_s = max(5.0, timeout_s / attempts)
        clients = {"ddg": ddg_client, "bing": bing_client}
        providers_built = []
        for name in names:
            kwargs: dict[str, Any] = {
                "retries": retries,
                "backoff_base_s": backoff_base_s,
                "request_timeout_s": request_timeout_s,
                "client": clients[name],
                "sleep": sleep,
            }
            if name == "ddg":
                kwargs["region"] = region
            providers_built.append(_PROVIDER_CLASSES[name](**kwargs))
        self._providers = tuple(providers_built)

    async def execute(self, args: BaseModel, context: ToolContext) -> dict[str, Any]:
        assert isinstance(args, _WebSearchArgs)  # enforced by Tool.call
        await self._gate.acquire()
        query = args.query
        failures: list[tuple[str, str]] = []
        attempted: list[str] = []
        for provider in self._providers:
            cached = self._cache.get(provider.name, query)
            if cached is not None:
                outcome, attempts = cached
                return self._success_payload(
                    query, outcome, attempts, cached=True
                )
            attempted.append(provider.name)
            outcome = await provider.search(query, self._max_results)
            if not outcome.failed:
                attempts = tuple(attempted)
                self._cache.put(outcome, query, attempts=attempts)
                return self._success_payload(
                    query, outcome, attempts, cached=False
                )
            failures.append((provider.name, outcome.reason or "blocked"))
        if not failures:
            raise ToolError("network_error", "no web search providers are configured")
        detail = ", ".join(f"{name} ({reason})" for name, reason in failures)
        if all(reason.startswith("network_error") for _, reason in failures):
            raise ToolError(
                "network_error",
                f"web_search failed for all providers: {detail}. {_GUIDANCE}",
            )
        raise ToolError(
            "provider_blocked",
            f"web_search failed for all providers: {detail}. {_GUIDANCE}",
        )

    def _success_payload(
        self,
        query: str,
        outcome: SearchOutcome,
        attempts: Sequence[str],
        *,
        cached: bool,
    ) -> dict[str, Any]:
        return {
            "query": query,
            "provider": outcome.provider,
            "results": [dict(result) for result in outcome.results],
            "truncated": outcome.truncated,
            "attempts": list(attempts),
            "cached": cached,
        }
