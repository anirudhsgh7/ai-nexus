"""Search providers (the only `app/tools/search/` module importing httpx)."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from app import logging_config as log
from app.tools.search.constants import (
    BING_EXTRA_HEADERS,
    BING_URL,
    DDG_EXTRA_HEADERS,
    DDG_URL,
    headers_for,
    ua_for,
)
from app.tools.search.parsers import (
    classify_response,
    parse_bing_results,
    parse_results,
)

logger = logging.getLogger("ai_nexus.tools.web_search")

#: Re-exported so `tool.py` can annotate injected clients without importing
#: httpx itself (the layer rule confines httpx to this module).
AsyncClient = httpx.AsyncClient


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    provider: str
    results: tuple[dict[str, str], ...]
    truncated: bool
    blocked: bool
    reason: str = ""

    @property
    def failed(self) -> bool:
        """Anti-bot block OR transport failure: both cascade and are uncached."""
        return self.blocked or self.reason.startswith("network_error")


def _attempts_label(attempts: int) -> str:
    return "attempt" if attempts == 1 else "attempts"


# Process-lifetime clients so cookies persist across calls (a first-visit
# stranger is exactly what the challenge detector screens for). Deliberately
# never closed: the server process owns their lifecycle. Tests inject their
# own clients instead of touching these.
_CLIENTS: dict[str, httpx.AsyncClient] = {}


def _default_client(provider: str) -> httpx.AsyncClient:
    client = _CLIENTS.get(provider)
    if client is None:
        client = httpx.AsyncClient(follow_redirects=True)
        _CLIENTS[provider] = client
    return client


def _is_bing_consent(response: httpx.Response) -> bool:
    """Bing occasionally lands on consent.bing.com / an inline consent wall."""
    location = response.headers.get("location", "")
    if "consent" in location.lower():
        return True
    if "consent" in (response.url.host or "").lower():
        return True
    low = response.text.lower()
    return "consent.bing.com" in low or 'id="b_consent"' in low


class DuckDuckGoProvider:
    """Primary provider: browser-faithful POST to the html endpoint."""

    name = "ddg"

    def __init__(
        self,
        *,
        retries: int = 2,
        backoff_base_s: float = 2.0,
        region: str = "us-en",
        request_timeout_s: float = 15.0,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._retries = retries
        self._backoff_base_s = backoff_base_s
        self._region = region
        self._request_timeout_s = request_timeout_s
        self._client = client
        self._sleep = sleep or asyncio.sleep

    def _http(self) -> httpx.AsyncClient:
        return self._client if self._client is not None else _default_client("ddg")

    async def search(self, query: str, max_results: int) -> SearchOutcome:
        attempts = self._retries + 1
        client = self._http()
        last_status = 0
        for attempt in range(attempts):
            headers = headers_for(ua_for(query, attempt), DDG_EXTRA_HEADERS)
            try:
                response = await client.post(
                    DDG_URL,
                    data={"q": query, "b": "", "kl": self._region},
                    headers=headers,
                    timeout=self._request_timeout_s,
                )
            except httpx.HTTPError as exc:
                # Transport failures are fed to the model once, never retried.
                return SearchOutcome(
                    provider=self.name,
                    results=(),
                    truncated=False,
                    blocked=False,
                    reason=f"network_error: {type(exc).__name__}",
                )
            body = response.text
            results: list[dict[str, str]] = []
            truncated = False
            if response.status_code == 200:
                results, truncated = parse_results(body, max_results)
            verdict = classify_response(response.status_code, body, len(results))
            backoff = self._backoff_for(attempt, attempts, verdict)
            log.web_search_attempt(
                logger,
                provider=self.name,
                attempt=attempt + 1,
                status=response.status_code,
                blocked=verdict != "ok",
                reason=verdict,
                backoff_ms=round(backoff * 1000),
            )
            if verdict == "ok":
                return SearchOutcome(
                    provider=self.name,
                    results=tuple(results),
                    truncated=truncated,
                    blocked=False,
                )
            last_status = response.status_code
            if backoff > 0:
                await self._sleep(backoff)
        return SearchOutcome(
            provider=self.name,
            results=(),
            truncated=False,
            blocked=True,
            reason=self._exhausted_reason(last_status, attempts),
        )

    def _backoff_for(self, attempt: int, attempts: int, verdict: str) -> float:
        if verdict == "ok" or attempt >= attempts - 1:
            return 0.0
        return self._backoff_base_s * (2**attempt) + random.uniform(0, 0.5)

    @staticmethod
    def _exhausted_reason(status: int, attempts: int) -> str:
        if status == 200:
            what = "empty/challenge page"
        else:
            what = f"HTTP {status} bot challenge"
        return f"{what} after {attempts} {_attempts_label(attempts)}"


class BingProvider:
    """Free fallback provider: plain GET against the Bing SERP."""

    name = "bing"

    def __init__(
        self,
        *,
        retries: int = 2,
        backoff_base_s: float = 2.0,
        request_timeout_s: float = 15.0,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._retries = retries
        self._backoff_base_s = backoff_base_s
        self._request_timeout_s = request_timeout_s
        self._client = client
        self._sleep = sleep or asyncio.sleep

    def _http(self) -> httpx.AsyncClient:
        return self._client if self._client is not None else _default_client("bing")

    async def search(self, query: str, max_results: int) -> SearchOutcome:
        attempts = self._retries + 1
        client = self._http()
        consent_retried = False
        last_status = 0
        for attempt in range(attempts):
            headers = headers_for(ua_for(query, attempt), BING_EXTRA_HEADERS)
            try:
                response = await client.get(
                    BING_URL,
                    params={
                        "q": query,
                        "setlang": "en",
                        "mkt": "en-US",
                        "count": max_results,
                        "form": "QBLH",
                    },
                    headers=headers,
                    timeout=self._request_timeout_s,
                )
            except httpx.HTTPError as exc:
                return SearchOutcome(
                    provider=self.name,
                    results=(),
                    truncated=False,
                    blocked=False,
                    reason=f"network_error: {type(exc).__name__}",
                )
            body = response.text
            consent = _is_bing_consent(response)
            results: list[dict[str, str]] = []
            truncated = False
            if response.status_code == 200 and not consent:
                results, truncated = parse_bing_results(body, max_results)
            verdict = (
                "challenge"
                if consent
                else classify_response(response.status_code, body, len(results))
            )
            backoff = self._backoff_for(attempt, attempts, verdict)
            if consent and not consent_retried and attempt < attempts - 1:
                # A locale preference for the consent wall, not a credential.
                client.cookies.set("SRCHHPGUSR", "SRCHLANG=en", domain=".bing.com")
                consent_retried = True
            log.web_search_attempt(
                logger,
                provider=self.name,
                attempt=attempt + 1,
                status=response.status_code,
                blocked=verdict != "ok",
                reason="consent" if consent else verdict,
                backoff_ms=round(backoff * 1000),
            )
            if verdict == "ok":
                return SearchOutcome(
                    provider=self.name,
                    results=tuple(results),
                    truncated=truncated,
                    blocked=False,
                )
            last_status = response.status_code
            if backoff > 0:
                await self._sleep(backoff)
        return SearchOutcome(
            provider=self.name,
            results=(),
            truncated=False,
            blocked=True,
            reason=(
                f"consent wall/challenge after {attempts} {_attempts_label(attempts)}"
                if last_status == 200
                else f"HTTP {last_status} after {attempts} {_attempts_label(attempts)}"
            ),
        )

    def _backoff_for(self, attempt: int, attempts: int, verdict: str) -> float:
        if verdict == "ok" or attempt >= attempts - 1:
            return 0.0
        return self._backoff_base_s * (2**attempt) + random.uniform(0, 0.5)
