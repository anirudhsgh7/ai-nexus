"""web_search: parsers, challenge classifier, retries, cascade, gate, cache.

Phase 8b PRD §9.1. Fully offline: respx for transports, injected clocks/sleeps
where retry timing matters, pure functions everywhere else.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest
import respx

from app.schemas import AgentRole
from app.tools import ToolContext
from app.tools.search import providers as providers_mod
from app.tools.web_search import (
    BING_URL,
    DDG_URL,
    SearchCache,
    SearchGate,
    SearchOutcome,
    WebSearchTool,
    classify_response,
    parse_bing_results,
    parse_results,
)

CTX = ToolContext(agent=AgentRole.RESEARCHER, run_id="run1")
FIXTURE = Path(__file__).parent / "fixtures" / "ddg_results.html"
BING_FIXTURE = Path(__file__).parent / "fixtures" / "bing_results.html"
CHALLENGE_FIXTURE = Path(__file__).parent / "fixtures" / "ddg_challenge.html"


class _RecordingGate:
    """Zero-wait gate that counts acquisitions (pacing is tested separately)."""

    def __init__(self) -> None:
        self.calls = 0

    async def acquire(self) -> None:
        self.calls += 1


def _gate() -> SearchGate:
    return SearchGate(min_interval_s=0.0, jitter_s=0.0)


async def _no_sleep(_seconds: float) -> None:
    return None


def _tool(
    client: httpx.AsyncClient | None = None,
    *,
    providers: tuple[str, ...] = ("ddg",),
    retries: int = 0,
    backoff_base_s: float = 0.001,
    cache_ttl_s: float = 900.0,
    max_results: int = 5,
    gate: object | None = None,
    cache: SearchCache | None = None,
    sleep: object | None = None,
) -> WebSearchTool:
    return WebSearchTool(
        max_results=max_results,
        timeout_s=45.0,
        result_max_chars=2000,
        providers=providers,
        retries=retries,
        backoff_base_s=backoff_base_s,
        region="us-en",
        cache_ttl_s=cache_ttl_s,
        gate=gate if gate is not None else _gate(),  # type: ignore[arg-type]
        cache=cache,
        ddg_client=client,
        bing_client=client,
        sleep=sleep,  # type: ignore[arg-type]
    )


# ------------------------------------------------------------------- DDG parser


def test_parse_results_from_live_capture():
    results, truncated = parse_results(FIXTURE.read_text(), limit=5)
    assert truncated is False
    assert len(results) == 2
    assert results[0] == {
        "title": "GDP growth (annual %) - United States | Data",
        "url": "https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG?locations=US",
        "snippet": results[0]["snippet"],
    }
    assert results[0]["snippet"], "snippet text should be extracted and unescaped"
    assert results[0]["url"].startswith("https://")
    assert "duckduckgo.com" not in results[0]["url"], "redirect must be decoded"
    assert results[1]["title"].startswith("CBO")


def test_parse_results_limit_marks_truncated():
    results, truncated = parse_results(FIXTURE.read_text(), limit=1)
    assert len(results) == 1
    assert truncated is True


def test_parse_results_empty_page():
    results, truncated = parse_results("<html><body>No results</body></html>", 5)
    assert results == []
    assert truncated is False


def test_parse_results_strips_tags_and_entities():
    html = (
        '<a rel="nofollow" class="result__a" '
        'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=x">'
        "A &amp; B <b>growth</b> report</a>"
        '<a class="result__snippet" href="//x">Snippet &quot;23%&quot;</a>'
    )
    results, _ = parse_results(html, 5)
    assert results[0]["title"] == "A & B growth report"
    assert results[0]["url"] == "https://example.com/a"
    assert results[0]["snippet"] == 'Snippet "23%"'


def test_parse_results_ignores_other_classes():
    html = '<a class="result__url" href="//other">not a title</a>'
    results, _ = parse_results(html, 5)
    assert results == []


# ------------------------------------------------------------------ Bing parser


def test_bing_parser_fixture():
    results, truncated = parse_bing_results(BING_FIXTURE.read_text(), limit=5)
    assert truncated is False
    assert len(results) == 3
    assert results[0]["title"] == "Kubernetes vs Docker: What's the difference? | IBM"
    assert results[0]["url"] == (
        "https://www.ibm.com/think/topics/kubernetes-vs-docker"
    )
    assert "bing.com" not in results[0]["url"]
    assert results[0]["snippet"].startswith("Kubernetes orchestrates")
    assert results[1]["title"] == "What is a Container? Docker & Container Basics"
    assert results[1]["url"] == "https://www.docker.com/resources/what-container/"
    assert "&" in results[1]["snippet"]  # entities unescaped inside <p>
    assert results[2]["url"] == "https://kubernetes.io/docs/concepts/overview/"


def test_bing_parser_limit_and_truncated():
    results, truncated = parse_bing_results(BING_FIXTURE.read_text(), limit=2)
    assert len(results) == 2
    assert truncated is True


def test_bing_parser_empty_page():
    results, truncated = parse_bing_results("<html>no results</html>", 5)
    assert results == []
    assert truncated is False


# ---------------------------------------------------------- challenge classifier


def test_challenge_classifier():
    assert classify_response(202, "<html>blocked</html>", 0) == "challenge"
    assert classify_response(403, "forbidden", 0) == "challenge"
    assert classify_response(429, "too many", 0) == "challenge"
    # captured challenge page: anomaly-modal / bots copy / cc=botnet / form
    assert classify_response(200, CHALLENGE_FIXTURE.read_text(), 0) == "challenge"
    assert classify_response(200, 'class="anomaly-modal"', 0) == "challenge"
    assert classify_response(200, 'src="anomaly.js"', 0) == "challenge"
    assert classify_response(200, "bots use DuckDuckGo", 0) == "challenge"
    assert classify_response(200, "cc=botnet", 0) == "challenge"
    # a bare "anomaly" in a real snippet with results must NOT block
    assert (
        classify_response(200, "<p>anomaly detection library</p>", 3) == "ok"
    )
    # 0 results + tiny body = soft block; 0 results + large body = real empty SERP
    assert classify_response(200, "<html>nothing here</html>", 0) == "challenge"
    assert classify_response(200, "x" * 20000, 0) == "ok"


# ------------------------------------------------------- DDG request behavior


@respx.mock
async def test_successful_search():
    respx.post(DDG_URL).mock(
        return_value=httpx.Response(200, text=FIXTURE.read_text())
    )
    async with httpx.AsyncClient() as client:
        result = await _tool(client, max_results=3).call(
            {"query": "growth 2025"}, CTX
        )
    assert result.error is None
    payload = json.loads(result.content)
    assert payload["ok"] is True
    assert payload["query"] == "growth 2025"
    assert len(payload["results"]) == 2
    assert payload["truncated"] is False
    assert payload["provider"] == "ddg"
    assert payload["attempts"] == ["ddg"]
    assert payload["cached"] is False
    assert "Mozilla" in respx.calls[0].request.headers["User-Agent"]


@respx.mock
async def test_ddg_post_form_shape():
    respx.post(DDG_URL).mock(
        return_value=httpx.Response(200, text=FIXTURE.read_text())
    )
    async with httpx.AsyncClient() as client:
        await _tool(client).call({"query": "growth 2025"}, CTX)
    request = respx.calls[0].request
    assert request.method == "POST"
    assert str(request.url) == DDG_URL, "clean URL: no query string on POST"
    body = request.content.decode()
    assert "q=growth+2025" in body or "q=growth%202025" in body
    assert "b=" in body
    assert "kl=us-en" in body


@respx.mock
async def test_ddg_headers_full_browser():
    respx.post(DDG_URL).mock(
        return_value=httpx.Response(200, text=FIXTURE.read_text())
    )
    async with httpx.AsyncClient() as client:
        await _tool(client).call({"query": "x"}, CTX)
    headers = respx.calls[0].request.headers
    assert headers["Accept-Language"] == "en-US,en;q=0.9"
    assert headers["Referer"] == "https://html.duckduckgo.com/"
    assert headers["Origin"] == "https://html.duckduckgo.com"
    assert headers["Sec-Fetch-Site"] == "same-origin"
    assert headers["Sec-Fetch-Mode"] == "navigate"
    assert headers["Sec-Fetch-Dest"] == "document"
    assert headers["Sec-Fetch-User"] == "?1"
    assert headers["Upgrade-Insecure-Requests"] == "1"
    assert "Chrome/124" in headers["User-Agent"]
    assert 'v="124"' in headers["sec-ch-ua"]
    assert headers["sec-ch-ua-platform"] in {'"macOS"', '"Windows"'}
    assert headers["sec-ch-ua-mobile"] == "?0"


@respx.mock
async def test_ua_rotates_on_retry():
    respx.post(DDG_URL).mock(
        side_effect=[
            httpx.Response(202, text="challenge"),
            httpx.Response(200, text=FIXTURE.read_text()),
        ]
    )
    async with httpx.AsyncClient() as client:
        result = await _tool(client, retries=1, sleep=_no_sleep).call(
            {"query": "x"}, CTX
        )
    assert result.error is None
    assert len(respx.calls) == 2
    user_agents = [call.request.headers["User-Agent"] for call in respx.calls]
    assert user_agents[0] != user_agents[1]


@respx.mock
async def test_retry_backoff_on_challenge(monkeypatch):
    respx.post(DDG_URL).mock(
        side_effect=[
            httpx.Response(202, text="challenge"),
            httpx.Response(202, text="challenge"),
            httpx.Response(200, text=FIXTURE.read_text()),
        ]
    )
    sleeps: list[float] = []

    async def _record(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(providers_mod.random, "uniform", lambda a, b: 0.0)
    async with httpx.AsyncClient() as client:
        result = await _tool(
            client, retries=2, backoff_base_s=0.01, sleep=_record
        ).call({"query": "x"}, CTX)
    assert result.error is None
    assert len(respx.calls) == 3
    assert sleeps == pytest.approx([0.01, 0.02], abs=1e-9)


@respx.mock
async def test_retry_exhausted_is_provider_blocked():
    respx.post(DDG_URL).mock(
        return_value=httpx.Response(202, text="challenge")
    )
    async with httpx.AsyncClient() as client:
        result = await _tool(client, retries=2, sleep=_no_sleep).call(
            {"query": "x"}, CTX
        )
    assert result.error == "provider_blocked"
    assert len(respx.calls) == 3
    assert "ddg" in result.content
    assert "HTTP 202" in result.content
    assert "3 attempts" in result.content


@respx.mock
async def test_no_retry_on_network_error():
    respx.post(DDG_URL).mock(side_effect=httpx.ConnectError("refused"))
    async with httpx.AsyncClient() as client:
        result = await _tool(client, retries=2).call({"query": "x"}, CTX)
    assert result.error == "network_error"
    assert len(respx.calls) == 1, "transport errors are never retried"


@respx.mock
async def test_zero_result_small_body_retried():
    respx.post(DDG_URL).mock(
        side_effect=[
            httpx.Response(200, text="<html>nothing yet</html>"),
            httpx.Response(200, text=FIXTURE.read_text()),
        ]
    )
    async with httpx.AsyncClient() as client:
        result = await _tool(client, retries=1, sleep=_no_sleep).call(
            {"query": "x"}, CTX
        )
    assert result.error is None
    payload = json.loads(result.content)
    assert len(payload["results"]) == 2
    assert len(respx.calls) == 2


@respx.mock
async def test_anomaly_page_is_provider_blocked():
    respx.post(DDG_URL).mock(
        return_value=httpx.Response(200, text=CHALLENGE_FIXTURE.read_text())
    )
    async with httpx.AsyncClient() as client:
        result = await _tool(client, retries=0).call({"query": "x"}, CTX)
    assert result.error == "provider_blocked"


@respx.mock
async def test_non_200_is_provider_blocked():
    respx.post(DDG_URL).mock(return_value=httpx.Response(202, text="challenge"))
    async with httpx.AsyncClient() as client:
        result = await _tool(client, retries=0).call({"query": "x"}, CTX)
    assert result.error == "provider_blocked"


@respx.mock
async def test_transport_error_is_network_error():
    respx.post(DDG_URL).mock(side_effect=httpx.ConnectError("refused"))
    async with httpx.AsyncClient() as client:
        result = await _tool(client).call({"query": "x"}, CTX)
    assert result.error == "network_error"


async def test_blank_query_rejected_before_any_request():
    result = await _tool().call({"query": ""}, CTX)
    assert result.error == "invalid_arguments"


@respx.mock
async def test_result_limit_respected():
    respx.post(DDG_URL).mock(
        return_value=httpx.Response(200, text=FIXTURE.read_text())
    )
    async with httpx.AsyncClient() as client:
        payload = json.loads(
            (await _tool(client, max_results=1).call({"query": "growth"}, CTX)).content
        )
    assert len(payload["results"]) == 1
    assert payload["truncated"] is True


# --------------------------------------------------------------------- cascade


@respx.mock
async def test_cascade_ddg_blocked_bing_ok():
    respx.post(DDG_URL).mock(
        return_value=httpx.Response(202, text="challenge")
    )
    respx.get(BING_URL).mock(
        return_value=httpx.Response(200, text=BING_FIXTURE.read_text())
    )
    async with httpx.AsyncClient() as client:
        result = await _tool(
            client, providers=("ddg", "bing"), retries=0
        ).call({"query": "kubernetes vs docker"}, CTX)
    assert result.error is None
    payload = json.loads(result.content)
    assert payload["provider"] == "bing"
    assert payload["attempts"] == ["ddg", "bing"]
    assert payload["cached"] is False
    assert len(payload["results"]) == 3


@respx.mock
async def test_cascade_all_blocked():
    respx.post(DDG_URL).mock(
        return_value=httpx.Response(202, text="challenge")
    )
    respx.get(BING_URL).mock(
        return_value=httpx.Response(429, text="rate limited")
    )
    async with httpx.AsyncClient() as client:
        result = await _tool(
            client, providers=("ddg", "bing"), retries=0
        ).call({"query": "x"}, CTX)
    assert result.error == "provider_blocked"
    assert "ddg" in result.content and "HTTP 202" in result.content
    assert "bing" in result.content and "429" in result.content


@respx.mock
async def test_cascade_all_network_errors_is_network_error():
    respx.post(DDG_URL).mock(side_effect=httpx.ConnectError("x"))
    respx.get(BING_URL).mock(side_effect=httpx.ConnectError("x"))
    async with httpx.AsyncClient() as client:
        result = await _tool(
            client, providers=("ddg", "bing"), retries=0
        ).call({"query": "x"}, CTX)
    assert result.error == "network_error"
    assert "ddg" in result.content and "bing" in result.content


@respx.mock
async def test_empty_serp_is_success_no_cascade():
    respx.post(DDG_URL).mock(
        return_value=httpx.Response(200, text="<html>" + "x" * 5000 + "</html>")
    )
    # No Bing route is registered: any cascade attempt would raise from respx.
    async with httpx.AsyncClient() as client:
        result = await _tool(
            client, providers=("ddg", "bing"), retries=0
        ).call({"query": "x"}, CTX)
    assert result.error is None
    payload = json.loads(result.content)
    assert payload["provider"] == "ddg"
    assert payload["results"] == []
    assert payload["truncated"] is False


@respx.mock
async def test_bing_consent_redirect_retries_with_cookie():
    respx.get(BING_URL).mock(
        side_effect=[
            httpx.Response(
                302, headers={"Location": "https://consent.bing.com/consent"}
            ),
            httpx.Response(200, text=BING_FIXTURE.read_text()),
        ]
    )
    respx.get("https://consent.bing.com/consent").mock(
        return_value=httpx.Response(
            200, text='<html><div id="b_consent">Choose your region</div></html>'
        )
    )
    async with httpx.AsyncClient() as client:
        result = await _tool(
            client, providers=("bing",), retries=1, sleep=_no_sleep
        ).call({"query": "x"}, CTX)
    assert result.error is None
    payload = json.loads(result.content)
    assert payload["provider"] == "bing"
    assert client.cookies.get("SRCHHPGUSR") == "SRCHLANG=en"


# ----------------------------------------------------------------------- cache


@respx.mock
async def test_cache_hit_skips_providers():
    respx.post(DDG_URL).mock(
        return_value=httpx.Response(200, text=FIXTURE.read_text())
    )
    async with httpx.AsyncClient() as client:
        tool = _tool(client, cache_ttl_s=900.0)
        first = json.loads(
            (await tool.call({"query": "growth 2025"}, CTX)).content
        )
        second = json.loads(
            (await tool.call({"query": "Growth   2025"}, CTX)).content
        )
    assert len(respx.calls) == 1, "second call must be served from cache"
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["provider"] == "ddg"
    assert second["attempts"] == ["ddg"], "cache must preserve provenance"


@respx.mock
async def test_cache_ttl_expiry():
    respx.post(DDG_URL).mock(
        return_value=httpx.Response(200, text=FIXTURE.read_text())
    )
    async with httpx.AsyncClient() as client:
        tool = _tool(client, cache_ttl_s=0.05)
        await tool.call({"query": "x"}, CTX)
        await asyncio.sleep(0.1)
        await tool.call({"query": "x"}, CTX)
    assert len(respx.calls) == 2


@respx.mock
async def test_cache_never_stores_blocks():
    respx.post(DDG_URL).mock(
        return_value=httpx.Response(202, text="challenge")
    )
    async with httpx.AsyncClient() as client:
        tool = _tool(client, retries=0)
        first = await tool.call({"query": "x"}, CTX)
        second = await tool.call({"query": "x"}, CTX)
    assert first.error == second.error == "provider_blocked"
    assert len(respx.calls) == 2, "a blocked outcome must never be cached"


def test_cache_put_ignores_failed_outcomes():
    cache = SearchCache(ttl_s=900.0)
    blocked = SearchOutcome(
        provider="ddg", results=(), truncated=False, blocked=True, reason="x"
    )
    cache.put(blocked, "blocked-query")
    assert cache.get("ddg", "blocked-query") is None
    network = SearchOutcome(
        provider="ddg",
        results=(),
        truncated=False,
        blocked=False,
        reason="network_error: ConnectError",
    )
    cache.put(network, "network-query")
    assert cache.get("ddg", "network-query") is None


# ------------------------------------------------------------------ gate/pacing


async def test_pacing_min_interval():
    gate = SearchGate(min_interval_s=0.05, jitter_s=0.0)
    started = time.monotonic()
    await gate.acquire()
    first = time.monotonic() - started
    second_started = time.monotonic()
    await gate.acquire()
    second = time.monotonic() - second_started
    assert first < 0.05, "first call must pass without artificial delay"
    assert second >= 0.04, "second call must wait out the pacing interval"


@respx.mock
async def test_tool_acquires_gate_once_per_call():
    respx.post(DDG_URL).mock(
        return_value=httpx.Response(200, text=FIXTURE.read_text())
    )
    gate = _RecordingGate()
    async with httpx.AsyncClient() as client:
        tool = _tool(client, gate=gate, cache_ttl_s=0.0)
        await tool.call({"query": "a"}, CTX)
        await tool.call({"query": "b"}, CTX)
    assert gate.calls == 2


# ---------------------------------------------------------- construction guards


def test_unknown_provider_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown web search provider"):
        WebSearchTool(providers=("ddg", "mojeek"))


def test_empty_provider_list_rejected():
    with pytest.raises(ValueError, match="at least one"):
        WebSearchTool(providers=())
