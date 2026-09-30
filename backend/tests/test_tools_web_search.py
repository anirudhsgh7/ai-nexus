"""web_search: pure parser against a live capture + request behavior (PRD §6.6.4)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

from app.schemas import AgentRole
from app.tools import ToolContext
from app.tools.web_search import DDG_URL, WebSearchTool, parse_results

CTX = ToolContext(agent=AgentRole.RESEARCHER, run_id="run1")
FIXTURE = Path(__file__).parent / "fixtures" / "ddg_results.html"


# ------------------------------------------------------------------- parser


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


# ------------------------------------------------------- request behavior


@respx.mock
async def test_successful_search():
    respx.get(DDG_URL).mock(
        return_value=httpx.Response(200, text=FIXTURE.read_text())
    )
    result = await WebSearchTool(max_results=3).call({"query": "growth 2025"}, CTX)
    assert result.error is None
    payload = json.loads(result.content)
    assert payload["ok"] is True
    assert payload["query"] == "growth 2025"
    assert len(payload["results"]) == 2
    assert payload["truncated"] is False
    request = respx.calls[0].request
    assert "q=growth+2025" in str(request.url) or "q=growth%202025" in str(request.url)
    assert "Mozilla" in request.headers["User-Agent"]


@respx.mock
async def test_anomaly_page_is_provider_blocked():
    respx.get(DDG_URL).mock(
        return_value=httpx.Response(200, text="<html>anomaly challenge</html>")
    )
    result = await WebSearchTool().call({"query": "x"}, CTX)
    assert result.error == "provider_blocked"


@respx.mock
async def test_non_200_is_provider_blocked():
    respx.get(DDG_URL).mock(return_value=httpx.Response(202, text="challenge"))
    result = await WebSearchTool().call({"query": "x"}, CTX)
    assert result.error == "provider_blocked"


@respx.mock
async def test_transport_error_is_network_error():
    respx.get(DDG_URL).mock(side_effect=httpx.ConnectError("refused"))
    result = await WebSearchTool().call({"query": "x"}, CTX)
    assert result.error == "network_error"


async def test_blank_query_rejected_before_any_request():
    result = await WebSearchTool().call({"query": ""}, CTX)
    assert result.error == "invalid_arguments"


@respx.mock
async def test_result_limit_respected():
    respx.get(DDG_URL).mock(
        return_value=httpx.Response(200, text=FIXTURE.read_text())
    )
    payload = json.loads(
        (await WebSearchTool(max_results=1).call({"query": "growth"}, CTX)).content
    )
    assert len(payload["results"]) == 1
    assert payload["truncated"] is True
