"""`web_search`: free DuckDuckGo HTML backend behind a flag (PRD §6.6.4).

The only module outside `app/llm/` permitted to import httpx (amended layer
rule). Parsing is a pure function so it is unit-testable against a fixture
without any network.
"""

from __future__ import annotations

import html as html_lib
import logging
import re
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import Tool, ToolContext
from app.tools.errors import ToolError

__all__ = ["DDG_URL", "WebSearchTool", "parse_results"]

logger = logging.getLogger("ai_nexus.tools.web_search")

DDG_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_A_TAG_RE = re.compile(r"<a\s([^>]*)>(.*?)</a>", re.DOTALL | re.IGNORECASE)
_HREF_RE = re.compile(r'href="([^"]*)"', re.IGNORECASE)
_CLASS_RE = re.compile(r'class="([^"]*)"', re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _anchors(fragment: str, wanted_class: str) -> list[tuple[str, str]]:
    """All `<a>` tags carrying `wanted_class` as one of their classes."""
    found: list[tuple[str, str]] = []
    for match in _A_TAG_RE.finditer(fragment):
        attrs, inner = match.group(1), match.group(2)
        class_match = _CLASS_RE.search(attrs)
        if class_match is None:
            continue
        if wanted_class not in class_match.group(1).split():
            continue
        href_match = _HREF_RE.search(attrs)
        found.append((href_match.group(1) if href_match else "", inner))
    return found


def _text(fragment: str) -> str:
    """Tags stripped, entities decoded, whitespace collapsed."""
    return " ".join(html_lib.unescape(_TAG_RE.sub(" ", fragment)).split())


def _decode_href(href: str) -> str:
    """DuckDuckGo redirect links carry the real URL percent-encoded in `uddg`."""
    href = html_lib.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    query = urlsplit(href).query
    if query:
        uddg = parse_qs(query).get("uddg")
        if uddg:
            return uddg[0]
    return href


def parse_results(html_text: str, limit: int) -> tuple[list[dict[str, str]], bool]:
    """Extract ordered `{title, url, snippet}` results; second value = truncated."""
    titles = _anchors(html_text, "result__a")
    snippets = [_text(inner) for _, inner in _anchors(html_text, "result__snippet")]
    results = [
        {
            "title": _text(inner),
            "url": _decode_href(href),
            "snippet": snippets[index] if index < len(snippets) else "",
        }
        for index, (href, inner) in enumerate(titles[:limit])
    ]
    return results, len(titles) > limit


class _WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1, max_length=300)


class WebSearchTool(Tool):
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = (
        "Search the web with DuckDuckGo and return the top results as "
        "title, url and snippet. Snippets are summaries, not full pages."
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
        self, *, max_results: int = 5, timeout_s: float = 20.0,
        result_max_chars: int = 2000,
    ) -> None:
        super().__init__(timeout_s=timeout_s, result_max_chars=result_max_chars)
        self._max_results = max_results

    async def execute(self, args: BaseModel, context: ToolContext) -> dict[str, Any]:
        assert isinstance(args, _WebSearchArgs)  # enforced by Tool.call
        headers = {"User-Agent": USER_AGENT, "Accept": "text/html"}
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_s, headers=headers, follow_redirects=True
            ) as client:
                response = await client.get(DDG_URL, params={"q": args.query})
        except httpx.HTTPError as exc:
            raise ToolError(
                "network_error", f"search request failed: {type(exc).__name__}"
            ) from exc

        if response.status_code != 200:
            raise ToolError(
                "provider_blocked",
                f"DuckDuckGo returned HTTP {response.status_code}",
            )
        body = response.text
        if "anomaly" in body.lower():
            raise ToolError(
                "provider_blocked",
                "DuckDuckGo served a bot-challenge page; try again later",
            )
        results, truncated = parse_results(body, self._max_results)
        return {"query": args.query, "results": results, "truncated": truncated}
