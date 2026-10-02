"""Compatibility shim: the web_search implementation lives in `app.tools.search`.

Phase 8b split the module (PRD §17.1) into `app/tools/search/{constants,
parsers, gate, cache, providers, tool}.py`; `httpx` is imported only by
`app/tools/search/providers.py` (layer rule amended). This path keeps every
existing import working — `DDG_URL`, `parse_results`, `WebSearchTool`, and
`KNOWN_SEARCH_PROVIDERS` — for the registry, `app.tools.__init__`, and tests.
"""

from __future__ import annotations

from app.tools.search import (
    BING_URL,
    DDG_URL,
    KNOWN_SEARCH_PROVIDERS,
    BingProvider,
    DuckDuckGoProvider,
    SearchCache,
    SearchGate,
    SearchOutcome,
    WebSearchTool,
    classify_response,
    parse_bing_results,
    parse_results,
)

__all__ = [
    "BING_URL",
    "DDG_URL",
    "KNOWN_SEARCH_PROVIDERS",
    "BingProvider",
    "DuckDuckGoProvider",
    "SearchCache",
    "SearchGate",
    "SearchOutcome",
    "WebSearchTool",
    "classify_response",
    "parse_bing_results",
    "parse_results",
]
