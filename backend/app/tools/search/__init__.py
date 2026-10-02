"""Search package: parsers, gate, cache, providers, tool (Phase 8b PRD §17.1).

`app/tools/web_search.py` remains the public import path (compatibility shim).
"""

from app.tools.search.cache import SearchCache
from app.tools.search.constants import (
    BING_EXTRA_HEADERS,
    BING_URL,
    BROWSER_HEADERS,
    DDG_EXTRA_HEADERS,
    DDG_URL,
    KNOWN_SEARCH_PROVIDERS,
    headers_for,
    ua_for,
)
from app.tools.search.gate import SearchGate
from app.tools.search.parsers import (
    classify_response,
    parse_bing_results,
    parse_results,
)
from app.tools.search.providers import (
    BingProvider,
    DuckDuckGoProvider,
    SearchOutcome,
)
from app.tools.search.tool import WebSearchTool

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
