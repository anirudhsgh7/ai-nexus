"""Static request material for the search providers (no I/O, no httpx)."""

from __future__ import annotations

import zlib

DDG_URL = "https://html.duckduckgo.com/html/"
BING_URL = "https://www.bing.com/search"
KNOWN_SEARCH_PROVIDERS = frozenset({"ddg", "bing"})

#: Chrome desktop UAs paired with their matching client hints. Mismatched
#: hints (UA version vs sec-ch-ua) are themselves a bot signal.
_UA_POOL: tuple[tuple[str, str, str], ...] = (
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        '"Google Chrome";v="124", "Chromium";v="124", "Not-A.Brand";v="99"',
        "macOS",
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        '"Google Chrome";v="124", "Chromium";v="124", "Not-A.Brand";v="99"',
        "Windows",
    ),
)

BROWSER_HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "no-cache",
}
DDG_EXTRA_HEADERS = {
    "Referer": "https://html.duckduckgo.com/",
    "Origin": "https://html.duckduckgo.com",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-User": "?1",
}
BING_EXTRA_HEADERS = {
    "Referer": "https://www.bing.com/",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-User": "?1",
}


def ua_for(query: str, attempt: int) -> tuple[str, str, str]:
    """Deterministic-within-query UA rotation: same query rotates per attempt."""
    index = (attempt + zlib.crc32(query.encode("utf-8"))) % len(_UA_POOL)
    return _UA_POOL[index]


def headers_for(ua: tuple[str, str, str], extra: dict[str, str]) -> dict[str, str]:
    browser_ua, ch_ua, platform = ua
    headers = dict(BROWSER_HEADERS)
    headers.update(extra)
    headers["User-Agent"] = browser_ua
    headers["sec-ch-ua"] = ch_ua
    headers["sec-ch-ua-platform"] = f'"{platform}"'
    headers["sec-ch-ua-mobile"] = "?0"
    return headers
