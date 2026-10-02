"""Pure SERP parsing and challenge classification (stdlib only; no I/O)."""

from __future__ import annotations

import base64
import html as html_lib
import re
from urllib.parse import parse_qs, urlsplit

_A_TAG_RE = re.compile(r"<a\s([^>]*)>(.*?)</a>", re.DOTALL | re.IGNORECASE)
_HREF_RE = re.compile(r'href="([^"]*)"', re.IGNORECASE)
_CLASS_RE = re.compile(r'class="([^"]*)"', re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_BING_BLOCK_RE = re.compile(
    r'<li[^>]*class="[^"]*\bb_algo\b[^"]*"[^>]*>', re.IGNORECASE
)
_BING_H2_RE = re.compile(r"<h2[^>]*>(.*?)</h2>", re.DOTALL | re.IGNORECASE)
_BING_P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.DOTALL | re.IGNORECASE)


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


def _decode_bing_href(href: str) -> str:
    """Bing wraps result URLs in `/ck/a?...&u=a1<base64>` redirects."""
    href = html_lib.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = "https://www.bing.com" + href
    parts = urlsplit(href)
    if "/ck/a" in parts.path:
        encoded = parse_qs(parts.query).get("u", [""])[0]
        if encoded.startswith("a1"):
            encoded = encoded[2:]
        if encoded:
            padded = encoded + "=" * (-len(encoded) % 4)
            try:
                decoded = base64.urlsafe_b64decode(padded).decode(
                    "utf-8", "replace"
                )
            except ValueError:
                decoded = ""
            if decoded.startswith(("http://", "https://")):
                return decoded
    return href


def parse_bing_results(
    html_text: str, limit: int
) -> tuple[list[dict[str, str]], bool]:
    """Extract ordered `{title, url, snippet}` from Bing SERP `li.b_algo` blocks."""
    starts = [match.start() for match in _BING_BLOCK_RE.finditer(html_text)]
    found: list[dict[str, str]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(html_text)
        block = html_text[start:end]
        title = ""
        url = ""
        heading = _BING_H2_RE.search(block)
        if heading:
            anchor = _A_TAG_RE.search(heading.group(1))
            if anchor:
                attrs, inner = anchor.group(1), anchor.group(2)
                href = _HREF_RE.search(attrs)
                url = _decode_bing_href(href.group(1)) if href else ""
                title = _text(inner)
        caption_index = block.lower().find("b_caption")
        snippet_src = block[caption_index:] if caption_index != -1 else block
        paragraph = _BING_P_RE.search(snippet_src)
        snippet = _text(paragraph.group(1)) if paragraph else ""
        if title or url:
            found.append({"title": title, "url": url, "snippet": snippet})
    return found[:limit], len(found) > limit


def classify_response(status: int, body: str, result_count: int) -> str:
    """Classify a provider response: `"ok"` (served) or `"challenge"` (blocked).

    `result_count` is required because a legitimate empty SERP must stay
    distinguishable: the soft-block page is tiny, a real (even empty) SERP is
    not. Markers are precise on purpose — a bare `"anomaly"` substring
    false-positives on real snippets (e.g. a query about `anomaly.js`).
    """
    if status == 202:
        return "challenge"
    if status in (403, 429):
        return "challenge"
    low = body.lower()
    if any(
        marker in low
        for marker in (
            "anomaly-modal",          # DDG captcha modal class
            "bots use duckduckgo",    # captcha page copy
            'src="anomaly.js"',       # precise script ref, NOT bare "anomaly"
            "cc=botnet",              # challenge redirect marker
            "challenge-form",         # generic challenge form
        )
    ):
        return "challenge"
    if result_count == 0 and len(body) < 2000:
        return "challenge"            # soft-block page is tiny; real SERP is not
    return "ok"
