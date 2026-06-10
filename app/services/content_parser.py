"""Fetch + parse content into the normalized form the AEO checks consume.

Responsibilities:
- Fetch a URL with httpx (10s timeout) and surface failures as
  ContentFetchError so the API layer can map them to a 422.
- Detect whether pasted input is HTML or plain text.
- Strip boilerplate (nav/footer/header/aside/script/style/...) before any
  text-based scoring.
- Extract: first paragraph, H-tags in DOM order, and the main body text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup

FETCH_TIMEOUT_SECONDS = 10.0
MAX_CONTENT_BYTES = 2_000_000  # refuse to parse pages larger than ~2 MB

# Tags that are navigation/chrome rather than content. Removed before the
# first-paragraph extraction and readability scoring.
BOILERPLATE_TAGS = (
    "nav", "footer", "header", "aside", "script", "style",
    "noscript", "form", "iframe", "svg", "button",
)

HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")

# A sane browser-ish UA: some sites return 403 to the default httpx UA.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
        "AEGIS-Bot/0.1"
    )
}


class ContentFetchError(Exception):
    """URL could not be fetched (timeout, non-2xx, not HTML, too large)."""

    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


class ContentParseError(Exception):
    """Content was fetched/supplied but no usable text could be extracted."""

    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


@dataclass
class ParsedContent:
    """Normalized content handed to every AEO check."""

    first_paragraph: str
    headings: list[tuple[str, str]] = field(default_factory=list)  # (tag, text), DOM order
    main_text: str = ""
    is_html: bool = False


def fetch_url(url: str) -> str:
    """Fetch a URL and return its body text, raising ContentFetchError on any failure."""
    if not re.match(r"^https?://", url, re.IGNORECASE):
        raise ContentFetchError(
            "Invalid URL: only http(s) URLs are supported.", detail=f"Got: {url[:200]}"
        )
    try:
        with httpx.Client(
            timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True, headers=_HEADERS
        ) as client:
            response = client.get(url)
    except httpx.TimeoutException as exc:
        raise ContentFetchError(
            "Could not retrieve content from the provided URL.",
            detail=f"Connection timeout after {int(FETCH_TIMEOUT_SECONDS)}s",
        ) from exc
    except httpx.HTTPError as exc:
        raise ContentFetchError(
            "Could not retrieve content from the provided URL.", detail=str(exc)
        ) from exc

    if response.status_code >= 400:
        raise ContentFetchError(
            "Could not retrieve content from the provided URL.",
            detail=f"HTTP {response.status_code} from {url}",
        )

    content_type = response.headers.get("content-type", "")
    if content_type and not any(
        t in content_type for t in ("text/html", "text/plain", "application/xhtml")
    ):
        raise ContentFetchError(
            "URL did not return HTML or plain text content.",
            detail=f"Content-Type: {content_type}",
        )

    if len(response.content) > MAX_CONTENT_BYTES:
        raise ContentFetchError(
            "Page is too large to analyze.",
            detail=f"{len(response.content)} bytes (limit {MAX_CONTENT_BYTES})",
        )

    return response.text


def looks_like_html(raw: str) -> bool:
    """Cheap heuristic: does the input contain real markup?"""
    return bool(re.search(r"<\s*(html|body|p|h[1-6]|div|article|section)\b", raw, re.IGNORECASE))


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _parse_html(raw: str) -> ParsedContent:
    soup = BeautifulSoup(raw, "html.parser")

    # Headings are collected BEFORE boilerplate stripping: an <h1> living in a
    # <header> wrapper is still the page's H1 and must count for hierarchy.
    headings = [
        (tag.name, _normalize_whitespace(tag.get_text()))
        for tag in soup.find_all(HEADING_TAGS)
    ]

    for tag in soup.find_all(BOILERPLATE_TAGS):
        tag.decompose()

    # First paragraph: first <p> with non-trivial text after boilerplate removal.
    first_paragraph = ""
    for p in soup.find_all("p"):
        text = _normalize_whitespace(p.get_text())
        if len(text.split()) >= 3:  # skip empty/decorative paragraphs
            first_paragraph = text
            break

    # Main text: prefer semantic content containers when present.
    container = soup.find("article") or soup.find("main") or soup.body or soup
    blocks: list[str] = []
    for el in container.find_all(["p", "li"] + list(HEADING_TAGS)):
        text = _normalize_whitespace(el.get_text())
        if text:
            blocks.append(text)
    main_text = "\n".join(blocks) if blocks else _normalize_whitespace(container.get_text(" "))

    if not first_paragraph and main_text:
        # JS-heavy or non-<p> page: fall back to the first text block.
        first_paragraph = main_text.split("\n")[0]

    return ParsedContent(
        first_paragraph=first_paragraph,
        headings=headings,
        main_text=main_text,
        is_html=True,
    )


def _parse_plain_text(raw: str) -> ParsedContent:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    first_paragraph = paragraphs[0] if paragraphs else ""
    main_text = "\n".join(paragraphs)
    return ParsedContent(
        first_paragraph=_normalize_whitespace(first_paragraph),
        headings=[],
        main_text=main_text,
        is_html=False,
    )


def parse_content(raw: str) -> ParsedContent:
    """Parse raw HTML or plain text into ParsedContent.

    Raises ContentParseError if no usable text survives parsing (e.g. a
    JavaScript-only page whose HTML body is empty).
    """
    raw = raw.strip()
    if not raw:
        raise ContentParseError("No content to analyze.", detail="Input was empty.")

    parsed = _parse_html(raw) if looks_like_html(raw) else _parse_plain_text(raw)

    if not parsed.main_text or len(parsed.main_text.split()) < 5:
        raise ContentParseError(
            "Could not extract readable text from the content.",
            detail=(
                "The page may be JavaScript-rendered, behind a login wall, "
                "or otherwise empty after boilerplate removal."
            ),
        )
    return parsed
