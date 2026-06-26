"""Document Agent.

Detects a URL in the user's message, fetches the page, strips HTML to plain
text, and loads the result as research findings so the Synthesis agent can
answer questions about the document.

Uses only stdlib (urllib + html.parser) — no extra dependencies.
"""

import logging
import re
import urllib.request
from html.parser import HTMLParser

from app.config import settings, truncate_to_budget
from app.state import AgentState

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s<>\"']+")


class _TextExtractor(HTMLParser):
    """Minimal HTML stripper — skips scripts/styles, collects visible text."""

    _SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript"}

    def __init__(self):
        super().__init__()
        self._skip = False
        self.chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS:
            self._skip = False

    def handle_data(self, data):
        if not self._skip and (text := data.strip()):
            self.chunks.append(text)

    def get_text(self) -> str:
        return "\n".join(self.chunks)


def _fetch_url(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.get_text()


def doc_agent_node(state: AgentState) -> dict:
    message = state["messages"][-1].content

    if settings.mock_mode:
        return {
            "findings": "Mock document: The article discusses AI research trends in 2025, focusing on multi-agent systems and real-time streaming pipelines.",
            "raw_research": "Mock raw document content.",
            "confidence_score": 8,
        }

    urls = _URL_RE.findall(message)
    if not urls:
        return {
            "findings": "No URL was found in your message. Please include a URL (starting with http:// or https://) to analyse.",
            "raw_research": "",
            "confidence_score": 1,
        }

    url = urls[0]
    logger.info(f"doc_agent fetching url={url}")

    try:
        raw = _fetch_url(url)
    except Exception as exc:
        logger.warning(f"doc_agent fetch_failed url={url} error={exc}")
        return {
            "findings": f"Could not retrieve the document at {url}. The site may be blocking automated access or the URL may be invalid.",
            "raw_research": "",
            "confidence_score": 1,
        }

    if not raw.strip():
        return {
            "findings": f"The page at {url} returned no readable text (it may be a JavaScript-heavy app or behind a login).",
            "raw_research": "",
            "confidence_score": 1,
        }

    raw, _ = truncate_to_budget(raw, label="doc_agent")
    logger.info(f"doc_agent fetched chars={len(raw)} url={url}")
    return {
        "findings": raw,
        "raw_research": raw,
        "confidence_score": 7,
        "source_url": url,
    }
