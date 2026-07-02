"""Web search tool backed by the Tavily API.

The research agent uses this to gather news, financials and recent
developments about a company.
"""

from tenacity import retry, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential
from tavily import TavilyClient

from app.config import settings

_client = TavilyClient(api_key=settings.tavily_api_key)


MOCK_RESULTS = [
    {
        "title": "Mock Company Overview",
        "url": "https://example.com/mock",
        "content": (
            "The company has shown strong revenue growth of 30% year-over-year. "
            "Recent funding round raised $500M at a $10B valuation. "
            "Leadership team expanded with three C-suite hires. "
            "New product lines launched in Q3 with positive market reception."
        ),
    }
]


def _is_retriable(exc: Exception) -> bool:
    """Only retry transient network/rate-limit errors, not auth or bad-request failures."""
    msg = str(exc).lower()
    # Never retry auth failures, invalid keys, or bad requests — they won't fix themselves.
    non_retriable = ("401", "403", "invalid api key", "unauthorized", "400", "bad request")
    if any(kw in msg for kw in non_retriable):
        return False
    return True


@retry(
    retry=retry_if_exception(_is_retriable),
    stop=(stop_after_attempt(3) | stop_after_delay(30)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def tavily_search(query: str, max_results: int | None = None) -> list[dict]:
    """Run a web search and return a list of result dicts.

    Each result contains at least `title`, `url` and `content` keys.
    """
    if settings.mock_mode:
        return MOCK_RESULTS

    results = max_results or settings.max_search_results
    response = _client.search(
        query=query,
        max_results=results,
        search_depth="advanced",
    )
    return response.get("results", [])
