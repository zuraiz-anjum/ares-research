"""Web search tool backed by the Tavily API.

The research agent uses this to gather news, financials and recent
developments about a company.
"""

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tavily import TavilyClient

from app.config import settings

_client = TavilyClient(api_key=settings.tavily_api_key)


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def tavily_search(query: str, max_results: int | None = None) -> list[dict]:
    """Run a web search and return a list of result dicts.

    Each result contains at least `title`, `url` and `content` keys.
    """
    results = max_results or settings.max_search_results
    response = _client.search(
        query=query,
        max_results=results,
        search_depth="advanced",
    )
    return response.get("results", [])
