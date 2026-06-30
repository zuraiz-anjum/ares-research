"""Research agent.

Gathers company information (news, financials, recent developments) using the
Tavily search tool, summarises the findings, and rates its own confidence in
how well the findings answer the user's question.

Uses asyncio.gather for true parallel search — no thread pool overhead.
"""

import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings, truncate_to_budget
from app.llm import get_llm
from app.state import AgentState
from app.tools.search import MOCK_RESULTS, tavily_search
from app.utils.search_cache import cached_search

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """You are the Research Agent in an AI research workspace.
Summarise the web search results below to answer the user's question.

RULES:
- If the results cover MULTIPLE distinct entities (companies, products, people),
  write a separate section for each using ## headings (e.g. ## OpenAI, ## Anthropic).
  Never blend entity-specific figures across sections.
- Include specific numbers, dates, and dollar amounts exactly as found.
  Use vague language only when the source itself is vague.
- Do NOT add information from your training knowledge.
  If a figure is absent from the results, write "Not found in search results" for that item.
- Rate your confidence 0-10 based solely on how well the results cover the question."""


class ResearchResult(BaseModel):
    findings: str = Field(description="Concise summary of the relevant findings.")
    confidence_score: int = Field(ge=0, le=10)


def _get_queries(state: AgentState) -> list[str]:
    """Return sub-queries from decomposer, or fall back to single query."""
    sub_queries = state.get("sub_queries")
    if sub_queries:
        return sub_queries
    question = state["original_query"]
    clarification = state.get("clarification")
    return [f"{question} {clarification}".strip() if clarification else question]


async def _fetch_one(query: str) -> tuple[list, int, list[dict]]:
    """Run a single Tavily search asynchronously using a thread executor."""
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, cached_search, query)
    srcs = [{"title": r.get("title", ""), "url": r.get("url", "")}
            for r in results if r.get("url")]
    return results, len(results), srcs


async def research_node(state: AgentState) -> dict:
    queries = _get_queries(state)
    original_question = state.get("original_query", queries[0])

    if settings.mock_mode:
        return {
            "findings": "Mock findings: The company has strong revenue growth of 30% YoY. Recent $500M funding round at $10B valuation. Leadership team expanded with three C-suite hires.",
            "raw_research": MOCK_RESULTS[0]["content"],
            "confidence_score": 8,
        }

    # Run all sub-queries in true parallel via asyncio.gather — no thread-pool overhead.
    fetch_results = await asyncio.gather(*[_fetch_one(q) for q in queries])

    result_counts = [r[1] for r in fetch_results]

    # Deduplicate sources and assign sequential citation numbers [1], [2], …
    all_sources: list[dict] = []
    seen_urls: set[str] = set()
    url_to_num: dict[str, int] = {}
    for _, _, srcs in fetch_results:
        for s in srcs:
            if s["url"] not in seen_urls:
                num = len(all_sources) + 1
                all_sources.append(s)
                seen_urls.add(s["url"])
                url_to_num[s["url"]] = num

    # Build per-query blocks with inline citation numbers so the summariser
    # and downstream agents can reference sources as [n].
    raw_blocks = []
    for query, (results, _, srcs) in zip(queries, fetch_results):
        lines = [f"=== Results for: {query} ==="]
        for r in results:
            num = url_to_num.get(r.get("url", ""), "?")
            lines.append(f"[{num}] {r.get('title','')}\n{r.get('content','')}")
        raw_blocks.append("\n\n".join(lines))

    raw_research = "\n\n".join(raw_blocks)
    raw_research, _ = truncate_to_budget(raw_research, label="research_raw")

    for q, count in zip(queries, result_counts):
        logger.info(f"search query={repr(q)} results={count}")
    logger.info(f"sources_collected count={len(all_sources)}")

    llm = get_llm().with_structured_output(ResearchResult)
    result: ResearchResult = llm.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=f"User question: {original_question}\n\nSearch results:\n{raw_research}"),
        ]
    )

    # Penalise confidence if any sub-query returned sparse or no results.
    confidence = result.confidence_score
    min_results = min(result_counts)
    if min_results == 0:
        confidence = max(0, confidence - 3)
    elif len(queries) > 1 and min_results < 2:
        confidence = max(0, confidence - 1)

    return {
        "findings": result.findings,
        "raw_research": raw_research,
        "confidence_score": confidence,
        "sources": all_sources,
    }
