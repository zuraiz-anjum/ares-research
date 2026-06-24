"""Research agent.

Gathers company information (news, financials, recent developments) using the
Tavily search tool, summarises the findings, and rates its own confidence in
how well the findings answer the user's question.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm import get_llm
from app.state import AgentState
from app.tools.search import tavily_search

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """You are the Research Agent in a company-research assistant.
You are given raw web search results about a company. Extract the relevant facts
(news, financials, leadership, recent developments) that help answer the user's
question, and summarise them clearly.

Then rate your confidence from 0 to 10 that the findings are sufficient and
relevant enough to answer the user's question well."""


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


def research_node(state: AgentState) -> dict:
    queries = _get_queries(state)
    original_question = state.get("original_query", queries[0])

    # Run all sub-queries in parallel — no sequential dependency between them.
    def fetch(query: str) -> tuple[str, int]:
        results = tavily_search(query)
        content = f"=== Results for: {query} ===\n" + "\n\n".join(r.get("content", "") for r in results)
        return content, len(results)

    with ThreadPoolExecutor() as executor:
        fetch_results = list(executor.map(fetch, queries))

    all_raw = [r[0] for r in fetch_results]
    result_counts = [r[1] for r in fetch_results]
    raw_research = "\n\n".join(all_raw)

    for q, count in zip(queries, result_counts):
        logger.info(f"search query={repr(q)} results={count}")

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
    }
