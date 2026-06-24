"""Research agent.

Gathers company information (news, financials, recent developments) using the
Tavily search tool, summarises the findings, and rates its own confidence in
how well the findings answer the user's question.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm import get_llm
from app.state import AgentState
from app.tools.search import tavily_search

RESEARCH_SYSTEM_PROMPT = """You are the Research Agent in a company-research assistant.
You are given raw web search results about a company. Extract the relevant facts
(news, financials, leadership, recent developments) that help answer the user's
question, and summarise them clearly.

Then rate your confidence from 0 to 10 that the findings are sufficient and
relevant enough to answer the user's question well."""


class ResearchResult(BaseModel):
    findings: str = Field(description="Concise summary of the relevant findings.")
    confidence_score: int = Field(ge=0, le=10)


def _build_query(state: AgentState) -> str:
    """Construct the search query from the user's question (and any clarification)."""
    question = state["original_query"]
    clarification = state.get("clarification")
    if clarification:
        return f"{question} {clarification}"
    return question


def research_node(state: AgentState) -> dict:
    query = _build_query(state)
    results = tavily_search(query)
    raw_research = "\n\n".join(r.get("content", "") for r in results)

    llm = get_llm().with_structured_output(ResearchResult)
    result: ResearchResult = llm.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=f"User question: {query}\n\nSearch results:\n{raw_research}"),
        ]
    )

    return {
        "findings": result.findings,
        "raw_research": raw_research,
        "confidence_score": result.confidence_score,
    }
