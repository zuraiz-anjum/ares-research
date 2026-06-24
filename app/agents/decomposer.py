"""Decomposer agent.

Analyses the user's question and decides whether it is a simple atomic query
or a compound question that benefits from parallel sub-queries.

Simple:  "Tell me about Stripe"            -> ["Stripe company overview"]
Compound: "Compare Stripe and Brex funding" -> ["Stripe recent funding", "Brex recent funding"]

Sub-queries are independent and executed in parallel by the research agent,
avoiding the sequential dependency assumption of Least-to-Most prompting.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm import get_llm
from app.state import AgentState

DECOMPOSER_SYSTEM_PROMPT = """You are the Query Decomposer in a company research assistant.

Your job is to analyse the user's question and break it into focused search queries.

Rules:
- If the question asks about ONE company or topic, return a single query.
- If the question compares multiple companies, asks about multiple entities, or has
  clearly independent sub-questions, return 2-3 focused queries — one per entity or sub-topic.
- Each query should be self-contained and specific enough to search directly.
- Never return more than 3 queries.
- Queries must be independent — no query should depend on the answer to another."""


class DecompositionResult(BaseModel):
    queries: list[str] = Field(
        description="List of 1-3 independent search queries.",
        min_length=1,
        max_length=3,
    )
    is_compound: bool = Field(
        description="True if the question was decomposed into multiple queries."
    )


def decomposer_node(state: AgentState) -> dict:
    question = state.get("original_query", "")
    clarification = state.get("clarification", "")
    full_question = f"{question} {clarification}".strip() if clarification else question

    llm = get_llm(temperature=0).with_structured_output(DecompositionResult)
    result: DecompositionResult = llm.invoke([
        SystemMessage(content=DECOMPOSER_SYSTEM_PROMPT),
        HumanMessage(content=full_question),
    ])

    return {"sub_queries": result.queries}
