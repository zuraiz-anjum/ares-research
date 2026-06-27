"""Decomposer agent.

Analyses the user's question and decides whether it is a simple atomic query
or a compound question that benefits from parallel sub-queries.

Simple:  "Tell me about Stripe"            -> ["Stripe company overview"]
Compound: "Compare Stripe and Brex funding" -> ["Stripe recent funding", "Brex recent funding"]
Dependent: "What did Stripe do after their funding?" -> single merged query (no parallel split)

Sub-queries are independent and executed in parallel by the research agent,
avoiding the sequential dependency assumption of Least-to-Most prompting.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

# Keywords that suggest a compound query worth decomposing.
# If none of these appear, skip the LLM call entirely — fast path.
COMPOUND_INDICATORS = [
    "compare", "vs", "versus", "versus", "and", "both",
    "between", "difference", "similar", "each", "all three",
]

DECOMPOSER_SYSTEM_PROMPT = """You are the Query Decomposer in a company research assistant.

Your job is to analyse the user's question and break it into focused search queries.

Rules:
- If the question asks about ONE company or topic, return a single query.
- If the question compares multiple companies or asks about multiple independent entities,
  return 2-3 focused queries — one per entity or sub-topic.
- Each query must be fully self-contained and searchable on its own.
- Never return more than 3 queries.
- IMPORTANT: If answering one sub-question requires knowing the answer to another first
  (sequential dependency), set has_dependencies=true and return a single merged query instead.
  Only decompose when sub-queries are truly independent."""


class DecompositionResult(BaseModel):
    # min_length/max_length omitted — Cerebras rejects those JSON schema keywords.
    queries: list[str] = Field(
        description="List of 1-3 independent search queries.",
    )
    is_compound: bool = Field(
        description="True if the question was decomposed into multiple queries."
    )
    has_dependencies: bool = Field(
        default=False,
        description="True if sub-questions have sequential dependencies. Forces single query."
    )


def _is_likely_compound(question: str) -> bool:
    """Fast heuristic — returns False for obviously simple queries to skip the LLM call."""
    q = question.lower()
    return any(indicator in q for indicator in COMPOUND_INDICATORS)


def decomposer_node(state: AgentState) -> dict:
    question = state.get("original_query", "")
    clarification = state.get("clarification", "")
    full_question = f"{question} {clarification}".strip() if clarification else question

    if settings.mock_mode:
        return {"sub_queries": [full_question]}

    # Fast path: skip LLM call for obviously simple queries.
    if not _is_likely_compound(full_question):
        logger.info(f"decomposition fast_path=true query={repr(full_question)}")
        return {"sub_queries": [full_question]}

    llm = get_llm(temperature=0).with_structured_output(DecompositionResult)
    result: DecompositionResult = llm.invoke([
        SystemMessage(content=DECOMPOSER_SYSTEM_PROMPT),
        HumanMessage(content=full_question),
    ])

    # Clamp to 1-3 queries (schema constraints removed for Cerebras compatibility).
    if not result.queries:
        result.queries = [full_question]
    result.queries = result.queries[:3]

    # If the LLM detected sequential dependencies, collapse back to single query.
    if result.has_dependencies:
        logger.info(f"decomposition has_dependencies=true collapsing to single query")
        return {"sub_queries": [full_question]}

    logger.info(
        f"decomposition compound={result.is_compound} "
        f"has_dependencies={result.has_dependencies} "
        f"queries={result.queries}"
    )
    return {"sub_queries": result.queries}
