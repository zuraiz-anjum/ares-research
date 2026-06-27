"""Shared graph state.

Every agent reads from and writes to this state object as the conversation
flows through the graph. The `messages` list is the running conversation
transcript that gives each agent the context of previous turns.
"""

from typing import Optional, TypedDict

from langchain_core.messages import AnyMessage


class AgentState(TypedDict, total=False):
    # Running conversation transcript, shared across every agent and turn.
    messages: list[AnyMessage]

    # Clarity agent output.
    clarity_status: str            # "clear" | "needs_clarification"
    clarification: Optional[str]   # the user's answer to a clarifying question

    # Research agent output.
    findings: str
    raw_research: str
    confidence_score: int          # 0-10

    # Validator agent output.
    validation_result: str         # "sufficient" | "insufficient"
    attempts: int                  # number of research attempts so far

    original_query: str

    # Decomposer agent output.
    sub_queries: list[str]          # parallel sub-queries for compound questions

    # Intent router output.
    mode: str                       # "research" | "report" | "chat" | "document"

    # Document agent output.
    source_url: str

    # Critic agent output.
    critique: str

    # Suggestions agent output.
    suggestions: list[str]

    # Fact Checker agent output.
    fact_check_results: list          # list[{"claim": str, "status": str, "note": str}]

    # Planner agent output.
    plan_steps: list[str]

    # Chart writer output.
    chart_url: str        # public path to the rendered PNG

    # PDF generator output.
    pdf_url: str          # public path to the generated PDF

    # Research sources collected from Tavily results.
    sources: list          # list[{"title": str, "url": str}]

    # Request telemetry (populated in main.py after graph completes).
    token_count: int      # approximate tokens used this turn
    provider_used: str    # which LLM provider served the request
    latency_ms: int       # wall-clock time for the graph run