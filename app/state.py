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

    # Report writer output (stored explicitly so pdf_generator can read it
    # even after chart_writer appends its own message to state["messages"]).
    report_content: str

    # Draft critic / writer revision loop.
    revision_count:    int   # number of times draft_critic has run this turn
    revision_feedback: str   # critic's feedback for the next writer pass; "" = accepted
    draft_score:       int   # quality score 1-10 from the most recent critic run

    # Chart writer / data_visualizer output.
    chart_url: str         # public path to the primary PNG (backward compat)
    chart_urls: list       # list[str] — all PNGs from data_visualizer / survey_analyst

    # PDF generator output.
    pdf_url: str           # public path to the generated PDF

    # Research sources collected from Tavily results.
    sources: list          # list[{"title": str, "url": str}]

    # RAG: ID of the document uploaded by the user for this session.
    doc_id: str            # hex UUID returned by POST /upload

    # Survey mode: True when the attached file is a CSV/Excel survey.
    is_survey: bool

    # Long-conversation memory: compressed summary of older turns.
    conversation_summary: str

    # Academic writer output (extracted from the paper for PDF rendering).
    paper_abstract: str   # 150-200 word abstract block
    paper_keywords: str   # comma-separated keyword list

    # Challenger (Research ↔ Synthesis debate) output.
    challenge_queries: list  # list[str] — Tavily queries used to find counter-evidence
    counter_evidence:  str   # raw counter-evidence text found by challenger
    synthesis_draft:   str   # first-pass synthesis answer (before reconciliation)

    # Voting synthesis output (research mode only).
    vote_winner:    str   # "analyst" | "devils_advocate" | "communicator"
    vote_reason:    str   # one-sentence judge explanation
    vote_responses: dict  # {"analyst": str, "devils_advocate": str, "communicator": str}

    # Request telemetry (populated in main.py after graph completes).
    token_count: int      # approximate tokens used this turn
    provider_used: str    # which LLM provider served the request
    latency_ms: int       # wall-clock time for the graph run