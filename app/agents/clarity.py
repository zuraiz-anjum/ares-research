"""Clarity agent.

Decides whether the user's request is specific enough to research, and
detects whether the user wants an academic paper so that it can collect
the four upfront requirements before the pipeline starts.

Academic detection is LLM-based — no hardcoded phrases — so it handles
any natural wording: "write a paper", "something like google scholar",
"make it look like a journal article", "scholarly write-up", etc.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import get_llm
from app.state import AgentState

ACADEMIC_MCQ = {
    "type": "academic_mcq",
    "questions": [
        {
            "id": "angle",
            "label": "Research angle",
            "text": "What specific angle should the paper argue?",
            "options": [
                "Comparative performance & accuracy benchmarks",
                "Efficiency, cost & resource usage",
                "Real-world applicability & deployment use cases",
                "Survey of existing literature & approaches",
            ],
        },
        {
            "id": "audience",
            "label": "Target audience",
            "text": "Who will read this paper?",
            "options": [
                "Undergraduate students",
                "Graduate researchers",
                "Industry professionals",
                "General academic audience",
            ],
        },
        {
            "id": "scope",
            "label": "Scope & constraints",
            "text": "Any time period or domain constraints?",
            "options": [
                "No constraints",
                "2020–2025 only",
                "Computer science / AI domain",
                "Industry-focused applications",
            ],
        },
        {
            "id": "style",
            "label": "Paper style",
            "text": "What type of paper should this be?",
            "options": [
                "Literature review",
                "Empirical analysis",
                "Survey paper",
                "Theoretical framework / case study",
            ],
        },
    ],
}

CLARITY_SYSTEM_PROMPT = """You are the Clarity Agent in an AI workspace assistant.
Your job is to analyse the user's request and return a JSON object with two fields.

--- FIELD 1: status ---
Return "clear" if the request can be acted on as-is:
  - It names at least one identifiable subject (company, person, topic, technology, etc.)
  - The intent is understandable (research, compare, explain, code, write, plan, chart, etc.)
  - Missing details like time period or format are NOT reasons to ask — the system makes
    reasonable assumptions.

Return "needs_clarification" ONLY if:
  - There is NO identifiable subject ("tell me about it", "what about them?")
  - The intent is completely unknowable ("just do the thing")
  - Without clarification the request literally cannot be started.

--- FIELD 2: is_academic_paper ---
Return true if the user is asking to produce an academic or scholarly paper —
regardless of the exact words used. Consider ALL of the following as academic:
  - Explicit: "write a paper on X", "research paper", "academic paper", "journal article",
    "conference paper", "literature review", "systematic review", "survey paper"
  - Implicit / paraphrased: "something like a Google Scholar paper", "make it look like
    a journal", "scholarly write-up", "in academic format", "write a study on",
    "academic-style report", "write it like a research publication", "I need a thesis on"
  - Any request where the clear output is a structured academic document with Abstract,
    numbered sections, and a References list.

Return false for everything else: general research, PDF reports, comparisons, emails,
code, chat, plans, data analysis, and standard reports.

--- FIELD 3: question ---
Only populated when status is "needs_clarification". Write one short, friendly question.

--- CONVERSATION HISTORY ---
If a RECENT CONVERSATION block is provided, the current message may be a
follow-up (e.g. "which benchmarks were used?", "what about the other one?").
Use the history to resolve pronouns and implicit references before deciding
status — a short question is NOT ambiguous if the history already establishes
its subject. When history resolves the reference, also fill FIELD 4 below.

--- FIELD 4: standalone_query ---
Only populated when the current message is a follow-up whose subject comes
from history (e.g. current message "which benchmarks were used?" + history
about "the transformer efficiency paper" -> "Which benchmarks were used in
the transformer efficiency paper?"). Leave empty if the current message is
already self-contained — do not rephrase it unnecessarily."""


class ClarityVerdict(BaseModel):
    status: Literal["clear", "needs_clarification"]
    is_academic_paper: bool = Field(
        default=False,
        description="True if the user wants an academic/scholarly paper output.",
    )
    question: str = Field(default="", description="Question to ask when status is needs_clarification.")
    standalone_query: str = Field(
        default="",
        description="Context-resolved rewrite of a follow-up question; empty if not needed.",
    )


# How many prior messages to show the clarity LLM for resolving follow-up
# references. Kept short — this is just enough for "which benchmarks were
# used in [the thing we discussed]?", not a substitute for synthesis's own
# full history/summary handling.
_HISTORY_WINDOW = 6


def _recent_history_text(state: AgentState) -> str:
    prior = state["messages"][:-1][-_HISTORY_WINDOW:]
    if not prior:
        return ""
    lines = [
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {str(m.content)[:300]}"
        for m in prior
    ]
    return "\n".join(lines)


def clarity_node(state: AgentState) -> dict:
    user_query = state["messages"][-1].content

    # When a document is attached, the query refers to that document — no need
    # to ask for clarification regardless of how vague the wording is.
    if state.get("doc_id"):
        return {"clarity_status": "clear", "original_query": user_query}

    if settings.mock_mode:
        return {"clarity_status": "clear", "original_query": user_query}

    history = _recent_history_text(state)
    human_content = (
        f"RECENT CONVERSATION:\n{history}\n\nCurrent message: {user_query}"
        if history else user_query
    )

    llm = get_llm(temperature=0).with_structured_output(ClarityVerdict)
    verdict: ClarityVerdict = llm.invoke(
        [SystemMessage(content=CLARITY_SYSTEM_PROMPT), HumanMessage(content=human_content)]
    )
    resolved_query = verdict.standalone_query or user_query

    # ── Academic paper: collect the four upfront requirements ──────────────
    # Always fires before general clarification so the pipeline gets the full
    # paper spec from the very start, even when the topic itself is clear.
    if verdict.is_academic_paper:
        answers = interrupt(ACADEMIC_MCQ)
        enriched_query = (
            f"{resolved_query}\n\n"
            f"--- Paper requirements (provided by user) ---\n"
            f"{answers}"
        )
        return {
            "clarity_status": "clear",
            "clarification": answers,
            "original_query": enriched_query,
        }

    # ── General clarity check ──────────────────────────────────────────────
    if verdict.status == "needs_clarification":
        answer = interrupt({"question": verdict.question})
        return {"clarity_status": "clear", "clarification": answer, "original_query": user_query}

    return {"clarity_status": "clear", "original_query": resolved_query}
