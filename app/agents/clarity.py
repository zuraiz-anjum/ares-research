"""Clarity agent.

Decides whether the user's request is specific enough to research. If a
company is not identifiable or the question is too vague, it pauses the graph
and asks the user a clarifying question (human-in-the-loop) before continuing.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import get_llm
from app.state import AgentState

CLARITY_SYSTEM_PROMPT = """You are the Clarity Agent in an AI workspace assistant.
Your job is to decide if a request can be acted on AS-IS, or if it truly cannot
proceed without more information.

Return "clear" if:
  - The request names at least one identifiable subject (company, person, topic, etc.)
  - The intent is understandable (research, compare, explain, code, write, plan)
  - This includes: company research, data comparisons (even if time period unspecified —
    assume most recent available), coding tasks, explanations, debates, plans, document
    analysis, or general knowledge questions.
  - IMPORTANT: Missing details like time period, specific metric, or format are NOT
    reasons to ask for clarification. The agent can make reasonable assumptions.

Return "needs_clarification" ONLY if:
  - There is NO identifiable subject (e.g. "tell me about it", "what about them?")
  - The intent is completely unknowable (e.g. "just do the thing")
  - Without clarification the request literally cannot be started.

Examples of "clear":
  - "Compare revenue numbers for Stripe and Brex" → clear (subjects + intent known)
  - "What is OpenAI's valuation?" → clear
  - "Write a function to sort a list" → clear
  - "Pros and cons of Python vs Go" → clear
  - "Tell me about Anthropic" → clear

Examples of "needs_clarification":
  - "Tell me about them" (no subject) → needs_clarification
  - "Look it up" (no subject or topic) → needs_clarification

If clarification IS needed, write one short, friendly question."""


class ClarityVerdict(BaseModel):
    status: Literal["clear", "needs_clarification"]
    question: str = Field(default="", description="Question to ask when unclear.")


def clarity_node(state: AgentState) -> dict:
    user_query = state["messages"][-1].content

    if settings.mock_mode:
        return {"clarity_status": "clear", "original_query": user_query}

    llm = get_llm(temperature=0).with_structured_output(ClarityVerdict)
    verdict: ClarityVerdict = llm.invoke(
        [SystemMessage(content=CLARITY_SYSTEM_PROMPT), HumanMessage(content=user_query)]
    )

    if verdict.status == "needs_clarification":
        # Pause the graph and surface the question to the user. Execution
        # resumes here once the user replies with a clarification.
        answer = interrupt({"question": verdict.question})
        return {"clarity_status": "clear", "clarification": answer, "original_query": user_query}

    return {"clarity_status": "clear", "original_query": user_query}
