"""Clarity agent.

Decides whether the user's request is specific enough to research. If a
company is not identifiable or the question is too vague, it pauses the graph
and asks the user a clarifying question (human-in-the-loop) before continuing.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.llm import get_llm
from app.state import AgentState

CLARITY_SYSTEM_PROMPT = """You are the Clarity Agent in a company-research assistant.
Decide whether the user's latest request is specific enough to begin research.

A request is "clear" when it names (or unambiguously implies) a specific company
and a researchable intent. It "needs_clarification" when no company can be
identified or the ask is too broad to act on.

If clarification is needed, write a single, friendly question that would unblock
you (e.g. "Which company are you asking about?")."""


class ClarityVerdict(BaseModel):
    status: Literal["clear", "needs_clarification"]
    question: str = Field(default="", description="Question to ask when unclear.")


def clarity_node(state: AgentState) -> dict:
    user_query = state["messages"][-1].content

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
