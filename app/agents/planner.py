"""Planner agent.

For complex, multi-part tasks the Planner creates an explicit numbered
research plan before any searching begins. The plan is surfaced to the user
via an SSE event so they can see the agent's reasoning before results arrive.

The Planner also sets sub_queries so the Research agent knows exactly what
to search for — replacing the Decomposer for plan-mode queries.
"""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are the Research Planner in an AI workspace.

Given a complex question or task, produce a concise numbered research plan.
Each step must be a specific, independently searchable query — not a vague
description.

Output exactly the numbered list, one step per line, no extra text.
Maximum 4 steps.

Example:
1. Stripe revenue growth and valuation 2023-2025
2. Stripe recent product launches and acquisitions
3. Stripe competitive position vs Adyen and PayPal
4. Stripe regulatory challenges and international expansion"""


class PlanResult(BaseModel):
    # min_length/max_length omitted — Cerebras rejects those JSON schema keywords.
    steps: list[str] = Field(
        description="Numbered research steps (just the text, no numbers).",
    )


def _parse_steps(text: str) -> list[str]:
    """Extract step text from a numbered list, stripping the leading number."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    steps = []
    for line in lines:
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if cleaned:
            steps.append(cleaned)
    return steps[:4]


def planner_node(state: AgentState) -> dict:
    question = state.get("original_query", "")
    clarification = state.get("clarification", "")
    full_question = f"{question} {clarification}".strip() if clarification else question

    if settings.mock_mode:
        steps = [
            f"Overview and background of: {full_question}",
            f"Recent news and developments",
            f"Key metrics and financial data",
            f"Competitive landscape and risks",
        ]
        return {"plan_steps": steps, "sub_queries": steps}

    llm = get_llm(temperature=0).with_structured_output(PlanResult)
    try:
        result: PlanResult = llm.invoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=full_question),
        ])
        steps = result.steps
    except Exception:
        # Fallback: plain text parse
        llm_plain = get_llm(temperature=0)
        response = llm_plain.invoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=full_question),
        ])
        steps = _parse_steps(response.content)

    if not steps:
        steps = [full_question]

    logger.info(f"planner steps={len(steps)} query={repr(full_question)}")
    return {"plan_steps": steps, "sub_queries": steps}
