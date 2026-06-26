"""Critic agent.

After the Synthesis agent writes an answer, the Critic challenges it —
surfacing gaps, risks, counter-arguments, and caveats the answer glossed over.

Only runs on research-mode queries; chat answers are not critiqued.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are the Critic Agent in an AI research workspace.
You have just seen a research answer written for a user.

Your job is to add perspective the answer is missing. Cover 2-4 of:
- Important caveats or data limitations the user should know
- Counter-arguments or alternative interpretations of the evidence
- Risks or downsides that were glossed over
- What is still unknown, unverified, or likely to change soon

Rules:
- Do NOT repeat what the answer already said — only add what is missing.
- Be specific. Generic warnings ("always verify sources") are useless.
- Be concise: 2-4 bullet points maximum.
- Use markdown bullet syntax."""


def critic_node(state: AgentState) -> dict:
    if settings.mock_mode:
        return {
            "critique": (
                "- Funding figures are often self-reported and unverified by independent parties.\n"
                "- Valuations reflect investor sentiment at a point in time and can shift quickly.\n"
                "- Leadership changes announced publicly may lag actual internal reorganisations."
            )
        }

    question = state.get("original_query", "")
    synthesis = state["messages"][-1].content

    llm = get_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=f"User question: {question}\n\nResearch answer:\n{synthesis}"),
    ])
    logger.info("critic_complete")
    return {"critique": response.content}
