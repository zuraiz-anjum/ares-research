"""Debate Writer agent.

For 'debate' mode. After the research pipeline gathers evidence, the Debate
Writer structures it into a rigorous FOR vs AGAINST analysis with a verdict.

The output is streamed token-by-token so the user sees the sections build up
in real time. No additional critic pass — the debate format is self-balancing.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

DEBATE_SYSTEM_PROMPT = """You are the Debate Writer in an AI research workspace.

Given research findings and a question, produce a rigorous, evidence-backed
debate analysis using this exact structure:

## Arguments For
Provide 4–5 specific, evidence-backed arguments in favour of the proposition.
Every argument MUST cite a real fact, figure, or event from the research.

## Arguments Against
Provide 4–5 specific counter-arguments with equal rigour. No strawmen.
Every argument MUST cite a real fact, figure, or event from the research.

## Verdict
2–3 sentences. Honest, nuanced conclusion. If the evidence strongly favours
one side, say so plainly. Don't dodge the complexity.

Rules:
- Never repeat the same point on both sides.
- Both sections must be equally rigorous.
- Cite specific numbers and named events — not vague claims."""


async def debate_writer_node(state: AgentState) -> dict:
    question = state.get("original_query", "")
    research = state.get("findings", "") or state.get("raw_research", "")

    if not research:
        research = "No specific research data available — rely on general knowledge."

    history = state.get("messages", [])

    llm = get_llm(streaming=True)
    response = await llm.ainvoke([
        SystemMessage(content=DEBATE_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {question}\n\nResearch findings:\n{research[:3000]}"),
        *history,
    ])

    logger.info("debate_writer complete")
    return {"messages": [AIMessage(content=response.content)]}
