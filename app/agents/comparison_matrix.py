"""Comparison Matrix agent.

Triggered by "X vs Y", "compare X and Y", "side by side".
Takes research findings about multiple entities and produces a structured
markdown comparison table highlighting winners per category, plus a verdict.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import settings, truncate_to_budget
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

COMPARISON_SYSTEM_PROMPT = """You are the Comparison Matrix agent in an AI research workspace.

You receive research findings about multiple entities (companies, products, tools, etc.).
Your job is to produce a high-quality side-by-side comparison.

Output format — always use ALL three sections:

## Comparison Matrix

A clean markdown table. Columns: Dimension | Entity A | Entity B | (Entity C if present).
Rows must cover relevant dimensions from this set (use only those applicable):
Founding / Age, Valuation / Market Cap, Annual Revenue, Funding Total, Last Funding Round,
Headcount, Key Products, CEO / Leadership, Headquarters, Business Model, Growth Rate,
Notable Customers, Strengths, Weaknesses.

Use ✓ to mark the clear winner per row (only when there's a clear winner — skip if tied or unclear).
Bold the winner's value in that row.

## Key Takeaways

3-5 bullets. Each starts with an entity name or a theme, then the key insight.
Be specific with numbers from the research.

## Verdict

2-3 sentences. Recommend which entity wins overall and for which use case / audience.
Be opinionated but nuanced.

Rules:
- Never fabricate numbers. If a figure is missing, write "N/A".
- Keep all cell values short (≤ 15 words).
- Prefer specific figures over adjectives ("$4.2B" beats "significant").
"""


async def comparison_matrix_node(state: AgentState) -> dict:
    question = state.get("original_query", "")
    raw = state.get("raw_research", "") or state.get("findings", "")
    sub_queries = state.get("sub_queries", [])

    if settings.mock_mode:
        mock = (
            "## Comparison Matrix\n\n"
            "| Dimension | Company A | Company B |\n"
            "|---|---|---|\n"
            "| Valuation | **$10B** ✓ | $8B |\n"
            "| Revenue Growth | 30% YoY | **35% YoY** ✓ |\n"
            "| Headcount | **2,500** ✓ | 1,800 |\n"
            "| Last Round | $500M Series D | $300M Series C |\n"
            "| Business Model | B2B SaaS | B2B + B2C |\n\n"
            "## Key Takeaways\n\n"
            "- **Company A** commands a higher valuation but Company B is growing faster\n"
            "- **Company B** has superior revenue per employee\n"
            "- Both are well-funded with 24+ months runway\n\n"
            "## Verdict\n\n"
            "Company A leads on scale and market presence. Company B is the better bet "
            "for growth-focused investors given its faster trajectory and better unit economics."
        )
        return {"messages": [AIMessage(content=mock)]}

    raw, _ = truncate_to_budget(raw, label="comparison_raw")

    context = f"Comparison question: {question}\n"
    if sub_queries:
        context += f"Entities: {', '.join(sub_queries)}\n"
    context += f"\nResearch findings:\n{raw}"

    llm = get_llm(streaming=True)
    response = await llm.ainvoke([
        SystemMessage(content=COMPARISON_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ])
    logger.info("comparison_matrix_complete")
    return {"messages": [AIMessage(content=response.content)]}
