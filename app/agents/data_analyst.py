"""Data Analyst agent.

Handles the 'data_analysis' pipeline mode. Extracts numerical data from
research findings, builds markdown comparison tables, and highlights key
metrics — all in a single streaming response.

Triggered when the user asks about metrics, comparisons, or statistics.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import settings, truncate_to_budget
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

DATA_ANALYST_SYSTEM_PROMPT = """You are the Data Analyst agent in an AI research workspace.

You have received raw research findings. Your job is to structure and present
the numerical and comparative data in a clear, scannable format.

Output format:
1. A markdown comparison table if multiple entities are being compared (required
   when data exists for more than one entity). Use clean | column | formatting.
2. **Key Metrics** — a bullet list of the 4-6 most significant data points with
   exact figures (revenue, growth %, valuations, headcount, dates, rankings).
3. **Analysis** — 2-3 sentences interpreting what the numbers mean. What is
   notable, surprising, or worth watching?

Rules:
- Only include numbers that appear in the source material. Never fabricate figures.
- If data is sparse, state that clearly rather than padding with vague claims.
- Prefer specificity: "$4.2B Series D in March 2025" beats "significant funding".
- Format all large numbers consistently (e.g. $1.2B, not $1,200,000,000)."""


async def data_analyst_node(state: AgentState) -> dict:
    question = state.get("original_query", "")
    raw = state.get("raw_research", "") or state.get("findings", "")
    sub_queries = state.get("sub_queries", [])

    if settings.mock_mode:
        mock = (
            "| Metric | Company A | Company B |\n"
            "|---|---|---|\n"
            "| Valuation | $10B | $8B |\n"
            "| Revenue Growth | 30% YoY | 22% YoY |\n"
            "| Last Funding | $500M Series D | $300M Series C |\n"
            "| Employees | 2,500 | 1,800 |\n\n"
            "**Key Metrics:**\n"
            "- Company A raised $500M at a $10B valuation in Q1 2025\n"
            "- Company B's revenue growth has accelerated from 18% to 22% YoY\n"
            "- Both companies are hiring aggressively in engineering\n"
            "- Company A leads on absolute valuation but B has higher revenue/employee\n\n"
            "**Analysis:**\n"
            "Company A holds a clear valuation premium, though Company B's improving "
            "unit economics suggest it may close the gap by 2026. Both remain well-funded "
            "with runway exceeding 24 months at current burn rates."
        )
        return {"messages": [AIMessage(content=mock)]}

    raw, _ = truncate_to_budget(raw, label="data_analyst")

    context = f"Research question: {question}\n"
    if sub_queries and len(sub_queries) > 1:
        context += f"Entities researched: {', '.join(sub_queries)}\n"
    context += f"\nRaw findings:\n{raw}"

    llm = get_llm(streaming=True)
    response = await llm.ainvoke([
        SystemMessage(content=DATA_ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ])
    logger.info("data_analyst_complete")
    return {"messages": [AIMessage(content=response.content)]}
