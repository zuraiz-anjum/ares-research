"""Report Writer agent.

Replaces Synthesis for "report" mode queries. Takes the research findings
and produces a structured, professional markdown report — not a conversational
answer.

Streams tokens (streaming=True) so the report appears in real time in the UI.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import settings, truncate_to_budget
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

REPORT_SYSTEM_PROMPT = """You are the Report Writer in an AI research workspace.
Turn raw research findings into a professional, structured markdown report that a
stakeholder can read directly — no filler, no padding.

Use these five sections (## headers):

## Executive Summary
2-3 sentence overview: what was researched and what the headline finding is.

## Key Findings
Bullet-point list of the most important facts (numbers, names, dates where available).

## Detailed Analysis
Expand on the findings with context, reasoning, and any notable patterns.

## Risks & Considerations
What should the reader watch out for? Counterpoints, data limitations, open questions.

## Conclusion
1-2 sentence closing statement: what this means and what to watch next.

Rules:
- Be specific — cite figures, names, and dates when available.
- Avoid generic phrases like "it is important to note" or "in conclusion".
- If multiple entities were researched, give each its own sub-section under Key Findings.
- Output ONLY the five sections above. Do NOT add appendices, chart code, mermaid diagrams,
  LaTeX, PDF assembly instructions, or any meta-content about how to generate charts or
  export the document. Charts and PDFs are handled automatically by separate pipeline agents.
- Use plain ASCII hyphens (-) for compound words and dashes. Do not use Unicode dashes,
  non-breaking hyphens, or special typographic characters."""


async def report_writer_node(state: AgentState) -> dict:
    question = state.get("original_query", "")
    findings = state.get("findings", "") or state.get("raw_research", "")
    sub_queries = state.get("sub_queries", [])

    if settings.mock_mode:
        mock = (
            "## Executive Summary\n"
            "The company shows strong momentum with 30% YoY revenue growth and a recent $500M funding round.\n\n"
            "## Key Findings\n"
            "- Revenue growth: 30% year-over-year\n"
            "- Funding: $500M Series D at $10B valuation\n"
            "- Leadership: Three C-suite hires in Q3 2025\n\n"
            "## Detailed Analysis\n"
            "Mock analysis: The funding round signals strong investor confidence...\n\n"
            "## Risks & Considerations\n"
            "- Valuation may be inflated relative to current revenue multiples\n"
            "- Leadership transitions introduce short-term execution risk\n\n"
            "## Conclusion\n"
            "The company is well-positioned for growth, though execution risk warrants monitoring."
        )
        return {"messages": [AIMessage(content=mock)], "report_content": mock}

    findings, _ = truncate_to_budget(findings, label="report_findings")

    context = f"Research topic: {question}\n"
    if sub_queries and len(sub_queries) > 1:
        context += f"Sub-topics researched in parallel: {', '.join(sub_queries)}\n"
    context += f"\nFindings:\n{findings}"

    llm = get_llm(streaming=True)
    response = await llm.ainvoke([
        SystemMessage(content=REPORT_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ])
    logger.info("report_writer_complete")
    return {
        "messages": [AIMessage(content=response.content)],
        "report_content": response.content,
    }
