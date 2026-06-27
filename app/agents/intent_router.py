"""Intent Router agent.

Classifies the user's request into a pipeline mode immediately after Clarity
confirms the question is actionable.

Modes
-----
research      Company/market research — conversational answer + fact-check + critique.
report        Same research pipeline but formatted as a structured multi-section report.
data_analysis Research for numerical comparison — outputs tables + key metrics.
debate        Research + structured FOR vs AGAINST analysis with a verdict.
plan          Complex multi-step task — Planner creates a visible research plan first.
chat          Answered from model knowledge, no web search needed.
document      User supplied a URL; fetch and analyse that document.
code          Write, debug, or explain code — no web search needed.
"""

import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://")

_REPORT_SIGNALS = {
    "write a report", "create a report", "make a report", "generate a report",
    "write an analysis", "create an analysis", "research report",
    "write me a report", "write me an analysis",
    "detailed report", "comprehensive report", "full report", "in-depth report",
    "short report", "brief report", "report on", "write report",
}

_DATA_SIGNALS = {
    "compare metrics", "compare numbers", "compare the data", "compare stats",
    "revenue numbers", "funding numbers", "show me the data", "show the numbers",
    "breakdown of", "statistics", "growth rate", "market share", "headcount",
    "employees", "revenue vs", "valuation vs", "by the numbers",
}

_CODE_SIGNALS = {
    "write code", "write a function", "write a class", "write a script",
    "implement", "code to", "code that", "program to", "algorithm",
    "function that", "how to code", "python code", "javascript code",
    "typescript", "debug", "fix this code", "fix the bug", "refactor",
    "optimize this code", "write a test", "unit test",
}

_RESEARCH_SIGNALS = {
    "company", "startup", "funding", "series a", "series b", "series c",
    "ipo", "valuation", "revenue", "ceo", "cto", "cfo", "founded",
    "acquisition", "merger", "stock", "earnings", "market cap", "investors",
    "venture", "vc", "competitor", "compete", "product launch", "layoffs",
    "recent news", "latest news",
}

_DEBATE_SIGNALS = {
    "pros and cons", "advantages and disadvantages", "should i",
    "steelman", "debate", "argument for", "argument against",
    "is it worth", "worth it", "for and against", "pros vs cons",
    "make the case", "best option", "which is better",
}

_PLAN_SIGNALS = {
    "step by step plan", "create a plan", "make a plan", "plan for",
    "strategy for", "roadmap for", "how to approach", "help me figure out",
    "what steps", "plan my", "plan to build", "plan to launch",
    "action plan", "how do i achieve", "how do i get",
}

_CHART_SIGNALS = {
    "chart", "bar chart", "line chart", "pie chart", "scatter plot",
    "histogram", "make a chart", "create a chart", "draw a chart",
    "show a chart", "plot this", "plot the", "visualize", "visualise",
    "visualization", "graph this", "graph the data",
}

_PDF_SIGNALS = {
    "generate pdf", "export pdf", "download pdf", "as a pdf",
    "pdf report", "pdf document", "create pdf", "make a pdf",
    "pdf version", "pdf file", "save as pdf", "export as pdf",
    "pdf export", "as pdf",
}

ROUTER_SYSTEM_PROMPT = """You are the Intent Router in an AI workspace.
Classify the user's request into exactly one mode.

research      — needs live web search: company news, funding, leadership, market
                data, competitive analysis, recent events about organisations.

report        — same as research but the user explicitly wants a formatted,
                structured report (e.g. "write a report on", "full analysis").

data_analysis — same as research but the user wants numbers, metrics, tables,
                or statistical comparisons (e.g. "compare the numbers for X and Y").

debate        — the user wants a balanced FOR vs AGAINST analysis of a topic,
                decision, or comparison (e.g. "pros and cons", "should I", "steelman").

plan          — the user has a complex multi-step task or goal and needs an
                explicit research plan before execution (e.g. "create a plan",
                "strategy for", "step by step", "roadmap").

chat          — answered from general knowledge without live search: history,
                science, definitions, writing help, explanations, opinions.

document      — the user provided a URL or wants a specific web page analysed.

code          — write, debug, explain, or refactor code; no web search needed.

Reply with the mode name only — no explanation, no punctuation."""


class RouteResult(BaseModel):
    mode: Literal["research", "report", "data_analysis", "debate", "plan", "chat", "document", "code", "chart", "pdf"] = Field(
        description="Pipeline mode that best fits the user's request."
    )


def intent_router_node(state: AgentState) -> dict:
    question = state.get("original_query", "")

    if settings.mock_mode:
        return {"mode": "research"}

    q_lower = question.lower()

    # Fast paths — ordered from most specific to least.
    if _URL_RE.search(question):
        logger.info("intent fast_path=document")
        return {"mode": "document"}

    if any(sig in q_lower for sig in _PDF_SIGNALS):
        logger.info("intent fast_path=pdf")
        return {"mode": "pdf"}

    if any(sig in q_lower for sig in _CHART_SIGNALS):
        logger.info("intent fast_path=chart")
        return {"mode": "chart"}

    if any(sig in q_lower for sig in _CODE_SIGNALS):
        logger.info("intent fast_path=code")
        return {"mode": "code"}

    if any(sig in q_lower for sig in _REPORT_SIGNALS):
        logger.info("intent fast_path=report")
        return {"mode": "report"}

    if any(sig in q_lower for sig in _DATA_SIGNALS):
        logger.info("intent fast_path=data_analysis")
        return {"mode": "data_analysis"}

    if any(sig in q_lower for sig in _DEBATE_SIGNALS):
        logger.info("intent fast_path=debate")
        return {"mode": "debate"}

    if any(sig in q_lower for sig in _PLAN_SIGNALS):
        logger.info("intent fast_path=plan")
        return {"mode": "plan"}

    if any(sig in q_lower for sig in _RESEARCH_SIGNALS):
        logger.info("intent fast_path=research")
        return {"mode": "research"}

    # LLM for ambiguous cases.
    llm = get_llm(temperature=0).with_structured_output(RouteResult)
    result: RouteResult = llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ])
    logger.info(f"intent mode={result.mode} query={repr(question)}")
    return {"mode": result.mode}
