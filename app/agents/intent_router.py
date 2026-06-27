"""Intent Router agent.

Classifies the user's request into a pipeline mode immediately after Clarity
confirms the question is actionable.

Modes
-----
research      Company/market research — conversational answer + fact-check + critique.
report        Same research pipeline but formatted as a structured multi-section report.
data_analysis Research for numerical comparison — outputs tables + key metrics + auto-chart.
comparison    Side-by-side comparison matrix of 2+ entities — auto-chart included.
debate        Research + structured FOR vs AGAINST analysis with a verdict.
plan          Complex multi-step task — Planner creates a visible research plan first.
chat          Answered from model knowledge, no web search needed.
document      User supplied a URL; fetch and analyse that document.
code          Write, debug, or explain code — no web search needed.
email         Research a topic, then draft a ready-to-send professional email.
chart         Render user-supplied data as a matplotlib chart (no web search).
pdf           Full research pipeline + export to a styled PDF document.
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

# Word-boundary regex patterns are used for PDF and chart detection
# instead of phrase lists — a human says "pdf" in too many ways to
# enumerate ("in pdf form", "as a pdf please", "pdf it", "pdf version",
# "give me a pdf", etc.).  A single \bpdf\b catches all of them.
_PDF_RE    = re.compile(r'\bpdf\b', re.IGNORECASE)
_CHART_RE  = re.compile(
    r'\b(chart|bar chart|line chart|pie chart|scatter|histogram|'
    r'plot|visuali[sz]e?|visualization|visualisation|graph the|graph this)\b',
    re.IGNORECASE,
)

# Legacy phrase sets kept for fallback completeness.
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
    "pdf export", "as pdf", "in pdf", "pdf form", "pdf format",
    "to pdf", "into a pdf", "in a pdf", "give me a pdf",
}

_COMPARISON_SIGNALS = {
    "vs ", "versus", "compare ", "side by side", "side-by-side",
    "compared to", "compared with", "difference between", "similarities between",
    "better: ", "which is better", "head to head", "head-to-head",
    "x vs y", "a vs b", "pros and cons of both",
}

_EMAIL_SIGNALS = {
    "draft an email", "write an email", "draft email", "write email",
    "compose an email", "compose email", "email about", "email to",
    "email summarizing", "email summarising", "email draft",
    "send an email", "prepare an email", "craft an email",
}

ROUTER_SYSTEM_PROMPT = """You are the Intent Router in an AI workspace.
Classify the user's request into exactly one mode.

research      — needs live web search: company news, funding, leadership, market
                data, competitive analysis, recent events about organisations.

report        — same as research but the user explicitly wants a formatted,
                structured report (e.g. "write a report on", "full analysis").

data_analysis — same as research but the user wants numbers, metrics, tables,
                or statistical comparisons (e.g. "compare the numbers for X and Y").

comparison    — side-by-side structured comparison of 2+ named entities (companies,
                tools, products) with a winner verdict (e.g. "Stripe vs Braintree",
                "compare OpenAI and Anthropic", "Tesla versus Rivian").

debate        — the user wants a balanced FOR vs AGAINST analysis of a topic,
                decision, or comparison (e.g. "pros and cons", "should I", "steelman").

plan          — the user has a complex multi-step task or goal and needs an
                explicit research plan before execution (e.g. "create a plan",
                "strategy for", "step by step", "roadmap").

chat          — answered from general knowledge without live search: history,
                science, definitions, writing help, explanations, opinions.

document      — the user provided a URL or wants a specific web page analysed.

code          — write, debug, explain, or refactor code; no web search needed.

email         — the user wants to draft a professional email about a topic
                (e.g. "draft an email to my manager about Stripe's funding").

chart         — render data as a visual chart. Triggered whenever the user
                mentions "chart", "graph", "plot", "visualize", or provides
                raw numbers they want rendered as a bar/line/pie chart.

pdf           — research a topic AND export the result as a downloadable PDF
                document. Triggered by ANY mention of "pdf", "pdf report",
                "in pdf form", "as a pdf", "pdf format", etc.
                IMPORTANT: if the user asks for BOTH a comparison/report AND
                a pdf, always choose pdf — the pdf pipeline includes the full
                research and report writing, then exports to PDF.

Reply with the mode name only — no explanation, no punctuation."""


class RouteResult(BaseModel):
    mode: Literal[
        "research", "report", "data_analysis", "comparison", "debate",
        "plan", "chat", "document", "code", "email", "chart", "pdf"
    ] = Field(description="Pipeline mode that best fits the user's request.")


def intent_router_node(state: AgentState) -> dict:
    question = state.get("original_query", "")

    if settings.mock_mode:
        return {"mode": "research"}

    q_lower = question.lower()

    # Fast paths — ordered from most specific to least.
    if _URL_RE.search(question):
        logger.info("intent fast_path=document")
        return {"mode": "document"}

    # Regex word-boundary check is the primary gate; phrase-list is backup.
    if _PDF_RE.search(question) or any(sig in q_lower for sig in _PDF_SIGNALS):
        logger.info("intent fast_path=pdf")
        return {"mode": "pdf"}

    if _CHART_RE.search(question) or any(sig in q_lower for sig in _CHART_SIGNALS):
        logger.info("intent fast_path=chart")
        return {"mode": "chart"}

    if any(sig in q_lower for sig in _EMAIL_SIGNALS):
        logger.info("intent fast_path=email")
        return {"mode": "email"}

    if any(sig in q_lower for sig in _CODE_SIGNALS):
        logger.info("intent fast_path=code")
        return {"mode": "code"}

    if any(sig in q_lower for sig in _REPORT_SIGNALS):
        logger.info("intent fast_path=report")
        return {"mode": "report"}

    if any(sig in q_lower for sig in _COMPARISON_SIGNALS):
        logger.info("intent fast_path=comparison")
        return {"mode": "comparison"}

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
