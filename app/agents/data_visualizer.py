"""Data Visualizer agent.

After Tavily research, extracts numerical data from findings and generates
2-3 charts by running LLM-authored Python code in isolated subprocesses.

Pipeline position (replaces chart_writer for research-derived modes):
  research → [writer] → data_visualizer → [pdf_generator | suggestions]
"""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings, truncate_to_budget
from app.llm import get_llm
from app.state import AgentState
from app.utils.chart_exec import run_chart_code

logger = logging.getLogger(__name__)

_DARK = dict(
    bg="#0f0f1a", panel="#1a1a2e", text="#cccccc",
    palette='["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#CCB974","#DA8BC3"]',
)

VIZ_SYSTEM_PROMPT = f"""You are a data visualisation expert.

Given research findings, extract all numerical data and write Python/matplotlib
code to generate 2-3 meaningful, distinct charts.

Dark-theme constants injected for you (use them):
  BG      = "{_DARK['bg']}"
  PANEL   = "{_DARK['panel']}"
  TEXT    = "{_DARK['text']}"
  PALETTE = {_DARK['palette']}

Each chart's code block MUST:
  1. Start with: fig, ax = plt.subplots(figsize=(10, 5.5))
  2. Apply: fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
  3. Set tick/label colours to TEXT
  4. Set a clear title and axis labels
  5. NOT contain import statements (already provided)
  6. NOT contain plt.savefig() or plt.show() (appended automatically)
  7. Contain only real data extracted from the findings — no placeholders

Chart-type guide:
  bar / horizontal_bar — comparing discrete entities or time periods
  line                 — trends over consecutive time points
  pie                  — proportions that sum to 100 %
  scatter              — correlation between two numeric variables

Fewer high-quality charts beat many mediocre ones. If there is not enough
numerical data for a second or third chart, return fewer."""


class ChartCode(BaseModel):
    chart_type: Literal["bar", "line", "pie", "horizontal_bar", "scatter"]
    title: str = Field(description="Descriptive chart title")
    description: str = Field(description="One sentence on what this chart shows")
    code: str = Field(description="Python matplotlib body — no imports, no savefig")


class VizPlan(BaseModel):
    charts: list[ChartCode] = Field(description="1-3 chart specs")
    data_summary: str = Field(
        description="One sentence summarising the numerical data found in findings"
    )


def _mock_charts() -> dict:
    return {
        "chart_urls": [],
        "chart_url": "",
        "messages": [AIMessage(content="(mock) data_visualizer skipped in mock mode.")],
    }


async def data_visualizer_node(state: AgentState) -> dict:
    if settings.mock_mode:
        return _mock_charts()

    # Prefer writer output (report_content) so we visualise the already-
    # synthesised text rather than raw scraped HTML.
    findings = (
        state.get("report_content")
        or state.get("findings")
        or state.get("raw_research")
        or ""
    )
    query = state.get("original_query", "")

    if not findings.strip():
        logger.warning("data_visualizer_no_findings")
        return {
            "chart_urls": [],
            "chart_url": "",
            "messages": [AIMessage(content="No numerical data found to visualise.")],
        }

    findings, _ = truncate_to_budget(findings, label="viz_findings")

    llm = get_llm(temperature=0).with_structured_output(VizPlan)
    plan: VizPlan = await llm.ainvoke([
        SystemMessage(content=VIZ_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Research topic: {query}\n\nFindings:\n{findings}"
        ),
    ])

    # Inject the dark-theme constants into each subprocess.
    setup = (
        f'BG = "{_DARK["bg"]}"\n'
        f'PANEL = "{_DARK["panel"]}"\n'
        f'TEXT = "{_DARK["text"]}"\n'
        f'PALETTE = {_DARK["palette"]}\n'
    )

    chart_urls: list[str] = []
    figure_lines: list[str] = []

    for i, chart in enumerate(plan.charts, 1):
        url = run_chart_code(chart.code, extra_setup=setup)
        if url:
            chart_urls.append(url)
            figure_lines.append(
                f"**Figure {i}: {chart.title}**\n_{chart.description}_"
            )
            logger.info(
                "data_visualizer chart=%d type=%s url=%s", i, chart.chart_type, url
            )
        else:
            logger.warning(
                "data_visualizer chart=%d failed title=%r", i, chart.title
            )

    summary = plan.data_summary
    if figure_lines:
        summary += "\n\n" + "\n\n".join(figure_lines)

    return {
        "chart_urls": chart_urls,
        # Keep chart_url populated for backward-compat (pdf_generator reads it).
        "chart_url": chart_urls[0] if chart_urls else "",
        "messages": [AIMessage(content=summary)],
    }
