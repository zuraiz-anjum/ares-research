
"""Data Visualizer agent.

Generates 2-3 matplotlib charts from research data and saves them as PNGs.

Two modes depending on upstream data_extractor output:

  CSV mode  — extracted_csv_path is set: loads structured DataFrames, gives
              the LLM precise numbers → higher quality charts.
  Text mode — no CSV: LLM extracts numbers from the report text (fallback).

Pipeline position:
  data_extractor → [writer] → draft_critic → data_visualizer → [pdf_generator | suggestions]
"""

import json
import logging
import os
from typing import Literal

import pandas as pd
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
_LIGHT = dict(
    bg="white", panel="#f5f7fa", text="#1a1a1a",
    palette='["#2563eb","#dc2626","#16a34a","#d97706","#7c3aed","#0891b2","#db2777"]',
)

# ── Theme constants injected into every chart subprocess ──────────────────────
_DARK_SETUP = (
    f'BG      = "{_DARK["bg"]}"\n'
    f'PANEL   = "{_DARK["panel"]}"\n'
    f'TEXT    = "{_DARK["text"]}"\n'
    f'PALETTE = {_DARK["palette"]}\n'
)
_LIGHT_SETUP = (
    f'BG      = "{_LIGHT["bg"]}"\n'
    f'PANEL   = "{_LIGHT["panel"]}"\n'
    f'TEXT    = "{_LIGHT["text"]}"\n'
    f'PALETTE = {_LIGHT["palette"]}\n'
)

_CHART_RULES = """\
Each chart code block MUST:
  1. Start with: fig, ax = plt.subplots(figsize=(10, 5.5))
  2. Apply: fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
  3. Set tick/label/title colours to TEXT
  4. Give a clear, descriptive title and labelled axes
  5. NOT contain import statements (already provided)
  6. NOT call plt.savefig() or plt.show() (appended automatically)
  7. Use only real data — no placeholders or example values

Chart-type guide:
  bar / horizontal_bar — comparing discrete entities or time periods
  line                 — trends over consecutive time points
  pie                  — proportions that sum to ~100 % (ONLY when there are 3+ entities)
  scatter              — correlation between two numeric variables

IMPORTANT: Never produce a pie chart with fewer than 3 slices — use a bar chart instead.
Fewer high-quality charts beat many mediocre ones."""

# ── CSV-mode system prompt ─────────────────────────────────────────────────────
CSV_VIZ_PROMPT = f"""You are a data visualisation expert creating publication-quality charts.

You receive structured DataFrames extracted from research findings. Write Python/
matplotlib code to produce 2-3 distinct, insightful charts using these DataFrames.

Theme constants already defined for you: BG, PANEL, TEXT, PALETTE
DataFrames are pre-loaded as variables named after each table (snake_case).

{_CHART_RULES}"""

# ── Text-mode system prompt ────────────────────────────────────────────────────
TEXT_VIZ_PROMPT = f"""You are a data visualisation expert creating publication-quality charts.

Given research findings, extract all numerical data and write Python/matplotlib
code to generate 2-3 meaningful, distinct charts.

Theme constants already defined: BG, PANEL, TEXT, PALETTE

{_CHART_RULES}"""


class ChartCode(BaseModel):
    chart_type: Literal["bar", "line", "pie", "horizontal_bar", "scatter"]
    title: str = Field(description="Descriptive chart title")
    description: str = Field(description="One sentence on what this chart shows")
    code: str = Field(description="Python matplotlib body — no imports, no savefig")


class VizPlan(BaseModel):
    charts: list[ChartCode] = Field(description="1-3 chart specs")
    data_summary: str = Field(description="One sentence summarising the data visualised")


# ── CSV loading helper ─────────────────────────────────────────────────────────

def _load_csv_tables(csv_path: str) -> dict[str, pd.DataFrame]:
    """Parse the multi-table CSV written by data_extractor.

    Returns {snake_case_table_name: DataFrame}.
    """
    tables: dict[str, pd.DataFrame] = {}
    current_name: str | None = None
    current_rows: list[list[str]] = []

    def _flush():
        nonlocal current_name, current_rows
        if current_name and len(current_rows) >= 2:
            header = current_rows[0]
            data   = current_rows[1:]
            df = pd.DataFrame(data, columns=header)
            # Coerce numeric columns (errors='ignore' removed in pandas 2.x)
            for col in df.columns:
                _tmp = pd.to_numeric(df[col], errors="coerce")
                if _tmp.notna().all():
                    df[col] = _tmp
            var_name = (
                current_name.lower()
                .replace(" ", "_")
                .replace("-", "_")
                .replace("/", "_")
                .replace("(", "")
                .replace(")", "")
                .strip("_")
            )
            tables[var_name] = df
        current_name = None
        current_rows = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        import csv as _csv
        reader = _csv.reader(f)
        for row in reader:
            if not row:
                _flush()
                continue
            if row[0].startswith("# Note:"):
                continue
            if row[0].startswith("# "):
                _flush()
                current_name = row[0][2:].strip()
                continue
            if current_name is not None:
                current_rows.append(row)

    _flush()
    return tables


def _build_df_setup(tables: dict[str, pd.DataFrame], theme: str = _DARK_SETUP) -> tuple[str, str]:
    """Return (extra_setup_code, profile_text) for the chart subprocess and LLM prompt.

    Each DataFrame is embedded as inline JSON so the subprocess does not need
    to re-parse the multi-table CSV format.
    """
    setup_lines = [theme, "import pandas as pd, json"]
    profile_parts: list[str] = []

    for var_name, df in tables.items():
        records = df.to_dict(orient="records")
        setup_lines.append(
            f"{var_name} = pd.DataFrame({json.dumps(records, default=str)})"
        )
        for col in df.select_dtypes(include="object").columns:
            setup_lines.append(
                f"_tmp = pd.to_numeric({var_name}[{col!r}], errors='coerce'); "
                f"{var_name}[{col!r}] = _tmp if _tmp.notna().all() else {var_name}[{col!r}]"
            )
        profile_parts.append(
            f"DataFrame '{var_name}':\n"
            f"  Columns: {list(df.columns)}\n"
            f"  Shape: {df.shape[0]} rows x {df.shape[1]} cols\n"
            f"  Sample:\n{df.head(5).to_string(index=False)}\n"
        )

    return "\n".join(setup_lines), "\n".join(profile_parts)


# ── Main node ──────────────────────────────────────────────────────────────────

async def data_visualizer_node(state: AgentState) -> dict:
    if settings.mock_mode:
        return {
            "chart_urls": [],
            "chart_url": "",
            "chart_titles": [],
            "messages": [AIMessage(content="(mock) data_visualizer skipped.")],
        }

    query      = state.get("original_query", "")
    csv_path   = state.get("extracted_csv_path", "") or ""
    has_csv    = bool(csv_path and os.path.exists(csv_path))
    mode       = state.get("mode", "")
    theme      = _LIGHT_SETUP if mode in ("academic", "pdf", "report") else _DARK_SETUP

    if has_csv:
        return await _run_csv_mode(state, query, csv_path, theme)
    else:
        return await _run_text_mode(state, query, theme)


async def _run_csv_mode(state: AgentState, query: str, csv_path: str, theme: str) -> dict:
    """Generate charts from the structured CSV produced by data_extractor."""
    tables = _load_csv_tables(csv_path)
    if not tables:
        logger.warning("data_visualizer csv_empty csv=%s", csv_path)
        return await _run_text_mode(state, query, theme)

    extra_setup, profile_text = _build_df_setup(tables, theme)

    df_list = "\n".join(
        f"  - '{name}': columns {list(df.columns)}" for name, df in tables.items()
    )
    user_prompt = (
        f"Research topic: {query}\n\n"
        f"Available DataFrames:\n{df_list}\n\n"
        f"DataFrame profiles:\n{profile_text}"
    )

    llm  = get_llm(temperature=0).with_structured_output(VizPlan)
    plan: VizPlan = await llm.ainvoke([
        SystemMessage(content=CSV_VIZ_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    return _execute_plan(plan, extra_setup, "csv")


async def _run_text_mode(state: AgentState, query: str, theme: str) -> dict:
    """Fallback: extract numbers from report text and generate charts."""
    findings = (
        state.get("report_content")
        or state.get("findings")
        or state.get("raw_research")
        or ""
    )
    if not findings.strip():
        logger.warning("data_visualizer no_findings")
        return {
            "chart_urls":   [],
            "chart_url":    "",
            "chart_titles": [],
            "messages":     [AIMessage(content="No data found to visualise.")],
        }

    findings, _ = truncate_to_budget(findings, label="viz_findings")

    llm  = get_llm(temperature=0).with_structured_output(VizPlan)
    plan: VizPlan = await llm.ainvoke([
        SystemMessage(content=TEXT_VIZ_PROMPT),
        HumanMessage(content=f"Research topic: {query}\n\nFindings:\n{findings}"),
    ])

    return _execute_plan(plan, theme, "text")


def _execute_plan(plan: VizPlan, extra_setup: str, mode: str) -> dict:
    chart_urls:   list[str] = []
    chart_titles: list[str] = []
    figure_lines: list[str] = []

    for i, chart in enumerate(plan.charts, 1):
        url = run_chart_code(chart.code, extra_setup=extra_setup)
        if url:
            chart_urls.append(url)
            chart_titles.append(chart.title)
            figure_lines.append(
                f"**Figure {i}: {chart.title}**\n_{chart.description}_"
            )
            logger.info(
                "data_visualizer mode=%s chart=%d type=%s url=%s",
                mode, i, chart.chart_type, url,
            )
        else:
            logger.warning(
                "data_visualizer mode=%s chart=%d failed title=%r",
                mode, i, chart.title,
            )

    summary = plan.data_summary
    if figure_lines:
        summary += "\n\n" + "\n\n".join(figure_lines)

    return {
        "chart_urls":   chart_urls,
        "chart_url":    chart_urls[0] if chart_urls else "",
        "chart_titles": chart_titles,
        "messages":     [AIMessage(content=summary)],
    }
