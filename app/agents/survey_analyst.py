"""Survey Analyst agent.

When the user uploads a CSV or Excel file, this agent:
  1. Reads it with pandas and builds a compact column profile.
  2. Sends the profile + a sample to the LLM.
  3. Gets back matplotlib code for each column / relationship.
  4. Runs every chart in an isolated subprocess.
  5. Returns chart_urls + a findings summary for embedding in reports.

Pipeline:  clarity → intent_router → survey_analyst → suggestions
"""

import json
import logging
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import get_llm
from app.state import AgentState
from app.utils.chart_exec import run_chart_code

UPLOADS_DIR = "uploads"
logger = logging.getLogger(__name__)

_DARK = dict(
    bg="#0f0f1a", panel="#1a1a2e", text="#cccccc",
    palette='["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#CCB974","#DA8BC3"]',
)

SURVEY_SYSTEM_PROMPT = f"""You are a survey data analysis expert.

You receive a JSON column profile and a markdown sample of a CSV/Excel survey.
Your job is to write Python/matplotlib code to generate the most insightful charts.

Dark-theme constants provided for you (already defined in the script):
  BG      = "{_DARK['bg']}"
  PANEL   = "{_DARK['panel']}"
  TEXT    = "{_DARK['text']}"
  PALETTE = {_DARK['palette']}

The pandas DataFrame is already loaded as `df`. You must NOT re-read the file.

Chart selection rules:
  - Categorical column (≤ 20 unique values)  → horizontal bar of value_counts()
  - Likert scale (numeric, range 1-5 or 1-7) → horizontal bar with gradient colours
  - Numeric column                            → histogram (bins=20) + optional KDE
  - Two correlated numeric columns            → scatter with regression line
  - Date / time column                        → line chart of counts per period
  - Avoid duplicating the same insight twice

For each chart code block:
  1. Start with: fig, ax = plt.subplots(figsize=(10, 5.5))
  2. Apply: fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
  3. Style ticks, labels, and spine colours with TEXT / "#333355"
  4. Give each chart a descriptive title that names the column
  5. Do NOT include import statements (already present)
  6. Do NOT call plt.savefig() or plt.show()

Return at most 6 charts. Prioritise columns with the most variance or insight."""


class SurveyChart(BaseModel):
    column: str = Field(description="Column or columns this chart visualises")
    chart_type: str = Field(description="e.g. bar, histogram, scatter, line")
    title: str
    description: str = Field(description="One sentence on the insight this reveals")
    code: str = Field(description="Python matplotlib body using `df` variable")


class SurveyPlan(BaseModel):
    charts: list[SurveyChart]
    key_findings: str = Field(
        description="2-3 sentence summary of the most important patterns"
    )
    data_shape: str = Field(description="e.g. '150 responses, 12 questions'")


def _find_survey_file(doc_id: str) -> str | None:
    for ext in (".csv", ".xlsx", ".xls"):
        path = os.path.join(UPLOADS_DIR, f"{doc_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def _build_profile(df) -> dict:
    import pandas as pd

    profile: dict = {
        "shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
        "columns": [],
    }
    for col in df.columns[:25]:  # cap at 25 columns
        entry: dict = {
            "name": col,
            "dtype": str(df[col].dtype),
            "unique_values": int(df[col].nunique()),
            "null_pct": round(df[col].isnull().mean() * 100, 1),
            "sample": df[col].dropna().head(5).tolist(),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            entry["min"]  = round(float(df[col].min()), 3)
            entry["max"]  = round(float(df[col].max()), 3)
            entry["mean"] = round(float(df[col].mean()), 3)
            entry["std"]  = round(float(df[col].std()), 3)
        profile["columns"].append(entry)
    return profile


async def survey_analyst_node(state: AgentState) -> dict:
    import pandas as pd

    doc_id = state.get("doc_id", "")
    query  = state.get("original_query", "Analyse this survey")

    if settings.mock_mode:
        return {
            "findings": "Mock survey analysis complete.",
            "chart_urls": [],
            "chart_url":  "",
            "messages": [AIMessage(content="Mock survey analysis complete.")],
        }

    # ── Locate the file ────────────────────────────────────────────────────
    csv_path = _find_survey_file(doc_id)
    if not csv_path:
        msg = "Survey file not found. Please upload a .csv or .xlsx file first."
        return {"findings": msg, "messages": [AIMessage(content=msg)]}

    # ── Load ───────────────────────────────────────────────────────────────
    try:
        df = (
            pd.read_csv(csv_path)
            if csv_path.endswith(".csv")
            else pd.read_excel(csv_path)
        )
    except Exception as exc:
        logger.error("survey_read_failed: %s", exc)
        return {
            "findings": str(exc),
            "messages": [AIMessage(content=f"Failed to read file: {exc}")],
        }

    profile     = _build_profile(df)
    profile_json = json.dumps(profile, default=str, ensure_ascii=False, indent=2)

    try:
        sample_md = df.head(3).to_markdown(index=False)
    except Exception:
        sample_md = df.head(3).to_string(index=False)

    # ── Plan charts with LLM ──────────────────────────────────────────────
    llm  = get_llm(temperature=0).with_structured_output(SurveyPlan)
    plan: SurveyPlan = await llm.ainvoke([
        SystemMessage(content=SURVEY_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"User request: {query}\n\n"
            f"Data profile:\n{profile_json}\n\n"
            f"Sample rows (first 3):\n{sample_md}"
        )),
    ])

    # ── Inject dark-theme constants + pre-loaded df into each subprocess ──
    abs_path  = os.path.abspath(csv_path)
    read_call = (
        f"pd.read_csv({abs_path!r})"
        if csv_path.endswith(".csv")
        else f"pd.read_excel({abs_path!r})"
    )
    df_setup = (
        f"import pandas as pd\n"
        f"df = {read_call}\n"
        f'BG      = "{_DARK["bg"]}"\n'
        f'PANEL   = "{_DARK["panel"]}"\n'
        f'TEXT    = "{_DARK["text"]}"\n'
        f"PALETTE = {_DARK['palette']}\n"
    )

    # ── Run each chart ─────────────────────────────────────────────────────
    chart_urls:   list[str] = []
    figure_lines: list[str] = []

    for i, chart in enumerate(plan.charts, 1):
        url = run_chart_code(chart.code, extra_setup=df_setup)
        if url:
            chart_urls.append(url)
            figure_lines.append(
                f"**Figure {i}: {chart.title}**\n_{chart.description}_"
            )
            logger.info(
                "survey_chart col=%r type=%s url=%s",
                chart.column, chart.chart_type, url,
            )
        else:
            logger.warning(
                "survey_chart_failed col=%r title=%r", chart.column, chart.title
            )

    findings = (
        f"## Survey Analysis — {plan.data_shape}\n\n"
        f"**Key findings:** {plan.key_findings}"
    )
    if figure_lines:
        findings += "\n\n" + "\n\n".join(figure_lines)

    return {
        "findings":   findings,
        "chart_urls": chart_urls,
        "chart_url":  chart_urls[0] if chart_urls else "",
        "messages":   [AIMessage(content=findings)],
    }
