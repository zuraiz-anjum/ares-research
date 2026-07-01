"""Data Extractor agent.

Sits between research and the writer agents. Examines Tavily findings,
classifies whether meaningful numerical/quantitative data exists, extracts
it into a clean structured CSV, and forwards that CSV path to data_visualizer
via state.

Pipeline position:
  research → data_extractor → [academic_writer | report_writer | data_analyst | comparison_matrix]
"""

import csv
import io
import json
import logging
import os
import re
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings, truncate_to_budget
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)
CSV_DIR = "static/charts"

EXTRACTOR_SYSTEM_PROMPT = """You are a quantitative data extraction specialist.

Given research findings, determine whether meaningful numerical data exists and
extract it into clean structured tables that can be charted.

WHAT COUNTS AS QUANTITATIVE DATA (extract these):
- Financial figures: revenue, funding raised, valuation, market cap, profit/loss
- Growth metrics: percentage change, YoY growth, CAGR
- User/customer counts, market share percentages
- Time-series: the same metric measured across multiple years or quarters
- Cross-entity comparisons: the same metric for multiple companies or products
- Rankings, scores, or ratings with numeric values

SKIP if data is only:
- Pure descriptive prose with no meaningful numbers
- A single isolated number with no comparison or trend context
- Numbers that are only dates (years, timestamps)

OUTPUT — respond with ONLY valid JSON, no markdown fences:
{
  "has_quantitative_data": true | false,
  "data_tables": [
    {
      "table_name": "Short descriptive name for what this dataset shows",
      "columns": ["Entity", "Metric ($B)", "Year"],
      "rows": [
        ["OpenAI", 57.9, 2025],
        ["Anthropic", 13.0, 2024]
      ],
      "notes": "optional: units, period, data source note"
    }
  ],
  "data_summary": "One sentence describing what quantitative data was found, or 'No significant quantitative data found' if none."
}

Rules:
- Extract ALL numeric values with their full context (entity name, time period, unit)
- Standardise units within each table — use the same scale for every row
- Put units in the column header, NOT in the cell values (e.g. column = "Funding ($B)", cell = 57.9)
- Numeric cells must be numbers only: no $, no commas, no % signs inside the cell
- Each table represents ONE coherent dataset (e.g. "Company Funding Comparison 2020-2025")
- Maximum 5 tables; return fewer if the data doesn't support more
- Column names must be under 25 characters"""


class DataTable(BaseModel):
    table_name: str
    columns: list[str]
    rows: list[list]
    notes: str = ""


class ExtractionResult(BaseModel):
    has_quantitative_data: bool
    data_tables: list[DataTable] = Field(default_factory=list)
    data_summary: str


def _parse_extraction(text: str) -> ExtractionResult:
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    start = text.find("{")
    if start == -1:
        return ExtractionResult(has_quantitative_data=False,
                                data_summary="Could not parse extraction result.")
    try:
        data = json.loads(text[start:])
        tables = [DataTable(**t) for t in data.get("data_tables", [])]
        return ExtractionResult(
            has_quantitative_data=bool(data.get("has_quantitative_data", False)),
            data_tables=tables,
            data_summary=str(data.get("data_summary", "")),
        )
    except Exception as exc:
        logger.warning("data_extractor parse_failed: %s", exc)
        return ExtractionResult(has_quantitative_data=False,
                                data_summary="Extraction parsing failed.")


def _tables_to_csv(tables: list[DataTable]) -> str:
    """Serialise multiple DataTable objects into a single CSV.

    Each table is preceded by a comment row '# <table_name>' so that
    data_visualizer can split them back into separate DataFrames.
    """
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for table in tables:
        writer.writerow([f"# {table.table_name}"])
        if table.notes:
            writer.writerow([f"# Note: {table.notes}"])
        writer.writerow(table.columns)
        for row in table.rows:
            writer.writerow([str(v) for v in row])
        writer.writerow([])
    return out.getvalue()


async def data_extractor_node(state: AgentState) -> dict:
    if settings.mock_mode:
        return {
            "extracted_csv_path": "",
            "data_is_quantitative": False,
            "messages": [AIMessage(content="(mock) data_extractor skipped.")],
        }

    findings = state.get("findings") or state.get("raw_research") or ""
    query    = state.get("original_query", "")

    if not findings.strip():
        return {
            "extracted_csv_path": "",
            "data_is_quantitative": False,
            "messages": [AIMessage(content="No research findings to extract data from.")],
        }

    findings_budget, _ = truncate_to_budget(findings, label="data_extractor")

    llm      = get_llm(temperature=0)
    response = await llm.ainvoke([
        SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Research topic: {query}\n\nFindings:\n{findings_budget}"),
    ])

    result = _parse_extraction(response.content)

    if not result.has_quantitative_data or not result.data_tables:
        logger.info("data_extractor no_quantitative_data query=%r", query[:80])
        return {
            "extracted_csv_path": "",
            "data_is_quantitative": False,
            "messages": [AIMessage(content=f"Data classifier: {result.data_summary}")],
        }

    os.makedirs(CSV_DIR, exist_ok=True)
    csv_filename = f"data_{uuid.uuid4().hex[:10]}.csv"
    csv_path     = os.path.join(CSV_DIR, csv_filename)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write(_tables_to_csv(result.data_tables))

    logger.info(
        "data_extractor tables=%d csv=%s summary=%r",
        len(result.data_tables), csv_path, result.data_summary[:100],
    )

    return {
        "extracted_csv_path": csv_path,
        "data_is_quantitative": True,
        "messages": [
            AIMessage(content=(
                f"Quantitative data extracted: {result.data_summary} "
                f"({len(result.data_tables)} table(s))"
            ))
        ],
    }
