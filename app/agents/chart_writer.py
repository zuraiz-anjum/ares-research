"""Chart Writer agent.

Extracts structured data from the user's query and renders it as a
matplotlib chart (bar, line, pie, scatter). The chart is saved as a PNG
to static/charts/ and the URL is stored in state for the frontend.
"""

import base64
import io
import logging
import os
import uuid
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

CHARTS_DIR = "static/charts"


class Dataset(BaseModel):
    label: str = Field(default="", description="Series name (used in legend)")
    values: list[float]


class ChartSpec(BaseModel):
    chart_type: Literal["bar", "line", "pie", "scatter", "horizontal_bar"]
    title: str
    labels: list[str] = Field(description="X-axis / category labels")
    datasets: list[Dataset] = Field(description="One or more data series")
    x_label: str = Field(default="")
    y_label: str = Field(default="")
    unit_note: str = Field(default="", description="e.g. 'in USD Billions'")


CHART_SYSTEM_PROMPT = """You are a data extraction and chart-planning assistant.

Given the user's message, extract any data and produce a chart specification.

Chart type selection:
- comparisons between items   → bar
- trends over time            → line
- part-of-whole proportions   → pie
- large quantity comparisons  → horizontal_bar
- two-variable correlation    → scatter

Rules:
- Strip currency symbols; record the unit in y_label (e.g. "Revenue (USD Billions)")
- Keep labels short (≤ 20 chars each)
- Values must be numbers only (no commas, no symbols)
- If multiple series exist, create multiple Dataset objects
- If the user provides no data at all, make a plausible illustrative example
  and note it in the title with "(Illustrative)"
"""


def _render(spec: ChartSpec) -> str:
    """Render ChartSpec → PNG file in static/charts/. Returns public URL."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(CHARTS_DIR, exist_ok=True)

    PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
               "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#cccccc")
    ax.xaxis.label.set_color("#aaaaaa")
    ax.yaxis.label.set_color("#aaaaaa")
    ax.title.set_color("#ffffff")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    n = len(spec.labels)

    if spec.chart_type == "pie":
        vals = spec.datasets[0].values if spec.datasets else []
        wedges, texts, autotexts = ax.pie(
            vals, labels=spec.labels, autopct="%1.1f%%",
            colors=PALETTE[:len(vals)], startangle=140,
            textprops={"color": "#cccccc"},
        )
        for at in autotexts:
            at.set_color("#ffffff")

    elif spec.chart_type in ("bar", "horizontal_bar"):
        x = np.arange(n)
        count = max(len(spec.datasets), 1)
        width = min(0.7 / count, 0.35)
        for i, ds in enumerate(spec.datasets):
            offset = (i - count / 2 + 0.5) * width
            color = PALETTE[i % len(PALETTE)]
            if spec.chart_type == "bar":
                bars = ax.bar(x + offset, ds.values, width,
                              label=ds.label or None, color=color,
                              edgecolor="#0f0f1a", linewidth=0.5)
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                            f"{h:,.0f}", ha="center", va="bottom",
                            fontsize=8, color="#cccccc")
            else:
                ax.barh(x + offset, ds.values, width,
                        label=ds.label or None, color=color,
                        edgecolor="#0f0f1a", linewidth=0.5)
        if spec.chart_type == "bar":
            ax.set_xticks(x)
            ax.set_xticklabels(spec.labels, rotation=20, ha="right", color="#cccccc")
            ax.yaxis.grid(True, alpha=0.2, color="#444466")
        else:
            ax.set_yticks(x)
            ax.set_yticklabels(spec.labels, color="#cccccc")
            ax.xaxis.grid(True, alpha=0.2, color="#444466")
        if count > 1:
            ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="#cccccc")

    elif spec.chart_type == "line":
        for i, ds in enumerate(spec.datasets):
            ax.plot(spec.labels, ds.values, marker="o", linewidth=2,
                    color=PALETTE[i % len(PALETTE)], label=ds.label or None,
                    markersize=6, markerfacecolor="#0f0f1a")
        ax.set_xticks(range(n))
        ax.set_xticklabels(spec.labels, rotation=20, ha="right", color="#cccccc")
        ax.yaxis.grid(True, alpha=0.2, color="#444466")
        if len(spec.datasets) > 1:
            ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="#cccccc")

    elif spec.chart_type == "scatter":
        for i, ds in enumerate(spec.datasets):
            ax.scatter(range(len(ds.values)), ds.values,
                       color=PALETTE[i % len(PALETTE)], s=80, alpha=0.8,
                       label=ds.label or None)
        ax.set_xticks(range(n))
        ax.set_xticklabels(spec.labels, rotation=20, ha="right", color="#cccccc")
        if len(spec.datasets) > 1:
            ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="#cccccc")

    title = spec.title
    if spec.unit_note:
        title += f"  ({spec.unit_note})"
    ax.set_title(title, fontsize=14, fontweight="bold", color="#ffffff", pad=14)
    if spec.x_label:
        ax.set_xlabel(spec.x_label, color="#aaaaaa")
    if spec.y_label:
        ax.set_ylabel(spec.y_label, color="#aaaaaa")

    plt.tight_layout(pad=1.5)
    filename = f"chart_{uuid.uuid4().hex[:10]}.png"
    filepath = os.path.join(CHARTS_DIR, filename)
    plt.savefig(filepath, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"/static/charts/{filename}"


def chart_writer_node(state: AgentState) -> dict:
    mode = state.get("mode", "chart")

    if mode in ("data_analysis", "comparison", "pdf", "report"):
        # Derive chart data from the previous agent's output (the
        # data_analyst, comparison_matrix, or report_writer content).
        prior_output = next(
            (m.content for m in reversed(state["messages"])
             if hasattr(m, "content") and m.content),
            state.get("original_query", ""),
        )
        prompt_content = (
            f"Original question: {state.get('original_query', '')}\n\n"
            f"Extract the numerical data from this content and produce a chart. "
            f"If multiple entities are compared, use a bar or horizontal_bar chart "
            f"with one bar per entity:\n\n{prior_output}"
        )
    else:
        prompt_content = state.get("original_query") or state["messages"][-1].content

    llm = get_llm(temperature=0).with_structured_output(ChartSpec)
    spec: ChartSpec = llm.invoke([
        SystemMessage(content=CHART_SYSTEM_PROMPT),
        HumanMessage(content=prompt_content),
    ])

    try:
        chart_url = _render(spec)
    except Exception as exc:
        logger.error(f"chart_render_failed: {exc}")
        chart_url = ""

    # Build a human-readable text summary alongside the visual
    lines = [f"**{spec.title}**"]
    if spec.unit_note:
        lines.append(f"_{spec.unit_note}_")
    for ds in spec.datasets:
        pairs = ", ".join(
            f"{lbl}: {val:,.1f}" for lbl, val in zip(spec.labels, ds.values)
        )
        prefix = f"**{ds.label}** — " if ds.label else ""
        lines.append(prefix + pairs)

    summary = "\n\n".join(lines)
    logger.info(f"chart_generated url={chart_url} type={spec.chart_type}")

    return {
        "messages": [AIMessage(content=summary)],
        "chart_url": chart_url,
    }
