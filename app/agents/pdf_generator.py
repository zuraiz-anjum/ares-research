"""PDF Generator agent.

Converts the report_writer output (Markdown text in state["messages"][-1])
into a professionally styled PDF using ReportLab. The file is saved to
static/reports/ and a download URL is stored in state.
"""

import logging
import os
import re
import uuid
from datetime import datetime

from app.state import AgentState

logger = logging.getLogger(__name__)

REPORTS_DIR = "static/reports"

# Brand colours
_NAVY  = "#1a1a2e"
_BLUE  = "#4C72B0"
_GREY  = "#555555"
_LIGHT = "#888888"
_TEXT  = "#222222"


def _build_styles():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    base = getSampleStyleSheet()
    C = colors.HexColor

    return {
        "cover_tag": ParagraphStyle("cover_tag", parent=base["Normal"],
            fontSize=9, textColor=C(_LIGHT), alignment=TA_CENTER, spaceAfter=4),
        "cover_title": ParagraphStyle("cover_title", parent=base["Normal"],
            fontSize=26, fontName="Helvetica-Bold", textColor=C(_NAVY),
            leading=32, spaceAfter=8, alignment=TA_LEFT),
        "cover_sub": ParagraphStyle("cover_sub", parent=base["Normal"],
            fontSize=11, textColor=C(_GREY), spaceAfter=20, alignment=TA_LEFT),
        "h1": ParagraphStyle("h1", parent=base["Heading1"],
            fontSize=18, fontName="Helvetica-Bold", textColor=C(_NAVY),
            spaceBefore=18, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=base["Heading2"],
            fontSize=14, fontName="Helvetica-Bold", textColor=C(_BLUE),
            spaceBefore=14, spaceAfter=6),
        "h3": ParagraphStyle("h3", parent=base["Heading3"],
            fontSize=12, fontName="Helvetica-Bold", textColor=C(_GREY),
            spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"],
            fontSize=10.5, leading=17, spaceAfter=6, textColor=C(_TEXT),
            alignment=TA_JUSTIFY),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"],
            fontSize=10.5, leading=17, spaceAfter=4, leftIndent=16,
            textColor=C(_TEXT), bulletIndent=0),
        "caption": ParagraphStyle("caption", parent=base["Normal"],
            fontSize=9, textColor=C(_LIGHT), spaceAfter=4, leftIndent=16),
        "footer": ParagraphStyle("footer", parent=base["Normal"],
            fontSize=8, textColor=C(_LIGHT), alignment=TA_CENTER),
        "fact_verified": ParagraphStyle("fact_verified", parent=base["Normal"],
            fontSize=10, leading=16, textColor=colors.HexColor("#2d6a2d"), spaceAfter=3),
        "fact_uncertain": ParagraphStyle("fact_uncertain", parent=base["Normal"],
            fontSize=10, leading=16, textColor=colors.HexColor("#7a5c00"), spaceAfter=3),
        "fact_contradicted": ParagraphStyle("fact_contradicted", parent=base["Normal"],
            fontSize=10, leading=16, textColor=colors.HexColor("#8b1a1a"), spaceAfter=3),
    }


def _inline_md(text: str) -> str:
    """Convert **bold** and _italic_ to ReportLab XML tags."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
    text = re.sub(r"_(.+?)_",       r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`",       r"<font face='Courier'>\1</font>", text)
    # Escape bare ampersands that aren't already entity refs
    text = re.sub(r"&(?!(?:amp|lt|gt|apos|quot);)", "&amp;", text)
    return text


def _md_to_flowables(text: str, styles: dict) -> list:
    from reportlab.platypus import Paragraph, Spacer

    flowables = []
    for raw in text.split("\n"):
        line = raw.rstrip()

        if not line:
            flowables.append(Spacer(1, 5))
            continue

        if line.startswith("### "):
            flowables.append(Paragraph(_inline_md(line[4:]), styles["h3"]))
        elif line.startswith("## "):
            flowables.append(Paragraph(_inline_md(line[3:]), styles["h2"]))
        elif line.startswith("# "):
            flowables.append(Paragraph(_inline_md(line[2:]), styles["h1"]))
        elif re.match(r"^[-*] ", line):
            flowables.append(Paragraph("• " + _inline_md(line[2:]), styles["bullet"]))
        elif re.match(r"^\d+\. ", line):
            m = re.match(r"^(\d+)\. (.*)", line)
            num, rest = m.group(1), m.group(2)
            flowables.append(
                Paragraph(f"{num}. {_inline_md(rest)}", styles["bullet"])
            )
        elif re.match(r"^-{3,}$|^\*{3,}$", line):
            from reportlab.platypus import HRFlowable
            from reportlab.lib import colors
            flowables.append(
                HRFlowable(width="100%", thickness=0.5,
                           color=colors.HexColor("#cccccc"),
                           spaceBefore=6, spaceAfter=6)
            )
        else:
            flowables.append(Paragraph(_inline_md(line), styles["body"]))

    return flowables


def pdf_generator_node(state: AgentState) -> dict:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm, inch
    from reportlab.platypus import (
        HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
        Table, TableStyle,
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)

    report_text  = state["messages"][-1].content
    query        = state.get("original_query", "Research Report")
    critique     = state.get("critique", "")
    fact_checks  = state.get("fact_check_results", []) or []
    plan_steps   = state.get("plan_steps", []) or []

    filename = f"ares_report_{uuid.uuid4().hex[:8]}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)

    styles = _build_styles()
    C = colors.HexColor

    doc = SimpleDocTemplate(
        filepath, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm,  bottomMargin=2.5*cm,
        title=query[:80],
        author="Ares — Autonomous Research & Evidence System",
    )

    story = []

    # ── Cover block ────────────────────────────────────────────────────────────
    story.append(Paragraph("ARES  ·  AUTONOMOUS RESEARCH &amp; EVIDENCE SYSTEM",
                           styles["cover_tag"]))
    story.append(Spacer(1, 10))

    title_text = (query[:90] + "…") if len(query) > 90 else query
    story.append(Paragraph(title_text, styles["cover_title"]))

    ts = datetime.utcnow().strftime("%B %d, %Y")
    story.append(Paragraph(f"Generated {ts}  ·  Multi-Agent AI Research",
                           styles["cover_sub"]))
    story.append(HRFlowable(width="100%", thickness=2,
                            color=C(_BLUE), spaceBefore=4, spaceAfter=20))

    # ── Research plan (if any) ─────────────────────────────────────────────────
    if plan_steps:
        story.append(Paragraph("Research Plan", styles["h2"]))
        for i, step in enumerate(plan_steps, 1):
            story.append(Paragraph(f"{i}. {_inline_md(step)}", styles["bullet"]))
        story.append(Spacer(1, 12))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=C("#cccccc"), spaceAfter=12))

    # ── Main report body ───────────────────────────────────────────────────────
    story.extend(_md_to_flowables(report_text, styles))

    # ── Data visualization (chart generated by chart_writer before this node) ──
    chart_url = state.get("chart_url", "")
    if chart_url:
        chart_path = chart_url.lstrip("/")          # "static/charts/chart_xxx.png"
        if os.path.exists(chart_path):
            story.append(Spacer(1, 16))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=C("#cccccc"), spaceBefore=8, spaceAfter=10))
            story.append(Paragraph("Data Visualization", styles["h2"]))
            # A4 usable width ≈ 21 cm − 5 cm margins = 16 cm ≈ 6.3 inch
            story.append(Image(chart_path, width=6.3 * inch, height=3.4 * inch))
            story.append(Spacer(1, 8))

    # ── Fact-check table ───────────────────────────────────────────────────────
    if fact_checks:
        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=C("#cccccc"), spaceBefore=8, spaceAfter=10))
        story.append(Paragraph("Fact-Check Summary", styles["h2"]))

        icons    = {"verified": "✓", "uncertain": "?", "contradicted": "✗"}
        st_style = {
            "verified":     styles["fact_verified"],
            "uncertain":    styles["fact_uncertain"],
            "contradicted": styles["fact_contradicted"],
        }
        for fc in fact_checks:
            status = fc.get("status", "uncertain")
            claim  = fc.get("claim", "")
            note   = fc.get("note", "")
            icon   = icons.get(status, "?")
            sty    = st_style.get(status, styles["body"])
            story.append(Paragraph(f"<b>{icon}  {_inline_md(claim)}</b>", sty))
            if note:
                story.append(Paragraph(_inline_md(note), styles["caption"]))
        story.append(Spacer(1, 8))

    # ── Critical review ────────────────────────────────────────────────────────
    if critique:
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=C("#cccccc"), spaceBefore=8, spaceAfter=10))
        story.append(Paragraph("Critical Review", styles["h2"]))
        story.extend(_md_to_flowables(critique, styles))

    # ── Footer ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=C("#cccccc"), spaceAfter=8))
    story.append(Paragraph(
        "Generated by Ares · Autonomous Research &amp; Evidence System · Powered by LangGraph",
        styles["footer"],
    ))

    doc.build(story)
    download_url = f"/static/reports/{filename}"
    logger.info(f"pdf_generated path={filepath}")
    return {"pdf_url": download_url}
