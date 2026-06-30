"""PDF Generator agent.

Converts the report_writer output (Markdown text in state["messages"][-1])
into a professionally styled PDF using ReportLab. The file is saved to
static/reports/ and a download URL is stored in state.
"""

import html as _html
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


def _sanitize_unicode(text: str) -> str:
    """Replace Unicode chars Helvetica cannot render (would show as ■)."""
    return (
        text
        .replace('‑', '-')    # non-breaking hyphen
        .replace('‒', '-')    # figure dash
        .replace('–', '-')    # en dash
        .replace('—', '--')   # em dash
        .replace('―', '--')   # horizontal bar
        .replace('­', '')     # soft hyphen
        .replace(' ', ' ')    # non-breaking space
        .replace('‘', "'")    # left single quotation mark
        .replace('’', "'")    # right single quotation mark
        .replace('“', '"')    # left double quotation mark
        .replace('”', '"')    # right double quotation mark
        .replace('…', '...')  # horizontal ellipsis
        .replace('•', '-')    # bullet (handled by _md_to_flowables but just in case)
        .replace('■', '-')    # black square (the tofu glyph itself)
    )


def _inline_md(text: str) -> str:
    """Convert **bold** and _italic_ to ReportLab XML tags."""
    text = _sanitize_unicode(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
    text = re.sub(r"_(.+?)_",       r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`",       r"<font face='Courier'>\1</font>", text)
    # Escape bare ampersands that aren't already entity refs
    text = re.sub(r"&(?!(?:amp|lt|gt|apos|quot);)", "&amp;", text)
    return text


def _md_to_flowables(text: str, styles: dict) -> list:
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, Paragraph, Spacer, Table, TableStyle,
    )

    C = colors.HexColor
    _th = ParagraphStyle("th", parent=styles["body"],
                         fontName="Helvetica-Bold", fontSize=9,
                         textColor=colors.white)
    _td = ParagraphStyle("td", parent=styles["body"], fontSize=9)

    flowables = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1

        if not line:
            flowables.append(Spacer(1, 5))
            continue

        # ── Markdown pipe table ────────────────────────────────────────────
        if line.startswith("|"):
            table_lines = [line]
            while i < len(lines) and lines[i].rstrip().startswith("|"):
                table_lines.append(lines[i].rstrip())
                i += 1
            rows = []
            for tl in table_lines:
                cells = [c.strip() for c in tl.strip("|").split("|")]
                non_empty = [c for c in cells if c]
                # Skip GFM separator rows (|---|:---:|---:|)
                if non_empty and all(re.match(r"^:?-+:?$", c) for c in non_empty):
                    continue
                rows.append(cells)
            if rows:
                col_count = max(len(r) for r in rows)
                col_w = (16 * cm) / col_count
                data = []
                for r_idx, row in enumerate(rows):
                    sty = _th if r_idx == 0 else _td
                    cells_p = [Paragraph(_inline_md(c), sty) for c in row]
                    while len(cells_p) < col_count:
                        cells_p.append(Paragraph("", _td))
                    data.append(cells_p)
                t = Table(data, colWidths=[col_w] * col_count, repeatRows=1)
                t.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, 0),  C("#1a1a2e")),
                    ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
                    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
                    ("FONTSIZE",      (0, 0), (-1, -1), 9),
                    ("ROWBACKGROUNDS",(0, 1), (-1, -1),
                     [C("#f5f5f5"), colors.white]),
                    ("GRID",          (0, 0), (-1, -1), 0.4, C("#cccccc")),
                    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING",    (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
                ]))
                flowables.append(Spacer(1, 8))
                flowables.append(t)
                flowables.append(Spacer(1, 8))
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
            flowables.append(
                HRFlowable(width="100%", thickness=0.5,
                           color=C("#cccccc"), spaceBefore=6, spaceAfter=6)
            )
        else:
            flowables.append(Paragraph(_inline_md(line), styles["body"]))

    return flowables


def _build_academic_styles():
    """Academic paper styles — Times-Roman body, clean hierarchy."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle

    C = colors.HexColor

    return {
        "paper_title": ParagraphStyle("paper_title",
            fontName="Times-Bold", fontSize=18, leading=24,
            alignment=TA_CENTER, spaceAfter=10, textColor=C("#111111")),
        "paper_authors": ParagraphStyle("paper_authors",
            fontName="Times-Roman", fontSize=10, leading=14,
            alignment=TA_CENTER, spaceAfter=4, textColor=C("#444444")),
        "abstract_label": ParagraphStyle("abstract_label",
            fontName="Times-Bold", fontSize=9.5, leading=13,
            alignment=TA_CENTER, spaceAfter=4, textColor=C("#111111")),
        "abstract_body": ParagraphStyle("abstract_body",
            fontName="Times-Roman", fontSize=9.5, leading=14,
            alignment=TA_JUSTIFY, leftIndent=36, rightIndent=36,
            spaceAfter=6, textColor=C("#222222")),
        "keywords_line": ParagraphStyle("keywords_line",
            fontName="Times-Roman", fontSize=9, leading=13,
            alignment=TA_CENTER, spaceAfter=12, textColor=C("#555555")),
        "h1": ParagraphStyle("h1",
            fontName="Times-Bold", fontSize=12, leading=16,
            spaceBefore=14, spaceAfter=5, textColor=C("#111111")),
        "h2": ParagraphStyle("h2",
            fontName="Times-Bold", fontSize=10.5, leading=14,
            spaceBefore=10, spaceAfter=4, textColor=C("#222222")),
        "body": ParagraphStyle("body",
            fontName="Times-Roman", fontSize=10, leading=15,
            alignment=TA_JUSTIFY, spaceAfter=5, textColor=C("#222222")),
        "figure_caption": ParagraphStyle("figure_caption",
            fontName="Times-Roman", fontSize=9, leading=13,
            alignment=TA_CENTER, spaceAfter=8, textColor=C("#555555"),
            fontStyle="italic"),
        "ref_entry": ParagraphStyle("ref_entry",
            fontName="Times-Roman", fontSize=9, leading=13,
            leftIndent=18, firstLineIndent=-18, spaceAfter=4,
            textColor=C("#222222")),
        "footer": ParagraphStyle("footer",
            fontName="Times-Roman", fontSize=8,
            alignment=TA_CENTER, textColor=C("#aaaaaa")),
    }


def _academic_md_to_flowables(text: str, styles: dict) -> list:
    """Parse the academic_writer markdown into ReportLab flowables."""
    from reportlab.lib import colors
    from reportlab.platypus import HRFlowable, Paragraph, Spacer

    C = colors.HexColor
    flowables = []
    lines = text.split("\n")
    in_refs = False

    for line in lines:
        line = line.rstrip()

        if not line:
            flowables.append(Spacer(1, 4))
            continue

        # Skip the title (handled separately), abstract, and keywords lines
        if line.startswith("# ") or line.startswith("**Abstract:**") or line.startswith("**Keywords:**"):
            continue

        if line.strip() == "---":
            flowables.append(HRFlowable(width="100%", thickness=0.5,
                                        color=C("#aaaaaa"), spaceBefore=6, spaceAfter=8))
            continue

        # Section headings — treat ### as h2 (subsection), ## as h1 (section)
        if line.startswith("### "):
            text_val = line[4:]
            flowables.append(Paragraph(_inline_md(text_val), styles["h2"]))
            in_refs = False
            continue

        if line.startswith("## "):
            heading = line[3:]
            in_refs = heading.strip().lower() in ("references", "reference list")
            flowables.append(Paragraph(_inline_md(heading), styles["h1"]))
            continue

        # Reference entries [N] ...
        if in_refs and re.match(r"^\[\d+\]", line):
            flowables.append(Paragraph(_inline_md(line), styles["ref_entry"]))
            continue

        # Body text
        if line and not line.startswith("---"):
            flowables.append(Paragraph(_inline_md(line), styles["body"]))

    return flowables


def _generate_academic_pdf(state: dict) -> str:
    """Render an academic-style PDF. Returns the file path."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm, inch
    from reportlab.platypus import (
        HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer,
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)

    report_text  = state.get("report_content") or state["messages"][-1].content
    query        = state.get("original_query", "Research Paper")
    abstract     = state.get("paper_abstract", "")
    keywords     = state.get("paper_keywords", "")

    # Extract title from the first # line of the paper
    title = query
    for line in report_text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break

    filename = f"ares_academic_{uuid.uuid4().hex[:8]}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)

    styles = _build_academic_styles()
    C = colors.HexColor

    doc = SimpleDocTemplate(
        filepath, pagesize=A4,
        leftMargin=2.8*cm, rightMargin=2.8*cm,
        topMargin=3.0*cm,  bottomMargin=3.0*cm,
        title=title,
        author="Ares — Autonomous Research & Evidence System",
    )

    story = []

    # ── Title block ────────────────────────────────────────────────────────
    story.append(Paragraph(_inline_md(title), styles["paper_title"]))
    story.append(Spacer(1, 4))

    ts = datetime.utcnow().strftime("%B %Y")
    story.append(Paragraph(
        f"Ares Research System  ·  Generated {ts}",
        styles["paper_authors"],
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.0,
                            color=C("#111111"), spaceBefore=0, spaceAfter=8))

    # ── Abstract ───────────────────────────────────────────────────────────
    if abstract:
        story.append(Paragraph("ABSTRACT", styles["abstract_label"]))
        story.append(Paragraph(_inline_md(abstract), styles["abstract_body"]))
        story.append(Spacer(1, 4))

    # ── Keywords ───────────────────────────────────────────────────────────
    if keywords:
        story.append(Paragraph(
            f"<b>Keywords:</b> {_inline_md(keywords)}",
            styles["keywords_line"],
        ))

    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=C("#aaaaaa"), spaceBefore=4, spaceAfter=10))

    # ── Paper body (sections 1-5) ──────────────────────────────────────────
    story.extend(_academic_md_to_flowables(report_text, styles))

    # ── Chart as Figure 1 ──────────────────────────────────────────────────
    chart_url = state.get("chart_url", "")
    if chart_url:
        chart_path = chart_url.lstrip("/")
        if os.path.exists(chart_path):
            story.append(Spacer(1, 12))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=C("#cccccc"), spaceBefore=4, spaceAfter=8))
            story.append(Image(chart_path, width=5.8 * inch, height=3.1 * inch))
            story.append(Paragraph(
                f"<i>Figure 1: Data visualisation for \"{_inline_md(query[:60])}\"</i>",
                styles["figure_caption"],
            ))

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=C("#cccccc"), spaceAfter=6))
    story.append(Paragraph(
        "Generated by Ares · Autonomous Research &amp; Evidence System",
        styles["footer"],
    ))

    doc.build(story)
    return filepath


def pdf_generator_node(state: AgentState) -> dict:
    # Academic mode uses a separate renderer with Times-Roman paper styling.
    if state.get("mode") == "academic":
        filepath = _generate_academic_pdf(state)
        pdf_url  = f"/static/reports/{os.path.basename(filepath)}"
        from langchain_core.messages import AIMessage
        return {
            "pdf_url": pdf_url,
            "messages": [AIMessage(content=f"Academic paper PDF ready: {pdf_url}")],
        }

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm, inch
    from reportlab.platypus import (
        HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
        Table, TableStyle,
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # report_content is set explicitly by report_writer so chart_writer's
    # message (which becomes messages[-1]) doesn't shadow the report text.
    report_text  = state.get("report_content") or state["messages"][-1].content
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

    if len(query) > 120:
        truncated = query[:120].rsplit(" ", 1)[0]
        title_text = truncated + "…"
    else:
        title_text = query
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

    # ── Sources & References ────────────────────────────────────────────────────
    sources = state.get("sources") or []
    if sources:
        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=C("#cccccc"), spaceBefore=8, spaceAfter=10))
        story.append(Paragraph("Sources &amp; References", styles["h2"]))
        for idx, src in enumerate(sources, 1):
            title = _sanitize_unicode(src.get("title") or "Untitled")
            url   = src.get("url", "")
            title_esc = _html.escape(title)
            if url:
                url_esc = _html.escape(url, quote=True)
                entry = (f'{idx}.  <link href="{url_esc}" color="#4C72B0">'
                         f'<u>{title_esc}</u></link>')
            else:
                entry = f"{idx}.  {title_esc}"
            story.append(Paragraph(entry, styles["body"]))
        story.append(Spacer(1, 8))

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
