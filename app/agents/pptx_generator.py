"""PPTX export agent.

Converts a markdown report into a PowerPoint presentation using python-pptx.
Each ## heading becomes a new slide; bullet points and paragraphs become
body content. A title slide is prepended automatically.
"""

import os
import re
import uuid
from pathlib import Path


def generate_pptx(title: str, markdown_content: str) -> str:
    """Convert markdown report to PPTX and return the output file path."""
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)

    # Colour palette
    NAVY    = RGBColor(0x08, 0x09, 0x0b)
    BLUE    = RGBColor(0x4a, 0x7c, 0xff)
    WHITE   = RGBColor(0xdd, 0xe1, 0xea)
    DIM     = RGBColor(0x85, 0x8f, 0xa5)

    def _blank_slide():
        blank_layout = prs.slide_layouts[6]  # completely blank
        slide = prs.slides.add_slide(blank_layout)
        # Dark background
        bg = slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = NAVY
        return slide

    def _add_textbox(slide, left, top, width, height, text, size, bold=False, color=WHITE, align=PP_ALIGN.LEFT, wrap=True):
        txb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        txb.word_wrap = wrap
        tf = txb.text_frame
        tf.word_wrap = wrap
        p = tf.paragraphs[0]
        p.alignment = align
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        return txb

    def _accent_bar(slide):
        """Thin blue accent line at top of slide."""
        from pptx.util import Pt as PtU
        line = slide.shapes.add_connector(
            1,  # straight connector
            Inches(0), Inches(0.38),
            Inches(13.33), Inches(0.38),
        )
        line.line.color.rgb = BLUE
        line.line.width = Pt(2)

    # ── Title slide ──────────────────────────────────────────────────────────
    slide = _blank_slide()

    # Logo box
    logo = slide.shapes.add_shape(
        1,  # rectangle
        Inches(0.6), Inches(2.8), Inches(0.55), Inches(0.55),
    )
    logo.fill.solid()
    logo.fill.fore_color.rgb = BLUE
    logo.line.color.rgb = BLUE
    logo_tf = logo.text_frame
    logo_tf.paragraphs[0].runs  # ensure paragraph exists
    logo_p = logo_tf.paragraphs[0]
    logo_p.alignment = PP_ALIGN.CENTER
    logo_run = logo_p.add_run()
    logo_run.text = "⚡"
    logo_run.font.size = Pt(18)

    _add_textbox(slide, 1.3, 2.7, 11, 0.7, "Ares", 32, bold=True, color=WHITE)
    _add_textbox(slide, 1.3, 3.3, 11, 0.5, "Autonomous Research & Evidence System", 14, color=DIM)
    _add_textbox(slide, 0.6, 4.1, 12, 1.0, title, 22, bold=True, color=BLUE)

    # ── Parse markdown into sections ─────────────────────────────────────────
    sections = []
    current_heading = None
    current_lines: list[str] = []

    for raw_line in markdown_content.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, current_lines))
            current_heading = line[3:].strip()
            current_lines = []
        elif line.startswith("# "):
            # Top-level heading becomes a section divider
            if current_heading is not None:
                sections.append((current_heading, current_lines))
            current_heading = line[2:].strip()
            current_lines = []
        else:
            if current_heading is not None or line:
                current_lines.append(line)

    if current_heading is not None:
        sections.append((current_heading, current_lines))

    # If no headings found, treat whole content as one section
    if not sections:
        sections = [("Summary", markdown_content.splitlines())]

    # ── Content slides ────────────────────────────────────────────────────────
    for heading, lines in sections:
        slide = _blank_slide()

        # Accent bar
        try:
            _accent_bar(slide)
        except Exception:
            pass

        # Slide heading
        _add_textbox(slide, 0.6, 0.5, 12, 0.65, heading, 24, bold=True, color=WHITE)

        # Body — combine lines, strip markdown noise
        body_lines = []
        for line in lines:
            if not line:
                continue
            # Convert bullet markers
            line = re.sub(r"^\*\*(.+?)\*\*", r"\1", line)  # bold
            line = re.sub(r"^\*(.+?)\*", r"\1", line)       # italic
            line = re.sub(r"`(.+?)`", r"\1", line)           # inline code
            line = re.sub(r"\[(\d+)\]", "", line)            # citation marks
            line = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", line)  # markdown links
            if line.startswith("- ") or line.startswith("* "):
                body_lines.append("  • " + line[2:])
            elif re.match(r"^\d+\.", line):
                body_lines.append("  " + line)
            else:
                body_lines.append(line)

        body_text = "\n".join(body_lines[:18])  # cap at 18 lines per slide

        if body_text:
            _add_textbox(slide, 0.6, 1.3, 12.1, 5.8, body_text, 14, color=DIM)

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs("static/reports", exist_ok=True)
    out_path = f"static/reports/report_{uuid.uuid4().hex[:10]}.pptx"
    prs.save(out_path)
    return out_path
