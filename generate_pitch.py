"""Generate the Ares pitch/presentation PDF."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import PageBreak

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY      = colors.HexColor("#0d1b2a")
ACCENT    = colors.HexColor("#4a7cff")
ACCENT2   = colors.HexColor("#7c4fff")
LIGHT_BG  = colors.HexColor("#f0f4ff")
MID_BG    = colors.HexColor("#e8eeff")
ROW_ALT   = colors.HexColor("#f7f9ff")
WHITE     = colors.white
DARK_TEXT = colors.HexColor("#0d1b2a")
DIM_TEXT  = colors.HexColor("#4a5568")
GREEN     = colors.HexColor("#22c55e")
RED       = colors.HexColor("#ef4444")

W, H = A4

def build_styles():
    base = getSampleStyleSheet()

    def s(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "cover_title": s("cover_title",
            fontName="Helvetica-Bold", fontSize=36, textColor=WHITE,
            leading=44, alignment=TA_CENTER, spaceAfter=10),
        "cover_sub": s("cover_sub",
            fontName="Helvetica", fontSize=14, textColor=colors.HexColor("#a0b4ff"),
            leading=20, alignment=TA_CENTER, spaceAfter=6),
        "cover_tag": s("cover_tag",
            fontName="Helvetica-Bold", fontSize=11, textColor=colors.HexColor("#c0d0ff"),
            alignment=TA_CENTER, spaceAfter=4),
        "section_h": s("section_h",
            fontName="Helvetica-Bold", fontSize=16, textColor=WHITE,
            leading=20, spaceAfter=0, spaceBefore=0),
        "h2": s("h2",
            fontName="Helvetica-Bold", fontSize=13, textColor=DARK_TEXT,
            leading=18, spaceAfter=4, spaceBefore=10),
        "h3": s("h3",
            fontName="Helvetica-Bold", fontSize=11, textColor=ACCENT,
            leading=15, spaceAfter=3, spaceBefore=8),
        "body": s("body",
            fontName="Helvetica", fontSize=9.5, textColor=DARK_TEXT,
            leading=15, spaceAfter=4, alignment=TA_JUSTIFY),
        "body_small": s("body_small",
            fontName="Helvetica", fontSize=8.5, textColor=DIM_TEXT,
            leading=13, spaceAfter=3),
        "bullet": s("bullet",
            fontName="Helvetica", fontSize=9.5, textColor=DARK_TEXT,
            leading=15, spaceAfter=3, leftIndent=14, bulletIndent=2),
        "mono": s("mono",
            fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#334155"),
            leading=12, spaceAfter=2, leftIndent=8),
        "quote": s("quote",
            fontName="Helvetica-BoldOblique", fontSize=11, textColor=ACCENT,
            leading=17, spaceAfter=6, leftIndent=12, alignment=TA_CENTER),
        "caption": s("caption",
            fontName="Helvetica-Oblique", fontSize=8, textColor=DIM_TEXT,
            alignment=TA_CENTER, spaceAfter=4),
        "tag": s("tag",
            fontName="Helvetica-Bold", fontSize=8, textColor=ACCENT,
            leading=12),
        "timing": s("timing",
            fontName="Helvetica-Bold", fontSize=9.5, textColor=ACCENT,
            leading=14, spaceAfter=2),
    }


def section_header(title, st):
    """Dark navy banner acting as a section divider."""
    tbl = Table([[Paragraph(title, st["section_h"])]], colWidths=[W - 4*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), NAVY),
        ("TOPPADDING",  (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 14),
        ("RIGHTPADDING",(0,0), (-1,-1), 14),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return [Spacer(1, 8), tbl, Spacer(1, 8)]


def hr(color=ACCENT, thickness=0.8):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=2)


def bullet_list(items, st, symbol="•"):
    return [Paragraph(f"{symbol}  {item}", st["bullet"]) for item in items]


def simple_table(headers, rows, col_widths, st, header_bg=NAVY, header_fg=WHITE):
    data = [[Paragraph(f"<b>{h}</b>", ParagraphStyle("th",
                fontName="Helvetica-Bold", fontSize=9, textColor=header_fg,
                leading=13)) for h in headers]]
    for i, row in enumerate(rows):
        data.append([
            Paragraph(str(c), ParagraphStyle("td",
                fontName="Helvetica", fontSize=8.8, textColor=DARK_TEXT, leading=13))
            for c in row
        ])
    style = TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), header_bg),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, ROW_ALT]),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#d0d8f0")),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ])
    return Table(data, colWidths=col_widths, style=style, repeatRows=1)


def check(val):
    """Return coloured tick or cross paragraph."""
    if val in ("✓", True, "Yes"):
        return Paragraph("<b>✓</b>", ParagraphStyle("ck",
            fontName="Helvetica-Bold", fontSize=10, textColor=GREEN,
            alignment=TA_CENTER, leading=13))
    elif val in ("✗", False, "No"):
        return Paragraph("<b>✗</b>", ParagraphStyle("cx",
            fontName="Helvetica-Bold", fontSize=10, textColor=RED,
            alignment=TA_CENTER, leading=13))
    return Paragraph(str(val), ParagraphStyle("cv",
        fontName="Helvetica", fontSize=8.8, textColor=DARK_TEXT,
        alignment=TA_CENTER, leading=13))


def build_cover(st):
    """Full-bleed dark cover simulation with a large table cell."""
    cover_data = [[
        Paragraph("ARES", st["cover_title"]),
    ]]
    cover = Table(cover_data, colWidths=[W - 4*cm])
    cover.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), NAVY),
        ("TOPPADDING",   (0,0),(-1,-1), 48),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("ALIGN",        (0,0),(-1,-1), "CENTER"),
    ]))

    subtitle_data = [[Paragraph("Autonomous Research &amp; Evidence System", st["cover_sub"])]]
    subtitle = Table(subtitle_data, colWidths=[W - 4*cm])
    subtitle.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), NAVY),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
    ]))

    tagline_data = [[Paragraph(
        "Multi-Agent AI · 20-Node LangGraph Pipeline · 12 Output Modes · RAG + Reranking · Live PDF Export",
        st["cover_tag"]
    )]]
    tagline = Table(tagline_data, colWidths=[W - 4*cm])
    tagline.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), NAVY),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 40),
    ]))

    divider_data = [[Paragraph("Product Pitch &amp; Technical Overview", ParagraphStyle("div",
        fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#8090c0"),
        alignment=TA_CENTER, leading=14))]]
    divider = Table(divider_data, colWidths=[W - 4*cm])
    divider.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), colors.HexColor("#0a1520")),
        ("TOPPADDING",   (0,0),(-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
    ]))

    return [cover, subtitle, tagline, divider, Spacer(1, 18)]


def build_hook(st):
    elems = section_header("01 · THE HOOK", st)
    elems += [
        Spacer(1, 6),
        Paragraph(
            '"Every professional spends hours doing the same thing: opening tabs, reading articles, '
            'cross-checking data, writing summaries, formatting reports. '
            '<b>Ares eliminates that.</b> You type one sentence. '
            'You get a fact-checked, sourced, exportable research report in under 60 seconds."',
            st["quote"]
        ),
        Spacer(1, 6),
        Paragraph(
            "Ares is not a chatbot wrapper. It is a <b>decision-making pipeline</b> where specialised "
            "AI agents collaborate and hand off work to each other — exactly like a real analyst team — "
            "except it runs in seconds, costs pennies, and never sleeps.",
            st["body"]
        ),
        Spacer(1, 4),
    ]
    return elems


def build_what(st):
    elems = section_header("02 · WHAT IS ARES", st)
    elems += [
        Paragraph("Core Definition", st["h2"]),
        Paragraph(
            "Ares is a <b>multi-agent AI research platform</b> built on a 20-node LangGraph stateful "
            "graph. It takes a plain-English query and autonomously handles the entire research, "
            "validation, writing, and export workflow without any human intervention.",
            st["body"]
        ),
        Spacer(1, 6),
        Paragraph("What it produces:", st["h3"]),
    ]
    elems += bullet_list([
        "Sourced research answers with fact-checking and critical perspective",
        "Side-by-side comparison matrices across companies, products, or technologies",
        "Multi-section structured reports and downloadable branded PDFs",
        "Matplotlib data visualisation charts embedded inline or exported",
        "Professional emails written from live research data",
        "Code generation, debate analysis, strategic plans",
        "Document Q&amp;A from uploaded PDFs, Word files, or text — with RAG + reranking",
    ], st)
    return elems


def build_workflow(st):
    elems = section_header("03 · HOW IT WORKS — THE PIPELINE", st)

    # Pipeline flow as a visual table
    nodes = [
        ("Clarity Agent",    "Validates the query — pauses and asks if ambiguous"),
        ("Intent Router",    "Classifies into 1 of 12 pipeline modes using LLM + fast-path rules"),
        ("Planner",          "Creates a visible multi-step research plan for complex tasks"),
        ("Decomposer",       "Breaks the query into parallel sub-questions"),
        ("Research Agent",   "Fires Tavily web searches in parallel for each sub-query"),
        ("Validator",        "Scores source quality and confidence — retries if weak"),
        ("Fact Checker",     "Cross-checks key claims against a second search pass"),
        ("Critic Agent",     "Peer-reviews the draft answer before it reaches the user"),
        ("Synthesis Agent",  "Writes the final answer using ONLY validated research findings"),
        ("Chart Writer",     "Renders a Matplotlib chart from any numerical data found"),
        ("Report / PDF",     "Formats as structured report and exports to branded PDF"),
        ("Suggestions",      "Generates 3 intelligent follow-up questions automatically"),
    ]

    data = [[
        Paragraph("<b>#</b>", ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=8.5, textColor=WHITE, leading=12)),
        Paragraph("<b>Agent / Node</b>", ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=8.5, textColor=WHITE, leading=12)),
        Paragraph("<b>Responsibility</b>", ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=8.5, textColor=WHITE, leading=12)),
    ]]
    for i, (name, desc) in enumerate(nodes, 1):
        data.append([
            Paragraph(str(i), ParagraphStyle("n", fontName="Helvetica-Bold", fontSize=8.5,
                textColor=ACCENT, alignment=TA_CENTER, leading=12)),
            Paragraph(f"<b>{name}</b>", ParagraphStyle("nm", fontName="Helvetica-Bold",
                fontSize=8.5, textColor=DARK_TEXT, leading=12)),
            Paragraph(desc, ParagraphStyle("ds", fontName="Helvetica", fontSize=8.5,
                textColor=DIM_TEXT, leading=12)),
        ])

    tbl = Table(data, colWidths=[1.0*cm, 4.5*cm, 10.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), NAVY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, ROW_ALT]),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#d0d8f0")),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 7),
        ("RIGHTPADDING",  (0,0),(-1,-1), 7),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    elems.append(tbl)

    elems += [
        Spacer(1, 10),
        Paragraph("Live Streaming Architecture", st["h3"]),
        Paragraph(
            "Every node activation is streamed to the frontend in real time via <b>Server-Sent Events (SSE)</b>. "
            "The user watches a live Agent Workspace panel: each agent lights up as it starts, shows elapsed "
            "time when it completes, and the answer text streams token-by-token as the model writes it. "
            "There are <b>12 typed SSE event types</b>: node_start, mode, token, chart, pdf_ready, "
            "sources, decomposed, critique, fact_check, plan_revealed, suggestions, and complete.",
            st["body"]
        ),
    ]
    return elems


def build_modes(st):
    elems = section_header("04 · 12 PIPELINE MODES", st)
    elems.append(Paragraph(
        "The Intent Router classifies every query into one of 12 specialised pipelines. "
        "Each mode activates a different subset of the 20 agents.",
        st["body"]
    ))
    elems.append(Spacer(1, 6))

    rows = [
        ("Research",       '"Tell me about Anthropic"',                  "Sourced conversational answer + fact-check + critique"),
        ("Comparison",     '"Stripe vs Braintree vs Adyen"',             "Side-by-side matrix, chart, winner verdict"),
        ("Report",         '"Write a report on OpenAI"',                 "Structured multi-section report"),
        ("PDF",            '"PDF report on Tesla"',                      "Full research pipeline → branded downloadable PDF"),
        ("Data Analysis",  '"Show me Apple revenue numbers"',            "Tables + key metrics + auto-chart"),
        ("Chart",          '"Bar chart: Apple, Google, Meta revenue"',   "Matplotlib visualisation from raw data"),
        ("Debate",         '"Pros and cons of AI regulation"',           "FOR / AGAINST structured analysis + verdict"),
        ("Plan",           '"How do I launch a SaaS startup?"',          "Step-by-step research plan with visible stages"),
        ("Code",           '"Write a FastAPI auth endpoint"',            "Working code, no web search needed"),
        ("Email",          '"Draft an email about Stripe funding"',      "Live research → professional ready-to-send email"),
        ("Document",       "Upload PDF → ask questions about it",        "RAG + cross-encoder reranking → doc Q&A"),
        ("Chat",           '"What is a B2B SaaS business model?"',       "General knowledge answer from model"),
    ]
    headers = ["Mode", "Example Query", "Output"]
    cw = [3.0*cm, 5.5*cm, 7.5*cm]
    elems.append(simple_table(headers, rows, cw, st))
    return elems


def build_capabilities(st):
    elems = section_header("05 · KEY TECHNICAL CAPABILITIES", st)

    caps = [
        ("RAG — Document Intelligence",
         [
             "Upload PDFs, Word docs, or plain text files via the paperclip button",
             "Chunks documents using RecursiveCharacterTextSplitter (1000 chars, 150 overlap)",
             "Embeds into ChromaDB persistent vector store with cosine similarity index",
             "Retrieves top-20 candidates by embedding similarity",
             "Re-ranks with FlashRank cross-encoder (ms-marco-MiniLM-L-12-v2) — returns top 6",
             "Document context committed silently to the thread — follow-ups remember the file automatically",
             "Supports PDF, DOCX, TXT, MD — up to 10 MB",
         ]),
        ("Multi-Provider LLM Fallback",
         [
             "Primary: Groq (ultra-fast inference)",
             "Fallback 1: Cerebras",
             "Fallback 2: Google Gemini",
             "Fallback 3: OpenRouter",
             "Automatic failover on rate-limit or error — zero user impact",
         ]),
        ("Persistent Conversation Checkpointing",
         [
             "Built on AsyncSqliteSaver (LangGraph checkpoint system)",
             "Full graph state (messages, sub-queries, findings, mode, doc_id) persists to SQLite",
             "Conversations survive server restarts — users can continue threads days later",
             "Per-thread isolation — each conversation has its own checkpoint",
         ]),
        ("Human-in-the-Loop Clarification",
         [
             "Ambiguous queries trigger a LangGraph graph interrupt",
             "Pipeline pauses, asks the user one targeted clarifying question",
             "Resumes from the exact checkpoint state once the user replies",
             "No work is lost or repeated during the pause",
         ]),
        ("PDF Export with Full Formatting",
         [
             "ReportLab generates branded PDFs: title, metadata, sections, footer",
             "GFM pipe tables rendered as styled ReportLab tables with alternating row colours",
             "Charts embedded inline as images from Matplotlib",
             'Sources and References section with clickable hyperlinks',
             "Unicode sanitisation — no rendering artefacts (■) from LLM output",
         ]),
        ("Rate Limiting & Production Safety",
         [
             "SlowAPI rate limiter: 20 requests/minute on chat, 10/minute on uploads",
             "Token budget enforcement — rejects oversized inputs before graph runs",
             "File type and size validation on upload (whitelist: .pdf .docx .txt .md, max 10 MB)",
             "CORS middleware, async throughout — production-ready FastAPI",
         ]),
    ]

    for title, bullets in caps:
        elems.append(KeepTogether([
            Paragraph(title, st["h3"]),
            *bullet_list(bullets, st),
            Spacer(1, 4),
        ]))

    return elems


def build_usecases(st):
    elems = section_header("06 · REAL-WORLD USE CASES", st)

    cases = [
        (
            "Investment Analyst",
            "compare funding rounds for Stripe, Braintree, and Adyen with a chart",
            "Comparison → 3 parallel Tavily searches → Validator → Comparison Matrix → Chart Writer → Synthesis",
            [
                "Side-by-side funding table across all three companies",
                "Matplotlib bar chart embedded in the answer",
                "Critical perspective highlighting risks",
                "3 intelligent follow-up questions",
            ],
            "~25 seconds. Replaces 2–3 hours of manual research per company.",
        ),
        (
            "Business Development Manager",
            "Write a PDF report on Anthropic's competitive position",
            "PDF → Planner → Decomposer (4 sub-queries) → Research → Report Writer → Chart Writer → PDF Generator",
            [
                "6-page downloadable PDF with branded header and footer",
                "Structured sections: Overview, Funding, Products, Competition, Outlook",
                "Chart embedded inside the PDF",
                "Sources and References with live hyperlinks",
            ],
            "~45 seconds. Replaces a day's work for a junior analyst.",
        ),
        (
            "Engineering Manager",
            "Upload internal spec PDF → 'Does this cover error handling for the payment flow?'",
            "Document (RAG) → ChromaDB similarity search → Cross-encoder reranking → Synthesis",
            [
                "Reads the actual uploaded document, not training knowledge",
                "Quotes specific passages from the spec",
                "Follow-up questions keep the document in context automatically",
                "Badge auto-dismisses after first answer — doc stays silently bound to thread",
            ],
            "~8 seconds. Instant technical due diligence on any internal document.",
        ),
        (
            "Startup Founder",
            "Draft an email to my CTO about OpenAI's latest funding round",
            "Email → Research (live Tavily search) → Email Drafter",
            [
                "Searches for the most recent OpenAI funding news",
                "Drafts a professional, accurate, ready-to-send email",
                "Cites real figures — no hallucinated numbers",
            ],
            "~20 seconds. Combines live research with professional writing in one step.",
        ),
        (
            "Data Analyst",
            "Bar chart: Apple $383B, Google $307B, Meta $116B revenue",
            "Chart → Chart Writer (Matplotlib)",
            [
                "Clean bar chart rendered and displayed inline",
                "Chart embedded in any subsequent PDF export",
                "No web search needed — model reads the numbers from your query",
            ],
            "~5 seconds. Instant visualisation from raw data.",
        ),
    ]

    for role, query, pipeline, outputs, time_val in cases:
        block = [
            Paragraph(role, st["h2"]),
            Paragraph(f'<b>Query:</b> <i>"{query}"</i>', st["body"]),
            Paragraph(f"<b>Pipeline:</b> {pipeline}", st["body_small"]),
            Paragraph("<b>Output:</b>", st["body"]),
            *bullet_list(outputs, st),
            Paragraph(f"<b>Value:</b> {time_val}", ParagraphStyle("val",
                fontName="Helvetica-Bold", fontSize=9, textColor=ACCENT,
                leading=14, spaceAfter=6)),
            Spacer(1, 4),
            hr(colors.HexColor("#e0e8ff"), 0.5),
        ]
        elems.append(KeepTogether(block))

    return elems


def build_differentiation(st):
    elems = section_header("07 · WHY ARES — NOT CHATGPT OR PERPLEXITY", st)
    elems.append(Spacer(1, 4))

    col_w = [6.0*cm, 2.5*cm, 2.8*cm, 4.7*cm]
    headers = ["Feature", "ChatGPT", "Perplexity", "Ares"]

    rows_raw = [
        ("Specialised agent pipelines",          "✗", "✗", "✓ 20 nodes, 12 modes"),
        ("Downloadable PDF export with charts",  "✗", "✗", "✓ ReportLab + Matplotlib"),
        ("Document RAG with cross-encoder",      "✓ basic", "✗", "✓ ChromaDB + FlashRank"),
        ("Persistent conversation checkpoints",  "✗", "✗", "✓ SQLite / AsyncSqliteSaver"),
        ("Multi-provider LLM auto-failover",     "✗", "✗", "✓ Groq→Cerebras→Gemini→OR"),
        ("Human-in-the-loop graph interrupt",    "✗", "✗", "✓ LangGraph interrupt"),
        ("Live streaming pipeline UI",           "✗", "partial", "✓ 12 SSE event types"),
        ("Comparison matrix output mode",        "✗", "✗", "✓ structured matrix + chart"),
        ("Fact-check + critic peer review",      "✗", "✗", "✓ dedicated agents"),
        ("Rate limiting + production safety",    "N/A", "N/A", "✓ SlowAPI + token budget"),
    ]

    def cell(val, col):
        if col in (1, 2):
            if val == "✗":
                return Paragraph("<b>✗</b>", ParagraphStyle("cx", fontName="Helvetica-Bold",
                    fontSize=10, textColor=RED, alignment=TA_CENTER, leading=13))
            elif val == "✓":
                return Paragraph("<b>✓</b>", ParagraphStyle("ck", fontName="Helvetica-Bold",
                    fontSize=10, textColor=GREEN, alignment=TA_CENTER, leading=13))
            elif val == "✓ basic":
                return Paragraph("✓ basic", ParagraphStyle("cb", fontName="Helvetica",
                    fontSize=8.5, textColor=colors.HexColor("#f59e0b"),
                    alignment=TA_CENTER, leading=13))
            else:
                return Paragraph(val, ParagraphStyle("cn", fontName="Helvetica",
                    fontSize=8.5, textColor=DIM_TEXT, alignment=TA_CENTER, leading=13))
        elif col == 3:
            return Paragraph(val, ParagraphStyle("ca", fontName="Helvetica-Bold",
                fontSize=8.5, textColor=ACCENT, leading=13))
        return Paragraph(val, ParagraphStyle("cf", fontName="Helvetica",
            fontSize=8.5, textColor=DARK_TEXT, leading=13))

    data = [[Paragraph(f"<b>{h}</b>", ParagraphStyle("th", fontName="Helvetica-Bold",
        fontSize=8.5, textColor=WHITE, leading=12, alignment=TA_CENTER))
        for h in headers]]

    for feat, gpt, perp, ares in rows_raw:
        data.append([
            Paragraph(feat, ParagraphStyle("f", fontName="Helvetica", fontSize=8.5,
                textColor=DARK_TEXT, leading=12)),
            cell(gpt, 1), cell(perp, 2), cell(ares, 3),
        ])

    tbl = Table(data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), NAVY),
        ("BACKGROUND",    (3,1),( 3,-1), LIGHT_BG),
        ("ROWBACKGROUNDS",(0,1),(2,-1),[WHITE, ROW_ALT]),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#d0d8f0")),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 7),
        ("RIGHTPADDING",  (0,0),(-1,-1), 7),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("LINEAFTER",     (2,0),(2,-1), 1.2, ACCENT),
    ]))
    elems.append(tbl)
    return elems


def build_tech_stack(st):
    elems = section_header("08 · TECH STACK", st)
    elems.append(Paragraph(
        "Built entirely from scratch as a take-home project — no pre-built agent frameworks or UI templates.",
        st["body"]
    ))
    elems.append(Spacer(1, 6))

    rows = [
        ("Orchestration",  "LangGraph ≥0.2.50",            "Stateful 20-node multi-agent graph with conditional edges"),
        ("API Layer",      "FastAPI + Uvicorn",             "Async REST + SSE streaming, CORS, rate limiting"),
        ("LLM Providers",  "Groq · Cerebras · Gemini · OpenRouter", "Auto-failover chain with shared interface"),
        ("Web Search",     "Tavily Python SDK",             "Parallel sub-query search with source metadata"),
        ("Vector DB",      "ChromaDB ≥0.5.0",              "Persistent cosine-similarity index for RAG"),
        ("Embeddings",     "all-MiniLM-L6-v2 (ONNX)",     "Default ChromaDB embedding function, ~79MB, cached"),
        ("Reranking",      "FlashRank ≥0.2.9",             "ms-marco-MiniLM-L-12-v2 cross-encoder, ~4MB"),
        ("Doc Parsing",    "pypdf · python-docx · LangChain splitter", "PDF, DOCX, TXT, MD with 1000-char chunks"),
        ("Checkpointing",  "AsyncSqliteSaver + aiosqlite", "Persistent LangGraph state across server restarts"),
        ("PDF Export",     "ReportLab ≥4.0.0",             "Tables, charts, citations, Unicode sanitisation"),
        ("Charts",         "Matplotlib ≥3.8.0",            "Bar / line / pie rendered as PNG, embedded in PDF"),
        ("Frontend",       "Vanilla JS + SSE",             "No framework — EventSource, marked.js, Prism.js"),
        ("Validation",     "Pydantic v2 + pydantic-settings","Type-safe config and request models"),
        ("Rate Limiting",  "SlowAPI ≥0.1.9",               "Per-IP limits on chat and upload endpoints"),
    ]

    headers = ["Layer", "Technology", "Role"]
    cw = [3.2*cm, 5.0*cm, 7.8*cm]
    elems.append(simple_table(headers, rows, cw, st))
    return elems


def build_pitch_structure(st):
    elems = section_header("09 · YOUR 5-MINUTE PITCH SCRIPT", st)
    elems.append(Paragraph(
        "Memorise this structure. Lead with the demo — let the product speak first.",
        st["body"]
    ))
    elems.append(Spacer(1, 6))

    steps = [
        ("0:00 – 0:30", "The Hook",
         'Open with: "Every professional wastes hours on manual research. Ares eliminates that — one sentence in, '
         'a fact-checked sourced report out, in under 60 seconds." Then immediately open the browser.'),
        ("0:30 – 1:30", "Live Demo — Comparison",
         'Type: "compare the funding rounds of OpenAI, Anthropic and Stripe with a chart". '
         'Narrate while it runs: "Watch the Agent Workspace — Clarity checks the question, '
         'Router picks Comparison mode, Decomposer fires three parallel searches simultaneously, '
         'Validator scores the sources…" Point out the streaming tokens.'),
        ("1:30 – 2:00", "Explain the Architecture",
         '"This is a 20-node LangGraph state machine. Each node is a specialised agent. '
         'The graph decides which agents to activate based on the query — 12 different pipelines for 12 types of work. '
         'No hardcoded scripts, no copy-paste prompts."'),
        ("2:00 – 2:45", "Live Demo — RAG",
         'Upload a PDF. Ask: "what is written in this?" Watch it route to Document mode. '
         '"I built a full RAG pipeline — ChromaDB vector store, cosine similarity retrieval, '
         'then a cross-encoder reranker to surface the most relevant chunks. '
         'The document stays in context for the whole conversation silently."'),
        ("2:45 – 3:30", "Live Demo — PDF Export",
         'Type: "write a PDF report on Anthropic". Download the PDF. '
         '"ReportLab generates a branded PDF with embedded charts, formatted tables, '
         'and a Sources section with clickable links. The chart is generated by Matplotlib and embedded automatically."'),
        ("3:30 – 4:15", "Differentiation",
         'Point to the comparison table: "This is not a ChatGPT wrapper. '
         'ChatGPT has none of these: persistent checkpointing, multi-provider failover, '
         'cross-encoder RAG, live pipeline UI, or PDF export. '
         'Each of those took real engineering decisions."'),
        ("4:15 – 5:00", "The Close",
         '"I built this end to end: LangGraph orchestration, FastAPI with SSE streaming, '
         'ChromaDB + FlashRank RAG, ReportLab PDF export, and a live streaming UI — all from scratch. '
         'It is production-ready today. Any question about the architecture I am happy to go deep on."'),
    ]

    for time_s, title, script in steps:
        block = [
            Table([[
                Paragraph(time_s, ParagraphStyle("tm", fontName="Helvetica-Bold",
                    fontSize=9, textColor=WHITE, leading=13)),
                Paragraph(title, ParagraphStyle("tt", fontName="Helvetica-Bold",
                    fontSize=9, textColor=colors.HexColor("#a0c0ff"), leading=13)),
            ]], colWidths=[2.8*cm, 13.2*cm],
            style=TableStyle([
                ("BACKGROUND",   (0,0),(-1,-1), NAVY),
                ("TOPPADDING",   (0,0),(-1,-1), 5),
                ("BOTTOMPADDING",(0,0),(-1,-1), 5),
                ("LEFTPADDING",  (0,0),(-1,-1), 8),
                ("RIGHTPADDING", (0,0),(-1,-1), 8),
            ])),
            Table([[Paragraph(script, ParagraphStyle("sc", fontName="Helvetica",
                fontSize=9, textColor=DARK_TEXT, leading=14))
            ]], colWidths=[W - 4*cm],
            style=TableStyle([
                ("BACKGROUND",   (0,0),(-1,-1), LIGHT_BG),
                ("TOPPADDING",   (0,0),(-1,-1), 7),
                ("BOTTOMPADDING",(0,0),(-1,-1), 7),
                ("LEFTPADDING",  (0,0),(-1,-1), 10),
                ("RIGHTPADDING", (0,0),(-1,-1), 10),
                ("LINEBELOW",    (0,0),(-1,-1), 0.5, colors.HexColor("#d0d8f0")),
            ])),
            Spacer(1, 3),
        ]
        elems.append(KeepTogether(block))

    elems += [
        Spacer(1, 10),
        Table([[Paragraph(
            "KEY RULE: Demo first, explain second. If the product impresses them visually, "
            "the technical explanation lands as proof — not as a claim.",
            ParagraphStyle("kr", fontName="Helvetica-Bold", fontSize=9.5,
                textColor=ACCENT, alignment=TA_CENTER, leading=14)
        )]], colWidths=[W - 4*cm],
        style=TableStyle([
            ("BACKGROUND",   (0,0),(-1,-1), colors.HexColor("#eef2ff")),
            ("TOPPADDING",   (0,0),(-1,-1), 10),
            ("BOTTOMPADDING",(0,0),(-1,-1), 10),
            ("LEFTPADDING",  (0,0),(-1,-1), 12),
            ("RIGHTPADDING", (0,0),(-1,-1), 12),
            ("BOX",          (0,0),(-1,-1), 1.5, ACCENT),
        ])),
    ]
    return elems


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(DIM_TEXT)
    canvas.drawString(2*cm, 1.2*cm, "Ares — Autonomous Research & Evidence System")
    canvas.drawRightString(W - 2*cm, 1.2*cm, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#e0e8ff"))
    canvas.setLineWidth(0.5)
    canvas.line(2*cm, 1.5*cm, W - 2*cm, 1.5*cm)
    canvas.restoreState()


def main():
    out = "static/reports/ares_pitch.pdf"
    doc = SimpleDocTemplate(
        out, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2.2*cm,
        title="Ares — Product Pitch & Technical Overview",
        author="Ares Research System",
    )

    st = build_styles()
    story = []

    story += build_cover(st)
    story += build_hook(st)
    story.append(PageBreak())
    story += build_what(st)
    story.append(PageBreak())
    story += build_workflow(st)
    story.append(PageBreak())
    story += build_modes(st)
    story.append(PageBreak())
    story += build_capabilities(st)
    story.append(PageBreak())
    story += build_usecases(st)
    story.append(PageBreak())
    story += build_differentiation(st)
    story.append(Spacer(1, 12))
    story += build_tech_stack(st)
    story.append(PageBreak())
    story += build_pitch_structure(st)

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"PDF written: {out}")


if __name__ == "__main__":
    main()
