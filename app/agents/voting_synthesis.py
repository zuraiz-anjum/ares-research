"""Voting Synthesis agent — parallel multi-perspective reasoning.

Replaces the single synthesis call for 'research' mode with three parallel
LLM calls (different reasoning styles), followed by a judge that picks the
strongest response and explains why.

Pipeline position (research mode only):
    research findings → voting_synthesis → challenger → fact_checker → ...

The three variants run concurrently via asyncio.gather — total wall-clock
time is ~= one LLM call, not three.
"""

import asyncio
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings, truncate_to_budget
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)


# ── Variant prompts ────────────────────────────────────────────────────────────

_BASE_RULES = """
Rules:
- Use ONLY the provided research findings. Do not fill gaps with training knowledge.
- Never invent figures, statistics, or valuations.
- If multiple entities are discussed, address ALL of them.
- Be concise but complete. Use markdown where it aids clarity.
"""

ANALYST_PROMPT = """You are a data-driven research analyst. Answer the user's question
from the research findings below with precision and rigour.

Style: numbers-first. Lead with the most significant quantitative facts.
Use bullet points for key data points, then a short paragraph for interpretation.
Be specific: cite exact figures, dates, and named entities.
Do NOT soften claims that are clearly supported — state them directly.
""" + _BASE_RULES

DEVILS_ADVOCATE_PROMPT = """You are a critical thinker who challenges obvious narratives.
Answer the user's question from the research findings below, but actively surface
the risks, limitations, and counterpoints that a surface-level read would miss.

Style: narrative paragraphs. Start with what seems true, then pivot to the tensions,
unknowns, or red flags the data also reveals. Balance is key — acknowledge strengths
but ensure risks get equal weight. Never make things up; only surface what the data hints at.
""" + _BASE_RULES

COMMUNICATOR_PROMPT = """You are a clear communicator explaining complex research to a
smart non-specialist. Answer the user's question from the research findings below.

Style: story-first. Open with the single most important insight in one sentence,
then build context around it in plain prose. Avoid jargon. Use analogies if they
genuinely clarify. Reserve bullet points for genuinely list-like information only.
End with one concrete takeaway.
""" + _BASE_RULES

# ── Judge prompt ────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are the Synthesis Judge. Three AI agents have answered the same
research question using the same underlying facts — each from a different angle.

Your job:
1. Read all three responses.
2. Pick the ONE that best answers the user's actual question — considering accuracy,
   completeness, and usefulness (not just style).
3. Output ONLY the winning response, verbatim, followed by a single line:

---
*[Analyst | Devil's Advocate | Communicator] perspective chosen — [one sentence explaining why]*

Do NOT merge, summarise, or rewrite the winning response. Output it exactly as written,
then append the verdict line above."""


# ── Structured output for judge ─────────────────────────────────────────────────

class JudgeVerdict(BaseModel):
    winner: str  = Field(description="One of: 'analyst', 'devils_advocate', 'communicator'")
    reason: str  = Field(description="One sentence explaining why this response best answers the question")


# ── Mock ────────────────────────────────────────────────────────────────────────

def _mock_response(query: str) -> dict:
    answer = (
        f"**Mock Voted Answer for:** {query}\n\n"
        "Three perspectives were evaluated; the analyst view was selected.\n\n"
        "- Revenue growth: 30% YoY\n"
        "- Funding: $500M Series D\n"
        "- Leadership: 3 C-suite hires in Q3 2025\n\n"
        "*Analyst perspective chosen — most precise match to the quantitative question.*"
    )
    return {
        "messages":          [AIMessage(content=answer)],
        "vote_winner":       "analyst",
        "vote_reason":       "Mock: analyst perspective selected",
        "vote_responses":    {"analyst": answer, "devils_advocate": answer, "communicator": answer},
    }


# ── Core logic ──────────────────────────────────────────────────────────────────

async def _run_variant(system_prompt: str, context: str, label: str) -> str:
    """Run a single synthesis variant. Returns the text content."""
    from app.utils.checkpoint import save_sub_agent_error
    llm = get_llm(streaming=False)
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context),
        ])
        logger.info(f"voting_synthesis: {label} completed len={len(response.content)}")
        return response.content
    except Exception as exc:
        await save_sub_agent_error("voting_synthesis", label, exc)
        return ""


async def voting_synthesis_node(state: AgentState) -> dict:
    if settings.mock_mode:
        return _mock_response(state.get("original_query", ""))

    query    = state.get("original_query", "")
    research = state.get("findings", "") or state.get("raw_research", "")
    research, _ = truncate_to_budget(research, label="voting_synthesis_findings")

    # Inject cross-session entity memory if available
    prior_context = ""
    try:
        from app.memory.entity_store import recall
        recalled = recall(query)
        if recalled:
            prior_context = f"Prior context from earlier sessions:\n{recalled}\n\n"
    except Exception:
        pass

    context = (
        f"{prior_context}"
        f"User question: {query}\n\n"
        f"Research findings:\n{research}"
    )

    # ── Step 1: run all three variants in parallel ────────────────────────────
    analyst_text, devils_text, comm_text = await asyncio.gather(
        _run_variant(ANALYST_PROMPT,         context, "analyst"),
        _run_variant(DEVILS_ADVOCATE_PROMPT, context, "devils_advocate"),
        _run_variant(COMMUNICATOR_PROMPT,    context, "communicator"),
    )

    # If all variants failed, fall back gracefully
    if not any([analyst_text, devils_text, comm_text]):
        logger.error("voting_synthesis: all variants failed, returning empty")
        return {"messages": [AIMessage(content="(synthesis failed)")]}

    responses = {
        "analyst":         analyst_text,
        "devils_advocate": devils_text,
        "communicator":    comm_text,
    }

    # ── Step 2: judge picks the winner ────────────────────────────────────────
    # Build a labelled comparison for the judge
    comparison = (
        f"QUESTION: {query}\n\n"
        "=== RESPONSE A (Analyst) ===\n"
        f"{analyst_text[:1800]}\n\n"
        "=== RESPONSE B (Devil's Advocate) ===\n"
        f"{devils_text[:1800]}\n\n"
        "=== RESPONSE C (Communicator) ===\n"
        f"{comm_text[:1800]}"
    )

    winner_key  = "analyst"
    winner_text = analyst_text or devils_text or comm_text
    reason      = "fallback — structured output unavailable"

    try:
        judge_llm = get_llm(streaming=False).with_structured_output(JudgeVerdict)
        verdict: JudgeVerdict = await judge_llm.ainvoke([
            SystemMessage(content=JUDGE_PROMPT),
            HumanMessage(content=comparison),
        ])
        winner_key  = verdict.winner if verdict.winner in responses else "analyst"
        reason      = verdict.reason
        winner_text = responses.get(winner_key) or winner_text
        logger.info(f"voting_synthesis: judge selected '{winner_key}' — {reason}")
    except Exception as exc:
        logger.warning(f"voting_synthesis: judge failed ({exc}), using analyst response")

    # Append verdict footnote to the winning answer
    label_map = {
        "analyst":         "Analyst",
        "devils_advocate": "Devil's Advocate",
        "communicator":    "Communicator",
    }
    final_answer = (
        winner_text.rstrip()
        + f"\n\n---\n*{label_map.get(winner_key, winner_key)} perspective chosen"
        + (f" — {reason}" if reason else "") + "*"
    )

    return {
        "messages":       [AIMessage(content=final_answer)],
        "vote_winner":    winner_key,
        "vote_reason":    reason,
        "vote_responses": responses,
    }
