"""Challenger agent — Research ↔ Synthesis debate.

Pipeline position (research / plan modes only):
    synthesis (draft) → challenger → fact_checker → critic → suggestions

What it does:
1. Reads the synthesis draft and asks the LLM to identify 2-3 specific,
   verifiable claims that could be challenged.
2. Searches Tavily for counter-evidence on each claim.
3. If counter-evidence is found, asks the LLM to reconcile both sides into
   a more nuanced, intellectually honest final answer (streamed).
4. If no counter-evidence is found, passes through silently so the original
   synthesis answer is used unchanged.
"""

import asyncio
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings, truncate_to_budget
from app.llm import get_llm
from app.state import AgentState
from app.tools.search import MOCK_RESULTS, tavily_search

logger = logging.getLogger(__name__)


# ── Structured output: claim extractor ────────────────────────────────────────

class DebatePoint(BaseModel):
    claim: str        = Field(description="A specific, verifiable claim from the draft answer")
    search_query: str = Field(description="Targeted Tavily query to find counter-evidence for this claim")


class ClaimAnalysis(BaseModel):
    is_debatable: bool              = Field(
        description="True if the answer makes verifiable factual claims worth stress-testing"
    )
    debate_points: list[DebatePoint] = Field(
        default_factory=list,
        description="Top 2 claims to challenge (empty if not debatable)"
    )


CLAIM_EXTRACTOR_PROMPT = """\
You are a rigorous fact-checker reviewing an AI-generated research answer.

Your job: identify the 2 most specific, verifiable factual claims in the answer
that a sceptic could reasonably challenge — not vague opinions, but concrete
assertions about numbers, causes, outcomes, or attributions.

Return is_debatable=False (and empty debate_points) if:
- The answer is merely descriptive or definitional with no contestable claims
- The answer already acknowledges all major counter-arguments
- The answer is very short (< 100 words) with no specific facts

For each debate point, write a targeted search query that would find opposing
evidence, criticism, or limitations — e.g. "OpenAI GPT-4 accuracy limitations
criticism" or "electric vehicles environmental cost manufacturing counter-argument".
"""

RECONCILE_PROMPT = """\
You are the Synthesis Reconciler in an AI research pipeline.

You have two sources of evidence:
1. PRIMARY EVIDENCE: what the initial research found and the draft answer was based on
2. COUNTER-EVIDENCE: new findings that challenge or qualify some of those claims

Your task: produce a single, improved, intellectually honest answer that:
- Retains what the primary evidence STRONGLY supports
- Honestly qualifies or walks back claims where counter-evidence is compelling
- Notes explicitly where the evidence is mixed or contested (e.g. "some studies argue... while others find...")
- Is MORE nuanced than the original draft, NOT a simple average of the two sides
- Picks the better-supported position where evidence clearly favours one side
- Is the same length or longer than the draft — do NOT truncate content

Rules:
- Do NOT add meta-commentary like "After reviewing counter-evidence..." — just write the answer
- Do NOT cite sources by number ([1], [2]) — use natural language ("According to X..." or "Research suggests...")
- Preserve any markdown headings from the original draft if they are still relevant
- Write in the same tone and style as the original draft
"""


def _format_counter_results(results: list[dict]) -> str:
    parts = []
    for r in results[:6]:
        title   = r.get("title", "")
        content = r.get("content", "")[:400]
        url     = r.get("url", "")
        parts.append(f"• {title}\n  {content}\n  Source: {url}")
    return "\n\n".join(parts)


async def _search_counter_evidence(debate_points: list[DebatePoint]) -> tuple[list[dict], list[str]]:
    """Run Tavily searches for each debate point in parallel."""
    if settings.mock_mode:
        return MOCK_RESULTS, [p.search_query for p in debate_points]

    async def _search(point: DebatePoint) -> list[dict]:
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: tavily_search(point.search_query, max_results=3)
            )
        except Exception as exc:
            logger.warning(f"challenger search failed for '{point.search_query}': {exc}")
            return []

    results_list = await asyncio.gather(*[_search(p) for p in debate_points])
    all_results  = [r for sublist in results_list for r in sublist]
    queries      = [p.search_query for p in debate_points]
    return all_results, queries


async def challenger_node(state: AgentState) -> dict:
    # Extract the synthesis draft from the last AI message.
    messages = state.get("messages", [])
    draft = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            draft = msg.content
            break

    if not draft or len(draft) < 80:
        logger.info("challenger: draft too short to debate, passing through")
        return {}

    original_query = state.get("original_query", "")
    findings       = state.get("findings", "") or state.get("raw_research", "")
    findings, _    = truncate_to_budget(findings, label="challenger_findings")

    llm = get_llm(streaming=False)

    # ── Step 1: identify debatable claims ────────────────────────────────────
    structured_llm = llm.with_structured_output(ClaimAnalysis)
    try:
        analysis: ClaimAnalysis = await structured_llm.ainvoke([
            SystemMessage(content=CLAIM_EXTRACTOR_PROMPT),
            HumanMessage(content=f"Question: {original_query}\n\nDraft answer:\n{draft[:3000]}"),
        ])
    except Exception as exc:
        logger.warning(f"challenger: claim extraction failed: {exc}")
        return {}

    if not analysis.is_debatable or not analysis.debate_points:
        logger.info("challenger: no debatable claims found, passing through")
        return {"challenge_queries": []}

    logger.info(f"challenger: found {len(analysis.debate_points)} debatable claims, searching for counter-evidence")

    # ── Step 2: search for counter-evidence ──────────────────────────────────
    counter_results, queries = await _search_counter_evidence(analysis.debate_points)

    if not counter_results:
        logger.info("challenger: no counter-evidence found, passing through")
        return {"challenge_queries": queries}

    counter_text = _format_counter_results(counter_results)

    # ── Step 3: reconcile into a nuanced final answer ─────────────────────────
    reconcile_context = (
        f"ORIGINAL QUESTION: {original_query}\n\n"
        f"DRAFT ANSWER (based on primary research):\n{draft}\n\n"
        f"CLAIMS THAT WERE CHALLENGED:\n"
        + "\n".join(f"- {p.claim}" for p in analysis.debate_points)
        + f"\n\nCOUNTER-EVIDENCE FOUND:\n{counter_text}"
    )

    streaming_llm = get_llm(streaming=True)
    reconciled = await streaming_llm.ainvoke([
        SystemMessage(content=RECONCILE_PROMPT),
        HumanMessage(content=reconcile_context),
    ])

    logger.info(f"challenger: reconciled answer produced (len={len(reconciled.content)})")

    return {
        "messages":        [AIMessage(content=reconciled.content)],
        "challenge_queries": queries,
        "counter_evidence":  counter_text,
    }
