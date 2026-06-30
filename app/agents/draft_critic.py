"""Draft Critic agent.

Evaluates a report or academic paper draft and provides a quality score plus
specific, actionable feedback. If the score is below the acceptance threshold
the draft is returned to the appropriate writer for revision.

Max revision cycles is controlled by MAX_REVISIONS (default 2 = one revision
pass before forcing the pipeline through regardless of score).
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import truncate_to_budget
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

MAX_REVISIONS    = 2   # critic runs this many times max before forcing acceptance
ACCEPT_THRESHOLD = 7   # score >= this means accepted


CRITIC_SYSTEM_PROMPT = """You are a rigorous Draft Quality Critic for an AI research pipeline.

Your job: evaluate a research report or academic paper draft and return a score plus specific
actionable feedback so the writer knows exactly what to fix.

SCORING (1-10):
  9-10  Excellent — thorough, evidence-backed, well-structured, ready to deliver
  7-8   Good — minor gaps or thin sections; still publishable with small edits
  5-6   Adequate — notable structural problems or missing content
  1-4   Poor — major issues: missing sections, unsupported claims, poor flow

FEEDBACK RULES:
  - Name the exact section or claim that needs work
  - Say WHAT to add or change, not just "it's weak"
  - Keep feedback to 5-8 bullet points, under 300 words total
  - If score >= 7, still note 1-2 improvements for the writer to consider

Return JSON with three fields:
  score     (int 1-10)
  feedback  (string — bullet-pointed critique)
  accepted  (bool — true when score >= 7)"""


class DraftVerdict(BaseModel):
    score:    int  = Field(ge=1, le=10, description="Quality score 1-10")
    feedback: str  = Field(description="Specific, actionable critique for the writer")
    accepted: bool = Field(description="True when score >= 7")


async def draft_critic_node(state: AgentState) -> dict:
    mode           = state.get("mode", "report")
    revision_count = state.get("revision_count", 0)
    draft          = state.get("report_content", "") or ""

    if not draft:
        logger.info(f"draft_critic no_draft mode={mode} — skipping")
        return {
            "draft_score":       10,
            "revision_feedback": "",
            "revision_count":    revision_count + 1,
            "messages":          [AIMessage(content="Draft critique skipped — no content to review.")],
        }

    draft_preview, _ = truncate_to_budget(draft, label="draft_critic_input")
    query = state.get("original_query", "")

    prompt = (
        f"Original research question: {query}\n\n"
        f"Mode: {mode}  |  Critique pass #{revision_count + 1}\n\n"
        f"--- DRAFT ---\n{draft_preview}"
    )

    llm = get_llm(temperature=0).with_structured_output(DraftVerdict)
    verdict: DraftVerdict = await llm.ainvoke([
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    new_revision_count = revision_count + 1

    # Force acceptance when we hit the revision ceiling.
    if new_revision_count >= MAX_REVISIONS:
        verdict.accepted = True
        logger.info(
            f"draft_critic max_revisions_reached score={verdict.score} "
            f"revision_count={new_revision_count} mode={mode}"
        )
    else:
        verdict.accepted = verdict.score >= ACCEPT_THRESHOLD

    if verdict.accepted:
        msg = f"✓ Draft accepted — score {verdict.score}/10.\n\n{verdict.feedback}"
        feedback_out = ""
    else:
        msg = (
            f"Draft score {verdict.score}/10 — revision #{new_revision_count} requested.\n\n"
            f"{verdict.feedback}"
        )
        feedback_out = verdict.feedback

    logger.info(
        f"draft_critic score={verdict.score} accepted={verdict.accepted} "
        f"revision_count={new_revision_count} mode={mode}"
    )

    return {
        "draft_score":       verdict.score,
        "revision_feedback": feedback_out,
        "revision_count":    new_revision_count,
        "messages":          [AIMessage(content=msg)],
    }
