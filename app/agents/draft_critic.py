"""Draft Critic agent.

Evaluates a report or academic paper draft and provides a quality score plus
specific, actionable feedback. If the score is below the acceptance threshold
the draft is returned to the appropriate writer for revision.

Max revision cycles is controlled by MAX_REVISIONS (default 2 = one revision
pass before forcing the pipeline through regardless of score).
"""

import json
import logging
import re

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

You MUST respond with ONLY a valid JSON object and nothing else — no markdown, no explanation:
{"score": <int 1-10>, "feedback": "<bullet-pointed critique>", "accepted": <true|false>}"""


class DraftVerdict(BaseModel):
    score:    int  = Field(ge=1, le=10, description="Quality score 1-10")
    feedback: str  = Field(description="Specific, actionable critique for the writer")
    accepted: bool = Field(description="True when score >= 7")


def _parse_verdict(text: str) -> DraftVerdict:
    """Extract JSON from LLM response, tolerating markdown fences, Hermes
    function wrappers, and malformed/unquoted string values.
    """
    # Strip markdown fences and Hermes function-call wrappers
    text = re.sub(r"```(?:json)?|```", "", text)
    text = re.sub(r"<function=[^>]+>|</function>", "", text).strip()

    # 1. Try strict JSON first (clean responses from Cerebras / Gemini)
    start = text.find("{")
    if start != -1:
        try:
            decoder = json.JSONDecoder()
            data, _ = decoder.raw_decode(text, start)
            score = int(data["score"])
            return DraftVerdict(
                score=score,
                feedback=str(data["feedback"]),
                accepted=bool(data.get("accepted", score >= ACCEPT_THRESHOLD)),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # 2. Fallback: regex-extract each field individually (handles Groq's
    #    malformed output where feedback value is not a quoted JSON string)
    score_m = re.search(r'"score"\s*:\s*(\d+)', text)
    accept_m = re.search(r'"accepted"\s*:\s*(true|false)', text, re.I)

    if not score_m:
        raise ValueError(f"Cannot extract score from draft_critic response: {text[:300]}")

    score = int(score_m.group(1))
    accepted = accept_m.group(1).lower() == "true" if accept_m else score >= ACCEPT_THRESHOLD

    # Extract feedback: everything between "feedback": and the next top-level key or end
    feedback_m = re.search(r'"feedback"\s*:\s*(.*?)(?=,\s*"(?:score|accepted)"|\s*\})', text, re.S)
    feedback = feedback_m.group(1).strip().strip('"') if feedback_m else text[:500]

    logger.warning(f"draft_critic used fallback field extraction (malformed JSON)")
    return DraftVerdict(score=score, feedback=feedback, accepted=bool(accepted))


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

    # Use plain text output + manual JSON parse to avoid Groq function-calling
    # format bug (model generates <function=DraftVerdict>...</function> which
    # LangChain's tool binding rejects with a 400 BadRequestError).
    llm = get_llm(temperature=0)
    response = await llm.ainvoke([
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    verdict = _parse_verdict(response.content)

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
