"""Suggestions agent.

The final node in every pipeline. Generates 3 short follow-up questions
the user might naturally want to ask next, based on the answer that was given.

Results are surfaced in the UI as clickable chips below the response.
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

SUGGESTIONS_SYSTEM_PROMPT = """You are a Follow-up Question Generator.

Given a user's question and the answer they just received, generate exactly 3
follow-up questions they would naturally want to ask next.

Rules:
- Each question should explore a DIFFERENT angle: e.g. deeper detail, comparison,
  implication, risk, or next action.
- Keep each question SHORT — ideally under 10 words.
- Do NOT repeat information already in the answer.
- Output ONLY a valid JSON array of exactly 3 strings. No other text.

Example output: ["Who are their main competitors?", "What's their runway?", "Any recent layoffs?"]"""


def suggestions_node(state: AgentState) -> dict:
    if settings.mock_mode:
        mode = state.get("mode", "research")
        defaults = {
            "code": ["How do I test this?", "Can this handle edge cases?", "How do I extend this?"],
            "report": ["Which metric is most important?", "How does this compare to last year?", "What should I watch next?"],
            "data_analysis": ["Which company has better margins?", "What's the growth forecast?", "How did they get there?"],
        }
        return {"suggestions": defaults.get(mode, [
            "Who are their main competitors?",
            "What's their revenue growth trend?",
            "Any recent leadership changes?",
        ])}

    question = state.get("original_query", "")
    messages = state.get("messages", [])
    answer = messages[-1].content if messages else ""
    mode = state.get("mode", "research")

    if mode == "code":
        prompt = f"The user asked for code: {question}\n\nGenerate 3 follow-up questions about testing, extending, or improving this code."
    else:
        prompt = f"User's question: {question}\n\nAnswer given (truncated):\n{answer[:800]}"

    llm = get_llm(temperature=0.6)
    try:
        response = llm.invoke([
            SystemMessage(content=SUGGESTIONS_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        content = response.content.strip()
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            questions = json.loads(content[start:end])
            if isinstance(questions, list):
                return {"suggestions": [str(q) for q in questions[:3]]}
    except Exception as exc:
        logger.warning(f"suggestions_failed: {exc}")

    return {"suggestions": []}
