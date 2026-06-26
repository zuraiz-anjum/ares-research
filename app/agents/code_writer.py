"""Code Writer agent.

Handles the 'code' pipeline mode. Produces clean, idiomatic, production-quality
code with a brief explanation — no web search needed.

Streams tokens so code appears character-by-character in the UI. Supports any
language (auto-detected from context, defaults to Python).
"""

import logging

from langchain_core.messages import AIMessage, SystemMessage

from app.config import settings
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

CODE_WRITER_SYSTEM_PROMPT = """You are an expert software engineer and coding assistant.

When given a coding request:
1. Detect the language from context; default to Python if unspecified.
2. Write one line of plain-text explanation before the code block.
3. Write clean, idiomatic, production-quality code in a fenced markdown code block
   with the correct language tag (e.g. ```python, ```typescript, ```bash).
4. Add concise inline comments only for non-obvious logic.
5. After the code, add a **How it works** section with 3-5 bullet points.
6. If the request has multiple reasonable approaches, implement the best one and
   briefly note the alternative in one sentence.

Do NOT add boilerplate disclaimers, unnecessary imports, or verbose explanations.
Keep it tight — a senior engineer reading this should get value immediately."""


async def code_writer_node(state: AgentState) -> dict:
    history = state["messages"][-10:]

    if settings.mock_mode:
        return {"messages": [AIMessage(content=(
            "Here's a Python implementation using a dictionary for O(1) lookups:\n\n"
            "```python\ndef two_sum(nums: list[int], target: int) -> list[int]:\n"
            "    seen = {}\n"
            "    for i, n in enumerate(nums):\n"
            "        diff = target - n\n"
            "        if diff in seen:\n"
            "            return [seen[diff], i]\n"
            "        seen[n] = i\n"
            "    return []\n```\n\n"
            "**How it works:**\n"
            "- Iterates through the list once — O(n) time, O(n) space\n"
            "- Stores each number's index in a hash map\n"
            "- For each number, checks if its complement (target - n) is already in the map\n"
            "- Returns indices immediately on match, avoiding a nested loop"
        ))]}

    llm = get_llm(streaming=True)
    response = await llm.ainvoke([
        SystemMessage(content=CODE_WRITER_SYSTEM_PROMPT),
        *history,
    ])
    logger.info("code_writer_complete")
    return {"messages": [AIMessage(content=response.content)]}
