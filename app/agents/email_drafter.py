"""Email Drafter agent.

Triggered by "draft an email", "write an email about", "email to [person] about".
Uses the research pipeline for factual grounding, then formats everything as a
ready-to-send professional email with subject line and body.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import settings, truncate_to_budget
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

EMAIL_SYSTEM_PROMPT = """You are the Email Drafter agent in an AI research workspace.

You have received research findings on a topic. Draft a professional, ready-to-send email.

Output format — always produce exactly this structure:

**Subject:** [concise, specific subject line]

---

[Opening salutation — use "Hi [Name]," if a name is specified, otherwise "Hi there,"]

[Body — 2-4 short paragraphs. Each paragraph has a single purpose:
  1. Context / reason for writing
  2. Key facts / main content (draw from research findings with specific numbers)
  3. Optional: secondary insight or context
  4. Clear call-to-action or next step]

[Closing — "Best regards," or "Kind regards,"]
[Sender placeholder — use "[Your Name]" unless specified]

---

Rules:
- Emails must be concise. Total body: 100-200 words.
- Tone: professional but not stiff. Conversational sentences.
- Embed specific facts and numbers from the research — this makes the email credible.
- If the user specified a recipient, recipient's role, or sender name, use them.
- If no call-to-action is implied, end with an offer to discuss further.
- Never fabricate facts. If research is sparse, note that clearly in the email body.
- After the email, add a short section:
  **Draft notes:** One sentence on what to personalise before sending.
"""


async def email_drafter_node(state: AgentState) -> dict:
    question = state.get("original_query", "")
    findings = state.get("findings", "") or state.get("raw_research", "")

    if settings.mock_mode:
        mock = (
            "**Subject:** Q3 Market Update — Strong Growth Signals\n\n"
            "---\n\n"
            "Hi there,\n\n"
            "I wanted to share a quick update on the market research you requested.\n\n"
            "The company recorded 30% YoY revenue growth in Q3, supported by a recent "
            "$500M Series D at a $10B valuation. Leadership has expanded with three "
            "C-suite hires, signalling continued scaling ambitions.\n\n"
            "Happy to jump on a call this week to discuss implications for our strategy.\n\n"
            "Best regards,\n"
            "[Your Name]\n\n"
            "---\n\n"
            "**Draft notes:** Personalise the opening with the recipient's name and "
            "adjust the call-to-action to match your specific ask."
        )
        return {"messages": [AIMessage(content=mock)]}

    findings, _ = truncate_to_budget(findings, label="email_findings")

    context = f"Email request: {question}\n"
    if findings:
        context += f"\nResearch findings to draw from:\n{findings}"

    llm = get_llm(streaming=True)
    response = await llm.ainvoke([
        SystemMessage(content=EMAIL_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ])
    logger.info("email_drafter_complete")
    return {"messages": [AIMessage(content=response.content)]}
