"""Synthesis agent.

Turns the validated research findings into a clear, well-structured answer for
the user, using the conversation history to stay on-topic across turns.

Long conversations are handled via a rolling summary: when the message history
exceeds SUMMARY_THRESHOLD turns, older messages are compressed into a single
SystemMessage summary so the context window stays lean without losing context.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import settings, truncate_to_budget
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

# Compress history when there are more than this many messages (human+AI pairs).
SUMMARY_THRESHOLD = 10

RESEARCH_SYNTHESIS_PROMPT = """You are the Synthesis Agent in a research workspace.
Write a clear, well-structured answer to the user's question using ONLY the
research findings below.

RULES:
- Cite sources inline using their bracketed numbers, e.g. "OpenAI raised $6.6B [3]."
- If multiple entities are discussed, address ALL of them with equal depth.
- Do NOT fill gaps with training knowledge. If data is missing, say so explicitly.
- Never invent figures, funding amounts, valuations, or statistics.
- If prior session context is provided, use it to supplement (not replace) the research.

{prior_context}Research findings:
{research}"""

DOCUMENT_SYNTHESIS_PROMPT = """You are a Document Analysis agent. Answer the user's
question using the document content below. Be specific and quote or paraphrase
relevant passages when helpful. If the document doesn't contain enough information
to answer, say so clearly.

Document content:
{content}"""

CHAT_SYNTHESIS_PROMPT = """You are a helpful AI assistant. Answer the user's
question clearly, accurately, and concisely. Use markdown formatting when it
genuinely helps clarity (headers, bullets, code blocks). Avoid unnecessary
padding or filler sentences."""

SUMMARISE_PROMPT = """You are a conversation summariser. Compress the following
conversation into a concise bullet-point summary (max 200 words) that captures
the key topics discussed, decisions made, and facts established. Preserve
specific numbers, company names, and conclusions."""


def _build_history(state: AgentState) -> list:
    """Return the most recent messages, injecting a summary for older turns."""
    all_msgs = state.get("messages", [])
    existing_summary = state.get("conversation_summary", "")

    if len(all_msgs) <= SUMMARY_THRESHOLD:
        return all_msgs[-SUMMARY_THRESHOLD:]

    # Split: keep the last 6 messages fresh; compress everything before that.
    fresh = all_msgs[-6:]
    to_summarise = all_msgs[:-6]

    try:
        llm = get_llm(temperature=0)
        text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:400]}"
            for m in to_summarise
        )
        if existing_summary:
            text = f"Previous summary:\n{existing_summary}\n\nNew messages:\n{text}"
        summary_msg = llm.invoke([
            SystemMessage(content=SUMMARISE_PROMPT),
            HumanMessage(content=text),
        ])
        summary = summary_msg.content
        logger.info(f"conversation_summarised turns={len(to_summarise)}")
    except Exception:
        logger.warning("summarisation_failed falling back to truncation", exc_info=True)
        summary = existing_summary or "(earlier conversation omitted)"

    return [SystemMessage(content=f"[Earlier conversation summary]\n{summary}"), *fresh]


async def synthesis_node(state: AgentState) -> dict:
    mode = state.get("mode", "research")

    if settings.mock_mode:
        labels = {"research": "research", "chat": "chat", "document": "document analysis"}
        return {"messages": [AIMessage(content=f"Mock {labels.get(mode, mode)} answer: placeholder response.")]}

    history = _build_history(state)

    if mode == "chat":
        system = CHAT_SYNTHESIS_PROMPT
    elif mode == "document":
        content = state.get("findings", "") or state.get("raw_research", "")
        content, _ = truncate_to_budget(content, label="doc_content")
        system = DOCUMENT_SYNTHESIS_PROMPT.format(content=content)
    else:
        research = state.get("findings", "") or state.get("raw_research", "")
        research, _ = truncate_to_budget(research, label="synthesis_findings")

        # Inject cross-session entity memory when available.
        # recall() does a synchronous full-file read — offload to a thread so
        # it doesn't block the event loop for every other in-flight request.
        prior_context = ""
        try:
            import asyncio
            from app.memory.entity_store import recall
            loop = asyncio.get_running_loop()
            recalled = await loop.run_in_executor(None, recall, state.get("original_query", ""))
            if recalled:
                prior_context = recalled + "\n\n"
        except Exception:
            pass

        system = RESEARCH_SYNTHESIS_PROMPT.format(research=research, prior_context=prior_context)

    # streaming=True causes LangGraph's astream_events to emit on_chat_model_stream
    # events per token, which the SSE endpoint forwards to the client in real time.
    llm = get_llm(streaming=True)
    response = await llm.ainvoke([SystemMessage(content=system), *history])

    # Store entity facts from this session for future recall (fire-and-forget).
    if mode not in ("chat", "document"):
        try:
            import asyncio
            from app.memory.entity_store import extract_and_store
            from app.utils.background import spawn
            findings = state.get("findings", "")
            query    = state.get("original_query", "")
            if findings and query:
                loop = asyncio.get_running_loop()
                spawn(loop.run_in_executor(None, extract_and_store, findings, query))
        except Exception:
            pass

    return {"messages": [AIMessage(content=response.content)]}
