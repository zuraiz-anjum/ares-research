"""Synthesis agent.

Turns the validated research findings into a clear, well-structured answer for
the user, using the conversation history to stay on-topic across turns.
"""

from langchain_core.messages import AIMessage, SystemMessage

from app.config import settings, truncate_to_budget
from app.llm import get_llm
from app.state import AgentState

RESEARCH_SYNTHESIS_PROMPT = """You are the Synthesis Agent in a research workspace.
Write a clear, well-structured answer to the user's question using the research
findings below. Be specific, cite concrete facts, and keep the conversation
context in mind so follow-up questions feel natural.

Research findings:
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


async def synthesis_node(state: AgentState) -> dict:
    mode = state.get("mode", "research")
    history = state["messages"][-10:]

    if settings.mock_mode:
        labels = {"research": "research", "chat": "chat", "document": "document analysis"}
        return {"messages": [AIMessage(content=f"Mock {labels.get(mode, mode)} answer: placeholder response.")]}

    if mode == "chat":
        system = CHAT_SYNTHESIS_PROMPT
    elif mode == "document":
        content = state.get("findings", "") or state.get("raw_research", "")
        content, _ = truncate_to_budget(content, label="doc_content")
        system = DOCUMENT_SYNTHESIS_PROMPT.format(content=content)
    else:
        research = state.get("findings", "") or state.get("raw_research", "")
        research, _ = truncate_to_budget(research, label="synthesis_findings")
        system = RESEARCH_SYNTHESIS_PROMPT.format(research=research)

    # streaming=True causes LangGraph's astream_events to emit on_chat_model_stream
    # events per token, which the SSE endpoint forwards to the client in real time.
    llm = get_llm(streaming=True)
    response = await llm.ainvoke([SystemMessage(content=system), *history])
    return {"messages": [AIMessage(content=response.content)]}
