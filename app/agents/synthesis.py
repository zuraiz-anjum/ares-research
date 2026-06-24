"""Synthesis agent.

Turns the validated research findings into a clear, well-structured answer for
the user, using the conversation history to stay on-topic across turns.
"""

from langchain_core.messages import AIMessage, SystemMessage

from app.llm import get_llm
from app.state import AgentState

SYNTHESIS_SYSTEM_PROMPT = """You are the Synthesis Agent in a company-research assistant.
Write a clear, well-structured answer to the user's question using the research
findings below. Be specific, cite concrete facts, and keep the conversation
context in mind so follow-up questions feel natural.

Research findings:
{research}"""


def synthesis_node(state: AgentState) -> dict:
    research =state.get("findings", "") or state.get("raw_research", "") 
    history = state["messages"][-10:]  # last 10 messages for context

    system = SYNTHESIS_SYSTEM_PROMPT.format(research=research)

    llm = get_llm()
    response = llm.invoke([SystemMessage(content=system), *history])

    return {"messages": [AIMessage(content=response.content)]}
