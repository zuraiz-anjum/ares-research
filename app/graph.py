"""Graph definition.

Wires the four agents into a LangGraph state machine:

    START -> clarity -> research -> (validator | synthesis)
    validator -> (research | synthesis)
    synthesis -> END

Routing is conditional on each agent's output:
  * research routes to synthesis when confidence is high enough, otherwise to
    the validator;
  * the validator loops back to research while the findings are insufficient
    and we still have attempts left, otherwise it proceeds to synthesis.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.clarity import clarity_node
from app.agents.research import research_node
from app.agents.synthesis import synthesis_node
from app.agents.validator import validator_node
from app.config import settings
from app.state import AgentState


def route_after_research(state: AgentState) -> str:
    """High-confidence research goes straight to synthesis; otherwise validate."""
    if state["confidence_score"] >= settings.confidence_threshold:
        return "synthesis"
    return "validator"


def route_after_validation(state: AgentState) -> str:
    """Loop back to research while insufficient and attempts remain."""
    if state["validation_result"] == "sufficient":
        return "synthesis"
    if state["attempts"] < settings.max_validation_attempts:
        return "research"
    return "synthesis"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("clarity", clarity_node)
    builder.add_node("research", research_node)
    builder.add_node("validator", validator_node)
    builder.add_node("synthesis", synthesis_node)

    builder.add_edge(START, "clarity")
    builder.add_edge("clarity", "research")
    builder.add_conditional_edges(
        "research", route_after_research, {"validator": "validator", "synthesis": "synthesis"}
    )
    builder.add_conditional_edges(
        "validator", route_after_validation, {"research": "research", "synthesis": "synthesis"}
    )
    builder.add_edge("synthesis", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Compiled once at import time and reused across requests.
graph = build_graph()
