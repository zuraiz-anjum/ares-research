"""Graph definition.

Wires all agents into a LangGraph state machine.

Flow by mode (set by intent_router after clarity):

  research      → decomposer → research → (validator | synthesis) → synthesis → critic → suggestions → END
  report        → decomposer → research → (validator | report_writer) → report_writer → suggestions → END
  data_analysis → decomposer → research → (validator | data_analyst) → data_analyst → suggestions → END
  chat          → synthesis → suggestions → END
  document      → doc_agent → synthesis → suggestions → END
  code          → code_writer → suggestions → END

suggestions is the universal last node before END across all pipelines.
"""

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.agents.clarity import clarity_node
from app.agents.code_writer import code_writer_node
from app.agents.critic import critic_node
from app.agents.data_analyst import data_analyst_node
from app.agents.decomposer import decomposer_node
from app.agents.doc_agent import doc_agent_node
from app.agents.intent_router import intent_router_node
from app.agents.report_writer import report_writer_node
from app.agents.research import research_node
from app.agents.suggestions import suggestions_node
from app.agents.synthesis import synthesis_node
from app.agents.validator import validator_node
from app.config import settings
from app.state import AgentState


def _after_research_pipeline(state: AgentState) -> str:
    """After sufficient research, route to the appropriate output node."""
    mode = state.get("mode", "research")
    if mode == "report":
        return "report_writer"
    if mode == "data_analysis":
        return "data_analyst"
    return "synthesis"


def route_after_intent(state: AgentState) -> str:
    mode = state.get("mode", "research")
    if mode == "chat":
        return "synthesis"
    if mode == "document":
        return "doc_agent"
    if mode == "code":
        return "code_writer"
    # research, report, data_analysis all go through the research pipeline.
    return "decomposer"


def route_after_research(state: AgentState) -> str:
    if state["confidence_score"] >= settings.confidence_threshold:
        return _after_research_pipeline(state)
    return "validator"


def route_after_validation(state: AgentState) -> str:
    if state["validation_result"] == "sufficient":
        return _after_research_pipeline(state)
    if state["attempts"] < settings.max_validation_attempts:
        return "research"
    return _after_research_pipeline(state)


def route_after_synthesis(state: AgentState) -> str:
    # Only research-mode queries get a critic pass.
    return "critic" if state.get("mode") == "research" else "suggestions"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("clarity", clarity_node)
    builder.add_node("intent_router", intent_router_node)
    builder.add_node("decomposer", decomposer_node)
    builder.add_node("research", research_node)
    builder.add_node("validator", validator_node)
    builder.add_node("doc_agent", doc_agent_node)
    builder.add_node("data_analyst", data_analyst_node)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("report_writer", report_writer_node)
    builder.add_node("code_writer", code_writer_node)
    builder.add_node("critic", critic_node)
    builder.add_node("suggestions", suggestions_node)

    builder.add_edge(START, "clarity")
    builder.add_edge("clarity", "intent_router")
    builder.add_conditional_edges(
        "intent_router", route_after_intent,
        {
            "decomposer": "decomposer",
            "synthesis":  "synthesis",
            "doc_agent":  "doc_agent",
            "code_writer":"code_writer",
        },
    )

    # Research pipeline.
    builder.add_edge("decomposer", "research")
    builder.add_conditional_edges(
        "research", route_after_research,
        {
            "validator":    "validator",
            "synthesis":    "synthesis",
            "report_writer":"report_writer",
            "data_analyst": "data_analyst",
        },
    )
    builder.add_conditional_edges(
        "validator", route_after_validation,
        {
            "research":     "research",
            "synthesis":    "synthesis",
            "report_writer":"report_writer",
            "data_analyst": "data_analyst",
        },
    )

    # Output nodes → suggestions → END.
    builder.add_edge("doc_agent", "synthesis")
    builder.add_conditional_edges(
        "synthesis", route_after_synthesis,
        {"critic": "critic", "suggestions": "suggestions"},
    )
    builder.add_edge("critic", "suggestions")
    builder.add_edge("report_writer", "suggestions")
    builder.add_edge("data_analyst", "suggestions")
    builder.add_edge("code_writer", "suggestions")
    builder.add_edge("suggestions", END)

    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)


# Compiled once at import time and reused across requests.
graph = build_graph()
