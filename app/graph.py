"""Graph definition.

Wires all agents into a LangGraph state machine.

Pipelines by mode (set by intent_router after clarity):

  research      → decomposer → research → validator* → synthesis → fact_checker → critic → suggestions → END
  report        → decomposer → research → validator* → report_writer → suggestions → END
  data_analysis → decomposer → research → validator* → data_analyst → suggestions → END
  debate        → decomposer → research → validator* → debate_writer → suggestions → END
  plan          → planner → research → validator* → synthesis → fact_checker → critic → suggestions → END
  chat          → synthesis → suggestions → END
  document      → doc_agent → synthesis → suggestions → END
  code          → code_writer → suggestions → END

  * validator only if confidence_score < threshold

suggestions is the universal terminal node across all pipelines.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.clarity import clarity_node
from app.agents.code_writer import code_writer_node
from app.agents.critic import critic_node
from app.agents.data_analyst import data_analyst_node
from app.agents.debate_writer import debate_writer_node
from app.agents.decomposer import decomposer_node
from app.agents.doc_agent import doc_agent_node
from app.agents.fact_checker import fact_checker_node
from app.agents.intent_router import intent_router_node
from app.agents.planner import planner_node
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
    if mode == "debate":
        return "debate_writer"
    return "synthesis"  # research, plan


def route_after_intent(state: AgentState) -> str:
    mode = state.get("mode", "research")
    if mode == "chat":
        return "synthesis"
    if mode == "document":
        return "doc_agent"
    if mode == "code":
        return "code_writer"
    if mode == "plan":
        return "planner"
    # research, report, data_analysis, debate → research pipeline
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
    # Research and plan modes go through fact_checker → critic before suggestions.
    if state.get("mode") in ("research", "plan"):
        return "fact_checker"
    return "suggestions"


_RESEARCH_PIPELINE_TARGETS = {
    "validator":    "validator",
    "synthesis":    "synthesis",
    "report_writer":"report_writer",
    "data_analyst": "data_analyst",
    "debate_writer":"debate_writer",
}


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("clarity",       clarity_node)
    builder.add_node("intent_router", intent_router_node)
    builder.add_node("planner",       planner_node)
    builder.add_node("decomposer",    decomposer_node)
    builder.add_node("research",      research_node)
    builder.add_node("validator",     validator_node)
    builder.add_node("doc_agent",     doc_agent_node)
    builder.add_node("data_analyst",  data_analyst_node)
    builder.add_node("debate_writer", debate_writer_node)
    builder.add_node("fact_checker",  fact_checker_node)
    builder.add_node("synthesis",     synthesis_node)
    builder.add_node("report_writer", report_writer_node)
    builder.add_node("code_writer",   code_writer_node)
    builder.add_node("critic",        critic_node)
    builder.add_node("suggestions",   suggestions_node)

    builder.add_edge(START, "clarity")
    builder.add_edge("clarity", "intent_router")
    builder.add_conditional_edges(
        "intent_router", route_after_intent,
        {
            "decomposer":  "decomposer",
            "synthesis":   "synthesis",
            "doc_agent":   "doc_agent",
            "code_writer": "code_writer",
            "planner":     "planner",
        },
    )

    # plan mode: planner feeds directly into research (it generates sub_queries).
    builder.add_edge("planner", "research")

    # Standard research pipeline.
    builder.add_edge("decomposer", "research")
    builder.add_conditional_edges("research",   route_after_research,   _RESEARCH_PIPELINE_TARGETS)
    builder.add_conditional_edges("validator",  route_after_validation, _RESEARCH_PIPELINE_TARGETS)

    # Output nodes.
    builder.add_edge("doc_agent", "synthesis")
    builder.add_conditional_edges(
        "synthesis", route_after_synthesis,
        {"fact_checker": "fact_checker", "suggestions": "suggestions"},
    )
    builder.add_edge("fact_checker",  "critic")
    builder.add_edge("critic",        "suggestions")
    builder.add_edge("report_writer", "suggestions")
    builder.add_edge("data_analyst",  "suggestions")
    builder.add_edge("debate_writer", "suggestions")
    builder.add_edge("code_writer",   "suggestions")
    builder.add_edge("suggestions",   END)

    # MemorySaver supports all async methods (ainvoke, astream_events, aget_state).
    # For production persistence swap this for AsyncPostgresSaver or AsyncSqliteSaver
    # initialised in a FastAPI lifespan context manager.
    return builder.compile(checkpointer=MemorySaver())


# Compiled once at import time and reused across requests.
graph = build_graph()
