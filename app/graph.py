"""Graph definition.

Wires all agents into a LangGraph state machine.

Pipelines by mode (set by intent_router after clarity):

  research      → decomposer → research → validator* → voting_synthesis (×3 parallel) → challenger → fact_checker → critic → suggestions → END
  report        → decomposer → research → validator* → report_writer → data_visualizer → suggestions → END
  pdf           → decomposer → research → validator* → report_writer → data_visualizer → pdf_generator → suggestions → END
  academic      → decomposer → research → validator* → academic_writer → data_visualizer → pdf_generator → suggestions → END
  data_analysis → decomposer → research → validator* → data_analyst → data_visualizer → suggestions → END
  comparison    → decomposer → research → validator* → comparison_matrix → data_visualizer → suggestions → END
  debate        → decomposer → research → validator* → debate_writer → suggestions → END
  email         → decomposer → research → validator* → email_drafter → suggestions → END
  plan          → planner → research → validator* → dynamic_spawner (parallel specialists) → challenger → fact_checker → critic → suggestions → END
  chat          → synthesis → suggestions → END
  document      → doc_agent → synthesis → suggestions → END
  code          → code_writer → suggestions → END
  chart         → chart_writer → suggestions → END   (single chart from user-supplied data)
  survey        → survey_analyst → suggestions → END

  * validator only if confidence_score < threshold

suggestions is the universal terminal node across all pipelines.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agents.academic_writer import academic_writer_node
from app.agents.data_extractor import data_extractor_node
from app.agents.chart_writer import chart_writer_node
from app.agents.draft_critic import draft_critic_node, MAX_REVISIONS
from app.agents.clarity import clarity_node
from app.agents.code_writer import code_writer_node
from app.agents.comparison_matrix import comparison_matrix_node
from app.agents.critic import critic_node
from app.agents.data_analyst import data_analyst_node
from app.agents.data_visualizer import data_visualizer_node
from app.agents.debate_writer import debate_writer_node
from app.agents.decomposer import decomposer_node
from app.agents.doc_agent import doc_agent_node
from app.agents.email_drafter import email_drafter_node
from app.agents.fact_checker import fact_checker_node
from app.agents.intent_router import intent_router_node
from app.agents.pdf_generator import pdf_generator_node
from app.agents.planner import planner_node
from app.agents.report_writer import report_writer_node
from app.agents.research import research_node
from app.agents.suggestions import suggestions_node
from app.agents.survey_analyst import survey_analyst_node
from app.agents.challenger import challenger_node
from app.agents.dynamic_spawner import dynamic_spawner_node
from app.agents.synthesis import synthesis_node
from app.agents.voting_synthesis import voting_synthesis_node
from app.agents.validator import validator_node
from app.config import settings
from app.state import AgentState
from app.utils.checkpoint import checkpoint


def _after_research_pipeline(state: AgentState) -> str:
    """After sufficient research, route to the appropriate output node."""
    mode = state.get("mode", "research")
    # Modes that produce visual reports go through data_extractor first so
    # the visualizer has structured CSV data rather than raw prose.
    if mode in ("report", "pdf", "academic", "data_analysis", "comparison"):
        return "data_extractor"
    if mode == "debate":
        return "debate_writer"
    if mode == "email":
        return "email_drafter"
    if mode == "research":
        return "voting_synthesis"   # parallel multi-perspective reasoning
    if mode == "plan":
        return "dynamic_spawner"    # findings-aware agent selection
    return "synthesis"  # document fallback


def route_after_data_extractor(state: AgentState) -> str:
    """Route from data_extractor to the appropriate writer based on mode."""
    mode = state.get("mode", "report")
    if mode in ("report", "pdf"):
        return "report_writer"
    if mode == "academic":
        return "academic_writer"
    if mode == "data_analysis":
        return "data_analyst"
    if mode == "comparison":
        return "comparison_matrix"
    return "report_writer"


def route_after_data_visualizer(state: AgentState) -> str:
    """pdf and academic modes continue to pdf_generator; all others end."""
    if state.get("mode") in ("pdf", "academic"):
        return "pdf_generator"
    return "suggestions"


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
    if mode == "chart":
        return "chart_writer"
    if mode == "survey":
        return "survey_analyst"
    # research, report, pdf, academic, data_analysis, comparison, debate, email → research pipeline
    return "decomposer"


def route_after_report_writer(state: AgentState) -> str:
    """Report/pdf writers feed the draft_critic before visualisation."""
    return "draft_critic"


def route_after_draft_critic(state: AgentState) -> str:
    """Accept or request a revision from the appropriate writer."""
    feedback  = state.get("revision_feedback", "")
    revisions = state.get("revision_count", 0)
    # revision_count was already incremented by draft_critic this run.
    if not feedback or revisions >= MAX_REVISIONS:
        return "data_visualizer"
    mode = state.get("mode", "report")
    return "academic_writer" if mode == "academic" else "report_writer"


def route_after_chart_writer(state: AgentState) -> str:
    """chart mode (user-supplied data): always ends at suggestions."""
    return "suggestions"


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
    # plan mode debates first; chat/document go straight to suggestions.
    if state.get("mode") == "plan":
        return "challenger"
    return "suggestions"

def route_after_voting_synthesis(_state: AgentState) -> str:
    # voting_synthesis is research mode only — always debates.
    return "challenger"


def route_after_dynamic_spawner(_state: AgentState) -> str:
    # dynamic_spawner is plan mode only — always debates then fact-checks.
    return "challenger"


_RESEARCH_PIPELINE_TARGETS = {
    "research":          "research",
    "validator":         "validator",
    "synthesis":         "synthesis",
    "voting_synthesis":  "voting_synthesis",
    "dynamic_spawner":   "dynamic_spawner",
    "data_extractor":    "data_extractor",
    "report_writer":     "report_writer",
    "academic_writer":   "academic_writer",
    "data_analyst":      "data_analyst",
    "comparison_matrix": "comparison_matrix",
    "debate_writer":     "debate_writer",
    "email_drafter":     "email_drafter",
    "data_visualizer":   "data_visualizer",
    "suggestions":       "suggestions",
}


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(AgentState)

    cp = checkpoint  # alias for readability
    builder.add_node("clarity",           cp(clarity_node))
    builder.add_node("intent_router",     cp(intent_router_node))
    builder.add_node("data_extractor",    cp(data_extractor_node))
    builder.add_node("planner",           cp(planner_node))
    builder.add_node("decomposer",        cp(decomposer_node))
    builder.add_node("research",          cp(research_node))
    builder.add_node("validator",         cp(validator_node))
    builder.add_node("doc_agent",         cp(doc_agent_node))
    builder.add_node("data_analyst",      cp(data_analyst_node))
    builder.add_node("comparison_matrix", cp(comparison_matrix_node))
    builder.add_node("debate_writer",     cp(debate_writer_node))
    builder.add_node("email_drafter",     cp(email_drafter_node))
    builder.add_node("fact_checker",      cp(fact_checker_node))
    builder.add_node("synthesis",         cp(synthesis_node))
    builder.add_node("voting_synthesis",  cp(voting_synthesis_node))
    builder.add_node("dynamic_spawner",   cp(dynamic_spawner_node))
    builder.add_node("challenger",        cp(challenger_node))
    builder.add_node("report_writer",     cp(report_writer_node))
    builder.add_node("academic_writer",   cp(academic_writer_node))
    builder.add_node("draft_critic",      cp(draft_critic_node))
    builder.add_node("data_visualizer",   cp(data_visualizer_node))
    builder.add_node("survey_analyst",    cp(survey_analyst_node))
    builder.add_node("pdf_generator",     cp(pdf_generator_node))
    builder.add_node("chart_writer",      cp(chart_writer_node))
    builder.add_node("code_writer",       cp(code_writer_node))
    builder.add_node("critic",            cp(critic_node))
    builder.add_node("suggestions",       cp(suggestions_node))

    builder.add_edge(START, "clarity")
    builder.add_edge("clarity", "intent_router")
    builder.add_conditional_edges(
        "intent_router", route_after_intent,
        {
            "decomposer":     "decomposer",
            "synthesis":      "synthesis",
            "doc_agent":      "doc_agent",
            "code_writer":    "code_writer",
            "planner":        "planner",
            "chart_writer":   "chart_writer",
            "survey_analyst": "survey_analyst",
        },
    )

    # data_extractor → writer (mode-specific)
    builder.add_conditional_edges(
        "data_extractor", route_after_data_extractor,
        {
            "report_writer":   "report_writer",
            "academic_writer": "academic_writer",
            "data_analyst":    "data_analyst",
            "comparison_matrix": "comparison_matrix",
        },
    )

    # academic_writer → draft_critic (may loop back) → data_visualizer
    builder.add_edge("academic_writer", "draft_critic")

    # plan mode: planner feeds directly into research (it generates sub_queries).
    builder.add_edge("planner", "research")

    # Standard research pipeline.
    builder.add_edge("decomposer", "research")
    builder.add_conditional_edges("research",  route_after_research,  _RESEARCH_PIPELINE_TARGETS)
    builder.add_conditional_edges("validator", route_after_validation, _RESEARCH_PIPELINE_TARGETS)

    # Output nodes.
    builder.add_edge("doc_agent", "synthesis")
    builder.add_conditional_edges(
        "synthesis", route_after_synthesis,
        {"challenger": "challenger", "suggestions": "suggestions"},
    )
    builder.add_conditional_edges(
        "voting_synthesis", route_after_voting_synthesis,
        {"challenger": "challenger"},
    )
    builder.add_conditional_edges(
        "dynamic_spawner", route_after_dynamic_spawner,
        {"challenger": "challenger"},
    )
    builder.add_edge("challenger",        "fact_checker")
    builder.add_edge("fact_checker",      "critic")
    builder.add_edge("critic",            "suggestions")
    builder.add_conditional_edges(
        "report_writer", route_after_report_writer,
        {"draft_critic": "draft_critic"},
    )
    builder.add_conditional_edges(
        "draft_critic", route_after_draft_critic,
        {
            "report_writer":   "report_writer",
            "academic_writer": "academic_writer",
            "data_visualizer": "data_visualizer",
        },
    )
    builder.add_edge("pdf_generator",     "suggestions")

    # data_analysis + comparison → data_visualizer (multi-chart) → suggestions
    builder.add_edge("data_analyst",      "data_visualizer")
    builder.add_edge("comparison_matrix", "data_visualizer")

    # data_visualizer routes to pdf_generator (pdf/academic) or suggestions (rest)
    builder.add_conditional_edges(
        "data_visualizer", route_after_data_visualizer,
        {"pdf_generator": "pdf_generator", "suggestions": "suggestions"},
    )

    # chart mode: single chart from user-supplied data → suggestions
    builder.add_edge("chart_writer",      "suggestions")

    builder.add_edge("survey_analyst",    "suggestions")
    builder.add_edge("debate_writer",     "suggestions")
    builder.add_edge("email_drafter",     "suggestions")
    builder.add_edge("code_writer",       "suggestions")
    builder.add_edge("suggestions",       END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


# Compiled once at import time and reused across requests.
graph = build_graph()
