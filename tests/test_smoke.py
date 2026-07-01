"""Smoke and routing tests.

These check that the graph compiles and that the routing functions send work to
the right next agent. They run without any network access or API keys.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.graph import build_graph, route_after_research, route_after_validation
from app.config import settings


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    """All 26 agent nodes should be registered in the compiled graph."""
    graph = build_graph()
    nodes = set(graph.get_graph().nodes.keys())
    expected = {
        "clarity", "intent_router", "decomposer", "research", "validator",
        "synthesis", "voting_synthesis", "challenger", "dynamic_spawner",
        "doc_agent", "fact_checker", "critic", "suggestions",
        "report_writer", "draft_critic", "academic_writer",
        "pdf_generator", "chart_writer", "data_analyst", "data_visualizer",
        "survey_analyst", "comparison_matrix", "debate_writer",
        "email_drafter", "code_writer", "planner",
    }
    missing = expected - nodes
    assert not missing, f"Missing nodes: {missing}"


# ---------------------------------------------------------------------------
# route_after_research — confidence-based routing
# ---------------------------------------------------------------------------

def test_high_confidence_routes_to_voting_synthesis():
    # Default mode is "research", which now uses voting_synthesis.
    state = {"confidence_score": 8, "mode": "research"}
    assert route_after_research(state) == "voting_synthesis"


def test_low_confidence_routes_to_validator():
    state = {"confidence_score": 2, "mode": "research"}
    assert route_after_research(state) == "validator"


def test_confidence_at_threshold_routes_to_voting_synthesis():
    """Score exactly at threshold should bypass validator."""
    state = {"confidence_score": settings.confidence_threshold, "mode": "research"}
    assert route_after_research(state) == "voting_synthesis"


def test_confidence_one_below_threshold_routes_to_validator():
    state = {"confidence_score": settings.confidence_threshold - 1, "mode": "research"}
    assert route_after_research(state) == "validator"


def test_zero_confidence_routes_to_validator():
    state = {"confidence_score": 0, "mode": "research"}
    assert route_after_research(state) == "validator"


def test_max_confidence_routes_to_voting_synthesis():
    state = {"confidence_score": 10, "mode": "research"}
    assert route_after_research(state) == "voting_synthesis"


# ---------------------------------------------------------------------------
# route_after_research — mode-aware routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,expected", [
    ("research",      "voting_synthesis"),   # parallel voting
    ("plan",          "dynamic_spawner"),    # findings-aware spawn
    # Visual modes now go through data_extractor before the writer
    ("report",        "data_extractor"),
    ("pdf",           "data_extractor"),
    ("data_analysis", "data_extractor"),
    ("comparison",    "data_extractor"),
    ("academic",      "data_extractor"),
    ("debate",        "debate_writer"),
    ("email",         "email_drafter"),
])
def test_mode_routing_after_research(mode, expected):
    """High-confidence research routes to the correct output node per mode."""
    state = {"confidence_score": 10, "mode": mode}
    assert route_after_research(state) == expected


# ---------------------------------------------------------------------------
# route_after_validation — loop and exit conditions
# ---------------------------------------------------------------------------

def test_sufficient_validation_routes_to_output_node():
    # Validation sufficient for report mode → data_extractor (then report_writer)
    state = {"validation_result": "sufficient", "attempts": 1, "mode": "report"}
    assert route_after_validation(state) == "data_extractor"


def test_sufficient_validation_research_routes_to_voting_synthesis():
    state = {"validation_result": "sufficient", "attempts": 1, "mode": "research"}
    assert route_after_validation(state) == "voting_synthesis"


def test_insufficient_validation_loops_back_to_research():
    state = {"validation_result": "insufficient", "attempts": 1, "mode": "research"}
    assert route_after_validation(state) == "research"


def test_max_attempts_reached_exits_loop():
    """When attempts hit the max, stop looping and proceed to output."""
    state = {"validation_result": "insufficient",
             "attempts": settings.max_validation_attempts, "mode": "report"}
    assert route_after_validation(state) == "data_extractor"


def test_attempts_one_below_max_still_loops():
    state = {"validation_result": "insufficient",
             "attempts": settings.max_validation_attempts - 1, "mode": "research"}
    assert route_after_validation(state) == "research"


def test_sufficient_at_max_attempts_exits_loop():
    state = {"validation_result": "sufficient",
             "attempts": settings.max_validation_attempts, "mode": "research"}
    assert route_after_validation(state) == "voting_synthesis"


# ---------------------------------------------------------------------------
# Mock mode — Tavily returns fake results, no network call
# ---------------------------------------------------------------------------

def test_mock_mode_tavily_returns_fake_results():
    with patch.object(settings, "mock_mode", True):
        from app.tools.search import tavily_search, MOCK_RESULTS
        results = tavily_search("test query")
        assert results == MOCK_RESULTS
        assert len(results) > 0
        assert "content" in results[0]


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

def test_token_budget_not_exceeded_for_short_message():
    from langchain_core.messages import HumanMessage
    short_msg = HumanMessage(content="hi")
    total_tokens = len(short_msg.content) // 4
    assert total_tokens <= settings.max_token_budget


def test_budget_error_helper_returns_none_for_short_messages():
    from app.main import _budget_error
    from langchain_core.messages import HumanMessage
    inputs = {"messages": [HumanMessage(content="Tell me about Stripe")], "attempts": 0}
    assert _budget_error(inputs, "test-thread") is None


def test_budget_error_helper_fires_on_oversize_input():
    from app.main import _budget_error
    from langchain_core.messages import HumanMessage
    huge = HumanMessage(content="x" * (settings.max_token_budget * 4 + 100))
    inputs = {"messages": [huge], "attempts": 0}
    result = _budget_error(inputs, "test-thread")
    assert result is not None
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Truncation utility
# ---------------------------------------------------------------------------

def test_truncate_no_truncation_for_short_text():
    from app.config import truncate_to_budget
    text = "Short content"
    result, was_truncated = truncate_to_budget(text)
    assert result == text
    assert was_truncated is False


def test_truncate_fires_on_large_content():
    from app.config import truncate_to_budget
    oversized = "x" * (settings.max_token_budget * 4)
    result, was_truncated = truncate_to_budget(oversized)
    assert was_truncated is True
    assert len(result) < len(oversized)
    assert "truncated" in result
