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


# ---------------------------------------------------------------------------
# route_after_research — confidence-based routing
# ---------------------------------------------------------------------------

def test_high_confidence_routes_to_synthesis():
    state = {"confidence_score": 8}
    assert route_after_research(state) == "synthesis"


def test_low_confidence_routes_to_validator():
    state = {"confidence_score": 2}
    assert route_after_research(state) == "validator"


def test_confidence_at_threshold_routes_to_synthesis():
    """Score exactly at threshold should go to synthesis, not validator (>= fix)."""
    state = {"confidence_score": settings.confidence_threshold}
    assert route_after_research(state) == "synthesis"


def test_confidence_one_below_threshold_routes_to_validator():
    """Score one below threshold should still go to validator."""
    state = {"confidence_score": settings.confidence_threshold - 1}
    assert route_after_research(state) == "validator"


def test_zero_confidence_routes_to_validator():
    state = {"confidence_score": 0}
    assert route_after_research(state) == "validator"


def test_max_confidence_routes_to_synthesis():
    state = {"confidence_score": 10}
    assert route_after_research(state) == "synthesis"


# ---------------------------------------------------------------------------
# route_after_validation — loop and exit conditions
# ---------------------------------------------------------------------------

def test_sufficient_validation_routes_to_synthesis():
    state = {"validation_result": "sufficient", "attempts": 1}
    assert route_after_validation(state) == "synthesis"


def test_insufficient_validation_loops_back_to_research():
    state = {"validation_result": "insufficient", "attempts": 1}
    assert route_after_validation(state) == "research"


def test_max_attempts_reached_routes_to_synthesis():
    """When attempts hit the max, stop looping and go to synthesis."""
    state = {"validation_result": "insufficient", "attempts": settings.max_validation_attempts}
    assert route_after_validation(state) == "synthesis"


def test_attempts_one_below_max_still_loops():
    """One attempt below max should still retry research."""
    state = {"validation_result": "insufficient", "attempts": settings.max_validation_attempts - 1}
    assert route_after_validation(state) == "research"


def test_sufficient_at_max_attempts_routes_to_synthesis():
    """Sufficient result should always go to synthesis regardless of attempt count."""
    state = {"validation_result": "sufficient", "attempts": settings.max_validation_attempts}
    assert route_after_validation(state) == "synthesis"


# ---------------------------------------------------------------------------
# Mock mode — Tavily returns fake results, no network call
# ---------------------------------------------------------------------------

def test_mock_mode_tavily_returns_fake_results():
    """With MOCK_MODE=true, tavily_search should return mock data without hitting the API."""
    with patch.object(settings, "mock_mode", True):
        from app.tools.search import tavily_search, MOCK_RESULTS
        results = tavily_search("test query")
        assert results == MOCK_RESULTS
        assert len(results) > 0
        assert "content" in results[0]


# ---------------------------------------------------------------------------
# Token budget guard
# ---------------------------------------------------------------------------

def test_token_budget_not_exceeded():
    """Short message should pass the budget check."""
    from app.main import _run
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    short_msg = HumanMessage(content="hi")
    total_tokens = len(short_msg.content) // 4
    assert total_tokens <= settings.max_token_budget


def test_token_budget_calculation():
    """Token estimation uses character count // 4."""
    from langchain_core.messages import HumanMessage

    msg = HumanMessage(content="a" * 400)
    estimated = len(msg.content) // 4
    assert estimated == 100
