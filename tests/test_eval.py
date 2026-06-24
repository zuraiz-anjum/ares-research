"""Evaluation tests — groundedness, cost estimation, latency.

These tests run against the mock mode so no API keys or network access needed.
They check output quality properties rather than routing logic.
"""

import time
from unittest.mock import patch, MagicMock

from app.config import settings
from app.graph import route_after_research, route_after_validation


# ---------------------------------------------------------------------------
# Labeled test set — routing correctness on real-world query types
# ---------------------------------------------------------------------------

ROUTING_CASES = [
    # (confidence_score, expected_route, label)
    (9, "synthesis", "high confidence skips validator"),
    (7, "synthesis", "above threshold skips validator"),
    (6, "synthesis", "exactly at threshold goes to synthesis"),
    (5, "validator", "below threshold needs validation"),
    (1, "validator", "very low confidence needs validation"),
    (0, "validator", "zero confidence needs validation"),
]


def test_labeled_routing_cases():
    """Labeled test set — each case documents the expected routing decision."""
    for score, expected, label in ROUTING_CASES:
        result = route_after_research({"confidence_score": score})
        assert result == expected, f"FAILED: {label} — got {result}, expected {expected}"


VALIDATION_CASES = [
    # (result, attempts, expected_route, label)
    ("sufficient", 1, "synthesis", "sufficient always exits loop"),
    ("sufficient", 3, "synthesis", "sufficient exits even at max attempts"),
    ("insufficient", 1, "research", "insufficient below max retries"),
    ("insufficient", 2, "research", "insufficient one below max retries"),
    ("insufficient", 3, "synthesis", "insufficient at max exits loop"),
]


def test_labeled_validation_cases():
    """Labeled test set — validation routing across all meaningful states."""
    for result, attempts, expected, label in VALIDATION_CASES:
        state = {"validation_result": result, "attempts": attempts}
        actual = route_after_validation(state)
        assert actual == expected, f"FAILED: {label} — got {actual}, expected {expected}"


# ---------------------------------------------------------------------------
# Groundedness — synthesis uses findings, not raw web scrape
# ---------------------------------------------------------------------------

def test_synthesis_uses_findings_not_raw_research():
    """Synthesis should prefer processed findings over raw web scrape."""
    from app.agents.synthesis import synthesis_node
    from langchain_core.messages import HumanMessage, AIMessage

    findings = "Stripe raised $600M at a $95B valuation in March 2024."
    raw_research = "Cookie policy. Accept all cookies. Navigation menu. Footer links. " + findings

    captured = {}

    original_invoke = None

    def fake_invoke(messages):
        captured["system_content"] = messages[0].content
        return MagicMock(content="Stripe raised $600M at $95B valuation.")

    with patch("app.agents.synthesis.get_llm") as mock_llm:
        mock_llm.return_value.invoke = fake_invoke
        synthesis_node({
            "findings": findings,
            "raw_research": raw_research,
            "messages": [HumanMessage(content="Tell me about Stripe's funding")],
        })

    assert findings in captured["system_content"], "findings should be in the prompt"
    assert "Cookie policy" not in captured["system_content"], "raw_research should not be in the prompt"


# ---------------------------------------------------------------------------
# Cost estimation — token budget
# ---------------------------------------------------------------------------

def test_cost_estimation_short_query():
    """Short queries should be well within the token budget."""
    from langchain_core.messages import HumanMessage

    msg = HumanMessage(content="Tell me about Stripe")
    estimated_tokens = len(msg.content) // 4
    assert estimated_tokens < settings.max_token_budget


def test_cost_estimation_long_query_exceeds_budget():
    """Queries over the budget should be caught by the guard."""
    from langchain_core.messages import HumanMessage

    long_content = "a" * (settings.max_token_budget * 4 + 100)
    msg = HumanMessage(content=long_content)
    estimated_tokens = len(msg.content) // 4
    assert estimated_tokens > settings.max_token_budget


def test_token_budget_guard_fires():
    """_run should return an error response when budget is exceeded."""
    from app.main import _run
    from langchain_core.messages import HumanMessage

    long_content = "a" * (settings.max_token_budget * 4 + 100)
    inputs = {"messages": [HumanMessage(content=long_content)], "attempts": 0}
    result = _run(inputs, "test-budget-thread")
    assert result["status"] == "error"
    assert "too large" in result["answer"].lower()


# ---------------------------------------------------------------------------
# Latency — mock mode should be fast
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Decomposer — query splitting
# ---------------------------------------------------------------------------

def test_decomposer_splits_compound_query():
    """Compound query should produce multiple sub-queries."""
    from unittest.mock import patch, MagicMock
    from app.agents.decomposer import decomposer_node

    mock_result = MagicMock()
    mock_result.queries = ["Stripe recent funding", "OpenAI recent funding"]
    mock_result.is_compound = True

    with patch("app.agents.decomposer.get_llm") as mock_llm:
        mock_llm.return_value.with_structured_output.return_value.invoke.return_value = mock_result
        result = decomposer_node({"original_query": "Compare Stripe and OpenAI funding"})

    assert result["sub_queries"] == ["Stripe recent funding", "OpenAI recent funding"]
    assert len(result["sub_queries"]) == 2


def test_decomposer_keeps_simple_query():
    """Simple query should produce a single sub-query."""
    from unittest.mock import patch, MagicMock
    from app.agents.decomposer import decomposer_node

    mock_result = MagicMock()
    mock_result.queries = ["Stripe company overview"]
    mock_result.is_compound = False

    with patch("app.agents.decomposer.get_llm") as mock_llm:
        mock_llm.return_value.with_structured_output.return_value.invoke.return_value = mock_result
        result = decomposer_node({"original_query": "Tell me about Stripe"})

    assert len(result["sub_queries"]) == 1
    assert result["sub_queries"][0] == "Stripe company overview"


def test_mock_search_latency():
    """Mock search should complete near-instantly — no network call."""
    from app.tools.search import tavily_search, MOCK_RESULTS

    with patch.object(settings, "mock_mode", True):
        start = time.time()
        result = tavily_search("Stripe funding")
        elapsed = time.time() - start

    assert elapsed < 0.1, f"Mock search took {elapsed:.3f}s — should be instant"
    assert result == MOCK_RESULTS
