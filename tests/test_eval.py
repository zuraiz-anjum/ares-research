"""Evaluation and quality tests.

Covers groundedness, latency, decomposer behaviour, and full mock-mode
end-to-end runs. No API keys or network access required.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.graph import route_after_research, route_after_validation


# ---------------------------------------------------------------------------
# Labeled routing test set
# ---------------------------------------------------------------------------

ROUTING_CASES = [
    # research mode now routes to voting_synthesis (parallel voting)
    (9,  "voting_synthesis", "high confidence skips validator → voting_synthesis"),
    (7,  "voting_synthesis", "above threshold skips validator → voting_synthesis"),
    (6,  "voting_synthesis", "exactly at threshold → voting_synthesis"),
    (5,  "validator",        "below threshold needs validation"),
    (1,  "validator",        "very low confidence needs validation"),
    (0,  "validator",        "zero confidence needs validation"),
]


def test_labeled_routing_cases():
    for score, expected, label in ROUTING_CASES:
        result = route_after_research({"confidence_score": score, "mode": "research"})
        assert result == expected, f"FAILED: {label} — got {result!r}, expected {expected!r}"


VALIDATION_CASES = [
    # research mode exits validation → voting_synthesis
    ("sufficient",   1, "voting_synthesis", "sufficient always exits loop → voting_synthesis"),
    ("sufficient",   3, "voting_synthesis", "sufficient exits even at max attempts → voting_synthesis"),
    ("insufficient", 1, "research",         "insufficient below max retries"),
    ("insufficient", 2, "research",         "insufficient one below max retries"),
    ("insufficient", 3, "voting_synthesis", "insufficient at max exits loop → voting_synthesis"),
]


def test_labeled_validation_cases():
    for result, attempts, expected, label in VALIDATION_CASES:
        actual = route_after_validation({
            "validation_result": result, "attempts": attempts, "mode": "research"
        })
        assert actual == expected, f"FAILED: {label} — got {actual!r}, expected {expected!r}"


# ---------------------------------------------------------------------------
# Groundedness — synthesis uses findings, not raw web scrape
# ---------------------------------------------------------------------------

def test_synthesis_uses_findings_not_raw_research():
    """Synthesis should pass processed findings to the LLM, not raw HTML scrape."""
    from app.agents.synthesis import synthesis_node
    from langchain_core.messages import HumanMessage

    findings = "Stripe raised $600M at a $95B valuation in March 2024."
    raw_research = "Cookie policy. Accept all cookies. Navigation menu." + findings

    captured = {}

    async def fake_ainvoke(messages, **kwargs):
        captured["system_content"] = messages[0].content
        return MagicMock(content="Stripe raised $600M at $95B valuation.")

    mock_llm = MagicMock()
    mock_llm.ainvoke = fake_ainvoke

    with patch("app.agents.synthesis.get_llm", return_value=mock_llm):
        asyncio.run(synthesis_node({
            "mode": "research",
            "findings": findings,
            "raw_research": raw_research,
            "original_query": "Tell me about Stripe's funding",
            "messages": [HumanMessage(content="Tell me about Stripe's funding")],
        }))

    assert findings in captured["system_content"], "findings should be injected into the system prompt"
    assert "Cookie policy" not in captured["system_content"], "raw HTML scrape should not leak into prompt"


def test_synthesis_uses_chat_prompt_for_chat_mode():
    """Chat mode should use CHAT_SYNTHESIS_PROMPT, not the research prompt."""
    from app.agents.synthesis import synthesis_node, CHAT_SYNTHESIS_PROMPT
    from langchain_core.messages import HumanMessage

    captured = {}

    async def fake_ainvoke(messages, **kwargs):
        captured["system_content"] = messages[0].content
        return MagicMock(content="4")

    mock_llm = MagicMock()
    mock_llm.ainvoke = fake_ainvoke

    with patch("app.agents.synthesis.get_llm", return_value=mock_llm):
        asyncio.run(synthesis_node({
            "mode": "chat",
            "messages": [HumanMessage(content="What is 2+2?")],
        }))

    assert captured["system_content"] == CHAT_SYNTHESIS_PROMPT


# ---------------------------------------------------------------------------
# Token budget guard
# ---------------------------------------------------------------------------

def test_token_budget_guard_fires_via_helper():
    """_budget_error should return an error dict when the message is oversized."""
    from app.main import _budget_error
    from langchain_core.messages import HumanMessage

    long_content = "a" * (settings.max_token_budget * 4 + 100)
    inputs = {"messages": [HumanMessage(content=long_content)], "attempts": 0}
    result = _budget_error(inputs, "test-budget-thread")
    assert result is not None
    assert result["status"] == "error"


def test_token_budget_guard_passes_short_message():
    from app.main import _budget_error
    from langchain_core.messages import HumanMessage
    inputs = {"messages": [HumanMessage(content="Tell me about OpenAI")], "attempts": 0}
    assert _budget_error(inputs, "test-short-thread") is None


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

def test_decomposer_splits_compound_query():
    from app.agents.decomposer import decomposer_node

    mock_result = MagicMock()
    mock_result.queries = ["Stripe recent funding", "OpenAI recent funding"]
    mock_result.is_compound = True
    mock_result.has_dependencies = False

    with patch("app.agents.decomposer.get_llm") as mock_llm:
        mock_llm.return_value.with_structured_output.return_value.invoke.return_value = mock_result
        result = decomposer_node({"original_query": "Compare Stripe and OpenAI funding"})

    assert result["sub_queries"] == ["Stripe recent funding", "OpenAI recent funding"]


def test_decomposer_fast_path_for_simple_query():
    """Simple single-entity query should skip the LLM entirely."""
    from app.agents.decomposer import decomposer_node

    with patch("app.agents.decomposer.get_llm") as mock_llm:
        result = decomposer_node({"original_query": "Tell me about Stripe"})
        mock_llm.assert_not_called()

    assert len(result["sub_queries"]) == 1
    assert result["sub_queries"][0] == "Tell me about Stripe"


def test_decomposer_collapses_dependent_queries():
    from app.agents.decomposer import decomposer_node

    mock_result = MagicMock()
    mock_result.queries = ["Stripe funding", "what happened after"]
    mock_result.is_compound = True
    mock_result.has_dependencies = True

    with patch("app.agents.decomposer.get_llm") as mock_llm:
        mock_llm.return_value.with_structured_output.return_value.invoke.return_value = mock_result
        result = decomposer_node({"original_query": "Stripe funding and what happened after"})

    assert len(result["sub_queries"]) == 1


# ---------------------------------------------------------------------------
# Truncation
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
    assert "truncated" in result


# ---------------------------------------------------------------------------
# Full mock-mode end-to-end
# ---------------------------------------------------------------------------

def test_full_graph_runs_in_mock_mode():
    """Full pipeline should complete with zero external API calls in mock mode."""
    from langchain_core.messages import HumanMessage
    from app.graph import build_graph

    with patch.object(settings, "mock_mode", True):
        g = build_graph()
        result = asyncio.run(g.ainvoke(
            {"messages": [HumanMessage(content="Tell me about Stripe")], "attempts": 0},
            {"configurable": {"thread_id": "mock-e2e-test"}},
        ))

    assert "messages" in result
    assert len(result["messages"]) > 0
    assert result["messages"][-1].content


def test_conversation_history_accumulates_across_turns():
    """Regression test: AgentState.messages must use the add_messages reducer.

    Previously this field had no reducer annotation, so LangGraph's default
    "last write wins" merge silently replaced the entire checkpointed
    transcript with just the current turn's message on every new turn —
    every multi-turn conversation lost all memory after one message, in
    every mode, not just an isolated clarity/routing glitch. Runs two turns
    against the same thread_id and checks both that message count grows and
    that turn 1's content survives into turn 2's state.
    """
    from langchain_core.messages import HumanMessage
    from app.graph import build_graph

    with patch.object(settings, "mock_mode", True):
        g = build_graph()
        config = {"configurable": {"thread_id": "history-accum-test"}}

        result1 = asyncio.run(g.ainvoke(
            {"messages": [HumanMessage(content="My favorite color is blue.")], "attempts": 0},
            config,
        ))
        turn1_count = len(result1["messages"])
        assert turn1_count >= 1

        result2 = asyncio.run(g.ainvoke(
            {"messages": [HumanMessage(content="What did I just tell you?")], "attempts": 0},
            config,
        ))

    assert len(result2["messages"]) > turn1_count, (
        "messages did not accumulate across turns — the add_messages reducer "
        "on AgentState.messages may have been removed"
    )
    contents = [str(m.content) for m in result2["messages"]]
    assert any("favorite color is blue" in c for c in contents), (
        "turn 1's message was lost — conversation history is being "
        "overwritten instead of appended"
    )


def test_mock_search_latency():
    """Mock search should complete well under 100ms — no network call."""
    from app.tools.search import tavily_search, MOCK_RESULTS

    with patch.object(settings, "mock_mode", True):
        start = time.time()
        result = tavily_search("Stripe funding")
        elapsed = time.time() - start

    assert elapsed < 0.1, f"Mock search took {elapsed:.3f}s — expected <0.1s"
    assert result == MOCK_RESULTS


def test_groundedness_answer_references_mock_findings():
    """Mock-mode answer should be non-empty and reference expected keywords."""
    from langchain_core.messages import HumanMessage
    from app.graph import build_graph

    with patch.object(settings, "mock_mode", True):
        g = build_graph()
        result = asyncio.run(g.ainvoke(
            {"messages": [HumanMessage(content="Tell me about Stripe")], "attempts": 0},
            {"configurable": {"thread_id": "groundedness-test"}},
        ))

    answer = result["messages"][-1].content.lower()
    assert len(answer) > 0
    assert "mock" in answer or "placeholder" in answer or "company" in answer


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def test_settings_has_langsmith_fields():
    assert hasattr(settings, "langsmith_api_key")
    assert hasattr(settings, "langsmith_project")


def test_settings_has_slack_webhook_field():
    assert hasattr(settings, "slack_webhook_url")


def test_settings_max_token_budget_is_high_enough():
    assert settings.max_token_budget >= 8000, \
        f"max_token_budget={settings.max_token_budget} is too low for multi-company queries"
