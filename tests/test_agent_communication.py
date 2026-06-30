"""Tests for the three agent communication patterns and error checkpointing.

Covers:
  - challenger      (Research ↔ Synthesis debate)
  - voting_synthesis (parallel agent voting)
  - dynamic_spawner  (dynamic agent spawning)
  - checkpoint       (error_type column + save_sub_agent_error + /errors endpoint)

All LLM and Tavily calls are mocked — no network access required.
"""

import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)


def _make_llm_mock(response_content: str = "Mock LLM response") -> MagicMock:
    """Return a mock LLM whose .ainvoke() resolves to an AIMessage."""
    msg = MagicMock()
    msg.content = response_content
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=msg)
    return mock


def _make_structured_mock(return_value) -> MagicMock:
    """Return a mock for llm.with_structured_output(...).ainvoke(...)."""
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=return_value)
    return structured


# ══════════════════════════════════════════════════════════════════════════════
# challenger — Research ↔ Synthesis debate
# ══════════════════════════════════════════════════════════════════════════════

class TestChallenger:

    def _state(self, draft: str = "A" * 200, findings: str = "Some research findings.") -> dict:
        return {
            "messages":       [AIMessage(content=draft)],
            "original_query": "Tell me about AI",
            "findings":       findings,
        }

    def test_passes_through_when_draft_too_short(self):
        from app.agents.challenger import challenger_node
        state = {
            "messages":       [AIMessage(content="Short answer")],
            "original_query": "test",
            "findings":       "findings",
        }
        result = _run(challenger_node(state))
        assert result == {}

    def test_passes_through_when_not_debatable(self):
        from app.agents.challenger import challenger_node
        from app.agents.challenger import ClaimAnalysis

        not_debatable = ClaimAnalysis(is_debatable=False, debate_points=[])
        structured_mock = _make_structured_mock(not_debatable)
        llm_mock = MagicMock()
        llm_mock.with_structured_output.return_value = structured_mock

        with patch("app.agents.challenger.get_llm", return_value=llm_mock):
            result = _run(challenger_node(self._state()))

        assert result == {"challenge_queries": []}

    def test_reconciles_when_counter_evidence_found(self):
        from app.agents.challenger import challenger_node, ClaimAnalysis, DebatePoint

        claim = DebatePoint(
            claim="AI will take all jobs",
            search_query="AI job replacement evidence counter",
        )
        analysis = ClaimAnalysis(is_debatable=True, debate_points=[claim])

        structured_mock = _make_structured_mock(analysis)
        non_streaming_llm = MagicMock()
        non_streaming_llm.with_structured_output.return_value = structured_mock

        streaming_llm = _make_llm_mock("Nuanced reconciled answer about AI and jobs.")

        def _get_llm(streaming=False, **_):
            return streaming_llm if streaming else non_streaming_llm

        counter_results = [
            {"title": "AI Automation Limits", "url": "http://example.com/1",
             "content": "Studies show AI automates tasks, not whole jobs."},
        ]

        with patch("app.agents.challenger.get_llm", side_effect=_get_llm), \
             patch("app.agents.challenger.tavily_search", return_value=counter_results):
            result = _run(challenger_node(self._state()))

        assert "messages" in result
        assert "Nuanced" in result["messages"][0].content
        assert result["challenge_queries"] == [claim.search_query]
        assert "counter_evidence" in result

    def test_passes_through_when_no_counter_evidence_found(self):
        from app.agents.challenger import challenger_node, ClaimAnalysis, DebatePoint

        claim = DebatePoint(claim="Claim", search_query="counter query")
        analysis = ClaimAnalysis(is_debatable=True, debate_points=[claim])
        structured_mock = _make_structured_mock(analysis)
        non_streaming_llm = MagicMock()
        non_streaming_llm.with_structured_output.return_value = structured_mock

        with patch("app.agents.challenger.get_llm", return_value=non_streaming_llm), \
             patch("app.agents.challenger.tavily_search", return_value=[]):
            result = _run(challenger_node(self._state()))

        # Claims identified but no counter evidence → no reconciled message
        assert "messages" not in result
        assert result.get("challenge_queries") == ["counter query"]

    def test_claim_extraction_failure_returns_empty(self):
        from app.agents.challenger import challenger_node

        structured_mock = MagicMock()
        structured_mock.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        llm_mock = MagicMock()
        llm_mock.with_structured_output.return_value = structured_mock

        with patch("app.agents.challenger.get_llm", return_value=llm_mock):
            result = _run(challenger_node(self._state()))

        assert result == {}


# ══════════════════════════════════════════════════════════════════════════════
# voting_synthesis — parallel agent voting
# ══════════════════════════════════════════════════════════════════════════════

class TestVotingSynthesis:

    def _state(self) -> dict:
        return {
            "original_query": "What is the state of AI?",
            "findings":       "AI is growing. GPT-4 was released. Llama 3 exists.",
        }

    def test_mock_mode_returns_response(self, monkeypatch):
        from app.agents.voting_synthesis import voting_synthesis_node
        from app.config import settings
        monkeypatch.setattr(settings, "mock_mode", True)

        result = _run(voting_synthesis_node(self._state()))

        assert "messages" in result
        assert result["vote_winner"] in ("analyst", "devils_advocate", "communicator")
        assert result["vote_responses"]

    def test_three_variants_run_and_judge_picks_winner(self):
        from app.agents.voting_synthesis import voting_synthesis_node, JudgeVerdict

        variant_response = _make_llm_mock("Variant response about AI.")
        verdict = JudgeVerdict(winner="analyst", reason="Most data-driven answer.")
        judge_structured = _make_structured_mock(verdict)
        judge_llm = MagicMock()
        judge_llm.with_structured_output.return_value = judge_structured

        call_count = {"n": 0}

        def _get_llm(streaming=False, **_):
            call_count["n"] += 1
            # first 3 calls are variants (streaming=False), 4th is judge (streaming=False)
            if judge_llm.with_structured_output.call_count < 1 and call_count["n"] == 4:
                return judge_llm
            return variant_response

        # Patch with a side_effect list: 3 variants + 1 judge
        with patch("app.agents.voting_synthesis.get_llm",
                   side_effect=[variant_response, variant_response, variant_response, judge_llm]):
            result = _run(voting_synthesis_node(self._state()))

        assert "messages" in result
        assert result["vote_winner"] == "analyst"
        assert "perspective chosen" in result["messages"][0].content
        assert "reason" in result["vote_reason"].lower() or result["vote_reason"]

    def test_verdict_footnote_always_appended(self):
        from app.agents.voting_synthesis import voting_synthesis_node, JudgeVerdict

        variant_llm  = _make_llm_mock("Detailed answer here.")
        verdict       = JudgeVerdict(winner="communicator", reason="Clearest narrative.")
        judge_struct  = _make_structured_mock(verdict)
        judge_llm     = MagicMock()
        judge_llm.with_structured_output.return_value = judge_struct

        with patch("app.agents.voting_synthesis.get_llm",
                   side_effect=[variant_llm, variant_llm, variant_llm, judge_llm]):
            result = _run(voting_synthesis_node(self._state()))

        assert "perspective chosen" in result["messages"][0].content

    def test_judge_failure_falls_back_gracefully(self):
        from app.agents.voting_synthesis import voting_synthesis_node

        variant_llm = _make_llm_mock("Fallback analyst answer.")

        judge_struct = MagicMock()
        judge_struct.ainvoke = AsyncMock(side_effect=Exception("judge timeout"))
        judge_llm = MagicMock()
        judge_llm.with_structured_output.return_value = judge_struct

        with patch("app.agents.voting_synthesis.get_llm",
                   side_effect=[variant_llm, variant_llm, variant_llm, judge_llm]):
            result = _run(voting_synthesis_node(self._state()))

        # Should still return something using the fallback winner
        assert "messages" in result
        assert result["messages"][0].content

    def test_all_variants_fail_returns_failure_message(self):
        from app.agents.voting_synthesis import voting_synthesis_node

        failing_llm = MagicMock()
        failing_llm.ainvoke = AsyncMock(side_effect=Exception("API down"))

        with patch("app.agents.voting_synthesis.get_llm", return_value=failing_llm), \
             patch("app.utils.checkpoint.save_sub_agent_error", new_callable=AsyncMock):
            result = _run(voting_synthesis_node(self._state()))

        assert "messages" in result
        assert "(synthesis failed)" in result["messages"][0].content


# ══════════════════════════════════════════════════════════════════════════════
# dynamic_spawner — dynamic agent spawning
# ══════════════════════════════════════════════════════════════════════════════

class TestDynamicSpawner:

    def _state(self, findings: str = "OpenAI raised $6.6B. Revenue grew 200%.") -> dict:
        return {
            "original_query": "Analyse the AI investment landscape",
            "findings":       findings,
        }

    def test_mock_mode_returns_response(self, monkeypatch):
        from app.agents.dynamic_spawner import dynamic_spawner_node
        from app.config import settings
        monkeypatch.setattr(settings, "mock_mode", True)

        result = _run(dynamic_spawner_node(self._state()))

        assert "messages" in result
        assert "spawned_agents" in result
        assert "Agents spawned:" in result["messages"][0].content

    def test_spawns_only_narrative_when_no_numbers_no_entities(self):
        from app.agents.dynamic_spawner import dynamic_spawner_node, SpawnDecision

        decision = SpawnDecision(
            spawn_quantitative=False, spawn_comparative=False,
            rationale="Simple qualitative question."
        )
        struct_mock = _make_structured_mock(decision)
        decider_llm = MagicMock()
        decider_llm.with_structured_output.return_value = struct_mock

        agent_llm = _make_llm_mock("The narrative answer about AI.")

        with patch("app.agents.dynamic_spawner.get_llm",
                   side_effect=[decider_llm, agent_llm]):
            result = _run(dynamic_spawner_node(self._state()))

        assert result["spawned_agents"] == ["narrative"]
        assert "Narrative" in result["messages"][0].content

    def test_spawns_quantitative_when_decision_says_so(self):
        from app.agents.dynamic_spawner import dynamic_spawner_node, SpawnDecision

        decision = SpawnDecision(
            spawn_quantitative=True, spawn_comparative=False,
            rationale="Findings contain revenue figures."
        )
        struct_mock = _make_structured_mock(decision)
        decider_llm = MagicMock()
        decider_llm.with_structured_output.return_value = struct_mock

        agent_llm   = _make_llm_mock("Agent response content.")
        weaver_llm  = _make_llm_mock("Woven comprehensive response.")

        # decider + 2 agents (narrative, quantitative) + weaver
        with patch("app.agents.dynamic_spawner.get_llm",
                   side_effect=[decider_llm, agent_llm, agent_llm, weaver_llm]):
            result = _run(dynamic_spawner_node(self._state()))

        assert set(result["spawned_agents"]) == {"narrative", "quantitative"}
        assert "Narrative, Quantitative" in result["messages"][0].content

    def test_spawns_all_three_for_multi_entity_numerical(self):
        from app.agents.dynamic_spawner import dynamic_spawner_node, SpawnDecision

        decision = SpawnDecision(
            spawn_quantitative=True, spawn_comparative=True,
            rationale="Multiple companies + financial data."
        )
        struct_mock = _make_structured_mock(decision)
        decider_llm = MagicMock()
        decider_llm.with_structured_output.return_value = struct_mock

        agent_llm  = _make_llm_mock("Agent content.")
        weaver_llm = _make_llm_mock("Combined woven answer.")

        # decider + 3 agents + weaver = 5 calls
        with patch("app.agents.dynamic_spawner.get_llm",
                   side_effect=[decider_llm, agent_llm, agent_llm, agent_llm, weaver_llm]):
            result = _run(dynamic_spawner_node(self._state()))

        assert set(result["spawned_agents"]) == {"narrative", "quantitative", "comparative"}

    def test_spawn_decision_failure_defaults_to_narrative_only(self):
        from app.agents.dynamic_spawner import dynamic_spawner_node

        failing_struct = MagicMock()
        failing_struct.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        decider_llm = MagicMock()
        decider_llm.with_structured_output.return_value = failing_struct

        agent_llm = _make_llm_mock("Narrative fallback response.")

        with patch("app.agents.dynamic_spawner.get_llm",
                   side_effect=[decider_llm, agent_llm]):
            result = _run(dynamic_spawner_node(self._state()))

        assert result["spawned_agents"] == ["narrative"]

    def test_agents_footnote_always_present(self):
        from app.agents.dynamic_spawner import dynamic_spawner_node, SpawnDecision

        decision = SpawnDecision(spawn_quantitative=False, spawn_comparative=False,
                                 rationale="Simple query.")
        struct_mock = _make_structured_mock(decision)
        decider_llm = MagicMock()
        decider_llm.with_structured_output.return_value = struct_mock
        agent_llm = _make_llm_mock("Answer content.")

        with patch("app.agents.dynamic_spawner.get_llm",
                   side_effect=[decider_llm, agent_llm]):
            result = _run(dynamic_spawner_node(self._state()))

        assert "Agents spawned:" in result["messages"][0].content


# ══════════════════════════════════════════════════════════════════════════════
# Error checkpoint
# ══════════════════════════════════════════════════════════════════════════════

class TestErrorCheckpoint:

    def _make_db(self, tmp_path) -> str:
        db_path = str(tmp_path / "test.db")
        from app.utils.checkpoint import _TABLE_DDL, _MIGRATION_DDL
        with sqlite3.connect(db_path) as db:
            db.execute(_TABLE_DDL)
            try:
                db.execute(_MIGRATION_DDL)
            except Exception:
                pass
        return db_path

    def test_error_type_saved_when_node_fails(self, tmp_path):
        from app.utils import checkpoint as cp_mod

        db_path = self._make_db(tmp_path)

        async def _failing_node(state):
            raise ValueError("bad input value")

        with patch.object(cp_mod, "ANALYTICS_DB", db_path), \
             patch.object(cp_mod, "_current_thread_id") as tid_mock:
            tid_mock.get.return_value = "test-thread-001"
            wrapped = cp_mod.checkpoint(_failing_node)
            try:
                _run(wrapped({"mode": "research", "messages": []}))
            except ValueError:
                pass

        with sqlite3.connect(db_path) as db:
            rows = db.execute(
                "SELECT status, error_type, error FROM node_checkpoints WHERE status='failed'"
            ).fetchall()

        assert rows, "No failed checkpoint recorded"
        status, error_type, error = rows[0]
        assert status == "failed"
        assert error_type == "ValueError"
        assert "bad input value" in error

    def test_save_sub_agent_error_writes_record(self, tmp_path):
        from app.utils import checkpoint as cp_mod

        db_path = self._make_db(tmp_path)

        async def _run_it():
            with patch.object(cp_mod, "ANALYTICS_DB", db_path), \
                 patch.object(cp_mod, "_current_thread_id") as tid_mock:
                tid_mock.get.return_value = "thread-sub-001"
                await cp_mod.save_sub_agent_error(
                    "voting_synthesis", "analyst", TimeoutError("variant timed out")
                )

        _run(_run_it())

        with sqlite3.connect(db_path) as db:
            rows = db.execute(
                "SELECT node, status, error_type, error FROM node_checkpoints WHERE status='failed'"
            ).fetchall()

        assert rows
        node, status, error_type, error = rows[0]
        assert node == "voting_synthesis"
        assert error_type == "TimeoutError"
        assert error  # traceback stored

    def test_completed_nodes_have_no_error_type(self, tmp_path):
        from app.utils import checkpoint as cp_mod

        db_path = self._make_db(tmp_path)

        async def _ok_node(state):
            return {"mode": "research"}

        with patch.object(cp_mod, "ANALYTICS_DB", db_path), \
             patch.object(cp_mod, "_current_thread_id") as tid_mock:
            tid_mock.get.return_value = "thread-ok"
            wrapped = cp_mod.checkpoint(_ok_node)
            _run(wrapped({"mode": "research", "messages": []}))

        with sqlite3.connect(db_path) as db:
            rows = db.execute(
                "SELECT status, error_type FROM node_checkpoints WHERE status='completed'"
            ).fetchall()

        assert rows
        assert all(r[1] is None for r in rows), "Completed rows should have no error_type"


# ══════════════════════════════════════════════════════════════════════════════
# Graph routing — new nodes registered
# ══════════════════════════════════════════════════════════════════════════════

class TestGraphRouting:

    def test_graph_has_all_new_nodes(self):
        from app.graph import build_graph
        graph = build_graph()
        nodes = set(graph.get_graph().nodes.keys())
        new_nodes = {"challenger", "voting_synthesis", "dynamic_spawner", "draft_critic",
                     "academic_writer", "data_visualizer", "survey_analyst"}
        missing = new_nodes - nodes
        assert not missing, f"Missing graph nodes: {missing}"

    def test_research_mode_routes_to_voting_synthesis(self):
        from app.graph import _after_research_pipeline
        state = {"mode": "research", "confidence_score": 9}
        assert _after_research_pipeline(state) == "voting_synthesis"

    def test_plan_mode_routes_to_dynamic_spawner(self):
        from app.graph import _after_research_pipeline
        state = {"mode": "plan", "confidence_score": 9}
        assert _after_research_pipeline(state) == "dynamic_spawner"

    def test_synthesis_routes_to_challenger_for_plan_mode(self):
        from app.graph import route_after_synthesis
        assert route_after_synthesis({"mode": "plan"}) == "challenger"

    def test_synthesis_routes_to_suggestions_for_chat_mode(self):
        from app.graph import route_after_synthesis
        assert route_after_synthesis({"mode": "chat"}) == "suggestions"

    def test_draft_critic_routes_to_data_visualizer_when_accepted(self):
        from app.graph import route_after_draft_critic
        from app.agents.draft_critic import MAX_REVISIONS
        # Score accepted (empty feedback)
        state = {"revision_feedback": "", "revision_count": 1, "mode": "report"}
        assert route_after_draft_critic(state) == "data_visualizer"

    def test_draft_critic_routes_back_to_report_writer_when_feedback(self):
        from app.graph import route_after_draft_critic
        state = {"revision_feedback": "Section 2 is thin.", "revision_count": 1, "mode": "report"}
        assert route_after_draft_critic(state) == "report_writer"

    def test_draft_critic_forces_accept_at_max_revisions(self):
        from app.graph import route_after_draft_critic
        from app.agents.draft_critic import MAX_REVISIONS
        state = {"revision_feedback": "Still needs work.", "revision_count": MAX_REVISIONS, "mode": "report"}
        assert route_after_draft_critic(state) == "data_visualizer"
