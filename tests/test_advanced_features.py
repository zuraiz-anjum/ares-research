"""Tests for the four advanced features added in the final sprint.

Covers:
  1. Search result cache  (app/utils/search_cache.py)
  2. Cost-aware routing   (app/utils/cost_tracker.py)
  3. Confidence calibration (app/utils/calibration.py)
  4. Prompt health reporting (app/utils/calibration.py)

All tests use temporary SQLite DBs and patched settings so no real DB files
are modified and no network calls are made.
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings


# ── helpers ────────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


# ==============================================================================
# 1. Search cache
# ==============================================================================

class TestSearchCache:
    def test_store_and_retrieve(self, tmp_path, monkeypatch):
        from app.utils import search_cache as sc
        monkeypatch.setattr(sc, "_CACHE_DB", str(tmp_path / "cache.db"))
        monkeypatch.setattr(settings, "search_cache_enabled", True)
        monkeypatch.setattr(settings, "search_cache_ttl_seconds", 3600)

        results = [{"title": "Test", "content": "hello", "url": "https://example.com"}]
        sc.store_cached("openai funding", results)
        hit = sc.get_cached("openai funding")
        assert hit is not None
        assert hit[0]["title"] == "Test"

    def test_cache_miss_returns_none(self, tmp_path, monkeypatch):
        from app.utils import search_cache as sc
        monkeypatch.setattr(sc, "_CACHE_DB", str(tmp_path / "cache.db"))
        monkeypatch.setattr(settings, "search_cache_enabled", True)
        monkeypatch.setattr(settings, "search_cache_ttl_seconds", 3600)

        hit = sc.get_cached("query that was never stored")
        assert hit is None

    def test_expired_entry_returns_none(self, tmp_path, monkeypatch):
        from app.utils import search_cache as sc
        monkeypatch.setattr(sc, "_CACHE_DB", str(tmp_path / "cache.db"))
        monkeypatch.setattr(settings, "search_cache_enabled", True)
        monkeypatch.setattr(settings, "search_cache_ttl_seconds", 1)  # 1-second TTL

        sc.store_cached("stale query", [{"title": "old", "url": "x"}])
        time.sleep(1.1)
        hit = sc.get_cached("stale query")
        assert hit is None

    def test_disabled_cache_always_returns_none(self, tmp_path, monkeypatch):
        from app.utils import search_cache as sc
        monkeypatch.setattr(sc, "_CACHE_DB", str(tmp_path / "cache.db"))
        monkeypatch.setattr(settings, "search_cache_enabled", False)

        sc.store_cached("anything", [{"url": "x"}])
        assert sc.get_cached("anything") is None

    def test_normalisation_ignores_case_and_whitespace(self, tmp_path, monkeypatch):
        from app.utils import search_cache as sc
        monkeypatch.setattr(sc, "_CACHE_DB", str(tmp_path / "cache.db"))
        monkeypatch.setattr(settings, "search_cache_enabled", True)
        monkeypatch.setattr(settings, "search_cache_ttl_seconds", 3600)

        sc.store_cached("OpenAI", [{"title": "a"}])
        # same key, different capitalisation → should be a cache hit
        hit = sc.get_cached("openai")
        assert hit is not None

    def test_cached_search_calls_tavily_on_miss(self, tmp_path, monkeypatch):
        from app.utils import search_cache as sc
        monkeypatch.setattr(sc, "_CACHE_DB", str(tmp_path / "cache.db"))
        monkeypatch.setattr(settings, "search_cache_enabled", True)
        monkeypatch.setattr(settings, "search_cache_ttl_seconds", 3600)

        fake_results = [{"title": "live", "url": "https://live.example.com"}]
        with patch("app.tools.search.tavily_search", return_value=fake_results) as mock_tv:
            result = sc.cached_search("brand new query")
        mock_tv.assert_called_once_with("brand new query", max_results=None)
        assert result == fake_results

    def test_cached_search_skips_tavily_on_hit(self, tmp_path, monkeypatch):
        from app.utils import search_cache as sc
        monkeypatch.setattr(sc, "_CACHE_DB", str(tmp_path / "cache.db"))
        monkeypatch.setattr(settings, "search_cache_enabled", True)
        monkeypatch.setattr(settings, "search_cache_ttl_seconds", 3600)

        sc.store_cached("cached query", [{"title": "cached"}])
        with patch("app.tools.search.tavily_search") as mock_tv:
            result = sc.cached_search("cached query")
        mock_tv.assert_not_called()
        assert result[0]["title"] == "cached"

    def test_cache_stats_counts_hits_and_misses(self, tmp_path, monkeypatch):
        from app.utils import search_cache as sc
        monkeypatch.setattr(sc, "_CACHE_DB", str(tmp_path / "cache.db"))
        monkeypatch.setattr(settings, "search_cache_enabled", True)
        monkeypatch.setattr(settings, "search_cache_ttl_seconds", 3600)

        sc.store_cached("q1", [{"title": "r1"}])  # 1 miss recorded
        sc.get_cached("q1")                        # 1 hit recorded
        sc.get_cached("q1")                        # 2nd hit
        stats = sc.cache_stats()
        assert stats.get("miss", 0) >= 1
        assert stats.get("hit", 0) >= 2

    def test_invalidate_removes_entry(self, tmp_path, monkeypatch):
        from app.utils import search_cache as sc
        monkeypatch.setattr(sc, "_CACHE_DB", str(tmp_path / "cache.db"))
        monkeypatch.setattr(settings, "search_cache_enabled", True)
        monkeypatch.setattr(settings, "search_cache_ttl_seconds", 3600)

        sc.store_cached("to-remove", [{"title": "x"}])
        removed = sc.invalidate("to-remove")
        assert removed is True
        assert sc.get_cached("to-remove") is None


# ==============================================================================
# 2. Cost-aware routing / budget guard
# ==============================================================================

class TestCostTracker:
    def test_estimate_query_tokens_increases_with_length(self):
        from app.utils.cost_tracker import estimate_query_tokens
        short = estimate_query_tokens("hi")
        long  = estimate_query_tokens("x" * 4000)
        assert long > short

    def test_estimate_is_higher_for_expensive_modes(self):
        from app.utils.cost_tracker import estimate_query_tokens
        plan_est     = estimate_query_tokens("query", mode="plan")
        chat_est     = estimate_query_tokens("query", mode="chat")
        assert plan_est > chat_est

    def test_session_token_limit_is_positive(self):
        from app.utils.cost_tracker import session_token_limit
        limit = session_token_limit()
        assert limit > 0

    def test_budget_summary_fields(self):
        from app.utils.cost_tracker import budget_summary
        s = budget_summary(1000)
        assert "tokens_used" in s
        assert "token_limit" in s
        assert "cost_used_usd" in s
        assert "remaining_usd" in s
        assert s["tokens_used"] == 1000

    def test_check_cost_budget_passes_when_thread_is_new(self, tmp_path):
        """A brand-new thread (0 tokens so far) should never be blocked."""
        from app.utils.cost_tracker import check_cost_budget
        import sqlite3
        db_path = str(tmp_path / "sessions.db")
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE sessions (thread_id TEXT, token_count INTEGER)")
            db.commit()
        result = _run(check_cost_budget("new-thread", "Tell me about Stripe", db_path))
        assert result is None

    def test_check_cost_budget_fires_when_limit_exceeded(self, tmp_path, monkeypatch):
        """Thread with massive token history should trigger the budget guard."""
        from app.utils.cost_tracker import check_cost_budget
        import sqlite3
        monkeypatch.setattr(settings, "session_cost_budget_usd", 0.001)   # tiny budget
        monkeypatch.setattr(settings, "cost_per_1k_tokens_usd", 0.001)
        db_path = str(tmp_path / "sessions.db")
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE sessions (thread_id TEXT, token_count INTEGER)")
            db.execute("INSERT INTO sessions VALUES ('busy-thread', 5000)")  # already used 5000 tokens
            db.commit()

        result = _run(check_cost_budget("busy-thread", "Another query", db_path))
        assert result is not None
        assert result["status"] == "error"
        assert "budget" in result["answer"].lower()

    def test_check_cost_budget_logs_warning_on_fire(self, tmp_path, monkeypatch, caplog):
        import logging
        import sqlite3
        from app.utils.cost_tracker import check_cost_budget
        monkeypatch.setattr(settings, "session_cost_budget_usd", 0.001)
        monkeypatch.setattr(settings, "cost_per_1k_tokens_usd", 0.001)
        db_path = str(tmp_path / "sessions.db")
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE sessions (thread_id TEXT, token_count INTEGER)")
            db.execute("INSERT INTO sessions VALUES ('t', 9999)")
            db.commit()
        with caplog.at_level(logging.WARNING):
            _run(check_cost_budget("t", "x", db_path))
        assert any("cost_budget_guard_fired" in r.message for r in caplog.records)


# ==============================================================================
# 3. Confidence calibration
# ==============================================================================

class TestCalibration:
    def _patch_db(self, monkeypatch, tmp_path):
        from app.utils import calibration as cal
        monkeypatch.setattr(cal, "ANALYTICS_DB", str(tmp_path / "cal.db"))

    def test_record_stores_row(self, tmp_path, monkeypatch):
        from app.utils import calibration as cal
        self._patch_db(monkeypatch, tmp_path)
        _run(cal.record(
            thread_id="t1", mode="research", confidence_score=8,
            fact_check_results=[{"result": "true"}, {"result": "true"}],
            draft_score=7, challenger_found=True, vote_winner="analyst", latency_ms=1200,
        ))
        import sqlite3
        with sqlite3.connect(str(tmp_path / "cal.db")) as db:
            count = db.execute("SELECT COUNT(*) FROM calibration").fetchone()[0]
        assert count == 1

    def test_calibration_report_empty_db(self, tmp_path, monkeypatch):
        from app.utils import calibration as cal
        self._patch_db(monkeypatch, tmp_path)
        report = _run(cal.calibration_report())
        assert report["total_records"] == 0

    def test_calibration_report_with_data(self, tmp_path, monkeypatch):
        from app.utils import calibration as cal
        self._patch_db(monkeypatch, tmp_path)

        # Seed several records: high confidence but low pass rate (over-confident)
        for _ in range(5):
            _run(cal.record(
                "t", "research", confidence_score=9,
                fact_check_results=[{"result": "false"}, {"result": "false"}],
                draft_score=4, challenger_found=False, vote_winner="", latency_ms=1000,
            ))
        report = _run(cal.calibration_report())
        assert report["total_records"] == 5
        assert report["is_over_confident"] is True
        assert "bucket_stats" in report
        assert "recommended_threshold" in report

    def test_calibration_report_well_calibrated(self, tmp_path, monkeypatch):
        from app.utils import calibration as cal
        self._patch_db(monkeypatch, tmp_path)

        # High confidence + high pass rate → well calibrated
        for _ in range(5):
            _run(cal.record(
                "t", "research", confidence_score=8,
                fact_check_results=[{"result": "true"}, {"result": "true"}],
                draft_score=8, challenger_found=True, vote_winner="analyst", latency_ms=800,
            ))
        report = _run(cal.calibration_report())
        assert report["is_over_confident"] is False

    def test_record_prompt_metric(self, tmp_path, monkeypatch):
        from app.utils import calibration as cal
        self._patch_db(monkeypatch, tmp_path)
        _run(cal.record_prompt_metric("draft_critic", "draft_score", 7.5, "research"))
        import sqlite3
        with sqlite3.connect(str(tmp_path / "cal.db")) as db:
            row = db.execute("SELECT node, metric, value FROM prompt_metrics").fetchone()
        assert row[0] == "draft_critic"
        assert row[1] == "draft_score"
        assert row[2] == pytest.approx(7.5)


# ==============================================================================
# 4. Prompt health report
# ==============================================================================

class TestPromptHealth:
    def _patch_db(self, monkeypatch, tmp_path):
        from app.utils import calibration as cal
        monkeypatch.setattr(cal, "ANALYTICS_DB", str(tmp_path / "cal.db"))

    def test_empty_db_returns_no_alerts(self, tmp_path, monkeypatch):
        from app.utils import calibration as cal
        self._patch_db(monkeypatch, tmp_path)
        report = _run(cal.prompt_health_report())
        assert "alerts" in report
        assert report["alerts"] == []

    def test_low_pass_rate_generates_alert(self, tmp_path, monkeypatch):
        from app.utils import calibration as cal
        self._patch_db(monkeypatch, tmp_path)

        # Seed low pass-rate data and a bad draft score
        for _ in range(5):
            _run(cal.record(
                "t", "research", confidence_score=5,
                fact_check_results=[{"result": "false"}, {"result": "false"}, {"result": "false"}],
                draft_score=3, challenger_found=False, vote_winner="", latency_ms=1000,
            ))
        report = _run(cal.prompt_health_report(window_hours=24 * 365))  # large window
        assert any(a["node"] == "fact_checker" for a in report["alerts"])
        assert any(a["node"] == "draft_critic"  for a in report["alerts"])

    def test_dominant_vote_winner_generates_alert(self, tmp_path, monkeypatch):
        from app.utils import calibration as cal
        self._patch_db(monkeypatch, tmp_path)

        # Seed vote data where analyst wins every time
        for _ in range(10):
            _run(cal.record(
                "t", "research", confidence_score=7,
                fact_check_results=[{"result": "true"}],
                draft_score=7, challenger_found=True, vote_winner="analyst", latency_ms=1000,
            ))
        report = _run(cal.prompt_health_report(window_hours=24 * 365))
        assert any(a["node"] == "voting_synthesis" for a in report["alerts"])

    def test_prompt_metrics_surface_in_health_report(self, tmp_path, monkeypatch):
        from app.utils import calibration as cal
        self._patch_db(monkeypatch, tmp_path)

        _run(cal.record_prompt_metric("draft_critic", "draft_score", 6.0, "research"))
        _run(cal.record_prompt_metric("fact_checker", "pass_rate", 0.8, "research"))

        report = _run(cal.prompt_health_report(window_hours=24 * 365))
        assert "draft_critic" in report["node_metrics"]
        assert "fact_checker" in report["node_metrics"]
