"""Confidence calibration engine.

Tracks whether the research agent's self-reported confidence scores actually
predict output quality. Over time, the system learns whether it is over- or
under-confident and can recommend a better threshold.

How it works:
  1. After each completed request, record:
       (confidence_score, fact_check_pass_rate, draft_score, mode)
  2. Periodically call `calibration_report()` to analyse:
       - Is high confidence → high quality? (Ideal: yes)
       - Which confidence bucket is most miscalibrated?
       - What threshold minimises false-positives (skipping validation when quality is low)?
  3. Expose via GET /calibration and GET /prompt-health endpoints.

Terminology:
  calibration error = |predicted_quality - observed_quality|
  over-confident    = high confidence_score but low fact_check pass rate
  under-confident   = low confidence_score but high fact_check pass rate (wastes validator calls)
"""

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

from app.config import settings, _data_path

logger = logging.getLogger(__name__)

# Must match ANALYTICS_DB in app/main.py and app/utils/checkpoint.py — all
# three modules write to the same physical ares_analytics.db.
ANALYTICS_DB = _data_path("ares_analytics.db")

_DDL = """
CREATE TABLE IF NOT EXISTS calibration (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id            TEXT NOT NULL,
    mode                 TEXT NOT NULL DEFAULT '',
    confidence_score     INTEGER DEFAULT -1,
    fact_check_pass_rate REAL DEFAULT -1,
    draft_score          INTEGER DEFAULT -1,
    challenger_found     INTEGER DEFAULT -1,  -- 1=counter evidence found, 0=not, -1=not run
    vote_winner          TEXT DEFAULT '',
    latency_ms           INTEGER DEFAULT 0,
    timestamp            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_metrics (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    node       TEXT NOT NULL,
    metric     TEXT NOT NULL,
    value      REAL NOT NULL,
    mode       TEXT DEFAULT '',
    timestamp  TEXT NOT NULL
);
"""

_CONFIDENCE_BUCKETS = [(0, 3, "low"), (4, 6, "medium"), (7, 10, "high")]


def _ensure_tables_sync() -> None:
    with sqlite3.connect(ANALYTICS_DB) as db:
        db.executescript(_DDL)


async def _ensure_tables() -> None:
    async with aiosqlite.connect(ANALYTICS_DB) as db:
        await db.executescript(_DDL)
        await db.commit()


# ── Record calibration data ────────────────────────────────────────────────────

async def record(
    thread_id: str,
    mode: str,
    confidence_score: int,
    fact_check_results: list,
    draft_score: int,
    challenger_found: bool | None,
    vote_winner: str,
    latency_ms: int,
) -> None:
    """Persist quality signals after a completed request."""
    try:
        await _ensure_tables()

        # Compute fact-check pass rate (fraction of claims that passed).
        pass_rate: float = -1.0
        if fact_check_results:
            passed = sum(
                1 for r in fact_check_results
                if isinstance(r, dict) and r.get("result", "").lower() in ("true", "pass", "supported", "verified")
            )
            pass_rate = passed / len(fact_check_results)

        challenger_int = (1 if challenger_found else 0) if challenger_found is not None else -1

        async with aiosqlite.connect(ANALYTICS_DB) as db:
            await db.execute(
                "INSERT INTO calibration "
                "(thread_id, mode, confidence_score, fact_check_pass_rate, draft_score, "
                "challenger_found, vote_winner, latency_ms, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    thread_id, mode, confidence_score, pass_rate, draft_score,
                    challenger_int, vote_winner, latency_ms,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()

        logger.debug(
            f"calibration_recorded thread={thread_id} mode={mode} "
            f"conf={confidence_score} pass_rate={pass_rate:.2f} draft={draft_score}"
        )
    except Exception:
        logger.warning("calibration.record failed", exc_info=True)


async def record_prompt_metric(node: str, metric: str, value: float, mode: str = "") -> None:
    """Record a per-node quality metric (e.g. draft_score, pass_rate)."""
    try:
        await _ensure_tables()
        async with aiosqlite.connect(ANALYTICS_DB) as db:
            await db.execute(
                "INSERT INTO prompt_metrics (node, metric, value, mode, timestamp) VALUES (?,?,?,?,?)",
                (node, metric, value, mode, datetime.now(timezone.utc).isoformat()),
            )
            await db.commit()
    except Exception:
        logger.warning("calibration.record_prompt_metric failed", exc_info=True)


# ── Analysis ───────────────────────────────────────────────────────────────────

async def calibration_report(window: int | None = None) -> dict:
    """Compute calibration statistics over the most recent *window* records.

    Returns:
      bucket_stats  — per bucket: count, avg pass_rate, avg draft_score
      recommended_threshold — threshold that best separates high vs low quality
      is_over_confident — True if high-confidence queries have low pass rate
      calibration_error — mean |confidence/10 - pass_rate| across all records
      total_records     — how many records were analysed
    """
    limit = window or settings.calibration_window
    try:
        await _ensure_tables()
        async with aiosqlite.connect(ANALYTICS_DB) as db:
            cur = await db.execute(
                "SELECT confidence_score, fact_check_pass_rate, draft_score "
                "FROM calibration "
                "WHERE confidence_score >= 0 "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = await cur.fetchall()
    except Exception:
        logger.warning("calibration_report: DB read failed", exc_info=True)
        return {"error": "calibration DB not available", "total_records": 0}

    if not rows:
        return {
            "total_records": 0,
            "message": "No calibration data yet. Data accumulates after each completed research request.",
        }

    bucket_stats = []
    calibration_errors = []

    for lo, hi, label in _CONFIDENCE_BUCKETS:
        bucket_rows = [(r[1], r[2]) for r in rows if lo <= r[0] <= hi and r[1] >= 0]
        count = len(bucket_rows)
        if count == 0:
            bucket_stats.append({
                "bucket": label, "range": f"{lo}–{hi}",
                "count": 0, "avg_pass_rate": None, "avg_draft_score": None,
            })
            continue
        avg_pass  = sum(r[0] for r in bucket_rows) / count
        avg_draft = sum(r[1] for r in bucket_rows if r[1] >= 0) / max(1, sum(1 for r in bucket_rows if r[1] >= 0))
        bucket_stats.append({
            "bucket": label, "range": f"{lo}–{hi}", "count": count,
            "avg_pass_rate": round(avg_pass, 3),
            "avg_draft_score": round(avg_draft, 1) if avg_draft else None,
        })

    # Calibration error: how close is confidence_score/10 to actual pass_rate?
    for r in rows:
        conf, pass_rate, _ = r
        if conf >= 0 and pass_rate >= 0:
            calibration_errors.append(abs(conf / 10.0 - pass_rate))

    mean_cal_error = sum(calibration_errors) / len(calibration_errors) if calibration_errors else None

    # Determine recommended threshold: find the confidence score that best splits
    # high-quality (pass_rate >= 0.7) from low-quality requests.
    best_threshold = settings.confidence_threshold
    best_f1 = -1.0
    for t in range(1, 10):
        tp = sum(1 for r in rows if r[0] >= t and r[1] >= 0.70)
        fp = sum(1 for r in rows if r[0] >= t and 0 <= r[1] < 0.70)
        fn = sum(1 for r in rows if r[0] < t and r[1] >= 0.70)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    # Over-confident = high-confidence bucket has avg_pass_rate < 0.7
    high_bucket = next((b for b in bucket_stats if b["bucket"] == "high"), None)
    is_over_confident = (
        high_bucket is not None
        and high_bucket["avg_pass_rate"] is not None
        and high_bucket["avg_pass_rate"] < 0.70
    )

    return {
        "total_records":         len(rows),
        "bucket_stats":          bucket_stats,
        "current_threshold":     settings.confidence_threshold,
        "recommended_threshold": best_threshold,
        "threshold_changed":     best_threshold != settings.confidence_threshold,
        "mean_calibration_error": round(mean_cal_error, 3) if mean_cal_error is not None else None,
        "is_over_confident":     is_over_confident,
        "diagnosis": (
            "Over-confident: system skips validation too often"
            if is_over_confident
            else "Well-calibrated or under-confident (conservative)"
        ),
    }


async def prompt_health_report(window_hours: int = 24) -> dict:
    """Return per-node quality trends for the last *window_hours* hours.

    High-value signals:
      - fact_checker:  pass_rate trending down → prompts may be too aggressive
      - draft_critic:  draft_score trending down → writer or critic may need tuning
      - challenger:    counter_found_rate → how often the debater finds real disagreement
      - voting_synthesis: winner distribution → are all 3 perspectives competitive?
    """
    try:
        await _ensure_tables()
        async with aiosqlite.connect(ANALYTICS_DB) as db:
            cur = await db.execute(
                "SELECT node, metric, AVG(value) as avg_val, MIN(value), MAX(value), COUNT(*) "
                "FROM prompt_metrics "
                "WHERE timestamp >= datetime('now', ?) "
                "GROUP BY node, metric ORDER BY node, metric",
                (f"-{window_hours} hours",),
            )
            rows = await cur.fetchall()

            # Also pull raw calibration signals for additional context
            cur2 = await db.execute(
                "SELECT "
                "  AVG(CASE WHEN fact_check_pass_rate >= 0 THEN fact_check_pass_rate END) as avg_pass_rate, "
                "  AVG(CASE WHEN draft_score >= 0 THEN draft_score END) as avg_draft_score, "
                "  AVG(CASE WHEN challenger_found >= 0 THEN challenger_found END) as avg_challenger, "
                "  COUNT(*) as total "
                "FROM calibration WHERE timestamp >= datetime('now', ?)",
                (f"-{window_hours} hours",),
            )
            cal_row = await cur2.fetchone()

            # Vote winner distribution
            cur3 = await db.execute(
                "SELECT vote_winner, COUNT(*) FROM calibration "
                "WHERE vote_winner != '' AND timestamp >= datetime('now', ?) "
                "GROUP BY vote_winner",
                (f"-{window_hours} hours",),
            )
            vote_rows = await cur3.fetchall()
    except Exception:
        logger.warning("prompt_health_report: DB read failed", exc_info=True)
        return {"error": "prompt health DB not available"}

    node_metrics: dict = {}
    for node_name, metric, avg_val, min_val, max_val, count in rows:
        node_metrics.setdefault(node_name, {})[metric] = {
            "avg": round(avg_val, 3) if avg_val is not None else None,
            "min": round(min_val, 3) if min_val is not None else None,
            "max": round(max_val, 3) if max_val is not None else None,
            "count": count,
        }

    overall: dict = {}
    if cal_row and cal_row[3]:
        overall = {
            "avg_fact_check_pass_rate":  round(cal_row[0], 3) if cal_row[0] is not None else None,
            "avg_draft_score":           round(cal_row[1], 1) if cal_row[1] is not None else None,
            "avg_challenger_found_rate": round(cal_row[2], 3) if cal_row[2] is not None else None,
            "total_requests":            cal_row[3],
        }

    vote_dist = {r[0]: r[1] for r in vote_rows}

    # Simple health alerts
    alerts = []
    if overall.get("avg_fact_check_pass_rate") is not None and overall["avg_fact_check_pass_rate"] < 0.6:
        alerts.append({
            "severity": "warning",
            "node": "fact_checker",
            "message": f"Fact-check pass rate {overall['avg_fact_check_pass_rate']:.0%} is below 60% — review fact_checker or synthesis prompts.",
        })
    if overall.get("avg_draft_score") is not None and overall["avg_draft_score"] < 5:
        alerts.append({
            "severity": "warning",
            "node": "draft_critic",
            "message": f"Average draft score {overall['avg_draft_score']:.1f}/10 is below 5 — consider tightening writer or critic prompts.",
        })
    if vote_dist:
        total_votes = sum(vote_dist.values())
        dominant = max(vote_dist, key=vote_dist.get)
        if vote_dist[dominant] / total_votes > 0.80:
            alerts.append({
                "severity": "info",
                "node": "voting_synthesis",
                "message": f"'{dominant}' perspective wins {vote_dist[dominant]/total_votes:.0%} of votes — other perspectives may need stronger prompts.",
            })

    return {
        "window_hours":  window_hours,
        "overall":       overall,
        "node_metrics":  node_metrics,
        "vote_distribution": vote_dist,
        "alerts":        alerts,
    }
