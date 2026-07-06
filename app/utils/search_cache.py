"""Search result cache — avoids redundant Tavily API calls.

SQLite-backed, TTL-based cache keyed on a normalised query hash.
Repeated or near-identical queries within the TTL window return instantly
from the local DB without hitting the Tavily API or consuming token budget.

Usage (transparent drop-in for tavily_search):
    from app.utils.search_cache import cached_search
    results = cached_search("OpenAI funding 2025", max_results=5)
"""

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path

from app.config import settings, _data_path

logger = logging.getLogger(__name__)

_CACHE_DB = _data_path("ares_cache.db")

_DDL = """
CREATE TABLE IF NOT EXISTS search_cache (
    query_hash  TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    results     TEXT NOT NULL,      -- JSON array of result dicts
    created_at  REAL NOT NULL,      -- unix timestamp
    hits        INTEGER DEFAULT 0   -- how many times this entry was served from cache
);
CREATE TABLE IF NOT EXISTS search_cache_stats (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event      TEXT NOT NULL,   -- 'hit' | 'miss' | 'evict'
    query_hash TEXT,
    timestamp  TEXT NOT NULL
);
"""


_last_prune: float = 0.0


def _ensure_db() -> None:
    global _last_prune
    with sqlite3.connect(_CACHE_DB) as db:
        # WAL mode allows concurrent readers + one writer without "database is locked".
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(_DDL)
        now = time.time()
        if now - _last_prune > 3600:
            db.execute(
                "DELETE FROM search_cache_stats WHERE timestamp < datetime('now', '-7 days')"
            )
            db.commit()
            _last_prune = now


def _query_hash(query: str) -> str:
    normalised = query.lower().strip()
    return hashlib.sha256(normalised.encode()).hexdigest()[:32]


def get_cached(query: str) -> list[dict] | None:
    """Return cached results for *query*, or None if absent / expired."""
    if not settings.search_cache_enabled:
        return None
    qh  = _query_hash(query)
    ttl = settings.search_cache_ttl_seconds
    try:
        _ensure_db()
        with sqlite3.connect(_CACHE_DB) as db:
            row = db.execute(
                "SELECT results, created_at FROM search_cache WHERE query_hash = ?", (qh,)
            ).fetchone()
            if row is None:
                return None
            results_json, created_at = row
            age = time.time() - created_at
            if age > ttl:
                db.execute("DELETE FROM search_cache WHERE query_hash = ?", (qh,))
                db.execute(
                    "INSERT INTO search_cache_stats (event, query_hash, timestamp) VALUES (?,?,datetime('now'))",
                    ("evict", qh),
                )
                db.commit()
                logger.debug(f"search_cache: evicted stale entry qh={qh} age={age:.0f}s")
                return None
            # Update hit counter
            db.execute("UPDATE search_cache SET hits = hits + 1 WHERE query_hash = ?", (qh,))
            db.execute(
                "INSERT INTO search_cache_stats (event, query_hash, timestamp) VALUES (?,?,datetime('now'))",
                ("hit", qh),
            )
            db.commit()
            logger.info(f"search_cache: HIT qh={qh} age={age:.0f}s query='{query[:60]}'")
            return json.loads(results_json)
    except Exception:
        logger.warning("search_cache: get_cached failed", exc_info=True)
        return None


def store_cached(query: str, results: list[dict]) -> None:
    """Persist search *results* for *query* in the cache."""
    if not settings.search_cache_enabled:
        return
    qh = _query_hash(query)
    try:
        _ensure_db()
        with sqlite3.connect(_CACHE_DB) as db:
            db.execute(
                "INSERT OR REPLACE INTO search_cache (query_hash, query, results, created_at, hits) "
                "VALUES (?, ?, ?, ?, 0)",
                (qh, query, json.dumps(results), time.time()),
            )
            db.execute(
                "INSERT INTO search_cache_stats (event, query_hash, timestamp) VALUES (?,?,datetime('now'))",
                ("miss", qh),
            )
            db.commit()
        logger.debug(f"search_cache: stored qh={qh} results={len(results)} query='{query[:60]}'")
    except Exception:
        logger.warning("search_cache: store_cached failed", exc_info=True)


def cached_search(query: str, max_results: int | None = None) -> list[dict]:
    """Tavily search with transparent caching.  Drop-in replacement."""
    from app.tools.search import tavily_search

    hit = get_cached(query)
    if hit is not None:
        return hit[:max_results] if max_results else hit

    results = tavily_search(query, max_results=max_results)
    store_cached(query, results)
    return results


def cache_stats() -> dict:
    """Return cache hit/miss/evict counts for the last 24 hours."""
    try:
        _ensure_db()
        with sqlite3.connect(_CACHE_DB) as db:
            rows = db.execute(
                "SELECT event, COUNT(*) FROM search_cache_stats "
                "WHERE timestamp >= datetime('now', '-24 hours') GROUP BY event"
            ).fetchall()
            size = db.execute("SELECT COUNT(*) FROM search_cache").fetchone()[0]
        stats = {r[0]: r[1] for r in rows}
        stats["cached_entries"] = size
        return stats
    except Exception:
        return {}


def invalidate(query: str) -> bool:
    """Remove a specific query from the cache. Returns True if it existed."""
    qh = _query_hash(query)
    try:
        _ensure_db()
        with sqlite3.connect(_CACHE_DB) as db:
            n = db.execute("DELETE FROM search_cache WHERE query_hash = ?", (qh,)).rowcount
            db.commit()
        return n > 0
    except Exception:
        return False
