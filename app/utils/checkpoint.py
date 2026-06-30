"""Node checkpoint system.

Wraps every LangGraph node with structured entry/exit/error logging and
persists each checkpoint to the `node_checkpoints` table in ares_analytics.db.

Usage (graph.py):
    from app.utils.checkpoint import checkpoint
    builder.add_node("research", checkpoint(research_node))

Usage (before graph invocation in main.py):
    from app.utils.checkpoint import set_thread_id
    set_thread_id(thread_id)
"""

import asyncio
import contextvars
import functools
import json
import logging
import sqlite3
import time
import traceback
from datetime import datetime, timezone

import aiosqlite

logger = logging.getLogger(__name__)

ANALYTICS_DB = "ares_analytics.db"

# Propagated automatically to child coroutines/tasks by Python's asyncio.
_current_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar("cp_thread_id", default="-")


def set_thread_id(tid: str) -> None:
    _current_thread_id.set(tid)


# ── DB helpers ──────────────────────────────────────────────────────────────

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS node_checkpoints (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id      TEXT NOT NULL,
    node           TEXT NOT NULL,
    status         TEXT NOT NULL,      -- started | completed | failed | interrupted
    input_summary  TEXT,               -- JSON compact state snapshot
    output_summary TEXT,               -- JSON compact output snapshot
    error          TEXT,               -- full traceback on failed
    error_type     TEXT,               -- exception class name (e.g. "ValueError")
    duration_ms    INTEGER,
    timestamp      TEXT NOT NULL
)
"""

# Add error_type column to existing DBs that were created before it existed.
_MIGRATION_DDL = "ALTER TABLE node_checkpoints ADD COLUMN error_type TEXT"


def _ensure_table_sync() -> None:
    try:
        with sqlite3.connect(ANALYTICS_DB) as db:
            db.execute(_TABLE_DDL)
            try:
                db.execute(_MIGRATION_DDL)
                db.commit()
            except Exception:
                pass  # column already exists
    except Exception:
        pass


def _save_sync(
    thread_id: str,
    node: str,
    status: str,
    input_summary: dict | None = None,
    output_summary: dict | None = None,
    error: str | None = None,
    error_type: str | None = None,
    duration_ms: int | None = None,
) -> None:
    try:
        with sqlite3.connect(ANALYTICS_DB) as db:
            db.execute(_TABLE_DDL)
            try:
                db.execute(_MIGRATION_DDL); db.commit()
            except Exception:
                pass
            db.execute(
                "INSERT INTO node_checkpoints "
                "(thread_id, node, status, input_summary, output_summary, error, error_type, duration_ms, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    thread_id, node, status,
                    json.dumps(input_summary) if input_summary else None,
                    json.dumps(output_summary) if output_summary else None,
                    error, error_type, duration_ms,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            db.commit()
    except Exception:
        logger.warning(f"checkpoint_save_sync_failed node={node}", exc_info=True)


async def _save_async(
    thread_id: str,
    node: str,
    status: str,
    input_summary: dict | None = None,
    output_summary: dict | None = None,
    error: str | None = None,
    error_type: str | None = None,
    duration_ms: int | None = None,
) -> None:
    try:
        async with aiosqlite.connect(ANALYTICS_DB) as db:
            await db.execute(_TABLE_DDL)
            try:
                await db.execute(_MIGRATION_DDL)
                await db.commit()
            except Exception:
                pass
            await db.execute(
                "INSERT INTO node_checkpoints "
                "(thread_id, node, status, input_summary, output_summary, error, error_type, duration_ms, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    thread_id, node, status,
                    json.dumps(input_summary) if input_summary else None,
                    json.dumps(output_summary) if output_summary else None,
                    error, error_type, duration_ms,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
    except Exception:
        logger.warning(f"checkpoint_save_async_failed node={node}", exc_info=True)


async def save_sub_agent_error(node: str, sub_label: str, exc: Exception) -> None:
    """Record an error from a parallel sub-agent (e.g. a variant in voting_synthesis).

    Called from within parallel runners (asyncio.gather blocks) where exceptions
    are caught to avoid cancelling sibling tasks. Without this call those errors
    would be silently swallowed and invisible in /diagnostics or /errors.
    """
    tid = _current_thread_id.get()
    tb  = traceback.format_exc()
    await _save_async(
        tid, node, "failed",
        input_summary={"sub_label": sub_label},
        error=tb[:3000],
        error_type=type(exc).__name__,
    )
    logger.warning(
        f"[CP] sub_agent_error node={node} sub={sub_label} "
        f"error_type={type(exc).__name__} thread={tid}: {exc!r}"
    )


# ── State summarisers ────────────────────────────────────────────────────────

def _summarise_state(state: dict) -> dict:
    """Compact, safe snapshot of AgentState for diagnostic logging."""
    s: dict = {}
    for k in ("mode", "clarity_status", "confidence_score", "validation_result",
              "attempts", "is_survey", "paper_abstract"):
        if k in state and state[k] is not None:
            v = state[k]
            s[k] = str(v)[:80] if isinstance(v, str) and len(str(v)) > 80 else v

    for k in ("sub_queries", "sources", "suggestions", "fact_check_results", "plan_steps", "chart_urls"):
        v = state.get(k)
        if v:
            s[f"{k}_count"] = len(v)

    for k in ("findings", "report_content", "raw_research"):
        v = state.get(k)
        if v:
            s[f"{k}_chars"] = len(v)

    doc_id = state.get("doc_id")
    if doc_id:
        s["doc_id"] = doc_id[:8] + "…"

    msgs = state.get("messages")
    if msgs:
        s["messages_count"] = len(msgs)
        last = msgs[-1]
        content = getattr(last, "content", "") or ""
        s["last_msg_preview"] = content[:100] + ("…" if len(content) > 100 else "")

    return s


def _summarise_output(output: dict) -> dict:
    """Compact snapshot of what a node returned."""
    s: dict = {}
    for k, v in output.items():
        if k == "messages":
            s["messages_added"] = len(v) if isinstance(v, list) else 1
            if isinstance(v, list) and v:
                content = getattr(v[-1], "content", "") or ""
                s["last_msg_preview"] = content[:120] + ("…" if len(content) > 120 else "")
        elif isinstance(v, str):
            s[k] = v[:120] + ("…" if len(v) > 120 else "")
        elif isinstance(v, (int, float, bool)):
            s[k] = v
        elif isinstance(v, list):
            s[f"{k}_count"] = len(v)
        elif v is None:
            pass
        else:
            s[k] = type(v).__name__
    return s


# ── Decorator ────────────────────────────────────────────────────────────────

def checkpoint(fn):
    """Wrap a LangGraph node with entry/exit/error checkpoint recording."""
    from langgraph.errors import GraphInterrupt  # local import avoids circular deps

    node_name = fn.__name__.replace("_node", "")

    if asyncio.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def _async_wrapper(state, *args, **kwargs):
            tid = _current_thread_id.get()
            in_sum = _summarise_state(dict(state))
            t0 = time.monotonic()

            logger.info(
                f"[CP] node={node_name} status=started thread={tid} "
                f"mode={in_sum.get('mode','-')} msgs={in_sum.get('messages_count',0)}"
            )
            await _save_async(tid, node_name, "started", input_summary=in_sum)

            try:
                result = await fn(state, *args, **kwargs)
            except GraphInterrupt:
                ms = int((time.monotonic() - t0) * 1000)
                logger.info(f"[CP] node={node_name} status=interrupted thread={tid} duration_ms={ms}")
                await _save_async(tid, node_name, "interrupted", duration_ms=ms)
                raise
            except Exception as exc:
                ms = int((time.monotonic() - t0) * 1000)
                tb = traceback.format_exc()
                etype = type(exc).__name__
                logger.error(
                    f"[CP] node={node_name} status=FAILED thread={tid} "
                    f"error_type={etype} duration_ms={ms} error={exc!r}\n{tb}"
                )
                await _save_async(
                    tid, node_name, "failed",
                    input_summary=in_sum, error=tb[:3000], error_type=etype, duration_ms=ms,
                )
                raise

            ms = int((time.monotonic() - t0) * 1000)
            out_sum = _summarise_output(result) if isinstance(result, dict) else {}
            logger.info(
                f"[CP] node={node_name} status=completed thread={tid} "
                f"duration_ms={ms} outputs={list(out_sum.keys())}"
            )
            await _save_async(tid, node_name, "completed", output_summary=out_sum, duration_ms=ms)
            return result

        return _async_wrapper

    else:
        @functools.wraps(fn)
        def _sync_wrapper(state, *args, **kwargs):
            tid = _current_thread_id.get()
            in_sum = _summarise_state(dict(state))
            t0 = time.monotonic()

            logger.info(
                f"[CP] node={node_name} status=started thread={tid} "
                f"mode={in_sum.get('mode','-')} msgs={in_sum.get('messages_count',0)}"
            )
            _save_sync(tid, node_name, "started", input_summary=in_sum)

            try:
                result = fn(state, *args, **kwargs)
            except GraphInterrupt:
                ms = int((time.monotonic() - t0) * 1000)
                logger.info(f"[CP] node={node_name} status=interrupted thread={tid} duration_ms={ms}")
                _save_sync(tid, node_name, "interrupted", duration_ms=ms)
                raise
            except Exception as exc:
                ms = int((time.monotonic() - t0) * 1000)
                tb = traceback.format_exc()
                etype = type(exc).__name__
                logger.error(
                    f"[CP] node={node_name} status=FAILED thread={tid} "
                    f"error_type={etype} duration_ms={ms} error={exc!r}\n{tb}"
                )
                _save_sync(tid, node_name, "failed",
                           input_summary=in_sum, error=tb[:3000], error_type=etype, duration_ms=ms)
                raise

            ms = int((time.monotonic() - t0) * 1000)
            out_sum = _summarise_output(result) if isinstance(result, dict) else {}
            logger.info(
                f"[CP] node={node_name} status=completed thread={tid} "
                f"duration_ms={ms} outputs={list(out_sum.keys())}"
            )
            _save_sync(tid, node_name, "completed", output_summary=out_sum, duration_ms=ms)
            return result

        return _sync_wrapper
