"""Cost-aware routing — session budget enforcement.

Tracks cumulative token usage per thread and aborts queries that would push
the session over the configured cost budget.

This implements the assignment video-task requirement:
  "Add a guard that aborts a query and returns a graceful message if the
  projected token cost would exceed the configured budget, and log when it fires."

How it works:
  1. Before each graph invocation, estimate how many tokens this query will consume.
  2. Sum historical usage for this thread from the sessions DB.
  3. If (historical + estimate) > session_token_limit, return a graceful error and log.
  4. Post-request, record actual token cost so the running total stays accurate.

Budget is configured via:
  SESSION_COST_BUDGET_USD   (default 0.50)
  COST_PER_1K_TOKENS_USD   (default 0.001)
"""

import logging
from typing import Optional

import aiosqlite

from app.config import settings

logger = logging.getLogger(__name__)


# ── Cost model ─────────────────────────────────────────────────────────────────

# Rough estimated LLM call counts per mode.
# Used to weight the token estimate when mode is known before invocation.
_MODE_CALL_WEIGHT: dict[str, float] = {
    "plan":          2.0,   # dynamic_spawner spawns up to 3 agents + weaver
    "research":      1.8,   # voting_synthesis runs 3 parallel + judge
    "academic":      1.6,   # draft_critic loop + pdf
    "report":        1.5,   # report_writer + draft_critic + pdf
    "pdf":           1.5,
    "comparison":    1.3,
    "data_analysis": 1.2,
    "debate":        1.2,
    "email":         1.0,
    "code":          1.0,
    "chart":         0.8,
    "survey":        0.8,
    "document":      0.8,
    "chat":          0.5,
}
_DEFAULT_WEIGHT = 1.5   # used when mode is not yet known


def session_token_limit() -> int:
    """Convert USD budget + cost rate into a token limit."""
    return int(settings.session_cost_budget_usd / settings.cost_per_1k_tokens_usd * 1000)


def estimate_query_tokens(message: str, mode: str | None = None) -> int:
    """Estimate tokens this query will consume (conservative upper bound).

    Components:
      - input_tokens:    approx token count of the user message
      - output_overhead: typical generated output for the mode (~1500–3000 tokens)
      - chain_overhead:  system prompts + intermediate LLM calls
    """
    input_tokens   = max(len(message) // 4, 50)
    weight         = _MODE_CALL_WEIGHT.get(mode or "", _DEFAULT_WEIGHT)
    base_overhead  = 2500   # output + system prompts for a single-LLM query
    estimated      = input_tokens + int(base_overhead * weight)
    return estimated


def tokens_to_usd(tokens: int) -> float:
    return tokens * settings.cost_per_1k_tokens_usd / 1000


# ── DB helpers ─────────────────────────────────────────────────────────────────

async def get_thread_token_total(thread_id: str, db_path: str) -> int:
    """Return the cumulative tokens used by *thread_id* so far."""
    try:
        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute(
                "SELECT COALESCE(SUM(token_count), 0) FROM sessions WHERE thread_id = ?",
                (thread_id,),
            )
            row = await cur.fetchone()
            return int(row[0]) if row else 0
    except Exception:
        logger.warning("cost_tracker: get_thread_token_total failed", exc_info=True)
        return 0


# ── Budget guard ───────────────────────────────────────────────────────────────

async def check_cost_budget(
    thread_id: str,
    message: str,
    db_path: str,
    mode: str | None = None,
) -> Optional[dict]:
    """Return an error dict if this query would exceed the session cost budget.

    Returns None when the budget is fine (caller should proceed normally).
    Logs a warning when the guard fires so the event is observable.
    """
    limit = session_token_limit()
    existing   = await get_thread_token_total(thread_id, db_path)
    estimated  = estimate_query_tokens(message, mode)
    projected  = existing + estimated

    cost_used_usd      = tokens_to_usd(existing)
    projected_cost_usd = tokens_to_usd(projected)
    budget_usd         = settings.session_cost_budget_usd

    if projected > limit:
        logger.warning(
            f"cost_budget_guard_fired thread={thread_id} "
            f"existing_tokens={existing} estimated_new={estimated} "
            f"projected={projected} limit={limit} "
            f"cost_used=${cost_used_usd:.4f} projected_cost=${projected_cost_usd:.4f} "
            f"budget=${budget_usd:.2f}"
        )
        return {
            "status": "error",
            "answer": (
                f"⚠️ **Session budget limit reached.**\n\n"
                f"This session has consumed approximately **{existing:,} tokens** "
                f"(~${cost_used_usd:.3f}). Adding this query would exceed the "
                f"configured session budget of **${budget_usd:.2f}**.\n\n"
                "Please start a new conversation or ask your administrator to "
                "increase `SESSION_COST_BUDGET_USD`."
            ),
            "thread_id": thread_id,
        }

    # Warn at 80% so the user has advance notice before hitting the wall.
    if projected > limit * 0.80:
        remaining_usd = budget_usd - cost_used_usd
        logger.info(
            f"cost_budget_warning thread={thread_id} "
            f"usage_pct={100 * existing / limit:.0f}% remaining=${remaining_usd:.3f}"
        )

    return None


def budget_summary(thread_tokens: int) -> dict:
    """Return a human-readable cost summary for a thread."""
    limit = session_token_limit()
    pct   = min(100.0, thread_tokens / limit * 100) if limit > 0 else 0
    return {
        "tokens_used":      thread_tokens,
        "token_limit":      limit,
        "usage_pct":        round(pct, 1),
        "cost_used_usd":    round(tokens_to_usd(thread_tokens), 4),
        "cost_budget_usd":  settings.session_cost_budget_usd,
        "remaining_usd":    round(settings.session_cost_budget_usd - tokens_to_usd(thread_tokens), 4),
    }
