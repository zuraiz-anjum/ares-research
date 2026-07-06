"""LLM factory with automatic provider fallback.

Priority order (all configured keys are tried in this sequence):
  1. Groq        — GROQ_API_KEY        — llama-3.3-70b, 100k tok/day
  2. Cerebras    — CEREBRAS_API_KEY    — gpt-oss-120b, fast fallback
  3. Gemini      — GEMINI_API_KEY      — gemini-2.0-flash (last: slow retry on quota)
  4. OpenRouter  — OPENROUTER_API_KEY  — llama-3.3-70b:free, unlimited

When a provider returns a rate-limit / quota-exceeded error the factory
advances the global active index so every subsequent call in that process
skips the exhausted provider — no manual intervention needed.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone

from app.config import settings

# How long to wait before giving the whole provider chain one bounded second
# pass, once every provider has failed on the same call. Short on purpose —
# this is a "give transient bursts a moment to clear" delay, not an attempt
# to wait out a provider's full suggested retry-after (which can be 60s+).
_EXHAUSTED_RETRY_DELAY_SECONDS = 3


class AllProvidersExhaustedError(Exception):
    """Raised when the whole fallback chain fails and at least one provider
    along the way failed on a genuine rate-limit/quota error.

    The *last* exception in the chain isn't always itself a rate limit — e.g.
    Groq/Cerebras/Gemini can all be legitimately quota-exhausted, and then the
    final fallback (OpenRouter) fails for an unrelated reason (a bad API key,
    say), and that unrelated exception is what would otherwise be raised.
    From the caller's perspective the useful fact is "every provider failed
    and it was capacity-related for most of them" — this exception lets
    main.py show a clear "today's free quota is used up" message instead of
    a generic internal error, without hiding that the final failure might
    also need its own separate fix (see its message for the real cause).
    """

logger = logging.getLogger(__name__)

# Global index: which provider to start from. Advances automatically when
# a provider is rate-limited or errors out. Resets on server restart, and
# also on UTC day-rollover (see _maybe_reset_active_idx) — otherwise a
# provider that's still broken after its daily quota resets (or a dead
# last-resort provider with nothing left to fall back to) would wedge every
# request until someone manually restarts the process.
_active_idx: int = 0
_active_idx_date = None


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in [
        "rate limit", "429", "resource_exhausted", "quota exceeded",
        "too many requests", "rate_limit_exceeded", "tokens per day",
        "limit: 0", "requests per day", "exceeded your current quota",
    ])


def _maybe_reset_active_idx() -> None:
    """Reset the fallback index when a new UTC day starts.

    Rate limits like Groq's 100k-tokens/day reset daily, but `_active_idx`
    previously only reset on process restart. If the chain had advanced past
    a provider whose quota later recovered — or landed on a broken
    last-resort provider (nothing left to fall back to) — every request
    stayed broken until someone noticed and restarted the server manually.
    Resetting on day-rollover bounds an outage to "at most a day" instead of
    "until someone restarts it."
    """
    global _active_idx, _active_idx_date
    today = datetime.now(timezone.utc).date()
    if _active_idx != 0 and _active_idx_date != today:
        logger.warning(f"active_idx_reset previous_idx={_active_idx} reason=new_utc_day")
        _active_idx = 0
    _active_idx_date = today


def _is_provider_bug(exc: Exception) -> bool:
    """Detect provider-specific bugs that warrant falling back to the next provider.

    Groq's llama models fail structured output in two ways:
      1. tool_use_failed — model emits <function=...> Hermes format instead of
         proper JSON tool calls, rejected by Groq's own API.
      2. json_object prompt constraint — json_mode requires the word 'json' in
         the prompt, which not every agent system prompt includes.
    Both are Groq deficiencies; fall through to Cerebras/Gemini instead.
    """
    msg = str(exc)
    return any(kw in msg for kw in [
        "tool_use_failed",
        "failed_generation",
        "must contain the word 'json'",
    ])


def _build_llms(temperature: float, streaming: bool) -> list[tuple[str, object]]:
    """Build the ordered list of available LLM instances."""
    providers: list[tuple[str, object]] = []

    if settings.groq_api_key:
        from langchain_groq import ChatGroq
        providers.append(("groq", ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=settings.groq_api_key,
            streaming=streaming,
        )))

    # Cerebras before Gemini: Gemini has long internal retries on quota errors
    # which block the thread pool for 30+ seconds; Cerebras responds immediately.
    if settings.cerebras_api_key:
        from langchain_openai import ChatOpenAI
        providers.append(("cerebras", ChatOpenAI(
            model="gpt-oss-120b",
            temperature=temperature,
            api_key=settings.cerebras_api_key,
            base_url="https://api.cerebras.ai/v1",
            streaming=streaming,
        )))

    if settings.gemini_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            providers.append(("gemini", ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=temperature,
                google_api_key=settings.gemini_api_key,
                streaming=streaming,
            )))
        except ImportError:
            pass

    if settings.openrouter_api_key:
        from langchain_openai import ChatOpenAI
        providers.append(("openrouter", ChatOpenAI(
            model="meta-llama/llama-3.3-70b-instruct:free",
            temperature=temperature,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            streaming=streaming,
        )))

    return providers


class _FallbackLLM:
    """Transparent LLM wrapper that falls back to the next provider on rate limits.

    Mirrors the invoke / ainvoke / with_structured_output interface of
    LangChain chat models so it can be used as a drop-in replacement.
    """

    def __init__(self, providers: list[tuple[str, object]], start: int = 0):
        self._providers = providers
        self._idx = start
        # Whether this instance has already used its one bounded retry of the
        # full chain after every provider was exhausted. Reset per instance
        # (a fresh _FallbackLLM is created per get_llm() / with_structured_output()
        # call), so it's a one-shot budget per call chain, not global.
        self._retried_full_chain = False
        # True if ANY provider failed on a genuine rate-limit/quota error
        # during this call chain — even if the exception that ultimately gets
        # raised (e.g. from the last, otherwise-broken fallback) isn't itself
        # a rate limit. See AllProvidersExhaustedError.
        self._saw_rate_limit = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _llm(self):
        return self._providers[self._idx][1]

    def _advance(self, exc: Exception) -> bool:
        global _active_idx
        current_name = self._providers[self._idx][0]
        if self._idx + 1 < len(self._providers):
            self._idx += 1
            _active_idx = self._idx
            next_name = self._providers[self._idx][0]
            logger.warning(
                f"provider_fallback from={current_name} to={next_name} "
                f"reason={str(exc)[:60]}"
            )
            return True
        logger.error(f"all_providers_exhausted last={current_name}")
        return False

    def _should_retry_exhausted(self, exc: Exception) -> bool:
        """True once, the first time every provider has failed on a rate-limit.

        Most free-tier 429s are short-lived (the providers themselves often
        report a retry-after of well under a minute). Previously, once every
        provider failed, the exception was raised immediately with no second
        chance — turning a transient burst into a hard failure for the user.
        This gives the whole chain one bounded extra pass after a short wait.
        Not applied to _is_provider_bug() failures (structured-output quirks,
        etc.) since those won't be fixed by waiting and retrying the same call.
        """
        if self._retried_full_chain or not _is_rate_limited(exc):
            return False
        self._retried_full_chain = True
        return True

    def _reset_for_retry(self) -> None:
        global _active_idx
        self._idx = 0
        _active_idx = 0

    # ── Public interface ───────────────────────────────────────────────────────

    def invoke(self, *args, **kwargs):
        while True:
            try:
                return self._llm().invoke(*args, **kwargs)
            except Exception as exc:
                rate_limited = _is_rate_limited(exc)
                self._saw_rate_limit = self._saw_rate_limit or rate_limited
                if (rate_limited or _is_provider_bug(exc)) and self._advance(exc):
                    continue
                if self._should_retry_exhausted(exc):
                    logger.warning(
                        f"all_providers_exhausted_retrying_once delay={_EXHAUSTED_RETRY_DELAY_SECONDS}s"
                    )
                    time.sleep(_EXHAUSTED_RETRY_DELAY_SECONDS)
                    self._reset_for_retry()
                    continue
                if self._saw_rate_limit:
                    raise AllProvidersExhaustedError(str(exc)) from exc
                raise

    async def ainvoke(self, *args, **kwargs):
        while True:
            try:
                return await self._llm().ainvoke(*args, **kwargs)
            except Exception as exc:
                rate_limited = _is_rate_limited(exc)
                self._saw_rate_limit = self._saw_rate_limit or rate_limited
                if (rate_limited or _is_provider_bug(exc)) and self._advance(exc):
                    continue
                if self._should_retry_exhausted(exc):
                    logger.warning(
                        f"all_providers_exhausted_retrying_once delay={_EXHAUSTED_RETRY_DELAY_SECONDS}s"
                    )
                    await asyncio.sleep(_EXHAUSTED_RETRY_DELAY_SECONDS)
                    self._reset_for_retry()
                    continue
                if self._saw_rate_limit:
                    raise AllProvidersExhaustedError(str(exc)) from exc
                raise

    def with_structured_output(self, schema, **kwargs):
        """Return a new _FallbackLLM where every provider uses structured output."""
        structured = [
            (name, llm.with_structured_output(schema, **kwargs))
            for name, llm in self._providers
        ]
        return _FallbackLLM(structured, self._idx)

    def __getattr__(self, name: str):
        return getattr(self._llm(), name)


def get_llm(temperature: float | None = None, streaming: bool = False) -> _FallbackLLM:
    """Return a fallback-aware LLM starting from the current healthy provider."""
    _maybe_reset_active_idx()
    temp = settings.llm_temperature if temperature is None else temperature
    providers = _build_llms(temp, streaming)
    if not providers:
        raise RuntimeError(
            "No LLM provider configured. Set at least one of: "
            "GROQ_API_KEY, GEMINI_API_KEY, CEREBRAS_API_KEY, OPENROUTER_API_KEY"
        )
    start = min(_active_idx, len(providers) - 1)
    return _FallbackLLM(providers, start)
