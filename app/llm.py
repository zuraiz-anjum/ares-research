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

import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Global index: which provider to start from. Advances automatically when
# a provider is rate-limited. Resets on server restart (next day).
_active_idx: int = 0


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in [
        "rate limit", "429", "resource_exhausted", "quota exceeded",
        "too many requests", "rate_limit_exceeded", "tokens per day",
        "limit: 0", "requests per day", "exceeded your current quota",
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

    # ── Public interface ───────────────────────────────────────────────────────

    def invoke(self, *args, **kwargs):
        while True:
            try:
                return self._llm().invoke(*args, **kwargs)
            except Exception as exc:
                if _is_rate_limited(exc) and self._advance(exc):
                    continue
                raise

    async def ainvoke(self, *args, **kwargs):
        while True:
            try:
                return await self._llm().ainvoke(*args, **kwargs)
            except Exception as exc:
                if _is_rate_limited(exc) and self._advance(exc):
                    continue
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
    temp = settings.llm_temperature if temperature is None else temperature
    providers = _build_llms(temp, streaming)
    if not providers:
        raise RuntimeError(
            "No LLM provider configured. Set at least one of: "
            "GROQ_API_KEY, GEMINI_API_KEY, CEREBRAS_API_KEY, OPENROUTER_API_KEY"
        )
    start = min(_active_idx, len(providers) - 1)
    return _FallbackLLM(providers, start)
