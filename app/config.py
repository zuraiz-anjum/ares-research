"""Application configuration.

Settings are loaded from environment variables (and a local .env file in
development). See `.env.example` for the full list of supported variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Provider credentials -------------------------------------------------
    groq_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    cerebras_api_key: str = ""
    tavily_api_key: str = ""

    # --- Model selection ------------------------------------------------------
    # Auto-selected per provider in llm.py — override here if needed.
    llm_model: str = ""
    llm_temperature: float = 0.2

    # --- Research behaviour ---------------------------------------------------
    max_search_results: int = 5
    max_validation_attempts: int = 3
    confidence_threshold: int = 6
    max_token_budget: int = 24000
    mock_mode: bool = False

    # --- Server ---------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Auth -----------------------------------------------------------------
    # Set APP_PASSWORD in .env to enable the password gate.
    # Leave empty to disable auth (development default).
    app_password: str = ""

    # --- Cost & budget --------------------------------------------------------
    # Estimated cost per 1K tokens in USD (adjust per provider).
    cost_per_1k_tokens_usd: float = 0.001
    # Max cumulative USD cost allowed per session (thread).
    # Queries that would exceed this are aborted with a graceful message.
    session_cost_budget_usd: float = 0.50
    # When True, downgrade expensive modes before hard-aborting.
    # e.g. "plan" (7 LLM calls) → "research" (6) → "chat" (1)
    downgrade_on_budget_pressure: bool = True

    # --- Search result cache --------------------------------------------------
    # Cache Tavily results so repeated / similar queries skip the API.
    search_cache_enabled: bool = True
    search_cache_ttl_seconds: int = 3600   # 1 hour default

    # --- Confidence calibration -----------------------------------------------
    # How many completed requests to look back when recalibrating threshold.
    calibration_window: int = 100

    # --- Observability --------------------------------------------------------
    # Set LANGSMITH_API_KEY to enable LangSmith tracing of every graph run.
    langsmith_api_key: str = ""
    langsmith_project: str = "ares-research"

    # --- Integrations ---------------------------------------------------------
    # POST to this URL after every completed research report (Slack incoming webhook format).
    slack_webhook_url: str = ""
    # SMTP credentials for scheduled email report delivery.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "ares@localhost"

    # --- Persistent storage ---------------------------------------------------
    # Set DATA_DIR to a Railway/Fly volume mount path (e.g. /app/data) so that
    # SQLite databases, charts, and reports survive container restarts.
    # Leave empty in local development — files are written to the working dir.
    data_dir: str = ""


settings = Settings()


def _data_path(filename: str) -> str:
    """Return an absolute path inside DATA_DIR (or cwd if not set)."""
    import os
    base = settings.data_dir.rstrip("/") if settings.data_dir else "."
    return os.path.join(base, filename)


def truncate_to_budget(text: str, label: str = "content") -> tuple[str, bool]:
    """Truncate text to half the token budget before passing to an LLM.

    Returns (text, was_truncated). Using half the budget leaves room for
    the system prompt, history, and the model's response.
    """
    import logging
    logger = logging.getLogger(__name__)
    max_chars = (settings.max_token_budget // 2) * 4
    if len(text) <= max_chars:
        return text, False
    truncated = text[:max_chars] + "\n\n[content truncated to fit token budget]"
    logger.warning(f"token_budget_truncated label={label} original_chars={len(text)} max_chars={max_chars}")
    return truncated, True


def validate_settings() -> None:
    """Fail fast on startup if required credentials are missing."""
    missing = []
    if not settings.mock_mode:
        has_llm = any([
            settings.groq_api_key,
            settings.gemini_api_key,
            settings.openrouter_api_key,
            settings.cerebras_api_key,
        ])
        if not has_llm:
            missing.append("one of: GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, CEREBRAS_API_KEY")
        if not settings.tavily_api_key:
            missing.append("TAVILY_API_KEY")
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill in the values."
        )
