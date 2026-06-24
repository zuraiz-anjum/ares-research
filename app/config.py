"""Application configuration.

Settings are loaded from environment variables (and a local .env file in
development). See `.env.example` for the full list of supported variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Provider credentials -------------------------------------------------
    groq_api_key: str = ""
    tavily_api_key: str = ""

    # --- Model selection ------------------------------------------------------
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.2

    # --- Research behaviour ---------------------------------------------------
    max_search_results: int = 5
    max_validation_attempts: int = 3
    confidence_threshold: int = 6

    # --- Server ---------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()


