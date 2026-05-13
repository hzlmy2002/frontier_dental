"""Runtime configuration loaded from environment / .env file.

All secrets and tunables flow through this single object so the rest of the
codebase never reaches into ``os.environ`` directly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Target site ---
    base_url: str = "https://www.safcodental.com"
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36 frontier-dental-poc/0.1"
    )

    # --- Crawl tuning ---
    rate_limit_rps: float = 1.0
    request_timeout_s: float = 30.0
    max_retries: int = 3
    max_products_per_category: int | None = None
    navigator_max_steps: int = 60
    # Hard cap on pages walked by ListingCrawler per category. The crawler
    # stops earlier when a page yields no new product URLs (Safco's empty
    # state past the last page); this cap is purely a runaway safety net.
    listing_max_pages: int = 50

    # --- LLM (OpenAI-compatible vLLM endpoint) ---
    vllm_base_url: str = Field(default="http://localhost:8000/v1")
    vllm_api_key: str = Field(default="EMPTY")
    vllm_model: str = Field(default="Qwen/Qwen3.5-35B-A3B")
    vllm_temperature: float = 0.0
    vllm_max_tokens: int = 2048

    # --- Adobe Commerce recommendations API (alternative_products source) ---
    # Defaults are Safco's public values — overridable via .env.
    recs_endpoint: str = "https://commerce.adobe.io/recs/v1/precs/preconfigured"
    recs_environment_id: str = "1a3bb2e5-c41c-400c-9910-a71d7d006be4"
    recs_api_key: str = "recs_open"
    recs_store_code: str = "main_website_store"
    recs_store_view_code: str = "default"
    recs_website_code: str = "base"
    recs_max_alternatives: int = 8

    # --- Storage ---
    output_dir: Path = Path("output")
    sqlite_path: Path = Path("output/products.sqlite")

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = True


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide singleton for ``Settings``."""

    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.output_dir.mkdir(parents=True, exist_ok=True)
    return _settings


def reset_settings_for_tests() -> None:
    """Drop the cached singleton — used by tests that override env vars."""

    global _settings
    _settings = None
