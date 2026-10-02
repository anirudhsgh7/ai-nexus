from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for AI Nexus. Env prefix AI_NEXUS_, `.env` supported."""

    model_config = SettingsConfigDict(
        env_prefix="AI_NEXUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AI Nexus"
    app_version: str = "0.1.0"

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    ollama_base_url: str = "http://localhost:11434"
    primary_model: str = "qwen2.5:14b-instruct"
    fallback_model: str = "qwen2.5:7b-instruct"

    num_ctx: int = Field(default=8192, ge=2048, le=32768)
    max_rounds: int = Field(default=3, ge=1, le=5)
    keep_alive: str = "30m"
    request_timeout_s: float = Field(default=300.0, gt=0)
    connect_timeout_s: float = Field(default=5.0, gt=0)
    max_concurrent_generations: int = Field(default=1, ge=1)
    health_cache_ttl_s: float = Field(default=5.0, ge=0)

    log_level: str = "INFO"
    log_token_usage: bool = True

    default_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # --- Tools (Phase 6) ---
    tool_max_steps: int = Field(default=5, ge=1, le=10)
    tool_timeout_s: float = Field(default=20.0, gt=0.0, le=120.0)
    tool_result_max_chars: int = Field(default=2000, ge=200, le=8000)
    tool_results_budget_chars: int = Field(default=6000, ge=1000, le=24000)
    tool_files_root: str = ""           # empty = file tools disabled
    tool_web_search_enabled: bool = False
    tool_web_search_max_results: int = Field(default=5, ge=1, le=10)
    # Phase 8b: reliability settings behind the same flag.
    tool_web_search_providers: list[str] = Field(
        default_factory=lambda: ["ddg", "bing"]
    )
    tool_web_search_min_interval_s: float = Field(default=3.0, ge=0.5, le=30.0)
    tool_web_search_retries: int = Field(default=2, ge=0, le=5)
    tool_web_search_backoff_base_s: float = Field(default=2.0, ge=0.5, le=15.0)
    tool_web_search_timeout_s: float = Field(default=45.0, ge=5.0, le=120.0)
    tool_web_search_cache_ttl_s: float = Field(default=900.0, ge=0.0, le=86400.0)
    tool_web_search_region: str = "us-en"

    # --- Persistence (Phase 7) ---
    db_path: str = "data/ai_nexus.db"   # empty = no persistence (Phase 6 behavior)
    db_retention_runs: int = Field(default=500, ge=0)  # 0 = keep everything

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    @field_validator("ollama_base_url")
    @classmethod
    def _validate_base_url(cls, v: str) -> str:
        v = v.rstrip("/")
        if not v.startswith(("http://", "https://")):
            raise ValueError("ollama_base_url must start with http:// or https://")
        return v

    @field_validator("primary_model", "fallback_model")
    @classmethod
    def _validate_model_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model name must be non-empty")
        return v.strip()

    @field_validator("fallback_model")
    @classmethod
    def _models_must_differ(cls, v: str, info: Any) -> str:
        primary = info.data.get("primary_model")
        if primary is not None and primary == v:
            raise ValueError("fallback_model must differ from primary_model")
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        level = v.upper()
        if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError(f"invalid log_level: {v}")
        return level

    @field_validator("db_path")
    @classmethod
    def _normalize_db_path(cls, v: str) -> str:
        return v.strip()

    @field_validator("tool_web_search_providers")
    @classmethod
    def _validate_web_search_providers(
        cls, v: list[str], info: ValidationInfo
    ) -> list[str]:
        # Lazy import keeps `app.config` free of a module-level `app.tools`
        # dependency (registry imports config; tools importing config would
        # otherwise be a cycle).
        from app.tools.web_search import KNOWN_SEARCH_PROVIDERS

        cleaned: list[str] = []
        for raw in v:
            name = str(raw).strip().lower()
            if name and name not in cleaned:
                cleaned.append(name)
        unknown = [name for name in cleaned if name not in KNOWN_SEARCH_PROVIDERS]
        if unknown:
            known = ", ".join(sorted(KNOWN_SEARCH_PROVIDERS))
            raise ValueError(
                f"unknown web search provider(s): {', '.join(unknown)} "
                f"(known: {known})"
            )
        if not cleaned and info.data.get("tool_web_search_enabled"):
            raise ValueError(
                "tool_web_search_providers must list at least one provider "
                "when tool_web_search_enabled is true"
            )
        return cleaned

    @field_validator("tool_web_search_region")
    @classmethod
    def _validate_region(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("tool_web_search_region must be non-empty")
        return v

    @model_validator(mode="after")
    def _tool_budget_covers_result_cap(self) -> "Settings":
        if self.tool_results_budget_chars < self.tool_result_max_chars:
            raise ValueError(
                "tool_results_budget_chars must be >= tool_result_max_chars "
                f"({self.tool_results_budget_chars} < {self.tool_result_max_chars})"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
