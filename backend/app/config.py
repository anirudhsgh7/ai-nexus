from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
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
    keep_alive: str = "30m"
    request_timeout_s: float = Field(default=300.0, gt=0)
    connect_timeout_s: float = Field(default=5.0, gt=0)
    max_concurrent_generations: int = Field(default=1, ge=1)
    health_cache_ttl_s: float = Field(default=5.0, ge=0)

    log_level: str = "INFO"
    log_token_usage: bool = True

    default_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
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


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
