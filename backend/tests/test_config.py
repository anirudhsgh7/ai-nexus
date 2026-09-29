import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings

# Defaults must match the PHASE_1_PRD.md §6.1 table exactly.
PRD_DEFAULTS = {
    "app_name": "AI Nexus",
    "app_version": "0.1.0",
    "host": "127.0.0.1",
    "port": 8000,
    "ollama_base_url": "http://localhost:11434",
    "primary_model": "qwen2.5:14b-instruct",
    "fallback_model": "qwen2.5:7b-instruct",
    "num_ctx": 8192,
    "max_rounds": 3,
    "keep_alive": "30m",
    "request_timeout_s": 300.0,
    "connect_timeout_s": 5.0,
    "max_concurrent_generations": 1,
    "health_cache_ttl_s": 5.0,
    "log_level": "INFO",
    "log_token_usage": True,
    "default_temperature": 0.0,
}


def test_defaults_match_prd(settings: Settings):
    for field, expected in PRD_DEFAULTS.items():
        assert getattr(settings, field) == expected, field
    assert settings.cors_origins == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def test_env_override(monkeypatch):
    monkeypatch.setenv("AI_NEXUS_NUM_CTX", "4096")
    monkeypatch.setenv("AI_NEXUS_PRIMARY_MODEL", "llama3.2:3b")
    monkeypatch.setenv("AI_NEXUS_LOG_TOKEN_USAGE", "false")
    s = Settings()
    assert s.num_ctx == 4096
    assert s.primary_model == "llama3.2:3b"
    assert s.log_token_usage is False


def test_invalid_num_ctx_rejected():
    with pytest.raises(ValidationError):
        Settings(num_ctx=1024)
    with pytest.raises(ValidationError):
        Settings(num_ctx=65536)


def test_max_rounds_bounds():
    assert Settings(max_rounds=1).max_rounds == 1
    assert Settings(max_rounds=5).max_rounds == 5
    with pytest.raises(ValidationError):
        Settings(max_rounds=0)
    with pytest.raises(ValidationError):
        Settings(max_rounds=6)


def test_max_rounds_env_override(monkeypatch):
    monkeypatch.setenv("AI_NEXUS_MAX_ROUNDS", "2")
    assert Settings().max_rounds == 2


def test_equal_models_rejected():
    with pytest.raises(ValidationError):
        Settings(primary_model="qwen2.5:14b-instruct", fallback_model="qwen2.5:14b-instruct")


def test_invalid_base_url_rejected():
    with pytest.raises(ValidationError):
        Settings(ollama_base_url="localhost:11434")


def test_base_url_trailing_slash_normalized(settings):
    s = Settings(ollama_base_url="http://localhost:11434/")
    assert s.ollama_base_url == "http://localhost:11434"


def test_invalid_log_level_rejected():
    with pytest.raises(ValidationError):
        Settings(log_level="VERBOSE")


def test_get_settings_cached():
    assert get_settings() is get_settings()
