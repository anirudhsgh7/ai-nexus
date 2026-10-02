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


# ------------------------------------------------------- tools (Phase 6 PRD §6.4)

TOOL_DEFAULTS = {
    "tool_max_steps": 5,
    "tool_timeout_s": 20.0,
    "tool_result_max_chars": 2000,
    "tool_results_budget_chars": 6000,
    "tool_files_root": "",
    "tool_web_search_enabled": False,
    "tool_web_search_max_results": 5,
}


def test_tool_defaults(settings: Settings):
    for field, expected in TOOL_DEFAULTS.items():
        assert getattr(settings, field) == expected, field


def test_tool_env_override(monkeypatch):
    monkeypatch.setenv("AI_NEXUS_TOOL_MAX_STEPS", "3")
    monkeypatch.setenv("AI_NEXUS_TOOL_WEB_SEARCH_ENABLED", "true")
    monkeypatch.setenv("AI_NEXUS_TOOL_FILES_ROOT", "/tmp/corpus")
    s = Settings()
    assert s.tool_max_steps == 3
    assert s.tool_web_search_enabled is True
    assert s.tool_files_root == "/tmp/corpus"


def test_tool_bounds():
    assert Settings(tool_max_steps=1).tool_max_steps == 1
    assert Settings(tool_max_steps=10).tool_max_steps == 10
    for bad in ({"tool_max_steps": 0}, {"tool_max_steps": 11},
                {"tool_result_max_chars": 100}, {"tool_result_max_chars": 9000},
                {"tool_web_search_max_results": 0},
                {"tool_web_search_max_results": 11},
                {"tool_timeout_s": 0.0}):
        with pytest.raises(ValidationError):
            Settings(**bad)


def test_tool_budget_must_cover_result_cap():
    assert Settings(
        tool_result_max_chars=2000, tool_results_budget_chars=6000
    ).tool_results_budget_chars == 6000
    with pytest.raises(ValidationError):
        Settings(tool_result_max_chars=4000, tool_results_budget_chars=3000)
    # raising the cap alone trips the cross-field validator
    with pytest.raises(ValidationError):
        Settings(tool_result_max_chars=8000)


# ------------------------------------- web search reliability (Phase 8b PRD §6.9)

WEB_SEARCH_DEFAULTS = {
    "tool_web_search_providers": ["ddg", "bing"],
    "tool_web_search_min_interval_s": 3.0,
    "tool_web_search_retries": 2,
    "tool_web_search_backoff_base_s": 2.0,
    "tool_web_search_timeout_s": 45.0,
    "tool_web_search_cache_ttl_s": 900.0,
    "tool_web_search_region": "us-en",
}


def test_web_search_defaults(settings: Settings):
    for field, expected in WEB_SEARCH_DEFAULTS.items():
        assert getattr(settings, field) == expected, field


def test_web_search_env_override(monkeypatch):
    monkeypatch.setenv("AI_NEXUS_TOOL_WEB_SEARCH_ENABLED", "true")
    monkeypatch.setenv("AI_NEXUS_TOOL_WEB_SEARCH_PROVIDERS", '["bing","ddg"]')
    monkeypatch.setenv("AI_NEXUS_TOOL_WEB_SEARCH_RETRIES", "4")
    monkeypatch.setenv("AI_NEXUS_TOOL_WEB_SEARCH_MIN_INTERVAL_S", "5")
    monkeypatch.setenv("AI_NEXUS_TOOL_WEB_SEARCH_REGION", "uk-en")
    s = Settings()
    assert s.tool_web_search_providers == ["bing", "ddg"]  # order preserved
    assert s.tool_web_search_retries == 4
    assert s.tool_web_search_min_interval_s == 5.0
    assert s.tool_web_search_region == "uk-en"


def test_web_search_provider_list_dedupes_and_normalizes():
    s = Settings(tool_web_search_providers=["DDG", "ddg", " bing "])
    assert s.tool_web_search_providers == ["ddg", "bing"]


def test_web_search_unknown_provider_rejected():
    with pytest.raises(ValidationError, match="unknown web search provider"):
        Settings(tool_web_search_providers=["ddg", "mojeek"])


def test_web_search_empty_providers_fail_only_when_enabled():
    assert Settings(tool_web_search_providers=[]).tool_web_search_providers == []
    with pytest.raises(ValidationError, match="at least one provider"):
        Settings(tool_web_search_enabled=True, tool_web_search_providers=[])


def test_web_search_bounds():
    for bad in (
        {"tool_web_search_min_interval_s": 0.1},
        {"tool_web_search_min_interval_s": 31.0},
        {"tool_web_search_retries": -1},
        {"tool_web_search_retries": 6},
        {"tool_web_search_backoff_base_s": 0.1},
        {"tool_web_search_backoff_base_s": 20.0},
        {"tool_web_search_timeout_s": 4.0},
        {"tool_web_search_timeout_s": 121.0},
        {"tool_web_search_cache_ttl_s": -1.0},
    ):
        with pytest.raises(ValidationError):
            Settings(**bad)


# -------------------------------------------------------- persistence (Phase 7 PRD §6.9)

def test_db_defaults(monkeypatch):
    # the autouse conftest fixture sets AI_NEXUS_DB_PATH for app tests;
    # the documented default is asserted with the env var cleared
    monkeypatch.delenv("AI_NEXUS_DB_PATH", raising=False)
    s = Settings()
    assert s.db_path == "data/ai_nexus.db"
    assert s.db_retention_runs == 500


def test_db_settings_env_override(monkeypatch):
    monkeypatch.setenv("AI_NEXUS_DB_PATH", "  /tmp/other.db ")
    monkeypatch.setenv("AI_NEXUS_DB_RETENTION_RUNS", "0")
    s = Settings()
    assert s.db_path == "/tmp/other.db"  # whitespace stripped
    assert s.db_retention_runs == 0


def test_db_path_empty_disables_persistence(monkeypatch):
    monkeypatch.setenv("AI_NEXUS_DB_PATH", "")
    assert Settings().db_path == ""


def test_db_retention_bounds():
    assert Settings(db_retention_runs=0).db_retention_runs == 0
    with pytest.raises(ValidationError):
        Settings(db_retention_runs=-1)
