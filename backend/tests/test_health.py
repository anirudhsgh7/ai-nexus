"""Health endpoint: status semantics, codes, hints, caching (PRD §6.6)."""

from fastapi.testclient import TestClient

from app.api.health import build_health_payload
from app.config import Settings, clear_settings_cache
from app.llm import LLMProvider, ModelInfo, ProviderHealth
from app.llm.base import ChatMessage, CompletionResult, StreamChunk, TokenUsage
from app.main import create_app


class StubProvider(LLMProvider):
    def __init__(self, health_result: ProviderHealth | None = None, exc: Exception | None = None):
        self._health_result = health_result
        self._exc = exc
        self.health_calls = 0

    @property
    def name(self) -> str:
        return "ollama"

    async def chat(self, messages, **kwargs) -> CompletionResult:
        raise NotImplementedError

    async def stream(self, messages, **kwargs):
        yield StreamChunk(delta="", done=True)
        return

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health(self) -> ProviderHealth:
        self.health_calls += 1
        if self._exc:
            raise self._exc
        return self._health_result

    async def aclose(self) -> None:
        pass


def _reachable(models: list[str]) -> ProviderHealth:
    return ProviderHealth(
        reachable=True,
        latency_ms=3.0,
        server_version="0.34.4",
        models=[ModelInfo(name=m, context_length=32768) for m in models],
    )


# ---------------------------------------------------------------- pure payload

def test_payload_ok_status():
    s = Settings()
    code, payload = build_health_payload(
        s, "ollama", _reachable([s.primary_model, s.fallback_model]), started=0.0
    )
    assert code == 200
    assert payload["status"] == "ok"
    assert "hint" not in payload
    assert payload["checks"] == {
        "primary_model_available": True,
        "fallback_model_available": True,
    }


def test_payload_degraded_status_has_pull_hint():
    s = Settings(fallback_model="llama3.2:3b")
    code, payload = build_health_payload(s, "ollama", _reachable(["llama3.2:3b"]), started=0.0)
    assert code == 200
    assert payload["status"] == "degraded"
    assert f"ollama pull {s.primary_model}" in payload["hint"]


def test_payload_unavailable_status_has_serve_hint():
    s = Settings()
    code, payload = build_health_payload(
        s, "ollama",
        ProviderHealth(reachable=False, error="connection refused"),
        started=0.0,
    )
    assert code == 503
    assert payload["status"] == "unavailable"
    assert "ollama serve" in payload["hint"]
    assert payload["provider"]["error"] == "connection refused"


# ---------------------------------------------------------------- HTTP behavior

def _client(monkeypatch, ttl: float = 5.0, stub: StubProvider | None = None):
    monkeypatch.setenv("AI_NEXUS_HEALTH_CACHE_TTL_S", str(ttl))
    clear_settings_cache()
    app = create_app()
    with TestClient(app) as client:
        if stub is not None:
            app.state.provider = stub
            app.state.health_cache = None
        return client, app


def test_ok_response_codes_and_schema(monkeypatch):
    stub = StubProvider(_reachable(["qwen2.5:14b-instruct", "qwen2.5:7b-instruct"]))
    client, _ = _client(monkeypatch, stub=stub)
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["app"]["name"] == "AI Nexus"
    assert body["provider"]["name"] == "ollama"
    assert body["config"]["num_ctx"] == 8192
    assert len(body["provider"]["models"]) == 2
    assert "took_ms" in body


def test_degraded_response(monkeypatch):
    stub = StubProvider(_reachable(["llama3.2:3b"]))
    client, _ = _client(monkeypatch, stub=stub)
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "degraded"
    assert "ollama pull" in body["hint"]


def test_unavailable_response(monkeypatch):
    stub = StubProvider(ProviderHealth(reachable=False, error="connection refused"))
    client, _ = _client(monkeypatch, stub=stub)
    res = client.get("/api/health")
    assert res.status_code == 503
    body = res.json()
    assert body["status"] == "unavailable"
    assert "ollama serve" in body["hint"]


def test_provider_exception_becomes_unavailable_not_500(monkeypatch):
    stub = StubProvider(exc=RuntimeError("boom"))
    client, _ = _client(monkeypatch, stub=stub)
    res = client.get("/api/health")
    assert res.status_code == 503
    assert res.json()["provider"]["error"].startswith("internal error")


def test_cache_prevents_upstream_calls_and_fresh_bypasses(monkeypatch):
    stub = StubProvider(_reachable(["qwen2.5:14b-instruct"]))
    client, _ = _client(monkeypatch, ttl=60.0, stub=stub)

    assert client.get("/api/health").status_code == 200
    assert stub.health_calls == 1

    assert client.get("/api/health").status_code == 200
    assert stub.health_calls == 1, "second call must be served from cache"

    assert client.get("/api/health?fresh=true").status_code == 200
    assert stub.health_calls == 2, "?fresh=true must bypass cache"
