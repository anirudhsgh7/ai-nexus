"""Ollama-down behavior end-to-end (Phase 10 PRD §6.4).

The provider-level mapping has always been tested; this locks the *full*
contract — health payload, background run failure, and SSE replay — against a
deterministically dead upstream (loopback port with nothing listening), with
no Ollama and no real network.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.main import create_app

DEAD_URL = "http://127.0.0.1:9"  # discard port: connection refused on loopback


def _wait_terminal(client: TestClient, run_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/runs/{run_id}").json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(0.02)
    raise AssertionError("run did not reach a terminal state")


def test_health_reports_unavailable_with_serve_hint(monkeypatch):
    monkeypatch.setenv("AI_NEXUS_OLLAMA_BASE_URL", DEAD_URL)
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unavailable"
        assert body["provider"]["reachable"] is False
        assert "ollama serve" in body["hint"]


def test_run_fails_visibly_with_provider_unavailable_hint(monkeypatch):
    monkeypatch.setenv("AI_NEXUS_OLLAMA_BASE_URL", DEAD_URL)
    app = create_app()
    with TestClient(app) as client:
        created = client.post("/api/runs", json={"task": "ping upstream"}).json()
        assert _wait_terminal(client, created["run_id"])["status"] == "failed"

        body = client.get(f"/api/runs/{created['run_id']}").json()
        assert body["error"]["type"] == "ProviderUnavailableError"
        assert "ollama serve" in body["error"]["hint"]
        assert body["final_message"] is None
        assert body["steps"][0]["status"] == "failed"

        # the failure is replayable over SSE with the same hint
        with client.stream("GET", f"/api/runs/{created['run_id']}/events") as stream:
            text = "".join(stream.iter_text())
        assert "event: run_failed" in text
        assert "ProviderUnavailableError" in text
        assert "ollama serve" in text
