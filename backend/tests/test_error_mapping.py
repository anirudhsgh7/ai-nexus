"""Error mapping: upstream failures become our taxonomy with actionable hints."""

import httpx
import pytest
import respx

from app.llm.base import (
    ChatMessage,
    ChatRole,
    ModelNotFoundError,
    ProviderUnavailableError,
    ResponseParseError,
    RequestTimeoutError,
    UpstreamError,
)

CHAT_URL = "http://localhost:11434/api/chat"
VERSION_URL = "http://localhost:11434/api/version"


def _msgs() -> list[ChatMessage]:
    return [ChatMessage(role=ChatRole.USER, content="hi")]


@respx.mock
async def test_connect_error_maps_to_provider_unavailable(provider):
    respx.post(CHAT_URL).mock(side_effect=httpx.ConnectError("connection refused"))
    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.chat(_msgs())
    assert "ollama serve" in exc_info.value.hint


@respx.mock
async def test_read_timeout_maps_to_request_timeout(provider):
    respx.post(CHAT_URL).mock(side_effect=httpx.ReadTimeout("timed out"))
    with pytest.raises(RequestTimeoutError) as exc_info:
        await provider.chat(_msgs())
    assert "AI_NEXUS_REQUEST_TIMEOUT_S" in exc_info.value.hint


@respx.mock
async def test_model_not_found_maps_with_pull_hint(provider):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(
        404, json={"error": "model 'qwen2.5:14b-instruct' not found, try pulling it first"}
    ))
    with pytest.raises(ModelNotFoundError) as exc_info:
        await provider.chat(_msgs())
    assert "ollama pull qwen2.5:14b-instruct" in exc_info.value.hint


@respx.mock
async def test_upstream_500_maps_to_upstream_error(provider):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(UpstreamError) as exc_info:
        await provider.chat(_msgs())
    assert exc_info.value.status == 500


@respx.mock
async def test_malformed_json_body_maps_to_response_parse_error(provider):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, text="not json"))
    with pytest.raises(ResponseParseError):
        await provider.chat(_msgs())


@respx.mock
async def test_health_reports_unreachable_instead_of_raising(provider):
    respx.get(VERSION_URL).mock(side_effect=httpx.ConnectError("refused"))
    health = await provider.health()
    assert health.reachable is False
    assert health.error
