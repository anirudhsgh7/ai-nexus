"""Integration tests against live Ollama (auto-skipped when unreachable)."""

import asyncio

import httpx
import pytest

from app.config import Settings
from app.llm import ChatMessage, ChatRole, ModelNotFoundError, OllamaProvider

pytestmark = pytest.mark.integration

OLLAMA_URL = "http://localhost:11434"


def _ollama_reachable() -> bool:
    try:
        httpx.get(f"{OLLAMA_URL}/api/version", timeout=2.0).raise_for_status()
        return True
    except Exception:
        return False


if not _ollama_reachable():
    pytest.skip("Ollama is not running; skipping integration suite", allow_module_level=True)


@pytest.fixture
async def live_provider():
    provider = OllamaProvider(Settings())
    yield provider
    await provider.aclose()


@pytest.fixture(scope="module")
def installed_model() -> str:
    """Prefer configured primary, then fallback, then whatever is installed."""
    settings = Settings()
    try:
        names = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0).json()["models"]
    except Exception:
        pytest.skip("cannot list Ollama models")
    installed = {m["name"] for m in names}
    for candidate in (settings.primary_model, settings.fallback_model, *installed):
        if candidate in installed:
            return candidate
    pytest.skip("no models installed")


async def test_list_models_non_empty(live_provider):
    models = await live_provider.list_models()
    assert models, "expected at least one installed model"
    assert all(m.name for m in models)


async def test_chat_round_trip_with_usage(live_provider, installed_model):
    result = await live_provider.chat(
        [ChatMessage(role=ChatRole.USER, content="What is 2+2? Reply with only the number.")],
        model=installed_model,
        max_tokens=20,
    )
    assert result.content.strip()
    assert result.usage.prompt_tokens is not None
    assert result.usage.completion_tokens is not None
    # G3: explicit num_ctx must actually bound what Ollama evaluated
    assert result.usage.prompt_tokens <= 8192
    assert result.finish_reason in {"stop", "length", "unload"}


async def test_stream_yields_deltas_and_final_usage(live_provider, installed_model):
    chunks = [
        c async for c in live_provider.stream(
            [ChatMessage(role=ChatRole.USER, content="Count from 1 to 5.")],
            model=installed_model,
            max_tokens=60,
        )
    ]
    assert len(chunks) >= 2, "expected multiple streamed chunks"
    text = "".join(c.delta for c in chunks)
    assert text.strip()
    final = chunks[-1]
    assert final.done is True
    assert final.usage is not None
    assert final.usage.completion_tokens is not None


async def test_model_not_found_with_pull_hint(live_provider):
    with pytest.raises(ModelNotFoundError) as exc_info:
        await live_provider.chat(
            [ChatMessage(role=ChatRole.USER, content="hi")],
            model="does-not-exist:1b",
            keep_alive="0",
        )
    assert "ollama pull" in exc_info.value.hint
    assert "does-not-exist:1b" in str(exc_info.value)


async def test_health_reports_reachable(live_provider):
    health = await live_provider.health()
    assert health.reachable is True
    assert health.server_version
    assert health.models


async def test_concurrent_chats_never_overlap(live_provider, installed_model):
    """Acceptance #7: semaphore(1) serializes upstream generation on M4/16GB.

    Measures actual in-flight POSTs at the transport level, since a caller-side
    start timestamp includes semaphore queue wait and cannot prove this.
    """
    state = {"in_flight": 0, "max_in_flight": 0}
    original_post = live_provider._client.post

    async def tracking_post(*args, **kwargs):
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        try:
            return await original_post(*args, **kwargs)
        finally:
            state["in_flight"] -= 1

    live_provider._client.post = tracking_post
    try:
        await asyncio.gather(
            live_provider.chat(
                [ChatMessage(role=ChatRole.USER, content="Say the word: ok")],
                model=installed_model, max_tokens=10,
            ),
            live_provider.chat(
                [ChatMessage(role=ChatRole.USER, content="Say the word: ready")],
                model=installed_model, max_tokens=10,
            ),
        )
    finally:
        live_provider._client.post = original_post

    assert state["max_in_flight"] == 1, (
        f"expected serialized generation, saw {state['max_in_flight']} in flight"
    )
