"""Deterministic test doubles for the LLM provider (no network)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import AsyncIterator

from app.llm.base import (
    ChatMessage,
    CompletionResult,
    LLMProvider,
    ModelInfo,
    ProviderHealth,
    TokenUsage,
)


class FakeProvider(LLMProvider):
    """Records chat() calls; returns queued results or raises queued errors.

    Usage:
        provider = FakeProvider()
        provider.queue_result(CompletionResult(content="hello", ...))
        await provider.chat(...)   # -> recorded + canned response
    """

    def __init__(self, results: list[CompletionResult | Exception] | None = None) -> None:
        self.results: list[CompletionResult | Exception] = list(results or [])
        self.chat_calls: list[dict] = []

    @property
    def name(self) -> str:
        return "fake"

    def queue_result(self, result: CompletionResult | Exception) -> None:
        self.results.append(result)

    @staticmethod
    def make_result(content: str = "ok", *, prompt_tokens: int = 10,
                    completion_tokens: int = 5) -> CompletionResult:
        return CompletionResult(
            content=content,
            tool_calls=[],
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model="fake-model",
            finish_reason="stop",
        )

    async def chat(self, messages: Sequence[ChatMessage], **kwargs) -> CompletionResult:
        self.chat_calls.append({"messages": list(messages), "kwargs": dict(kwargs)})
        if not self.results:
            raise AssertionError("FakeProvider.queue_result() was not called")
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def stream(self, messages: Sequence[ChatMessage], **kwargs) -> AsyncIterator:
        raise NotImplementedError("FakeProvider does not implement stream()")

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health(self) -> ProviderHealth:
        return ProviderHealth(reachable=True)

    async def aclose(self) -> None:
        pass
