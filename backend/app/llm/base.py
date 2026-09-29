"""Provider-agnostic LLM abstractions: ABC, message/result types, error taxonomy.

This module is the hard boundary that keeps AI Nexus provider-agnostic.
Nothing outside `app/llm/` may import httpx or know any provider's wire format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Sequence


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class ChatMessage:
    role: ChatRole
    content: str
    tool_calls: list[dict] | None = None
    tool_name: str | None = None


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict
    id: str | None = None


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    load_duration_ms: float | None = None
    prompt_eval_duration_ms: float | None = None
    eval_duration_ms: float | None = None


@dataclass(slots=True)
class CompletionResult:
    content: str
    tool_calls: list[ToolCall]
    usage: TokenUsage
    model: str
    finish_reason: str | None = None


@dataclass(slots=True)
class StreamChunk:
    delta: str
    done: bool
    usage: TokenUsage | None = None
    model: str = ""


@dataclass(slots=True)
class ModelInfo:
    name: str
    size_bytes: int | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    capabilities: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProviderHealth:
    reachable: bool
    latency_ms: float | None = None
    server_version: str | None = None
    models: list[ModelInfo] = field(default_factory=list)
    error: str | None = None


class LLMError(Exception):
    """Base for all provider errors; `hint` is surfaced to API clients."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


class ProviderUnavailableError(LLMError):
    def __init__(self, message: str = "LLM provider is unreachable", *, hint: str = "") -> None:
        super().__init__(message, hint=hint or "Is Ollama running? Start it with: ollama serve")


class ModelNotFoundError(LLMError):
    def __init__(self, model: str) -> None:
        super().__init__(
            f"Model not found: {model}",
            hint=f"Model missing. Pull it with: ollama pull {model}",
        )
        self.model = model


class RequestTimeoutError(LLMError):
    def __init__(self, timeout_s: float) -> None:
        super().__init__(
            f"LLM request timed out after {timeout_s}s",
            hint="Cold model load can be slow; raise AI_NEXUS_REQUEST_TIMEOUT_S.",
        )
        self.timeout_s = timeout_s


class UpstreamError(LLMError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ResponseParseError(LLMError):
    def __init__(self, message: str) -> None:
        super().__init__(message, hint="Malformed upstream response; check Ollama version.")


class LLMProvider(ABC):
    """Contract every provider implements; keyword params resolve from Settings."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        keep_alive: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | str | None = None,
    ) -> CompletionResult: ...

    @abstractmethod
    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        keep_alive: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | str | None = None,
    ) -> AsyncIterator[StreamChunk]: ...

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]: ...

    @abstractmethod
    async def health(self) -> ProviderHealth: ...

    @abstractmethod
    async def aclose(self) -> None: ...
