"""OllamaProvider: talks to a local Ollama server over /api/chat.

This is the only module allowed to know Ollama's wire format or import httpx
for transport purposes (tests and scripts excepted).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterator, NoReturn, Sequence

import httpx

from app import logging_config as log
from app.config import Settings
from app.llm.base import (
    ChatMessage,
    ChatRole,
    CompletionResult,
    ContextOverflowError,
    LLMProvider,
    ModelInfo,
    ProviderHealth,
    ProviderUnavailableError,
    ModelNotFoundError,
    ResponseParseError,
    RequestTimeoutError,
    StreamChunk,
    TokenUsage,
    ToolCall,
    UpstreamError,
)

logger = logging.getLogger("ai_nexus.llm.ollama")

_TOKENS_PER_CHAR = 3.5  # rough heuristic for pre-flight context pressure check
_NS_PER_MS = 1e6


def _extract_missing_model(body: str) -> str:
    """Pull the model name out of Ollama's 404 body, whatever shape it has."""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = {}
    if isinstance(data, dict) and data.get("model"):
        return str(data["model"])
    match = re.search(r"model ['\"]([^'\"]+)['\"]", str(data.get("error", "")))
    if match:
        return match.group(1)
    return "unknown"


class OllamaProvider(LLMProvider):

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        timeout = httpx.Timeout(
            connect=settings.connect_timeout_s,
            read=settings.request_timeout_s,
            write=30.0,
            pool=settings.connect_timeout_s,
        )
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=timeout,
        )
        self._sem = asyncio.Semaphore(settings.max_concurrent_generations)
        if settings.max_concurrent_generations > 2:
            logger.warning(
                "max_concurrent_generations=%s exceeds hardware-safe value on 16GB",
                settings.max_concurrent_generations,
            )

    @property
    def name(self) -> str:
        return "ollama"

    # ------------------------------------------------------------------ utils

    def _resolve(
        self,
        *,
        model: str | None,
        temperature: float | None,
        num_ctx: int | None,
        keep_alive: str | None,
    ) -> tuple[str, float, int, str]:
        s = self._settings
        return (
            model or s.primary_model,
            s.default_temperature if temperature is None else temperature,
            s.num_ctx if num_ctx is None else num_ctx,
            s.keep_alive if keep_alive is None else keep_alive,
        )

    @staticmethod
    def _map_message(message: ChatMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": message.role.value if isinstance(message.role, ChatRole) else str(message.role),
            "content": message.content,
        }
        if message.tool_calls is not None:
            payload["tool_calls"] = message.tool_calls
        if message.tool_name is not None:
            payload["tool_name"] = message.tool_name
        return payload

    def _build_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float,
        num_ctx: int,
        keep_alive: str,
        stream: bool,
        max_tokens: int | None,
        tools: list[dict] | None,
        response_format: dict | str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [self._map_message(m) for m in messages],
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                # Defaults equal Ollama's own defaults (1.1 / 64), so this is
                # byte-identical behavior until an operator tunes it
                # (Phase 10 PRD §6.2).
                "repeat_penalty": self._settings.repeat_penalty,
                "repeat_last_n": self._settings.repeat_last_n,
            },
            "keep_alive": keep_alive,
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        if tools:
            payload["tools"] = tools
        if response_format is not None:
            payload["format"] = response_format
        return payload

    @staticmethod
    def _estimate_prompt_tokens(messages: Sequence[ChatMessage]) -> int:
        chars = sum(len(m.content) for m in messages)
        return int(chars / _TOKENS_PER_CHAR)

    def _precheck_context(self, messages: Sequence[ChatMessage], num_ctx: int) -> None:
        estimate = self._estimate_prompt_tokens(messages)
        if estimate >= num_ctx:
            raise ContextOverflowError(estimate, num_ctx)
        if estimate >= 0.9 * num_ctx:
            log.context_pressure(
                logger, estimated_tokens=estimate, num_ctx=num_ctx
            )

    @staticmethod
    def _postcheck_context(usage: TokenUsage, num_ctx: int) -> None:
        prompt = usage.prompt_tokens
        if prompt is not None and prompt >= num_ctx:
            raise ContextOverflowError(prompt, num_ctx)
        if prompt is not None and prompt >= 0.95 * num_ctx:
            log.context_truncation_suspected(
                logger, prompt_tokens=prompt, num_ctx=num_ctx
            )

    @staticmethod
    def _parse_usage(data: dict[str, Any]) -> TokenUsage:
        prompt = data.get("prompt_eval_count")
        completion = data.get("eval_count")
        total = (prompt + completion) if prompt is not None and completion is not None else None

        def _ms(key: str) -> float | None:
            ns = data.get(key)
            return None if ns is None else round(ns / _NS_PER_MS, 2)

        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            load_duration_ms=_ms("load_duration"),
            prompt_eval_duration_ms=_ms("prompt_eval_duration"),
            eval_duration_ms=_ms("eval_duration"),
        )

    @staticmethod
    def _parse_tool_calls(raw: list[Any] | None) -> list[ToolCall]:
        if not raw:
            return []
        calls: list[ToolCall] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ResponseParseError(f"tool_call is not an object: {item!r}")
            fn = item.get("function") or {}
            name = fn.get("name")
            if not name:
                raise ResponseParseError("tool_call missing function.name")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError as exc:
                    raise ResponseParseError(
                        f"tool_call arguments not valid JSON: {args[:200]}"
                    ) from exc
            if not isinstance(args, dict):
                raise ResponseParseError("tool_call arguments must be an object")
            calls.append(ToolCall(name=name, arguments=args, id=item.get("id")))
        return calls

    def _parse_chat_response(self, data: dict[str, Any], model: str) -> CompletionResult:
        message = data.get("message")
        if not isinstance(message, dict):
            raise ResponseParseError(
                f"response missing 'message': {json.dumps(data)[:500]}"
            )
        content = message.get("content") or ""
        usage = self._parse_usage(data)
        return CompletionResult(
            content=content,
            tool_calls=self._parse_tool_calls(message.get("tool_calls")),
            usage=usage,
            model=data.get("model") or model,
            finish_reason=data.get("done_reason"),
        )

    @staticmethod
    def _parse_model_info(entry: dict[str, Any]) -> ModelInfo:
        details = entry.get("details") or {}
        # Ollama versions place context_length either in details or model_info.
        context_length = details.get("context_length") or (
            entry.get("model_info") or {}
        ).get("context_length")
        return ModelInfo(
            name=entry.get("name", ""),
            size_bytes=entry.get("size"),
            parameter_size=details.get("parameter_size"),
            quantization=details.get("quantization_level"),
            context_length=context_length,
            capabilities=entry.get("capabilities") or [],
        )

    def _map_httpx_error(self, exc: Exception) -> Exception:
        if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
            return ProviderUnavailableError()
        if isinstance(exc, httpx.ReadTimeout | httpx.PoolTimeout):
            return RequestTimeoutError(self._settings.request_timeout_s)
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            body = exc.response.text[:500]
            if status == 404 and "not found" in body.lower():
                model = _extract_missing_model(body)
                return ModelNotFoundError(model)
            if status == 500 and "token repeat limit" in body.lower():
                # The model looped during a long (usually structured)
                # generation — give the operator a tuning path instead of a
                # raw upstream dump (Phase 10 PRD §6.2). No auto-retry.
                return UpstreamError(
                    f"Ollama returned {status}: {body}",
                    status=status,
                    hint=(
                        "The model looped during generation. Re-run the task; "
                        "if it recurs, raise AI_NEXUS_REPEAT_PENALTY (e.g. 1.2) "
                        "or lower the step's max_tokens."
                    ),
                )
            return UpstreamError(f"Ollama returned {status}: {body}", status=status)
        if isinstance(exc, httpx.TransportError):
            return ProviderUnavailableError(str(exc))
        return exc  # unknown; caller logs it as-is

    def _raise_mapped(self, exc: httpx.HTTPError) -> "NoReturn":
        mapped = self._map_httpx_error(exc)
        if isinstance(mapped, ProviderUnavailableError):
            log.provider_unavailable(logger, error=str(mapped))
        else:
            log.upstream_error(logger, error=str(mapped))
        raise mapped from exc

    # --------------------------------------------------------------- interface

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
    ) -> CompletionResult:
        model, temperature, num_ctx, keep_alive = self._resolve(
            model=model, temperature=temperature, num_ctx=num_ctx, keep_alive=keep_alive
        )
        payload = self._build_payload(
            messages, model=model, temperature=temperature, num_ctx=num_ctx,
            keep_alive=keep_alive, stream=False, max_tokens=max_tokens,
            tools=tools, response_format=response_format,
        )
        self._precheck_context(messages, num_ctx)
        log.llm_request_start(
            logger, provider=self.name, model=model, num_ctx=num_ctx,
            keep_alive=keep_alive, message_count=len(messages), stream=False,
            temperature=temperature,
        )
        started = time.monotonic()
        try:
            async with self._sem:
                response = await self._client.post("/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            self._raise_mapped(exc)
        except json.JSONDecodeError as exc:
            error = ResponseParseError(f"invalid JSON from Ollama: {exc}")
            log.llm_request_error(logger, error=str(error))
            raise error from exc

        try:
            result = self._parse_chat_response(data, model)
        except ResponseParseError as error:
            log.llm_request_error(logger, error=str(error))
            raise

        self._postcheck_context(result.usage, num_ctx)
        log.llm_request_end(
            logger, model=result.model,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            upstream_total_ms=result.usage.load_duration_ms,
            eval_duration_ms=result.usage.eval_duration_ms,
            finish_reason=result.finish_reason,
            wall_ms=round((time.monotonic() - started) * 1000, 1),
        )
        return result

    async def stream(
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
    ) -> AsyncIterator[StreamChunk]:
        model, temperature, num_ctx, keep_alive = self._resolve(
            model=model, temperature=temperature, num_ctx=num_ctx, keep_alive=keep_alive
        )
        payload = self._build_payload(
            messages, model=model, temperature=temperature, num_ctx=num_ctx,
            keep_alive=keep_alive, stream=True, max_tokens=max_tokens,
            tools=tools, response_format=response_format,
        )
        self._precheck_context(messages, num_ctx)
        log.llm_request_start(
            logger, provider=self.name, model=model, num_ctx=num_ctx,
            keep_alive=keep_alive, message_count=len(messages), stream=True,
            temperature=temperature,
        )
        started = time.monotonic()
        final_usage: TokenUsage | None = None

        async with self._sem:
            try:
                async with self._client.stream("POST", "/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError as exc:
                            error = ResponseParseError(f"bad NDJSON line: {line[:200]}")
                            log.llm_request_error(logger, error=str(error))
                            raise error from exc
                        message = chunk.get("message") or {}
                        done = bool(chunk.get("done"))
                        if done:
                            final_usage = self._parse_usage(chunk)
                        yield StreamChunk(
                            delta=message.get("content") or "",
                            done=done,
                            usage=final_usage if done else None,
                            model=chunk.get("model") or model,
                        )
                        if done:
                            break
            except httpx.HTTPError as exc:
                mapped = self._map_httpx_error(exc)
                if isinstance(mapped, ProviderUnavailableError):
                    log.provider_unavailable(logger, error=str(mapped))
                else:
                    log.upstream_error(logger, error=str(mapped))
                raise mapped from exc

        if final_usage is not None:
            self._postcheck_context(final_usage, num_ctx)
            log.llm_request_end(
                logger, model=model,
                prompt_tokens=final_usage.prompt_tokens,
                completion_tokens=final_usage.completion_tokens,
                total_tokens=final_usage.total_tokens,
                eval_duration_ms=final_usage.eval_duration_ms,
                finish_reason="stream",
                wall_ms=round((time.monotonic() - started) * 1000, 1),
            )

    async def list_models(self) -> list[ModelInfo]:
        data = await self._get_json("/api/tags", retry=True)
        return [self._parse_model_info(m) for m in data.get("models", [])]

    async def health(self) -> ProviderHealth:
        started = time.monotonic()
        try:
            version_data = await self._get_json("/api/version", retry=True)
            tags_data = await self._get_json("/api/tags", retry=True)
        except (ProviderUnavailableError, UpstreamError, RequestTimeoutError) as exc:
            return ProviderHealth(reachable=False, error=str(exc))
        return ProviderHealth(
            reachable=True,
            latency_ms=round((time.monotonic() - started) * 1000, 2),
            server_version=version_data.get("version"),
            models=[self._parse_model_info(m) for m in tags_data.get("models", [])],
        )

    async def _get_json(self, path: str, *, retry: bool = False) -> dict[str, Any]:
        attempts = 2 if retry else 1
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                response = await self._client.get(path)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as exc:
                last_error = self._map_httpx_error(exc)
                if isinstance(last_error, UpstreamError) and last_error.status and last_error.status < 500:
                    break  # client errors will not improve on retry
        assert last_error is not None
        raise last_error

    async def aclose(self) -> None:
        await self._client.aclose()
