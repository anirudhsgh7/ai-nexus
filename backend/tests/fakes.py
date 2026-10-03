"""Deterministic test doubles for the LLM provider (no network)."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import AsyncIterator

from app.agents.structured import ACCOUNTABILITY_SCHEMA, VERIFICATION_SCHEMA

from app.llm.base import (
    ChatMessage,
    ChatRole,
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


# ---------------------------------------------------- Phase 11 audit fake


class AuditAwareFakeProvider(FakeProvider):
    """Synthesizes the two Phase 11 audit responses from the request itself.

    `validate_verification` requires coverage of the exact claim ids the
    Verifier received, which differs per chain — so the report is built from
    the `CLAIMS TO EVALUATE` block of the actual request. `unverifiable`
    statuses carry no evidence constraints, keeping orchestration chains
    focused on shape (content rules live in test_audits / test_agents_audit).
    """

    def __init__(self, results=None, *, fail_audit: Exception | None = None,
                 fail_verify: Exception | None = None) -> None:
        super().__init__(results)
        self.fail_audit = fail_audit
        self.fail_verify = fail_verify

    async def chat(self, messages, **kwargs):  # type: ignore[override]
        schema = kwargs.get("response_format")
        if schema is VERIFICATION_SCHEMA or schema is ACCOUNTABILITY_SCHEMA:
            self.chat_calls.append({"messages": list(messages), "kwargs": dict(kwargs)})
            if schema is ACCOUNTABILITY_SCHEMA:
                if self.fail_audit is not None:
                    raise self.fail_audit
                payload = json.dumps({
                    "content": "process audit",
                    "trace_completeness": True,
                    "final_claim_provenance": [],
                    "flags": [],
                    "overall_status": "clean",
                    "summary": "no process gaps found",
                })
            else:
                if self.fail_verify is not None:
                    raise self.fail_verify
                user = next(
                    (
                        m.content for m in reversed(messages)
                        if m.role is ChatRole.USER
                    ),
                    "",
                )
                ids = list(dict.fromkeys(re.findall(r"\[(c\d+)\]", user)))
                payload = json.dumps({
                    "content": "verified against the provided records",
                    "claims": [
                        {
                            "claim_id": cid,
                            "verification_status": "unverifiable",
                            "evidence_checked": [],
                            "supporting_evidence": [],
                            "contradicting_evidence": [],
                            "source_references": [],
                            "explanation": "no authoritative source in this context",
                            "confidence": 0.5,
                        }
                        for cid in ids
                    ],
                })
            return FakeProvider.make_result(payload)
        return await super().chat(messages, **kwargs)
