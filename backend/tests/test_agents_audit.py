"""Verifier/Accountability agent paths (Phase 11 PRD §9.1).

Scripted FakeProvider chains: the independent-check rule feeds the structured
retry-once loop, reports count as output even when empty of claims, and the
accountability path runs the no-tools structured route (facts are enforced by
the orchestrator, not the agent).
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.agents import Agent, AgentConfig
from app.agents.errors import StructuredOutputError
from app.agents.structured import OutputKind
from app.llm.base import CompletionResult, ToolCall, TokenUsage
from app.schemas import (
    AgentRole,
    Claim,
    ClaimStatus,
    MessageType,
    VerificationStatus,
)
from app.tools import Tool, ToolContext
from tests.fakes import FakeProvider

# --------------------------------------------------------------- payloads

VERIFIED_OK = json.dumps({
    "content": "opened the file and checked",
    "claims": [{
        "claim_id": "c1",
        "verification_status": "verified",
        "evidence_checked": ["growth_report.txt"],
        "supporting_evidence": [
            {"source": "growth_report.txt", "quote": "23%"}
        ],
        "contradicting_evidence": [],
        "source_references": ["growth_report.txt"],
        "explanation": "The file states 23%.",
        "confidence": 0.9,
    }],
})

VERIFIED_WITHOUT_PROOFS = json.dumps({
    "content": "looks right",
    "claims": [{
        "claim_id": "c1",
        "verification_status": "verified",
        "evidence_checked": [],
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "source_references": [],
        "explanation": "the skeptic said so",
        "confidence": 0.9,
    }],
})

UNVERIFIABLE_OK = json.dumps({
    "content": "nothing authoritative exists",
    "claims": [{
        "claim_id": "c1",
        "verification_status": "unverifiable",
        "evidence_checked": [],
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "source_references": [],
        "explanation": "no source can be found",
        "confidence": 0.3,
    }],
})

EMPTY_REPORT = json.dumps({
    "content": "nothing to verify",
    "claims": [],
})

ACCOUNTABILITY_OK = json.dumps({
    "content": "trace audit complete",
    "trace_completeness": True,
    "final_claim_provenance": [
        {"claim_id": "c1", "origin": "researcher",
         "verdict": "supported", "evidence_count": 1},
    ],
    "flags": [],
    "overall_status": "clean",
    "summary": "no process gaps found",
})

ACCOUNTABILITY_EMPTY_CONTENT = json.dumps({
    "content": "",
    "trace_completeness": True,
    "final_claim_provenance": [],
    "flags": [],
    "overall_status": "clean",
    "summary": "clean",
})

ACCOUNTABILITY_BAD_ENUM_FIRST = json.dumps({
    "content": "x",
    "trace_completeness": True,
    "final_claim_provenance": [],
    "flags": [{"kind": "not_a_kind", "severity": "info",
               "refs": [], "explanation": "x"}],
    "overall_status": "clean",
    "summary": "x",
})


# --------------------------------------------------------------- fixtures


def _claims() -> list[Claim]:
    return [
        Claim(id="c1", statement="The rate was 23%.", status=ClaimStatus.FACT),
    ]


def _verifier_config(*, capabilities: frozenset[str] = frozenset()) -> AgentConfig:
    return AgentConfig(
        role=AgentRole.VERIFIER,
        display_name="Verifier",
        instructions="Independently verify each claim; prior verdicts are context only.",
        temperature=0.0,
        output_type=MessageType.VERIFICATION,
        output_kind=OutputKind.VERIFICATION,
        capabilities=capabilities,
    )


def _accountability_config() -> AgentConfig:
    return AgentConfig(
        role=AgentRole.ACCOUNTABILITY,
        display_name="Accountability",
        instructions="Audit the trace for process and provenance gaps only.",
        temperature=0.0,
        output_type=MessageType.ACCOUNTABILITY,
        output_kind=OutputKind.ACCOUNTABILITY,
    )


def _result(content: str, tool_calls: list[ToolCall] | None = None) -> CompletionResult:
    return CompletionResult(
        content=content,
        tool_calls=tool_calls or [],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="fake-model",
        finish_reason="stop",
    )


class _SearchArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1)


class FakeCheckTool(Tool):
    """Minimal tool so the verifier has something executable to 'check with'."""

    name: ClassVar[str] = "file_search"
    description: ClassVar[str] = "Search the corpus."
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    args_model: ClassVar[type[BaseModel]] = _SearchArgs

    async def execute(self, args: BaseModel, context: ToolContext) -> dict:
        return {"matches": [{"file": "growth_report.txt", "line": 2}]}


# ----------------------------------------------------------- no-tools verifier


async def test_verifier_invalid_report_retried_then_succeeds():
    provider = FakeProvider()
    provider.queue_result(_result(VERIFIED_WITHOUT_PROOFS))
    provider.queue_result(_result(VERIFIED_OK))
    agent = Agent(_verifier_config(), provider)

    message = await agent.run("verify the final claims", claims=_claims())

    assert provider.chat_calls.__len__() == 2, "violation must trigger one retry"
    assert (
        provider.chat_calls[0]["kwargs"]["response_format"]
        is provider.chat_calls[1]["kwargs"]["response_format"]
    )
    assert message.verification is not None
    assert message.verification.claims[0].verification_status is (
        VerificationStatus.VERIFIED
    )
    assert message.retries == 1


async def test_verifier_double_failure_raises_structured_error():
    provider = FakeProvider()
    provider.queue_result(_result(VERIFIED_WITHOUT_PROOFS))
    provider.queue_result(_result(VERIFIED_WITHOUT_PROOFS))
    agent = Agent(_verifier_config(), provider)

    with pytest.raises(StructuredOutputError) as exc_info:
        await agent.run("verify", claims=_claims())
    assert exc_info.value.attempts == 2
    assert "independent tool check" in exc_info.value.last_error or (
        "without supporting evidence" in exc_info.value.last_error
    )


async def test_verifier_unverifiable_passes_without_evidence():
    provider = FakeProvider()
    provider.queue_result(_result(UNVERIFIABLE_OK))
    agent = Agent(_verifier_config(), provider)

    message = await agent.run("verify", claims=_claims())
    assert message.verification is not None
    assert message.retries == 0
    assert (
        message.verification.claims[0].verification_status
        is VerificationStatus.UNVERIFIABLE
    )


async def test_verifier_empty_final_claims_report_is_valid_output():
    provider = FakeProvider()
    provider.queue_result(
        _result(json.dumps({
            "content": "no final claims to verify",
            "claims": [],
        }))
    )
    agent = Agent(_verifier_config(), provider)

    message = await agent.run("verify", claims=[])
    # an empty claim list is still a report -> empty-rule must accept it
    assert message.verification is not None
    assert message.verification.claims == []
    assert message.content == "no final claims to verify"


# ---------------------------------------------------------- tool-enabled path


async def test_verifier_with_tools_runs_bounded_loop_then_validates():
    provider = FakeProvider()
    # Phase A: requests a tool call
    provider.queue_result(_result("", tool_calls=[
        ToolCall(name="file_search", arguments={"query": "growth"}),
    ]))
    # Phase A turn 2: no more calls
    provider.queue_result(_result("checked the corpus"))
    # Phase B: valid report
    provider.queue_result(_result(VERIFIED_OK))
    agent = Agent(_verifier_config(capabilities=frozenset({"file_search"})),
                  provider, tools=[FakeCheckTool()])

    message = await agent.run("verify", claims=_claims(), run_id="r1")

    assert provider.chat_calls.__len__() == 3, "Phase A x2 + Phase B"
    assert message.tool_results is not None and len(message.tool_results) == 1
    assert message.verification is not None
    assert message.retries == 0


async def test_verifier_tools_available_but_zero_calls_rejects_verified():
    """The critical invariant: verified without an independent tool check."""
    provider = FakeProvider()
    # Phase A: no tool calls at all
    provider.queue_result(_result("just thinking"))
    # Phase B attempt 1: verified without having checked anything -> rejected
    provider.queue_result(_result(VERIFIED_OK))
    # Phase B attempt 2 (correction): honest downgrade
    provider.queue_result(_result(UNVERIFIABLE_OK))
    agent = Agent(_verifier_config(capabilities=frozenset({"file_search"})),
                  provider, tools=[FakeCheckTool()])

    message = await agent.run("verify", claims=_claims(), run_id="r1")

    assert provider.chat_calls.__len__() == 3, "Phase A + two Phase B attempts"
    assert message.retries == 1
    assert (
        message.verification.claims[0].verification_status
        is VerificationStatus.UNVERIFIABLE
    )


# ------------------------------------------------------------ accountability


async def test_accountability_structured_no_tools_path():
    payload = json.dumps({
        "content": "trace looks complete",
        "trace_completeness": True,
        "final_claim_provenance": [
            {"claim_id": "c1", "origin": "researcher",
             "verdict": "supported", "evidence_count": 1},
        ],
        "flags": [],
        "overall_status": "clean",
        "summary": "no gaps",
    })
    provider = FakeProvider()
    provider.queue_result(_result(payload))
    agent = Agent(_accountability_config(), provider)

    message = await agent.run("audit the run", context="RUN TRACE FACTS: ...")

    assert message.accountability is not None
    assert message.accountability.overall_status.value == "clean"
    # no-tools structured path: response_format set, never tools=
    kwargs = provider.chat_calls[0]["kwargs"]
    assert "tools" not in kwargs
    assert kwargs["response_format"] is not None
    assert message.retries == 0


async def test_accountability_malformed_enum_retried():
    bad = json.dumps({
        "content": "x", "trace_completeness": True,
        "final_claim_provenance": [],
        "flags": [{"kind": "not_a_kind", "severity": "info",
                   "refs": [], "explanation": "x"}],
        "overall_status": "clean", "summary": "x",
    })
    good = json.dumps({
        "content": "x", "trace_completeness": True,
        "final_claim_provenance": [],
        "flags": [],
        "overall_status": "clean", "summary": "x",
    })
    provider = FakeProvider()
    provider.queue_result(_result(bad))
    provider.queue_result(_result(good))
    agent = Agent(_accountability_config(), provider)

    message = await agent.run("audit", context="facts")
    assert message.retries == 1
    assert message.accountability is not None


async def test_accountability_empty_summary_content_still_outputs_report():
    """A report with empty prose counts as structured output (empty rule)."""
    payload = json.dumps({
        "content": "",
        "trace_completeness": True,
        "final_claim_provenance": [],
        "flags": [],
        "overall_status": "clean",
        "summary": "fine",
    })
    provider = FakeProvider()
    provider.queue_result(_result(payload))
    agent = Agent(_accountability_config(), provider)

    message = await agent.run("audit", context="facts")
    assert message.content == ""
    assert message.accountability is not None
