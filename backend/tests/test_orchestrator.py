"""Pipeline contract (PRD §6.3–6.5): qualification, goldens, execution, errors."""

import asyncio
import json

import pytest

from app.agents import build_registry
from app.agents.structured import DEFAULT_STRUCTURED_MAX_TOKENS
from app.orchestrator import (
    PIPELINE,
    ClaimOrigin,
    Orchestrator,
    qualify_claims,
    render_evidence_board,
    render_synthesis_context,
    resolve_claim,
)
from app.runs import RunEventType, RunManager, RunStatus, StepKind, StepStatus
from app.schemas import AgentRole, Claim, ClaimStatus, ClaimVerdict, Evidence, Verdict
from tests.fakes import FakeProvider

# ------------------------------------------------------------------ payloads
# FakeProvider consumes queued results in exact request order, which for the
# fixed pipeline is: plan, research, ideation, critique, synthesis.


def _claims_payload(content: str, claims: list[dict]) -> str:
    return json.dumps({"content": content, "claims": claims})


def _verdicts_payload(content: str, verdicts: list[dict]) -> str:
    return json.dumps({"content": content, "verdicts": verdicts})


PLAN_JSON = _claims_payload("Plan prose.", [])
RESEARCH_JSON = _claims_payload(
    "Research prose.",
    [
        {"statement": "Finding one", "status": "unverified", "evidence": []},
        {"statement": "Finding two", "status": "fact",
         "evidence": [{"source": "report"}]},
    ],
)
IDEATION_JSON = _claims_payload(
    "Ideation prose.",
    [{"statement": "Idea A", "status": "hypothesis", "evidence": []}],
)
CRITIQUE_JSON = _verdicts_payload(
    "Critique prose.",
    [
        {"claim_id": "c1", "verdict": "refuted", "objection": "no source", "evidence": []},
        {"claim_id": "c2", "verdict": "supported", "objection": "report holds",
         "evidence": []},
        {"claim_id": "c3", "verdict": "unverifiable", "objection": "not checkable",
         "evidence": []},
    ],
)
SYNTHESIS_JSON = _claims_payload("Final answer.", [])

TASK = "Should we build X?"


async def _run_pipeline(payloads: list[str | Exception]):
    provider = FakeProvider()
    for item in payloads:
        provider.queue_result(
            item if isinstance(item, Exception) else FakeProvider.make_result(item)
        )
    store = RunManager()
    run = store.create(TASK)
    orchestrator = Orchestrator(build_registry(provider), store)
    await orchestrator.execute(run.id)
    return provider, store, run


# ------------------------------------------------------------------ qualification

def test_qualify_claims_deterministic_ids_and_origins():
    batches = [
        (AgentRole.RESEARCHER, [
            Claim(id="x1", statement="A", status=ClaimStatus.UNVERIFIED),
            Claim(id="x2", statement="B", status=ClaimStatus.FACT,
                  evidence=[Evidence(source="s")]),
        ]),
        (AgentRole.IDEATOR, [Claim(id="x1", statement="C")]),
    ]
    qualified = qualify_claims(batches)
    assert [q.claim.id for q in qualified] == ["c1", "c2", "c3"]
    assert qualified[0].origin == ClaimOrigin(AgentRole.RESEARCHER, "x1")
    assert qualified[1].origin == ClaimOrigin(AgentRole.RESEARCHER, "x2")
    assert qualified[2].origin == ClaimOrigin(AgentRole.IDEATOR, "x1")
    assert qualified[1].claim.status is ClaimStatus.FACT
    assert qualified[1].claim.evidence[0].source == "s"


def test_qualify_claims_empty():
    assert qualify_claims([]) == []
    assert qualify_claims([(AgentRole.RESEARCHER, [])]) == []


def test_resolve_claim():
    qualified = qualify_claims([(AgentRole.RESEARCHER, [Claim(id="x", statement="A")])])
    assert resolve_claim(qualified, "c1").origin.original_id == "x"
    assert resolve_claim(qualified, "c9") is None


# ------------------------------------------------------------------ board golden

def test_evidence_board_golden():
    qualified = qualify_claims([
        (AgentRole.RESEARCHER, [
            Claim(id="x1", statement="X grew 40%.", status=ClaimStatus.UNVERIFIED,
                  confidence=0.55,
                  evidence=[Evidence(source="blog", quote="up 40%")]),
        ]),
        (AgentRole.IDEATOR, [Claim(id="x1", statement="Idea A.",
                                   status=ClaimStatus.HYPOTHESIS)]),
    ])
    verdicts = [
        Verdict(claim_id="c1", verdict=ClaimVerdict.REFUTED,
                objection="marketing post, not audited"),
    ]
    assert render_evidence_board(qualified, verdicts) == (
        "[c1] (researcher) status=unverified confidence=0.55\n"
        "Claim: X grew 40%.\n"
        "Evidence:\n"
        "- source=blog\n"
        '  quote="up 40%"\n'
        "Verdict: refuted — marketing post, not audited\n"
        "\n"
        "[c2] (ideator) status=hypothesis confidence=none\n"
        "Claim: Idea A.\n"
        "Evidence: none\n"
        "Verdict: none"
    )


def test_evidence_board_empty():
    assert render_evidence_board([], []) == ""


# ------------------------------------------------------------------ synthesis golden

def test_synthesis_context_golden():
    out = render_synthesis_context("R prose", "I prose", "C prose", "BOARD")
    assert out == (
        "RESEARCHER FINDINGS:\nR prose\n\n"
        "IDEATOR OPTIONS:\nI prose\n\n"
        "SKEPTIC CRITIQUE:\nC prose\n\n"
        "EVALUATED CLAIMS:\nBOARD"
    )


def test_synthesis_context_skipped_variant():
    out = render_synthesis_context("R", "I", None, "")
    assert "SKEPTIC CRITIQUE:\n(skipped: no claims were produced to evaluate)" in out
    assert out.endswith("EVALUATED CLAIMS:\n(none)")


# ------------------------------------------------------------------ happy path

async def test_pipeline_happy_path():
    provider, store, run = await _run_pipeline(
        [PLAN_JSON, RESEARCH_JSON, IDEATION_JSON, CRITIQUE_JSON, SYNTHESIS_JSON]
    )

    assert len(provider.chat_calls) == 5
    assert run.status is RunStatus.COMPLETED
    assert run.error is None
    assert run.final_message is not None
    assert run.final_message.content == "Final answer."
    assert run.duration_ms is not None

    # steps: exact kinds/order, all completed
    assert [s.kind for s in run.steps] == [k for k, _ in PIPELINE]
    assert all(s.status is StepStatus.COMPLETED for s in run.steps)
    assert all(s.duration_ms is not None for s in run.steps)


async def test_researcher_and_ideator_context_is_byte_identical():
    provider, _, _ = await _run_pipeline(
        [PLAN_JSON, RESEARCH_JSON, IDEATION_JSON, CRITIQUE_JSON, SYNTHESIS_JSON]
    )
    researcher_user = provider.chat_calls[1]["messages"][1].content
    ideator_user = provider.chat_calls[2]["messages"][1].content
    assert researcher_user == ideator_user
    assert researcher_user == f"TASK:\n{TASK}\n\nCONTEXT:\nPlan prose."

    plan_user = provider.chat_calls[0]["messages"][1].content
    assert plan_user == f"TASK:\n{TASK}"


async def test_skeptic_receives_records_not_peer_prose():
    provider, _, _ = await _run_pipeline(
        [PLAN_JSON, RESEARCH_JSON, IDEATION_JSON, CRITIQUE_JSON, SYNTHESIS_JSON]
    )
    skeptic_user = provider.chat_calls[3]["messages"][1].content
    assert "CLAIMS TO EVALUATE:" in skeptic_user
    assert "[c1] status=unverified" in skeptic_user
    assert "[c3] status=hypothesis" in skeptic_user
    assert "Research prose." not in skeptic_user
    assert "Ideation prose." not in skeptic_user
    assert "Plan prose." not in skeptic_user


async def test_synthesis_context_has_sections_and_provenance():
    provider, _, _ = await _run_pipeline(
        [PLAN_JSON, RESEARCH_JSON, IDEATION_JSON, CRITIQUE_JSON, SYNTHESIS_JSON]
    )
    synthesis_user = provider.chat_calls[4]["messages"][1].content
    for header in ("RESEARCHER FINDINGS:", "IDEATOR OPTIONS:",
                   "SKEPTIC CRITIQUE:", "EVALUATED CLAIMS:"):
        assert header in synthesis_user
    assert "[c1] (researcher) status=unverified" in synthesis_user
    assert "[c3] (ideator) status=hypothesis" in synthesis_user
    assert "Verdict: refuted — no source" in synthesis_user
    assert "Verdict: supported — report holds" in synthesis_user


async def test_event_sequence_exact():
    _, store, run = await _run_pipeline(
        [PLAN_JSON, RESEARCH_JSON, IDEATION_JSON, CRITIQUE_JSON, SYNTHESIS_JSON]
    )
    types = [e.type for e in run.events]
    expected = [RunEventType.RUN_STARTED]
    for _ in range(5):
        expected += [RunEventType.STEP_STARTED, RunEventType.STEP_COMPLETED]
    expected += [RunEventType.RUN_COMPLETED]
    assert types == expected
    assert [e.seq for e in run.events] == list(range(1, 13))

    step_events = [e for e in run.events if e.type is RunEventType.STEP_COMPLETED]
    assert [e.step for e in step_events] == [1, 2, 3, 4, 5]
    assert [e.kind for e in step_events] == [k for k, _ in PIPELINE]
    terminal = run.events[-1]
    assert terminal.message is not None
    assert terminal.message.content == "Final answer."


# ------------------------------------------------------------------ failure paths

async def test_researcher_failure_aborts_run_with_hint():
    from app.llm.base import ProviderUnavailableError

    provider, store, run = await _run_pipeline([PLAN_JSON, ProviderUnavailableError()])

    assert len(provider.chat_calls) == 2  # plan + failed researcher; nothing after
    assert run.status is RunStatus.FAILED
    assert run.final_message is None
    assert [s.status for s in run.steps] == [StepStatus.COMPLETED, StepStatus.FAILED]
    assert run.steps[1].error.type == "ProviderUnavailableError"

    last = run.events[-1]
    assert last.type is RunEventType.RUN_FAILED
    assert last.step == 2 and last.kind is StepKind.RESEARCH
    assert last.agent is AgentRole.RESEARCHER
    assert "ollama serve" in last.error.hint
    assert not any(e.type is RunEventType.RUN_COMPLETED for e in run.events)


async def test_structured_failure_at_skeptic_uses_retry_then_aborts():
    provider, store, run = await _run_pipeline(
        [PLAN_JSON, RESEARCH_JSON, IDEATION_JSON, "not json", "still not json"]
    )
    assert len(provider.chat_calls) == 5  # 3 ok + skeptic attempt + retry
    assert run.status is RunStatus.FAILED
    assert [s.status for s in run.steps] == [
        StepStatus.COMPLETED, StepStatus.COMPLETED,
        StepStatus.COMPLETED, StepStatus.FAILED,
    ]
    assert run.steps[3].error.type == "StructuredOutputError"
    last = run.events[-1]
    assert last.type is RunEventType.RUN_FAILED
    assert last.step == 4 and last.kind is StepKind.CRITIQUE


# ------------------------------------------------------------------ skipped path

async def test_zero_claims_skips_skeptic_and_completes():
    empty_research = _claims_payload("Research prose.", [])
    empty_ideation = _claims_payload("Ideation prose.", [])
    provider, store, run = await _run_pipeline(
        [PLAN_JSON, empty_research, empty_ideation, SYNTHESIS_JSON]
    )
    assert len(provider.chat_calls) == 4  # skeptic skipped entirely
    assert run.status is RunStatus.COMPLETED
    assert [s.status for s in run.steps] == [
        StepStatus.COMPLETED, StepStatus.COMPLETED, StepStatus.COMPLETED,
        StepStatus.SKIPPED, StepStatus.COMPLETED,
    ]
    skipped = run.steps[3]
    assert skipped.skipped is True and skipped.message is None

    skipped_event = next(
        e for e in run.events
        if e.type is RunEventType.STEP_COMPLETED and e.step == 4
    )
    assert skipped_event.skipped is True and skipped_event.message is None

    synthesis_user = provider.chat_calls[3]["messages"][1].content
    assert "(skipped: no claims were produced to evaluate)" in synthesis_user
    assert synthesis_user.endswith("EVALUATED CLAIMS:\n(none)")


# ------------------------------------------------------------------ cancellation

class SlowProvider(FakeProvider):
    async def chat(self, messages, **kwargs):  # type: ignore[override]
        self.chat_calls.append({"messages": list(messages), "kwargs": dict(kwargs)})
        await asyncio.sleep(30)
        raise AssertionError("unreachable")


async def test_cancellation_marks_run_failed_and_step_failed():
    provider = SlowProvider()
    store = RunManager()
    run = store.create(TASK)
    orchestrator = Orchestrator(build_registry(provider), store)

    task = asyncio.create_task(orchestrator.execute(run.id))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert run.status is RunStatus.FAILED
    assert run.error is not None and run.error.type == "ServerShutdown"
    assert run.steps[-1].status is StepStatus.FAILED
    assert run.events[-1].type is RunEventType.RUN_FAILED


# ------------------------------------------------------------------ misc

def test_structured_cap_untouched_for_plain_calls():
    # guards a regression where the orchestrator might pass a cap directly
    assert DEFAULT_STRUCTURED_MAX_TOKENS == 2048
