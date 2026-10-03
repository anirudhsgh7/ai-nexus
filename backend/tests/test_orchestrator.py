"""Iterative orchestration contract (Phase 5 PRD §6.3–6.7 + Phase 11 §6.7).

Fixtures are queued to FakeProvider in exact request order (documented per
test). Two chains exist: SIMPLE (all claims supported -> deterministic finish,
Phase 4-compatible shape) and ITERATIVE (unresolved -> decide -> revise ->
re-critique -> decide -> synthesize). Every completed chain then runs the two
Phase 11 audit steps (verify, audit); `AuditAwareFakeProvider` synthesizes
those responses from the request itself because static payloads cannot satisfy
`validate_verification`'s coverage rule (each chain's selected-round claim ids
differ). Audit content semantics are tested in test_audits/test_agents_audit;
this module owns orchestration shape and isolation.
"""

import asyncio
import json
import re
from datetime import UTC, datetime

import pytest

from app.agents import build_registry
from app.agents.structured import ACCOUNTABILITY_SCHEMA, VERIFICATION_SCHEMA
from app.audits import compute_audit_facts
from app.llm.base import ChatRole
from app.orchestrator import (
    MAX_TOTAL_STEPS,
    ROUND_ONE_STEPS,
    ClaimOrigin,
    Orchestrator,
    PoolState,
    QualifiedClaim,
    qualify_claims,
    render_accountability_context,
    render_decision_summary,
    render_evidence_board,
    render_iteration_history,
    render_revision_context,
    render_synthesis_context,
    render_verification_context,
    resolve_claim,
    select_best_round,
)
from app.runs import (
    RoundSnapshot,
    RunEventType,
    RunManager,
    RunRecord,
    RunStatus,
    StepKind,
    StepRecord,
    StepStatus,
)
from app.schemas import (
    AccountabilityFlagKind,
    AccountabilityStatus,
    AgentMessage,
    AgentRole,
    Claim,
    ClaimStatus,
    ClaimVerdict,
    ClaimVerification,
    DecisionAction,
    Evidence,
    ManagerDecision,
    MessageType,
    VerificationReport,
    VerificationStatus,
    Verdict,
)
from tests.fakes import AuditAwareFakeProvider, FakeProvider

TASK = "Should we build X?"


# ---------------------------------------------------------------- payloads

def _claims_payload(content: str, claims: list[dict]) -> str:
    return json.dumps({"content": content, "claims": claims})


def _verdicts_payload(content: str, verdicts: list[dict]) -> str:
    return json.dumps({"content": content, "verdicts": verdicts})


def _decision_payload(action: str, **overrides) -> str:
    payload = {"action": action, "reason": "evidence gap remains", "confidence": 0.8}
    if action == "call_agent":
        payload.update({"target": "researcher",
                        "instruction": "Find a published source for the 40% claim."})
    payload.update(overrides)
    return json.dumps(payload)


PLAN_JSON = _claims_payload("Plan prose.", [])

# simple chain: every claim carries evidence so supported verdicts validate
RESEARCH_SIMPLE = _claims_payload("Research prose.", [
    {"statement": "Finding one", "status": "unverified",
     "evidence": [{"source": "doc A"}]},
    {"statement": "Finding two", "status": "fact",
     "evidence": [{"source": "report"}]},
])
IDEATION_SIMPLE = _claims_payload("Ideation prose.", [
    {"statement": "Idea A", "status": "hypothesis",
     "evidence": [{"source": "survey"}]},
])
CRITIQUE_ALL_SUPPORTED = _verdicts_payload("Critique prose.", [
    {"claim_id": "c1", "verdict": "supported", "objection": "doc A holds",
     "evidence": [{"source": "doc A"}]},
    {"claim_id": "c2", "verdict": "supported", "objection": "report holds",
     "evidence": [{"source": "report"}]},
    {"claim_id": "c3", "verdict": "supported", "objection": "survey holds",
     "evidence": [{"source": "survey"}]},
])
SYNTHESIS_JSON = _claims_payload("Final answer.", [])

# iterative chain: round-1 evidence gaps
RESEARCH_JSON = _claims_payload("Research prose.", [
    {"statement": "Finding one", "status": "unverified", "evidence": []},
    {"statement": "Finding two", "status": "fact",
     "evidence": [{"source": "report"}]},
])
IDEATION_JSON = _claims_payload("Ideation prose.", [
    {"statement": "Idea A", "status": "hypothesis", "evidence": []},
])
CRITIQUE_R1 = _verdicts_payload("Critique prose.", [
    {"claim_id": "c1", "verdict": "refuted", "objection": "no source exists",
     "evidence": []},
    {"claim_id": "c2", "verdict": "supported", "objection": "report holds",
     "evidence": [{"source": "report"}]},
    {"claim_id": "c3", "verdict": "unverifiable", "objection": "no data",
     "evidence": []},
])
DECISION_CALL = _decision_payload("call_agent")
DECISION_FINISH = _decision_payload("finish", reason="no useful work remains")
REVISION_JSON = _claims_payload("Revision prose.", [
    {"statement": "Revised finding A", "status": "fact",
     "evidence": [{"source": "audit report"}]},
    {"statement": "Revised finding B", "status": "unverified",
     "evidence": [{"source": "press release"}]},
])
CRITIQUE_R2 = _verdicts_payload("Critique prose round 2.", [
    {"claim_id": "c4", "verdict": "supported", "objection": "audit confirms",
     "evidence": [{"source": "audit report"}]},
    {"claim_id": "c5", "verdict": "unverifiable", "objection": "single source",
     "evidence": []},
])
CRITIQUE_R2_ALL_UNVER = _verdicts_payload("Critique prose round 2.", [
    {"claim_id": "c4", "verdict": "unverifiable", "objection": "no corroboration",
     "evidence": []},
    {"claim_id": "c5", "verdict": "unverifiable", "objection": "no corroboration",
     "evidence": []},
])


async def _run_pipeline(payloads, *, max_rounds: int = 3,
                        audit_error: Exception | None = None,
                        verify_error: Exception | None = None):
    provider = AuditAwareFakeProvider(fail_audit=audit_error,
                                      fail_verify=verify_error)
    for item in payloads:
        provider.queue_result(
            item if isinstance(item, Exception) else FakeProvider.make_result(item)
        )
    store = RunManager()
    run = store.create(TASK)
    orchestrator = Orchestrator(build_registry(provider), store, max_rounds=max_rounds)
    await orchestrator.execute(run.id)
    return provider, store, run


# ================================================================= pure helpers

def test_qualify_claims_continues_ids_with_start():
    batch = [(AgentRole.RESEARCHER, [Claim(id="x1", statement="A")])]
    assert [q.claim.id for q in qualify_claims(batch, start=4)] == ["c4"]
    assert qualify_claims(batch)[0].claim.id == "c1"


def test_resolve_claim():
    qualified = qualify_claims(
        [(AgentRole.IDEATOR, [Claim(id="x", statement="A")])]
    )
    assert resolve_claim(qualified, "c1").origin.agent is AgentRole.IDEATOR
    assert resolve_claim(qualified, "zz") is None


# ---------------------------------------------------------------- evidence board

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


# ---------------------------------------------------------------- decision summary

def _summary_pool():
    qualified = qualify_claims([
        (AgentRole.RESEARCHER, [
            Claim(id="x1", statement="Alpha holds.", status=ClaimStatus.UNVERIFIED),
            Claim(id="x2", statement="X grew 40% in 2025.",
                  status=ClaimStatus.UNVERIFIED),
            Claim(id="x3", statement="Beta holds.", status=ClaimStatus.UNVERIFIED),
            Claim(id="x4", statement="Gamma holds.", status=ClaimStatus.UNVERIFIED),
        ]),
        (AgentRole.IDEATOR, [
            Claim(id="x1", statement="Y is the cheaper option.",
                  status=ClaimStatus.UNVERIFIED),
            Claim(id="x2", statement="Delta idea.", status=ClaimStatus.UNVERIFIED),
        ]),
    ])
    verdicts = {
        "c1": Verdict(claim_id="c1", verdict=ClaimVerdict.SUPPORTED,
                      objection="doc holds"),
        "c2": Verdict(claim_id="c2", verdict=ClaimVerdict.UNVERIFIABLE,
                      objection="No published source was cited."),
        "c3": Verdict(claim_id="c3", verdict=ClaimVerdict.SUPPORTED,
                      objection="doc holds"),
        "c4": Verdict(claim_id="c4", verdict=ClaimVerdict.REFUTED,
                      objection="contradicted by the report."),
        "c5": Verdict(claim_id="c5", verdict=ClaimVerdict.REFUTED,
                      objection="Contradicts the pricing table."),
        "c6": Verdict(claim_id="c6", verdict=ClaimVerdict.SUPPORTED,
                      objection="survey holds"),
    }
    return qualified, verdicts


def test_decision_summary_golden_with_previous_instruction():
    active, verdicts = _summary_pool()
    previous = ManagerDecision(
        action=DecisionAction.CALL_AGENT, target=AgentRole.RESEARCHER,
        instruction="Find a published source for the 40% growth claim.",
        reason="gap", confidence=0.8,
    )
    out = render_decision_summary(2, 3, active, verdicts, previous)
    assert out == (
        "ROUND 2 OF 3\n"
        "\n"
        "PREVIOUS INSTRUCTION:\n"
        "researcher: Find a published source for the 40% growth claim.\n"
        "\n"
        "AGENT STATUS:\n"
        "- researcher: 4 claims (2 supported, 2 unresolved)\n"
        "- ideator: 2 claims (1 supported, 1 unresolved)\n"
        "\n"
        "UNRESOLVED CLAIMS:\n"
        "[c2] (researcher) verdict=unverifiable\n"
        "  Claim: X grew 40% in 2025.\n"
        "  Objection: No published source was cited.\n"
        "[c4] (researcher) verdict=refuted\n"
        "  Claim: Gamma holds.\n"
        "  Objection: contradicted by the report.\n"
        "[c5] (ideator) verdict=refuted\n"
        "  Claim: Y is the cheaper option.\n"
        "  Objection: Contradicts the pricing table."
    )


def test_decision_summary_without_previous_omits_block():
    active, verdicts = _summary_pool()
    out = render_decision_summary(1, 3, active, verdicts, None)
    assert out.startswith("ROUND 1 OF 3\n")
    assert "PREVIOUS INSTRUCTION" not in out
    assert "AGENT STATUS:" in out


def test_decision_summary_clips_long_fields():
    long_statement = "S" * 250
    long_objection = "O" * 250
    active = qualify_claims([
        (AgentRole.RESEARCHER, [Claim(id="x1", statement=long_statement)]),
    ])
    verdicts = {"c1": Verdict(claim_id="c1", verdict=ClaimVerdict.UNVERIFIABLE,
                              objection=long_objection)}
    out = render_decision_summary(1, 3, active, verdicts, None)
    assert "S" * 200 + "..." in out
    assert long_statement not in out
    assert "O" * 200 + "..." in out
    assert long_objection not in out


def test_decision_summary_caps_entries():
    active = qualify_claims([
        (AgentRole.RESEARCHER, [
            Claim(id=f"x{i}", statement=f"Claim {i}") for i in range(1, 14)
        ]),
    ])
    out = render_decision_summary(1, 3, active, {}, None)
    assert out.count("  Claim: ") == 10
    assert "... and 3 more unresolved claims" in out


def test_decision_summary_empty_unresolved():
    out = render_decision_summary(1, 3, [], {}, None)
    assert out.endswith("UNRESOLVED CLAIMS:\n(none)")


def test_decision_summary_renders_none_verdict():
    active = qualify_claims([(AgentRole.IDEATOR, [Claim(id="x1", statement="S")])])
    out = render_decision_summary(1, 3, active, {}, None)
    assert "[c1] (ideator) verdict=none" in out
    assert "Objection: none" in out


# ---------------------------------------------------------------- revision context

def test_revision_context_golden():
    board = (
        "[c1] (researcher) status=unverified confidence=none\n"
        "Claim: X grew 40%.\n"
        "Evidence: none\n"
        "Verdict: unverifiable — no source"
    )
    assert render_revision_context("Find a source.", board) == (
        "MANAGER INSTRUCTION:\n"
        "Find a source.\n"
        "\n"
        "YOUR CURRENT CLAIMS AND THEIR VERDICTS:\n" + board
    )


def test_revision_context_empty_board():
    out = render_revision_context("Explore X.", "")
    assert out.endswith("YOUR CURRENT CLAIMS AND THEIR VERDICTS:\n(none)")


# ---------------------------------------------------------------- iteration history

def _snap(number: int, supported: int, unresolved: int, **overrides) -> RoundSnapshot:
    return RoundSnapshot(
        round_number=number,
        claims=overrides.get("claims", []),
        origins=overrides.get("origins", {}),
        verdicts=overrides.get("verdicts", []),
        worker_content=overrides.get("worker_content", {}),
        skeptic_content=overrides.get("skeptic_content", ""),
        supported_count=supported,
        unresolved_count=unresolved,
    )


def test_iteration_history_golden():
    claims = [
        Claim(id="c4", statement="fixed claim"),
        Claim(id="c7", statement="still open"),
    ]
    verdicts = [
        Verdict(claim_id="c4", verdict=ClaimVerdict.SUPPORTED, objection="holds"),
        Verdict(claim_id="c7", verdict=ClaimVerdict.UNVERIFIABLE,
                objection="no data"),
    ]
    snap1 = _snap(1, 3, 5)
    snap2 = _snap(2, 6, 2, claims=claims,
                  origins={"c4": AgentRole.RESEARCHER,
                           "c7": AgentRole.IDEATOR},
                  verdicts=verdicts)
    out = render_iteration_history([snap1, snap2], snap2)
    assert out == (
        "ITERATION HISTORY:\n"
        "- round 1: 3 supported, 5 unresolved\n"
        "- round 2: 6 supported, 2 unresolved\n"
        "Selected round: 2 (net evidence score 4)\n"
        "\n"
        "REMAINING UNRESOLVED:\n"
        "[c7] (ideator) verdict=unverifiable"
    )


def test_iteration_history_no_remaining():
    out = render_iteration_history([_snap(1, 2, 0)], _snap(1, 2, 0))
    assert out.endswith("REMAINING UNRESOLVED:\n(none)")


# ---------------------------------------------------------------- synthesis context

def test_synthesis_context_forbids_meta_text():
    """Thin-evidence answers must be answers, not descriptions of answers."""
    context = render_synthesis_context(
        "researcher prose", "ideator prose", None, "board body"
    )
    lowered = context.lower()
    assert "write the answer itself" in lowered
    assert "never describe what an answer should contain" in lowered
    assert "do not make the missing evidence the whole answer" in lowered
    # materials still included
    assert "researcher prose" in context
    assert "ideator prose" in context
    assert "board body" in context


# ---------------------------------------------------------------- best round

def test_select_best_round_matrix():
    def snap(n, s, u):
        return _snap(n, s, u)

    assert select_best_round([snap(1, 3, 5), snap(2, 6, 2)]).round_number == 2
    assert select_best_round([snap(1, 6, 2), snap(2, 3, 5)]).round_number == 1
    # tie -> later round
    assert select_best_round([snap(1, 2, 0), snap(2, 2, 0)]).round_number == 2
    # net-positive beats an empty later round
    assert select_best_round([snap(1, 2, 0), snap(2, 0, 0)]).round_number == 1
    assert select_best_round([snap(1, 1, 1)]).round_number == 1


# ---------------------------------------------------------------- pool rules

def test_pool_add_assigns_ids_and_dedupes():
    pool = PoolState()
    added = pool.add(AgentRole.RESEARCHER, [
        Claim(id="x1", statement="Alpha holds"),
        Claim(id="x2", statement="ALPHA   HOLDS"),   # duplicate after normalize
    ])
    assert [q.claim.id for q in added] == ["c1"]
    assert pool.counter == 1
    pool.add(AgentRole.IDEATOR, [Claim(id="x1", statement="Totally different.")])
    assert [q.claim.id for q in pool.claims] == ["c1", "c2"]
    assert pool.counter == 2


def test_pool_unresolved_and_counts():
    pool = PoolState()
    pool.add(AgentRole.RESEARCHER, [
        Claim(id="x1", statement="A"),
        Claim(id="x2", statement="B"),
    ])
    pool.record_verdicts([
        Verdict(claim_id="c1", verdict=ClaimVerdict.SUPPORTED, objection="holds"),
        Verdict(claim_id="c2", verdict=ClaimVerdict.REFUTED, objection="nope"),
    ])
    assert pool.supported_count() == 1
    assert [q.claim.id for q in pool.unresolved()] == ["c2"]
    # claim without any verdict counts as unresolved
    pool.add(AgentRole.IDEATOR, [Claim(id="x1", statement="C")])
    assert [q.claim.id for q in pool.unresolved()] == ["c2", "c3"]


def test_pool_revision_keeps_supported_drops_unresolved_adds_new():
    pool = PoolState()
    pool.add(AgentRole.RESEARCHER, [
        Claim(id="x1", statement="kept claim"),
        Claim(id="x2", statement="dropped claim"),
    ])
    pool.record_verdicts([
        Verdict(claim_id="c1", verdict=ClaimVerdict.SUPPORTED, objection="holds"),
        Verdict(claim_id="c2", verdict=ClaimVerdict.REFUTED, objection="bad"),
    ])
    added, dropped = pool.apply_revision(AgentRole.RESEARCHER, [
        Claim(id="x1", statement="revised claim"),
        Claim(id="x2", statement="kept claim"),  # duplicate of active -> dropped
    ])
    assert [q.claim.id for q in dropped] == ["c2"]
    assert [q.claim.id for q in added] == ["c3"]          # IDs continue
    assert "c2" not in pool.verdicts                      # verdict pruned
    assert [q.claim.id for q in pool.claims] == ["c1", "c3"]
    # other agents untouched
    pool.add(AgentRole.IDEATOR, [Claim(id="x1", statement="ideator claim")])
    pool.record_verdicts([                            # settle the new revision
        Verdict(claim_id="c3", verdict=ClaimVerdict.SUPPORTED, objection="holds"),
    ])
    added2, dropped2 = pool.apply_revision(AgentRole.RESEARCHER, [])
    assert added2 == [] and dropped2 == []                # all supported -> no drop
    assert len(pool.claims) == 3


# ================================================================= simple path

async def _simple_pipeline():
    return await _run_pipeline(
        [PLAN_JSON, RESEARCH_SIMPLE, IDEATION_SIMPLE, CRITIQUE_ALL_SUPPORTED,
         SYNTHESIS_JSON]
    )


async def test_simple_path_stops_at_round_one():
    provider, store, run = await _simple_pipeline()
    assert run.status is RunStatus.COMPLETED
    assert run.error is None
    # 5 pipeline calls + verify + audit; NO decision call may happen
    assert len(provider.chat_calls) == 7, "no decision call may happen"
    assert [s.kind for s in run.steps] == [
        StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE,
        StepKind.CRITIQUE, StepKind.SYNTHESIZE, StepKind.VERIFY,
        StepKind.AUDIT,
    ]
    assert all(s.status is StepStatus.COMPLETED for s in run.steps)
    # no synthetic decision: all steps ran through the LLM
    assert not any(s.skipped for s in run.steps)


async def test_simple_path_event_shape_matches_phase4():
    _, _, run = await _simple_pipeline()
    types = [e.type for e in run.events]
    expected = [RunEventType.RUN_STARTED]
    for _ in range(7):  # plan research ideate critique synthesize verify audit
        expected += [RunEventType.STEP_STARTED, RunEventType.STEP_COMPLETED]
    expected += [RunEventType.RUN_COMPLETED]
    assert types == expected
    assert [e.seq for e in run.events] == list(range(1, 17))


async def test_simple_path_single_round_snapshot():
    _, _, run = await _simple_pipeline()
    assert len(run.rounds) == 1
    snapshot = run.rounds[0]
    assert snapshot.round_number == 1
    assert snapshot.supported_count == 3
    assert snapshot.unresolved_count == 0
    assert snapshot.decision is None, "deterministic finish records no decision"


async def test_researcher_and_ideator_context_is_byte_identical():
    provider, _, _ = await _simple_pipeline()
    researcher_user = provider.chat_calls[1]["messages"][1].content
    ideator_user = provider.chat_calls[2]["messages"][1].content
    assert researcher_user == ideator_user
    assert researcher_user == f"TASK:\n{TASK}\n\nCONTEXT:\nPlan prose."


async def test_skeptic_receives_records_not_peer_prose():
    provider, _, _ = await _simple_pipeline()
    skeptic_user = provider.chat_calls[3]["messages"][1].content
    assert "CLAIMS TO EVALUATE:" in skeptic_user
    assert "Research prose." not in skeptic_user
    assert "Ideation prose." not in skeptic_user
    assert "Plan prose." not in skeptic_user


async def test_simple_synthesis_has_sections_and_history():
    provider, _, _ = await _simple_pipeline()
    synthesis_user = provider.chat_calls[4]["messages"][1].content
    for header in ("RESEARCHER FINDINGS:", "IDEATOR OPTIONS:",
                   "SKEPTIC CRITIQUE:", "EVALUATED CLAIMS:", "ITERATION HISTORY:"):
        assert header in synthesis_user
    assert "[c1] (researcher) status=unverified" in synthesis_user
    assert "Verdict: supported — doc A holds" in synthesis_user
    assert synthesis_user.endswith("REMAINING UNRESOLVED:\n(none)")


async def test_final_message_and_step_rounds():
    _, _, run = await _simple_pipeline()
    assert run.final_message is not None
    assert run.final_message.content == "Final answer."
    assert [s.round for s in run.steps] == [None, 1, 1, 1, None, None, None]


# ================================================================= iterative path

async def _iterative_pipeline():
    return await _run_pipeline([
        PLAN_JSON, RESEARCH_JSON, IDEATION_JSON, CRITIQUE_R1,
        DECISION_CALL, REVISION_JSON, CRITIQUE_R2, DECISION_FINISH,
        SYNTHESIS_JSON,
    ])


async def test_iterative_full_step_trace():
    provider, store, run = await _iterative_pipeline()
    assert run.status is RunStatus.COMPLETED
    assert len(provider.chat_calls) == 11  # 9 + verify + audit
    assert [s.kind for s in run.steps] == [
        StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE, StepKind.CRITIQUE,
        StepKind.DECIDE, StepKind.REVISE, StepKind.CRITIQUE, StepKind.DECIDE,
        StepKind.SYNTHESIZE, StepKind.VERIFY, StepKind.AUDIT,
    ]
    assert [s.round for s in run.steps] == [
        None, 1, 1, 1, 1, 2, 2, 2, None, None, None,
    ]
    assert all(s.status is StepStatus.COMPLETED for s in run.steps)


async def test_iterative_event_sequence_and_rounds():
    _, _, run = await _iterative_pipeline()
    types = [e.type for e in run.events]
    expected = [RunEventType.RUN_STARTED]
    for _ in range(11):
        expected += [RunEventType.STEP_STARTED, RunEventType.STEP_COMPLETED]
    expected += [RunEventType.RUN_COMPLETED]
    assert types == expected
    assert [e.seq for e in run.events] == list(range(1, 25))
    decide_events = [
        e for e in run.events
        if e.type is RunEventType.STEP_COMPLETED and e.kind is StepKind.DECIDE
    ]
    assert [e.round for e in decide_events] == [1, 2]


async def test_iterative_round_snapshots_and_decisions():
    _, _, run = await _iterative_pipeline()
    assert len(run.rounds) == 2
    r1, r2 = run.rounds
    assert (r1.supported_count, r1.unresolved_count) == (1, 2)
    assert (r2.supported_count, r2.unresolved_count) == (2, 2)
    assert r1.decision is not None
    assert r1.decision.action is DecisionAction.CALL_AGENT
    assert r1.decision.target is AgentRole.RESEARCHER
    assert r2.decision is not None
    assert r2.decision.action is DecisionAction.FINISH
    assert "no useful work remains" in r2.decision.reason


async def test_decision_context_has_summary_not_prose():
    provider, _, _ = await _iterative_pipeline()
    decision_user = provider.chat_calls[4]["messages"][1].content
    assert decision_user.startswith(f"TASK:\n{TASK}\n\nCONTEXT:\nROUND 1 OF 3")
    assert "AGENT STATUS:" in decision_user
    assert "UNRESOLVED CLAIMS:" in decision_user
    assert "[c1] (researcher) verdict=refuted" in decision_user
    assert "[c3] (ideator) verdict=unverifiable" in decision_user
    # worker prose must never reach the router
    assert "Research prose." not in decision_user
    assert "Ideation prose." not in decision_user
    assert "Critique prose." not in decision_user
    # second decision includes the previous instruction
    second_decision_user = provider.chat_calls[7]["messages"][1].content
    assert "PREVIOUS INSTRUCTION:" in second_decision_user
    assert "Find a published source for the 40% claim." in second_decision_user
    assert second_decision_user.startswith(f"TASK:\n{TASK}\n\nCONTEXT:\nROUND 2 OF 3")


async def test_revision_context_isolation():
    provider, _, _ = await _iterative_pipeline()
    revision_user = provider.chat_calls[5]["messages"][1].content
    assert "MANAGER INSTRUCTION:\nFind a published source for the 40% claim." in revision_user
    # the researcher sees only its own claims (c1 dropped-unresolved, c2 supported)
    assert "[c2] (researcher)" in revision_user
    assert "Verdict: supported — report holds" in revision_user
    # never the peer's claims, peer prose, plan, or skeptic prose
    assert "Idea A" not in revision_user
    assert "[c3]" not in revision_user
    assert "Ideation prose." not in revision_user
    assert "Research prose." not in revision_user
    assert "Plan prose." not in revision_user
    assert "Critique prose" not in revision_user
    # revision turn goes to the researcher's real role config
    assert provider.chat_calls[5]["messages"][0].content.startswith(
        "You are the Researcher in AI Nexus"
    )


async def test_second_critique_evaluates_only_new_claims():
    provider, _, _ = await _iterative_pipeline()
    second_critique_user = provider.chat_calls[6]["messages"][1].content
    assert "[c4]" in second_critique_user
    assert "[c5]" in second_critique_user
    # carried-forward supported claim and prior findings are NOT re-evaluated
    assert "[c2]" not in second_critique_user
    assert "Finding two" not in second_critique_user
    assert "Idea A" not in second_critique_user


async def test_synthesis_uses_best_round_material():
    provider, _, _ = await _iterative_pipeline()
    synthesis_user = provider.chat_calls[8]["messages"][1].content
    # best round = 2 (net 0) over round 1 (net -1)
    assert "Selected round: 2 (net evidence score 0)" in synthesis_user
    assert "Revision prose." in synthesis_user      # researcher content from round 2
    assert "Research prose." not in synthesis_user  # round-1 researcher content dropped
    assert "Ideation prose." in synthesis_user      # never revised
    assert "[c4] (researcher)" in synthesis_user
    assert "REMAINING UNRESOLVED:" in synthesis_user
    assert "[c3] (ideator) verdict=unverifiable" in synthesis_user
    assert "- round 1: 1 supported, 2 unresolved" in synthesis_user
    assert "- round 2: 2 supported, 2 unresolved" in synthesis_user


# ================================================================= guards


def _synthetic_decide(run) -> StepRecord:
    """The guard's skipped DECIDE (its position moves once verify/audit follow)."""
    return next(s for s in run.steps if s.skipped and s.kind is StepKind.DECIDE)


async def test_guard_round_cap_forces_finish():
    provider, store, run = await _run_pipeline(
        [PLAN_JSON, RESEARCH_JSON, IDEATION_JSON, CRITIQUE_R1, SYNTHESIS_JSON],
        max_rounds=1,
    )
    assert run.status is RunStatus.COMPLETED
    # 4 pipeline calls + synthesis + verify + audit; NO decision call
    assert len(provider.chat_calls) == 7, "no decision call once capped"
    assert [s.kind for s in run.steps] == [
        StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE, StepKind.CRITIQUE,
        StepKind.DECIDE, StepKind.SYNTHESIZE, StepKind.VERIFY, StepKind.AUDIT,
    ]
    synthetic = _synthetic_decide(run)
    assert synthetic.skipped and synthetic.status is StepStatus.SKIPPED
    assert synthetic.kind is StepKind.DECIDE
    assert synthetic.message is not None
    decision = synthetic.message.decision
    assert decision.action is DecisionAction.FINISH
    assert "round cap reached (max_rounds=1)" == decision.reason
    assert synthetic.round == 1


async def test_guard_repeated_decision_forces_finish():
    provider, store, run = await _run_pipeline([
        PLAN_JSON, RESEARCH_JSON, IDEATION_JSON, CRITIQUE_R1,
        DECISION_CALL, REVISION_JSON, CRITIQUE_R2_ALL_UNVER, DECISION_CALL,
        SYNTHESIS_JSON,
    ])
    assert run.status is RunStatus.COMPLETED
    assert len(provider.chat_calls) == 11  # 9 + verify + audit
    assert [s.kind for s in run.steps] == [
        StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE, StepKind.CRITIQUE,
        StepKind.DECIDE, StepKind.REVISE, StepKind.CRITIQUE, StepKind.DECIDE,
        StepKind.DECIDE, StepKind.SYNTHESIZE, StepKind.VERIFY, StepKind.AUDIT,
    ]
    synthetic = _synthetic_decide(run)
    assert synthetic.skipped and synthetic.message.decision.reason == "repeated decision"


async def test_guard_no_progress_revision_forces_finish():
    critique_mixed = _verdicts_payload("Critique prose.", [
        {"claim_id": "c1", "verdict": "supported", "objection": "doc A holds",
         "evidence": [{"source": "doc A"}]},
        {"claim_id": "c2", "verdict": "supported", "objection": "report holds",
         "evidence": [{"source": "report"}]},
        {"claim_id": "c3", "verdict": "unverifiable", "objection": "no data",
         "evidence": []},
    ])
    provider, store, run = await _run_pipeline([
        PLAN_JSON, RESEARCH_SIMPLE, IDEATION_SIMPLE, critique_mixed,
        DECISION_CALL,                      # targets researcher (all supported)
        _claims_payload("Revision prose.", []),  # adds nothing; drops nothing
        SYNTHESIS_JSON,
    ])
    assert run.status is RunStatus.COMPLETED
    assert len(provider.chat_calls) == 9  # 7 + verify + audit
    assert [s.kind for s in run.steps] == [
        StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE, StepKind.CRITIQUE,
        StepKind.DECIDE, StepKind.REVISE, StepKind.DECIDE, StepKind.SYNTHESIZE,
        StepKind.VERIFY, StepKind.AUDIT,
    ]
    synthetic = _synthetic_decide(run)
    assert synthetic.skipped
    assert synthetic.message.decision.reason == "revision produced no progress"
    # The synthetic decide closes the round the revise opened: both carry the
    # pending round (2), never the previous round — group labels stay monotonic
    # (Setup / Round 1 / Round 2 / Wrap-up, no phantom Round 1 after Round 2).
    revise = next(s for s in run.steps if s.kind is StepKind.REVISE)
    assert revise.round == 2
    assert synthetic.round == 2
    # An aborted round is not recorded as a snapshot: best-round selection
    # keeps round 1 (identical counts would otherwise tiebreak to a round with
    # no evaluated claims).
    assert len(run.rounds) == 1
    assert [s.round for s in run.steps] == [
        None, 1, 1, 1, 1, 2, 2, None, None, None,
    ]


async def test_guard_step_cap_forces_finish(monkeypatch):
    monkeypatch.setattr("app.orchestrator.MAX_TOTAL_STEPS", 4)
    provider, store, run = await _run_pipeline(
        [PLAN_JSON, RESEARCH_JSON, IDEATION_JSON, CRITIQUE_R1, SYNTHESIS_JSON]
    )
    assert run.status is RunStatus.COMPLETED
    assert len(provider.chat_calls) == 7  # 4 capped steps + synthesis + audits
    assert _synthetic_decide(run).message.decision.reason == "step cap reached"
    assert len(run.steps) == 8  # 4 + synthetic decide + synthesize + verify + audit


# ================================================================= degenerate paths

async def test_zero_claims_skips_critique_and_finishes():
    empty_research = _claims_payload("Research prose.", [])
    empty_ideation = _claims_payload("Ideation prose.", [])
    provider, store, run = await _run_pipeline(
        [PLAN_JSON, empty_research, empty_ideation, SYNTHESIS_JSON]
    )
    assert len(provider.chat_calls) == 6  # plan research ideate synth verify audit
    assert run.status is RunStatus.COMPLETED
    assert [s.status for s in run.steps] == [
        StepStatus.COMPLETED, StepStatus.COMPLETED, StepStatus.COMPLETED,
        StepStatus.SKIPPED, StepStatus.COMPLETED, StepStatus.COMPLETED,
        StepStatus.COMPLETED,
    ]
    assert [s.round for s in run.steps] == [None, 1, 1, 1, None, None, None]
    assert len(run.rounds) == 1 and run.rounds[0].unresolved_count == 0

    synthesis_user = provider.chat_calls[3]["messages"][1].content
    assert "(skipped: no claims were produced to evaluate)" in synthesis_user
    assert "EVALUATED CLAIMS:\n(none)" in synthesis_user
    assert synthesis_user.endswith("REMAINING UNRESOLVED:\n(none)")

    # Phase 11 §6.7: the verifier still runs and must return an empty report
    verify = next(s for s in run.steps if s.kind is StepKind.VERIFY)
    assert verify.message is not None
    assert verify.message.verification is not None
    assert verify.message.verification.claims == []


async def test_researcher_failure_aborts_run():
    from app.llm.base import ProviderUnavailableError

    provider, store, run = await _run_pipeline(
        [PLAN_JSON, ProviderUnavailableError()]
    )
    assert len(provider.chat_calls) == 2
    assert run.status is RunStatus.FAILED
    assert [s.status for s in run.steps] == [StepStatus.COMPLETED, StepStatus.FAILED]
    last = run.events[-1]
    assert last.type is RunEventType.RUN_FAILED
    assert last.step == 2 and last.kind is StepKind.RESEARCH
    assert "ollama serve" in last.error.hint


async def test_structured_failure_at_skeptic_aborts():
    provider, store, run = await _run_pipeline(
        [PLAN_JSON, RESEARCH_JSON, IDEATION_JSON, "not json", "still not json"]
    )
    assert len(provider.chat_calls) == 5
    assert run.status is RunStatus.FAILED
    assert run.steps[-1].error.type == "StructuredOutputError"
    assert run.events[-1].type is RunEventType.RUN_FAILED
    assert run.events[-1].kind is StepKind.CRITIQUE


async def test_structured_failure_at_decision_aborts():
    provider, store, run = await _run_pipeline(
        [PLAN_JSON, RESEARCH_JSON, IDEATION_JSON, CRITIQUE_R1,
         "garbage one", "garbage two"]
    )
    assert run.status is RunStatus.FAILED
    assert run.steps[-1].error.type == "StructuredOutputError"
    assert run.steps[-1].kind is StepKind.DECIDE
    assert run.final_message is None


# ================================================================= cancellation

class SlowProvider(FakeProvider):
    async def chat(self, messages, **kwargs):  # type: ignore[override]
        self.chat_calls.append({"messages": list(messages), "kwargs": dict(kwargs)})
        await asyncio.sleep(30)
        raise AssertionError("unreachable")


async def test_cancellation_marks_run_failed():
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
    assert run.error.type == "ServerShutdown"
    assert run.steps[-1].status is StepStatus.FAILED
    assert run.events[-1].type is RunEventType.RUN_FAILED


# ================================================== Phase 11 audit steps


def _step_record(index, kind, agent, *, round=None, skipped=False,
                 message=None) -> StepRecord:
    return StepRecord(
        index=index, kind=kind, agent=agent,
        status=StepStatus.SKIPPED if skipped else StepStatus.COMPLETED,
        started_at=datetime.now(UTC), round=round, skipped=skipped,
        message=message,
    )


def _verification_entry(cid: str, status: VerificationStatus,
                        *, supporting=None, sources=None) -> ClaimVerification:
    return ClaimVerification(
        claim_id=cid,
        verification_status=status,
        evidence_checked=["doc"] if supporting else [],
        supporting_evidence=supporting or [],
        source_references=sources or [],
        explanation="checked the record",
        confidence=0.8,
    )


def test_render_verification_context_golden():
    claims = [
        Claim(id="c1", statement="A", status=ClaimStatus.FACT,
              evidence=[Evidence(source="doc A")]),
        Claim(id="c2", statement="B"),
        Claim(id="c3", statement="C"),
    ]
    verdicts = [
        Verdict(claim_id="c1", verdict=ClaimVerdict.SUPPORTED,
                objection="doc A holds"),
        Verdict(claim_id="c2", verdict=ClaimVerdict.REFUTED,
                objection="no source exists"),
    ]
    out = render_verification_context("Final answer.", claims, verdicts)
    assert out == (
        "FINAL ANSWER:\nFinal answer.\n\n"
        "PRIOR VERDICTS (context only — never grounds for verification):\n"
        "[c1] supported — doc A holds\n"
        "[c2] refuted — no source exists\n"
        "[c3] none"
    )


def test_render_verification_context_empty_claims():
    out = render_verification_context("Answer.", [], [])
    assert out == (
        "FINAL ANSWER:\nAnswer.\n\n"
        "PRIOR VERDICTS (context only — never grounds for verification): (none)"
    )


def _golden_audit_run() -> RunRecord:
    run = RunManager().create(TASK)
    synth_msg = AgentMessage(
        from_agent=AgentRole.MANAGER, type=MessageType.SYNTHESIS,
        content="Final answer.",
    )
    decide_msg = AgentMessage(
        from_agent=AgentRole.MANAGER, type=MessageType.DECISION, content="",
        decision=ManagerDecision(
            action=DecisionAction.CALL_AGENT, target=AgentRole.RESEARCHER,
            instruction="find a source", reason="evidence gap", confidence=0.8,
        ),
    )
    guard_msg = AgentMessage(
        from_agent=AgentRole.MANAGER, type=MessageType.DECISION, content="",
        decision=ManagerDecision(
            action=DecisionAction.FINISH,
            reason="revision produced no progress", confidence=0.0,
        ),
    )
    run.steps.extend([
        _step_record(1, StepKind.PLAN, AgentRole.MANAGER),
        _step_record(2, StepKind.RESEARCH, AgentRole.RESEARCHER, round=1),
        _step_record(3, StepKind.IDEATE, AgentRole.IDEATOR, round=1),
        _step_record(4, StepKind.CRITIQUE, AgentRole.SKEPTIC, round=1),
        _step_record(5, StepKind.DECIDE, AgentRole.MANAGER, round=1,
                    message=decide_msg),
        _step_record(6, StepKind.REVISE, AgentRole.RESEARCHER, round=2),
        _step_record(7, StepKind.DECIDE, AgentRole.MANAGER, round=2,
                    skipped=True, message=guard_msg),
        _step_record(8, StepKind.SYNTHESIZE, AgentRole.MANAGER, message=synth_msg),
        _step_record(9, StepKind.VERIFY, AgentRole.VERIFIER),
    ])
    claims = [
        Claim(id="c1", statement="Holds.", status=ClaimStatus.FACT,
              evidence=[Evidence(source="doc")]),
        Claim(id="c2", statement="Open."),
    ]
    verdicts = [
        Verdict(claim_id="c1", verdict=ClaimVerdict.SUPPORTED, objection="holds"),
        Verdict(claim_id="c2", verdict=ClaimVerdict.UNVERIFIABLE,
                objection="no corroboration"),
    ]
    origins = {"c1": AgentRole.RESEARCHER, "c2": AgentRole.IDEATOR}
    for number in (1, 2):
        run.rounds.append(RoundSnapshot(
            round_number=number, claims=claims, origins=origins,
            verdicts=verdicts, worker_content={}, skeptic_content="",
            supported_count=1, unresolved_count=1,
        ))
    return run


def test_render_accountability_context_golden():
    run = _golden_audit_run()
    facts = compute_audit_facts(run.rounds, run.steps, "Final answer.")
    verification = VerificationReport(claims=[
        _verification_entry("c1", VerificationStatus.VERIFIED,
                            supporting=[Evidence(source="doc")],
                            sources=["doc"]),
        _verification_entry("c2", VerificationStatus.UNVERIFIABLE),
    ])
    out = render_accountability_context(run, facts, verification)
    assert out == (
        "RUN TRACE FACTS:\n"
        "steps: plan=completed research=completed ideate=completed "
        "critique=completed decide=completed revise=completed decide=skipped "
        "synthesize=completed verify=completed\n"
        "decisions: r1 call_agent→researcher confidence=0.80 | "
        "r2 guard_finish(revision produced no progress)\n"
        "guards: round 2 revision produced no progress\n"
        "failed_steps: none\n"
        "retries: total=0 steps=[]\n"
        "selected_round: 2 supported=1 unresolved=1\n"
        "final_claims: 2\n"
        "\n"
        "FINAL CLAIM PROVENANCE:\n"
        "[c1] origin=researcher verdict=supported evidence=1\n"
        "[c2] origin=ideator verdict=unverifiable evidence=0\n"
        "\n"
        "UNRESOLVED AT SYNTHESIS:\n"
        "[c2] (ideator) verdict=unverifiable — no corroboration\n"
        "\n"
        "VERIFICATION SUMMARY:\n"
        "verified=1 partially_verified=0 contradicted=0 unverifiable=1\n"
        "\n"
        "FINAL ANSWER:\n"
        "Final answer.\n"
        "\n"
        "FACT ENTRIES (every entry MUST appear in your flags with the same "
        "kind/severity/refs):\n"
        "- kind=unsupported_final_claim severity=warning refs=[c2] "
        "detail=final claim has no supported verdict\n"
        "- kind=unresolved_claim_suppressed severity=warning refs=[c2] "
        "detail=final answer neither names the unresolved claims nor admits "
        "uncertainty\n"
        "- kind=premature_stop severity=warning refs=[2] "
        "detail=guard stopped the run while unresolved claims remained"
    )


def test_render_accountability_context_caps_unresolved_at_ten():
    run = RunManager().create(TASK)
    claims = [
        Claim(id=f"c{i}", statement=f"s{i}") for i in range(1, 13)
    ]
    run.rounds.append(RoundSnapshot(
        round_number=1, claims=claims,
        origins={c.id: AgentRole.RESEARCHER for c in claims},
        verdicts=[], worker_content={}, skeptic_content="",
        supported_count=0, unresolved_count=12,
    ))
    facts = compute_audit_facts(run.rounds, [], None)
    out = render_accountability_context(run, facts, None)
    assert "... and 2 more unresolved claims" in out


async def test_audit_inputs_contain_no_worker_prose():
    provider, _, _ = await _simple_pipeline()
    verify_call = next(
        c for c in provider.chat_calls
        if c["kwargs"].get("response_format") is VERIFICATION_SCHEMA
    )
    verify_user = verify_call["messages"][1].content
    assert "FINAL ANSWER:\nFinal answer." in verify_user
    assert "CLAIMS TO EVALUATE:" in verify_user
    assert "PRIOR VERDICTS (context only" in verify_user
    for prose in ("Research prose.", "Ideation prose.",
                  "Critique prose.", "Plan prose."):
        assert prose not in verify_user, prose

    audit_call = next(
        c for c in provider.chat_calls
        if c["kwargs"].get("response_format") is ACCOUNTABILITY_SCHEMA
    )
    audit_user = audit_call["messages"][1].content
    assert "RUN TRACE FACTS:" in audit_user
    assert "FACT ENTRIES" in audit_user
    assert "FINAL ANSWER:\nFinal answer." in audit_user
    for prose in ("Research prose.", "Ideation prose.", "Critique prose."):
        assert prose not in audit_user, prose


async def test_audit_enforcement_merges_facts_into_report():
    _, _, run = await _iterative_pipeline()
    audit_step = run.steps[-1]
    assert audit_step.kind is StepKind.AUDIT
    report = audit_step.message.accountability
    assert report is not None
    # The provider said "clean" with empty provenance — code-canonical facts
    # must have overridden both (PRD §6.5.4).
    assert report.overall_status is AccountabilityStatus.WARNINGS
    kinds = {flag.kind for flag in report.flags}
    assert AccountabilityFlagKind.UNSUPPORTED_FINAL_CLAIM in kinds
    assert AccountabilityFlagKind.UNRESOLVED_CLAIM_SUPPRESSED in kinds
    unsupported = next(
        flag for flag in report.flags
        if flag.kind is AccountabilityFlagKind.UNSUPPORTED_FINAL_CLAIM
    )
    # c3 is the ideator's carried claim (only the researcher's claims are
    # dropped on revision) and c5 is the unverifiable revision addition
    assert unsupported.refs == ["c3", "c5"]
    assert [p.claim_id for p in report.final_claim_provenance] == [
        "c2", "c3", "c4", "c5",
    ]
    assert report.trace_completeness is True
    # enforcement left the persisted message and event identical (transform
    # runs before publication — assert the step record carries it)
    assert audit_step.message is not None


async def test_audit_step_failure_fails_run():
    from app.llm.base import ProviderUnavailableError

    provider, store, run = await _run_pipeline(
        [PLAN_JSON, RESEARCH_SIMPLE, IDEATION_SIMPLE, CRITIQUE_ALL_SUPPORTED,
         SYNTHESIS_JSON],
        audit_error=ProviderUnavailableError(),
    )
    assert run.status is RunStatus.FAILED
    audit = run.steps[-1]
    assert audit.kind is StepKind.AUDIT
    assert audit.status is StepStatus.FAILED
    assert audit.error is not None
    assert audit.error.type == "ProviderUnavailableError"
    # the answer is not final without its audits
    assert run.final_message is None
    assert run.events[-1].type is RunEventType.RUN_FAILED
    assert run.events[-1].kind is StepKind.AUDIT
    # verify step completed before the audit failure
    assert run.steps[-2].kind is StepKind.VERIFY
    assert run.steps[-2].status is StepStatus.COMPLETED


async def test_verify_step_failure_fails_run_and_skips_audit():
    """A VERIFY failure follows the single failure path: step FAILED, run
    FAILED, AUDIT never runs, no final message (PRD §6.7 / §8)."""
    from app.llm.base import ProviderUnavailableError

    provider, store, run = await _run_pipeline(
        [PLAN_JSON, RESEARCH_SIMPLE, IDEATION_SIMPLE, CRITIQUE_ALL_SUPPORTED,
         SYNTHESIS_JSON],
        verify_error=ProviderUnavailableError(),
    )
    assert run.status is RunStatus.FAILED
    verify = run.steps[-1]
    assert verify.kind is StepKind.VERIFY
    assert verify.status is StepStatus.FAILED
    assert verify.error is not None
    assert verify.error.type == "ProviderUnavailableError"
    # the audit never runs after a failed verification
    assert not any(step.kind is StepKind.AUDIT for step in run.steps)
    assert run.final_message is None
    assert run.events[-1].type is RunEventType.RUN_FAILED
    assert run.events[-1].kind is StepKind.VERIFY
    # synthesis completed before the verification failure
    assert run.steps[-2].kind is StepKind.SYNTHESIZE
    assert run.steps[-2].status is StepStatus.COMPLETED


async def test_audits_complete_without_injected_errors():
    """Baseline: both audit steps run to completion on a healthy chain."""
    provider, store, run = await _run_pipeline(
        [PLAN_JSON, RESEARCH_SIMPLE, IDEATION_SIMPLE, CRITIQUE_ALL_SUPPORTED,
         SYNTHESIS_JSON],
    )
    assert run.status is RunStatus.COMPLETED
    assert [s.kind for s in run.steps][-2:] == [StepKind.VERIFY, StepKind.AUDIT]
    assert all(
        step.status is StepStatus.COMPLETED for step in run.steps
    )
    # the answer is only final after both audits passed
    assert run.final_message is not None
