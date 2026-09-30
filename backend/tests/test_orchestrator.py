"""Iterative orchestration contract (Phase 5 PRD §6.3–6.7).

Fixtures are queued to FakeProvider in exact request order (documented per
test). Two chains exist: SIMPLE (all claims supported -> deterministic finish,
Phase 4-compatible 5-step shape) and ITERATIVE (unresolved -> decide -> revise
-> re-critique -> decide -> synthesize).
"""

import asyncio
import json

import pytest

from app.agents import build_registry
from app.orchestrator import (
    MAX_TOTAL_STEPS,
    ROUND_ONE_STEPS,
    ClaimOrigin,
    Orchestrator,
    PoolState,
    QualifiedClaim,
    qualify_claims,
    render_decision_summary,
    render_evidence_board,
    render_iteration_history,
    render_revision_context,
    render_synthesis_context,
    resolve_claim,
    select_best_round,
)
from app.runs import (
    RoundSnapshot,
    RunEventType,
    RunManager,
    RunStatus,
    StepKind,
    StepStatus,
)
from app.schemas import (
    AgentRole,
    Claim,
    ClaimStatus,
    ClaimVerdict,
    DecisionAction,
    Evidence,
    ManagerDecision,
    MessageType,
    Verdict,
)
from tests.fakes import FakeProvider

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


async def _run_pipeline(payloads, *, max_rounds: int = 3):
    provider = FakeProvider()
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
    assert len(provider.chat_calls) == 5, "no decision call may happen"
    assert [s.kind for s in run.steps] == [
        StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE,
        StepKind.CRITIQUE, StepKind.SYNTHESIZE,
    ]
    assert all(s.status is StepStatus.COMPLETED for s in run.steps)
    # no synthetic decision: all steps ran through the LLM
    assert not any(s.skipped for s in run.steps)


async def test_simple_path_event_shape_matches_phase4():
    _, _, run = await _simple_pipeline()
    types = [e.type for e in run.events]
    expected = [RunEventType.RUN_STARTED]
    for _ in range(5):
        expected += [RunEventType.STEP_STARTED, RunEventType.STEP_COMPLETED]
    expected += [RunEventType.RUN_COMPLETED]
    assert types == expected
    assert [e.seq for e in run.events] == list(range(1, 13))


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
    assert [s.round for s in run.steps] == [None, 1, 1, 1, None]


# ================================================================= iterative path

async def _iterative_pipeline():
    return await _run_pipeline([
        PLAN_JSON, RESEARCH_JSON, IDEATION_JSON, CRITIQUE_R1,
        DECISION_CALL, REVISION_JSON, CRITIQUE_R2, DECISION_FINISH,
        SYNTHESIS_JSON,
    ])


async def test_iterative_nine_step_trace():
    provider, store, run = await _iterative_pipeline()
    assert run.status is RunStatus.COMPLETED
    assert len(provider.chat_calls) == 9
    assert [s.kind for s in run.steps] == [
        StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE, StepKind.CRITIQUE,
        StepKind.DECIDE, StepKind.REVISE, StepKind.CRITIQUE, StepKind.DECIDE,
        StepKind.SYNTHESIZE,
    ]
    assert [s.round for s in run.steps] == [None, 1, 1, 1, 1, 2, 2, 2, None]
    assert all(s.status is StepStatus.COMPLETED for s in run.steps)


async def test_iterative_event_sequence_and_rounds():
    _, _, run = await _iterative_pipeline()
    types = [e.type for e in run.events]
    expected = [RunEventType.RUN_STARTED]
    for _ in range(9):
        expected += [RunEventType.STEP_STARTED, RunEventType.STEP_COMPLETED]
    expected += [RunEventType.RUN_COMPLETED]
    assert types == expected
    assert [e.seq for e in run.events] == list(range(1, 21))
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

async def test_guard_round_cap_forces_finish():
    provider, store, run = await _run_pipeline(
        [PLAN_JSON, RESEARCH_JSON, IDEATION_JSON, CRITIQUE_R1, SYNTHESIS_JSON],
        max_rounds=1,
    )
    assert run.status is RunStatus.COMPLETED
    # 4 pipeline calls + synthesis; NO decision call
    assert len(provider.chat_calls) == 5, "no decision call once capped"
    assert [s.kind for s in run.steps] == [
        StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE, StepKind.CRITIQUE,
        StepKind.DECIDE, StepKind.SYNTHESIZE,
    ]
    synthetic = run.steps[-2]
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
    assert len(provider.chat_calls) == 9
    assert [s.kind for s in run.steps] == [
        StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE, StepKind.CRITIQUE,
        StepKind.DECIDE, StepKind.REVISE, StepKind.CRITIQUE, StepKind.DECIDE,
        StepKind.DECIDE, StepKind.SYNTHESIZE,
    ]
    synthetic = run.steps[-2]
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
    assert len(provider.chat_calls) == 7
    assert [s.kind for s in run.steps] == [
        StepKind.PLAN, StepKind.RESEARCH, StepKind.IDEATE, StepKind.CRITIQUE,
        StepKind.DECIDE, StepKind.REVISE, StepKind.DECIDE, StepKind.SYNTHESIZE,
    ]
    synthetic = run.steps[-2]
    assert synthetic.skipped
    assert synthetic.message.decision.reason == "revision produced no progress"


async def test_guard_step_cap_forces_finish(monkeypatch):
    monkeypatch.setattr("app.orchestrator.MAX_TOTAL_STEPS", 4)
    provider, store, run = await _run_pipeline(
        [PLAN_JSON, RESEARCH_JSON, IDEATION_JSON, CRITIQUE_R1, SYNTHESIS_JSON]
    )
    assert run.status is RunStatus.COMPLETED
    assert len(provider.chat_calls) == 5  # 4 capped steps + synthesis
    assert run.steps[-2].message.decision.reason == "step cap reached"
    assert len(run.steps) == 6


# ================================================================= degenerate paths

async def test_zero_claims_skips_critique_and_finishes():
    empty_research = _claims_payload("Research prose.", [])
    empty_ideation = _claims_payload("Ideation prose.", [])
    provider, store, run = await _run_pipeline(
        [PLAN_JSON, empty_research, empty_ideation, SYNTHESIS_JSON]
    )
    assert len(provider.chat_calls) == 4
    assert run.status is RunStatus.COMPLETED
    assert [s.status for s in run.steps] == [
        StepStatus.COMPLETED, StepStatus.COMPLETED, StepStatus.COMPLETED,
        StepStatus.SKIPPED, StepStatus.COMPLETED,
    ]
    assert [s.round for s in run.steps] == [None, 1, 1, 1, None]
    assert len(run.rounds) == 1 and run.rounds[0].unresolved_count == 0

    synthesis_user = provider.chat_calls[3]["messages"][1].content
    assert "(skipped: no claims were produced to evaluate)" in synthesis_user
    assert "EVALUATED CLAIMS:\n(none)" in synthesis_user
    assert synthesis_user.endswith("REMAINING UNRESOLVED:\n(none)")


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
