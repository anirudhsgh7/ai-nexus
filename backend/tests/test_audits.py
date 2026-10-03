"""Mechanical audit layer (Phase 11 PRD §6.5): facts, validation, enforcement.

Everything here is deterministic — a green live eval rests on these checks
never needing a model.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.audits import (
    HONESTY_CUES,
    AuditFacts,
    FactFlag,
    _select_snapshot,
    compute_audit_facts,
    enforce_accountability,
    validate_verification,
)
from app.runs import RoundSnapshot, StepKind, StepRecord, StepStatus
from app.schemas import (
    AccountabilityFlag,
    AccountabilityFlagKind,
    AccountabilityReport,
    AccountabilityStatus,
    AgentMessage,
    AgentRole,
    Claim,
    ClaimProvenance,
    ClaimStatus,
    ClaimVerdict,
    DecisionAction,
    Evidence,
    FlagSeverity,
    ManagerDecision,
    MessageType,
    VerificationReport,
    VerificationStatus,
    Verdict,
)

NOW = datetime.now(UTC)


# ------------------------------------------------------------------ helpers


def _claim(
    cid: str,
    *,
    status: ClaimStatus = ClaimStatus.UNVERIFIED,
    confidence: float | None = None,
    evidence: list[Evidence] | None = None,
) -> Claim:
    return Claim(
        id=cid, statement=f"statement {cid}", status=status,
        confidence=confidence, evidence=evidence or [],
    )


def _verdict(
    cid: str, verdict: ClaimVerdict, objection: str = "objection",
) -> Verdict:
    return Verdict(claim_id=cid, verdict=verdict, objection=objection)


def _snapshot(
    number: int,
    claims: list[Claim],
    verdicts: list[Verdict],
    origins: dict[str, AgentRole] | None = None,
    *,
    supported: int = 0,
    unresolved: int = 0,
) -> RoundSnapshot:
    return RoundSnapshot(
        round_number=number,
        claims=claims,
        origins=origins or {c.id: AgentRole.RESEARCHER for c in claims},
        verdicts=verdicts,
        worker_content={},
        skeptic_content="",
        supported_count=supported,
        unresolved_count=unresolved,
    )


def _step(
    index: int,
    kind: StepKind,
    agent: AgentRole = AgentRole.MANAGER,
    *,
    status: StepStatus = StepStatus.COMPLETED,
    skipped: bool = False,
    message: AgentMessage | None = None,
    round: int | None = None,
) -> StepRecord:
    return StepRecord(
        index=index, kind=kind, agent=agent, status=status,
        started_at=NOW, message=message, skipped=skipped, round=round,
    )


def _decide_message(action: DecisionAction, *, target: AgentRole | None = None,
                    instruction: str = "", reason: str = "r") -> AgentMessage:
    return AgentMessage(
        from_agent=AgentRole.MANAGER,
        type=MessageType.DECISION,
        content="",
        decision=ManagerDecision(
            action=action, target=target, instruction=instruction,
            reason=reason, confidence=0.7,
        ),
    )


def _healthy_trace() -> list[StepRecord]:
    """plan research ideate critique synthesize verify (+ optional extras)."""
    return [
        _step(1, StepKind.PLAN),
        _step(2, StepKind.RESEARCH, AgentRole.RESEARCHER, round=1),
        _step(3, StepKind.IDEATE, AgentRole.IDEATOR, round=1),
        _step(4, StepKind.CRITIQUE, AgentRole.SKEPTIC, round=1),
        _step(5, StepKind.SYNTHESIZE),
        _step(6, StepKind.VERIFY, AgentRole.VERIFIER),
    ]


def _final(c1_supported: bool = True) -> str:
    return (
        "Growth was 23%; the 40% claim is incorrect and no data remains "
        "unresolved beyond labeled assumptions."
    )


# --------------------------------------------------------- compute: trace


def test_healthy_trace_is_complete():
    snapshot = _snapshot(
        1, [_claim("c1", status=ClaimStatus.FACT,
                   evidence=[Evidence(source="doc")])],
        [_verdict("c1", ClaimVerdict.SUPPORTED)],
        supported=1,
    )
    facts = compute_audit_facts([snapshot], _healthy_trace(), _final())
    assert facts.trace_complete is True
    assert facts.trace_missing == ()
    # fully supported + honest final answer -> no required flags at all
    assert facts.required_flags == ()


def test_missing_kind_and_failed_step_flagged():
    steps = _healthy_trace()
    steps = [s for s in steps if s.kind is not StepKind.SYNTHESIZE]
    facts = compute_audit_facts([], steps, "answer")
    assert facts.trace_complete is False
    assert facts.trace_missing == ("synthesize",)
    trace_flags = [
        f for f in facts.required_flags
        if f.kind is AccountabilityFlagKind.TRACE_INCOMPLETENESS
    ]
    assert len(trace_flags) == 1
    assert trace_flags[0].severity is FlagSeverity.WARNING
    assert trace_flags[0].refs == ("synthesize",)

    failed = _healthy_trace()
    failed[3] = _step(4, StepKind.CRITIQUE, AgentRole.SKEPTIC,
                      status=StepStatus.FAILED, round=1)
    facts = compute_audit_facts([], failed, "answer")
    trace_flag = next(
        f for f in facts.required_flags
        if f.kind is AccountabilityFlagKind.TRACE_INCOMPLETENESS
    )
    assert trace_flag.severity is FlagSeverity.VIOLATION
    assert trace_flag.refs == ("4",)


def test_no_final_message_is_incomplete():
    facts = compute_audit_facts([], _healthy_trace(), None)
    assert facts.trace_complete is False


# ------------------------------------------------- compute: claim facts


def test_unsupported_claim_and_provenance():
    claims = [
        _claim("c1", status=ClaimStatus.FACT, evidence=[Evidence(source="doc")]),
        _claim("c2"),
    ]
    verdicts = [
        _verdict("c1", ClaimVerdict.SUPPORTED),
        _verdict("c2", ClaimVerdict.UNVERIFIABLE),
    ]
    snapshot = _snapshot(1, claims, verdicts,
                         origins={"c1": AgentRole.RESEARCHER,
                                  "c2": AgentRole.IDEATOR},
                         supported=1, unresolved=1)
    facts = compute_audit_facts(
        [snapshot], _healthy_trace(),
        "Growth was 23%; this remains speculative beyond the labeled assumption.",
    )
    assert [p.claim_id for p in facts.provenance] == ["c1", "c2"]
    assert facts.provenance[0].origin is AgentRole.RESEARCHER
    assert facts.provenance[0].verdict is ClaimVerdict.SUPPORTED
    assert facts.provenance[1].origin is AgentRole.IDEATOR
    assert facts.provenance[1].verdict is ClaimVerdict.UNVERIFIABLE
    unsupported = next(
        f for f in facts.required_flags
        if f.kind is AccountabilityFlagKind.UNSUPPORTED_FINAL_CLAIM
    )
    assert unsupported.severity is FlagSeverity.WARNING
    assert unsupported.refs == ("c2",)
    # answer admits uncertainty -> no suppression flag
    assert not any(
        f.kind is AccountabilityFlagKind.UNRESOLVED_CLAIM_SUPPRESSED
        for f in facts.required_flags
    )


def test_suppression_flag_fires_only_when_answer_hides():
    claims = [_claim("c4")]
    verdicts = [_verdict("c4", ClaimVerdict.REFUTED)]
    snapshot = _snapshot(1, claims, verdicts, unresolved=1)
    steps = _healthy_trace()

    # hidden: no cue, no claim id
    facts = compute_audit_facts([snapshot], steps, "Everything is settled here.")
    assert any(
        f.kind is AccountabilityFlagKind.UNRESOLVED_CLAIM_SUPPRESSED
        and f.refs == ("c4",)
        for f in facts.required_flags
    )
    # honesty cue present
    facts = compute_audit_facts([snapshot], steps, "This remains speculative.")
    assert not any(
        f.kind is AccountabilityFlagKind.UNRESOLVED_CLAIM_SUPPRESSED
        for f in facts.required_flags
    )
    # claim id named
    facts = compute_audit_facts([snapshot], steps, "Claim c4 could not be resolved.")
    assert not any(
        f.kind is AccountabilityFlagKind.UNRESOLVED_CLAIM_SUPPRESSED
        for f in facts.required_flags
    )


def test_evidence_provenance_gap_rules():
    claims = [
        _claim("c1", status=ClaimStatus.FACT),            # fact, no evidence
        _claim("c2"),                                     # supported w/o evidence
    ]
    verdicts = [_verdict("c2", ClaimVerdict.SUPPORTED)]
    snapshot = _snapshot(1, claims, verdicts, supported=1)
    facts = compute_audit_facts([snapshot], _healthy_trace(), "ok unverifiable")
    gap = next(
        f for f in facts.required_flags
        if f.kind is AccountabilityFlagKind.EVIDENCE_PROVENANCE_GAP
    )
    assert gap.severity is FlagSeverity.WARNING
    assert set(gap.refs) == {"c1", "c2"}


def test_confidence_evidence_mismatch_rules():
    claims = [
        _claim("c1", confidence=0.9),                       # high, unsupported
        _claim("c2", confidence=0.9),                       # high, no evidence
        _claim("c3", confidence=0.9, status=ClaimStatus.FACT,
               evidence=[Evidence(source="doc")]),          # high but healthy
    ]
    verdicts = [_verdict("c3", ClaimVerdict.SUPPORTED)]
    snapshot = _snapshot(1, claims, verdicts, supported=1, unresolved=2)
    facts = compute_audit_facts([snapshot], _healthy_trace(), "speculative only")
    mismatch = next(
        f for f in facts.required_flags
        if f.kind is AccountabilityFlagKind.CONFIDENCE_EVIDENCE_MISMATCH
    )
    assert set(mismatch.refs) == {"c1", "c2"}
    assert "c3" not in mismatch.refs


def test_honesty_cues_locked():
    assert HONESTY_CUES == {
        "uncertain", "unresolved", "unverifiable", "assumption", "estimate",
        "no data", "cannot", "speculative", "lack specific data",
    }


# ------------------------------------------------- compute: step facts


def test_tool_use_inconsistency():
    from app.llm.base import ToolCall

    ok_message = AgentMessage(
        from_agent=AgentRole.RESEARCHER, type=MessageType.FINDING,
        tool_calls=[ToolCall(name="web_search", arguments={"query": "q"})],
        tool_results=[{"name": "web_search", "content": "{}"}],  # type: ignore[list-item]
    )
    steps = _healthy_trace()
    steps.insert(2, _step(7, StepKind.RESEARCH, AgentRole.RESEARCHER,
                          message=ok_message, round=1))
    facts = compute_audit_facts([], steps, "ok")
    assert not any(
        f.kind is AccountabilityFlagKind.TOOL_USE_INCONSISTENCY
        for f in facts.required_flags
    )

    bad_message = AgentMessage(
        from_agent=AgentRole.RESEARCHER, type=MessageType.FINDING,
        tool_calls=[ToolCall(name="web_search", arguments={"query": "q"})],
        tool_results=None,
    )
    steps = _healthy_trace()
    steps.insert(2, _step(7, StepKind.RESEARCH, AgentRole.RESEARCHER,
                          message=bad_message, round=1))
    facts = compute_audit_facts([], steps, "ok")
    flag = next(
        f for f in facts.required_flags
        if f.kind is AccountabilityFlagKind.TOOL_USE_INCONSISTENCY
    )
    assert flag.severity is FlagSeverity.VIOLATION
    assert flag.refs == ("7",)


def test_retry_activity_flag():
    retried = AgentMessage(
        from_agent=AgentRole.SKEPTIC, type=MessageType.CRITIQUE, retries=1,
    )
    steps = _healthy_trace()
    steps[3] = _step(4, StepKind.CRITIQUE, AgentRole.SKEPTIC,
                     message=retried, round=1)
    facts = compute_audit_facts([], steps, "ok")
    flag = next(
        f for f in facts.required_flags
        if f.kind is AccountabilityFlagKind.RETRY_ACTIVITY
    )
    assert flag.severity is FlagSeverity.INFO
    assert flag.refs == ("4",)


def test_premature_stop_requires_unresolved_claims():
    guard_step = _step(7, StepKind.DECIDE, skipped=True, round=2,
                       message=_decide_message(DecisionAction.FINISH,
                                               reason="no progress"))
    supported_snapshot = _snapshot(
        1, [_claim("c1", status=ClaimStatus.FACT,
                   evidence=[Evidence(source="doc")])],
        [_verdict("c1", ClaimVerdict.SUPPORTED)], supported=1,
    )
    facts = compute_audit_facts(
        [supported_snapshot], [*_healthy_trace(), guard_step], "done unverifiable"
    )
    assert not any(
        f.kind is AccountabilityFlagKind.PREMATURE_STOP
        for f in facts.required_flags
    )

    unresolved_snapshot = _snapshot(
        1, [_claim("c1")], [_verdict("c1", ClaimVerdict.UNVERIFIABLE)],
        unresolved=1,
    )
    facts = compute_audit_facts(
        [unresolved_snapshot], [*_healthy_trace(), guard_step], "done speculative"
    )
    flag = next(
        f for f in facts.required_flags
        if f.kind is AccountabilityFlagKind.PREMATURE_STOP
    )
    assert flag.severity is FlagSeverity.WARNING
    assert flag.refs == ("2",)


# ------------------------------------------------- decision consistency


def _decision_flag(facts: AuditFacts) -> FactFlag | None:
    return next(
        (f for f in facts.required_flags
         if f.kind is AccountabilityFlagKind.DECISION_INCONSISTENCY),
        None,
    )


def test_call_agent_honored_by_revision_is_clean():
    steps = [
        *_healthy_trace()[:4],
        _step(5, StepKind.DECIDE, round=1,
              message=_decide_message(DecisionAction.CALL_AGENT,
                                      target=AgentRole.RESEARCHER,
                                      instruction="find a source")),
        _step(6, StepKind.REVISE, AgentRole.RESEARCHER, round=2),
        _step(7, StepKind.SYNTHESIZE),
        _step(8, StepKind.VERIFY, AgentRole.VERIFIER),
    ]
    facts = compute_audit_facts([], steps, "ok unverifiable")
    assert _decision_flag(facts) is None


def test_call_agent_stopped_by_guard_is_not_inconsistent():
    steps = [
        *_healthy_trace()[:4],
        _step(5, StepKind.DECIDE, round=1,
              message=_decide_message(DecisionAction.CALL_AGENT,
                                      target=AgentRole.RESEARCHER,
                                      instruction="find a source")),
        _step(6, StepKind.DECIDE, skipped=True, round=1,
              message=_decide_message(DecisionAction.FINISH, reason="repeated decision")),
        _step(7, StepKind.SYNTHESIZE),
        _step(8, StepKind.VERIFY, AgentRole.VERIFIER),
    ]
    facts = compute_audit_facts([], steps, "ok speculative")
    assert _decision_flag(facts) is None


def test_unhonored_call_agent_is_violation():
    steps = [
        *_healthy_trace()[:4],
        _step(5, StepKind.DECIDE, round=1,
              message=_decide_message(DecisionAction.CALL_AGENT,
                                      target=AgentRole.RESEARCHER,
                                      instruction="find a source")),
        _step(6, StepKind.SYNTHESIZE),
        _step(7, StepKind.VERIFY, AgentRole.VERIFIER),
    ]
    facts = compute_audit_facts([], steps, "ok speculative")
    flag = _decision_flag(facts)
    assert flag is not None
    assert flag.severity is FlagSeverity.VIOLATION
    assert flag.refs == ("5",)


def test_decide_after_finish_is_violation():
    steps = [
        *_healthy_trace()[:4],
        _step(5, StepKind.DECIDE, round=1,
              message=_decide_message(DecisionAction.FINISH)),
        _step(6, StepKind.DECIDE, round=1,
              message=_decide_message(DecisionAction.CALL_AGENT,
                                      target=AgentRole.IDEATOR,
                                      instruction="more")),
        _step(7, StepKind.SYNTHESIZE),
        _step(8, StepKind.VERIFY, AgentRole.VERIFIER),
    ]
    facts = compute_audit_facts([], steps, "ok speculative")
    flag = _decision_flag(facts)
    assert flag is not None
    assert "5" in flag.refs


def test_unparseable_and_malformed_guard_decisions_flagged():
    # non-skipped decide whose decision never parsed
    bare = AgentMessage(from_agent=AgentRole.MANAGER, type=MessageType.DECISION)
    steps = [
        _step(1, StepKind.DECIDE, round=1, message=bare),
        _step(2, StepKind.SYNTHESIZE),
        _step(3, StepKind.VERIFY, AgentRole.VERIFIER),
    ]
    facts = compute_audit_facts([], steps, "ok")
    assert _decision_flag(facts) is not None

    # guard-skipped decide that is not a synthetic finish
    bad_guard = AgentMessage(
        from_agent=AgentRole.MANAGER, type=MessageType.DECISION,
        decision=ManagerDecision(
            action=DecisionAction.CALL_AGENT, target=AgentRole.IDEATOR,
            instruction="x", reason="r", confidence=0.5,
        ),
    )
    steps = [
        _step(1, StepKind.DECIDE, skipped=True, round=1, message=bad_guard),
        _step(2, StepKind.SYNTHESIZE),
        _step(3, StepKind.VERIFY, AgentRole.VERIFIER),
    ]
    facts = compute_audit_facts([], steps, "ok")
    assert _decision_flag(facts) is not None


def test_skipped_guard_without_message_flags():
    steps = [
        _step(1, StepKind.DECIDE, skipped=True, round=1, message=None),
        _step(2, StepKind.SYNTHESIZE),
        _step(3, StepKind.VERIFY, AgentRole.VERIFIER),
    ]
    facts = compute_audit_facts([], steps, "ok")
    assert _decision_flag(facts) is not None


# --------------------------------------------------------- selection drift


def test_selected_snapshot_matches_orchestrator_rule():
    from app.orchestrator import select_best_round

    matrices = [
        [
            _snapshot(1, [], [], supported=3, unresolved=2),
            _snapshot(2, [], [], supported=6, unresolved=2),
        ],
        [
            _snapshot(1, [], [], supported=3, unresolved=2),
            _snapshot(2, [], [], supported=3, unresolved=2),
        ],
        [_snapshot(1, [], [], supported=5, unresolved=0)],
        [
            _snapshot(1, [], [], supported=9, unresolved=1),
            _snapshot(2, [], [], supported=2, unresolved=0),
        ],
    ]
    for snapshots in matrices:
        assert _select_snapshot(snapshots) is select_best_round(snapshots)
    assert _select_snapshot([]) is None


# ------------------------------------------------- verify validation


def _entry(
    cid: str,
    status: VerificationStatus,
    *,
    supporting: list[Evidence] | None = None,
    contradicting: list[Evidence] | None = None,
    sources: list[str] | None = None,
    checked: list[str] | None = None,
) -> "object":
    from app.schemas import ClaimVerification
    return ClaimVerification(
        claim_id=cid,
        verification_status=status,
        evidence_checked=["doc"] if checked is None else checked,
        supporting_evidence=supporting or [],
        contradicting_evidence=contradicting or [],
        source_references=sources or [],
        explanation="checked the record",
        confidence=0.8,
    )


def _verified_entry(cid: str) -> "object":
    return _entry(
        cid, VerificationStatus.VERIFIED,
        supporting=[Evidence(source="doc", quote="q")],
        sources=["doc"],
    )


def test_verification_clean_report_passes():
    claims = [_claim("c1"), _claim("c2")]
    report = VerificationReport(claims=[  # type: ignore[arg-type]
        _verified_entry("c1"),
        _entry("c2", VerificationStatus.UNVERIFIABLE, checked=[],
               sources=[]),
    ])
    assert validate_verification(report, claims, 1, True) == []


def test_verification_coverage_rules():
    claims = [_claim("c1"), _claim("c2")]
    report = VerificationReport(claims=[_verified_entry("c1")])  # type: ignore[arg-type]
    problems = validate_verification(report, claims, 1, True)
    assert problems == ["claim c2 was not verified"]

    dup = VerificationReport(claims=[  # type: ignore[arg-type]
        _verified_entry("c1"), _verified_entry("c1"),
    ])
    problems = validate_verification(dup, [_claim("c1")], 1, True)
    assert problems == ["claim c1 verified more than once"]

    unknown = VerificationReport(claims=[_verified_entry("c9")])  # type: ignore[arg-type]
    problems = validate_verification(unknown, [_claim("c1")], 1, True)
    assert "claim c1 was not verified" in problems
    assert "verification for unknown claim c9" in problems


def test_verification_status_evidence_rules():
    claims = [_claim("c1")]

    no_support = VerificationReport(claims=[_entry("c1", VerificationStatus.VERIFIED)])  # type: ignore[arg-type]
    assert validate_verification(no_support, claims, 1, True) == [
        "claim c1 verified without supporting evidence",
        "claim c1 verified without source references",
    ]

    with_contradiction = VerificationReport(claims=[  # type: ignore[arg-type]
        _entry("c1", VerificationStatus.VERIFIED,
               supporting=[Evidence(source="a")],
               contradicting=[Evidence(source="b")]),
    ])
    assert "claim c1 verified despite contradicting evidence" in validate_verification(
        with_contradiction, claims, 1, True,
    )

    no_checked = VerificationReport(claims=[  # type: ignore[arg-type]
        _entry("c1", VerificationStatus.VERIFIED,
               supporting=[Evidence(source="a")],
               checked=[], sources=["a"]),
    ])
    assert validate_verification(no_checked, claims, 1, True) == [
        "claim c1 verified without recording checked evidence",
    ]

    contradicted_no_evidence = VerificationReport(claims=[  # type: ignore[arg-type]
        _entry("c1", VerificationStatus.CONTRADICTED, contradicting=[]),
    ])
    assert validate_verification(contradicted_no_evidence, claims, 1, True) == [
        "claim c1 contradicted without contradicting evidence",
    ]

    partial_nothing = VerificationReport(claims=[  # type: ignore[arg-type]
        _entry("c1", VerificationStatus.PARTIALLY_VERIFIED,
               supporting=[], contradicting=[]),
    ])
    assert validate_verification(partial_nothing, claims, 1, True) == [
        "claim c1 partially verified without evidence",
    ]


def test_verification_independent_tool_check_rule():
    claims = [_claim("c1")]
    report = VerificationReport(claims=[_verified_entry("c1")])  # type: ignore[arg-type]
    # tools available but none executed -> rejected
    assert validate_verification(report, claims, 0, True) == [
        "verified claims require at least one independent tool check",
    ]
    # executed -> clean
    assert validate_verification(report, claims, 2, True) == []
    # no tools configured -> record-level verification allowed
    assert validate_verification(report, claims, 0, False) == []
    # unverifiable never needs a tool check
    unverifiable = VerificationReport(claims=[  # type: ignore[arg-type]
        _entry("c1", VerificationStatus.UNVERIFIABLE, checked=[], sources=[]),
    ])
    assert validate_verification(unverifiable, claims, 0, True) == []


def test_verification_missing_report():
    assert validate_verification(None, [_claim("c1")], 0, False) == [
        "verification report missing",
    ]


# -------------------------------------------------- accountability enforcement


def _report(
    *,
    trace: bool = True,
    provenance: list | None = None,
    flags: list[AccountabilityFlag] | None = None,
    status: AccountabilityStatus = AccountabilityStatus.CLEAN,
    summary: str = "model summary",
) -> AccountabilityReport:
    return AccountabilityReport(
        trace_completeness=trace,
        final_claim_provenance=provenance or [],
        flags=flags or [],
        overall_status=status,
        summary=summary,
    )


def _facts(
    *,
    trace: bool = True,
    provenance: tuple = (),
    flags: tuple[FactFlag, ...] = (),
) -> AuditFacts:
    return AuditFacts(
        trace_complete=trace, trace_missing=(),
        provenance=provenance, required_flags=flags,
    )


def test_enforce_overwrites_code_canonical_fields():
    model_claim = ClaimProvenance(
        claim_id="c9", origin=AgentRole.SKEPTIC, verdict=None,
    )
    fact_claim = ClaimProvenance(
        claim_id="c1", origin=AgentRole.RESEARCHER,
        verdict=ClaimVerdict.SUPPORTED, evidence_count=2,
    )
    enforced = enforce_accountability(
        _report(trace=False, provenance=[model_claim]),
        _facts(trace=True, provenance=(fact_claim,)),
    )
    assert enforced.trace_completeness is True
    assert [p.claim_id for p in enforced.final_claim_provenance] == ["c1"]
    assert enforced.overall_status is AccountabilityStatus.CLEAN
    assert enforced.summary == "model summary"  # content preserved


def test_enforce_appends_missing_fact_flags_and_derives_status():
    fact = FactFlag(
        kind=AccountabilityFlagKind.UNSUPPORTED_FINAL_CLAIM,
        severity=FlagSeverity.WARNING,
        refs=("c4",),
        detail="final claim has no supported verdict",
    )
    enforced = enforce_accountability(
        _report(flags=[], status=AccountabilityStatus.CLEAN),
        _flags_facts(fact),
    )
    assert [f.kind for f in enforced.flags] == [
        AccountabilityFlagKind.UNSUPPORTED_FINAL_CLAIM
    ]
    assert enforced.flags[0].severity is FlagSeverity.WARNING
    assert enforced.flags[0].refs == ["c4"]
    assert enforced.flags[0].explanation == "final claim has no supported verdict"
    assert enforced.overall_status is AccountabilityStatus.WARNINGS


def _flags_facts(*facts: FactFlag) -> AuditFacts:
    return _facts(flags=facts)


def test_enforce_raises_model_understatement():
    fact = FactFlag(
        kind=AccountabilityFlagKind.PREMATURE_STOP,
        severity=FlagSeverity.WARNING,
        refs=("2",),
        detail="guard stopped the run",
    )
    model_flag = AccountabilityFlag(
        kind=AccountabilityFlagKind.PREMATURE_STOP,
        severity=FlagSeverity.INFO,   # understated on purpose
        refs=["2"],
        explanation="the model's own wording",
    )
    enforced = enforce_accountability(
        _report(flags=[model_flag], status=AccountabilityStatus.CLEAN),
        _facts(flags=(fact,)),
    )
    assert len(enforced.flags) == 1
    assert enforced.flags[0].severity is FlagSeverity.WARNING  # raised to fact
    assert enforced.flags[0].explanation == "the model's own wording"
    assert enforced.overall_status is AccountabilityStatus.WARNINGS


def test_enforce_violation_fact_forces_violations_status():
    fact = FactFlag(
        kind=AccountabilityFlagKind.DECISION_INCONSISTENCY,
        severity=FlagSeverity.VIOLATION,
        refs=("5",),
        detail="decision chain broken",
    )
    enforced = enforce_accountability(
        _report(status=AccountabilityStatus.CLEAN),
        _facts(flags=(fact,)),
    )
    assert enforced.overall_status is AccountabilityStatus.VIOLATIONS


def test_enforce_is_identity_when_report_already_consistent():
    fact = FactFlag(
        kind=AccountabilityFlagKind.RETRY_ACTIVITY,
        severity=FlagSeverity.INFO,
        refs=("4",),
        detail="retry occurred",
    )
    model_flag = AccountabilityFlag(
        kind=AccountabilityFlagKind.RETRY_ACTIVITY,
        severity=FlagSeverity.INFO,
        refs=["4"],
        explanation="one correction retry",
    )
    report = _report(flags=[model_flag], status=AccountabilityStatus.CLEAN)
    enforced = enforce_accountability(report, _facts(flags=(fact,)))
    assert enforced == report.model_copy(
        update={"overall_status": AccountabilityStatus.CLEAN}
    )
    assert enforced.flags[0].explanation == "one correction retry"


def test_enforce_keeps_model_semantic_flags():
    model_flag = AccountabilityFlag(
        kind=AccountabilityFlagKind.DECISION_INCONSISTENCY,
        severity=FlagSeverity.WARNING,
        refs=["5"],
        explanation="instruction repeated a previous one",
    )
    enforced = enforce_accountability(
        _report(flags=[model_flag]), _facts(flags=()),
    )
    assert enforced.flags == [model_flag]
    assert enforced.overall_status is AccountabilityStatus.WARNINGS
