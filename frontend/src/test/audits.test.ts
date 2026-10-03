/**
 * Audit derivation (Phase 11 PRD §9.2): location, counts, severity grouping.
 * Pure — no DOM, no I/O (architecture.test.ts enforces the no-I/O rule).
 */

import { describe, expect, it } from "vitest";

import {
  accountabilityFor,
  accountabilityLabel,
  claimsAtStatus,
  flagsBySeverity,
  verificationCounts,
  verificationFor,
} from "../audits";
import type {
  AccountabilityReport,
  AgentMessage,
  VerificationReport,
} from "../types";

function message(overrides: Partial<AgentMessage>): AgentMessage {
  return {
    id: "m1",
    from_agent: "manager",
    to_agent: null,
    type: "synthesis",
    content: "",
    claims: null,
    verdicts: null,
    decision: null,
    verification: null,
    accountability: null,
    confidence: null,
    tool_calls: null,
    tool_results: null,
    retries: 0,
    round: null,
    created_at: "2026-10-03T12:00:00Z",
    ...overrides,
  };
}

const VERIFICATION: VerificationReport = {
  claims: [
    {
      claim_id: "c1",
      verification_status: "verified",
      evidence_checked: ["doc"],
      supporting_evidence: [{ source: "doc", quote: "23%" }],
      contradicting_evidence: [],
      source_references: ["doc"],
      explanation: "checked",
      confidence: 0.9,
    },
    {
      claim_id: "c2",
      verification_status: "partially_verified",
      evidence_checked: ["doc"],
      supporting_evidence: [{ source: "doc", quote: null }],
      contradicting_evidence: [],
      source_references: [],
      explanation: "thin",
      confidence: 0.5,
    },
    {
      claim_id: "c3",
      verification_status: "contradicted",
      evidence_checked: ["doc"],
      supporting_evidence: [],
      contradicting_evidence: [{ source: "other", quote: null }],
      source_references: ["other"],
      explanation: "against",
      confidence: 0.8,
    },
    {
      claim_id: "c4",
      verification_status: "unverifiable",
      evidence_checked: [],
      supporting_evidence: [],
      contradicting_evidence: [],
      source_references: [],
      explanation: "nothing to check",
      confidence: 0.2,
    },
  ],
};

const ACCOUNTABILITY: AccountabilityReport = {
  trace_completeness: true,
  final_claim_provenance: [
    { claim_id: "c1", origin: "researcher", verdict: "supported", evidence_count: 1 },
    { claim_id: "c3", origin: "ideator", verdict: "refuted", evidence_count: 0 },
  ],
  flags: [
    { kind: "retry_activity", severity: "info", refs: ["4"], explanation: "one retry" },
    { kind: "premature_stop", severity: "warning", refs: ["2"], explanation: "guard fired" },
    {
      kind: "tool_use_inconsistency",
      severity: "violation",
      refs: ["7"],
      explanation: "calls without results",
    },
    { kind: "decision_inconsistency", severity: "warning", refs: ["5"], explanation: "broken chain" },
  ],
  overall_status: "violations",
  summary: "one tool inconsistency",
};

describe("verificationFor / accountabilityFor", () => {
  it("locates the reports in the step trace", () => {
    const steps = [
      { kind: "plan", message: null },
      { kind: "synthesize", message: message({ content: "final" }) },
      { kind: "verify", message: message({ verification: VERIFICATION }) },
      { kind: "audit", message: message({ accountability: ACCOUNTABILITY }) },
    ];
    expect(verificationFor(steps)).toBe(VERIFICATION);
    expect(accountabilityFor(steps)).toBe(ACCOUNTABILITY);
  });

  it("returns null when the step or message is missing", () => {
    expect(verificationFor([])).toBeNull();
    expect(accountabilityFor([])).toBeNull();
    expect(
      verificationFor([{ kind: "verify", message: null }]),
    ).toBeNull();
    // message present but no report (legacy runs)
    expect(
      verificationFor([{ kind: "verify", message: message({}) }]),
    ).toBeNull();
    expect(
      accountabilityFor([{ kind: "audit", message: message({}) }]),
    ).toBeNull();
  });

  it("prefers the latest audit step", () => {
    const newer: VerificationReport = { claims: [] };
    const steps = [
      { kind: "verify", message: message({ verification: VERIFICATION }) },
      { kind: "plan", message: null },
      { kind: "verify", message: message({ verification: newer }) },
    ];
    expect(verificationFor(steps)).toBe(newer);
  });

  it("works on StepEntry-shaped objects from the reducer", () => {
    const entryLike = {
      kind: "audit" as const,
      message: message({ accountability: ACCOUNTABILITY }),
      round: null,
      status: "completed" as const,
      index: 7,
      durationMs: 4,
      skipped: false,
      error: null,
    };
    expect(accountabilityFor([entryLike])).toBe(ACCOUNTABILITY);
  });
});

describe("verificationCounts", () => {
  it("tallies the four statuses", () => {
    expect(verificationCounts(VERIFICATION)).toEqual({
      verified: 1,
      partially: 1,
      contradicted: 1,
      unverifiable: 1,
      total: 4,
    });
    expect(verificationCounts({ claims: [] })).toEqual({
      verified: 0,
      partially: 0,
      contradicted: 0,
      unverifiable: 0,
      total: 0,
    });
  });
});

describe("flagsBySeverity", () => {
  it("groups and preserves order within each severity", () => {
    const grouped = flagsBySeverity(ACCOUNTABILITY);
    expect(grouped.violations.map((f) => f.kind)).toEqual([
      "tool_use_inconsistency",
    ]);
    expect(grouped.warnings.map((f) => f.kind)).toEqual([
      "premature_stop",
      "decision_inconsistency",
    ]);
    expect(grouped.info.map((f) => f.kind)).toEqual(["retry_activity"]);
  });

  it("handles empty flag lists", () => {
    const grouped = flagsBySeverity({
      ...ACCOUNTABILITY,
      flags: [],
    });
    expect(grouped).toEqual({ violations: [], warnings: [], info: [] });
  });
});

describe("labels", () => {
  it("maps overall status to readable labels", () => {
    expect(accountabilityLabel("clean")).toBe("Clean");
    expect(accountabilityLabel("warnings")).toBe("Warnings");
    expect(accountabilityLabel("violations")).toBe("Violations");
  });

  it("collects claim ids at a verification status", () => {
    expect(claimsAtStatus(VERIFICATION, "verified")).toEqual(["c1"]);
    expect(claimsAtStatus(VERIFICATION, "unverifiable")).toEqual(["c4"]);
    expect(claimsAtStatus({ claims: [] }, "verified")).toEqual([]);
  });
});
