/**
 * Pure audit derivation (Phase 11 PRD §6.11).
 *
 * Locates and summarizes the two audit reports carried by the step trace.
 * No I/O, no React, no clock — enforced by `architecture.test.ts`.
 * The UI folds events into `RunView`; these helpers only read the result.
 */

import type {
  AccountabilityFlag,
  AccountabilityReport,
  AccountabilityStatus,
  AgentMessage,
  ClaimVerification,
  VerificationReport,
  VerificationStatus,
} from "./types";

/** Structural minimum satisfied by both `StepEntry` and `StepPayload`. */
export type StepLike = {
  kind: string;
  message: AgentMessage | null;
};

/** The Verifier's report — the latest verify step wins (steps are ordered). */
export function verificationFor(
  steps: readonly StepLike[],
): VerificationReport | null {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const step = steps[index];
    if (step !== undefined && step.kind === "verify" && step.message !== null) {
      return step.message.verification;
    }
  }
  return null;
}

/** The Accountability agent's report — latest audit step wins. */
export function accountabilityFor(
  steps: readonly StepLike[],
): AccountabilityReport | null {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const step = steps[index];
    if (step !== undefined && step.kind === "audit" && step.message !== null) {
      return step.message.accountability;
    }
  }
  return null;
}

export interface VerificationCounts {
  verified: number;
  partially: number;
  contradicted: number;
  unverifiable: number;
  total: number;
}

export function verificationCounts(report: VerificationReport): VerificationCounts {
  const counts: VerificationCounts = {
    verified: 0,
    partially: 0,
    contradicted: 0,
    unverifiable: 0,
    total: report.claims.length,
  };
  for (const entry of report.claims) {
    if (entry.verification_status === "verified") counts.verified += 1;
    else if (entry.verification_status === "partially_verified") counts.partially += 1;
    else if (entry.verification_status === "contradicted") counts.contradicted += 1;
    else counts.unverifiable += 1;
  }
  return counts;
}

export interface FlagsBySeverity {
  violations: AccountabilityFlag[];
  warnings: AccountabilityFlag[];
  info: AccountabilityFlag[];
}

export function flagsBySeverity(report: AccountabilityReport): FlagsBySeverity {
  const grouped: FlagsBySeverity = { violations: [], warnings: [], info: [] };
  for (const flag of report.flags) {
    if (flag.severity === "violation") grouped.violations.push(flag);
    else if (flag.severity === "warning") grouped.warnings.push(flag);
    else grouped.info.push(flag);
  }
  return grouped;
}

/** Round-friendly status name for the overall accountability chip. */
export function accountabilityLabel(status: AccountabilityStatus): string {
  if (status === "clean") return "Clean";
  if (status === "warnings") return "Warnings";
  return "Violations";
}

/** Convenience: claim ids of entries at a given status (panel highlighting). */
export function claimsAtStatus(
  report: VerificationReport,
  status: VerificationStatus,
): string[] {
  return report.claims
    .filter((entry: ClaimVerification) => entry.verification_status === status)
    .map((entry) => entry.claim_id);
}
