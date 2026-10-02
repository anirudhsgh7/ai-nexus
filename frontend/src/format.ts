/** Pure display helpers (Phase 8 PRD §6.6/§6.7, Phase 9 PRD §6.8). No I/O. */

import type { AgentRole, Claim, Verdict } from "./types";

export const AGENT_LABELS: Record<AgentRole, string> = {
  manager: "Manager",
  researcher: "Researcher",
  ideator: "Ideator",
  skeptic: "Skeptic",
};

export function agentLabel(role: AgentRole): string {
  return AGENT_LABELS[role];
}

/** "0.1s" | "118.9s" | "1m 58.9s" | "—" for null. */
export function formatDuration(ms: number | null): string {
  if (ms === null || Number.isNaN(ms)) return "—";
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}m ${rest.toFixed(1)}s`;
}

/** Elapsed wall time between two ISO timestamps (null-safe). */
export function formatElapsed(
  startedAt: string | null,
  finishedAt: string | null,
  nowMs: number,
): string {
  if (startedAt === null) return "—";
  const start = Date.parse(startedAt);
  if (Number.isNaN(start)) return "—";
  const end = finishedAt !== null ? Date.parse(finishedAt) : nowMs;
  if (Number.isNaN(end)) return "—";
  return formatDuration(Math.max(0, end - start));
}

export function pluralize(count: number, singular: string): string {
  return `${count} ${count === 1 ? singular : `${singular}s`}`;
}

// ------------------------------------------------- Phase 9: evidence labels

export const CLAIM_STATUS_LABELS: Record<Claim["status"], string> = {
  fact: "Fact",
  assumption: "Assumption",
  hypothesis: "Hypothesis",
  opinion: "Opinion",
  inference: "Inference",
  unverified: "Unverified",
};

/** CSS class suffixes; always paired with the text label above (a11y). */
export const CLAIM_STATUS_TONES: Record<Claim["status"], string> = {
  fact: "claim-fact",
  assumption: "claim-assumption",
  hypothesis: "claim-hypothesis",
  opinion: "claim-opinion",
  inference: "claim-inference",
  unverified: "claim-unverified",
};

export const VERDICT_LABELS: Record<Verdict["verdict"], string> = {
  supported: "Supported",
  refuted: "Refuted",
  unverifiable: "Unverifiable",
};

export const VERDICT_TONES: Record<Verdict["verdict"], string> = {
  supported: "verdict-supported",
  refuted: "verdict-refuted",
  unverifiable: "verdict-unverifiable",
};

/** "90%" (nearest integer) | "—" for null/non-finite/out-of-range. */
export function formatConfidence(value: number | null): string {
  if (value === null || !Number.isFinite(value) || value < 0 || value > 1) {
    return "—";
  }
  return `${Math.round(value * 100)}%`;
}

/** Verdict badge text; a claim with no verdict is "Not evaluated". */
export function verdictLabel(verdict: Verdict["verdict"] | null): string {
  return verdict === null ? "Not evaluated" : VERDICT_LABELS[verdict];
}
