/** Pure display helpers (Phase 8 PRD §6.6/§6.7). No I/O. */

import type { AgentRole } from "./types";

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
