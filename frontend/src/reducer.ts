/**
 * The event fold (Phase 8 PRD §6.6) — the heart of the UI.
 *
 * The run view is a pure function of the backend's event log. The UI never
 * decides, sequences, or filters work: it folds `RunEvent`s and renders the
 * result. Every screen state is therefore unit-testable without a browser.
 */

import { agentLabel, formatDuration, pluralize } from "./format";
import type {
  AgentMessage,
  AgentRole,
  ConnectionState,
  ErrorInfo,
  RunEvent,
  RunStatus,
  StepKind,
} from "./types";

export type StepStatus = "running" | "completed" | "failed" | "skipped";

export interface StepEntry {
  index: number;
  kind: StepKind;
  agent: AgentRole;
  round: number | null;
  status: StepStatus;
  durationMs: number | null;
  message: AgentMessage | null;
  skipped: boolean;
  error: ErrorInfo | null;
}

export interface RunView {
  runId: string | null;
  task: string | null;
  status: RunStatus;
  steps: StepEntry[];
  finalMessage: AgentMessage | null;
  error: ErrorInfo | null;
  startedAt: string | null;
  finishedAt: string | null;
  lastSeq: number;
  connection: ConnectionState;
}

export const initialRunView: RunView = {
  runId: null,
  task: null,
  status: "running",
  steps: [],
  finalMessage: null,
  error: null,
  startedAt: null,
  finishedAt: null,
  lastSeq: 0,
  connection: "connecting",
};

export function createRunView(runId: string, task: string | null = null): RunView {
  return { ...initialRunView, runId, task };
}

export function setConnection(view: RunView, connection: ConnectionState): RunView {
  return { ...view, connection };
}

function upsertStep(steps: StepEntry[], entry: StepEntry): StepEntry[] {
  const position = steps.findIndex((step) => step.index === entry.index);
  if (position === -1) {
    return [...steps, entry].sort((a, b) => a.index - b.index);
  }
  const next = steps.slice();
  next[position] = entry;
  return next;
}

function markStepFailed(
  steps: StepEntry[],
  index: number,
  error: ErrorInfo | null,
): StepEntry[] {
  const position = steps.findIndex((step) => step.index === index);
  const existing = steps[position];
  if (existing === undefined) return steps;
  const next = steps.slice();
  next[position] = { ...existing, status: "failed", error };
  return next;
}

/** Pure fold: `view + event -> view`. Idempotent by `seq` watermark. */
export function foldEvent(view: RunView, event: RunEvent): RunView {
  if (event.seq <= view.lastSeq) return view;
  const next: RunView = { ...view, lastSeq: event.seq };

  switch (event.type) {
    case "run_started":
      return {
        ...next,
        task: event.task ?? view.task,
        status: "running",
        startedAt: event.ts,
      };
    case "step_started": {
      if (event.step === null) return next;
      const existing = view.steps.find((step) => step.index === event.step);
      return {
        ...next,
        steps: upsertStep(view.steps, {
          index: event.step,
          kind: event.kind ?? existing?.kind ?? "plan",
          agent: event.agent ?? existing?.agent ?? "manager",
          round: event.round ?? existing?.round ?? null,
          status: "running",
          durationMs: event.duration_ms,
          message: event.message,
          skipped: false,
          error: event.error,
        }),
      };
    }
    case "step_completed": {
      // Skipped/forced steps never emit `step_started` — create on complete.
      if (event.step === null) return next;
      const existing = view.steps.find((step) => step.index === event.step);
      const skipped = event.skipped === true;
      return {
        ...next,
        steps: upsertStep(view.steps, {
          index: event.step,
          kind: event.kind ?? existing?.kind ?? "plan",
          agent: event.agent ?? existing?.agent ?? "manager",
          round: event.round ?? existing?.round ?? null,
          status: skipped ? "skipped" : "completed",
          durationMs: event.duration_ms,
          message: event.message ?? existing?.message ?? null,
          skipped,
          error: event.error,
        }),
      };
    }
    case "run_completed":
      return {
        ...next,
        status: "completed",
        finalMessage: event.message,
        finishedAt: event.ts,
      };
    case "run_failed":
      return {
        ...next,
        status: "failed",
        error: event.error,
        finishedAt: event.ts,
        steps:
          event.step === null
            ? view.steps
            : markStepFailed(view.steps, event.step, event.error),
      };
  }
}

/** Last agent that is currently working, or null. */
export function currentAgent(view: RunView): AgentRole | null {
  for (let index = view.steps.length - 1; index >= 0; index -= 1) {
    const step = view.steps[index];
    if (step !== undefined && step.status === "running") return step.agent;
  }
  return null;
}

export interface StepGroup {
  label: string;
  steps: StepEntry[];
}

/** Setup (leading round-less plan) / Round N / Wrap-up (trailing synthesis). */
export function stepGroups(view: RunView): StepGroup[] {
  const groups: StepGroup[] = [];
  let current: StepGroup | null = null;
  let seenRound = false;
  for (const step of view.steps) {
    let label: string;
    if (step.round !== null) {
      seenRound = true;
      label = `Round ${step.round}`;
    } else {
      label = seenRound ? "Wrap-up" : "Setup";
    }
    if (current === null || current.label !== label) {
      current = { label, steps: [] };
      groups.push(current);
    }
    current.steps.push(step);
  }
  return groups;
}

/** One-line summary fragment set rendered in the step card header. */
export function stepSummary(entry: StepEntry): string {
  const parts: string[] = [`#${entry.index}`, agentLabel(entry.agent), entry.kind];
  if (entry.round !== null) parts.push(`r${entry.round}`);
  const message = entry.message;
  if (message !== null) {
    if (message.tool_calls !== null && message.tool_calls.length > 0) {
      parts.push(`${message.tool_calls.length} tools`);
    }
    if (message.claims !== null && message.claims.length > 0) {
      parts.push(pluralize(message.claims.length, "claim"));
    }
    if (message.verdicts !== null && message.verdicts.length > 0) {
      parts.push(pluralize(message.verdicts.length, "verdict"));
    }
    if (message.decision !== null) {
      const target = message.decision.target;
      parts.push(
        `decision: ${message.decision.action}${target !== null ? ` → ${target}` : ""}`,
      );
    }
  }
  parts.push(formatDuration(entry.durationMs));
  return parts.join(" · ");
}
