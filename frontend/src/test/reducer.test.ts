/**
 * The fold contract (Phase 8 PRD §6.6). Fixtures are wire-accurate dumps of
 * real backend output (see the phase report for how they were generated).
 */

import { describe, expect, it } from "vitest";

import { formatDuration } from "../format";
import {
  createRunView,
  currentAgent,
  foldEvent,
  initialRunView,
  setConnection,
  stepGroups,
  stepSummary,
  type RunView,
  type StepEntry,
} from "../reducer";
import type { RunEvent } from "../types";
import failedTrace from "./fixtures/failed_trace.json";
import iterativeTrace from "./fixtures/iterative_trace.json";

const events = iterativeTrace as unknown as RunEvent[];
const failedEvents = failedTrace as unknown as RunEvent[];

function foldAll(
  list: RunEvent[],
  view: RunView = createRunView("run1"),
): RunView {
  return list.reduce(foldEvent, view);
}

function makeEvent(
  partial: Partial<RunEvent> & Pick<RunEvent, "seq" | "type">,
): RunEvent {
  return {
    run_id: "run1",
    ts: "2026-09-30T12:00:00.000Z",
    step: null,
    kind: null,
    agent: null,
    round: null,
    task: null,
    message: null,
    duration_ms: null,
    skipped: null,
    error: null,
    ...partial,
  };
}

function stepAt(view: RunView, index: number): StepEntry {
  const step = view.steps[index];
  if (step === undefined) throw new Error(`no step at position ${index}`);
  return step;
}

describe("golden iterative trace", () => {
  const view = foldAll(events);

  it("folds the full 20-event log into a completed run with 9 steps", () => {
    expect(view.status).toBe("completed");
    expect(view.steps).toHaveLength(9);
    expect(view.steps.map((step) => step.kind)).toEqual([
      "plan", "research", "ideate", "critique", "decide",
      "revise", "critique", "decide", "synthesize",
    ]);
    expect(view.steps.map((step) => step.status)).toEqual(
      Array.from({ length: 9 }, () => "completed"),
    );
    expect(view.steps.map((step) => step.round)).toEqual([
      null, 1, 1, 1, 1, 2, 2, 2, null,
    ]);
    expect(view.lastSeq).toBe(events.length);
    expect(view.startedAt).not.toBeNull();
    expect(view.finishedAt).not.toBeNull();
    expect(view.error).toBeNull();
  });

  it("keeps the final message and run task", () => {
    expect(view.finalMessage?.content).toBe("Final answer.");
    expect(view.task).toBe("Should we build X?");
  });

  it("preserves structured message payloads verbatim", () => {
    const research = stepAt(view, 1);
    expect(research.message?.claims).toHaveLength(2);
    const critique = stepAt(view, 3);
    expect(critique.message?.verdicts).toHaveLength(3);
    expect(critique.message?.claims).toBeNull();
    const decide = stepAt(view, 4);
    expect(decide.message?.decision?.action).toBe("call_agent");
    expect(decide.message?.decision?.target).toBe("researcher");
  });

  it("re-folding the same log is deterministic", () => {
    expect(foldAll(events)).toEqual(foldAll(events));
  });

  it("reports no current agent once completed", () => {
    expect(currentAgent(view)).toBeNull();
  });

  it("groups steps into Setup / Round 1 / Round 2 / Wrap-up", () => {
    const groups = stepGroups(view);
    expect(groups.map((group) => group.label)).toEqual([
      "Setup", "Round 1", "Round 2", "Wrap-up",
    ]);
    expect(groups.map((group) => group.steps.length)).toEqual([1, 4, 3, 1]);
  });

  it("summarizes a step with counts and duration", () => {
    const entry = stepAt(view, 1);
    expect(stepSummary(entry)).toBe(
      `#2 · Researcher · research · r1 · 2 claims · ${formatDuration(entry.durationMs)}`,
    );
  });
});

describe("golden failed trace", () => {
  const view = foldAll(failedEvents);

  it("folds a ServerRestart-style failure into a failed run", () => {
    expect(view.status).toBe("failed");
    expect(view.error?.type).toBe("RequestTimeoutError");
    expect(view.finishedAt).not.toBeNull();
  });

  it("marks the in-flight step failed with the run error", () => {
    const step = stepAt(view, 0);
    expect(step.status).toBe("failed");
    expect(step.error?.type).toBe("RequestTimeoutError");
  });

  it("keeps the human-readable hint", () => {
    expect(view.error?.hint).toContain("AI_NEXUS_REQUEST_TIMEOUT_S");
  });
});

describe("fold edge cases", () => {
  it("creates skipped steps on step_completed alone", () => {
    const view = foldAll([
      makeEvent({ seq: 1, type: "run_started", task: "t" }),
      makeEvent({
        seq: 2, type: "step_completed", step: 6, kind: "decide",
        agent: "manager", round: 2, skipped: true, duration_ms: 0,
      }),
    ]);
    expect(view.steps).toHaveLength(1);
    expect(stepAt(view, 0).status).toBe("skipped");
    expect(stepAt(view, 0).skipped).toBe(true);
  });

  it("ignores events at or below the seq watermark", () => {
    const first = foldEvent(
      initialRunView,
      makeEvent({ seq: 1, type: "run_started", task: "t" }),
    );
    const second = foldEvent(
      first,
      makeEvent({ seq: 1, type: "run_started", task: "other" }),
    );
    expect(second).toBe(first);
  });

  it("upserts duplicate step_started events instead of duplicating", () => {
    let view = createRunView("run1");
    view = foldEvent(
      view,
      makeEvent({ seq: 1, type: "step_started", step: 1, kind: "plan", agent: "manager" }),
    );
    view = foldEvent(
      view,
      makeEvent({ seq: 2, type: "step_started", step: 1, kind: "plan", agent: "manager" }),
    );
    expect(view.steps).toHaveLength(1);
    expect(stepAt(view, 0).status).toBe("running");
    view = foldEvent(
      view,
      makeEvent({ seq: 3, type: "step_completed", step: 1, kind: "plan", agent: "manager", duration_ms: 10 }),
    );
    expect(view.steps).toHaveLength(1);
    expect(stepAt(view, 0).status).toBe("completed");
  });

  it("keeps steps ordered by index even if events arrive out of order", () => {
    let view = createRunView("run1");
    view = foldEvent(
      view,
      makeEvent({ seq: 1, type: "step_started", step: 2, kind: "research", agent: "researcher", round: 1 }),
    );
    view = foldEvent(
      view,
      makeEvent({ seq: 2, type: "step_started", step: 1, kind: "plan", agent: "manager" }),
    );
    expect(view.steps.map((step) => step.index)).toEqual([1, 2]);
  });

  it("tracks the current agent while a step is running", () => {
    const view = foldAll(events.slice(0, 4));
    expect(currentAgent(view)).toBe("researcher");
  });

  it("marks a step failed on run_failed when the index matches", () => {
    let view = createRunView("run1");
    view = foldEvent(
      view,
      makeEvent({ seq: 1, type: "step_started", step: 1, kind: "research", agent: "researcher", round: 1 }),
    );
    const error = { type: "ServerRestart", message: "interrupted", hint: "" };
    view = foldEvent(
      view,
      makeEvent({ seq: 2, type: "run_failed", step: 1, error }),
    );
    expect(stepAt(view, 0).status).toBe("failed");
    expect(stepAt(view, 0).error).toEqual(error);
  });

  it("survives run_failed with no matching step", () => {
    const view = foldAll([
      makeEvent({ seq: 1, type: "run_started", task: "t" }),
      makeEvent({
        seq: 2, type: "run_failed",
        error: { type: "X", message: "boom", hint: "" },
      }),
    ]);
    expect(view.status).toBe("failed");
    expect(view.steps).toEqual([]);
  });

  it("groups Setup / Round 1 / Wrap-up when rounds are present", () => {
    const plan = makeEvent({ seq: 1, type: "step_started", step: 1, kind: "plan", agent: "manager" });
    const research = makeEvent({ seq: 2, type: "step_started", step: 2, kind: "research", agent: "researcher", round: 1 });
    const synth = makeEvent({ seq: 3, type: "step_started", step: 3, kind: "synthesize", agent: "manager" });
    const view = foldAll([plan, research, synth]);
    expect(stepGroups(view).map((group) => group.label)).toEqual([
      "Setup", "Round 1", "Wrap-up",
    ]);
  });

  it("groups a degenerate round-less log under Setup (round 1 always exists in real runs)", () => {
    const plan = makeEvent({ seq: 1, type: "step_started", step: 1, kind: "plan", agent: "manager" });
    const synth = makeEvent({ seq: 2, type: "step_started", step: 2, kind: "synthesize", agent: "manager" });
    const view = foldAll([plan, synth]);
    expect(stepGroups(view).map((group) => group.label)).toEqual(["Setup"]);
  });

  it("setConnection returns a new view with only the connection changed", () => {
    const view = createRunView("run1", "task");
    const next = setConnection(view, "live");
    expect(next).not.toBe(view);
    expect(next.connection).toBe("live");
    expect(next.task).toBe("task");
    expect(view.connection).toBe("connecting");
  });

  it("initialRunView is an empty connecting state", () => {
    expect(initialRunView).toMatchObject({
      runId: null,
      steps: [],
      status: "running",
      lastSeq: 0,
      connection: "connecting",
    });
  });
});
