/**
 * App wiring flows (Phase 8 PRD §6.9): submit, live fold, final result,
 * failure, 409 adoption, validation, history replay, degraded backend.
 * api and sse are mocked; the fold itself is the real reducer.
 */

import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "../App";
import { ApiError } from "../api";
import type { RunStreamHandlers } from "../sse";
import type {
  HealthPayload,
  RunEvent,
  RunPayload,
  RunSummary,
} from "../types";
import failedTrace from "./fixtures/failed_trace.json";
import iterativeTrace from "./fixtures/iterative_trace.json";
import runDetail from "./fixtures/run_detail.json";
import runSummaries from "./fixtures/run_summaries.json";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return {
    ...actual,
    createRun: vi.fn(),
    listRuns: vi.fn(),
    getRun: vi.fn(),
    getHealth: vi.fn(),
  };
});
vi.mock("../sse", () => ({ openRunStream: vi.fn() }));

import * as api from "../api";
import { openRunStream } from "../sse";

const events = iterativeTrace as unknown as RunEvent[];
const failedEvents = failedTrace as unknown as RunEvent[];
const summaries = runSummaries as unknown as RunSummary[];

/** Resolution hook for the deferred getRun in the stale-refresh test. */
let deferredResolve: ((payload: RunPayload) => void) | null = null;

const mockCreateRun = vi.mocked(api.createRun);
const mockListRuns = vi.mocked(api.listRuns);
const mockGetRun = vi.mocked(api.getRun);
const mockGetHealth = vi.mocked(api.getHealth);
const mockOpenStream = vi.mocked(openRunStream);

function health(overrides: Partial<HealthPayload> = {}): HealthPayload {
  return {
    status: "ok",
    app: { name: "AI Nexus", version: "0.1.0" },
    provider: {
      name: "ollama",
      reachable: true,
      latency_ms: 1,
      server_version: "0.34.4",
    },
    config: {
      primary_model: "qwen2.5:14b-instruct",
      fallback_model: "qwen2.5:7b-instruct",
      num_ctx: 8192,
      max_concurrent_generations: 1,
    },
    checks: { primary_model_available: true, fallback_model_available: true },
    took_ms: 1,
    ...overrides,
  };
}

function payload(runId: string, task: string): RunPayload {
  return {
    run_id: runId,
    task,
    status: "running",
    created_at: "2026-09-30T12:00:00+00:00",
    started_at: null,
    finished_at: null,
    duration_ms: null,
    steps: [],
    rounds: [],
    selected_round: null,
    verification: null,
    accountability: null,
    final_message: null,
    error: null,
  };
}

function stubStream() {
  const close = vi.fn();
  const opened: string[] = [];
  let handlers: RunStreamHandlers | null = null;
  mockOpenStream.mockImplementation((runId, captured) => {
    opened.push(runId);
    handlers = captured;
    return { close, url: "memory://stream" };
  });
  return {
    close,
    opened,
    emit(list: RunEvent[]): void {
      if (handlers === null) throw new Error("stream not opened yet");
      const current = handlers;
      act(() => {
        for (const event of list) current.onEvent(event);
      });
    },
  };
}

beforeEach(() => {
  mockCreateRun.mockReset();
  mockListRuns.mockReset();
  mockGetRun.mockReset();
  mockGetHealth.mockReset();
  mockOpenStream.mockReset();
  deferredResolve = null;
  mockGetHealth.mockResolvedValue(health());
  mockListRuns.mockResolvedValue([]);
});

describe("shell", () => {
  it("renders header, empty history, and the empty-run hint", async () => {
    render(<App />);
    expect(screen.getByText("AI Nexus")).toBeInTheDocument();
    expect(await screen.findByText("No runs yet")).toBeInTheDocument();
    expect(screen.getByText(/Enter a task and press Run/)).toBeInTheDocument();
  });

  it("shows the unavailable banner and disables Run", async () => {
    mockGetHealth.mockResolvedValue(
      health({ status: "unavailable", hint: "Is Ollama running?" }),
    );
    render(<App />);
    expect(await screen.findByText(/Backend unavailable/)).toBeInTheDocument();
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Task"), "x");
    // even with a valid task the button stays disabled while unavailable
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Run" })).toBeDisabled(),
    );
  });

  it("shows the degraded banner but keeps Run enabled", async () => {
    mockGetHealth.mockResolvedValue(
      health({
        status: "degraded",
        hint: "Pull the primary model",
        checks: {
          primary_model_available: false,
          fallback_model_available: true,
        },
      }),
    );
    render(<App />);
    expect(await screen.findByText(/Backend degraded/)).toBeInTheDocument();
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Task"), "x");
    expect(screen.getByRole("button", { name: "Run" })).toBeEnabled();
  });
});

describe("running a task", () => {
  it("submits, opens the stream, and folds the live trace to the final result", async () => {
    mockCreateRun.mockResolvedValue({ run_id: "run-1", status: "running" });
    mockGetRun.mockResolvedValue(payload("run-1", "Should we build X?"));
    const stream = stubStream();

    render(<App />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Task"), "Should we build X?");
    await user.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(stream.opened).toEqual(["run-1"]));
    expect(mockCreateRun).toHaveBeenCalledWith("Should we build X?");

    stream.emit(events);

    expect(await screen.findByText("FINAL RESULT")).toBeInTheDocument();
    const finalCard = screen.getByLabelText("Final result");
    expect(
      within(finalCard).getByText(/Recommendation: adopt AI tooling/),
    ).toBeInTheDocument();
    expect(screen.getByText(/#2 · Researcher · research · r1/)).toBeInTheDocument();
    expect(screen.getByText(/#9 · Manager · synthesize/)).toBeInTheDocument();
    expect(screen.getByText("Round 1")).toBeInTheDocument();
    expect(screen.getByText("Round 2")).toBeInTheDocument();
    // Run is re-enabled once the run reached a terminal state
    expect(screen.getByRole("button", { name: "Run" })).toBeEnabled();
  });

  it("shows the failed-run error card for a failed trace", async () => {
    mockCreateRun.mockResolvedValue({ run_id: "run-2", status: "running" });
    mockGetRun.mockResolvedValue(payload("run-2", "What is 2+2?"));
    const stream = stubStream();

    render(<App />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Task"), "What is 2+2?");
    await user.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(stream.opened).toEqual(["run-2"]));

    stream.emit(failedEvents);

    expect(
      await screen.findByText(/RUN FAILED · RequestTimeoutError/),
    ).toBeInTheDocument();
    // the same error appears twice: on the failed step card and the run card
    expect(
      screen.getAllByText(/LLM request timed out/).length,
    ).toBeGreaterThanOrEqual(2);
    expect(
      screen.getAllByText(/AI_NEXUS_REQUEST_TIMEOUT_S/).length,
    ).toBeGreaterThanOrEqual(2);
  });

  it("adopts the active run on 409", async () => {
    mockCreateRun.mockRejectedValue(
      new ApiError({
        status: 409,
        type: "RunActiveError",
        message: "a run is already active",
        hint: "wait for run active-9 to finish",
        activeRunId: "active-9",
      }),
    );
    mockGetRun.mockResolvedValue(payload("active-9", "Other task"));
    const stream = stubStream();

    render(<App />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Task"), "Another task");
    await user.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(stream.opened).toEqual(["active-9"]));
    expect(
      await screen.findByText(/A run is already active — watching it\./),
    ).toBeInTheDocument();
  });

  it("renders an inline validation error on 422", async () => {
    mockCreateRun.mockRejectedValue(
      new ApiError({
        status: 422,
        type: "ValidationError",
        message: "String should have at least 1 character",
      }),
    );
    render(<App />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Task"), "x");
    await user.click(screen.getByRole("button", { name: "Run" }));

    expect(
      await screen.findByText("task must be 1–4000 characters"),
    ).toBeInTheDocument();
    expect(mockOpenStream).not.toHaveBeenCalled();
  });

  it("renders a network error card when the backend is unreachable", async () => {
    mockCreateRun.mockRejectedValue(
      new ApiError({
        status: 0,
        type: "NetworkError",
        message: "Cannot reach the backend",
        hint: "Is the backend running at http://127.0.0.1:8000?",
      }),
    );
    render(<App />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Task"), "x");
    await user.click(screen.getByRole("button", { name: "Run" }));

    expect(
      await screen.findByText("Cannot reach the backend"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Is the backend running at http:\/\/127\.0\.0\.1:8000\?/),
    ).toBeInTheDocument();
  });
});

describe("history", () => {
  it("loads a stored run through the stream and renders its trace", async () => {
    mockListRuns.mockResolvedValue(summaries);
    mockGetRun.mockResolvedValue(payload(summaries[0]?.run_id ?? "", "Should we build X?"));
    const stream = stubStream();

    render(<App />);
    expect(await screen.findByText("Should we build X?")).toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(
      screen.getByRole("button", { name: /Should we build X\?/ }),
    );

    await waitFor(() =>
      expect(stream.opened).toEqual([summaries[0]?.run_id]),
    );
    stream.emit(events);
    expect(await screen.findByText("FINAL RESULT")).toBeInTheDocument();
  });

  it("shows stored summaries with their status chips", async () => {
    mockListRuns.mockResolvedValue(summaries);
    render(<App />);
    expect(await screen.findByText("Is the 40% growth claim correct?")).toBeInTheDocument();
    expect(screen.getAllByText("completed").length).toBeGreaterThan(0);
    expect(screen.getAllByText("running").length).toBeGreaterThan(0);
  });
});

// ------------------------------------------------------ Phase 9: evidence

const detail = runDetail as unknown as RunPayload;

async function submitAndWatch(task: string) {
  const stream = stubStream();
  render(<App />);
  const user = userEvent.setup();
  await user.type(screen.getByLabelText("Task"), task);
  await user.click(screen.getByRole("button", { name: "Run" }));
  await waitFor(() => expect(stream.opened.length).toBeGreaterThan(0));
  return stream;
}

describe("evidence panels (Phase 9)", () => {
  it("renders round panels with verdicts, provenance, and the selected round", async () => {
    mockCreateRun.mockResolvedValue({ run_id: "run-1", status: "running" });
    mockGetRun.mockResolvedValue({ ...detail, run_id: "run-1" });
    const stream = await submitAndWatch(detail.task);
    stream.emit(events);

    expect(await screen.findByText("FINAL RESULT")).toBeInTheDocument();
    // one panel per persisted snapshot round — rounds arrive via the terminal
    // refresh, so the first render may not have them yet
    const panels = await screen.findAllByLabelText(/^Round \d+ evidence$/);
    expect(panels).toHaveLength(detail.rounds.length);
    // counts + exactly one selected round
    expect(
      screen.getAllByText(/\d+ supported · \d+ unresolved/),
    ).toHaveLength(detail.rounds.length);
    expect(await screen.findByText("SELECTED")).toBeInTheDocument();
    // verdicts render as text labels (not color-only)
    expect(
      screen.getAllByText(/^(Supported|Refuted|Unverifiable|Not evaluated)$/)
        .length,
    ).toBeGreaterThan(0);
    // the final card names the selected round's unresolved minority
    expect(
      await screen.findByText(/FINAL EVIDENCE · selected round \d+/),
    ).toBeInTheDocument();
    // decisions carry reason + instruction
    expect(screen.getAllByText(/^decision$/).length).toBeGreaterThan(0);

    // tool provenance: the SSE trace carries step messages, so a critique
    // re-emitted with its (real, 8b-shaped) tool envelope renders it
    const critique = events.find(
      (e) => e.type === "step_completed" && e.kind === "critique",
    );
    expect(critique?.message).not.toBeNull();
    stream.emit([
      {
        ...critique!,
        seq: Math.max(...events.map((e) => e.seq)) + 1,
        message: {
          ...critique!.message!,
          tool_calls: [
            {
              name: "web_search",
              arguments: { query: "INRIX 2025 traffic scorecard" },
              id: null,
            },
          ],
          tool_results: [
            {
              name: "web_search",
              content: JSON.stringify({
                ok: true,
                tool: "web_search",
                query: "INRIX 2025 traffic scorecard",
                provider: "ddg",
                results: [{ title: "t", url: "u", snippet: "s" }],
                truncated: false,
                attempts: ["ddg"],
                cached: true,
              }),
              error: null,
              duration_ms: 12.5,
            },
          ],
        },
      },
    ]);
    expect(
      await screen.findByText(/source: ddg \(cached\)/),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText(/web_search\("INRIX 2025 traffic scorecard"\)/)
        .length,
    ).toBeGreaterThan(0);
  });

  it("refreshes the detail payload on critique/decide steps and terminal events", async () => {
    mockCreateRun.mockResolvedValue({ run_id: "run-1", status: "running" });
    mockGetRun.mockResolvedValue(payload("run-1", "Should we build X?"));
    const stream = await submitAndWatch("Should we build X?");
    const before = mockGetRun.mock.calls.length;
    stream.emit(events);

    expect(await screen.findByText("FINAL RESULT")).toBeInTheDocument();
    // 2 critiques + 2 decides + 1 terminal refresh (no polling in between)
    await waitFor(() =>
      expect(mockGetRun.mock.calls.length).toBe(before + 5),
    );
  });

  it("labels a synthetic guard decision on a skipped decide step", async () => {
    mockCreateRun.mockResolvedValue({ run_id: "run-1", status: "running" });
    mockGetRun.mockResolvedValue(payload("run-1", "Guard task"));
    const stream = await submitAndWatch("Guard task");
    stream.emit([
      {
        seq: 7,
        type: "step_completed",
        run_id: "run-1",
        ts: "2026-10-02T12:00:00.000Z",
        step: 7,
        kind: "decide",
        agent: "manager",
        round: 2,
        task: null,
        message: {
          id: "guard-msg",
          from_agent: "manager",
          to_agent: null,
          type: "decision",
          content: "",
          claims: null,
          verdicts: null,
          decision: {
            action: "finish",
            target: null,
            instruction: "",
            reason: "revision produced no progress",
            confidence: 0,
          },
          verification: null,
          accountability: null,
          confidence: null,
          tool_calls: null,
          tool_results: null,
          retries: 0,
          round: 2,
          created_at: "2026-10-02T12:00:00.000Z",
        },
        duration_ms: 0,
        skipped: true,
        error: null,
      },
    ]);

    expect(await screen.findByText("guard")).toBeInTheDocument();
    expect(
      screen.getByText(/revision produced no progress/),
    ).toBeInTheDocument();
    // no snapshot exists for the aborted round -> no fabricated panel
    expect(screen.queryAllByLabelText(/^Round \d+ evidence$/)).toHaveLength(0);
  });

  it("shows no evidence panels for a run without round snapshots", async () => {
    mockCreateRun.mockResolvedValue({ run_id: "run-2", status: "running" });
    mockGetRun.mockResolvedValue(payload("run-2", "What is 2+2?"));
    const stream = await submitAndWatch("What is 2+2?");
    stream.emit(failedEvents);

    expect(
      await screen.findByText(/RUN FAILED · RequestTimeoutError/),
    ).toBeInTheDocument();
    expect(screen.queryAllByLabelText(/^Round \d+ evidence$/)).toHaveLength(0);
  });

  it("ignores a stale round refresh after switching runs", async () => {
    mockCreateRun.mockResolvedValue({ run_id: "run-a", status: "running" });
    mockListRuns.mockResolvedValue(summaries);
    // initial fetch for run-a, then one deferred refresh, then run-b payloads
    mockGetRun
      .mockResolvedValueOnce(payload("run-a", "Task A"))
      .mockImplementationOnce(
        () =>
          new Promise<RunPayload>((resolve) => {
            deferredResolve = resolve;
          }),
      )
      .mockResolvedValue(payload(summaries[0]?.run_id ?? "run-b", "Task B"));
    const stream = await submitAndWatch("Task A");

    // a decisive step fires the (deferred) refresh for run-a
    const critique = events.find(
      (e) => e.type === "step_completed" && e.kind === "critique",
    );
    expect(critique).toBeDefined();
    stream.emit([critique!]);
    await waitFor(() => expect(deferredResolve).not.toBeNull());

    // switch runs while that refresh is in flight
    const user = userEvent.setup();
    await user.click(
      screen.getByRole("button", { name: /Should we build X\?/ }),
    );
    await waitFor(() =>
      expect(stream.opened).toEqual(["run-a", summaries[0]?.run_id]),
    );

    // the stale response (carrying run-a's rounds) must be discarded
    deferredResolve!({ ...detail, run_id: "run-a" });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(screen.queryAllByLabelText(/^Round \d+ evidence$/)).toHaveLength(0);
  });
});

// ------------------------------------------------- Phase 11: audit panels

describe("audit panels (Phase 11)", () => {
  it("renders the Verifier's report from the step trace", async () => {
    mockCreateRun.mockResolvedValue({ run_id: "run-1", status: "running" });
    mockGetRun.mockResolvedValue({ ...detail, run_id: "run-1" });
    const stream = await submitAndWatch(detail.task);
    stream.emit(events);

    expect(await screen.findByText("FINAL RESULT")).toBeInTheDocument();
    const panel = await screen.findByLabelText(
      "Verification of the final answer",
    );
    // counts line mirrors the fixture's five entries
    expect(
      within(panel).getByText(
        /VERIFICATION · 5 claims · 2 verified · 1 partial · 1 contradicted · 1 unverifiable/,
      ),
    ).toBeInTheDocument();
    // every status renders as a text label, scoped to the panel
    expect(within(panel).getAllByText("Verified").length).toBe(2);
    expect(within(panel).getByText("Partially verified")).toBeInTheDocument();
    expect(within(panel).getByText("Contradicted")).toBeInTheDocument();
    expect(within(panel).getByText("Unverifiable")).toBeInTheDocument();
    // explanations and source references render
    expect(
      within(panel).getByText(/Primary source states the figure verbatim/),
    ).toBeInTheDocument();
    expect(
      within(panel).getByText(/sources: INRIX Global Traffic Scorecard/),
    ).toBeInTheDocument();
  });

  it("renders the Accountability report with enforced flags and provenance", async () => {
    mockCreateRun.mockResolvedValue({ run_id: "run-1", status: "running" });
    mockGetRun.mockResolvedValue({ ...detail, run_id: "run-1" });
    const stream = await submitAndWatch(detail.task);
    stream.emit(events);

    expect(await screen.findByText("FINAL RESULT")).toBeInTheDocument();
    const panel = await screen.findByLabelText(
      "Accountability process audit",
    );
    expect(
      within(panel).getByText("Warnings"),
    ).toBeInTheDocument(); // overall status chip
    expect(within(panel).getByText(/trace complete —/)).toBeInTheDocument();
    // provenance rows (scoped: claim ids also appear in round panels)
    expect(within(panel).getByText(/\[c6\]/)).toBeInTheDocument();
    // provenance rows for c2 and c6 are both supported
    expect(within(panel).getAllByText("Supported").length).toBeGreaterThanOrEqual(2);
    // flags: two echoed by the model + two appended by enforcement
    expect(
      within(panel).getAllByText(/unsupported final claim/).length,
    ).toBeGreaterThan(0);
    expect(within(panel).getByText(/premature stop/)).toBeInTheDocument();
    expect(within(panel).getByText(/unresolved claim suppressed/)).toBeInTheDocument();
    expect(
      within(panel).getByText(/confidence evidence mismatch/),
    ).toBeInTheDocument();
    // severity chips are text-labeled
    expect(within(panel).getAllByText("warning").length).toBeGreaterThan(0);
  });

  it("shows no audit panels before the audit steps exist", async () => {
    mockCreateRun.mockResolvedValue({ run_id: "run-1", status: "running" });
    mockGetRun.mockResolvedValue(payload("run-1", "Should we build X?"));
    const stream = await submitAndWatch("Should we build X?");
    // emit only the pre-audit portion of the trace
    const withoutAudits = events.filter(
      (e) => e.kind !== "verify" && e.kind !== "audit",
    );
    stream.emit(withoutAudits);

    expect(await screen.findByText("FINAL RESULT")).toBeInTheDocument();
    expect(
      screen.queryByLabelText("Verification of the final answer"),
    ).toBeNull();
    expect(
      screen.queryByLabelText("Accountability process audit"),
    ).toBeNull();
  });

  it("surfaces structured-output retries on the step card", async () => {
    mockCreateRun.mockResolvedValue({ run_id: "run-1", status: "running" });
    mockGetRun.mockResolvedValue(payload("run-1", "Should we build X?"));
    const stream = await submitAndWatch("Should we build X?");
    stream.emit([
      {
        seq: 90,
        type: "step_completed",
        run_id: "run-1",
        ts: "2026-10-03T12:00:00.000Z",
        step: 4,
        kind: "critique",
        agent: "skeptic",
        round: 1,
        task: null,
        message: {
          id: "retried",
          from_agent: "skeptic",
          to_agent: null,
          type: "critique",
          content: "Checked the claims twice.",
          claims: null,
          verdicts: null,
          decision: null,
          verification: null,
          accountability: null,
          confidence: null,
          tool_calls: null,
          tool_results: null,
          retries: 2,
          round: 1,
          created_at: "2026-10-03T12:00:00.000Z",
        },
        duration_ms: 12.0,
        skipped: null,
        error: null,
      },
    ]);

    expect(await screen.findByText(/2 retries/)).toBeInTheDocument();
  });
});
