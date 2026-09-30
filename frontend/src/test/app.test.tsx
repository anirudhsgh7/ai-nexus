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
    expect(within(finalCard).getByText(/Final answer\./)).toBeInTheDocument();
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
