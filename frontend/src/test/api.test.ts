/**
 * API client contract (Phase 8 PRD §6.4): success parsing and the dual error
 * shapes (app envelope + FastAPI 422 detail) plus transport failures.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { API_BASE, ApiError, createRun, getHealth, getRun, listRuns } from "../api";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

async function caught(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    if (error instanceof ApiError) return error;
    throw error;
  }
  throw new Error("expected the promise to reject");
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("createRun", () => {
  it("parses the 202 payload", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(202, {
        run_id: "abc",
        status: "running",
        task: "t",
        created_at: "2026-09-30T12:00:00+00:00",
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(createRun("t")).resolves.toMatchObject({
      run_id: "abc",
      status: "running",
    });
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${API_BASE}/api/runs`);
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({ task: "t" });
  });

  it("maps the 409 envelope including active_run_id", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(409, {
          error: {
            type: "RunActiveError",
            message: "a run is already active",
            hint: "wait for run other to finish",
            active_run_id: "other",
          },
        }),
      ),
    );

    const error = await caught(createRun("t"));
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(409);
    expect(error.type).toBe("RunActiveError");
    expect(error.activeRunId).toBe("other");
    expect(error.hint).toContain("other");
  });

  it("maps FastAPI's 422 detail shape", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(422, {
          detail: [
            {
              loc: ["body", "task"],
              msg: "String should have at least 1 character",
              type: "string_too_short",
            },
          ],
        }),
      ),
    );

    const error = await caught(createRun(""));
    expect(error.status).toBe(422);
    expect(error.type).toBe("ValidationError");
    expect(error.message).toBe("String should have at least 1 character");
  });

  it("maps transport failures to NetworkError with the base URL hint", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fail")));

    const error = await caught(createRun("t"));
    expect(error.status).toBe(0);
    expect(error.type).toBe("NetworkError");
    expect(error.hint).toContain(API_BASE);
  });

  it("falls back to HttpError for unknown bodies", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("boom", { status: 500 })),
    );

    const error = await caught(createRun("t"));
    expect(error.status).toBe(500);
    expect(error.type).toBe("HttpError");
    expect(error.message).toBe("HTTP 500");
  });
});

describe("listRuns and getRun", () => {
  it("passes the limit and parses summaries", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(200, [
        {
          run_id: "r1",
          status: "completed",
          task_preview: "x",
          created_at: "2026-09-30T12:00:00+00:00",
          duration_ms: 1000,
        },
      ]),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(listRuns(10)).resolves.toHaveLength(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe(`${API_BASE}/api/runs?limit=10`);
  });

  it("maps 404 envelopes", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(404, {
          error: { type: "RunNotFoundError", message: "unknown run: x", hint: "" },
        }),
      ),
    );

    const error = await caught(getRun("x"));
    expect(error.status).toBe(404);
    expect(error.type).toBe("RunNotFoundError");
  });
});

describe("getHealth", () => {
  it("parses 200 payloads", async () => {
    const payload = {
      status: "ok",
      app: { name: "AI Nexus", version: "0.1.0" },
      provider: { name: "ollama", reachable: true, latency_ms: 1, server_version: "0.34.4" },
      config: { primary_model: "m", fallback_model: "f", num_ctx: 8192, max_concurrent_generations: 1 },
      checks: { primary_model_available: true, fallback_model_available: true },
      took_ms: 1,
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(200, payload)));

    await expect(getHealth()).resolves.toMatchObject({ status: "ok" });
  });

  it("parses the 503 payload instead of throwing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(503, {
          status: "unavailable",
          hint: "Is Ollama running?",
          app: { name: "AI Nexus", version: "0.1.0" },
          provider: { name: "ollama", reachable: false, latency_ms: null, server_version: null },
          config: { primary_model: "m", fallback_model: "f", num_ctx: 8192, max_concurrent_generations: 1 },
          checks: { primary_model_available: false, fallback_model_available: false },
          took_ms: 1,
        }),
      ),
    );

    await expect(getHealth()).resolves.toMatchObject({
      status: "unavailable",
      hint: "Is Ollama running?",
    });
  });

  it("throws NetworkError when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fail")));
    const error = await caught(getHealth());
    expect(error.type).toBe("NetworkError");
  });
});
