/**
 * HTTP client (Phase 8 PRD §6.4) — the ONLY module in `src/` that calls
 * `fetch()` (enforced by `architecture.test.ts`).
 */

import type { HealthPayload, RunPayload, RunStatus, RunSummary } from "./types";

const rawBase: string | undefined = import.meta.env.VITE_API_BASE_URL;

export const API_BASE: string = (rawBase ?? "http://127.0.0.1:8000").replace(
  /\/+$/,
  "",
);

interface ApiErrorArgs {
  status: number;
  type: string;
  message: string;
  hint?: string;
  activeRunId?: string;
}

export class ApiError extends Error {
  readonly status: number;
  readonly type: string;
  readonly hint: string;
  readonly activeRunId: string | undefined;

  constructor(args: ApiErrorArgs) {
    super(args.message);
    this.name = "ApiError";
    this.status = args.status;
    this.type = args.type;
    this.hint = args.hint ?? "";
    this.activeRunId = args.activeRunId;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function networkError(): ApiError {
  return new ApiError({
    status: 0,
    type: "NetworkError",
    message: "Cannot reach the backend",
    hint: `Is the backend running at ${API_BASE}?`,
  });
}

function mapError(status: number, body: unknown): ApiError {
  if (isRecord(body) && isRecord(body.error)) {
    const error = body.error;
    return new ApiError({
      status,
      type: typeof error.type === "string" ? error.type : "HttpError",
      message:
        typeof error.message === "string" ? error.message : `HTTP ${status}`,
      hint: typeof error.hint === "string" ? error.hint : "",
      activeRunId:
        typeof error.active_run_id === "string"
          ? error.active_run_id
          : undefined,
    });
  }
  if (status === 422 && isRecord(body) && Array.isArray(body.detail)) {
    const first: unknown = body.detail[0];
    const message =
      isRecord(first) && typeof first.msg === "string"
        ? first.msg
        : "Invalid request";
    return new ApiError({ status, type: "ValidationError", message });
  }
  return new ApiError({ status, type: "HttpError", message: `HTTP ${status}` });
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw networkError();
  }
  const body = await readJson(response);
  if (!response.ok) throw mapError(response.status, body);
  return body as T;
}

export function createRun(
  task: string,
): Promise<{ run_id: string; status: RunStatus }> {
  return request("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ task }),
  });
}

export function listRuns(limit = 10): Promise<RunSummary[]> {
  return request(`/api/runs?limit=${limit}`, {
    headers: { Accept: "application/json" },
  });
}

export function getRun(runId: string): Promise<RunPayload> {
  return request(`/api/runs/${encodeURIComponent(runId)}`, {
    headers: { Accept: "application/json" },
  });
}

/**
 * Health special case: the backend returns 503 WITH a health payload when
 * unavailable, so parse the payload first and only map genuine failures.
 */
export async function getHealth(): Promise<HealthPayload> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/health`, {
      headers: { Accept: "application/json" },
    });
  } catch {
    throw networkError();
  }
  const body = await readJson(response);
  if (isRecord(body) && typeof body.status === "string") {
    return body as unknown as HealthPayload;
  }
  throw mapError(response.status, body);
}
