/**
 * SSE client (Phase 8 PRD §6.5) — the ONLY module in `src/` that constructs
 * an `EventSource` (enforced by `architecture.test.ts`).
 *
 * Lifecycle rules that matter:
 * - Named events need one listener per type (they never fire `onmessage`).
 * - Terminal events close the stream; otherwise EventSource reconnects to a
 *   finished stream forever.
 * - Other errors leave the stream open (browser retry + Last-Event-ID).
 */

import { API_BASE } from "./api";
import type { ConnectionState, EventType, RunEvent } from "./types";

const EVENT_TYPES: readonly EventType[] = [
  "run_started",
  "step_started",
  "step_completed",
  "run_completed",
  "run_failed",
];

export function isKnownEventType(value: unknown): value is EventType {
  return (
    typeof value === "string" &&
    (EVENT_TYPES as readonly string[]).includes(value)
  );
}

/** Forward-compatible guard: unknown/malformed events are dropped, not folded. */
export function parseRunEvent(raw: string): RunEvent | null {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    console.warn("sse: dropping malformed event payload");
    return null;
  }
  if (typeof data !== "object" || data === null) {
    console.warn("sse: dropping non-object event payload");
    return null;
  }
  const candidate = data as Record<string, unknown>;
  if (typeof candidate.seq !== "number") {
    console.warn("sse: dropping event without seq");
    return null;
  }
  if (!isKnownEventType(candidate.type)) {
    console.warn(`sse: dropping unknown event type ${String(candidate.type)}`);
    return null;
  }
  return data as unknown as RunEvent;
}

export interface RunStreamHandlers {
  onEvent: (event: RunEvent) => void;
  onState: (state: ConnectionState, detail?: string) => void;
}

export interface RunStreamHandle {
  close(): void;
  readonly url: string;
}

export function openRunStream(
  runId: string,
  handlers: RunStreamHandlers,
): RunStreamHandle {
  const url = `${API_BASE}/api/runs/${encodeURIComponent(runId)}/events`;
  const source = new EventSource(url);
  let closed = false;
  const listeners: Array<{ type: EventType; listener: (event: Event) => void }> =
    [];

  const close = (): void => {
    if (closed) return;
    closed = true;
    for (const { type, listener } of listeners) {
      source.removeEventListener(type, listener);
    }
    listeners.length = 0;
    source.close();
    handlers.onState("closed");
  };

  for (const type of EVENT_TYPES) {
    const listener = (event: Event): void => {
      const data = (event as MessageEvent).data;
      const raw = typeof data === "string" ? data : "";
      const runEvent = parseRunEvent(raw);
      if (runEvent === null) return;
      handlers.onEvent(runEvent);
      if (runEvent.type === "run_completed" || runEvent.type === "run_failed") {
        close();
      }
    };
    listeners.push({ type, listener });
    source.addEventListener(type, listener);
  }

  source.onopen = (): void => {
    if (!closed) handlers.onState("live");
  };
  source.onerror = (): void => {
    if (!closed) handlers.onState("reconnecting");
  };

  return { close, url };
}
