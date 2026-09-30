/**
 * SSE client lifecycle (Phase 8 PRD §6.5): listener registration, parsing,
 * terminal close (no reconnect loops), reconnect-on-error, idempotent close.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { openRunStream, parseRunEvent, type RunStreamHandlers } from "../sse";
import type { ConnectionState, RunEvent } from "../types";

class FakeEventSource {
  static instances: FakeEventSource[] = [];

  readonly url: string;
  readonly listeners = new Map<string, Set<(event: MessageEvent) => void>>();
  onopen: ((event: Event) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
    const set = this.listeners.get(type) ?? new Set();
    set.add(listener as (event: MessageEvent) => void);
    this.listeners.set(type, set);
  }

  removeEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
    this.listeners.get(type)?.delete(listener as (event: MessageEvent) => void);
  }

  close(): void {
    this.closed = true;
  }

  listenerCount(type: string): number {
    return this.listeners.get(type)?.size ?? 0;
  }

  emit(type: string, payload: string): void {
    for (const listener of this.listeners.get(type) ?? []) {
      listener({ data: payload } as MessageEvent);
    }
  }

  emitEvent(event: RunEvent): void {
    this.emit(event.type, JSON.stringify(event));
  }

  emitOpen(): void {
    this.onopen?.(new Event("open"));
  }

  emitError(): void {
    this.onerror?.(new Event("error"));
  }
}

function makeEvent(seq: number, type: RunEvent["type"]): RunEvent {
  return {
    seq,
    type,
    run_id: "r1",
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
  };
}

interface Capture {
  events: RunEvent[];
  states: ConnectionState[];
}

function captureHandlers(capture: Capture): RunStreamHandlers {
  return {
    onEvent: (event) => capture.events.push(event),
    onState: (state) => capture.states.push(state),
  };
}

beforeEach(() => {
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("parseRunEvent", () => {
  it("accepts known event shapes", () => {
    const event = parseRunEvent(JSON.stringify(makeEvent(3, "step_started")));
    expect(event?.seq).toBe(3);
    expect(event?.type).toBe("step_started");
  });

  it("drops malformed JSON", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(parseRunEvent("{not json")).toBeNull();
    expect(warn).toHaveBeenCalled();
  });

  it("drops payloads without a numeric seq", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(parseRunEvent(JSON.stringify({ type: "run_started" }))).toBeNull();
  });

  it("drops unknown event types (forward compatible)", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(
      parseRunEvent(JSON.stringify({ seq: 1, type: "html_delta" })),
    ).toBeNull();
  });
});

describe("openRunStream", () => {
  it("targets the run events endpoint and registers every listener", () => {
    const capture: Capture = { events: [], states: [] };
    openRunStream("run-9", captureHandlers(capture));
    const source = FakeEventSource.instances[0];
    expect(source?.url).toContain("/api/runs/run-9/events");
    for (const type of [
      "run_started", "step_started", "step_completed", "run_completed", "run_failed",
    ]) {
      expect(source?.listenerCount(type)).toBe(1);
    }
  });

  it("forwards parsed events and reports live on open", () => {
    const capture: Capture = { events: [], states: [] };
    openRunStream("r1", captureHandlers(capture));
    const source = FakeEventSource.instances[0];
    source?.emitOpen();
    source?.emitEvent(makeEvent(1, "run_started"));
    expect(capture.states).toEqual(["live"]);
    expect(capture.events.map((event) => event.seq)).toEqual([1]);
  });

  it("closes the stream on terminal events and removes listeners", () => {
    const capture: Capture = { events: [], states: [] };
    openRunStream("r1", captureHandlers(capture));
    const source = FakeEventSource.instances[0];
    source?.emitEvent(makeEvent(1, "run_completed"));
    expect(source?.closed).toBe(true);
    expect(capture.states).toContain("closed");
    expect(source?.listenerCount("run_completed")).toBe(0);
  });

  it("reports reconnecting on error without closing", () => {
    const capture: Capture = { events: [], states: [] };
    openRunStream("r1", captureHandlers(capture));
    const source = FakeEventSource.instances[0];
    source?.emitError();
    expect(capture.states).toEqual(["reconnecting"]);
    expect(source?.closed).toBe(false);
  });

  it("close() is idempotent and silent afterwards", () => {
    const capture: Capture = { events: [], states: [] };
    const handle = openRunStream("r1", captureHandlers(capture));
    const source = FakeEventSource.instances[0];
    handle.close();
    handle.close();
    expect(capture.states).toEqual(["closed"]);
    expect(source?.closed).toBe(true);
    // late errors after close must not emit states
    source?.emitError();
    source?.emitEvent(makeEvent(5, "run_completed"));
    expect(capture.states).toEqual(["closed"]);
  });

  it("drops malformed frames without disturbing the stream", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const capture: Capture = { events: [], states: [] };
    openRunStream("r1", captureHandlers(capture));
    const source = FakeEventSource.instances[0];
    source?.emit("step_completed", "{not json");
    source?.emitEvent(makeEvent(2, "step_completed"));
    expect(capture.events.map((event) => event.seq)).toEqual([2]);
    expect(source?.closed).toBe(false);
  });
});
