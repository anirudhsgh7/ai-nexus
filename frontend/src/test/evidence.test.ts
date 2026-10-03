/**
 * Evidence derivation (Phase 9 PRD §9.2): golden build over the wire-accurate
 * fixture, claim/verdict/retirement joins, tool provenance (incl. the
 * overflow-preview recovery path), and degenerate inputs.
 */

import { describe, expect, it } from "vitest";

import {
  buildEvidence,
  evidenceFor,
  unresolvedAtSynthesis,
} from "../evidence";
import type {
  AgentMessage,
  RoundSummary,
  RunPayload,
  ToolResult,
} from "../types";
import runDetail from "./fixtures/run_detail.json";

const payload = runDetail as unknown as RunPayload;

function round(overrides: Partial<RoundSummary>): RoundSummary {
  return {
    round: 1,
    supported: 0,
    unresolved: 0,
    decision: null,
    claims: [],
    origins: {},
    verdicts: [],
    ...overrides,
  };
}

function message(overrides: Partial<AgentMessage>): AgentMessage {
  return {
    id: "m1",
    from_agent: "researcher",
    to_agent: null,
    type: "finding",
    content: "",
    claims: null,
    verdicts: null,
    decision: null,
    verification: null,
    accountability: null,
    confidence: null,
    tool_calls: null,
    tool_results: null,
    retries: 0,
    round: null,
    created_at: "2026-10-02T12:00:00Z",
    ...overrides,
  };
}

function result(content: string, error: string | null = null): ToolResult {
  return { name: "web_search", content, error, duration_ms: 10 };
}

// ------------------------------------------------------------------- golden

describe("buildEvidence over the live fixture", () => {
  const evidence = buildEvidence(
    payload.rounds,
    payload.steps,
    payload.selected_round,
  );

  it("covers every snapshot in ascending order", () => {
    expect(evidence).toHaveLength(payload.rounds.length);
    expect(evidence.map((r) => r.round)).toEqual(
      payload.rounds.map((r) => r.round),
    );
    expect(evidence.length).toBeGreaterThanOrEqual(2);
  });

  it("marks exactly the selected round", () => {
    const selected = evidence.filter((r) => r.selected);
    expect(selected).toHaveLength(1);
    expect(selected[0]?.round).toBe(payload.selected_round);
  });

  it("joins origins and verdicts onto run-level claim ids", () => {
    const first = evidence[0];
    expect(first).toBeDefined();
    expect(first?.claims).toHaveLength(payload.rounds[0]?.claims.length ?? 0);
    for (const record of first?.claims ?? []) {
      expect(first?.claims.some((c) => c.id === record.id)).toBe(true);
      const snapshot = payload.rounds.find((r) => r.round === first?.round);
      expect(record.origin).toBe(snapshot?.origins[record.id] ?? null);
      const snapshotVerdict = snapshot?.verdicts.find(
        (v) => v.claim_id === record.id,
      );
      expect(record.verdict?.verdict ?? null).toBe(
        snapshotVerdict?.verdict ?? null,
      );
      expect(record.firstSeenRound).toBeLessThanOrEqual(record.lastSeenRound);
    }
  });

  it("classifies unresolved claims as anything without a supported verdict", () => {
    for (const record of evidence.flatMap((r) => r.unresolvedClaims)) {
      expect(record.verdict === null || record.verdict.verdict !== "supported");
    }
    const selected = evidence.find((r) => r.selected);
    if (selected !== undefined) {
      expect(unresolvedAtSynthesis(evidence)).toEqual(
        selected.unresolvedClaims,
      );
    }
  });

  it("retires claims the next revision dropped", () => {
    const retired = evidence.flatMap((r) => r.retiredClaims);
    expect(retired.length).toBeGreaterThan(0);
    for (const record of retired) {
      expect(record.retiredInRound).not.toBeNull();
      const dropping = evidence.find((r) => r.round === record.retiredInRound);
      // present in the previous snapshot, gone from the dropping round's claims
      expect(dropping?.claims.some((c) => c.id === record.id)).toBe(false);
      expect(dropping?.retiredClaims.some((c) => c.id === record.id)).toBe(true);
      const previous = evidence.find(
        (r) => r.round === (record.retiredInRound ?? 0) - 1,
      );
      expect(previous?.claims.some((c) => c.id === record.id)).toBe(true);
    }
  });

  it("surfaces tool provenance with provider and a cached hit", () => {
    const tools = evidence.flatMap((r) => r.tools);
    expect(tools.length).toBeGreaterThan(0);
    expect(tools.some((t) => t.provider !== null)).toBe(true);
    expect(tools.some((t) => t.cached === true)).toBe(true);
    expect(tools.some((t) => t.query !== null)).toBe(true);
  });
});

// ------------------------------------------------------------- degenerate

describe("buildEvidence edge cases", () => {
  it("returns [] for no rounds", () => {
    expect(buildEvidence([], [], 1)).toEqual([]);
  });

  it("handles missing origins and missing verdicts without crashing", () => {
    const evidence = buildEvidence(
      [
        round({
          claims: [
            {
              id: "c1",
              statement: "s",
              status: "unverified",
              confidence: null,
              evidence: [],
            },
          ],
          origins: {},
        }),
      ],
      [],
      null,
    );
    expect(evidence[0]?.claims[0]?.origin).toBeNull();
    expect(evidence[0]?.claims[0]?.verdict).toBeNull();
    expect(evidence[0]?.unresolvedClaims).toHaveLength(1);
    expect(evidence[0]?.selected).toBe(false);
    expect(unresolvedAtSynthesis(evidence)).toEqual([]);
  });

  it("excludes supported claims from the unresolved list", () => {
    const evidence = buildEvidence(
      [
        round({
          claims: [
            {
              id: "c1",
              statement: "a",
              status: "fact",
              confidence: 0.9,
              evidence: [],
            },
            {
              id: "c2",
              statement: "b",
              status: "opinion",
              confidence: null,
              evidence: [],
            },
          ],
          origins: { c1: "researcher", c2: "ideator" },
          verdicts: [
            {
              claim_id: "c1",
              verdict: "supported",
              objection: "holds",
              evidence: [],
            },
            {
              claim_id: "c2",
              verdict: "unverifiable",
              objection: "no data",
              evidence: [],
            },
          ],
        }),
      ],
      [],
      1,
    );
    expect(evidence[0]?.unresolvedClaims.map((c) => c.id)).toEqual(["c2"]);
    expect(unresolvedAtSynthesis(evidence).map((c) => c.id)).toEqual(["c2"]);
  });

  it("keeps first/last seen across rounds and retires only on disappearance", () => {
    const claim = {
      id: "c1",
      statement: "carried",
      status: "unverified" as const,
      confidence: null,
      evidence: [],
    };
    const evidence = buildEvidence(
      [
        round({ round: 1, claims: [claim] }),
        round({ round: 2, claims: [claim] }),
        round({ round: 3, claims: [] }),
      ],
      [],
      3,
    );
    expect(evidence[0]?.claims[0]).toMatchObject({
      firstSeenRound: 1,
      lastSeenRound: 2,
      // only the last standing copy carries the retirement
      retiredInRound: null,
    });
    expect(evidence[1]?.claims[0]).toMatchObject({
      firstSeenRound: 1,
      lastSeenRound: 2,
      retiredInRound: 3,
    });
    expect(evidence[2]?.retiredClaims.map((c) => c.id)).toEqual(["c1"]);
    expect(evidence[0]?.retiredClaims).toEqual([]);
    expect(evidence[1]?.retiredClaims).toEqual([]);
  });

  it("deduplicates verdict display by claim id (last wins)", () => {
    const evidence = buildEvidence(
      [
        round({
          claims: [
            {
              id: "c1",
              statement: "s",
              status: "unverified",
              confidence: null,
              evidence: [],
            },
          ],
          verdicts: [
            {
              claim_id: "c1",
              verdict: "unverifiable",
              objection: "first",
              evidence: [],
            },
            {
              claim_id: "c1",
              verdict: "refuted",
              objection: "last",
              evidence: [],
            },
          ],
        }),
      ],
      [],
      null,
    );
    expect(evidence[0]?.claims[0]?.verdict?.objection).toBe("last");
  });

  it("returns null evidence for null/unknown rounds", () => {
    expect(evidenceFor([], null)).toBeNull();
    expect(evidenceFor([], 7)).toBeNull();
  });
});

// ------------------------------------------------------- tool provenance

describe("tool provenance", () => {
  function toolsFor(content: string, error: string | null = null) {
    return buildEvidence(
      [round({ round: 1 })],
      [
        {
          round: 1,
          message: message({
            tool_calls: [
              {
                name: "web_search",
                arguments: { query: "growth 2025" },
                id: null,
              },
            ],
            tool_results: [result(content, error)],
          }),
        },
      ],
      null,
    )[0]?.tools;
  }

  it("reads provider/cached/attempts/results from a success envelope", () => {
    const tools = toolsFor(
      JSON.stringify({
        ok: true,
        tool: "web_search",
        query: "growth 2025",
        provider: "bing",
        results: [{ title: "t", url: "u", snippet: "s" }],
        truncated: false,
        attempts: ["ddg", "bing"],
        cached: true,
      }),
    );
    expect(tools).toHaveLength(1);
    expect(tools?.[0]).toMatchObject({
      tool: "web_search",
      query: "growth 2025",
      ok: true,
      provider: "bing",
      cached: true,
      attempts: ["ddg", "bing"],
      resultCount: 1,
    });
  });

  it("recovers provider from a Phase 6 overflow preview envelope", () => {
    const preview = JSON.stringify({
      ok: true,
      tool: "web_search",
      provider: "ddg",
      results: [{ title: "t" }, { title: "u" }],
      attempts: ["ddg"],
      cached: false,
    });
    const tools = toolsFor(
      JSON.stringify({
        ok: true,
        tool: "web_search",
        truncated: true,
        message: "result exceeded 2000 characters",
        preview,
      }),
    );
    expect(tools?.[0]).toMatchObject({
      provider: "ddg",
      cached: false,
      resultCount: 2,
    });
  });

  it("extracts provenance textually from a truncated overflow preview", () => {
    const inner =
      '{"ok":true,"tool":"web_search","query":"q","provider":"ddg",' +
      '"results":[{"title":"t"}],"truncated":false,' +
      '"attempts":["ddg","bing"],"cached":true}';
    // tail cut: preview is no longer valid JSON, but every key survived
    const tools = toolsFor(
      JSON.stringify({
        ok: true,
        tool: "web_search",
        truncated: true,
        message: "result exceeded 2000 characters",
        preview: inner.slice(0, inner.length - 1),
      }),
    );
    expect(tools?.[0]).toMatchObject({
      provider: "ddg",
      cached: true,
      attempts: ["ddg", "bing"],
      resultCount: null,
    });
    // head-only cut: `provider` sits before `results`, so it survives alone
    const head = inner.slice(0, inner.indexOf('"results"'));
    const headTools = toolsFor(
      JSON.stringify({
        ok: true,
        tool: "web_search",
        truncated: true,
        message: "result exceeded 2000 characters",
        preview: head,
      }),
    );
    expect(headTools?.[0]).toMatchObject({
      provider: "ddg",
      cached: null,
      attempts: [],
    });
  });

  it("falls back gracefully on unparseable and missing envelopes", () => {
    const unparseable = toolsFor("{not json");
    expect(unparseable?.[0]).toMatchObject({
      ok: true, // no ToolResult.error — envelope just unreadable
      provider: null,
      cached: null,
      attempts: [],
      resultCount: null,
    });
    const failed = toolsFor("whatever", "network_error");
    expect(failed?.[0]).toMatchObject({
      ok: false,
      error: "network_error",
      provider: null,
    });
  });

  it("marks calls without a parallel result", () => {
    const tools = buildEvidence(
      [round({ round: 1 })],
      [
        {
          round: 1,
          message: message({
            tool_calls: [
              { name: "file_search", arguments: { query: "x" }, id: null },
            ],
            tool_results: [],
          }),
        },
      ],
      null,
    )[0]?.tools;
    expect(tools?.[0]).toMatchObject({
      tool: "file_search",
      ok: false,
      error: "no result",
    });
  });

  it("only collects steps belonging to the round", () => {
    const evidence = buildEvidence(
      [round({ round: 1 }), round({ round: 2 })],
      [
        {
          round: 2,
          message: message({
            tool_calls: [
              { name: "web_search", arguments: { query: "q" }, id: null },
            ],
            tool_results: [
              result(JSON.stringify({ ok: true, provider: "ddg" })),
            ],
          }),
        },
      ],
      null,
    );
    expect(evidence[0]?.tools).toEqual([]);
    expect(evidence[1]?.tools).toHaveLength(1);
  });
});
