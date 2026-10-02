/**
 * Pure evidence derivation (Phase 9 PRD §6.3).
 *
 * Joins the backend's round snapshots (authoritative run-level claim ids,
 * origins, verdicts) with the step trace (tool provenance) into the data
 * the evidence panels render. No I/O, no React, no clock, no formatting —
 * enforced by `architecture.test.ts`.
 *
 * Why round-centric: agent messages carry message-local claim ids while the
 * orchestrator renumbers claims run-level in `RoundSnapshot`. Rendering both
 * would invite false matches, so message-local ids are never shown (§6.5).
 */

import type {
  AgentMessage,
  AgentRole,
  Claim,
  ManagerDecision,
  RoundSummary,
  ToolResult,
  Verdict,
} from "./types";

/** Structural minimum: `StepEntry` (reducer) satisfies this unchanged. */
export type EvidenceStep = Pick<
  { round: number | null; message: AgentMessage | null },
  "round" | "message"
>;

export interface ClaimRecord {
  id: string;
  statement: string;
  status: Claim["status"];
  confidence: number | null;
  evidence: Claim["evidence"];
  origin: AgentRole | null;
  verdict: Verdict | null;
  firstSeenRound: number;
  lastSeenRound: number;
  /** The round whose revision dropped it; null while the claim is active. */
  retiredInRound: number | null;
}

export interface ToolProvenance {
  tool: string;
  query: string | null;
  ok: boolean;
  error: string | null;
  provider: string | null;
  cached: boolean | null;
  attempts: string[];
  resultCount: number | null;
}

export interface RoundEvidence {
  round: number;
  supported: number;
  unresolved: number;
  decision: ManagerDecision | null;
  claims: ClaimRecord[];
  unresolvedClaims: ClaimRecord[];
  /** Claims this round's revision dropped (records from the previous round). */
  retiredClaims: ClaimRecord[];
  tools: ToolProvenance[];
  selected: boolean;
}

interface Envelope {
  ok?: boolean;
  error?: string;
  provider?: string;
  cached?: boolean;
  attempts?: unknown;
  results?: unknown[];
  preview?: string;
}

function strList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((a): a is string => typeof a === "string")
    : [];
}

/**
 * The Phase 6 overflow preview is *truncated* (shrunk until the envelope fits
 * `tool_result_max_chars`), so `JSON.parse(preview)` usually fails. The keys
 * we care about sit in fixed positions: `provider` right after `query`,
 * `cached`/`attempts` only when the tail survived — recover what exists
 * textually instead of dropping it (Phase 9 PRD §18 amendment).
 */
function extractPreviewFields(preview: string): {
  provider: string | null;
  cached: boolean | null;
  attempts: string[];
  resultCount: number | null;
} {
  const provider = /"provider":"([A-Za-z0-9_.-]+)"/.exec(preview)?.[1] ?? null;
  const cachedMatch = /"cached":(true|false)/.exec(preview);
  const cached = cachedMatch === null ? null : cachedMatch[1] === "true";
  let attempts: string[] = [];
  const attemptsMatch = /"attempts":\[([^\]]*)\]/.exec(preview);
  if (attemptsMatch !== null) {
    try {
      attempts = strList(JSON.parse(`[${attemptsMatch[1]}]`));
    } catch {
      attempts = [];
    }
  }
  return { provider, cached, attempts, resultCount: null };
}

/** Unwrap one tool-result content string into provenance fields. */
function parseEnvelope(result: ToolResult): {
  provider: string | null;
  cached: boolean | null;
  attempts: string[];
  resultCount: number | null;
} {
  const empty = {
    provider: null,
    cached: null,
    attempts: [] as string[],
    resultCount: null as number | null,
  };
  let parsed: Envelope;
  try {
    parsed = JSON.parse(result.content) as Envelope;
  } catch {
    return empty;
  }
  const provider =
    typeof parsed.provider === "string" ? parsed.provider : null;
  const cached = typeof parsed.cached === "boolean" ? parsed.cached : null;
  const attempts = strList(parsed.attempts);
  const resultCount = Array.isArray(parsed.results)
    ? parsed.results.length
    : null;
  if (provider !== null || cached !== null || resultCount !== null) {
    return { provider, cached, attempts, resultCount };
  }
  // Phase 6 overflow preview: recover provenance from inside `preview` —
  // first by parsing it (whole envelope), then textually (truncated head).
  if (typeof parsed.preview === "string") {
    try {
      const inner = JSON.parse(parsed.preview) as Envelope;
      return {
        provider: typeof inner.provider === "string" ? inner.provider : null,
        cached: typeof inner.cached === "boolean" ? inner.cached : null,
        attempts: strList(inner.attempts),
        resultCount: Array.isArray(inner.results)
          ? inner.results.length
          : null,
      };
    } catch {
      return extractPreviewFields(parsed.preview);
    }
  }
  return empty;
}

function toolProvenance(message: AgentMessage): ToolProvenance[] {
  const calls = message.tool_calls ?? [];
  const results = message.tool_results ?? [];
  return calls.map((call, index) => {
    const result = results[index] ?? null;
    const args = call.arguments;
    const query = typeof args.query === "string" ? args.query : null;
    if (result === null) {
      return {
        tool: call.name,
        query,
        ok: false,
        error: "no result",
        provider: null,
        cached: null,
        attempts: [],
        resultCount: null,
      };
    }
    const envelope = parseEnvelope(result);
    return {
      tool: call.name,
      query,
      ok: result.error === null,
      error: result.error,
      ...envelope,
    };
  });
}

export function buildEvidence(
  rounds: RoundSummary[],
  steps: EvidenceStep[],
  selectedRound: number | null,
): RoundEvidence[] {
  const sorted = [...rounds].sort((a, b) => a.round - b.round);
  if (sorted.length === 0) return [];

  // Cross-round spans (the future is unknown while iterating → final pass).
  const span = new Map<string, { first: number; last: number }>();
  for (const round of sorted) {
    for (const claim of round.claims) {
      const seen = span.get(claim.id);
      if (seen === undefined) {
        span.set(claim.id, { first: round.round, last: round.round });
      } else {
        seen.last = round.round;
      }
    }
  }

  // Retirement: an id in round N but not N+1 was dropped by round N+1's
  // revision. The dropped records (round N's copies) are shared into
  // round N+1's retiredClaims so the chip and the verdict travel with them.
  const retired = new Map<number, ClaimRecord[]>();
  let previousIds: Set<string> | null = null;
  let previousRecords: ClaimRecord[] | null = null;

  return sorted.map((round) => {
    const verdictById = new Map(
      round.verdicts.map((verdict) => [verdict.claim_id, verdict]),
    );
    const records: ClaimRecord[] = round.claims.map((claim) => {
      const seen = span.get(claim.id);
      return {
        id: claim.id,
        statement: claim.statement,
        status: claim.status,
        confidence: claim.confidence,
        evidence: claim.evidence,
        origin: round.origins[claim.id] ?? null,
        verdict: verdictById.get(claim.id) ?? null,
        firstSeenRound: seen?.first ?? round.round,
        lastSeenRound: seen?.last ?? round.round,
        retiredInRound: null,
      };
    });
    const ids = new Set(records.map((record) => record.id));
    if (previousIds !== null && previousRecords !== null) {
      const dropped = previousRecords.filter((record) => !ids.has(record.id));
      if (dropped.length > 0) {
        for (const record of dropped) record.retiredInRound = round.round;
        retired.set(round.round, dropped);
      }
    }
    previousIds = ids;
    previousRecords = records;

    const roundSteps = steps.filter((step) => step.round === round.round);
    const tools: ToolProvenance[] = [];
    for (const step of roundSteps) {
      if (step.message !== null) tools.push(...toolProvenance(step.message));
    }

    const unresolvedClaims = records.filter(
      (record) => record.verdict === null || record.verdict.verdict !== "supported",
    );

    return {
      round: round.round,
      supported: round.supported,
      unresolved: round.unresolved,
      decision: round.decision,
      claims: records,
      unresolvedClaims,
      retiredClaims: retired.get(round.round) ?? [],
      tools,
      selected: round.round === selectedRound,
    };
  });
}

/** The unresolved minority of the selected round (empty when nothing selected). */
export function unresolvedAtSynthesis(evidence: RoundEvidence[]): ClaimRecord[] {
  const selected = evidence.find((round) => round.selected);
  return selected === undefined ? [] : selected.unresolvedClaims;
}

export function evidenceFor(
  evidence: RoundEvidence[],
  round: number | null,
): RoundEvidence | null {
  if (round === null) return null;
  return evidence.find((entry) => entry.round === round) ?? null;
}
