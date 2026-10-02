import { agentLabel, pluralize } from "../format";
import type { RoundEvidence, ToolProvenance } from "../evidence";
import { ClaimCard } from "./ClaimCard";
import { DecisionRecord } from "./DecisionRecord";
import { VerdictBadge } from "./VerdictBadge";

const TOOL_LINE_CAP = 10;

function provenanceLine(tool: ToolProvenance): string {
  const parts = [
    tool.query !== null ? `${tool.tool}("${tool.query}")` : `${tool.tool}()`,
  ];
  if (tool.provider !== null) {
    parts.push(
      `source: ${tool.provider}${tool.cached === true ? " (cached)" : ""}`,
    );
  }
  if (tool.resultCount !== null) parts.push(pluralize(tool.resultCount, "result"));
  if (tool.ok && parts.length === 1) parts.push("ok");
  if (!tool.ok) parts.push(`error=${tool.error ?? "unknown"}`);
  return parts.join(" · ");
}

/**
 * The canonical evidence view for one round: counts, the Manager's decision,
 * every claim with status/origin/confidence/citations/verdict, claims the
 * revision retired, tool provenance, and the unresolved remainder
 * (Phase 9 PRD §6.6). Intermediate rounds render collapsed; the selected
 * round renders open (PRD open decision #2 default).
 */
export function RoundPanel({ evidence }: { evidence: RoundEvidence }) {
  return (
    <details
      className="round-panel"
      open={evidence.selected}
      aria-label={`Round ${evidence.round} evidence`}
    >
      <summary className="round-panel-head">
        <span className="round-panel-title">Round {evidence.round}</span>
        <span className="round-counts">
          {evidence.supported} supported · {evidence.unresolved} unresolved
        </span>
        {evidence.selected ? (
          <span className="chip selected-chip">SELECTED</span>
        ) : null}
      </summary>
      <div className="round-panel-body">
        {evidence.decision !== null ? (
          <DecisionRecord decision={evidence.decision} skipped={false} />
        ) : null}

        <h4 className="evidence-heading">
          Claims ({evidence.claims.length})
        </h4>
        <div className="claim-list">
          {evidence.claims.map((record) => (
            <ClaimCard key={record.id} record={record} />
          ))}
        </div>

        {evidence.retiredClaims.length > 0 ? (
          <>
            <h4 className="evidence-heading">
              Dropped by this round&apos;s revision
            </h4>
            <div className="retired-claims">
              {evidence.retiredClaims.map((record) => (
                <ClaimCard key={record.id} record={record} />
              ))}
            </div>
          </>
        ) : null}

        {evidence.tools.length > 0 ? (
          <>
            <h4 className="evidence-heading">External sources checked</h4>
            {evidence.tools.slice(0, TOOL_LINE_CAP).map((tool, index) => (
              <p className="tool-provenance" key={index}>
                {provenanceLine(tool)}
              </p>
            ))}
            {evidence.tools.length > TOOL_LINE_CAP ? (
              <p className="muted">
                … and {evidence.tools.length - TOOL_LINE_CAP} more
              </p>
            ) : null}
          </>
        ) : null}

        {evidence.unresolvedClaims.length > 0 ? (
          <>
            <h4 className="evidence-heading">
              Unresolved after round {evidence.round}
            </h4>
            <ul className="unresolved-list">
              {evidence.unresolvedClaims.map((record) => (
                <li key={record.id}>
                  <span className="claim-id">[{record.id}]</span>{" "}
                  <VerdictBadge verdict={record.verdict?.verdict ?? null} />{" "}
                  <span className="unresolved-origin">
                    {record.origin !== null
                      ? agentLabel(record.origin)
                      : "unknown"}{" "}
                    —{" "}
                    {record.verdict !== null
                      ? record.verdict.objection
                      : "never evaluated"}
                  </span>
                </li>
              ))}
            </ul>
          </>
        ) : null}
      </div>
    </details>
  );
}
