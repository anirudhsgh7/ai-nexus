import {
  CLAIM_STATUS_LABELS,
  CLAIM_STATUS_TONES,
  agentLabel,
  formatConfidence,
} from "../format";
import type { ClaimRecord } from "../evidence";
import { SourceCitation } from "./SourceCitation";
import { VerdictBadge } from "./VerdictBadge";

/**
 * One claim with its epistemic record: status, origin, confidence, cited
 * sources, and the Skeptic's verdict + objection (Phase 9 PRD §6.4).
 *
 * Always a `<details>`: evidence is keyboard-expandable, and the summary row
 * carries every signal as text so nothing depends on color or expansion.
 */
export function ClaimCard({ record }: { record: ClaimRecord }) {
  const verdict = record.verdict;
  return (
    <details className="claim-card">
      <summary className="claim-head">
        <span className="claim-id">[{record.id}]</span>
        <span className={`chip ${CLAIM_STATUS_TONES[record.status]}`}>
          {CLAIM_STATUS_LABELS[record.status]}
        </span>
        <span
          className={`claim-origin agent-text agent-${record.origin ?? "unknown"}`}
        >
          {record.origin !== null ? agentLabel(record.origin) : "unknown origin"}
        </span>
        <span
          className="confidence"
          title="claim-level confidence; messages are not aggregated by design"
        >
          {formatConfidence(record.confidence)}
        </span>
        <VerdictBadge verdict={verdict?.verdict ?? null} />
        {record.retiredInRound !== null ? (
          <span className="chip retired-chip">
            dropped in round {record.retiredInRound} (revision)
          </span>
        ) : null}
      </summary>
      <div className="claim-body">
        <p className="claim-statement">{record.statement}</p>
        {record.evidence.length > 0 ? (
          record.evidence.map((item, index) => (
            <SourceCitation key={index} evidence={item} />
          ))
        ) : (
          <p className="muted">no citations</p>
        )}
        {verdict !== null ? (
          <div className="claim-verdict">
            <VerdictBadge verdict={verdict.verdict} />
            <span className="verdict-objection">
              {verdict.verdict === "supported" ? "why:" : "why not:"}{" "}
              {verdict.objection}
            </span>
            {verdict.evidence.length > 0 ? (
              verdict.evidence.map((item, index) => (
                <SourceCitation
                  key={index}
                  evidence={item}
                  label={
                    verdict.verdict === "supported"
                      ? "supporting evidence"
                      : "refutation evidence"
                  }
                />
              ))
            ) : null}
          </div>
        ) : (
          <p className="muted">not evaluated</p>
        )}
      </div>
    </details>
  );
}
