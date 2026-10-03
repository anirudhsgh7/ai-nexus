import { formatConfidence } from "../format";
import {
  VERIFICATION_STATUS_LABELS,
  VERIFICATION_STATUS_TONES,
} from "../format";
import { verificationCounts } from "../audits";
import type { VerificationReport } from "../types";
import { SourceCitation } from "./SourceCitation";

/**
 * The Verifier's per-claim audit of the final answer (Phase 11 PRD §6.11).
 *
 * Renders as `<details>` rows: status + confidence in the summary (text, not
 * color-only), evidence checked and both evidence directions inside. This is
 * the independent final-answer audit — prior verdicts are not shown here
 * because they are exactly what this panel re-checks.
 */
export function VerificationPanel({ report }: { report: VerificationReport }) {
  const counts = verificationCounts(report);
  return (
    <section
      className="verification-panel"
      aria-label="Verification of the final answer"
    >
      <h3 className="evidence-heading">
        VERIFICATION · {report.claims.length} claims · {counts.verified}{" "}
        verified · {counts.partially} partial · {counts.contradicted}{" "}
        contradicted · {counts.unverifiable} unverifiable
      </h3>
      {report.claims.length === 0 ? (
        <p className="muted">(no final claims to verify)</p>
      ) : (
        <div className="claim-list">
          {report.claims.map((entry) => (
            <details className="claim-card" key={entry.claim_id}>
              <summary className="claim-head">
                <span className="claim-id">[{entry.claim_id}]</span>
                <span
                  className={`chip ${VERIFICATION_STATUS_TONES[entry.verification_status]}`}
                >
                  {VERIFICATION_STATUS_LABELS[entry.verification_status]}
                </span>
                <span className="confidence">
                  {formatConfidence(entry.confidence)}
                </span>
              </summary>
              <div className="claim-body">
                <p className="claim-statement">{entry.explanation}</p>
                {entry.evidence_checked.length > 0 ? (
                  <p className="tool-provenance">
                    checked: {entry.evidence_checked.join(", ")}
                  </p>
                ) : null}
                {entry.supporting_evidence.map((item, index) => (
                  <SourceCitation
                    key={`s${index}`}
                    evidence={item}
                    label="supporting evidence"
                  />
                ))}
                {entry.contradicting_evidence.map((item, index) => (
                  <SourceCitation
                    key={`c${index}`}
                    evidence={item}
                    label="contradicting evidence"
                  />
                ))}
                {entry.source_references.length > 0 ? (
                  <p className="tool-provenance">
                    sources: {entry.source_references.join(", ")}
                  </p>
                ) : null}
              </div>
            </details>
          ))}
        </div>
      )}
    </section>
  );
}
