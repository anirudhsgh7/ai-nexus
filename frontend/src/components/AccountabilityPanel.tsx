import {
  ACCOUNTABILITY_STATUS_LABELS,
  ACCOUNTABILITY_STATUS_TONES,
  FLAG_SEVERITY_LABELS,
  FLAG_SEVERITY_TONES,
  agentLabel,
} from "../format";
import { flagsBySeverity } from "../audits";
import type { AccountabilityReport } from "../types";
import { VerdictBadge } from "./VerdictBadge";

/**
 * The Accountability agent's process/trace audit (Phase 11 PRD §6.11).
 *
 * A provenance table (code-canonical, enforced server-side), the flag list
 * grouped by severity, and the overall status. Never a quality judgment —
 * only trace, provenance, and process findings.
 */
export function AccountabilityPanel({
  report,
}: {
  report: AccountabilityReport;
}) {
  const grouped = flagsBySeverity(report);
  return (
    <section
      className="accountability-panel"
      aria-label="Accountability process audit"
    >
      <h3 className="evidence-heading">
        ACCOUNTABILITY ·{" "}
        <span
          className={`chip ${ACCOUNTABILITY_STATUS_TONES[report.overall_status]}`}
        >
          {ACCOUNTABILITY_STATUS_LABELS[report.overall_status]}
        </span>
      </h3>
      <p className="muted">
        trace {report.trace_completeness ? "complete" : "incomplete"} —{" "}
        {report.summary}
      </p>

      {report.final_claim_provenance.length > 0 ? (
        <>
          <h4 className="evidence-heading">Final claim provenance</h4>
          <ul className="unresolved-list">
            {report.final_claim_provenance.map((item) => (
              <li key={item.claim_id}>
                <span className="claim-id">[{item.claim_id}]</span>{" "}
                <span className="claim-origin">{agentLabel(item.origin)}</span>{" "}
                <VerdictBadge verdict={item.verdict} />{" "}
                <span className="muted">
                  {item.evidence_count} evidence
                </span>
              </li>
            ))}
          </ul>
        </>
      ) : null}

      <h4 className="evidence-heading">
        Flags ({report.flags.length})
      </h4>
      {report.flags.length === 0 ? (
        <p className="muted">(no flags raised)</p>
      ) : (
        <ul className="unresolved-list">
          {[
            ...grouped.violations,
            ...grouped.warnings,
            ...grouped.info,
          ].map((flag, index) => (
            <li key={`${flag.kind}-${index}`}>
              <span className={`chip ${FLAG_SEVERITY_TONES[flag.severity]}`}>
                {FLAG_SEVERITY_LABELS[flag.severity]}
              </span>{" "}
              <strong>{flag.kind.replace(/_/g, " ")}</strong>
              {flag.refs.length > 0 ? (
                <span className="claim-id"> [{flag.refs.join(", ")}]</span>
              ) : null}
              <div className="muted">{flag.explanation}</div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
