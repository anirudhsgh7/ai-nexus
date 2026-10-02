import { formatConfidence } from "../format";
import type { ManagerDecision } from "../types";

/**
 * One vocabulary for decision display: step cards (real + synthetic guard
 * decisions) and round panels (Phase 9 PRD §6.6).
 */
export function DecisionRecord({
  decision,
  skipped,
}: {
  decision: ManagerDecision;
  skipped: boolean;
}) {
  if (skipped) {
    // Synthetic guard finish: the reason IS the stop condition (round cap,
    // no progress, repeated decision, step cap). Confidence is 0.0 by
    // construction and adds nothing.
    return (
      <p className="decision-line">
        <span className="label">decision</span>{" "}
        <span className="chip guard-chip">guard</span> finish ·{" "}
        <q>{decision.reason}</q>
      </p>
    );
  }
  const target = decision.target !== null ? ` → ${decision.target}` : "";
  return (
    <p className="decision-line">
      <span className="label">decision</span>{" "}
      {decision.action}
      {target} · confidence {formatConfidence(decision.confidence)}
      {decision.instruction !== "" ? (
        <>
          {" "}
          · <q>{decision.instruction}</q>
        </>
      ) : null}{" "}
      — {decision.reason}
    </p>
  );
}
