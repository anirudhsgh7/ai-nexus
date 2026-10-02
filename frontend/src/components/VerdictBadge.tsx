import { VERDICT_TONES, verdictLabel } from "../format";
import type { Verdict } from "../types";

/** Verdict chip; a claim with no verdict reads "Not evaluated" (text, not color). */
export function VerdictBadge({
  verdict,
}: {
  verdict: Verdict["verdict"] | null;
}) {
  const tone = verdict === null ? "verdict-none" : VERDICT_TONES[verdict];
  return <span className={`chip ${tone}`}>{verdictLabel(verdict)}</span>;
}
