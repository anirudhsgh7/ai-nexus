import type { Evidence } from "../types";

/**
 * One cited source. Plain text only — model-authored source strings are never
 * linkified or rendered as HTML (Phase 9 PRD §10 security).
 */
export function SourceCitation({
  evidence,
  label,
}: {
  evidence: Evidence;
  label?: string;
}) {
  return (
    <div className="citation">
      {label !== undefined ? (
        <span className="citation-label">{label}</span>
      ) : null}
      <p className="citation-source">{evidence.source}</p>
      {evidence.quote !== null && evidence.quote !== "" ? (
        <blockquote className="citation-quote">{evidence.quote}</blockquote>
      ) : null}
    </div>
  );
}
