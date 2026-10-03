import { formatDuration, pluralize } from "../format";
import { stepSummary, type StepEntry } from "../reducer";
import type { AgentMessage, ToolCall, ToolResult } from "../types";
import { DecisionRecord } from "./DecisionRecord";
import { ErrorCard } from "./ErrorCard";

function toolLine(
  call: ToolCall,
  index: number,
  results: ToolResult[] | null,
): string {
  const result = results !== null ? (results[index] ?? null) : null;
  const args = JSON.stringify(call.arguments);
  const status =
    result === null
      ? "no result"
      : result.error === null
        ? "ok"
        : `error=${result.error}`;
  const duration =
    result !== null && result.duration_ms !== null
      ? ` ${result.duration_ms.toFixed(1)}ms`
      : "";
  return `${call.name}(${args}) → ${status}${duration}`;
}

function structureLine(message: AgentMessage): string | null {
  const fragments: string[] = [];
  if (message.claims !== null && message.claims.length > 0) {
    fragments.push(pluralize(message.claims.length, "claim"));
  }
  if (message.verdicts !== null && message.verdicts.length > 0) {
    fragments.push(pluralize(message.verdicts.length, "verdict"));
  }
  if (message.tool_calls !== null && message.tool_calls.length > 0) {
    fragments.push(pluralize(message.tool_calls.length, "tool call"));
  }
  if (message.retries > 0) {
    // pluralize() naively appends "s"; "retr" + "ies" is spelled by hand
    fragments.push(
      message.retries === 1 ? "1 retry" : `${message.retries} retries`,
    );
  }
  return fragments.length > 0 ? fragments.join(" · ") : null;
}

/**
 * One expandable step. Phase 9 adds the full decision record (instruction,
 * confidence, guard chip). Claim/verdict evidence stays in the round panels:
 * message-local claim ids must never be rendered next to run-level verdict
 * ids (Phase 9 PRD §6.5).
 */
export function StepCard({ entry }: { entry: StepEntry }) {
  const message = entry.message;
  const tools = message?.tool_calls ?? null;
  const results = message?.tool_results ?? null;
  const structure = message !== null ? structureLine(message) : null;
  const hasBody =
    entry.skipped ||
    entry.error !== null ||
    (message !== null &&
      (message.content.trim() !== "" ||
        message.decision !== null ||
        structure !== null ||
        (tools !== null && tools.length > 0)));

  return (
    <details className="step-card">
      <summary className="step-summary">
        <span
          className={`agent-dot agent-${entry.agent}`}
          aria-hidden="true"
        />
        <span className={`chip status-${entry.status}`}>{entry.status}</span>
        <span className="step-title">{stepSummary(entry)}</span>
      </summary>
      <div className="step-body">
        {entry.skipped ? <p className="muted">(step skipped)</p> : null}
        {entry.error !== null ? (
          <ErrorCard error={entry.error} title="STEP FAILED" />
        ) : null}
        {message !== null && message.content.trim() !== "" ? (
          <pre className="prose">{message.content}</pre>
        ) : null}
        {message?.decision != null ? (
          <DecisionRecord
            decision={message.decision}
            skipped={entry.skipped && entry.kind === "decide"}
          />
        ) : null}
        {structure !== null ? (
          <p className="structure-line">{structure}</p>
        ) : null}
        {tools?.map((call, index) => (
          <p className="tool-line" key={`${call.name}-${index}`}>
            {toolLine(call, index, results)}
          </p>
        ))}
        {!hasBody ? <p className="muted">(no content)</p> : null}
        {entry.durationMs !== null && hasBody ? (
          <p className="step-duration">took {formatDuration(entry.durationMs)}</p>
        ) : null}
      </div>
    </details>
  );
}
