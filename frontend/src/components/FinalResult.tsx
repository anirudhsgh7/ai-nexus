import { formatElapsed, agentLabel } from "../format";
import type { ClaimRecord } from "../evidence";
import type { AgentMessage } from "../types";
import { VerdictBadge } from "./VerdictBadge";

const MINORITY_CAP = 10;

interface FinalResultProps {
  message: AgentMessage;
  startedAt: string | null;
  finishedAt: string | null;
  selectedRound: number | null;
  unresolved: ClaimRecord[];
}

export function FinalResult({
  message,
  startedAt,
  finishedAt,
  selectedRound,
  unresolved,
}: FinalResultProps) {
  return (
    <section className="final-card" aria-label="Final result">
      <h2>FINAL RESULT</h2>
      <p className="final-meta">
        total time {formatElapsed(startedAt, finishedAt, Date.now())}
      </p>
      <pre className="prose final-prose">{message.content}</pre>
      {selectedRound !== null ? (
        <div className="minority-report" aria-label="Unresolved at synthesis">
          <h3 className="minority-heading">
            FINAL EVIDENCE · selected round {selectedRound}
          </h3>
          {unresolved.length === 0 ? (
            <p className="muted">
              (none — all evaluated claims supported)
            </p>
          ) : (
            <>
              <ul className="minority-list">
                {unresolved.slice(0, MINORITY_CAP).map((record) => (
                  <li key={record.id}>
                    <span className="claim-id">[{record.id}]</span>{" "}
                    <VerdictBadge verdict={record.verdict?.verdict ?? null} />{" "}
                    <span className="minority-origin">
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
                {unresolved.length > MINORITY_CAP ? (
                  <li className="muted">
                    … and {unresolved.length - MINORITY_CAP} more
                  </li>
                ) : null}
              </ul>
            </>
          )}
        </div>
      ) : null}
    </section>
  );
}
