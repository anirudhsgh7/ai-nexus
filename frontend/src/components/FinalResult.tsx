import { formatElapsed } from "../format";
import type { AgentMessage } from "../types";

interface FinalResultProps {
  message: AgentMessage;
  startedAt: string | null;
  finishedAt: string | null;
}

export function FinalResult({
  message,
  startedAt,
  finishedAt,
}: FinalResultProps) {
  return (
    <section className="final-card" aria-label="Final result">
      <h2>FINAL RESULT</h2>
      <p className="final-meta">
        total time {formatElapsed(startedAt, finishedAt, Date.now())}
      </p>
      <pre className="prose final-prose">{message.content}</pre>
    </section>
  );
}
