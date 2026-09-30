import { formatDuration } from "../format";
import type { RunSummary } from "../types";

interface RunHistoryProps {
  items: RunSummary[];
  activeRunId: string | null;
  onSelect: (runId: string) => void;
}

export function RunHistory({ items, activeRunId, onSelect }: RunHistoryProps) {
  return (
    <section className="history" aria-label="Run history">
      <h2>HISTORY</h2>
      {items.length === 0 ? (
        <p className="muted">No runs yet</p>
      ) : (
        <ul className="history-list">
          {items.map((item) => (
            <li key={item.run_id}>
              <button
                type="button"
                className={
                  item.run_id === activeRunId
                    ? "history-item history-active"
                    : "history-item"
                }
                onClick={() => onSelect(item.run_id)}
              >
                <span className={`chip status-${item.status}`}>
                  {item.status}
                </span>
                <span className="history-task">{item.task_preview}</span>
                <span className="history-meta">
                  {formatDuration(item.duration_ms)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
