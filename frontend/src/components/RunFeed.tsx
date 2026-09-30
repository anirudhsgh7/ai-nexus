import { useEffect, useState } from "react";

import { agentLabel, formatElapsed } from "../format";
import {
  currentAgent,
  stepGroups,
  type RunView,
  type StepEntry,
} from "../reducer";
import { StepCard } from "./StepCard";

function ConnectionBadge({ state }: { state: RunView["connection"] }) {
  if (state === "live") return <span className="conn conn-live">live</span>;
  if (state === "connecting")
    return <span className="conn conn-idle">connecting…</span>;
  if (state === "reconnecting")
    return <span className="conn conn-warn">reconnecting…</span>;
  return null;
}

function Group({ label, steps }: { label: string; steps: StepEntry[] }) {
  return (
    <section className="round-group" aria-label={label}>
      <h3 className="round-label">{label}</h3>
      {steps.map((step) => (
        <StepCard key={step.index} entry={step} />
      ))}
    </section>
  );
}

export function RunFeed({ view }: { view: RunView }) {
  const [now, setNow] = useState<number>(() => Date.now());

  useEffect(() => {
    if (view.status !== "running") return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [view.status]);

  const agent = currentAgent(view);

  return (
    <div className="run-view">
      <div className="run-header">
        <div className="run-title">
          <span className={`chip status-${view.status}`} role="status">
            {view.status}
          </span>
          <strong>{view.task ?? "…"}</strong>
        </div>
        <div className="run-meta">
          <span>
            current agent:{" "}
            <span className={agent !== null ? `agent-text agent-${agent}` : ""}>
              {agent !== null ? agentLabel(agent) : "—"}
            </span>
          </span>
          <span>
            elapsed:{" "}
            {formatElapsed(view.startedAt, view.finishedAt, now)}
          </span>
          <ConnectionBadge state={view.connection} />
        </div>
      </div>
      <div className="feed" aria-live="polite">
        {stepGroups(view).map((group) => (
          <Group key={group.label} label={group.label} steps={group.steps} />
        ))}
      </div>
    </div>
  );
}
