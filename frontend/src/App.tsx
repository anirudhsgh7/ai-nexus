import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, createRun, getHealth, getRun, listRuns } from "./api";
import { buildEvidence, unresolvedAtSynthesis } from "./evidence";
import {
  createRunView,
  foldEvent,
  setConnection,
  type RunView,
} from "./reducer";
import { openRunStream, type RunStreamHandle } from "./sse";
import { ErrorCard } from "./components/ErrorCard";
import { FinalResult } from "./components/FinalResult";
import { RunFeed } from "./components/RunFeed";
import { RunHistory } from "./components/RunHistory";
import { StatusBanner } from "./components/StatusBanner";
import { TaskForm } from "./components/TaskForm";
import type { HealthPayload, RoundSummary, RunSummary } from "./types";

/** Event kinds after which round snapshots change (bounded: ≤ 2 per round). */
function changesRounds(kind: string | null): boolean {
  return kind === "critique" || kind === "decide";
}

export default function App() {
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [history, setHistory] = useState<RunSummary[]>([]);
  const [view, setView] = useState<RunView | null>(null);
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [actionError, setActionError] = useState<ApiError | null>(null);
  // Round evidence arrives via GET (not SSE): snapshots only exist on disk
  // after a critique/decision, so Phase 9 refreshes the detail payload at
  // those points and on terminal events — no polling (Phase 9 PRD §6.7).
  const [rounds, setRounds] = useState<RoundSummary[]>([]);
  const [selectedRound, setSelectedRound] = useState<number | null>(null);
  const [roundRefreshError, setRoundRefreshError] = useState<string | null>(null);
  const roundsReqRef = useRef(0);
  const streamRef = useRef<RunStreamHandle | null>(null);

  const refreshRounds = useCallback(async (runId: string): Promise<void> => {
    const token = roundsReqRef.current + 1;
    roundsReqRef.current = token;
    try {
      const payload = await getRun(runId);
      if (token !== roundsReqRef.current) return; // superseded / run switched
      setRounds(payload.rounds);
      setSelectedRound(payload.selected_round);
      setRoundRefreshError(null);
    } catch (error) {
      // Keep whatever evidence we already have; the feed itself is unaffected.
      console.error("round refresh failed", error);
      setRoundRefreshError(
        "Evidence unavailable — reselect the run to retry"
      );
    }
  }, []);

  const refreshHistory = useCallback(async () => {
    try {
      setHistory(await listRuns(10));
    } catch (error) {
      console.error("history refresh failed", error);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((payload) => {
        if (cancelled) return;
        setHealth(payload);
        setHealthError(null);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setHealth(null);
        setHealthError(
          error instanceof ApiError
            ? `${error.message}${error.hint !== "" ? ` — ${error.hint}` : ""}`
            : "Backend unreachable",
        );
      });
    void refreshHistory();
    return () => {
      cancelled = true;
      streamRef.current?.close();
    };
  }, [refreshHistory]);

  const startRunView = useCallback(async (runId: string): Promise<void> => {
    const payload = await getRun(runId); // existence check before streaming
    roundsReqRef.current += 1; // invalidate any in-flight round refresh
    setRounds(payload.rounds);
    setSelectedRound(payload.selected_round);
    streamRef.current?.close();
    const base = createRunView(runId, payload.task);
    setView(setConnection(base, "connecting"));
    streamRef.current = openRunStream(runId, {
      onEvent: (event) => {
        setView((previous) =>
          previous !== null && previous.runId === runId
            ? foldEvent(previous, event)
            : previous,
        );
        if (
          (event.type === "step_completed" && changesRounds(event.kind)) ||
          event.type === "run_completed" ||
          event.type === "run_failed"
        ) {
          void refreshRounds(runId);
        }
      },
      onState: (state) => {
        setView((previous) =>
          previous !== null && previous.runId === runId
            ? setConnection(previous, state)
            : previous,
        );
      },
    });
  }, [refreshRounds]);

  const handleSubmit = useCallback(
    async (task: string): Promise<void> => {
      setFormError(null);
      setActionError(null);
      setNotice(null);
      setRoundRefreshError(null);
      setBusy(true);
      try {
        const created = await createRun(task);
        await startRunView(created.run_id);
        await refreshHistory();
      } catch (error) {
        if (
          error instanceof ApiError &&
          error.type === "RunActiveError" &&
          error.activeRunId !== undefined
        ) {
          // Adoption, not failure: watch the run that is already active.
          setNotice("A run is already active — watching it.");
          try {
            await startRunView(error.activeRunId);
          } catch (inner) {
            setActionError(
              inner instanceof ApiError
                ? inner
                : new ApiError({
                    status: 0,
                    type: "NetworkError",
                    message: "Cannot reach the backend",
                  }),
            );
          }
        } else if (
          error instanceof ApiError &&
          error.type === "ValidationError"
        ) {
          setFormError("task must be 1–4000 characters");
        } else if (error instanceof ApiError) {
          setActionError(error);
        } else {
          setActionError(
            new ApiError({
              status: 0,
              type: "UnknownError",
              message: String(error),
            }),
          );
        }
      } finally {
        setBusy(false);
      }
    },
    [startRunView, refreshHistory],
  );

  const handleSelect = useCallback(
    async (runId: string): Promise<void> => {
      setActionError(null);
      setNotice(null);
      setRoundRefreshError(null);
      try {
        await startRunView(runId);
      } catch (error) {
        if (error instanceof ApiError) setActionError(error);
      }
    },
    [startRunView],
  );

  const runDisabled =
    view?.status === "running" || health?.status === "unavailable";

  // Pure join: backend snapshots + step trace -> panel data (no I/O here).
  const evidence = useMemo(
    () => buildEvidence(rounds, view?.steps ?? [], selectedRound),
    [rounds, view?.steps, selectedRound],
  );
  const unresolved = useMemo(
    () => unresolvedAtSynthesis(evidence),
    [evidence],
  );

  return (
    <div className="app">
      <header className="app-header">
        <h1>AI Nexus</h1>
        <span className="subtitle">four agents, one answer</span>
      </header>

      <StatusBanner health={health} error={healthError} />
      <TaskForm
        disabled={runDisabled}
        busy={busy}
        error={formError}
        onSubmit={handleSubmit}
      />

      {notice !== null ? <p className="notice">{notice}</p> : null}
      {actionError !== null ? <ErrorCard error={actionError} /> : null}
      {roundRefreshError !== null ? (
        <p className="notice" style={{ color: "#d97706" }}>
          {roundRefreshError}
        </p>
      ) : null}

      {view !== null ? (
        <>
          <RunFeed view={view} evidence={evidence} />
          {view.status === "completed" && view.finalMessage !== null ? (
            <FinalResult
              message={view.finalMessage}
              startedAt={view.startedAt}
              finishedAt={view.finishedAt}
              selectedRound={selectedRound}
              unresolved={unresolved}
            />
          ) : null}
          {view.status === "failed" && view.error !== null ? (
            <ErrorCard error={view.error} title="RUN FAILED" />
          ) : null}
        </>
      ) : (
        <p className="muted empty-hint">Enter a task and press Run.</p>
      )}

      <RunHistory
        items={history}
        activeRunId={view?.runId ?? null}
        onSelect={handleSelect}
      />
    </div>
  );
}
