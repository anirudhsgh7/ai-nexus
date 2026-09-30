import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, createRun, getHealth, getRun, listRuns } from "./api";
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
import type { HealthPayload, RunSummary } from "./types";

export default function App() {
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [history, setHistory] = useState<RunSummary[]>([]);
  const [view, setView] = useState<RunView | null>(null);
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [actionError, setActionError] = useState<ApiError | null>(null);
  const streamRef = useRef<RunStreamHandle | null>(null);

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
      },
      onState: (state) => {
        setView((previous) =>
          previous !== null && previous.runId === runId
            ? setConnection(previous, state)
            : previous,
        );
      },
    });
  }, []);

  const handleSubmit = useCallback(
    async (task: string): Promise<void> => {
      setFormError(null);
      setActionError(null);
      setNotice(null);
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

      {view !== null ? (
        <>
          <RunFeed view={view} />
          {view.status === "completed" && view.finalMessage !== null ? (
            <FinalResult
              message={view.finalMessage}
              startedAt={view.startedAt}
              finishedAt={view.finishedAt}
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
