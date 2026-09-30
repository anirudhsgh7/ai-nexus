/**
 * Wire types (Phase 8 PRD §6.3).
 *
 * Backend field names are used verbatim (snake_case): no mapping layer, no
 * camelCase transform — zero mapping bugs and greppable names. Drift against
 * the backend is guarded by `backend/tests/test_ui_contract.py`, which is the
 * reason these types may be hand-written.
 */

export type AgentRole = "manager" | "researcher" | "ideator" | "skeptic";

export type StepKind =
  | "plan"
  | "research"
  | "ideate"
  | "critique"
  | "decide"
  | "revise"
  | "synthesize";

export type RunStatus = "running" | "completed" | "failed";

export type EventType =
  | "run_started"
  | "step_started"
  | "step_completed"
  | "run_completed"
  | "run_failed";

export type ConnectionState =
  | "connecting"
  | "live"
  | "reconnecting"
  | "closed"
  | "error";

export interface Evidence {
  source: string;
  quote: string | null;
}

export interface Claim {
  id: string;
  statement: string;
  status:
    | "fact"
    | "assumption"
    | "hypothesis"
    | "opinion"
    | "inference"
    | "unverified";
  confidence: number | null;
  evidence: Evidence[];
}

export interface Verdict {
  claim_id: string;
  verdict: "supported" | "refuted" | "unverifiable";
  objection: string;
  evidence: Evidence[];
}

export interface ManagerDecision {
  action: "call_agent" | "finish";
  target: AgentRole | null;
  instruction: string;
  reason: string;
  confidence: number;
}

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
  id: string | null;
}

export interface ToolResult {
  name: string;
  content: string;
  error: string | null;
  duration_ms: number | null;
}

export interface ErrorInfo {
  type: string;
  message: string;
  hint: string;
}

export interface AgentMessage {
  id: string;
  from_agent: AgentRole;
  to_agent: AgentRole | null;
  type: string;
  content: string;
  claims: Claim[] | null;
  verdicts: Verdict[] | null;
  decision: ManagerDecision | null;
  confidence: number | null;
  tool_calls: ToolCall[] | null;
  tool_results: ToolResult[] | null;
  round: number | null;
  created_at: string;
}

export interface RunEvent {
  seq: number;
  type: EventType;
  run_id: string;
  ts: string;
  step: number | null;
  kind: StepKind | null;
  agent: AgentRole | null;
  round: number | null;
  task: string | null;
  message: AgentMessage | null;
  duration_ms: number | null;
  skipped: boolean | null;
  error: ErrorInfo | null;
}

export interface RunSummary {
  run_id: string;
  status: RunStatus;
  task_preview: string;
  created_at: string;
  duration_ms: number | null;
}

export interface StepPayload {
  index: number;
  kind: StepKind;
  agent: AgentRole;
  status: string;
  round: number | null;
  duration_ms: number | null;
  skipped: boolean;
  message: AgentMessage | null;
  error: ErrorInfo | null;
}

export interface RoundSummary {
  round: number;
  supported: number;
  unresolved: number;
  decision: unknown;
}

export interface RunPayload {
  run_id: string;
  task: string;
  status: RunStatus;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  duration_ms: number | null;
  steps: StepPayload[];
  rounds: RoundSummary[];
  final_message: AgentMessage | null;
  error: ErrorInfo | null;
}

export interface HealthPayload {
  status: "ok" | "degraded" | "unavailable";
  app: { name: string; version: string };
  provider: {
    name: string;
    reachable: boolean;
    latency_ms: number | null;
    server_version: string | null;
    error?: string;
  };
  config: {
    primary_model: string;
    fallback_model: string;
    num_ctx: number;
    max_concurrent_generations: number;
  };
  checks: {
    primary_model_available: boolean;
    fallback_model_available: boolean;
  };
  took_ms: number;
  hint?: string;
}
