/**
 * Wire types (Phase 8 PRD §6.3, extended by Phase 9 PRD §6.2).
 *
 * Backend field names are used verbatim (snake_case): no mapping layer, no
 * camelCase transform — zero mapping bugs and greppable names. Drift against
 * the backend is guarded by `backend/tests/test_ui_contract.py`, which is the
 * reason these types may be hand-written.
 */

export type AgentRole =
  | "manager"
  | "researcher"
  | "ideator"
  | "skeptic"
  | "verifier"
  | "accountability";

export type StepKind =
  | "plan"
  | "research"
  | "ideate"
  | "critique"
  | "decide"
  | "revise"
  | "synthesize"
  | "verify"
  | "audit";

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

// ------------------------------------------------- Phase 11 audit reports

export type VerificationStatus =
  | "verified"
  | "contradicted"
  | "unverifiable"
  | "partially_verified";

export interface ClaimVerification {
  claim_id: string;
  verification_status: VerificationStatus;
  evidence_checked: string[];
  supporting_evidence: Evidence[];
  contradicting_evidence: Evidence[];
  source_references: string[];
  explanation: string;
  confidence: number;
}

export interface VerificationReport {
  claims: ClaimVerification[];
}

export type AccountabilityFlagKind =
  | "trace_incompleteness"
  | "unsupported_final_claim"
  | "unresolved_claim_suppressed"
  | "decision_inconsistency"
  | "evidence_provenance_gap"
  | "tool_use_inconsistency"
  | "peer_prose_exposure"
  | "premature_stop"
  | "confidence_evidence_mismatch"
  | "retry_activity";

export type FlagSeverity = "info" | "warning" | "violation";

export interface AccountabilityFlag {
  kind: AccountabilityFlagKind;
  severity: FlagSeverity;
  refs: string[];
  explanation: string;
}

export type AccountabilityStatus = "clean" | "warnings" | "violations";

export interface ClaimProvenance {
  claim_id: string;
  origin: AgentRole;
  verdict: Verdict["verdict"] | null;
  evidence_count: number;
}

export interface AccountabilityReport {
  trace_completeness: boolean;
  final_claim_provenance: ClaimProvenance[];
  flags: AccountabilityFlag[];
  overall_status: AccountabilityStatus;
  summary: string;
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
  verification: VerificationReport | null;
  accountability: AccountabilityReport | null;
  confidence: number | null;
  tool_calls: ToolCall[] | null;
  tool_results: ToolResult[] | null;
  retries: number;
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
  decision: ManagerDecision | null;
  claims: Claim[];
  origins: Record<string, AgentRole>;
  verdicts: Verdict[];
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
  selected_round: number | null;
  verification: AgentMessage | null;
  accountability: AgentMessage | null;
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
