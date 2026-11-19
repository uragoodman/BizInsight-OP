export type ExplorerMetric = string;
export type ExplorerGranularity = "day" | "week" | "month" | "quarter" | "year";

export interface ExplorerQuery {
  seller_id: string | null;
  metric_name: ExplorerMetric;
  start_date: string;
  end_date: string;
  granularity: ExplorerGranularity;
}

export interface ExplorerPreview {
  context: ExplorerQuery & { locked: boolean; scope: string };
  summary: { metric_name: ExplorerMetric; value: number; unit: string; status: "ok" | "not_found" };
  series: Array<{ bucket: string; value: number }>;
  table: { columns: Array<{ key: string; label: string; kind: string; unit?: string }>; rows: Array<{ bucket: string; value: number }> };
  lineage: { source_table: string; query_time: string; latest_data_date: string | null };
  coverage?: { requested_start: string; requested_end: string; start_date: string; end_date: string; truncated: boolean; data_max_date: string };
}

export interface SellerOption { seller_id: string; label: string; order_count?: number; first_date?: string; last_date?: string }
export type SellerScopeType = "single" | "selected" | "all";
export interface SellerScopeResolveRequest {
  seller_scope_type: SellerScopeType;
  seller_ids?: string[];
  start_date: string;
  end_date: string;
  metric_ids?: string[];
  granularity?: ExplorerGranularity;
  include_unattributed?: boolean;
}
export interface SellerScopeIntentPayload {
  seller_scope_type: SellerScopeType;
  seller_ids: string[];
  start_date: string;
  end_date: string;
  metric_ids: string[];
  dimensions: ["seller"];
  granularity: ExplorerGranularity;
  include_unattributed: boolean;
}
export interface SignedSellerScopePayload {
  seller_scope_intent: SellerScopeIntentPayload;
  provenance: {
    source: "explorer_selector";
    token: string;
    expires_at: string;
  };
}
export interface SellerScopeResolveResponse { seller_scope_payload: SignedSellerScopePayload }
export interface MemoryItem { memory_id: string; seller_id: string; start_date: string; end_date: string; insight_text: string; limitations: string; evidence_ids: string[]; source_trace_id: string; status: "active" | "pending_review" | "superseded" | "stale" | "revoked" | "rejected"; created_at: string; scope_fingerprint?: string; review_reason?: string }
export interface MemoryConsolidationJob { job_id: string; status: "queued" | "generating" | "completed" | "failed" | "cancelled"; scope_fingerprint: string; proposed_memory_id?: string | null; error_code?: string | null; created_at?: string }
export interface MetricAttributionScope { status: "supported" | "unsupported" | "scope_unavailable"; methods: string[]; dimensions: string[]; required_data: Record<string, string[]>; factor_specs: Record<string, Array<{ factor_id: string; kind: string; metric_id?: string | null; numerator_metric?: string | null; denominator_metric?: string | null }>>; limitation: string }
export interface MetricDefinition { metric_name: string; display_name?: string; unit?: string; description?: string; available?: boolean; available_scopes?: string[]; report_available?: boolean; explorer_available?: boolean; report_available_scopes?: string[]; explorer_available_scopes?: string[]; surfaces?: string[]; attribution?: { support_by_scope: Record<"platform" | "seller", MetricAttributionScope>; limitation: string } }
export interface TurnClarification extends Record<string, unknown> { status?: string; reason_code?: string; reason?: string; missing_slots?: string[]; dimension_candidates?: string[]; ambiguous_candidates?: Array<{ metric_id: string; display_name: string }>; candidate_templates?: Array<{ template_id: string; display_name: string }>; candidates?: Array<Record<string, string>> }
export interface DebugTurn { id: string; threadId: string; clientTurnId?: string; message: string; answer?: string; error?: string; presentation?: AnalysisPresentation; taskType: string; traceId: string; createdAt: string; status: string; interactionState?: "awaiting_human_input" | "awaiting_metric_confirmation"; interactionReasonCode?: "premise_direction_mismatch"; clarification?: TurnClarification | null; scopeKey?: string; scopeFingerprint?: string }
export interface ComplexPlan { plan_id: string; template_name: string; display_name: string; purpose: string; scope: Record<string, string>; params: Record<string, string>; estimated_cost: string; limitations: string[]; status: string; expires_at: string; question: string; chat_session_id?: string; chat_turn_id?: string; parent_plan_id?: string | null; revision?: number; preferences?: { focus_key?: string; note?: string | null; source?: string }; focus_options?: Array<{ key: string; label: string }>; revision_chain?: Array<{ plan_id: string; revision: number; status: string; created_at: string }> }
export interface ActionPlan { action_plan_id: string; title: string; content: Record<string, unknown>; status: "draft" | "accepted" | "archived"; version: number; complex_plan_id: string; created_at: string; updated_at: string }
export type CodeAnalysisFileType = "csv" | "xlsx" | "parquet";
export type CodeAnalysisStatus = "awaiting_confirmation" | "queued" | "generating" | "validating" | "running" | "completed" | "failed" | "cancelled" | "expired" | "superseded";
export interface CodeAnalysisDataContract extends Record<string, unknown> {
  file_type?: CodeAnalysisFileType;
  rows?: number | null;
  columns?: Array<{ name: string; null_rate?: number; sensitive?: boolean }>;
  samples?: Array<Record<string, unknown>>;
  warnings?: string[];
}

export interface InventorySnapshotQuery {
  seller_id: string;
  start_date: string;
  end_date: string;
  as_of: string;
  quantity_basis: "on_hand" | "available";
  warehouse_id?: string;
  product_id?: string;
  limit?: number;
}
export interface InventorySnapshotBatchInput {
  source_system: string;
  source_batch_id: string;
  rows: Array<{
    seller_id: string;
    warehouse_id: string;
    product_id: string;
    sku?: string | null;
    snapshot_at: string;
    snapshot_semantics: "daily_close" | "point_in_time" | "latest_observed";
    on_hand_quantity?: number | string | null;
    reserved_quantity?: number | string | null;
    available_quantity?: number | string | null;
    quantity_unit: string;
  }>;
}
export interface InventorySnapshotBatchResult {
  status: "ok";
  seller_id: string;
  source_system: string;
  source_batch_id: string;
  received_rows: number;
  inserted_rows: number;
  duplicate_rows_ignored: number;
  snapshot_semantics: string[];
  quantity_units: string[];
}
export interface InventorySnapshotResult {
  status: "ok" | "not_found";
  seller_id: string;
  as_of: string;
  quantity_basis: "on_hand" | "available";
  observed_product_warehouse_count: number;
  truncated: boolean;
  items: Array<{
    warehouse_id: string;
    product_id: string;
    snapshot_at: string;
    snapshot_semantics: "daily_close" | "point_in_time" | "latest_observed";
    quantity: number | null;
    quantity_unit: string;
    source_system: string;
    measurement_status: "observed" | "missing";
  }>;
  coverage_status: "unknown_without_authoritative_product_warehouse_catalog";
  missing_policy: string;
  source_table: string;
}
export interface CausalExperimentScope { seller_id: string; start_date: string; end_date: string }
export interface EligibleCausalExperiment {
  experiment_id: string;
  outcome_metric_id: string;
  outcome_metric_name: string;
  assignment_unit: string;
  starts_at: string;
  ends_at: string;
  observation_count: number;
}
export interface EligibleCausalExperiments {
  items: EligibleCausalExperiment[];
  truncated: boolean;
  scope: CausalExperimentScope;
}
export interface CausalExperimentEstimate extends Record<string, unknown> {
  status: "ok" | "not_available" | string;
  causal_effect_estimated: boolean;
  effect?: number;
  confidence_interval?: [number, number] | number[];
  p_value?: number;
  treatment_n?: number;
  control_n?: number;
  metric_id?: string;
  causal_claim?: string;
  limitations?: string[];
}
export interface CodeAnalysisResourceLimits extends Record<string, unknown> {
  timeout_seconds?: number;
  memory_mb?: number;
  cpu_cores?: number;
  pids?: number;
  max_output_bytes?: number;
  max_result_bytes?: number;
  runner_image?: string;
}
export interface CodeAnalysisFile {
  file_id: string;
  filename: string;
  file_type: CodeAnalysisFileType;
  size_bytes: number;
  sha256: string;
  expires_at: string;
  purged_at?: string | null;
  data_contract: CodeAnalysisDataContract;
}
export interface CodeAnalysisPlan extends Record<string, unknown> {
  question: string;
  file_id: string;
  file_sha256: string;
  data_contract: CodeAnalysisDataContract;
  retention_hours: number;
  expected_output: string[];
  resource_limits?: CodeAnalysisResourceLimits;
}
export interface CodeAnalysisArtifactMetadata extends Record<string, unknown> { name?: string; content_type?: "image/png" | string; candidate_attempt?: number; failed?: boolean }
export interface CodeAnalysisArtifact { artifact_id: string; kind: "candidate_code" | "validation" | "runner_log" | "chart" | "result" | string; sha256: string; size_bytes: number; metadata?: CodeAnalysisArtifactMetadata; created_at: string }
export interface CodeAnalysisChart { artifact_id: string; name?: string | null; content_type?: "image/png" | string }
export interface CodeAnalysisResult extends Record<string, unknown> {
  conclusion: string;
  method: string | Record<string, unknown>;
  evidence: Array<Record<string, unknown>>;
  summary: Record<string, unknown>;
  charts: CodeAnalysisChart[];
  limitations: string[];
  recommendations: string[];
  reproducibility: Record<string, unknown>;
}
export interface CodeAnalysisAudit extends Record<string, unknown> {
  status: CodeAnalysisStatus;
  provider: string;
  generation_model?: string | null;
  provider_version?: string | null;
  candidate_attempts?: number;
  repair_attempts?: number;
  trace_id?: string | null;
  duration_ms: number;
  error_code?: string | null;
  error_phase?: string | null;
  error_http_status?: number | null;
  error_detail?: string | null;
  error_diagnostics?: Record<string, unknown> | null;
  provider_request_id?: string | null;
  created_at: string;
}
export interface CodeAnalysisJobSummary { job_id: string; question: string; status: CodeAnalysisStatus; file_id: string | null; error_code?: string | null; created_at: string; updated_at: string; completed_at?: string | null }
export interface CodeAnalysisJob extends CodeAnalysisJobSummary {
  plan_hash: string;
  plan_data: CodeAnalysisPlan;
  result_data?: CodeAnalysisResult | null;
  expires_at: string;
  chat_session_id?: string | null;
  confirmed_at?: string | null;
  started_at?: string | null;
  parent_job_id?: string | null;
  source_turn_id?: string | null;
  revision: number;
  resource_limits?: CodeAnalysisResourceLimits;
  artifacts?: CodeAnalysisArtifact[];
  latest_audit?: CodeAnalysisAudit | null;
}
export interface CodeAnalysisStatusEvent { event: "status"; sequence_number: number; job_id: string; status: CodeAnalysisStatus; error_code?: string | null }
export interface CodeAnalysisHeartbeatEvent { event: "heartbeat"; sequence_number: number; job_id: string; status: CodeAnalysisStatus }
export interface CodeAnalysisTerminalEvent { event: "completed" | "failed" | "cancelled" | "expired" | "superseded"; sequence_number: number; job?: CodeAnalysisJob; detail?: string }
export interface CodeAnalysisTimeoutEvent { event: "timeout"; sequence_number: number; job_id: string; detail: string }
export type CodeAnalysisEvent = CodeAnalysisStatusEvent | CodeAnalysisHeartbeatEvent | CodeAnalysisTerminalEvent | CodeAnalysisTimeoutEvent;

// === AnalysisPresentation（结构化结论卡）===
export interface AnalysisScope { seller_id?: string | null; start_date?: string; end_date?: string; granularity?: string }
export interface FactCard { metric_name: string; display_name?: string; value?: number | null; unit?: string; date_or_range?: string; evidence_ids?: string[] }
export interface ChangeCard { metric_name: string; display_name?: string; current_value?: number | null; baseline_value?: number | null; delta_value?: number | null; unit?: string | null; change_unit?: string | null; delta_ratio?: number | null; delta_ratio_unit?: string | null; current_period?: string; baseline_period?: string; evidence_ids?: string[] }
export interface ContributionEffect { component_type: string; value: number; unit: string; evidence_ids?: string[] }
export interface ContributionItem { dimension_value: string; delta_value: number; contribution_ratio?: number | null; current_value?: number | null; baseline_value?: number | null; components?: ContributionEffect[] }
export interface ContributionCard { dimension: string; metric_name?: string; metric_definition?: string; definition_version?: string; unit?: string; contribution_unit?: string | null; aggregation?: string; current_date?: string; baseline_date?: string; total_delta?: number | null; current_period?: Record<string, unknown> | null; baseline_period?: Record<string, unknown> | null; residual_value?: number | null; residual_type?: "coverage" | "rounding" | "model" | "unknown"; residual_explainable?: boolean; residual_tolerance?: number | null; residual_note?: string; analysis_nature?: "descriptive_contribution"; method?: string; items?: ContributionItem[]; evidence_ids?: string[] }
export interface RecommendationCard { text: string; evidence_ids?: string[] }
export interface FunnelStage { label: string; count: number; rate?: number | null }
export interface FunnelCard { date: string; stages: FunnelStage[]; evidence_ids?: string[] }
export interface JointClusterMetric { metric_name: string; display_name?: string; value?: number | null; unit?: string; comparison?: ChangeCard | null; evidence_ids?: string[] }
export interface JointAnalysisCluster { question?: string; seller_id?: string; period?: string; grain?: string; allocation_policy?: string; metrics?: JointClusterMetric[]; assessment?: string; sample_size?: number; sample_note?: string; association_limitation?: string; evidence_ids?: string[] }
export interface ReviewSample { order_id_abbr: string; score: number; comment_excerpt: string; has_title?: boolean; purchase_date?: string }
export interface ReviewSamplesCard { summary?: string; comment_rate?: number | null; avg_length?: number | null; samples?: ReviewSample[]; evidence_ids?: string[] }
export interface RiskDimension { name: string; value: number; weight: number; contribution: number }
export interface RiskAssessment { score: number; level: "low" | "medium" | "high"; dimensions?: RiskDimension[]; sample_size?: number; evidence_ids?: string[] }
export interface EvidenceRef { evidence_id: string; source_type: string; summary?: string }
export interface SemanticInterpretation { source_text?: string; metric_id?: string; display_name?: string; dimension?: string | null; dimension_label?: string; dimensions?: string[]; interpretation?: string; confidence?: number; reason?: string; requires_confirmation?: boolean }
export interface AnalysisPresentation { status: string; scope?: AnalysisScope | null; semantic_interpretation?: SemanticInterpretation | null; facts?: FactCard[]; changes?: ChangeCard[]; contributions?: ContributionCard[]; funnel?: FunnelCard[]; joint_clusters?: JointAnalysisCluster[]; review_samples?: ReviewSamplesCard | null; risk?: RiskAssessment | null; template_analyses?: AnalysisTemplateCard[]; limitations?: string[]; recommendations?: RecommendationCard[]; hypothesis_chains_by_metric?: Record<string, HypothesisStepCard[]>; evidence?: EvidenceRef[]; trace_id?: string; plan_preferences?: { focus_key?: string; note?: string | null; source?: string }; execution_governance?: { execution_id?: string; duration_ms?: number; input_fingerprint?: string; template_code_hash?: string } }
export interface AnalysisTemplateCard { template_name: string; display_name?: string; params?: Record<string, unknown>; result: Record<string, unknown>; evidence_ids?: string[] }
export interface HypothesisStepCard { step: number; hypothesis: string; purpose: string; evidence_ids: string[]; verdict: string; confidence?: number | null; rationale: string; next_action: string }
export interface DebugSession { id: string; title: string; createdAt: string; updatedAt: string; turns: DebugTurn[]; scope?: ExplorerQuery | null; scopeFingerprint?: string }
export interface ChatSessionPage { items: Array<{ id: string; title: string; scope?: ExplorerQuery | null; scope_fingerprint?: string | null; created_at: string; updated_at: string; turn_count: number }>; next_offset: number | null }
export interface PreferenceItem { registry_id: string; memory_key: string; memory_type: "presentation" | "analysis_method" | "scope_default"; value: Record<string, unknown> | null; scope: { seller_id?: string; type?: string }; status: "active" | "pending_create" | "pending_update" | "pending_delete" | "failed" | "superseded"; version: number; source_turn_id?: string | null; requires_confirmation: boolean; updated_at: string; expires_at?: string | null; overlay?: { status: string; error_code?: string | null; expires_at?: string | null } | null }
export interface PreferenceMutationResult { registry_id: string; operation_id?: string; status: string; changed: boolean; message?: string }
export interface PreferenceHistoryItem { audit_id: string; operation_id: string; event_type: string; status: string; reason_code?: string | null; created_at: string }
export interface ChatTurnPage { items: Array<{ id: string; client_turn_id?: string; message: string; answer: string | null; task_type: string; trace_id: string; status?: string; interaction_state?: DebugTurn["interactionState"] | "none"; interaction_reason_code?: DebugTurn["interactionReasonCode"]; thread_id?: string; scope_key?: string; scope_fingerprint?: string; clarification?: TurnClarification | null; presentation?: AnalysisPresentation | null; created_at: string }>; next_offset: number | null }

// SSE events — generated from backend Pydantic models
export type { SseEvent, ChatResult, SseFailedEvent, SseSuspendedEvent } from "./generated/sse-events";
