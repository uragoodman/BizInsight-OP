// Auto-generated from app/schemas/events.py — do not edit by hand.
// Run: npm run gen:events

export interface SseStartedEvent {
  sequence_number: number;
  event: 'started';
  trace_id: string;
}

export interface SseNodeStartedEvent {
  sequence_number: number;
  event: 'node_started';
  node: string;
}

export interface SseNodeCompletedEvent {
  sequence_number: number;
  event: 'node_completed';
  node: string;
  task_type: string | null;
  error: unknown | null;
}

export interface SsePlanEvent {
  sequence_number: number;
  event: 'plan';
  task_type: string | null;
  agents?: string[];
}

export interface SseInvestigationProgressEvent {
  sequence_number: number;
  event: 'investigation_progress';
  completed_actions: number;
  remaining_budget: number;
  status: 'in_progress' | 'sufficient' | 'stalled' | 'blocked' | 'abstained';
  current_action: string | null;
  stop_reason: string | null;
}

export interface SseHeartbeatEvent {
  sequence_number: number;
  event: 'heartbeat';
  trace_id: string;
}

export interface SseCompletedEvent {
  sequence_number: number;
  event: 'completed';
  task_type: string;
  answer: string;
  trace_id: string;
  error: boolean;
  error_code: string | null;
  error_detail: string | null;
  terminal_status: 'completed' | 'abstained' | 'failed' | 'cancelled' | 'interrupted' | null;
  clarification: Record<string, unknown> | null;
  presentation: unknown | null;
  complex_plan: Record<string, unknown> | null;
  durable_turn_id: string | null;
  execution_id: string | null;
  commit_version: number | null;
}

export interface SseSuspendedEvent {
  sequence_number: number;
  event: 'suspended';
  task_type: string;
  answer: string;
  trace_id: string;
  interaction_state: 'awaiting_human_input' | 'awaiting_metric_confirmation';
  interaction_reason_code: 'premise_direction_mismatch' | null;
  scope_fingerprint: string | null;
  clarification: Record<string, unknown> | null;
  complex_plan: Record<string, unknown> | null;
}

export interface SseFailedEvent {
  sequence_number: number;
  event: 'failed';
  detail: string;
}

export type SseEvent =
  | SseStartedEvent
  | SseNodeStartedEvent
  | SseNodeCompletedEvent
  | SsePlanEvent
  | SseInvestigationProgressEvent
  | SseHeartbeatEvent
  | SseCompletedEvent
  | SseSuspendedEvent
  | SseFailedEvent;

export const isSseStartedEvent = (e: SseEvent): e is SseStartedEvent => e.event === "started";
export const isSseNodeStartedEvent = (e: SseEvent): e is SseNodeStartedEvent => e.event === "node_started";
export const isSseNodeCompletedEvent = (e: SseEvent): e is SseNodeCompletedEvent => e.event === "node_completed";
export const isSsePlanEvent = (e: SseEvent): e is SsePlanEvent => e.event === "plan";
export const isSseInvestigationProgressEvent = (e: SseEvent): e is SseInvestigationProgressEvent => e.event === "investigation_progress";
export const isSseHeartbeatEvent = (e: SseEvent): e is SseHeartbeatEvent => e.event === "heartbeat";
export const isSseCompletedEvent = (e: SseEvent): e is SseCompletedEvent => e.event === "completed";
export const isSseSuspendedEvent = (e: SseEvent): e is SseSuspendedEvent => e.event === "suspended";
export const isSseFailedEvent = (e: SseEvent): e is SseFailedEvent => e.event === "failed";

export type ChatResult = SseCompletedEvent | (SseSuspendedEvent & { error: boolean; error_detail: string | null; presentation: unknown | null; complex_plan: Record<string, unknown> | null; durable_turn_id: string | null; execution_id: string | null; commit_version: number | null; clarification: Record<string, unknown> | null });
