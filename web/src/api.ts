// ============================================================
// 前端 HTTP API 客户端
// 作用：封装所有与后端 FastAPI 的通信——统一携带身份/令牌请求头、统一异常处理，
//       以及 SSE 流式解析（聊天、复杂分析确认、Code Agent 任务进度）。
// 说明：普通接口走 request<T>() 返回 JSON；文件下载走 requestBlob()；流式接口
//       走 stream* 系列（内部解析 data: 块并逐个回调事件）。
// ============================================================
import type { ActionPlan, CausalExperimentEstimate, CausalExperimentScope, ChatSessionPage, ChatTurnPage, CodeAnalysisEvent, CodeAnalysisFile, CodeAnalysisJob, CodeAnalysisJobSummary, ComplexPlan, EligibleCausalExperiments, ExplorerPreview, ExplorerQuery, InventorySnapshotBatchInput, InventorySnapshotBatchResult, InventorySnapshotQuery, InventorySnapshotResult, MemoryConsolidationJob, MemoryItem, MetricDefinition, PreferenceHistoryItem, PreferenceItem, PreferenceMutationResult, SellerOption, SellerScopeResolveRequest, SellerScopeResolveResponse } from "./types";
import type { ChatResult, SseEvent, SseFailedEvent, SseSuspendedEvent } from "./types";

// API 请求身份：actor 为调试身份 ID；analysisToken 用于分析写接口；
// bearerToken 为登录后的会话令牌（存在时以 Authorization: Bearer 发送）。
export interface ApiAuth { actor: string; analysisToken: string; bearerToken: string }

// 统一的 API 错误类型：携带 HTTP 状态码，便于调用方区分 401/403/5xx。
export class ApiError extends Error { constructor(message: string, readonly status?: number) { super(message); } }

// 组装请求头：始终携带 X-Actor-ID；按需追加 Content-Type、X-Analysis-Token 与 Bearer 令牌。
export const AUTH_EXPIRED_EVENT = "bizinsight:auth-expired";

function notifyUnauthorized(auth: ApiAuth, status: number): void {
  if (status !== 401 || !auth.bearerToken || typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(AUTH_EXPIRED_EVENT));
}

function headers(auth: ApiAuth, json = false): HeadersInit {
  const values: Record<string, string> = { "X-Actor-ID": auth.actor };
  if (json) values["Content-Type"] = "application/json";
  if (auth.analysisToken) values["X-Analysis-Token"] = auth.analysisToken;
  if (auth.bearerToken) values.Authorization = `Bearer ${auth.bearerToken}`;
  return values;
}

// 通用 JSON 请求：发起 fetch，失败时解析 detail 抛 ApiError，成功返回泛型结果。
export async function request<T>(auth: ApiAuth, path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, { ...init, headers: { ...headers(auth, Boolean(init.body)), ...(init.headers ?? {}) } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) notifyUnauthorized(auth, response.status);
  if (!response.ok) throw new ApiError(String(payload.detail ?? `请求失败（${response.status}）`), response.status);
  return payload as T;
}

// 通用二进制请求（用于图表/报告文件下载），失败同样抛 ApiError。
export async function requestBlob(auth: ApiAuth, path: string): Promise<Blob> {
  const response = await fetch(path, { headers: headers(auth) });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    notifyUnauthorized(auth, response.status);
    throw new ApiError(String(payload.detail ?? `请求失败 (${response.status})`), response.status);
  }
  return response.blob();
}

// 全部业务接口集合（每个属性对应一个后端端点）。
export const api = {
  // 首次运行引导：用一次性 Bootstrap Token 创建平台管理员账号。
  bootstrap: async (bootstrapToken: string, email: string, password: string, displayName: string) => {
    const response = await fetch("/auth/bootstrap", { method: "POST", headers: { "Content-Type": "application/json", "X-Bootstrap-Token": bootstrapToken }, body: JSON.stringify({ email, password, display_name: displayName }) });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new ApiError(String(payload.detail ?? "初始化失败"), response.status);
    return payload as { bootstrapped: boolean };
  },
  login: async (email: string, password: string) => {
    const response = await fetch("/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }) });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new ApiError(String(payload.detail ?? "登录失败"), response.status);
    return payload as { access_token: string; expires_at: string; profile: { user_id: string; display_name: string; role: string; tenant_name: string; permissions: string[] } };
  },
  me: (auth: ApiAuth) => request<{ user_id: string; email: string; display_name: string; role: string; tenant_name: string; seller_grants: Array<{ seller_id: string; start_date?: string | null; end_date?: string | null }>; permissions: string[] }>(auth, "/auth/me"),
  usage: (auth: ApiAuth) => request<{ period_days: number; summary: { input_tokens: number; output_tokens: number; total_tokens: number; estimated_cost: number; calls: number }; items: Array<{ event_id: string; trace_id?: string | null; operation: string; model: string; input_tokens: number; output_tokens: number; total_tokens: number; estimated_cost: number; usage_source: string; created_at: string }> }>(auth, "/auth/usage?days=30&limit=20"),
  // 读取质量仪表盘数据（最新快照 + 历史快照列表）。
  quality: (auth: ApiAuth) => request<{ latest: any; history: any[] }>(auth, "/api/quality"),
  evaluations: (auth: ApiAuth) => request<{ items: any[] }>(auth, "/evaluations"),
  // 手动触发一次质量快照捕获（会持久化到后端 snapshots.json）。
  captureQuality: (auth: ApiAuth) => request<any>(auth, "/api/quality/snapshots", { method: "POST" }),
  logout: (auth: ApiAuth) => request<{ logged_out: boolean }>(auth, "/auth/logout", { method: "POST" }),
  members: (auth: ApiAuth) => request<{ items: Array<Record<string, unknown>> }>(auth, "/auth/members"),
  abacPolicies: (auth: ApiAuth) => request<{ items: Array<Record<string, unknown>> }>(auth, "/auth/abac/policies"),
  saveAbacPolicy: (auth: ApiAuth, id: string, payload: { rule: Record<string, unknown>; status?: "active" | "disabled" }) => request<Record<string, unknown>>(auth, `/auth/abac/policies/${encodeURIComponent(id)}`, { method: "PUT", body: JSON.stringify(payload) }),
  deleteAbacPolicy: (auth: ApiAuth, id: string) => request<{ deleted: boolean }>(auth, `/auth/abac/policies/${encodeURIComponent(id)}`, { method: "DELETE" }),
  health: (auth: ApiAuth) => request<{ status: string }>(auth, "/health"),
  sellers: (auth: ApiAuth, query: string) => request<{ items: SellerOption[] }>(auth, `/data-explorer/sellers?query=${encodeURIComponent(query)}`),
  resolveSellerScope: (auth: ApiAuth, payload: SellerScopeResolveRequest) => request<SellerScopeResolveResponse>(auth, "/data-explorer/seller-scope/resolve", { method: "POST", body: JSON.stringify(payload) }),
  preview: (auth: ApiAuth, query: ExplorerQuery) => request<ExplorerPreview>(auth, "/data-explorer/preview", { method: "POST", body: JSON.stringify(query) }),
  inventorySnapshot: (auth: ApiAuth, query: InventorySnapshotQuery) => request<InventorySnapshotResult>(auth, "/data-explorer/inventory-snapshots/query", { method: "POST", body: JSON.stringify(query) }),
  eligibleCausalExperiments: (auth: ApiAuth, scope: CausalExperimentScope) => request<EligibleCausalExperiments>(auth, "/causal-experiments/eligible", { method: "POST", body: JSON.stringify(scope) }),
  estimateCausalExperiment: (auth: ApiAuth, payload: CausalExperimentScope & { experiment_id: string }) => request<CausalExperimentEstimate>(auth, "/causal-experiments/estimate", { method: "POST", body: JSON.stringify(payload) }),
  ingestInventorySnapshotBatch: (auth: ApiAuth, batch: InventorySnapshotBatchInput) => request<InventorySnapshotBatchResult>(auth, "/data-explorer/inventory-snapshots/ingest", { method: "POST", body: JSON.stringify(batch) }),
  timeBounds: (auth: ApiAuth) => request<{ min_date: string; max_date: string; relative_anchor: string }>(auth, "/data-explorer/time-bounds"),
  resolveTimeRange: (auth: ApiAuth, payload: { kind: string; year?: number; value?: number }) => request<{ start_date: string; end_date: string; label: string; truncated: boolean; data_max_date: string }>(auth, "/data-explorer/time-range/resolve", { method: "POST", body: JSON.stringify(payload) }),
  metrics: (auth: ApiAuth, scope: "platform" | "seller" = "platform") => request<MetricDefinition[]>(auth, `/analysis/semantic/metrics?scope=${scope}`),
  trace: (auth: ApiAuth, id: string) => request<Record<string, unknown>>(auth, `/traces/${encodeURIComponent(id)}`),
  reports: (auth: ApiAuth) => request<Array<Record<string, unknown>>>(auth, "/analysis/reports"),
  reportCapabilities: (auth: ApiAuth) => request<{ allowed_scopes: Array<"platform" | "seller">; locked_seller: SellerOption | null }>(auth, "/analysis/reports/capabilities"),
  report: (auth: ApiAuth, id: string) => request<Record<string, unknown>>(auth, `/analysis/reports/${encodeURIComponent(id)}`),
  exportReport: (auth: ApiAuth, id: string, version?: number) => requestBlob(auth, `/analysis/reports/${encodeURIComponent(id)}/export${version ? `?version=${version}` : ""}`),
  createReport: (auth: ApiAuth, payload: Record<string, unknown>) => request<Record<string, unknown>>(auth, "/analysis/reports", { method: "POST", body: JSON.stringify(payload) }),
  reviewReport: (auth: ApiAuth, id: string, payload: Record<string, unknown>) => request(auth, `/analysis/reports/${encodeURIComponent(id)}/review`, { method: "POST", body: JSON.stringify(payload) }),
  publishReport: (auth: ApiAuth, id: string, payload: Record<string, unknown>) => request(auth, `/analysis/reports/${encodeURIComponent(id)}/publish`, { method: "POST", body: JSON.stringify(payload) }),
  // —— AI 报告洞察：生成（异步秒回 generating 占位）、列表、人工审核、SSE 进度 ——
  createReportInsights: (auth: ApiAuth, id: string, version: number, payload: Record<string, unknown>) => request<Record<string, unknown>>(auth, `/analysis/reports/${encodeURIComponent(id)}/versions/${version}/insights`, { method: "POST", body: JSON.stringify(payload) }),
  reportInsights: (auth: ApiAuth, id: string, version: number) => request<Array<Record<string, unknown>>>(auth, `/analysis/reports/${encodeURIComponent(id)}/versions/${version}/insights`),
  reviewReportInsight: (auth: ApiAuth, id: string, version: number, insightVersion: number, payload: Record<string, unknown>) => request<Record<string, unknown>>(auth, `/analysis/reports/${encodeURIComponent(id)}/versions/${version}/insights/${insightVersion}/review`, { method: "POST", body: JSON.stringify(payload) }),
  // 洞察生成进度 SSE：逐条回调 stage/status/completed/failed/timeout 事件，直到终态或手动中止。
  streamInsightProgress: async (auth: ApiAuth, id: string, version: number, insightVersion: number, onEvent: (event: Record<string, unknown>) => void, signal?: AbortSignal): Promise<void> => {
    const response = await fetch(`/analysis/reports/${encodeURIComponent(id)}/versions/${version}/insights/${insightVersion}/events`, { method: "GET", headers: headers(auth), signal });
    if (!response.ok || !response.body) notifyUnauthorized(auth, response.status);
    if (!response.ok || !response.body) throw new ApiError(`进度流连接失败（${response.status}）`, response.status);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
      let separator = buffer.indexOf("\n\n");
      while (separator >= 0) {
        const block = buffer.slice(0, separator);
        buffer = buffer.slice(separator + 2);
        const line = block.split("\n").find((item) => item.startsWith("data: "));
        if (line) {
          const event = JSON.parse(line.slice(6)) as Record<string, unknown>;
          onEvent(event);
          if (event.event === "completed" || event.event === "failed" || event.event === "timeout") return;
        }
        separator = buffer.indexOf("\n\n");
      }
      if (done) break;
    }
  },
  saveTarget: (auth: ApiAuth, metric: string, payload: Record<string, unknown>) => request(auth, `/analysis/targets/${encodeURIComponent(metric)}`, { method: "PUT", body: JSON.stringify(payload) }),
  targets: (auth: ApiAuth, query = "") => request<{ items: Array<Record<string, unknown>>; total: number; next_offset: number | null }>(auth, `/analysis/targets${query ? `?${query}` : ""}`),
  deleteTarget: (auth: ApiAuth, metric: string, query: string) => request<{ deleted: boolean }>(auth, `/analysis/targets/${encodeURIComponent(metric)}?${query}`, { method: "DELETE" }),
  knowledge: (auth: ApiAuth) => request<Record<string, unknown>>(auth, "/knowledge/documents"),
  indexKnowledge: (auth: ApiAuth, collection: string) => request<Record<string, unknown>>(auth, `/knowledge/index?collection=${encodeURIComponent(collection)}`, { method: "POST" }),
  testLlm: (auth: ApiAuth) => request<{ status: string; reply?: string; latency_ms?: number; detail?: string }>(auth, "/chat/test-llm", { method: "POST" })
  ,confirmSemanticMapping: (auth: ApiAuth, mappingId: string, payload: { chat_session_id?: string; explorer_context?: ExplorerQuery }) => request<{ status: string; mapping_id: string; execution_request_id?: string }>(auth, `/chat/semantic-mappings/${encodeURIComponent(mappingId)}/confirm`, { method: "POST", body: JSON.stringify(payload) })
  ,skipSemanticMapping: (auth: ApiAuth, mappingId: string, payload: { chat_session_id?: string; explorer_context?: ExplorerQuery }) => request<{ status: string; mapping_id: string; mention_id?: string }>(auth, `/chat/semantic-mappings/${encodeURIComponent(mappingId)}/skip`, { method: "POST", body: JSON.stringify(payload) })
  ,sessionMemory: (auth: ApiAuth, id: string) => request<Record<string, unknown>>(auth, `/memory/sessions/${encodeURIComponent(id)}`)
  ,clearSessionMemory: (auth: ApiAuth, id: string) => request<{ cleared: boolean }>(auth, `/memory/sessions/${encodeURIComponent(id)}`, { method: "DELETE" })
  ,memories: (auth: ApiAuth, status?: string) => request<{ items: MemoryItem[] }>(auth, `/memories${status ? `?status=${encodeURIComponent(status)}` : ""}`)
  ,preferences: (auth: ApiAuth, chatSessionId?: string) => request<{ items: PreferenceItem[]; personalization_enabled: boolean }>(auth, `/memory/preferences${chatSessionId ? `?chat_session_id=${encodeURIComponent(chatSessionId)}` : ""}`)
  ,setPersonalization: (auth: ApiAuth, enabled: boolean) => request<{ personalization_enabled: boolean }>(auth, "/memory/preferences/settings", { method: "PUT", body: JSON.stringify({ enabled }) })
  ,createPreference: (auth: ApiAuth, payload: { memory_key: string; memory_type: string; value: Record<string, unknown>; chat_session_id: string; idempotency_key: string; confirmed: boolean }) => request<PreferenceMutationResult>(auth, "/memory/preferences", { method: "POST", body: JSON.stringify(payload) })
  ,updatePreference: (auth: ApiAuth, id: string, payload: { value: Record<string, unknown>; chat_session_id: string; expected_version: number; idempotency_key: string; confirmed: boolean }) => request<PreferenceMutationResult>(auth, `/memory/preferences/${encodeURIComponent(id)}`, { method: "PUT", body: JSON.stringify(payload) })
  ,deletePreference: (auth: ApiAuth, id: string, payload: { expected_version: number; idempotency_key: string; confirmed: boolean }) => request<PreferenceMutationResult>(auth, `/memory/preferences/${encodeURIComponent(id)}`, { method: "DELETE", body: JSON.stringify(payload) })
  ,deleteAllPreferences: (auth: ApiAuth, idempotencyKey: string) => request<{ queued_count: number; deleted_count: number; pending_registry_ids: string[]; unfinished: Array<{ registry_id: string; reason: string }>; complete: boolean }>(auth, "/memory/preferences", { method: "DELETE", body: JSON.stringify({ confirmation_phrase: "DELETE_ALL_PREFERENCES", idempotency_key: idempotencyKey }) })
  ,retryPreference: (auth: ApiAuth, id: string) => request<{ registry_id: string; status: string }>(auth, `/memory/preferences/${encodeURIComponent(id)}/retry`, { method: "POST" })
  ,preferenceHistory: (auth: ApiAuth, id: string) => request<{ items: PreferenceHistoryItem[] }>(auth, `/memory/preferences/${encodeURIComponent(id)}/history`)
  ,revokeMemory: (auth: ApiAuth, id: string) => request<{ revoked: boolean }>(auth, `/memories/${encodeURIComponent(id)}/revoke`, { method: "POST" })
  ,memorySupersedeCandidates: (auth: ApiAuth, id: string) => request<{ items: MemoryItem[] }>(auth, `/memories/${encodeURIComponent(id)}/supersede-candidates`)
  ,reviewMemory: (auth: ApiAuth, id: string, payload: { action: "supersede" | "approve_independent" | "reject"; actor: string; reason?: string; old_memory_id?: string }) => request<MemoryItem>(auth, `/memories/${encodeURIComponent(id)}/review`, { method: "POST", body: JSON.stringify(payload) })
  ,createMemoryConsolidationJob: (auth: ApiAuth, payload: { scope_fingerprint: string; actor: string }) => request<{ job_id: string; status: string }>(auth, "/memories/consolidation-jobs", { method: "POST", body: JSON.stringify(payload) })
  ,memoryConsolidationJobs: (auth: ApiAuth) => request<{ items: MemoryConsolidationJob[] }>(auth, "/memories/consolidation-jobs")
  ,chatSessions: (auth: ApiAuth, offset = 0, limit = 20) => request<ChatSessionPage>(auth, `/chat/sessions?offset=${offset}&limit=${limit}`)
  ,createChatSession: (auth: ApiAuth, title = "偏好设置") => request<{ id: string; title: string; created_at: string; updated_at: string }>(auth, "/chat/sessions", { method: "POST", body: JSON.stringify({ title }) })
  ,updateChatSessionScope: (auth: ApiAuth, sessionId: string, scope: ExplorerQuery | null) => request<{ id: string; title: string; scope: ExplorerQuery | null; scope_fingerprint?: string | null; created_at: string; updated_at: string }>(auth, `/chat/sessions/${encodeURIComponent(sessionId)}/scope`, { method: "PUT", body: JSON.stringify({ scope }) })
  ,chatTurns: (auth: ApiAuth, sessionId: string, offset = 0, limit = 20) => request<ChatTurnPage>(auth, `/chat/sessions/${encodeURIComponent(sessionId)}/turns?offset=${offset}&limit=${limit}`)
  ,taskContext: (auth: ApiAuth, sessionId: string) => request<Record<string, any>>(auth, `/chat/sessions/${encodeURIComponent(sessionId)}/task-context`)
  ,createTask: (auth: ApiAuth, sessionId: string, payload: { objective: string; scope?: Record<string, unknown> | null }) => request<Record<string, any>>(auth, `/chat/sessions/${encodeURIComponent(sessionId)}/tasks`, { method: "POST", body: JSON.stringify(payload) })
  ,deleteChatSession: (auth: ApiAuth, sessionId: string) => request<{ deleted: boolean }>(auth, `/chat/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" })
  ,deleteChatTurn: (auth: ApiAuth, sessionId: string, turnId: string) => request<{ deleted: boolean }>(auth, `/chat/sessions/${encodeURIComponent(sessionId)}/turns/${encodeURIComponent(turnId)}`, { method: "DELETE" })
  ,createComplexPlan: (auth: ApiAuth, payload: Record<string, unknown>) => request<ComplexPlan>(auth, "/complex-analysis/plans", { method: "POST", body: JSON.stringify(payload) })
  ,reviseComplexPlan: (auth: ApiAuth, id: string, payload: Record<string, unknown>) => request<ComplexPlan>(auth, `/complex-analysis/plans/${encodeURIComponent(id)}/revise`, { method: "POST", body: JSON.stringify(payload) })
  ,cancelComplexPlan: (auth: ApiAuth, id: string) => request<{ cancelled: boolean }>(auth, `/complex-analysis/plans/${encodeURIComponent(id)}/cancel`, { method: "POST" })
  ,actionPlans: (auth: ApiAuth, offset = 0, limit = 20) => request<{ items: ActionPlan[]; next_offset: number | null }>(auth, `/complex-analysis/action-plans?offset=${offset}&limit=${limit}`)
  ,actionPlan: (auth: ApiAuth, id: string) => request<ActionPlan>(auth, `/complex-analysis/action-plans/${encodeURIComponent(id)}`)
  ,updateActionPlan: (auth: ApiAuth, id: string, payload: Record<string, unknown>) => request<ActionPlan>(auth, `/complex-analysis/action-plans/${encodeURIComponent(id)}`, { method: "PUT", body: JSON.stringify(payload) })
  ,deleteActionPlan: (auth: ApiAuth, id: string) => request<{ deleted: boolean }>(auth, `/complex-analysis/action-plans/${encodeURIComponent(id)}`, { method: "DELETE" })
  ,uploadCodeAnalysisFile: async (auth: ApiAuth, file: File): Promise<CodeAnalysisFile> => { const form = new FormData(); form.append("file", file); const response = await fetch("/code-analysis/files", { method: "POST", headers: headers(auth), body: form }); const payload = await response.json().catch(() => ({})); if (!response.ok) { notifyUnauthorized(auth, response.status); throw new ApiError(String(payload.detail ?? "上传文件失败"), response.status); } return payload as CodeAnalysisFile; }
  ,prefetchDataset: (auth: ApiAuth, payload: { seller_id: string; start_date: string; end_date: string; metric_names: string[] }) => request<Record<string, unknown>>(auth, "/code-analysis/datasets", { method: "POST", body: JSON.stringify(payload) })
  ,createCodeAnalysisJob: (auth: ApiAuth, payload: { file_id: string; question: string; chat_session_id?: string; source_turn_id?: string; parent_job_id?: string }) => request<CodeAnalysisJob>(auth, "/code-analysis/jobs", { method: "POST", body: JSON.stringify(payload) })
  ,codeAnalysisJobs: (auth: ApiAuth, offset = 0, limit = 20, chatSessionId?: string, createdAfter?: string, fileId?: string) => {
    const query = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (chatSessionId) query.set("chat_session_id", chatSessionId);
    if (createdAfter) query.set("created_after", createdAfter);
    if (fileId) query.set("file_id", fileId);
    return request<{ items: CodeAnalysisJobSummary[]; next_offset: number | null; total: number }>(auth, `/code-analysis/jobs?${query.toString()}`);
  }
  ,codeAnalysisJob: (auth: ApiAuth, id: string) => request<CodeAnalysisJob>(auth, `/code-analysis/jobs/${encodeURIComponent(id)}`)
  ,confirmCodeAnalysisJob: (auth: ApiAuth, id: string, plan_hash: string) => request<CodeAnalysisJob>(auth, `/code-analysis/jobs/${encodeURIComponent(id)}/confirm`, { method: "POST", body: JSON.stringify({ plan_hash }) })
  ,reviseCodeAnalysisJob: (auth: ApiAuth, id: string, question: string) => request<CodeAnalysisJob>(auth, `/code-analysis/jobs/${encodeURIComponent(id)}/revise`, { method: "POST", body: JSON.stringify({ question }) })
  ,cancelCodeAnalysisJob: (auth: ApiAuth, id: string) => request<{ cancelled: boolean }>(auth, `/code-analysis/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" })
  ,deleteCodeAnalysisJob: (auth: ApiAuth, id: string) => request<{ deleted: boolean }>(auth, `/code-analysis/jobs/${encodeURIComponent(id)}`, { method: "DELETE" })
  ,codeAnalysisArtifact: (auth: ApiAuth, jobId: string, artifactId: string) => requestBlob(auth, `/code-analysis/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeURIComponent(artifactId)}`)
};

// 聊天流式接口：POST /chat/stream，把每个 SSE 事件回调给调用方。
export async function streamChat(auth: ApiAuth, payload: Record<string, unknown>, onEvent: (event: SseEvent) => void, signal?: AbortSignal): Promise<ChatResult> {
  return streamEndpoint(auth, "/chat/stream", payload, onEvent, signal);
}
// 复杂分析确认流式接口：确认执行计划后，接收执行进度的 SSE 事件。
export async function streamComplexPlan(auth: ApiAuth, planId: string, onEvent: (event: SseEvent) => void, signal?: AbortSignal): Promise<ChatResult> {
  return streamEndpoint(auth, `/complex-analysis/plans/${encodeURIComponent(planId)}/confirm`, {}, onEvent, signal);
}
// SSE 流解析通用实现：按 "\n\n" 切分事件块，取 "data: " 行解析 JSON 并回调；
// completed / suspended 都是终态，failed 抛错；流结束仍未收到终态则抛错兜底。
async function streamEndpoint(auth: ApiAuth, path: string, payload: Record<string, unknown>, onEvent: (event: SseEvent) => void, signal?: AbortSignal): Promise<ChatResult> {
  const response = await fetch(path, { method: "POST", headers: headers(auth, true), body: JSON.stringify(payload), signal });
  if (!response.ok || !response.body) notifyUnauthorized(auth, response.status);
  if (!response.ok || !response.body) throw new ApiError(`请求失败（${response.status}）`, response.status);
  const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = ""; let complete: ChatResult | undefined;
  while (true) {
    const { value, done } = await reader.read(); buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    let separator = buffer.indexOf("\n\n");
    while (separator >= 0) {
      const block = buffer.slice(0, separator); buffer = buffer.slice(separator + 2);
      const line = block.split("\n").find((item) => item.startsWith("data: "));
      if (line) {
        const event = JSON.parse(line.slice(6)) as SseEvent;
        onEvent(event);
        if (event.event === "failed") throw new ApiError(String((event as SseFailedEvent).detail ?? "流式分析失败"));
        if (event.event === "completed") complete = event as unknown as ChatResult;
        if (event.event === "suspended") {
          const suspended = event as SseSuspendedEvent;
          complete = {
            ...suspended,
            error: false,
            error_detail: null,
            clarification: suspended.clarification ?? null,
            presentation: null,
            complex_plan: suspended.complex_plan ?? null,
            durable_turn_id: null,
            execution_id: null,
            commit_version: null,
          };
        }
      }
      separator = buffer.indexOf("\n\n");
    }
    if (done) break;
  }
  if (!complete) throw new ApiError("分析流未返回最终结果");
  return complete;
}

// Code Agent 任务进度流：GET 事件流直到任务结束，返回最终任务对象（支持 AbortSignal 取消）。
export async function streamCodeAnalysisJob(
  auth: ApiAuth,
  jobId: string,
  onEvent: (event: CodeAnalysisEvent) => void,
  signal?: AbortSignal,
): Promise<CodeAnalysisJob | null> {
  const response = await fetch(`/code-analysis/jobs/${encodeURIComponent(jobId)}/events`, {
    method: "GET",
    headers: headers(auth),
    signal,
  });
  if (!response.ok || !response.body) notifyUnauthorized(auth, response.status);
  if (!response.ok || !response.body) {
    const payload = await response.json().catch(() => ({}));
    throw new ApiError(String(payload.detail ?? `任务状态连接失败（${response.status}）`), response.status);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let terminal: CodeAnalysisJob | null = null;
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done }).replace(/\r\n/g, "\n");
    let separator = buffer.indexOf("\n\n");
    while (separator >= 0) {
      const block = buffer.slice(0, separator);
      buffer = buffer.slice(separator + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) {
        let event: CodeAnalysisEvent;
        try {
          event = JSON.parse(data) as CodeAnalysisEvent;
        } catch {
          throw new ApiError("任务状态流返回了无效数据");
        }
        onEvent(event);
        if ("job" in event && event.job) terminal = event.job;
        if (event.event === "failed" && !event.job) {
          throw new ApiError(event.detail ?? "分析任务失败");
        }
      }
      separator = buffer.indexOf("\n\n");
    }
    if (done) break;
  }
  return terminal;
}
