import { expect, test, type Page, type TestInfo } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

type ExpectedTurn = {
  event?: string;
  allowed_events?: string[];
  task_type?: string;
  metric?: string;
  trace_metric?: string;
  query_type?: string;
  dimension?: string;
  date_contains?: string;
  date_equals?: string;
  answer_keywords_any?: string[];
  answer_keywords_any_if_completed?: string[];
  answer_keywords_all?: string[];
  clarification_status?: string;
  clarification_reason_code?: string;
  terminal_status?: "completed" | "abstained" | "failed" | "cancelled" | "interrupted";
  require_fact_evidence?: boolean;
  require_evidence_if_completed?: boolean;
  require_comparison_card?: boolean;
  require_clarification_card?: boolean;
  require_clarification_input?: boolean;
  require_candidate_buttons?: boolean;
  clarification_text_contains?: string;
  no_business_query_tool?: boolean;
  no_state_errors?: boolean;
  require_query_success?: boolean;
  require_result_rows?: boolean;
  expected_result_status?: string;
  presentation_status?: string;
  forbid_contribution_cards?: boolean;
  require_contribution_cards?: boolean;
  forbid_numeric_fact_evidence?: boolean;
  require_fact_evidence_if_completed?: boolean;
  no_workflow_errors?: boolean;
  require_terminal_error?: boolean;
  required_trace_error_codes?: string[];
  value_matches_turn?: number;
  scope_seller_id_matches_session?: boolean;
  answer_matches_top_dimension_value_from_prior_query?: boolean;
  answer_dimension_values_at_most_from_prior_query?: number;
  result_limit_at_most?: number;
  require_error_presentation?: boolean;
  require_rag_citations?: boolean;
  expected_rag_source_contains?: string;
  trace_metrics_all_if_completed?: string[];
  require_clarification_card_if_suspended?: boolean;
};
type ExpandedTurn = { question: string; clarification_candidate?: string; expect?: ExpectedTurn };
type ComplexWorkflow = {
  template_name: string;
  template_display_name: string;
  start_date: string;
  end_date: string;
  select_template?: string;
  revise_note?: string;
  result_status?: "ok" | "insufficient_data";
};
type ComplexEvidence = {
  plan_ids: string[];
  trace_id: string;
  template_name: string;
  plan_status: string;
  execution_status: string;
  result_status: string;
  interactions: Array<{ kind: string; status: string }>;
};
type ReportWorkflow = {
  scope_type: "platform" | "seller";
  granularity: "week" | "month";
  start_date: string;
  end_date: string;
  metric_names: string[];
  include_charts: boolean;
  expected_create_status?: number;
  expected_error_contains?: string;
  expected_status?: string;
  expected_seller_id?: string;
  expected_null_metrics?: string[];
  expected_non_null_metrics?: string[];
  expected_risks?: string[];
  expected_chart_statuses?: string[];
  actions?: Array<"approve" | "edit_approve" | "reject" | "publish" | "export" | "guard_export" | "stale_publish_v1">;
  review_reason?: string;
  edit_marker?: string;
  export_contains?: string;
  unsupported_metric_probe?: string;
  unsupported_metric_error?: string;
};
type ReportEvidence = {
  status: "verified";
  create_status: number | null;
  create_error: string | null;
  report_id: string | null;
  report_type: string | null;
  period: { start_date: string; end_date: string } | null;
  scope_type: string | null;
  seller_id: string | null;
  metric_names: string[];
  metric_values: Record<string, number | null>;
  chart_statuses: string[];
  version_states: Array<{ version: number; status: string }>;
  final_status: string | null;
  export_status: number | null;
  export_policy: string | null;
  export_filename: string | null;
  export_contains_expected: boolean | null;
  draft_export_guard_status: number | null;
  stale_publish_status: number | null;
  unsupported_metric_ui_disabled: boolean | null;
  unsupported_metric_probe_status: number | null;
  unsupported_metric_probe_error: string | null;
};
type TargetSpec = {
  metric_name: string;
  target_value: number;
  version: string;
  period_start: string;
  period_end: string;
  effective_at: string;
};
type MetricTargetWorkflow = {
  mode: "definition_search" | "write_targets" | "filter_target" | "delete_target" | "unsupported_metric" | "invalid_period";
  scope_type: "platform" | "seller";
  search?: string;
  metric_name?: string;
  expected_unit?: string;
  expected_description_contains?: string;
  targets?: TargetSpec[];
  filter_version?: string;
  unauthorized_seller_id?: string;
  expected_error_contains?: string;
};
type MetricTargetEvidence = {
  status: "verified";
  mode: MetricTargetWorkflow["mode"];
  scope_type: MetricTargetWorkflow["scope_type"];
  metric_name: string | null;
  visible_metric_row_count: number | null;
  metric_row_text: string | null;
  unit: string | null;
  write_statuses: Array<{ metric_name: string; version: string; status: number }>;
  target_list_status: number | null;
  target_rows: Array<Record<string, unknown>>;
  filtered_row_count: number | null;
  deleted_version: string | null;
  delete_status: number | null;
  rejected_probes: Array<{ kind: string; status: number; error: string }>;
};
type KnowledgeWorkflow = {
  mode: "catalog" | "refresh_readback" | "include_text" | "index_control" | "index_denied";
  expected_metric?: string;
};
type KnowledgeEvidence = {
  status: "verified";
  mode: KnowledgeWorkflow["mode"];
  document_count: number;
  document_ids: string[];
  index_version: string;
  default_text_fields_absent: boolean;
  expected_metric_found: boolean | null;
  refresh_stable: boolean | null;
  include_text_document_count: number | null;
  include_text_total_chars: number | null;
  update_button_visible: boolean | null;
  index_probe_status: number | null;
  index_probe_error: string | null;
};
type PreferenceWorkflow = {
  mode: "default_settings_cancel" | "create_confirm" | "create_cancel" | "personalization_off" | "session_overlay" | "update_active" | "create_then_delete" | "history_read" | "cross_user_denied" | "delete_all_exact";
  memory_key?: string;
  memory_type?: "presentation" | "analysis_method" | "scope_default";
  foreign_registry_id?: string;
  seed_fixture?: boolean;
};
type PreferenceMutationEvidence = {
  registry_id: string;
  operation_id: string;
  status: string;
  session_id?: string | null;
  event_type?: string | null;
};
type PreferenceEvidence = {
  status: "verified";
  mode: PreferenceWorkflow["mode"];
  personalization_enabled: boolean;
  session_ids: string[];
  mutations: PreferenceMutationEvidence[];
  checked_registry_ids: string[];
  readback_statuses: Array<{ registry_id: string; status: string; overlay_status: string | null }>;
  cancel_left_no_write: boolean | null;
  denied_status: number | null;
  denied_reason: string | null;
  history_row_count: number | null;
  delete_all_queued_count: number | null;
  confirmation_phrase_rejected: boolean | null;
  bulk_deleted_registry_ids: string[];
  raw_preference_values_recorded: false;
};
type ActionPlanWorkflow = {
  mode: "read_evidence" | "pagination" | "edit_title" | "edit_content" | "invalid_json" | "accept" | "archive" | "delete_cancel" | "delete_confirm" | "cross_user_denied";
  action_plan_id?: string;
  foreign_action_plan_id?: string;
  expected_title?: string;
  seed_fixture?: boolean;
};
type ActionPlanEvidence = {
  status: "verified";
  mode: ActionPlanWorkflow["mode"];
  action_plan_id: string | null;
  foreign_action_plan_id?: string | null;
  complex_plan_id: string | null;
  trace_linked: boolean | null;
  evidence_linked: boolean | null;
  initial_version: number | null;
  final_version: number | null;
  final_status: string | null;
  mutation_status: number | null;
  delete_cancelled_without_write: boolean | null;
  deleted: boolean | null;
  cross_user_statuses: number[];
  visible_plan_count: number | null;
  pagination_page_counts: number[];
  no_execution_request: boolean;
  raw_plan_content_recorded: false;
};
type QualityWorkflow = {
  mode: "platform_read" | "platform_capture" | "platform_capture_twice" | "platform_bucket" | "platform_evaluations" | "owner_read" | "analyst_route_denied" | "analyst_api_denied" | "analyst_snapshot_denied" | "owner_capture";
};
type QualityEvidence = {
  status: "verified";
  mode: QualityWorkflow["mode"];
  route_path: string | null;
  quality_status: number | null;
  latest_scope: string | null;
  history_count_before: number | null;
  history_count_after: number | null;
  evaluation_status: number | null;
  evaluation_rows: number | null;
  evaluation_request_count: number;
  capture_statuses: number[];
  denied_statuses: number[];
  trace_bucket_count: number | null;
  selected_bucket: string | null;
  no_unexpected_writes: boolean;
};
type ExpandedCase = {
  case_id: string;
  feature: string;
  persona: "platform_admin" | "seller_owner" | "seller_analyst";
  entry: "chat" | "seller_scoped_chat" | "reports" | "metrics" | "knowledge" | "preferences" | "action_plans" | "quality";
  scope?: { start_date: string; end_date: string };
  turns: ExpandedTurn[];
  expected?: { same_session?: boolean; scope_must_not_widen?: boolean };
  // One expectation per request in first-seen order: a string requires that
  // durable status, an array accepts any listed durable status, and null means
  // the request must not create a durable turn.
  expected_terminal_statuses?: Array<
    | "completed" | "abstained" | "failed" | "cancelled" | "interrupted" | "suspended"
    | Array<"completed" | "abstained" | "failed" | "cancelled" | "interrupted" | "suspended">
    | null
  >;
  expected_non_durable_requests?: Array<{
    request_index: number;
    reason_code: "pre_scope_template_disambiguation";
    expected_clarification_status: "complex_template_required";
  }>;
  complex_workflow?: ComplexWorkflow;
  report_workflow?: ReportWorkflow;
  metric_target_workflow?: MetricTargetWorkflow;
  knowledge_workflow?: KnowledgeWorkflow;
  preference_workflow?: PreferenceWorkflow;
  action_plan_workflow?: ActionPlanWorkflow;
  quality_workflow?: QualityWorkflow;
  outcome_class?: "functional_success" | "capability_boundary" | "expected_no_data" | "safe_refusal" | "blocked_product";
  notes?: string;
};
type SseEvent = Record<string, any>;
type TurnSummary = {
  event: string;
  task_type: string;
  trace_id: string;
  metric_names: string[];
  date_labels: string[];
  evidence_count: number;
  fact_evidence_count: number;
  fact_values: Array<number | null>;
  clarification_status: string | null;
  clarification_reason_code: string | null;
  terminal_status: string | null;
  clarification_candidate_count: number | null;
  has_error: boolean;
  terminal_failure_code: string | null;
  answer: string;
  presentation: Record<string, any>;
};
type ChatRequest = {
  chat_session_id?: string;
  chat_turn_id?: string;
  seller_scope_payload?: any;
  explorer_context?: any;
  use_current_explorer_context?: boolean;
};

const here = path.dirname(fileURLToPath(import.meta.url));
const caseFile = path.join(here, "cases", "expanded-cases.jsonl");
const enabled = process.env.E2E_SUITE === "expanded";
const selectedIds = new Set((process.env.E2E_EXPANDED_CASE_IDS ?? "").split(",").filter(Boolean));
const allCases: ExpandedCase[] = enabled
  ? readFileSync(caseFile, "utf8").replace(/^\uFEFF/, "").split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line) as ExpandedCase)
  : [];
const cases = allCases.filter((item) => selectedIds.has(item.case_id));
if (enabled && (!selectedIds.size || cases.length !== selectedIds.size)) {
  throw new Error("Expanded suite case selection does not match the case JSONL");
}

async function installSseCapture(page: Page): Promise<void> {
  await page.addInitScript(() => {
    type Capture = { url: string; status: number; body: string; ready: boolean };
    const host = window as Window & { __expandedSse?: Capture[] };
    host.__expandedSse = [];
    const original = window.fetch.bind(window);
    window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const response = await original(input, init);
      const url = input instanceof Request ? input.url : String(input);
      const method = String(init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
      const pathname = new URL(url, window.location.href).pathname;
      const isWorkflowStream = pathname.endsWith("/chat/stream")
        || /\/complex-analysis\/plans\/[^/]+\/confirm$/.test(pathname);
      if (method === "POST" && isWorkflowStream) {
        const record: Capture = { url: pathname, status: response.status, body: "", ready: false };
        host.__expandedSse!.push(record);
        void response.clone().text().then((body) => { record.body = body; record.ready = true; })
          .catch(() => { record.ready = true; });
      }
      return response;
    };
  });
}

async function login(page: Page, persona: ExpandedCase["persona"]): Promise<void> {
  const key = persona === "platform_admin" ? "PLATFORM" : persona === "seller_owner" ? "SELLER" : "SELLER_ANALYST";
  const email = process.env[`E2E_${key}_EMAIL`] ?? "";
  const password = process.env[`E2E_${key}_PASSWORD`] ?? "";
  const displayName = process.env[`E2E_${key}_DISPLAY_NAME`] ?? "";
  if (!email || !password || !displayName) throw new Error(`Missing isolated ${persona} credentials`);
  await page.goto("/settings");
  await page.locator('input[autocomplete="email"]').fill(email);
  await page.locator('input[autocomplete="current-password"]').fill(password);
  await page.getByRole("button", { name: "登录", exact: true }).click();
  await expect(page.locator(".side-footer").getByText(displayName, { exact: true })).toBeVisible();
}

async function startFreshChat(page: Page): Promise<void> {
  await page.goto("/chat");
  const newChat = page.getByRole("button", { name: "+ 新对话", exact: true });
  await expect(newChat).toBeVisible();
  await newChat.click();
  await expect(page.locator(".conversation-head h1")).toHaveText("新的调试会话");
}

async function selectSellerScope(page: Page, scope: NonNullable<ExpandedCase["scope"]>): Promise<void> {
  await page.goto("/explorer");
  await expect(page.getByRole("heading", { name: "电商数据浏览" })).toBeVisible();
  await page.locator(".filter-grid .el-select").first().click();
  await expect(page.locator(".el-select-dropdown__item:visible").first()).toBeVisible();
  await page.locator(".el-select-dropdown__item:visible").first().click();
  const dates = page.locator(".filter-grid .el-date-editor input");
  await dates.nth(0).fill(scope.start_date);
  await dates.nth(0).press("Enter");
  await dates.nth(1).fill(scope.end_date);
  await dates.nth(1).press("Enter");
  await page.getByRole("button", { name: "查询数据", exact: true }).click();
  await expect(page.getByText("指标趋势", { exact: true })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("向当前范围的 Agent 提问", { exact: true })).toBeVisible();
}

async function waitForSse(page: Page, start: number): Promise<{ status: number; events: SseEvent[] }> {
  await page.waitForFunction((index) => {
    const host = window as Window & { __expandedSse?: Array<{ ready: boolean }> };
    return Boolean(host.__expandedSse?.slice(index).some((item) => item.ready));
  }, start, { timeout: 12 * 60_000 });
  return page.evaluate((index) => {
    const host = window as Window & { __expandedSse?: Array<{ status: number; body: string; ready: boolean }> };
    const item = host.__expandedSse?.slice(index).find((record) => record.ready);
    if (!item) throw new Error("No completed chat SSE capture was found");
    return { status: item.status, events: (item.body ?? "").split(/\r?\n\r?\n/).flatMap((block) => {
      const data = block.split(/\r?\n/).filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim()).join("\n");
      if (!data) return [];
      try { return [JSON.parse(data) as Record<string, any>]; } catch { return []; }
    }) };
  }, start);
}

async function sendChatTurn(page: Page, entry: ExpandedCase["entry"], question: string): Promise<{ status: number; events: SseEvent[] }> {
  const start = await page.evaluate(() => {
    const windowHost = window as Window & { __expandedSse?: unknown[] };
    return windowHost.__expandedSse?.length ?? 0;
  });
  const input = entry === "seller_scoped_chat"
    ? page.locator(".scoped-chat-panel textarea")
    : page.locator('textarea[placeholder="输入经营问题，Enter 发送，Shift + Enter 换行"]');
  const button = entry === "seller_scoped_chat"
    ? page.getByRole("button", { name: "开始分析", exact: true })
    : page.getByRole("button", { name: "发送", exact: true });
  await input.fill(question);
  await button.click();
  await expect(page.locator(".thinking")).toHaveCount(0, { timeout: 12 * 60_000 });
  return waitForSse(page, start);
}

async function currentClarificationCard(page: Page) {
  const activeClarification = page.locator(".message-stream .clarification:visible");
  if (await activeClarification.count()) return activeClarification.last();
  return page.locator(".message-stream .assistant-message").last().locator(".history-clarification:visible");
}

async function selectClarificationCandidate(page: Page, displayName: string): Promise<{ status: number; events: SseEvent[] }> {
  const start = await page.evaluate(() => {
    const windowHost = window as Window & { __expandedSse?: unknown[] };
    return windowHost.__expandedSse?.length ?? 0;
  });
  const clarificationCard = await currentClarificationCard(page);
  const candidate = clarificationCard.getByRole("button", { name: `确认使用：${displayName}`, exact: true });
  await expect(candidate).toHaveCount(1);
  await candidate.click();
  await expect(page.locator(".thinking")).toHaveCount(0, { timeout: 12 * 60_000 });
  return waitForSse(page, start);
}

async function selectComplexTemplate(page: Page, templateName: string): Promise<{ status: number; events: SseEvent[] }> {
  const start = await page.evaluate(() => (window as Window & { __expandedSse?: unknown[] }).__expandedSse?.length ?? 0);
  const responsePromise = page.waitForResponse((response) =>
    response.url().includes("/chat/stream") && response.request().method() === "POST",
  );
  const card = page.locator(".complex-plan").filter({ hasText: "选择复杂分析模板" });
  await card.getByRole("button", { name: templateName, exact: true }).click();
  const response = await responsePromise;
  await expect(page.locator(".thinking")).toHaveCount(0, { timeout: 12 * 60_000 });
  const captured = await waitForSse(page, start);
  expect(captured.status).toBe(response.status());
  return captured;
}

async function readComplexPlan(page: Page, planId: string): Promise<{ status: number; body: Record<string, any> }> {
  return page.evaluate(async (id) => {
    const stored = JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}") as { bearerToken?: string };
    if (!stored.bearerToken) throw new Error("Complex plan readback requires the authenticated session");
    const response = await fetch(`/complex-analysis/plans/${encodeURIComponent(id)}`, {
      headers: { Authorization: `Bearer ${stored.bearerToken}` },
    });
    return { status: response.status, body: await response.json() as Record<string, any> };
  }, planId);
}

function summarize(events: SseEvent[]): TurnSummary {
  const terminal = [...events].reverse().find((item) => ["completed", "suspended", "failed"].includes(item.event));
  expect(terminal, `Missing terminal SSE event: ${events.map((item) => item.event).join(",")}`).toBeTruthy();
  const presentation = terminal!.presentation && typeof terminal!.presentation === "object" ? terminal!.presentation : {};
  const facts = Array.isArray(presentation.facts) ? presentation.facts : [];
  const changes = Array.isArray(presentation.changes) ? presentation.changes : [];
  const collectEvidenceIds = (value: unknown, ids = new Set<string>()): Set<string> => {
    if (Array.isArray(value)) {
      for (const item of value) collectEvidenceIds(item, ids);
    } else if (value && typeof value === "object") {
      for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
        if (key === "evidence_id" && typeof item === "string" && item) ids.add(item);
        if (key === "evidence_ids" && Array.isArray(item)) {
          for (const evidenceId of item) if (typeof evidenceId === "string" && evidenceId) ids.add(evidenceId);
        }
        collectEvidenceIds(item, ids);
      }
    }
    return ids;
  };
  const semanticInterpretation = presentation.semantic_interpretation && typeof presentation.semantic_interpretation === "object"
    ? presentation.semantic_interpretation
    : {};
  const answer = String(terminal!.answer ?? "");
  const factMetricNames = facts.map((item: Record<string, unknown>) => String(item.metric_name ?? "")).filter(Boolean);
  // Comparison results are carried by the validated change cards, not by
  // `facts`; the trace still proves the metric and periods executed.
  const changeMetricNames = changes.map((item: Record<string, unknown>) => String(item.metric_name ?? "")).filter(Boolean);
  const contributionMetricNames = (Array.isArray(presentation.contributions) ? presentation.contributions : [])
    .map((item: Record<string, unknown>) => String(item.metric_name ?? ""))
    .filter(Boolean);
  const semanticMetricName = typeof semanticInterpretation.metric_id === "string" ? semanticInterpretation.metric_id : "";
  const safeFailureCodes = new Set([
    "chat_turn_commit_not_persisted", "turn_commit_not_persisted", "durable_turn_missing",
    "stream_terminated_without_terminal", "token_budget_exceeded", "critic_revision_limit_exceeded",
    "INTERNAL_ERROR",
  ]);
  const terminalDetail = String(terminal!.detail ?? terminal!.error_detail?.code ?? "");
  const failureCode = safeFailureCodes.has(terminalDetail)
    ? terminalDetail
    : (/^([A-Za-z][A-Za-z0-9]*(?:Error|Exception))\b/.exec(terminalDetail)?.[1]
      ?? (terminal!.event === "failed" ? "unclassified_workflow_failure" : null));
  return {
    event: String(terminal!.event),
    task_type: String(terminal!.task_type ?? ""),
    trace_id: String(terminal!.trace_id ?? ""),
    metric_names: [...new Set([...factMetricNames, ...changeMetricNames, ...contributionMetricNames, semanticMetricName].filter(Boolean))],
    date_labels: facts.map((item: Record<string, unknown>) => String(item.date_or_range ?? "")),
    evidence_count: collectEvidenceIds(presentation).size,
    fact_evidence_count: facts.reduce((count: number, item: Record<string, unknown>) => count + (Array.isArray(item.evidence_ids) ? item.evidence_ids.length : 0), 0),
    fact_values: facts.map((item: Record<string, unknown>) => typeof item.value === "number" ? item.value : null),
    clarification_status: terminal!.clarification?.status ?? null,
    clarification_reason_code: terminal!.clarification?.reason_code ?? null,
    terminal_status: terminal!.terminal_status ?? null,
    clarification_candidate_count: Array.isArray(terminal!.clarification?.ambiguous_candidates)
      ? terminal!.clarification.ambiguous_candidates.length
      : null,
    has_error: Boolean(terminal!.error) || terminal!.event === "failed",
    terminal_failure_code: failureCode,
    answer,
    presentation,
  };
}

async function readTrace(page: Page, traceId: string): Promise<Record<string, any>> {
  return page.evaluate(async (id) => {
    const stored = JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}") as { bearerToken?: string };
    if (!stored.bearerToken) throw new Error("Authenticated trace inspection requires the signed-in bearer token");
    const response = await fetch(`/traces/${encodeURIComponent(id)}`, {
      headers: { Authorization: `Bearer ${stored.bearerToken}` },
    });
    if (!response.ok) throw new Error(`Trace lookup failed with HTTP ${response.status}`);
    return await response.json() as Record<string, any>;
  }, traceId);
}

function valuesForKey(value: unknown, keyName: string, output: unknown[] = []): unknown[] {
  if (!value || typeof value !== "object") return output;
  if (Array.isArray(value)) {
    for (const item of value) valuesForKey(item, keyName, output);
    return output;
  }
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    if (key === keyName) output.push(child);
    valuesForKey(child, keyName, output);
  }
  return output;
}

function resultPayloads(queryResults: unknown[]): Array<Record<string, any>> {
  const nestedKeys = ["metric_result", "trend_result", "compare_result", "breakdown_result", "contribution_result", "statistics_result", "association_result"];
  return queryResults.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const outer = (item as Record<string, any>).result;
    if (!outer || typeof outer !== "object") return [];
    const nested = nestedKeys.map((key) => outer[key]).filter((value) => value && typeof value === "object");
    return nested.length ? nested : [outer];
  });
}

async function assertTurn(page: Page, summary: TurnSummary, expected: ExpectedTurn | undefined, prior: TurnSummary[]): Promise<void> {
  expect(summary.event).not.toBe("failed");
  expect(summary.trace_id.length, "each turn must expose a trace id").toBeGreaterThan(0);
  if (summary.event === "completed" && summary.has_error) {
    expect(expected?.require_terminal_error, "completed SSE with error=true must be explicitly classified as an error outcome").toBeTruthy();
  } else if (summary.event === "completed") {
    expect(summary.answer.trim().length, "successful completed turns must expose a non-empty answer").toBeGreaterThan(0);
  } else {
    expect(summary.event).toBe("suspended");
  }
  if (!expected) return;
  let traceCache: Record<string, any> | undefined;
  const traceForTurn = async (): Promise<Record<string, any>> => {
    if (!traceCache) traceCache = await readTrace(page, summary.trace_id);
    return traceCache;
  };
  if (expected.require_rag_citations || expected.expected_rag_source_contains) {
    const trace = await traceForTurn();
    const snapshot = trace.state_snapshot ?? {};
    const citations = Array.isArray(snapshot.rag_citations) ? snapshot.rag_citations : [];
    expect(citations.length, "metric-definition workflow must persist RAG citations in its trace").toBeGreaterThan(0);
    const fusedHitIds = snapshot.rag_query_trace?.fused_hit_ids;
    expect(Array.isArray(fusedHitIds) && fusedHitIds.length > 0, "RAG trace must show retrieved Qdrant hit ids").toBeTruthy();
    if (expected.expected_rag_source_contains) {
      const sourcePaths = citations.map((item: Record<string, unknown>) => String(item.source_path ?? "").toLowerCase());
      expect(sourcePaths.some((source: string) => source.includes(expected.expected_rag_source_contains!.toLowerCase())),
        `RAG citations must include ${expected.expected_rag_source_contains}`).toBeTruthy();
    }
  }
  if (expected.event) expect(summary.event).toBe(expected.event);
  if (expected.allowed_events) expect(expected.allowed_events).toContain(summary.event);
  if (expected.presentation_status) expect(summary.presentation.status).toBe(expected.presentation_status);
  if (expected.forbid_contribution_cards) {
    expect(summary.presentation.contributions ?? [], "unsupported contribution must not expose numeric cards").toEqual([]);
  }
  if (expected.require_contribution_cards) {
    const cards = Array.isArray(summary.presentation.contributions) ? summary.presentation.contributions : [];
    expect(cards.length, "registered period contribution must expose a validated contribution card").toBeGreaterThan(0);
    expect(
      cards.some((card: Record<string, unknown>) => Array.isArray(card.items) && card.items.length > 0),
      "contribution card must contain non-empty dimension rows",
    ).toBeTruthy();
  }
  if (expected.task_type) expect(summary.task_type).toBe(expected.task_type);
  if (expected.metric) expect(summary.metric_names).toContain(expected.metric);
  if (expected.trace_metric) {
    const trace = await traceForTurn();
    const snapshot = trace.state_snapshot ?? {};
    const queryMetrics = (Array.isArray(snapshot.query_results) ? snapshot.query_results : [])
      .map((item: Record<string, any>) => String(item?.result?.parsed?.metric_name ?? ""))
      .filter(Boolean);
    const authoritativeMetric = String(snapshot.metric_authoritative_definition?.metric_id ?? "");
    expect([...queryMetrics, String(snapshot.premise_check?.metric_name ?? ""), authoritativeMetric]).toContain(expected.trace_metric);
  }
  if (expected.trace_metrics_all_if_completed && summary.event === "completed") {
    const trace = await traceForTurn();
    const snapshot = trace.state_snapshot ?? {};
    const queryMetrics = (Array.isArray(snapshot.query_results) ? snapshot.query_results : [])
      .map((item: Record<string, any>) => String(item?.result?.parsed?.metric_name ?? ""))
      .filter(Boolean);
    for (const metric of expected.trace_metrics_all_if_completed) {
      expect(queryMetrics, `completed multi-metric answer must execute ${metric}`).toContain(metric);
    }
  }
  if (expected.query_type || expected.dimension || expected.require_query_success || expected.require_result_rows
    || expected.expected_result_status || expected.no_business_query_tool || expected.result_limit_at_most !== undefined
    || expected.required_trace_error_codes?.length) {
    const trace = await traceForTurn();
    const snapshot = trace.state_snapshot ?? {};
    const queryResults = Array.isArray(snapshot.query_results) ? snapshot.query_results : [];
    const payloads = resultPayloads(queryResults);
    if (expected.query_type) {
      expect(valuesForKey(snapshot, "query_type"), "trace must show the expected registered query operation").toContain(expected.query_type);
    }
    if (expected.dimension) {
      const dimensions = [...valuesForKey(snapshot, "dimension"), ...valuesForKey(snapshot, "dimension_id")];
      expect(dimensions, "trace must show the requested grouping dimension").toContain(expected.dimension);
    }
    if (expected.require_query_success || expected.expected_result_status) {
      expect(payloads.length, "trace must contain a database query result").toBeGreaterThan(0);
      const statuses = payloads.map((item) => String(item?.status ?? "missing"));
      const requiredStatus = expected.expected_result_status ?? "ok";
      for (const status of statuses) expect(status).toBe(requiredStatus);
    }
    if (expected.require_result_rows || expected.result_limit_at_most !== undefined) {
      const itemArrays = payloads.flatMap((item) => Array.isArray(item.items) ? [item.items] : []);
      const pointArrays = payloads.flatMap((item) => Array.isArray(item.points) ? [item.points] : []);
      expect([...itemArrays, ...pointArrays].some((rows: unknown[]) => rows.length > 0), "trace must contain non-empty query rows").toBeTruthy();
      if (expected.result_limit_at_most !== undefined) {
        expect(itemArrays.some((rows: unknown[]) => rows.length <= expected.result_limit_at_most!), "result rows must honor the requested Top-N limit").toBeTruthy();
      }
    }
  }
  if (expected.date_contains) {
    expect(summary.date_labels.some((label) => label.includes(expected.date_contains!)) || summary.answer.includes(expected.date_contains)).toBeTruthy();
  }
  if (expected.date_equals) {
    const trace = await traceForTurn();
    const exactMetricDates = valuesForKey(trace.state_snapshot, "metric_date");
    expect(exactMetricDates).toContain(expected.date_equals);
  }
  if (expected.answer_keywords_any) {
    expect(expected.answer_keywords_any.some((keyword) => summary.answer.includes(keyword))).toBeTruthy();
  }
  if (expected.answer_keywords_any_if_completed && summary.event === "completed") {
    expect(expected.answer_keywords_any_if_completed.some((keyword) => summary.answer.includes(keyword))).toBeTruthy();
  }
  if (expected.answer_keywords_all) {
    for (const keyword of expected.answer_keywords_all) expect(summary.answer).toContain(keyword);
  }
  if (expected.clarification_status) expect(summary.clarification_status).toBe(expected.clarification_status);
  if (expected.clarification_reason_code) expect(summary.clarification_reason_code).toBe(expected.clarification_reason_code);
  if (expected.terminal_status) expect(summary.terminal_status).toBe(expected.terminal_status);
  if (expected.no_workflow_errors) expect(summary.has_error).toBeFalsy();
  if (expected.require_terminal_error) expect(summary.has_error).toBeTruthy();
  if (expected.required_trace_error_codes?.length) {
    const trace = await traceForTurn();
    const errors = Array.isArray(trace.state_snapshot?.errors) ? trace.state_snapshot.errors : [];
    const codes = errors.map((item: Record<string, unknown>) => String(item.code ?? ""));
    for (const code of expected.required_trace_error_codes) {
      expect(codes, `trace must preserve error code ${code}`).toContain(code);
    }
  }
  if (expected.require_error_presentation) {
    expect(summary.has_error, "the browser terminal payload must expose the unsuccessful outcome").toBeTruthy();
    expect(summary.event).toBe("completed");
    await expect(page.locator(".message-stream .assistant-message").last().locator(".turn-error")).toBeVisible();
  }
  if (expected.require_clarification_card) {
    expect(summary.event).toBe("suspended");
    const clarificationCard = await currentClarificationCard(page);
    await expect(clarificationCard).toHaveCount(1);
    await expect(clarificationCard).toBeVisible();
  }
  if (expected.require_clarification_card_if_suspended && summary.event === "suspended") {
    const clarificationCard = await currentClarificationCard(page);
    await expect(clarificationCard).toHaveCount(1);
    await expect(clarificationCard).toBeVisible();
  }
  if (expected.require_clarification_input) {
    expect(summary.event).toBe("suspended");
    await expect(page.locator(".clarification:visible").first()).toBeVisible();
    await expect(page.getByRole("textbox", { name: /补充确认/ })).toBeVisible();
  }
  if (expected.clarification_text_contains) {
    await expect(page.getByText(expected.clarification_text_contains, { exact: false }).first()).toBeVisible();
  }
  if (expected.presentation_status === "unsupported") {
    await expect(page.getByText("未执行数据查询", { exact: true }).first()).toBeVisible();
    await expect(page.getByText("已按此口径执行", { exact: true })).toHaveCount(0);
  }
  if (expected.require_candidate_buttons) {
    expect(summary.event).toBe("suspended");
    const clarificationCard = await currentClarificationCard(page);
    const candidateButtons = clarificationCard.locator(".clarification-actions button").filter({ hasText: "确认使用：" });
    expect(summary.clarification_candidate_count, "the server clarification must include its candidate set").not.toBeNull();
    expect(summary.clarification_candidate_count!).toBeGreaterThan(0);
    await expect(candidateButtons).toHaveCount(summary.clarification_candidate_count!);
    await expect(candidateButtons.first()).toBeVisible();
  }
  if (expected.no_business_query_tool) {
    const trace = await traceForTurn();
    const snapshot = trace.state_snapshot ?? {};
    expect(snapshot.query_results ?? []).toEqual([]);
    const toolNames = valuesForKey(snapshot.tool_calls ?? [], "tool_name").filter((value): value is string => typeof value === "string");
    expect(toolNames.filter((name) => /^(query_|registry_|breakdown_|compare_)/.test(name))).toEqual([]);
  }
  if (expected.no_state_errors) {
    const trace = await traceForTurn();
    expect(trace.state_snapshot?.errors ?? [], "workflow trace must not contain internal errors").toEqual([]);
  }
  if (expected.require_fact_evidence) {
    expect(summary.metric_names.length).toBeGreaterThan(0);
    expect(summary.fact_evidence_count).toBeGreaterThan(0);
    expect(summary.fact_values.some((value) => value !== null)).toBeTruthy();
  }
  if (expected.require_evidence_if_completed && summary.event === "completed") {
    expect(summary.evidence_count, "completed result must reference structured presentation evidence").toBeGreaterThan(0);
  }
  if (expected.require_fact_evidence_if_completed && summary.event === "completed") {
    expect(summary.fact_evidence_count, "completed metric fact must be backed by evidence").toBeGreaterThan(0);
  }
  if (expected.forbid_numeric_fact_evidence) {
    expect(summary.fact_values.every((value) => value === null), "no-data result must not present a numeric fact").toBeTruthy();
  }
  if (expected.require_comparison_card) {
    expect(Array.isArray(summary.presentation.changes) && summary.presentation.changes.length > 0).toBeTruthy();
  }
  if (expected.value_matches_turn) {
    const previous = prior[expected.value_matches_turn - 1];
    expect(previous?.fact_values[0]).not.toBeNull();
    expect(summary.fact_values[0]).toBeCloseTo(previous.fact_values[0] as number, 4);
  }
  if (expected.answer_matches_top_dimension_value_from_prior_query) {
    const previous = prior[prior.length - 1];
    expect(previous?.trace_id).toBeTruthy();
    const priorTrace = await readTrace(page, previous!.trace_id);
    const priorResults = Array.isArray(priorTrace.state_snapshot?.query_results) ? priorTrace.state_snapshot.query_results : [];
    const priorItems = resultPayloads(priorResults).flatMap((item) => Array.isArray(item.items) ? item.items : []);
    const top = priorItems
      .filter((item: Record<string, any>) => typeof item?.dimension_value === "string" && item?.value !== null && Number.isFinite(Number(item?.value)))
      .sort((left: Record<string, any>, right: Record<string, any>) => Number(right.value) - Number(left.value))[0];
    expect(top?.dimension_value, "previous breakdown must expose a top category").toBeTruthy();
    expect(summary.answer).toContain(top.dimension_value);
  }
  if (expected.answer_dimension_values_at_most_from_prior_query !== undefined) {
    const previous = prior[prior.length - 1];
    expect(previous?.trace_id).toBeTruthy();
    const priorTrace = await readTrace(page, previous!.trace_id);
    const priorResults = Array.isArray(priorTrace.state_snapshot?.query_results) ? priorTrace.state_snapshot.query_results : [];
    const dimensionValues = [...new Set(resultPayloads(priorResults).flatMap((item) =>
      Array.isArray(item.items)
        ? item.items.map((row: Record<string, any>) => row?.dimension_value).filter((value: unknown) => typeof value === "string")
        : []))] as string[];
    const mentionedValues = dimensionValues.filter((value) => summary.answer.includes(value));
    expect(mentionedValues.length, "follow-up must stay within the requested Top-N result set")
      .toBeLessThanOrEqual(expected.answer_dimension_values_at_most_from_prior_query);
  }
  if (expected.scope_seller_id_matches_session) {
    const sellerId = process.env.E2E_SELLER_ID ?? "";
    expect(sellerId.length).toBeGreaterThan(0);
    expect(summary.presentation.scope?.seller_id).toBe(sellerId);
  }
}

async function attachCaseEvidence(
  testInfo: TestInfo,
  matrixCase: ExpandedCase,
  summaries: TurnSummary[],
  chatRequests: ChatRequest[],
  status: "in_progress" | "passed",
  api5xxCount = 0,
  pageErrorCount = 0,
  complexEvidence?: ComplexEvidence,
  reportEvidence?: ReportEvidence,
  metricTargetEvidence?: MetricTargetEvidence,
  knowledgeEvidence?: KnowledgeEvidence,
  preferenceEvidence?: PreferenceEvidence,
  actionPlanEvidence?: ActionPlanEvidence,
  qualityEvidence?: QualityEvidence,
): Promise<void> {
  await testInfo.attach("expanded-case-evidence.json", {
    body: Buffer.from(JSON.stringify({
      test: testInfo.title,
      case_id: matrixCase.case_id,
      feature: matrixCase.feature,
      evidence_requirement: matrixCase.report_workflow
        ? "report_api_readback"
        : matrixCase.metric_target_workflow ? "metric_target_api_readback"
          : matrixCase.knowledge_workflow ? "knowledge_api_readback"
            : matrixCase.preference_workflow ? "preference_api_readback"
              : matrixCase.action_plan_workflow ? "action_plan_api_readback"
                : matrixCase.quality_workflow ? "quality_ui_api_readback" : "workflow_trace",
      outcome_class: matrixCase.outcome_class ?? null,
      persona: matrixCase.persona,
      status,
      turns: summaries.map((item, index) => ({
        event: item.event,
        error: item.has_error,
        error_code: item.terminal_failure_code,
        task_type: item.task_type,
        trace_id: item.trace_id,
        request_client_turn_id: chatRequests[index]?.chat_turn_id ?? null,
        metric_names: item.metric_names,
        date_labels: item.date_labels,
        evidence_count: item.evidence_count,
        fact_evidence_count: item.fact_evidence_count,
        clarification_status: item.clarification_status,
        clarification_reason_code: item.clarification_reason_code,
        terminal_status: item.terminal_status,
      })),
      trace_ids: [...new Set([...summaries.map((item) => item.trace_id).filter(Boolean), ...(complexEvidence?.trace_id ? [complexEvidence.trace_id] : [])])],
      plan_ids: complexEvidence?.plan_ids ?? [],
      complex_analysis: complexEvidence ?? null,
      report_evidence: reportEvidence ?? null,
      metric_target_evidence: metricTargetEvidence ?? null,
      knowledge_evidence: knowledgeEvidence ?? null,
      preference_evidence: preferenceEvidence ?? null,
      action_plan_evidence: actionPlanEvidence ?? null,
      quality_evidence: qualityEvidence ?? null,
      client_turn_ids: [...new Set(chatRequests.map((request) => request.chat_turn_id).filter(Boolean))],
      session_ids: chatRequests.map((request) => request.chat_session_id).filter(Boolean),
      expected_browser_terminal_events: matrixCase.turns.map((turn) => turn.expect?.event ?? null),
      observed_browser_terminal_events: summaries.map((item) => item.event),
      expected_terminal_statuses: matrixCase.expected_terminal_statuses ?? null,
      expected_non_durable_requests: matrixCase.expected_non_durable_requests ?? [],
      api_5xx_count: api5xxCount,
      page_error_count: pageErrorCount,
    }, null, 2)),
    contentType: "application/json",
  });
}

function reportPayloadFromRecord(record: Record<string, any>): Record<string, any> {
  const raw = record.report_json ?? record.report;
  if (typeof raw === "string") return JSON.parse(raw) as Record<string, any>;
  if (raw && typeof raw === "object") return raw as Record<string, any>;
  throw new Error("Report API readback did not include a report payload");
}

async function readReportRecord(page: Page, reportId: string, version?: number): Promise<{ status: number; body: Record<string, any> }> {
  return page.evaluate(async ({ id, reportVersion }) => {
    const stored = JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}") as { bearerToken?: string };
    if (!stored.bearerToken) throw new Error("Report readback requires the authenticated session");
    const query = reportVersion ? `?version=${reportVersion}` : "";
    const response = await fetch(`/analysis/reports/${encodeURIComponent(id)}${query}`, {
      headers: { Authorization: `Bearer ${stored.bearerToken}` },
    });
    return { status: response.status, body: await response.json() as Record<string, any> };
  }, { id: reportId, reportVersion: version ?? null });
}

async function browserReportRequest(
  page: Page,
  path: string,
  method: "GET" | "POST",
  body?: Record<string, unknown>,
): Promise<{ status: number; body: string; resultPolicy: string | null; disposition: string | null }> {
  return page.evaluate(async ({ requestPath, requestMethod, requestBody }) => {
    const stored = JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}") as { bearerToken?: string };
    if (!stored.bearerToken) throw new Error("Report boundary probe requires the authenticated session");
    const response = await fetch(requestPath, {
      method: requestMethod,
      headers: {
        Authorization: `Bearer ${stored.bearerToken}`,
        ...(requestBody ? { "Content-Type": "application/json" } : {}),
      },
      ...(requestBody ? { body: JSON.stringify(requestBody) } : {}),
    });
    return {
      status: response.status,
      body: await response.text(),
      resultPolicy: response.headers.get("X-Result-Policy"),
      disposition: response.headers.get("Content-Disposition"),
    };
  }, { requestPath: path, requestMethod: method, requestBody: body ?? null });
}

async function runReportWorkflow(page: Page, workflow: ReportWorkflow): Promise<ReportEvidence> {
  const evidence: ReportEvidence = {
    status: "verified", create_status: null, create_error: null, report_id: null,
    report_type: null, period: null, scope_type: null, seller_id: null,
    metric_names: [], metric_values: {}, chart_statuses: [], version_states: [],
    final_status: null, export_status: null, export_policy: null, export_filename: null,
    export_contains_expected: null, draft_export_guard_status: null,
    stale_publish_status: null, unsupported_metric_ui_disabled: null,
    unsupported_metric_probe_status: null, unsupported_metric_probe_error: null,
  };

  await page.goto("/reports");
  await expect(page.getByRole("heading", { name: "周期报告", exact: true })).toBeVisible();
  const form = page.locator(".page .el-form").first();
  const reportPanel = page.locator(".page .el-row").first().locator(".el-col").nth(1);
  const scopeItem = form.locator(".el-form-item").filter({ hasText: "报告范围" });
  const scopeRadios = scopeItem.locator(".el-radio-group");
  const sellerId = process.env.E2E_SELLER_ID ?? "";

  if (workflow.scope_type === "platform") {
    const platformRadio = scopeRadios.getByText("全平台", { exact: true });
    await expect(platformRadio).toBeVisible();
    await platformRadio.click();
    await expect(scopeRadios.locator(".el-radio.is-checked")).toContainText("全平台");
  } else {
    expect(sellerId, "the synthetic seller must be configured for seller reports").not.toBe("");
    await expect(scopeRadios.getByText("单店", { exact: true })).toBeVisible();
    await expect(scopeRadios.locator(".el-radio")).toHaveCount(1);
    await expect(scopeRadios.locator(".el-radio.is-checked")).toContainText("单店");
    await expect(scopeRadios.getByRole("radio", { name: "单店", exact: true })).toBeDisabled();
  }

  const granularityItem = form.locator(".el-form-item").filter({ hasText: "周期粒度" });
  const granularityLabel = workflow.granularity === "month" ? "月报" : "周报";
  await granularityItem.getByText(granularityLabel, { exact: true }).click();
  const dateInputs = form.locator(".el-form-item").filter({ hasText: "起止日期" }).locator(".el-date-editor input");
  await dateInputs.nth(0).fill(workflow.start_date);
  await dateInputs.nth(0).press("Enter");
  // The form aligns a selected date to its complete natural week/month.
  await expect(dateInputs.nth(0)).toHaveValue(workflow.start_date);

  const metricItem = form.locator(".el-form-item").filter({ hasText: "指标" });
  const metricSelect = metricItem.locator(".el-select");
  await metricSelect.click();
  const options = page.locator(".el-select-dropdown:visible .el-select-dropdown__item");
  await expect(options.first()).toBeVisible();
  if (workflow.unsupported_metric_probe) {
    const blocked = options.filter({ hasText: workflow.unsupported_metric_probe });
    await expect(blocked).toHaveCount(1);
    await expect(blocked).toHaveClass(/is-disabled/);
    evidence.unsupported_metric_ui_disabled = true;
  } else {
    for (let index = 0; index < await options.count(); index += 1) {
      const option = options.nth(index);
      const label = await option.innerText();
      const disabled = (await option.getAttribute("class") ?? "").includes("is-disabled");
      const selected = (await option.getAttribute("class") ?? "").includes("is-selected");
      const wanted = workflow.metric_names.some((name) => label.includes(`(${name})`) || label.trim() === name);
      if (!disabled && selected !== wanted) await option.click();
    }
    for (const name of workflow.metric_names) {
      await expect(metricSelect).toContainText(name);
    }
  }
  await page.keyboard.press("Escape");

  const chartsCheckbox = form.locator(".el-checkbox").filter({ hasText: "生成趋势与贡献图" });
  const chartsCheckboxInput = chartsCheckbox.locator("input.el-checkbox__original");
  await expect(chartsCheckbox).toBeVisible();
  if (await chartsCheckboxInput.isChecked() !== workflow.include_charts) await chartsCheckbox.click();
  await expect(chartsCheckboxInput).toBeChecked({ checked: workflow.include_charts });

  if (workflow.unsupported_metric_probe) {
    const probe = await browserReportRequest(page, "/analysis/reports", "POST", {
      start_date: workflow.start_date,
      end_date: workflow.end_date,
      granularity: workflow.granularity,
      scope_type: "seller",
      seller_id: sellerId,
      metric_names: [workflow.unsupported_metric_probe],
      include_charts: workflow.include_charts,
    });
    evidence.unsupported_metric_probe_status = probe.status;
    try {
      evidence.unsupported_metric_probe_error = String((JSON.parse(probe.body) as Record<string, unknown>).detail ?? "");
    } catch {
      evidence.unsupported_metric_probe_error = probe.body.slice(0, 300);
    }
    expect(probe.status).toBe(400);
    expect(evidence.unsupported_metric_probe_error).toContain(workflow.unsupported_metric_error ?? "Seller reports do not support");
    return evidence;
  }

  const createResponsePromise = page.waitForResponse((response) =>
    new URL(response.url()).pathname === "/analysis/reports" && response.request().method() === "POST",
  );
  await form.getByRole("button", { name: "生成报告草稿", exact: true }).click();
  const createResponse = await createResponsePromise;
  evidence.create_status = createResponse.status();
  const createBody = await createResponse.json().catch(() => ({})) as Record<string, any>;
  evidence.create_error = String(createBody.detail ?? "") || null;
  expect(createResponse.status()).toBe(workflow.expected_create_status ?? 200);
  if (workflow.expected_create_status && workflow.expected_create_status >= 400) {
    expect(evidence.create_error).toContain(workflow.expected_error_contains ?? "");
    await expect(page.locator(".el-message--error").last()).toContainText(workflow.expected_error_contains ?? "");
    return evidence;
  }

  const reportId = String(createBody.report?.report_id ?? "");
  expect(reportId).not.toBe("");
  evidence.report_id = reportId;
  const createdRecord = await readReportRecord(page, reportId);
  expect(createdRecord.status).toBe(200);
  let record = createdRecord.body;
  let payload = reportPayloadFromRecord(record);
  const actualScope = payload.scope ?? {};
  const period = payload.period ?? {};
  evidence.report_type = String(payload.report_type ?? "");
  evidence.period = { start_date: String(period.start_date ?? ""), end_date: String(period.end_date ?? "") };
  evidence.scope_type = String(actualScope.scope_type ?? "");
  evidence.seller_id = actualScope.seller_id ? String(actualScope.seller_id) : null;
  const reportMetrics = Array.isArray(payload.metrics) ? payload.metrics as Array<Record<string, any>> : [];
  evidence.metric_names = reportMetrics.map((metric) => String(metric.metric_name));
  evidence.metric_values = Object.fromEntries(reportMetrics.map((metric) => [String(metric.metric_name), metric.value == null ? null : Number(metric.value)]));
  evidence.chart_statuses = Array.isArray(payload.charts) ? payload.charts.map((chart: Record<string, any>) => String(chart.status)) : [];
  expect(payload.report_id).toBe(reportId);
  expect(payload.report_type).toBe(workflow.granularity === "month" ? "monthly" : "weekly");
  expect(evidence.period).toEqual({ start_date: workflow.start_date, end_date: workflow.end_date });
  expect(actualScope.scope_type).toBe(workflow.scope_type);
  if (workflow.scope_type === "seller") {
    expect(actualScope.seller_id).toBe(workflow.expected_seller_id ?? sellerId);
  } else {
    expect(actualScope.seller_id ?? null).toBeNull();
  }
  expect(evidence.metric_names.sort()).toEqual([...workflow.metric_names].sort());
  for (const metric of workflow.expected_null_metrics ?? []) expect(evidence.metric_values[metric]).toBeNull();
  for (const metric of workflow.expected_non_null_metrics ?? []) expect(evidence.metric_values[metric]).not.toBeNull();
  for (const risk of workflow.expected_risks ?? []) expect(payload.risks ?? []).toContain(risk);
  if (workflow.expected_chart_statuses) expect(evidence.chart_statuses).toEqual(workflow.expected_chart_statuses);
  if (workflow.include_charts && evidence.chart_statuses.includes("ok")) {
    await expect(reportPanel.getByRole("heading", { name: "图表", exact: true })).toBeVisible();
    await expect(reportPanel.locator(".chart-evidence").first()).toBeVisible();
  }
  expect(Array.isArray(payload.lineage_refs) && payload.lineage_refs.length).toBeTruthy();
  expect(payload.evidence_records && Object.keys(payload.evidence_records).length).toBeGreaterThan(0);
  evidence.version_states.push({ version: Number(record.version), status: String(record.status) });

  await expect(reportPanel.locator(".card-header")).toContainText(reportId);
  await expect(reportPanel.locator(".card-header")).toContainText("draft");

  if (workflow.actions?.includes("guard_export")) {
    await expect(reportPanel.getByRole("button", { name: "导出 Markdown", exact: true })).toHaveCount(0);
    const response = await browserReportRequest(page, `/analysis/reports/${encodeURIComponent(reportId)}/export?version=${record.version}`, "GET");
    evidence.draft_export_guard_status = response.status;
    expect(response.status).toBe(409);
    expect(response.body).toContain("Only published reports can be exported");
  }

  for (const action of workflow.actions ?? []) {
    if (action === "guard_export") continue;
    if (action === "approve" || action === "reject") {
      const responsePromise = page.waitForResponse((response) =>
        new URL(response.url()).pathname === `/analysis/reports/${encodeURIComponent(reportId)}/review`
          && response.request().method() === "POST",
      );
      await reportPanel.getByRole("button", { name: action === "approve" ? "审核通过" : "驳回", exact: true }).click();
      const dialog = page.locator(".el-message-box:visible");
      await expect(dialog).toBeVisible();
      await dialog.locator("input").fill(workflow.review_reason ?? `B6 ${action} review`);
      await dialog.getByRole("button", { name: "提交", exact: true }).click();
      const response = await responsePromise;
      expect(response.status()).toBe(200);
      record = (await readReportRecord(page, reportId)).body;
      evidence.version_states.push({ version: Number(record.version), status: String(record.status) });
      await expect(reportPanel.locator(".card-header")).toContainText(String(record.status));
    } else if (action === "edit_approve") {
      await reportPanel.getByRole("button", { name: "编辑 Markdown", exact: true }).click();
      const editor = reportPanel.locator("textarea").first();
      const original = await editor.inputValue();
      const marker = workflow.edit_marker ?? "B6_REVIEW_EDITED";
      await editor.fill(`${original}\n\n${marker}`);
      const responsePromise = page.waitForResponse((response) =>
        new URL(response.url()).pathname === `/analysis/reports/${encodeURIComponent(reportId)}/review`
          && response.request().method() === "POST",
      );
      await reportPanel.getByRole("button", { name: "保存编辑并通过", exact: true }).click();
      const dialog = page.locator(".el-message-box:visible");
      await dialog.locator("input").fill(workflow.review_reason ?? "B6 edit and approve");
      await dialog.getByRole("button", { name: "提交", exact: true }).click();
      const response = await responsePromise;
      expect(response.status()).toBe(200);
      record = (await readReportRecord(page, reportId)).body;
      evidence.version_states.push({ version: Number(record.version), status: String(record.status) });
      expect(Number(record.version)).toBe(2);
      expect(String(record.status)).toBe("approved");
      expect(String(record.content)).toContain(marker);
      const v1 = await readReportRecord(page, reportId, 1);
      expect(v1.status).toBe(200);
      expect(v1.body.status).toBe("rejected");
      evidence.version_states.push({ version: 1, status: String(v1.body.status) });
    } else if (action === "publish") {
      const responsePromise = page.waitForResponse((response) =>
        new URL(response.url()).pathname === `/analysis/reports/${encodeURIComponent(reportId)}/publish`
          && response.request().method() === "POST",
      );
      await reportPanel.getByRole("button", { name: "发布", exact: true }).click();
      const dialog = page.locator(".el-message-box:visible");
      await expect(dialog).toBeVisible();
      await dialog.getByRole("button", { name: "OK", exact: true }).click();
      const response = await responsePromise;
      expect(response.status()).toBe(200);
      record = (await readReportRecord(page, reportId)).body;
      evidence.version_states.push({ version: Number(record.version), status: String(record.status) });
      await expect(reportPanel.locator(".card-header")).toContainText("published");
    } else if (action === "stale_publish_v1") {
      const actor = process.env.E2E_PLATFORM_USER_ID ?? "";
      expect(actor, "the synthetic platform actor must be configured for the stale-version probe").not.toBe("");
      const response = await browserReportRequest(page, `/analysis/reports/${encodeURIComponent(reportId)}/publish`, "POST", {
        version: 1,
        actor,
      });
      evidence.stale_publish_status = response.status;
      expect(response.status).toBe(409);
      record = (await readReportRecord(page, reportId)).body;
      expect(Number(record.version)).toBe(2);
      expect(String(record.status)).toBe("approved");
    } else if (action === "export") {
      const downloadPromise = page.waitForEvent("download");
      const responsePromise = page.waitForResponse((response) =>
        new URL(response.url()).pathname === `/analysis/reports/${encodeURIComponent(reportId)}/export`,
      );
      await reportPanel.getByRole("button", { name: "导出 Markdown", exact: true }).click();
      const response = await responsePromise;
      const download = await downloadPromise;
      evidence.export_status = response.status();
      evidence.export_policy = response.headers()["x-result-policy"] ?? null;
      evidence.export_filename = download.suggestedFilename();
      const downloadedPath = await download.path();
      expect(downloadedPath, "the browser should materialize the exported report file").not.toBeNull();
      const exported = readFileSync(downloadedPath!, "utf8");
      evidence.export_contains_expected = workflow.export_contains ? exported.includes(workflow.export_contains) : exported.length > 0;
      expect(response.status()).toBe(200);
      expect(evidence.export_policy).toBe("approved-published-artifact");
      expect(evidence.export_filename).toBe(`${reportId}.md`);
      expect(evidence.export_contains_expected).toBeTruthy();
      if (workflow.export_contains) expect(exported).toContain(workflow.export_contains);
    }
  }

  record = (await readReportRecord(page, reportId)).body;
  evidence.final_status = String(record.status);
  evidence.version_states.push({ version: Number(record.version), status: String(record.status) });
  if (workflow.expected_status) expect(evidence.final_status).toBe(workflow.expected_status);
  return evidence;
}

async function browserMetricTargetRequest(
  page: Page,
  path: string,
  method: "GET" | "PUT" | "DELETE",
  body?: Record<string, unknown>,
): Promise<{ status: number; body: Record<string, any>; text: string }> {
  return page.evaluate(async ({ requestPath, requestMethod, requestBody }) => {
    const stored = JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}") as { bearerToken?: string };
    if (!stored.bearerToken) throw new Error("Metric/target readback requires the authenticated session");
    const response = await fetch(requestPath, {
      method: requestMethod,
      headers: {
        Authorization: `Bearer ${stored.bearerToken}`,
        ...(requestBody ? { "Content-Type": "application/json" } : {}),
      },
      ...(requestBody ? { body: JSON.stringify(requestBody) } : {}),
    });
    const text = await response.text();
    let parsed: Record<string, any> = {};
    try { parsed = JSON.parse(text) as Record<string, any>; } catch { /* keep the bounded raw response for failure evidence */ }
    return { status: response.status, body: parsed, text };
  }, { requestPath: path, requestMethod: method, requestBody: body ?? null });
}

async function browserKnowledgeRead(page: Page, includeText = false): Promise<{
  status: number;
  knowledgeRoot: string;
  collection: string;
  indexVersion: string;
  total: number;
  documents: Array<{ docId: string; title: string; sourcePath: string; metricName: string | null; indexVersion: string }>;
  defaultTextFieldsAbsent: boolean;
  textDocumentCount: number;
  textTotalChars: number;
}> {
  return page.evaluate(async (withText) => {
    const stored = JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}") as { bearerToken?: string };
    if (!stored.bearerToken) throw new Error("Knowledge readback requires the authenticated session");
    const response = await fetch(`/knowledge/documents?include_text=${withText ? "true" : "false"}`, {
      headers: { Authorization: `Bearer ${stored.bearerToken}` },
    });
    const payload = await response.json() as Record<string, any>;
    const documents = Array.isArray(payload.documents) ? payload.documents as Array<Record<string, any>> : [];
    // Reduce the response inside the page. Raw document text is never returned to
    // Playwright, written to attachments, or included in E2E reports.
    return {
      status: response.status,
      knowledgeRoot: String(payload.knowledge_root ?? ""),
      collection: String(payload.collection ?? ""),
      indexVersion: String(payload.index_version ?? ""),
      total: Number(payload.total ?? 0),
      documents: documents.map((item) => ({
        docId: String(item.doc_id ?? ""),
        title: String(item.title ?? ""),
        sourcePath: String(item.source_path ?? ""),
        metricName: item.metric_name == null ? null : String(item.metric_name),
        indexVersion: String(item.index_version ?? ""),
      })),
      defaultTextFieldsAbsent: documents.every((item) => !Object.prototype.hasOwnProperty.call(item, "text")),
      textDocumentCount: documents.filter((item) => typeof item.text === "string").length,
      textTotalChars: documents.reduce((sum, item) => sum + (typeof item.text === "string" ? item.text.length : 0), 0),
    };
  }, includeText);
}

async function browserKnowledgeIndexProbe(page: Page, collection: string): Promise<{ status: number; error: string }> {
  return page.evaluate(async (targetCollection) => {
    const stored = JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}") as { bearerToken?: string };
    if (!stored.bearerToken) throw new Error("Knowledge index permission probe requires the authenticated session");
    const params = new URLSearchParams({ collection: targetCollection, dry_run: "true" });
    const response = await fetch(`/knowledge/index?${params.toString()}`, {
      method: "POST",
      headers: { Authorization: `Bearer ${stored.bearerToken}` },
    });
    const text = await response.text();
    let error = "";
    try {
      const payload = JSON.parse(text) as Record<string, unknown>;
      error = String(payload.detail ?? "").slice(0, 240);
    } catch {
      error = text.slice(0, 240);
    }
    return { status: response.status, error };
  }, collection);
}

async function runKnowledgeWorkflow(page: Page, workflow: KnowledgeWorkflow): Promise<KnowledgeEvidence> {
  await page.goto("/knowledge");
  await expect(page.getByRole("heading", { name: "知识库", exact: true })).toBeVisible();
  await expect(page.getByText("文档列表", { exact: true })).toBeVisible();

  const indexRequests: string[] = [];
  const captureIndexRequest = (request: import("@playwright/test").Request) => {
    if (new URL(request.url()).pathname === "/knowledge/index" && request.method() === "POST") {
      indexRequests.push(request.method());
    }
  };
  page.on("request", captureIndexRequest);
  const initial = await browserKnowledgeRead(page);
  expect(initial.status).toBe(200);
  expect(initial.knowledgeRoot).not.toBe("");
  expect(initial.collection).not.toBe("");
  expect(initial.indexVersion).not.toBe("");
  expect(initial.total).toBeGreaterThan(0);
  expect(initial.documents).toHaveLength(initial.total);
  expect(initial.documents.every((item) => item.docId && item.title && item.sourcePath && item.indexVersion)).toBeTruthy();
  expect(initial.defaultTextFieldsAbsent, "default document-list API must omit raw text").toBeTruthy();
  const rows = page.locator(".section .el-table__body tbody tr");
  await expect(rows).toHaveCount(initial.total);

  const expectedMetricFound = workflow.expected_metric
    ? initial.documents.some((item) => item.metricName === workflow.expected_metric)
    : null;
  if (workflow.expected_metric) {
    expect(expectedMetricFound, `catalog metadata must include ${workflow.expected_metric}`).toBeTruthy();
    await expect(rows.filter({ hasText: workflow.expected_metric })).toHaveCount(1);
  }

  const updateButton = page.getByRole("button", { name: "更新索引", exact: true });
  const updateButtonVisible = await updateButton.isVisible().catch(() => false);
  let refreshStable: boolean | null = null;
  let includeTextDocumentCount: number | null = null;
  let includeTextTotalChars: number | null = null;
  let indexProbeStatus: number | null = null;
  let indexProbeError: string | null = null;

  if (workflow.mode === "refresh_readback") {
    await page.getByRole("button", { name: "刷新", exact: true }).click();
    const refreshed = await browserKnowledgeRead(page);
    expect(refreshed.status).toBe(200);
    expect(refreshed.documents.map((item) => item.docId).sort()).toEqual(initial.documents.map((item) => item.docId).sort());
    expect(refreshed.indexVersion).toBe(initial.indexVersion);
    refreshStable = true;
  }
  if (workflow.mode === "include_text") {
    const withText = await browserKnowledgeRead(page, true);
    expect(withText.status).toBe(200);
    expect(withText.textDocumentCount).toBe(withText.total);
    expect(withText.textTotalChars).toBeGreaterThan(0);
    includeTextDocumentCount = withText.textDocumentCount;
    includeTextTotalChars = withText.textTotalChars;
  }
  if (workflow.mode === "index_control") {
    expect(updateButtonVisible, "platform administrators should see the shared-index control").toBeTruthy();
    // Deliberately do not click: this E2E suite treats the configured Qdrant collection as read-only.
  }
  if (workflow.mode === "index_denied") {
    expect(updateButtonVisible, "seller users must not see shared-index mutation controls").toBeFalsy();
    const probe = await browserKnowledgeIndexProbe(page, initial.collection);
    indexProbeStatus = probe.status;
    indexProbeError = probe.error;
    expect(probe.status).toBe(403);
    expect(probe.error).toContain("Only platform administrators");
  }
  expect(indexRequests, "this read-only workflow must not trigger a shared-index write").toEqual(
    workflow.mode === "index_denied" ? ["POST"] : [],
  );
  page.off("request", captureIndexRequest);

  return {
    status: "verified",
    mode: workflow.mode,
    document_count: initial.total,
    document_ids: initial.documents.map((item) => item.docId).sort(),
    index_version: initial.indexVersion,
    default_text_fields_absent: initial.defaultTextFieldsAbsent,
    expected_metric_found: expectedMetricFound,
    refresh_stable: refreshStable,
    include_text_document_count: includeTextDocumentCount,
    include_text_total_chars: includeTextTotalChars,
    update_button_visible: updateButtonVisible,
    index_probe_status: indexProbeStatus,
    index_probe_error: indexProbeError,
  };
}

async function browserPreferenceRequest(
  page: Page,
  path: string,
  method: "GET" | "POST" | "PUT" | "DELETE",
  body?: Record<string, unknown>,
): Promise<{ status: number; body: Record<string, any> }> {
  return page.evaluate(async ({ requestPath, requestMethod, requestBody }) => {
    const stored = JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}") as { bearerToken?: string; actor?: string };
    if (!stored.bearerToken) throw new Error("Preference readback requires the authenticated session");
    const response = await fetch(requestPath, {
      method: requestMethod,
      headers: {
        Authorization: `Bearer ${stored.bearerToken}`,
        "X-Actor-ID": stored.actor ?? "",
        ...(requestBody ? { "Content-Type": "application/json" } : {}),
      },
      ...(requestBody ? { body: JSON.stringify(requestBody) } : {}),
    });
    const text = await response.text();
    let parsed: Record<string, any> = {};
    try { parsed = JSON.parse(text) as Record<string, any>; } catch { parsed = { detail: text.slice(0, 160) }; }
    return { status: response.status, body: parsed };
  }, { requestPath: path, requestMethod: method, requestBody: body ?? null });
}

async function preferenceReadback(page: Page, sessionId?: string): Promise<{
  status: number; personalization_enabled: boolean; items: Array<{
    registry_id: string; memory_key: string; status: string; overlay_status: string | null;
  }>;
}> {
  const query = sessionId ? `?chat_session_id=${encodeURIComponent(sessionId)}` : "";
  const result = await browserPreferenceRequest(page, `/memory/preferences${query}`, "GET");
  return {
    status: result.status,
    personalization_enabled: Boolean(result.body.personalization_enabled),
    items: Array.isArray(result.body.items) ? result.body.items.map((item: Record<string, any>) => ({
      registry_id: String(item.registry_id ?? ""),
      memory_key: String(item.memory_key ?? ""),
      status: String(item.status ?? ""),
      overlay_status: item.overlay?.status == null ? null : String(item.overlay.status),
    })) : [],
  };
}

async function createPreferenceSession(page: Page): Promise<string> {
  const responsePromise = page.waitForResponse((response) =>
    new URL(response.url()).pathname.endsWith("/chat/sessions")
    && response.request().method() === "POST");
  await page.getByRole("button", { name: "新建会话", exact: true }).click();
  const response = await responsePromise;
  expect(response.status()).toBe(201);
  const body = await response.json() as Record<string, any>;
  expect(typeof body.id).toBe("string");
  expect(body.id).not.toBe("");
  return String(body.id);
}

async function confirmMessageBox(page: Page, label: string): Promise<void> {
  const box = page.locator(".el-message-box:visible");
  await expect(box).toBeVisible();
  await box.getByRole("button", { name: label, exact: true }).click();
}

async function runPreferenceWorkflow(page: Page, workflow: PreferenceWorkflow): Promise<PreferenceEvidence> {
  await page.goto("/preferences");
  await expect(page.getByRole("heading", { name: "记忆与偏好", exact: true })).toBeVisible();
  const evidence: PreferenceEvidence = {
    status: "verified", mode: workflow.mode, personalization_enabled: true,
    session_ids: [], mutations: [], readback_statuses: [], cancel_left_no_write: null,
    checked_registry_ids: [],
    denied_status: null, denied_reason: null, history_row_count: null,
    delete_all_queued_count: null, confirmation_phrase_rejected: null,
    bulk_deleted_registry_ids: [],
    raw_preference_values_recorded: false,
  };
  const readCurrent = async (sessionId?: string) => {
    const readback = await preferenceReadback(page, sessionId);
    expect(readback.status).toBe(200);
    evidence.personalization_enabled = readback.personalization_enabled;
    evidence.readback_statuses = readback.items.map((item) => ({
      registry_id: item.registry_id, status: item.status, overlay_status: item.overlay_status,
    }));
    return readback;
  };
  const captureMutation = async (
    response: import("@playwright/test").Response,
    sessionId?: string | null,
    eventType?: string,
  ) => {
    expect(response.ok()).toBeTruthy();
    const body = await response.json() as Record<string, any>;
    expect(body.registry_id).toBeTruthy();
    expect(body.operation_id).toBeTruthy();
    evidence.mutations.push({
      registry_id: String(body.registry_id), operation_id: String(body.operation_id),
      status: String(body.status), session_id: sessionId ?? null, event_type: eventType ?? null,
    });
    return body;
  };

  const initial = await readCurrent();
  if (workflow.mode === "default_settings_cancel") {
    expect(initial.personalization_enabled).toBeTruthy();
    expect(initial.items).toHaveLength(0);
    const settingRequests: number[] = [];
    const capture = (response: import("@playwright/test").Response) => {
      if (new URL(response.url()).pathname.endsWith("/memory/preferences/settings") && response.request().method() === "PUT") {
        settingRequests.push(response.status());
      }
    };
    page.on("response", capture);
    await page.locator(".preference-toolbar .el-switch__core").click();
    await confirmMessageBox(page, "取消");
    await expect(page.locator(".preference-toolbar .el-switch__input")).toHaveAttribute("aria-checked", "true");
    await page.waitForTimeout(100);
    expect(settingRequests).toEqual([]);
    page.off("response", capture);
    evidence.cancel_left_no_write = true;
  } else if (workflow.mode === "create_confirm" || workflow.mode === "create_cancel") {
    const sessionId = await createPreferenceSession(page);
    evidence.session_ids.push(sessionId);
    await page.getByRole("button", { name: "添加偏好", exact: true }).click();
    await expect(page.getByRole("dialog", { name: "添加偏好" })).toBeVisible();
    if (workflow.memory_key && workflow.memory_key !== "response_style") {
      const labels: Record<string, string> = {
        report_style: "报告格式", response_language: "回答语言",
      };
      const keySelect = page.locator(".el-dialog:visible .el-form-item").filter({ hasText: "偏好字段" }).locator(".el-select");
      await keySelect.click();
      await page.locator(".el-select-dropdown__item:visible").filter({ hasText: labels[workflow.memory_key] ?? workflow.memory_key }).click();
    }
    const writes: string[] = [];
    const captureWrite = (request: import("@playwright/test").Request) => {
      if (new URL(request.url()).pathname.endsWith("/memory/preferences") && request.method() === "POST") writes.push("POST");
    };
    page.on("request", captureWrite);
    await page.getByRole("button", { name: "确认保存", exact: true }).click();
    if (workflow.mode === "create_cancel") {
      await confirmMessageBox(page, "取消");
      await page.waitForTimeout(150);
      const result = await readCurrent(sessionId);
      expect(result.items.some((item) => item.memory_key === workflow.memory_key)).toBeFalsy();
      expect(writes).toEqual([]);
      evidence.cancel_left_no_write = true;
    } else {
      const postPromise = page.waitForResponse((response) =>
        new URL(response.url()).pathname.endsWith("/memory/preferences")
        && response.request().method() === "POST");
      await confirmMessageBox(page, "确认保存");
      const response = await postPromise;
      const result = await captureMutation(response, sessionId, "add");
      expect(result.status).toBe("pending_create");
      const readback = await readCurrent(sessionId);
      const row = readback.items.find((item) => item.registry_id === result.registry_id);
      expect(row?.status).toBe("pending_create");
      expect(row?.overlay_status).toBe("pending");
      await expect(page.locator(".preference-item").filter({ hasText: workflow.memory_key! })).toContainText("等待保存");
    }
    page.off("request", captureWrite);
  } else if (workflow.mode === "personalization_off") {
    const sessionId = await createPreferenceSession(page);
    evidence.session_ids.push(sessionId);
    await page.locator(".preference-toolbar .el-switch__core").click();
    await confirmMessageBox(page, "确认");
    await expect(page.getByText("个性化已关闭；已有偏好不会参与后续回答。", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "添加偏好", exact: true })).toBeDisabled();
    const denied = await browserPreferenceRequest(page, "/memory/preferences", "POST", {
      memory_key: workflow.memory_key ?? "response_language", memory_type: "presentation",
      value: { language: "zh-CN" }, chat_session_id: sessionId,
      idempotency_key: `e2e-disabled-${crypto.randomUUID()}`, confirmed: true,
    });
    evidence.denied_status = denied.status;
    evidence.denied_reason = String(denied.body.detail ?? "");
    expect(denied.status).toBe(422);
    expect(evidence.denied_reason).toBe("personalization_disabled");
    await page.locator(".preference-toolbar .el-switch__core").click();
    await confirmMessageBox(page, "确认");
    const restored = await readCurrent(sessionId);
    expect(restored.personalization_enabled).toBeTruthy();
    expect(restored.items.some((item) => item.memory_key === workflow.memory_key)).toBeFalsy();
  } else if (workflow.mode === "session_overlay") {
    const firstSession = await createPreferenceSession(page);
    evidence.session_ids.push(firstSession);
    await page.getByRole("button", { name: "添加偏好", exact: true }).click();
    await page.getByRole("button", { name: "确认保存", exact: true }).click();
    const responsePromise = page.waitForResponse((response) =>
      new URL(response.url()).pathname.endsWith("/memory/preferences")
      && response.request().method() === "POST");
    await confirmMessageBox(page, "确认保存");
    const response = await responsePromise;
    const created = await captureMutation(response, firstSession, "add");
    expect(created.status).toBe("pending_create");
    const firstRead = await readCurrent(firstSession);
    const firstItem = firstRead.items.find((item) => item.registry_id === created.registry_id);
    expect(firstItem?.overlay_status).toBe("pending");

    const secondSession = await createPreferenceSession(page);
    evidence.session_ids.push(secondSession);
    const secondRead = await readCurrent(secondSession);
    const secondItem = secondRead.items.find((item) => item.registry_id === created.registry_id);
    expect(secondItem?.status).toBe("pending_create");
    expect(secondItem?.overlay_status).toBeNull();
  } else if (workflow.mode === "update_active") {
    const sessionId = await createPreferenceSession(page);
    evidence.session_ids.push(sessionId);
    const item = page.locator(".preference-item").filter({ hasText: workflow.memory_key! });
    await expect(item).toContainText("已保存");
    const oldId = initial.items.find((row) => row.memory_key === workflow.memory_key)?.registry_id;
    await item.getByRole("button", { name: "修改", exact: true }).click();
    await page.locator(".el-dialog:visible textarea").fill('{"metric":"gmv","dimension":"product_category","operation":"compare"}');
    const responsePromise = page.waitForResponse((response) =>
      new URL(response.url()).pathname.match(/\/memory\/preferences\/[^/]+$/)
      && response.request().method() === "PUT");
    await page.getByRole("button", { name: "确认修改", exact: true }).click();
    await confirmMessageBox(page, "确认更新");
    const response = await responsePromise;
    const updated = await captureMutation(response, sessionId, "update");
    expect(updated.status).toBe("pending_update");
    expect(String(updated.registry_id)).not.toBe(oldId);
    const readback = await readCurrent(sessionId);
    const row = readback.items.find((item) => item.registry_id === updated.registry_id);
    expect(row?.status).toBe("pending_update");
    expect(row?.overlay_status).toBe("pending");
  } else if (workflow.mode === "create_then_delete") {
    const sessionId = await createPreferenceSession(page);
    evidence.session_ids.push(sessionId);
    await page.getByRole("button", { name: "添加偏好", exact: true }).click();
    if (workflow.memory_key && workflow.memory_key !== "response_style") {
      const labels: Record<string, string> = { report_style: "报告格式", response_language: "回答语言" };
      const keySelect = page.locator(".el-dialog:visible .el-form-item").filter({ hasText: "偏好字段" }).locator(".el-select");
      await keySelect.click();
      await page.locator(".el-select-dropdown__item:visible").filter({ hasText: labels[workflow.memory_key] ?? workflow.memory_key }).click();
    }
    await page.getByRole("button", { name: "确认保存", exact: true }).click();
    const createPromise = page.waitForResponse((response) =>
      new URL(response.url()).pathname.endsWith("/memory/preferences")
      && response.request().method() === "POST");
    await confirmMessageBox(page, "确认保存");
    const createdResponse = await createPromise;
    const created = await captureMutation(createdResponse, sessionId, "add");
    expect(created.status).toBe("pending_create");
    const item = page.locator(".preference-item").filter({ hasText: workflow.memory_key! });
    await expect(item).toContainText("等待保存");
    const registryId = String(created.registry_id);
    const responsePromise = page.waitForResponse((response) =>
      new URL(response.url()).pathname.endsWith(`/memory/preferences/${encodeURIComponent(registryId!)}`)
      && response.request().method() === "DELETE");
    await item.getByRole("button", { name: "删除", exact: true }).click();
    await confirmMessageBox(page, "确认删除");
    const response = await responsePromise;
    const deleted = await captureMutation(response, null, "delete");
    expect(deleted.status).toBe("pending_delete");
    const readback = await readCurrent(sessionId);
    expect(readback.items.find((row) => row.registry_id === registryId)?.status).toBe("pending_delete");
  } else if (workflow.mode === "history_read") {
    const item = page.locator(".preference-item").filter({ hasText: workflow.memory_key! });
    await expect(item).toContainText("已保存");
    const registryId = initial.items.find((row) => row.memory_key === workflow.memory_key)?.registry_id;
    expect(registryId).toBeTruthy();
    evidence.checked_registry_ids.push(String(registryId));
    await item.getByRole("button", { name: "历史", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: `${workflow.memory_key} · 操作历史` });
    await expect(dialog).toBeVisible();
    const rows = dialog.locator(".el-table__body tbody tr");
    await expect(rows).toHaveCount(1);
    evidence.history_row_count = await rows.count();
  } else if (workflow.mode === "cross_user_denied") {
    const registryId = workflow.foreign_registry_id ?? "";
    expect(registryId).not.toBe("");
    evidence.checked_registry_ids.push(registryId);
    const denied = await browserPreferenceRequest(
      page, `/memory/preferences/${encodeURIComponent(registryId)}/history`, "GET",
    );
    evidence.denied_status = denied.status;
    evidence.denied_reason = String(denied.body.detail ?? "");
    expect(denied.status).toBe(404);
    expect(evidence.denied_reason).toBe("preference_not_found");
    const readback = await readCurrent();
    expect(readback.items.some((item) => item.registry_id === registryId)).toBeFalsy();
  } else if (workflow.mode === "delete_all_exact") {
    const sessionId = await createPreferenceSession(page);
    evidence.session_ids.push(sessionId);
    const createOne = async (keyLabel?: string) => {
      await page.getByRole("button", { name: "添加偏好", exact: true }).click();
      if (keyLabel) {
        const keySelect = page.locator(".el-dialog:visible .el-form-item").filter({ hasText: "偏好字段" }).locator(".el-select");
        await keySelect.click();
        await page.locator(".el-select-dropdown__item:visible").filter({ hasText: keyLabel }).click();
      }
      await page.getByRole("button", { name: "确认保存", exact: true }).click();
      const createPromise = page.waitForResponse((response) =>
        new URL(response.url()).pathname.endsWith("/memory/preferences")
        && response.request().method() === "POST");
      await confirmMessageBox(page, "确认保存");
      const result = await createPromise;
      const mutation = await captureMutation(result, sessionId, "add");
      expect(mutation.status).toBe("pending_create");
    };
    await createOne();
    await createOne("回答语言");
    await page.getByRole("button", { name: "删除全部", exact: true }).click();
    const prompt = page.locator(".el-message-box:visible");
    await expect(prompt).toBeVisible();
    const input = prompt.locator("input");
    await input.fill("NOT_THE_CONFIRMATION");
    await prompt.getByRole("button", { name: "继续", exact: true }).click();
    await expect(prompt.getByText("确认短语不匹配", { exact: true })).toBeVisible();
    evidence.confirmation_phrase_rejected = true;
    await input.fill("DELETE_ALL_PREFERENCES");
    const responsePromise = page.waitForResponse((response) =>
      new URL(response.url()).pathname.endsWith("/memory/preferences")
      && response.request().method() === "DELETE");
    await prompt.getByRole("button", { name: "继续", exact: true }).click();
    const response = await responsePromise;
    expect(response.status()).toBe(200);
    const result = await response.json() as Record<string, any>;
    evidence.delete_all_queued_count = Number(result.queued_count ?? 0);
    evidence.bulk_deleted_registry_ids = Array.isArray(result.pending_registry_ids)
      ? result.pending_registry_ids.map((value: unknown) => String(value)) : [];
    expect(evidence.delete_all_queued_count).toBe(2);
    expect(evidence.bulk_deleted_registry_ids).toHaveLength(2);
    const readback = await readCurrent(sessionId);
    expect(readback.items.filter((item) => item.status === "pending_delete").length).toBe(2);
  }
  return evidence;
}

async function browserActionPlanRequest(
  page: Page,
  path: string,
  method: "GET" | "PUT" | "DELETE",
  body?: Record<string, unknown>,
): Promise<{ status: number; body: Record<string, any> }> {
  return page.evaluate(async ({ requestPath, requestMethod, requestBody }) => {
    const stored = JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}") as { bearerToken?: string; actor?: string };
    if (!stored.bearerToken) throw new Error("Action plan readback requires the authenticated session");
    const response = await fetch(requestPath, {
      method: requestMethod,
      headers: {
        Authorization: `Bearer ${stored.bearerToken}`,
        "X-Actor-ID": stored.actor ?? "",
        ...(requestBody ? { "Content-Type": "application/json" } : {}),
      },
      ...(requestBody ? { body: JSON.stringify(requestBody) } : {}),
    });
    const raw = await response.text();
    let parsed: Record<string, any> = {};
    try { parsed = JSON.parse(raw) as Record<string, any>; } catch { parsed = { detail: raw.slice(0, 160) }; }
    return { status: response.status, body: parsed };
  }, { requestPath: path, requestMethod: method, requestBody: body ?? null });
}

async function readActionPlan(page: Page, actionPlanId: string): Promise<{ status: number; body: Record<string, any> }> {
  return browserActionPlanRequest(page, `/complex-analysis/action-plans/${encodeURIComponent(actionPlanId)}`, "GET");
}

async function selectActionPlanByTitle(page: Page, title: string): Promise<void> {
  const item = page.locator(".el-menu-item").filter({ hasText: title }).first();
  await expect(item).toBeVisible();
  const responsePromise = page.waitForResponse((response) =>
    new URL(response.url()).pathname.startsWith("/complex-analysis/action-plans/")
    && response.request().method() === "GET");
  await item.click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  await expect(page.getByText("证据化经营建议 · v", { exact: false })).toBeVisible();
}

async function runActionPlanWorkflow(page: Page, workflow: ActionPlanWorkflow): Promise<ActionPlanEvidence> {
  const actionPlanId = workflow.action_plan_id ?? "";
  const foreignActionPlanId = workflow.foreign_action_plan_id ?? "";
  const evidence: ActionPlanEvidence = {
    status: "verified", mode: workflow.mode, action_plan_id: actionPlanId || null,
    foreign_action_plan_id: foreignActionPlanId || null,
    complex_plan_id: null, trace_linked: null, evidence_linked: null,
    initial_version: null, final_version: null, final_status: null, mutation_status: null,
    delete_cancelled_without_write: null, deleted: null, cross_user_statuses: [],
    visible_plan_count: null, pagination_page_counts: [], no_execution_request: true,
    raw_plan_content_recorded: false,
  };
  const executionRequests: string[] = [];
  page.on("request", (request) => {
    const path = new URL(request.url()).pathname;
    if (request.method() === "POST" && path.startsWith("/complex-analysis/")) {
      executionRequests.push(path);
    }
  });
  await page.goto("/action-plans");
  await expect(page.getByRole("heading", { name: "经营建议", exact: true })).toBeVisible();
  await expect(page.getByText("系统不会自动执行业务动作。", { exact: false })).toBeVisible();

  if (workflow.mode === "pagination") {
    const firstPage = page.locator(".page > .el-row > .el-col").first().locator(".el-menu-item");
    await expect(firstPage).toHaveCount(20);
    evidence.pagination_page_counts.push(await firstPage.count());
    evidence.visible_plan_count = await firstPage.count();
    const listRequests: string[] = [];
    page.on("request", (request) => {
      const url = new URL(request.url());
      if (url.pathname === "/complex-analysis/action-plans" && request.method() === "GET") {
        listRequests.push(url.searchParams.get("offset") ?? "0");
      }
    });
    await page.getByRole("button", { name: "加载更多", exact: true }).click();
    await expect(firstPage).toHaveCount(21);
    evidence.pagination_page_counts.push(await firstPage.count());
    evidence.visible_plan_count = await firstPage.count();
    expect(listRequests).toEqual(["20"]);
    const list = await browserActionPlanRequest(page, "/complex-analysis/action-plans?offset=20&limit=20", "GET");
    expect(list.status).toBe(200);
    expect(list.body.items).toHaveLength(1);
    expect(list.body.next_offset).toBeNull();
  } else if (workflow.mode === "cross_user_denied") {
    const list = await browserActionPlanRequest(page, "/complex-analysis/action-plans?offset=0&limit=20", "GET");
    expect(list.status).toBe(200);
    const ownIds = Array.isArray(list.body.items) ? list.body.items.map((item: Record<string, unknown>) => String(item.action_plan_id)) : [];
    expect(ownIds).toContain(actionPlanId);
    expect(ownIds).not.toContain(foreignActionPlanId);
    evidence.visible_plan_count = ownIds.length;
    const getDenied = await readActionPlan(page, foreignActionPlanId);
    const updateDenied = await browserActionPlanRequest(page, `/complex-analysis/action-plans/${encodeURIComponent(foreignActionPlanId)}`, "PUT", { title: "E2E unauthorized edit" });
    const deleteDenied = await browserActionPlanRequest(page, `/complex-analysis/action-plans/${encodeURIComponent(foreignActionPlanId)}`, "DELETE");
    evidence.cross_user_statuses = [getDenied.status, updateDenied.status, deleteDenied.status];
    expect(evidence.cross_user_statuses).toEqual([404, 404, 404]);
    await expect(page.locator(".el-menu-item").filter({ hasText: workflow.expected_title! })).toBeVisible();
  } else {
    if (!actionPlanId || !workflow.expected_title) throw new Error("Action plan workflow requires an id and expected title");
    await expect(page.locator(".el-menu-item").filter({ hasText: workflow.expected_title })).toBeVisible();
    await selectActionPlanByTitle(page, workflow.expected_title);
    const before = await readActionPlan(page, actionPlanId);
    expect(before.status).toBe(200);
    evidence.initial_version = Number(before.body.version);
    evidence.complex_plan_id = String(before.body.complex_plan_id ?? "") || null;

    if (workflow.mode === "read_evidence") {
      const content = before.body.content as Record<string, any>;
      evidence.trace_linked = typeof content.trace_id === "string" && content.trace_id.length > 0;
      evidence.evidence_linked = Array.isArray(content.evidence_ids) && content.evidence_ids.length > 0;
      expect(evidence.trace_linked).toBe(true);
      expect(evidence.evidence_linked).toBe(true);
      await expect(page.locator("textarea")).toHaveValue(new RegExp(String(content.trace_id)));
      const parent = await browserActionPlanRequest(page, `/complex-analysis/plans/${encodeURIComponent(evidence.complex_plan_id!)}`, "GET");
      expect(parent.status).toBe(200);
      expect(parent.body.status).toBe("completed");
      expect(parent.body.trace_id).toBe(content.trace_id);
    } else if (workflow.mode === "edit_title") {
      await page.locator(".el-form-item").filter({ hasText: "标题" }).locator("input").fill("E2E 人工审核后的行动草案");
      const updatePromise = page.waitForResponse((response) =>
        new URL(response.url()).pathname.endsWith(`/action-plans/${encodeURIComponent(actionPlanId)}`)
        && response.request().method() === "PUT");
      await page.getByRole("button", { name: "保存", exact: true }).click();
      const update = await updatePromise;
      evidence.mutation_status = update.status();
      expect(update.status()).toBe(200);
    } else if (workflow.mode === "edit_content") {
      const content = { ...(before.body.content as Record<string, any>), e2e_review_marker: "human_reviewed" };
      await page.locator(".el-form-item").filter({ hasText: "内容（可编辑 JSON）" }).locator("textarea").fill(JSON.stringify(content, null, 2));
      const updatePromise = page.waitForResponse((response) =>
        new URL(response.url()).pathname.endsWith(`/action-plans/${encodeURIComponent(actionPlanId)}`)
        && response.request().method() === "PUT");
      await page.getByRole("button", { name: "保存", exact: true }).click();
      const update = await updatePromise;
      evidence.mutation_status = update.status();
      expect(update.status()).toBe(200);
    } else if (workflow.mode === "invalid_json") {
      await page.locator(".el-form-item").filter({ hasText: "内容（可编辑 JSON）" }).locator("textarea").fill("{\"actions\":");
      await page.getByRole("button", { name: "保存", exact: true }).click();
      await expect(page.locator(".el-message--error")).toBeVisible();
      await expect(page.locator(".el-message--error")).toContainText("JSON");
      evidence.final_version = Number((await readActionPlan(page, actionPlanId)).body.version);
      expect(evidence.final_version).toBe(evidence.initial_version);
      expect(evidence.mutation_status).toBeNull();
    } else if (workflow.mode === "accept" || workflow.mode === "archive") {
      const label = workflow.mode === "accept" ? "已采纳" : "已归档";
      await page.locator(".el-form-item").filter({ hasText: "状态" }).locator(".el-select").click();
      await page.locator(".el-select-dropdown__item:visible").filter({ hasText: label }).click();
      const updatePromise = page.waitForResponse((response) =>
        new URL(response.url()).pathname.endsWith(`/action-plans/${encodeURIComponent(actionPlanId)}`)
        && response.request().method() === "PUT");
      await page.getByRole("button", { name: "保存", exact: true }).click();
      const update = await updatePromise;
      evidence.mutation_status = update.status();
      expect(update.status()).toBe(200);
    } else if (workflow.mode === "delete_cancel" || workflow.mode === "delete_confirm") {
      let deleteRequestCount = 0;
      const onDelete = (request: import("@playwright/test").Request) => {
        if (request.method() === "DELETE" && new URL(request.url()).pathname.endsWith(`/action-plans/${encodeURIComponent(actionPlanId)}`)) deleteRequestCount += 1;
      };
      page.on("request", onDelete);
      await page.getByRole("button", { name: "删除", exact: true }).click();
      const box = page.locator(".el-message-box:visible");
      await expect(box).toBeVisible();
      await expect(box).toContainText("删除后无法恢复该行动草案");
      if (workflow.mode === "delete_cancel") {
        await box.getByRole("button", { name: "Cancel", exact: true }).click();
        await expect(box).toBeHidden();
        const readback = await readActionPlan(page, actionPlanId);
        evidence.delete_cancelled_without_write = deleteRequestCount === 0 && readback.status === 200;
        evidence.final_version = Number(readback.body.version);
        expect(evidence.delete_cancelled_without_write).toBe(true);
        expect(evidence.final_version).toBe(evidence.initial_version);
      } else {
        const deletePromise = page.waitForResponse((response) =>
          new URL(response.url()).pathname.endsWith(`/action-plans/${encodeURIComponent(actionPlanId)}`)
          && response.request().method() === "DELETE");
        await box.getByRole("button", { name: "OK", exact: true }).click();
        const deleted = await deletePromise;
        evidence.mutation_status = deleted.status();
        expect(deleted.status()).toBe(200);
        evidence.deleted = Boolean((await deleted.json() as Record<string, any>).deleted);
        expect(evidence.deleted).toBe(true);
        expect((await readActionPlan(page, actionPlanId)).status).toBe(404);
      }
      page.off("request", onDelete);
    }
    if (!["invalid_json", "delete_cancel", "delete_confirm"].includes(workflow.mode)) {
      const after = await readActionPlan(page, actionPlanId);
      expect(after.status).toBe(200);
      evidence.final_version = Number(after.body.version);
      evidence.final_status = String(after.body.status);
      if (workflow.mode === "edit_title") expect(after.body.title).toBe("E2E 人工审核后的行动草案");
      if (workflow.mode === "edit_content") {
        expect(after.body.content.e2e_review_marker).toBe("human_reviewed");
        expect(after.body.content.trace_id).toBe(before.body.content.trace_id);
        expect(after.body.content.evidence_ids).toEqual(before.body.content.evidence_ids);
      }
      if (workflow.mode === "accept") expect(after.body.status).toBe("accepted");
      if (workflow.mode === "archive") expect(after.body.status).toBe("archived");
      if (["edit_title", "edit_content", "accept", "archive"].includes(workflow.mode)) {
        expect(evidence.final_version).toBe(evidence.initial_version! + 1);
      } else {
        expect(evidence.final_version).toBe(evidence.initial_version);
      }
    }
  }
  evidence.no_execution_request = executionRequests.length === 0;
  expect(evidence.no_execution_request).toBe(true);
  return evidence;
}

async function browserQualityRequest(
  page: Page,
  path: string,
  method: "GET" | "POST",
): Promise<{ status: number; body: Record<string, any> }> {
  return page.evaluate(async ({ requestPath, requestMethod }) => {
    const stored = JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}") as { bearerToken?: string; actor?: string };
    if (!stored.bearerToken) throw new Error("Quality readback requires the authenticated session");
    const response = await fetch(requestPath, {
      method: requestMethod,
      headers: { Authorization: `Bearer ${stored.bearerToken}`, "X-Actor-ID": stored.actor ?? "" },
    });
    const raw = await response.text();
    let parsed: Record<string, any> = {};
    try { parsed = JSON.parse(raw) as Record<string, any>; } catch { parsed = { detail: raw.slice(0, 160) }; }
    return { status: response.status, body: parsed };
  }, { requestPath: path, requestMethod: method });
}

async function runQualityWorkflow(page: Page, workflow: QualityWorkflow): Promise<QualityEvidence> {
  const evidence: QualityEvidence = {
    status: "verified", mode: workflow.mode, route_path: null, quality_status: null,
    latest_scope: null, history_count_before: null, history_count_after: null,
    evaluation_status: null, evaluation_rows: null, evaluation_request_count: 0,
    capture_statuses: [], denied_statuses: [], trace_bucket_count: null,
    selected_bucket: null, no_unexpected_writes: true,
  };
  const evaluationPaths: string[] = [];
  const captureMethods: string[] = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname === "/evaluations") evaluationPaths.push(request.method());
    if (url.pathname === "/api/quality/snapshots") captureMethods.push(request.method());
  });

  if (workflow.mode === "analyst_route_denied") {
    await page.goto("/quality");
    await expect(page).toHaveURL(/\/chat$/);
    await expect(page.getByRole("heading", { name: "项目质量仪表盘", exact: true })).toHaveCount(0);
    evidence.route_path = new URL(page.url()).pathname;
    expect(evaluationPaths).toEqual([]);
    expect(captureMethods).toEqual([]);
  } else if (workflow.mode === "analyst_api_denied") {
    await page.goto("/chat");
    const result = await browserQualityRequest(page, "/api/quality", "GET");
    evidence.quality_status = result.status;
    evidence.denied_statuses.push(result.status);
    expect(result.status).toBe(403);
  } else if (workflow.mode === "analyst_snapshot_denied") {
    await page.goto("/chat");
    const result = await browserQualityRequest(page, "/api/quality/snapshots", "POST");
    evidence.capture_statuses.push(result.status);
    evidence.denied_statuses.push(result.status);
    expect(result.status).toBe(403);
  } else {
    const initialDashboard = page.waitForResponse((response) =>
      new URL(response.url()).pathname === "/api/quality" && response.request().method() === "GET");
    await page.goto("/quality");
    await expect(page.getByRole("heading", { name: "项目质量仪表盘", exact: true })).toBeVisible();
    const initialResponse = await initialDashboard;
    evidence.quality_status = initialResponse.status();
    expect(initialResponse.status()).toBe(200);
    evidence.route_path = new URL(page.url()).pathname;

    const before = await browserQualityRequest(page, "/api/quality", "GET");
    expect(before.status).toBe(200);
    evidence.latest_scope = String(before.body.latest?.scope ?? "") || null;
    evidence.history_count_before = Array.isArray(before.body.history) ? before.body.history.length : null;

    if (workflow.mode === "platform_read") {
      expect(evidence.latest_scope).toBe("platform");
      await expect(page.getByRole("button", { name: "保存当前快照", exact: true })).toBeVisible();
      const after = await browserQualityRequest(page, "/api/quality", "GET");
      evidence.history_count_after = Array.isArray(after.body.history) ? after.body.history.length : null;
      expect(evidence.history_count_after).toBe(evidence.history_count_before);
      expect(captureMethods).toEqual([]);
    } else if (workflow.mode === "platform_capture" || workflow.mode === "owner_capture") {
      const expectedScope = workflow.mode === "platform_capture" ? "platform" : "tenant";
      expect(evidence.latest_scope).toBe(expectedScope);
      const capture = page.waitForResponse((response) =>
        new URL(response.url()).pathname === "/api/quality/snapshots" && response.request().method() === "POST");
      await page.getByRole("button", { name: "保存当前快照", exact: true }).click();
      evidence.capture_statuses.push((await capture).status());
      await expect(page.getByText("已保存当前版本快照", { exact: true })).toBeVisible();
      const after = await browserQualityRequest(page, "/api/quality", "GET");
      expect(after.status).toBe(200);
      evidence.history_count_after = Array.isArray(after.body.history) ? after.body.history.length : null;
      expect(evidence.history_count_after).toBe((evidence.history_count_before ?? 0) + 1);
      expect(evidence.capture_statuses).toEqual([200]);
    } else if (workflow.mode === "platform_capture_twice") {
      expect(evidence.latest_scope).toBe("platform");
      const initialCount = evidence.history_count_before ?? 0;
      for (let index = 0; index < 2; index += 1) {
        const capture = page.waitForResponse((response) =>
          new URL(response.url()).pathname === "/api/quality/snapshots" && response.request().method() === "POST");
        await page.getByRole("button", { name: "保存当前快照", exact: true }).click();
        evidence.capture_statuses.push((await capture).status());
        await expect(page.getByText("已保存当前版本快照", { exact: true })).toBeVisible();
      }
      const after = await browserQualityRequest(page, "/api/quality", "GET");
      evidence.history_count_after = Array.isArray(after.body.history) ? after.body.history.length : null;
      expect(evidence.history_count_after).toBe(initialCount + 2);
      expect(evidence.capture_statuses).toEqual([200, 200]);
    } else if (workflow.mode === "platform_bucket") {
      expect(evidence.latest_scope).toBe("platform");
      evidence.trace_bucket_count = await page.locator(".section").filter({ hasText: "Trace 统计分代" }).locator(".el-select-dropdown__item").count().catch(() => 0);
      const selector = page.locator(".section").filter({ hasText: "Trace 统计分代" }).locator(".el-select");
      await selector.click();
      const legacy = page.locator(".el-select-dropdown__item:visible").filter({ hasText: "legacy" });
      await expect(legacy).toHaveCount(1);
      evidence.trace_bucket_count = await page.locator(".el-select-dropdown__item:visible").count();
      await legacy.click();
      evidence.selected_bucket = await selector.innerText();
      expect(evidence.selected_bucket).toContain("legacy");
      await expect(page.locator(".collection-status")).toContainText("2/2 条 Trace 已进入终态");
    } else if (workflow.mode === "platform_evaluations") {
      expect(evidence.latest_scope).toBe("platform");
      const result = await browserQualityRequest(page, "/evaluations", "GET");
      evidence.evaluation_status = result.status;
      evidence.evaluation_rows = Array.isArray(result.body.items) ? result.body.items.length : null;
      expect(result.status).toBe(200);
      expect(evaluationPaths).toContain("GET");
      await expect(page.getByText("Evaluation Evidence", { exact: true })).toBeVisible();
      expect(evaluationPaths.filter((method) => method !== "GET")).toEqual([]);
    } else if (workflow.mode === "owner_read") {
      expect(evidence.latest_scope).toBe("tenant");
      expect(evidence.history_count_before).toBe(0);
      expect((before.body.history ?? []).every((item: Record<string, unknown>) => item.scope === "tenant")).toBe(true);
      expect(evaluationPaths).toEqual([]);
      await expect(page.getByText("评测运行明细仅平台管理员可查看", { exact: false })).toBeVisible();
      await expect(page.getByText("Evaluation Evidence", { exact: true })).toHaveCount(0);
    }
  }

  evidence.evaluation_request_count = evaluationPaths.length;
  evidence.no_unexpected_writes = captureMethods.filter((method) => method === "POST").length === evidence.capture_statuses.length
    && evaluationPaths.every((method) => method === "GET");
  expect(evidence.no_unexpected_writes).toBe(true);
  return evidence;
}

function targetQueryPath(workflow: MetricTargetWorkflow, target: TargetSpec, sellerId: string, version?: string): string {
  const params = new URLSearchParams({
    metric_name: target.metric_name,
    period_start: target.period_start,
    period_end: target.period_end,
  });
  if (workflow.scope_type === "seller" || sellerId) params.set("seller_id", sellerId);
  if (version) params.set("version", version);
  return `/analysis/targets?${params.toString()}`;
}

function targetBody(target: TargetSpec, actor: string, sellerId: string): Record<string, unknown> {
  return {
    period_start: target.period_start,
    period_end: target.period_end,
    target_value: target.target_value,
    version: target.version,
    effective_at: target.effective_at,
    created_by: actor,
    seller_id: sellerId,
  };
}

function targetEvidenceRows(items: unknown): Array<Record<string, unknown>> {
  if (!Array.isArray(items)) return [];
  return items.map((item: Record<string, any>) => ({
    metric_name: String(item.metric_name ?? ""),
    seller_id: item.seller_id == null ? null : String(item.seller_id),
    period_start: String(item.period_start ?? ""),
    period_end: String(item.period_end ?? ""),
    target_value: item.target_value == null ? null : Number(item.target_value),
    version: String(item.version ?? ""),
    effective_at: String(item.effective_at ?? ""),
    scope: String(item.scope ?? ""),
  }));
}

async function runMetricTargetWorkflow(page: Page, workflow: MetricTargetWorkflow): Promise<MetricTargetEvidence> {
  const evidence: MetricTargetEvidence = {
    status: "verified",
    mode: workflow.mode,
    scope_type: workflow.scope_type,
    metric_name: workflow.metric_name ?? workflow.targets?.[0]?.metric_name ?? null,
    visible_metric_row_count: null,
    metric_row_text: null,
    unit: null,
    write_statuses: [],
    target_list_status: null,
    target_rows: [],
    filtered_row_count: null,
    deleted_version: null,
    delete_status: null,
    rejected_probes: [],
  };

  await page.goto("/metrics");
  await expect(page.getByRole("heading", { name: "指标与目标", exact: true })).toBeVisible();
  const metricsTable = page.locator(".metrics-page .el-table").first();
  await expect(metricsTable).toBeVisible();
  const definitionSearch = page.getByPlaceholder("搜索指标名称、编码或说明");

  if (workflow.mode === "definition_search" || workflow.mode === "unsupported_metric") {
    const searchTerm = workflow.search ?? workflow.metric_name ?? "";
    expect(searchTerm).not.toBe("");
    await definitionSearch.fill(searchTerm);
    const rows = metricsTable.locator(".el-table__body tbody tr");
    if (workflow.mode === "definition_search") {
      const metricName = workflow.metric_name ?? "";
      const matchingIndexes = await rows.evaluateAll((items, expectedName) => items
        .map((row, index) => ({
          index,
          code: row.querySelector("td")?.innerText.trim() ?? "",
        }))
        .filter((item) => item.code === expectedName)
        .map((item) => item.index), metricName);
      expect(matchingIndexes, `the metric-code column must contain exactly one ${metricName} row`).toHaveLength(1);
      const matchingRow = rows.nth(matchingIndexes[0]);
      const rowText = (await matchingRow.innerText()).trim();
      evidence.visible_metric_row_count = await rows.count();
      evidence.metric_row_text = rowText;
      evidence.metric_name = metricName;
      expect(rowText).toContain(metricName);
      if (workflow.expected_unit) expect(rowText).toContain(workflow.expected_unit);
      if (workflow.expected_description_contains) expect(rowText).toContain(workflow.expected_description_contains);
      if (workflow.mode === "definition_search") return evidence;
    } else {
      const metricName = workflow.metric_name ?? "";
      const matchingIndexes = await rows.evaluateAll((items, expectedName) => items
        .map((row, index) => ({ index, code: row.querySelector("td")?.innerText.trim() ?? "" }))
        .filter((item) => item.code === expectedName)
        .map((item) => item.index), metricName);
      expect(matchingIndexes, `the seller view must show one explicit capability state for ${metricName}`).toHaveLength(1);
      const unsupportedRow = rows.nth(matchingIndexes[0]);
      const rowText = (await unsupportedRow.innerText()).trim();
      await expect(unsupportedRow).toContainText("此范围不可用");
      evidence.visible_metric_row_count = 1;
      evidence.metric_row_text = rowText;
      evidence.metric_name = workflow.metric_name ?? null;
      const probeTarget = workflow.targets?.[0];
      expect(probeTarget, "unsupported-metric case needs one direct boundary probe").toBeTruthy();
      await page.getByRole("tab", { name: "目标管理", exact: true }).click();
      const sellerId = process.env.E2E_SELLER_ID ?? "";
      const actor = await page.evaluate(() => {
        const profile = JSON.parse(localStorage.getItem("bizinsight.debug-profile.v1") ?? "{}") as { id?: string };
        return profile.id ?? "";
      });
      expect(sellerId).not.toBe("");
      expect(actor).not.toBe("");
      const probe = await browserMetricTargetRequest(
        page,
        `/analysis/targets/${encodeURIComponent(probeTarget!.metric_name)}`,
        "PUT",
        targetBody(probeTarget!, actor, sellerId),
      );
      const error = String(probe.body.detail ?? probe.text).slice(0, 300);
      evidence.rejected_probes.push({ kind: "unsupported_metric", status: probe.status, error });
      expect(probe.status).toBe(400);
      expect(error).toContain(workflow.expected_error_contains ?? `Unsupported metric: ${probeTarget!.metric_name}`);
      const readback = await browserMetricTargetRequest(page, targetQueryPath(workflow, probeTarget!, sellerId, probeTarget!.version), "GET");
      evidence.target_list_status = readback.status;
      expect(readback.status).toBe(200);
      expect(Number(readback.body.total ?? 0)).toBe(0);
      return evidence;
    }
  }

  const targetTab = page.getByRole("tab", { name: "目标管理", exact: true });
  await targetTab.click();
  const form = page.locator(".target-form");
  await expect(form).toBeVisible();
  const sellerId = process.env.E2E_SELLER_ID ?? "";
  const actor = await page.evaluate(() => {
    const profile = JSON.parse(localStorage.getItem("bizinsight.debug-profile.v1") ?? "{}") as { id?: string };
    return profile.id ?? "";
  });
  expect(sellerId, "the isolated synthetic seller must be configured").not.toBe("");
  expect(actor, "the isolated actor must be configured").not.toBe("");

  const sellerItem = form.locator(".el-form-item").filter({ hasText: "店铺" });
  const sellerSelect = sellerItem.locator(".el-select");
  const targetSellerResponse = await browserMetricTargetRequest(
    page,
    `/data-explorer/sellers?query=${encodeURIComponent(sellerId)}`,
    "GET",
  );
  expect(targetSellerResponse.status).toBe(200);
  const targetSellerOptions = Array.isArray(targetSellerResponse.body.items)
    ? targetSellerResponse.body.items as Array<Record<string, any>>
    : [];
  const selectedTargetSeller = targetSellerOptions.find((item) => String(item.seller_id) === sellerId);
  expect(selectedTargetSeller, "authorized seller search must include the synthetic seller").toBeTruthy();
  const expectedSellerLabel = workflow.scope_type === "seller"
    ? sellerId
    : String(selectedTargetSeller!.label ?? sellerId);
  await expect(sellerSelect).toContainText(expectedSellerLabel);

  const targets = workflow.targets ?? [];
  const invalidTarget = workflow.mode === "invalid_period" ? targets[0] : undefined;
  const allowedWriteModes = new Set(["write_targets", "filter_target", "delete_target", "invalid_period"]);
  if (allowedWriteModes.has(workflow.mode)) expect(targets.length).toBeGreaterThan(0);
  const metadata = await browserMetricTargetRequest(page, `/analysis/semantic/metrics?scope=${workflow.scope_type}`, "GET");
  expect(metadata.status).toBe(200);
  const metricDefinitions = Array.isArray(metadata.body) ? metadata.body as Array<Record<string, any>> : [];

  for (const target of targets) {
    const metric = metricDefinitions.find((item) => String(item.metric_name) === target.metric_name);
    expect(metric, `metric ${target.metric_name} must be present in the ${workflow.scope_type} registry`).toBeTruthy();
    const metricLabel = String(metric!.display_name ?? target.metric_name);
    const metricSelect = form.locator(".el-form-item").filter({ hasText: "指标" }).locator(".el-select");
    await metricSelect.click();
    const metricOption = page.locator(".el-select-dropdown:visible .el-select-dropdown__item").filter({ hasText: metricLabel });
    await expect(metricOption.first()).toBeVisible();
    await metricOption.first().click();

    const periodInputs = form.locator(".el-form-item").filter({ hasText: "目标周期" }).locator(".el-date-editor input");
    await periodInputs.nth(0).fill(target.period_start);
    await periodInputs.nth(0).press("Enter");
    await periodInputs.nth(1).fill(target.period_end);
    await periodInputs.nth(1).press("Enter");
    const valueInput = form.locator(".el-form-item").filter({ hasText: "目标值" }).locator("input");
    await valueInput.fill(String(target.target_value));
    await valueInput.press("Tab");
    await form.locator(".el-form-item").filter({ hasText: "版本" }).locator("input").fill(target.version);
    const effectiveInput = form.locator(".el-form-item").filter({ hasText: "生效日期" }).locator(".el-date-editor input");
    await effectiveInput.fill(target.effective_at);
    await effectiveInput.press("Enter");

    const responsePromise = page.waitForResponse((response) =>
      new URL(response.url()).pathname === `/analysis/targets/${encodeURIComponent(target.metric_name)}`
        && response.request().method() === "PUT",
    );
    await form.getByRole("button", { name: "保存目标", exact: true }).click();
    const response = await responsePromise;
    evidence.write_statuses.push({ metric_name: target.metric_name, version: target.version, status: response.status() });
    if (workflow.mode === "invalid_period") {
      expect(response.status()).toBe(400);
      const body = await response.json().catch(() => ({})) as Record<string, any>;
      const error = String(body.detail ?? "");
      evidence.rejected_probes.push({ kind: "invalid_period", status: response.status(), error });
      expect(error).toContain(workflow.expected_error_contains ?? "target period_end must be on or after period_start");
      await expect(page.locator(".el-message--error").last()).toContainText(workflow.expected_error_contains ?? "target period_end must be on or after period_start");
    } else {
      expect(response.status()).toBe(200);
      await expect(page.locator(".el-message--success").last()).toContainText("目标已保存");
    }
  }

  if (workflow.mode === "invalid_period" && workflow.unauthorized_seller_id) {
    const probeTarget = invalidTarget!;
    const unauthorized = await browserMetricTargetRequest(
      page,
      `/analysis/targets/${encodeURIComponent(probeTarget.metric_name)}`,
      "PUT",
      targetBody({ ...probeTarget, period_start: "2018-08-01", period_end: "2018-08-31", version: `${probeTarget.version}-unauthorized` }, actor, workflow.unauthorized_seller_id),
    );
    const error = String(unauthorized.body.detail ?? unauthorized.text).slice(0, 300);
    evidence.rejected_probes.push({ kind: "unauthorized_seller", status: unauthorized.status, error });
    expect(unauthorized.status).toBe(400);
    expect(error).toContain("authorized seller");
    const forbiddenRead = await browserMetricTargetRequest(
      page,
      targetQueryPath(workflow, probeTarget, workflow.unauthorized_seller_id, `${probeTarget.version}-unauthorized`),
      "GET",
    );
    evidence.rejected_probes.push({
      kind: "unauthorized_seller_read",
      status: forbiddenRead.status,
      error: String(forbiddenRead.body.detail ?? forbiddenRead.text).slice(0, 300),
    });
    expect(forbiddenRead.status).toBe(403);
  }

  const queryTarget = targets[0];
  if (queryTarget && workflow.mode !== "invalid_period") {
    if (workflow.mode === "filter_target") {
      const filter = page.locator(".target-list-card .target-filters");
      await filter.locator("input").last().fill(workflow.filter_version ?? queryTarget.version);
      await filter.getByRole("button", { name: "查询", exact: true }).click();
      const rows = page.locator(".target-list-card .el-table__body tbody tr");
      await expect(rows).toHaveCount(1);
      await expect(rows.first()).toContainText(workflow.filter_version ?? queryTarget.version);
      evidence.filtered_row_count = await rows.count();
    }

    if (workflow.mode === "delete_target") {
      const row = page.locator(".target-list-card .el-table__body tbody tr").filter({ hasText: queryTarget.version });
      await expect(row).toHaveCount(1);
      await row.getByRole("button", { name: "删除", exact: true }).click();
      const dialog = page.locator(".el-message-box:visible");
      await expect(dialog).toBeVisible();
      const responsePromise = page.waitForResponse((response) =>
        new URL(response.url()).pathname === `/analysis/targets/${encodeURIComponent(queryTarget.metric_name)}`
          && response.request().method() === "DELETE",
      );
      await dialog.getByRole("button", { name: /^(OK|确定)$/ }).click();
      const response = await responsePromise;
      expect(response.status()).toBe(200);
      const body = await response.json() as Record<string, any>;
      expect(body.deleted).toBe(true);
      evidence.deleted_version = queryTarget.version;
      evidence.delete_status = response.status();
    }

    const readback = await browserMetricTargetRequest(
      page,
      targetQueryPath(workflow, queryTarget, sellerId, workflow.mode === "filter_target" ? workflow.filter_version ?? queryTarget.version : undefined),
      "GET",
    );
    evidence.target_list_status = readback.status;
    expect(readback.status).toBe(200);
    const rows = Array.isArray(readback.body.items) ? readback.body.items : [];
    evidence.target_rows = targetEvidenceRows(rows);
    if (workflow.mode === "delete_target") {
      expect(rows).toEqual([]);
    } else if (workflow.mode === "filter_target") {
      expect(rows).toHaveLength(1);
      expect(String(rows[0].version)).toBe(workflow.filter_version ?? queryTarget.version);
    } else {
      for (const target of targets) {
        const row = rows.find((item: Record<string, any>) => String(item.version) === target.version);
        expect(row, `target version ${target.version} must be persisted`).toBeTruthy();
        expect(Number(row.target_value)).toBe(target.target_value);
      }
    }
  }

  if (workflow.mode === "invalid_period" && invalidTarget) {
    const readback = await browserMetricTargetRequest(page, targetQueryPath(workflow, invalidTarget, sellerId, invalidTarget.version), "GET");
    evidence.target_list_status = readback.status;
    expect(readback.status).toBe(200);
    expect(Number(readback.body.total ?? 0)).toBe(0);
    evidence.target_rows = targetEvidenceRows(readback.body.items);
  }
  return evidence;
}

async function runComplexWorkflow(page: Page, workflow: ComplexWorkflow, testInfo: TestInfo, question: string): Promise<ComplexEvidence> {
  const sellerId = process.env.E2E_SELLER_ID ?? "";
  expect(sellerId, "the isolated seller identity must be configured").not.toBe("");
  const interactions: Array<{ kind: string; status: string }> = [];
  const planIds: string[] = [];

  if (workflow.select_template) {
    const selectionCard = page.locator(".complex-plan").filter({ hasText: "选择复杂分析模板" });
    await expect(selectionCard).toBeVisible();
    await expect(selectionCard.getByRole("button", { name: workflow.select_template, exact: true })).toBeVisible();
    const selected = await selectComplexTemplate(page, workflow.select_template);
    expect(selected.status).toBe(200);
    const selectedSummary = summarize(selected.events);
    expect(selectedSummary.event).toBe("suspended");
    expect(selectedSummary.clarification_status).toBe("complex_scope_required");
    expect(selectedSummary.trace_id).toBeTruthy();
    interactions.push({ kind: "template_selection", status: selectedSummary.clarification_status ?? "missing" });
    const selectedTrace = await readTrace(page, selectedSummary.trace_id);
    expect(selectedTrace.state_snapshot?.query_results ?? []).toEqual([]);
    expect(selectedTrace.state_snapshot?.tool_calls ?? []).toEqual([]);
  }

  const scopeCard = page.locator(".complex-plan").filter({ hasText: "需要补充分析范围" });
  await expect(scopeCard).toBeVisible();
  const sellerList = page.waitForResponse((response) =>
    response.url().includes("/data-explorer/sellers?query=") && response.request().method() === "GET",
  );
  await scopeCard.getByRole("button", { name: "选择店铺和日期范围", exact: true }).click();
  const scopeDrawer = page.locator(".el-drawer").filter({ hasText: "修改本会话范围" });
  await expect(scopeDrawer).toBeVisible();
  const sellerResponse = await sellerList;
  expect(sellerResponse.status()).toBe(200);
  const sellerBody = await sellerResponse.json() as { items: Array<{ seller_id: string }> };
  expect(sellerBody.items.map((item) => item.seller_id)).toEqual([sellerId]);
  await scopeDrawer.locator(".el-select").first().click();
  const sellerOptions = page.locator(".el-select-dropdown:visible .el-select-dropdown__item");
  await expect(sellerOptions).toHaveCount(1);
  // The UI truncates seller IDs in option labels; the API response above
  // already verifies the exact authorized seller returned for this tenant.
  await sellerOptions.first().click();

  const dates = scopeDrawer.locator(".el-date-editor input");
  await dates.nth(0).fill(workflow.start_date);
  await dates.nth(0).press("Enter");
  await dates.nth(1).fill(workflow.end_date);
  await dates.nth(1).press("Enter");
  await scopeDrawer.getByRole("button", { name: "应用到本会话", exact: true }).click();

  // The current UI saves scope to the chat session, then requires the user to
  // resend the original request. Plan creation remains behind that fresh route
  // evaluation and never happens from the scope editor itself.
  const planResponse = await sendChatTurn(page, "chat", question);
  expect(planResponse.status).toBe(200);
  const planTerminal = [...planResponse.events].reverse().find((event) => ["completed", "failed", "suspended"].includes(String(event.event)));
  expect(planTerminal?.event, `scope-resubmitted request should create a pending plan: ${JSON.stringify(planTerminal)}`).toBe("suspended");
  expect(planTerminal?.clarification?.status).toBe("confirmation_required");
  expect(planTerminal?.error).not.toBe(true);
  const responsePlan = planTerminal?.complex_plan as Record<string, any> | undefined;
  expect(responsePlan, "completed route response must include the pending complex plan").toBeTruthy();
  if (!responsePlan) throw new Error("completed route response omitted the pending complex plan");
  let plan = responsePlan;
  expect(plan.plan_id).toBeTruthy();
  expect(plan.template_name).toBe(workflow.template_name);
  expect(plan.status).toBe("pending_confirmation");
  expect(plan.scope?.seller_id).toBe(sellerId);
  expect(plan.params?.start_date).toBe(workflow.start_date);
  expect(plan.params?.end_date).toBe(workflow.end_date);
  planIds.push(String(plan.plan_id));
  interactions.push({ kind: "scope_applied_and_plan_created", status: String(plan.status) });
  await testInfo.attach("complex-plan-created.json", {
    body: Buffer.from(JSON.stringify({ plan_id: plan.plan_id, template_name: plan.template_name, status: plan.status })),
    contentType: "application/json",
  });

  const pendingRead = await readComplexPlan(page, String(plan.plan_id));
  expect(pendingRead.status).toBe(200);
  expect(pendingRead.body.status).toBe("pending_confirmation");
  const pendingPlanCard = page.locator(".complex-plan").filter({ hasText: "复杂分析计划（等待确认）" });
  await expect(pendingPlanCard).toBeVisible();

  if (workflow.revise_note) {
    const initialPlanId = String(plan.plan_id);
    await pendingPlanCard.getByRole("button", { name: "调整计划", exact: true }).click();
    const adjustment = page.locator(".complex-plan").filter({ hasText: "调整分析计划" });
    await expect(adjustment).toBeVisible();
    await adjustment.getByPlaceholder("例如：优先验证是否由某品类造成。备注只影响分析重点，不改变模板计算。").fill(workflow.revise_note);
    const revisePromise = page.waitForResponse((response) =>
      /\/complex-analysis\/plans\/[^/]+\/revise$/.test(new URL(response.url()).pathname)
        && response.request().method() === "POST",
    );
    await adjustment.getByRole("button", { name: "生成新版计划", exact: true }).click();
    const reviseResponse = await revisePromise;
    expect(reviseResponse.status()).toBe(200);
    plan = await reviseResponse.json() as Record<string, any>;
    expect(plan.revision).toBe(2);
    expect(plan.parent_plan_id).toBe(initialPlanId);
    expect(plan.preferences?.note).toBe(workflow.revise_note);
    planIds.push(String(plan.plan_id));
    const superseded = await readComplexPlan(page, initialPlanId);
    expect(superseded.status).toBe(200);
    expect(superseded.body.status).toBe("superseded");
    interactions.push({ kind: "plan_revision", status: String(plan.revision) });
  }

  const confirmStart = await page.evaluate(() => (window as Window & { __expandedSse?: unknown[] }).__expandedSse?.length ?? 0);
  const confirmPromise = page.waitForResponse((response) =>
    new URL(response.url()).pathname.endsWith(`/complex-analysis/plans/${encodeURIComponent(String(plan.plan_id))}/confirm`)
      && response.request().method() === "POST",
  );
  await pendingPlanCard
    .getByRole("button", { name: "确认执行", exact: true }).click();
  const confirmResponse = await confirmPromise;
  expect(confirmResponse.status()).toBe(200);
  await expect(page.locator(".thinking")).toHaveCount(0, { timeout: 12 * 60_000 });
  const execution = await waitForSse(page, confirmStart);
  expect(execution.status).toBe(200);
  const terminal = [...execution.events].reverse().find((event) => ["completed", "failed"].includes(String(event.event)));
  if (terminal?.event !== "completed") {
    const failedPlan = await readComplexPlan(page, String(plan.plan_id)).catch((error: unknown) => ({
      status: 0, body: { read_error: error instanceof Error ? error.message : String(error) },
    }));
    const executionAudits = await page.evaluate(async (id) => {
      const stored = JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}") as { bearerToken?: string };
      if (!stored.bearerToken) return { status: 0, body: { read_error: "missing_bearer_token" } };
      const response = await fetch(`/complex-analysis/executions?plan_id=${encodeURIComponent(id)}&limit=5`, {
        headers: { Authorization: `Bearer ${stored.bearerToken}` },
      });
      return { status: response.status, body: await response.json() as Record<string, any> };
    }, String(plan.plan_id)).catch((error: unknown) => ({
      status: 0, body: { read_error: error instanceof Error ? error.message : String(error) },
    }));
    await testInfo.attach("complex-execution-terminal.json", {
      body: Buffer.from(JSON.stringify({
        plan_id: plan.plan_id,
        event: terminal?.event ?? null,
        detail: terminal?.detail ?? null,
        trace_id: terminal?.trace_id ?? null,
        sequence_number: terminal?.sequence_number ?? null,
        persisted_plan: {
          status: failedPlan.status,
          plan_status: failedPlan.body.status ?? null,
          latest_execution: failedPlan.body.latest_execution ?? null,
        },
        execution_audits: {
          status: executionAudits.status,
          items: Array.isArray(executionAudits.body.items)
            ? executionAudits.body.items.map((item: Record<string, any>) => ({
              status: item.status ?? null,
              error_code: item.error_code ?? null,
              duration_ms: item.duration_ms ?? null,
              trace_id: item.trace_id ?? null,
            }))
            : [],
        },
      }, null, 2)),
      contentType: "application/json",
    });
  }
  expect(terminal?.event, `confirmed plan must complete; terminal=${JSON.stringify(terminal)}`).toBe("completed");
  expect(terminal?.error).not.toBe(true);
  const traceId = String(terminal?.trace_id ?? "");
  expect(traceId).not.toBe("");
  const presentation = terminal?.presentation && typeof terminal.presentation === "object" ? terminal.presentation : {};
  const analysis = Array.isArray(presentation.template_analyses) ? presentation.template_analyses[0] : null;
  expect(analysis?.template_name).toBe(workflow.template_name);
  const resultStatus = String(analysis?.result?.status ?? "missing");
  if (workflow.result_status) expect(resultStatus).toBe(workflow.result_status);

  const resultCard = page.locator(".analysis-result").last();
  await expect(resultCard).toBeVisible({ timeout: 60_000 });
  await expect(resultCard).toContainText(workflow.template_display_name);
  await expect(page.locator(".message-actions").last()).toContainText("Trace");

  const completedPlan = await readComplexPlan(page, String(plan.plan_id));
  expect(completedPlan.status).toBe(200);
  expect(completedPlan.body.status).toBe("completed");
  expect(completedPlan.body.latest_execution?.status).toBe("completed");
  expect(completedPlan.body.latest_execution?.trace_id).toBe(traceId);
  interactions.push({ kind: "confirmed_execution", status: String(completedPlan.body.status) });

  return {
    plan_ids: planIds,
    trace_id: traceId,
    template_name: workflow.template_name,
    plan_status: String(completedPlan.body.status),
    execution_status: String(completedPlan.body.latest_execution?.status ?? "missing"),
    result_status: resultStatus,
    interactions,
  };
}

test.afterEach(async ({ page }, testInfo) => {
  if (testInfo.status === "failed" || testInfo.status === "timedOut") {
    const masks = testInfo.title.includes("preferences") ? [page.locator(".preference-item code")] : [];
    await testInfo.attach("failure-screenshot.png", {
      body: await page.screenshot({ fullPage: true, animations: "disabled", mask: masks }).catch(() => Buffer.alloc(0)),
      contentType: "image/png",
    });
  }
});

for (const matrixCase of cases) {
  test(`expanded ${matrixCase.feature} ${matrixCase.case_id}`, async ({ page }, testInfo) => {
    test.setTimeout(15 * 60_000);
    await installSseCapture(page);
    const api5xx: string[] = [];
    const pageErrors: string[] = [];
    const chatRequests: ChatRequest[] = [];
    page.on("response", (response) => {
      if (response.status() >= 500) api5xx.push(`${response.status()} ${new URL(response.url()).pathname}`);
    });
    page.on("pageerror", (error) => pageErrors.push(error.message.slice(0, 300)));
    page.on("request", (request) => {
      if (!request.url().includes("/chat/stream") || request.method() !== "POST") return;
      try { chatRequests.push(request.postDataJSON() as typeof chatRequests[number]); } catch { chatRequests.push({}); }
    });

    await test.step("login with the isolated synthetic persona", () => login(page, matrixCase.persona));
    let reportEvidence: ReportEvidence | undefined;
    let metricTargetEvidence: MetricTargetEvidence | undefined;
    let knowledgeEvidence: KnowledgeEvidence | undefined;
    let preferenceEvidence: PreferenceEvidence | undefined;
    let actionPlanEvidence: ActionPlanEvidence | undefined;
    let qualityEvidence: QualityEvidence | undefined;
    if (matrixCase.entry === "reports") {
      expect(matrixCase.report_workflow, "report cases must declare report_workflow").toBeTruthy();
      expect(matrixCase.turns).toEqual([]);
      reportEvidence = await test.step("create/review/publish through the Reports page and verify API readback", () =>
        runReportWorkflow(page, matrixCase.report_workflow!));
    } else if (matrixCase.entry === "metrics") {
      expect(matrixCase.metric_target_workflow, "metric/target cases must declare metric_target_workflow").toBeTruthy();
      expect(matrixCase.turns).toEqual([]);
      metricTargetEvidence = await test.step("use the Metrics page and verify target state through the authenticated API", () =>
        runMetricTargetWorkflow(page, matrixCase.metric_target_workflow!));
    } else if (matrixCase.entry === "knowledge") {
      expect(matrixCase.knowledge_workflow, "knowledge cases must declare knowledge_workflow").toBeTruthy();
      knowledgeEvidence = await test.step("read the knowledge catalog and verify the read-only role contract", () =>
        runKnowledgeWorkflow(page, matrixCase.knowledge_workflow!));
      if (matrixCase.turns.length > 0) {
        await test.step("start an isolated conversation for the RAG retrieval case", () => startFreshChat(page));
      }
    } else if (matrixCase.entry === "preferences") {
      expect(matrixCase.preference_workflow, "preference cases must declare preference_workflow").toBeTruthy();
      expect(matrixCase.turns).toEqual([]);
      preferenceEvidence = await test.step("use the Preferences page and verify the authenticated API result", () =>
        runPreferenceWorkflow(page, matrixCase.preference_workflow!));
    } else if (matrixCase.entry === "action_plans") {
      expect(matrixCase.action_plan_workflow, "action-plan cases must declare action_plan_workflow").toBeTruthy();
      expect(matrixCase.turns).toEqual([]);
      actionPlanEvidence = await test.step("use the Action Plans page and verify owner-scoped API and database state", () =>
        runActionPlanWorkflow(page, matrixCase.action_plan_workflow!));
    } else if (matrixCase.entry === "quality") {
      expect(matrixCase.quality_workflow, "quality cases must declare quality_workflow").toBeTruthy();
      expect(matrixCase.turns).toEqual([]);
      qualityEvidence = await test.step("use the quality dashboard and verify its authorization and snapshot contract", () =>
        runQualityWorkflow(page, matrixCase.quality_workflow!));
    } else {
      await test.step("start an isolated conversation for this case", () => startFreshChat(page));
    }
    if (matrixCase.entry === "seller_scoped_chat") {
      await test.step("select and preview the granted seller/date scope in Explorer", async () => {
        if (!matrixCase.scope) throw new Error("seller_scoped_chat case is missing its date scope");
        await selectSellerScope(page, matrixCase.scope);
      });
    }

    const summaries: TurnSummary[] = [];
    for (let index = 0; !["reports", "metrics", "preferences", "action_plans", "quality"].includes(matrixCase.entry) && index < matrixCase.turns.length; index += 1) {
      const turn = matrixCase.turns[index];
      const response = await test.step(`chat turn ${index + 1}`, () => turn.clarification_candidate
        ? selectClarificationCandidate(page, turn.clarification_candidate)
        : sendChatTurn(page, matrixCase.entry, turn.question));
      expect(response.status, `turn ${index + 1} must use the real chat API`).toBe(200);
      const summary = summarize(response.events);
      summaries.push(summary);
      await attachCaseEvidence(testInfo, matrixCase, summaries, chatRequests, "in_progress", 0, 0, undefined, undefined, undefined, knowledgeEvidence);
      await assertTurn(page, summary, turn.expect, summaries.slice(0, -1));
    }

    let complexEvidence: ComplexEvidence | undefined;
    if (matrixCase.complex_workflow) {
      expect(matrixCase.entry).toBe("chat");
      expect(matrixCase.persona).toBe("seller_owner");
      expect(summaries).toHaveLength(1);
      complexEvidence = await test.step("validate scope, create a plan, and explicitly confirm execution", () =>
        runComplexWorkflow(page, matrixCase.complex_workflow!, testInfo, matrixCase.turns[0].question));
    }

    if (matrixCase.expected?.same_session) {
      const sessions = new Set(chatRequests.map((request) => request.chat_session_id).filter(Boolean));
      expect(chatRequests.length).toBe(matrixCase.turns.length + (matrixCase.complex_workflow ? 1 : 0) + (matrixCase.complex_workflow?.select_template ? 1 : 0));
      expect(sessions.size).toBe(1);
    }
    if (matrixCase.expected?.scope_must_not_widen) {
      const sellerId = process.env.E2E_SELLER_ID ?? "";
      expect(chatRequests).toHaveLength(matrixCase.turns.length);
      for (const request of chatRequests) {
        expect(request.use_current_explorer_context).toBe(true);
        expect(request.explorer_context?.seller_id).toBe(sellerId);
        expect(request.explorer_context?.start_date).toBe(matrixCase.scope?.start_date);
        expect(request.explorer_context?.end_date).toBe(matrixCase.scope?.end_date);
        expect(request.seller_scope_payload).toBeFalsy();
      }
    }
    expect(api5xx, `Unexpected API 5xx: ${api5xx.join("; ")}`).toEqual([]);
    expect(pageErrors, `Unhandled browser errors: ${pageErrors.join("; ")}`).toEqual([]);

    await attachCaseEvidence(testInfo, matrixCase, summaries, chatRequests, "passed", api5xx.length, pageErrors.length, complexEvidence, reportEvidence, metricTargetEvidence, knowledgeEvidence, preferenceEvidence, actionPlanEvidence, qualityEvidence);
    const screenshotMasks = matrixCase.preference_workflow ? [page.locator(".preference-item code")] : [];
    await testInfo.attach("final-browser-screen.png", {
      body: await page.screenshot({ fullPage: true, animations: "disabled", mask: screenshotMasks }),
      contentType: "image/png",
    });
  });
}
