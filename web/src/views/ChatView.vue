<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { DocumentCopy } from "@element-plus/icons-vue";
import { api, streamChat, streamComplexPlan } from "@/api";
import { isSseInvestigationProgressEvent, isSseNodeCompletedEvent, isSseNodeStartedEvent, isSseStartedEvent } from "../generated/sse-events";
import { formatInvestigationStopReason } from "../utils/format";
import AnalysisResultCard from "@/components/AnalysisResultCard.vue";
import MarkdownContent from "@/components/MarkdownContent.vue";
import ScopeTag from "@/components/ScopeTag.vue";
import TaskContextCard from "@/components/TaskContextCard.vue";
import { clarificationMessage, clarificationTitle } from "@/utils/clarificationCopy";
import { useAppStore } from "@/stores/app";
import type { AnalysisPresentation, ComplexPlan, DebugTurn, ExplorerQuery, MetricDefinition, SellerOption } from "@/types";
import { DEFAULT_ANALYSIS_RANGE, validateExplorerQuery } from "@/utils/scope";

const store = useAppStore();
const question = ref("");
const working = ref(false);
const pendingMessage = ref("");
const progress = ref<string[]>([]);
const currentNode = ref("");
const nodeStartedAt = ref(0);
const nodeElapsed = ref(0);
const clarification = ref<Record<string, unknown> | null>(null);
const clarificationInput = ref("");
const clarificationQuestion = ref("");
const presentation = ref<AnalysisPresentation | null>(null);
const chatScroll = ref<HTMLElement | null>(null);
const complexPlan = ref<ComplexPlan | null>(null);
const scopeForm = ref<ExplorerQuery>({ seller_id: null, metric_name: "gmv", start_date: "2018-01-01", end_date: "2018-12-31", granularity: "day" });
const sellerOptions = ref<SellerOption[]>([]);
const scopeEditorVisible = ref(false);
const scopeSaving = ref(false);
const scopeEditorForm = ref<ExplorerQuery>({ seller_id: null, metric_name: "gmv", ...DEFAULT_ANALYSIS_RANGE, granularity: "day" });
const scopeMetricOptions = ref<Array<{ value: string; label: string }>>([]);
const adjustingPlan = ref(false);
const revisionFocus = ref("");
const revisionNote = ref("");
const runningPlanId = ref<string | null>(null);

const activeSession = computed(() => store.activeSession);
const scope = computed(() => activeSession.value?.scope ?? null);
const turns = computed(() => [...(activeSession.value?.turns ?? [])].reverse());
function scopeContinuityKey(value: ExplorerQuery | null): string {
  if (!value) return "platform";
  const raw = value as ExplorerQuery & { seller_ids?: string[]; metric_ids?: string[] };
  return JSON.stringify([
    value.seller_id ?? (raw.seller_ids?.length === 1 ? raw.seller_ids[0] : null),
    value.metric_name ?? (raw.metric_ids?.length === 1 ? raw.metric_ids[0] : undefined),
    value.start_date, value.end_date, value.granularity,
  ]);
}
function dimensionLabel(value: string): string {
  const labels: Record<string, string> = {
    product_category: "商品品类", product: "商品", seller: "店铺",
    customer_state: "客户州/地区", payment_type: "支付方式",
    month_position: "月内时段", customer_city: "客户城市", seller_state: "卖家州",
  };
  return labels[value] ?? value;
}
function turnDimensionLabels(turn: DebugTurn): string[] {
  const candidates = turn.clarification?.dimension_candidates;
  return Array.isArray(candidates) ? candidates.map(dimensionLabel) : [];
}
function turnMetricCandidates(turn: DebugTurn): Array<{ metric_id: string; display_name: string }> {
  const candidates = turn.clarification?.ambiguous_candidates;
  return Array.isArray(candidates) ? candidates : [];
}
function isPremiseConfirmationMessage(message: string): boolean {
  const normalized = message.toLocaleLowerCase().trim();
  return ["确认", "继续", "好的", "好", "可以", "是", "yes", "ok", "continue", "proceed"].includes(normalized)
    || ["按实际", "按真实", "分析实际上升", "分析实际下降"].some((phrase) => normalized.includes(phrase));
}
function displayAnswer(answer: string): string {
  return answer.replace(/数据来源：\s*(\([\s\S]*\)\s+([A-Za-z_]\w*)|SELECT\s+[\s\S]*?)(?=。)/gi, (_match, _source, alias) =>
    `数据来源：服务端受控指标查询${alias ? `（${alias}）` : ""}`,
  );
}

async function copyMessage(text: string, label: string): Promise<void> {
  if (!text) {
    ElMessage.warning("没有可复制的内容");
    return;
  }
  let copied = false;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      copied = true;
    }
  } catch { /* Fall back when clipboard permissions are unavailable. */ }
  if (!copied) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    try {
      textarea.select();
      copied = document.execCommand("copy");
    } catch { /* The browser may block both clipboard APIs. */ }
    finally { textarea.remove(); }
  }
  if (copied) ElMessage.success(`${label}已复制到剪贴板`);
  else {
    ElMessage.warning("复制失败，请手动选择并复制内容");
  }
}
const clarificationCandidates = computed(() => Array.isArray(clarification.value?.ambiguous_candidates)
  ? clarification.value.ambiguous_candidates as Array<{ metric_id: string; display_name: string; mapping_id?: string }> : []);
const pendingMappingIds = computed(() => clarificationCandidates.value.map(item => item.mapping_id).filter((id): id is string => Boolean(id)));
const supportedMetricIds = computed(() => Array.isArray(clarification.value?.confirmed_metrics) ? clarification.value.confirmed_metrics as string[] : []);
const unavailableMetrics = computed(() => Array.isArray(clarification.value?.unavailable_metrics) ? clarification.value.unavailable_metrics as string[] : []);
const complexTemplateCandidates = computed(() => Array.isArray(clarification.value?.candidate_templates)
  ? clarification.value.candidate_templates as Array<{ template_id: string; display_name: string }> : []);
const semanticClarification = computed(() => clarification.value?.semantic_interpretation as { interpretation?: string; display_name?: string; dimension_label?: string } | undefined);
const timeClarificationMessage = computed(() => String(clarification.value?.original_message ?? ""));
const taskClarificationMessage = computed(() => String(clarification.value?.original_message ?? ""));
const timeNote = ref("");
const nodeText: Record<string, string> = {
  init_state: "正在准备分析", resolve_explorer_context: "正在验证数据范围", load_session_memory: "正在读取会话记忆",
  triage: "正在理解问题", retrieve_analysis_memories: "正在检索历史结论", metric_definition: "正在读取指标口径",
  data_query: "正在查询数据", root_cause: "正在分析原因", report: "正在生成报告",
  generate_answer: "正在整理结论", critic: "正在审核结论", human_review: "等待人工审核",
  persist_trace: "正在保存追踪记录", persist_memory_p0: "正在更新会话记忆",
};

watch([turns, working, progress], async () => {
  await nextTick();
  if (chatScroll.value) chatScroll.value.scrollTop = chatScroll.value.scrollHeight;
}, { deep: true });

onMounted(async () => {
  try {
    await store.loadSessions(true);
    if (store.activeSessionId) await store.loadTurns(store.activeSessionId, true);
  } catch {
    ElMessage.warning("聊天历史加载失败，请稍后重试");
  }
});

function newSession() {
  store.createSession(); complexPlan.value = null;
  adjustingPlan.value = false;
  question.value = ""; pendingMessage.value = ""; progress.value = []; clarification.value = null; presentation.value = null;
}

async function selectSession(id: string) {
  if (id === store.activeSessionId) return;
  store.activeSessionId = id;
  pendingMessage.value = ""; progress.value = []; clarification.value = null; presentation.value = null; complexPlan.value = null;
  adjustingPlan.value = false;
  await store.loadTurns(id, true);
}

async function removeSession(session: { id: string }) {
  try {
    await ElMessageBox.confirm("删除后无法恢复该会话及其全部问答记录。", "删除会话", { type: "warning", confirmButtonText: "删除", cancelButtonText: "取消" });
    await store.deleteSession(session.id);
    ElMessage.success("会话已删除");
  } catch { /* User cancelled or request failed. */ }
}

async function removeTurn(turn: DebugTurn) {
  if (!activeSession.value) return;
  try {
    await ElMessageBox.confirm("删除这段问答记录？", "删除记录", { type: "warning", confirmButtonText: "删除", cancelButtonText: "取消" });
    await store.deleteTurn(activeSession.value.id, turn.id);
  } catch { /* User cancelled or request failed. */ }
}

async function clearScope() {
  try {
    await store.saveActiveSessionScope(null);
    ElMessage.success("已清除本会话的数据范围");
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "清除数据范围失败");
  }
}

async function openScopeEditor() {
  scopeEditorForm.value = scope.value
    ? { ...scope.value }
    : { seller_id: null, metric_name: "gmv", ...DEFAULT_ANALYSIS_RANGE, granularity: "day" };
  scopeEditorVisible.value = true;
  try {
    const [sellerResult, metrics] = await Promise.all([
      api.sellers(store.auth, ""),
      scopeMetricOptions.value.length ? Promise.resolve(null) : api.metrics(store.auth),
    ]);
    sellerOptions.value = sellerResult.items;
    if (metrics) {
      scopeMetricOptions.value = metrics
        .filter((metric: MetricDefinition) => (metric.surfaces || []).includes("explorer"))
        .map((metric: MetricDefinition) => ({ value: metric.metric_name, label: metric.display_name || metric.metric_name }));
    }
    if (scopeEditorForm.value.seller_id && !sellerOptions.value.some((item) => item.seller_id === scopeEditorForm.value.seller_id)) {
      sellerOptions.value.unshift({ seller_id: scopeEditorForm.value.seller_id, label: scopeEditorForm.value.seller_id });
    }
    if (scopeEditorForm.value.metric_name && !scopeMetricOptions.value.some((item) => item.value === scopeEditorForm.value.metric_name)) {
      scopeMetricOptions.value.unshift({ value: scopeEditorForm.value.metric_name, label: scopeEditorForm.value.metric_name });
    }
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "范围选项加载失败");
  }
}

async function applyScopeFromEditor() {
  const validationError = validateExplorerQuery(scopeEditorForm.value);
  if (validationError) return ElMessage.warning(validationError);
  scopeSaving.value = true;
  try {
    await store.saveActiveSessionScope({ ...scopeEditorForm.value });
    store.clearPreview();
    scopeEditorVisible.value = false;
    ElMessage.success("范围已更新；新范围用于本会话后续提问，历史回答保持不变");
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "范围保存失败");
  } finally {
    scopeSaving.value = false;
  }
}

async function submit(metricConfirmation: string[] | null = null, messageOverride?: string, taskId?: string, semanticMappingIds: string[] = [], skippedSemanticMappingIds: string[] = []) {
  const message = (messageOverride ?? question.value).trim();
  if (!message || working.value) return;
  // session.id 是 durable chat 的唯一会话标识；长期偏好有独立的 server-side scope。
  const session = activeSession.value ?? store.createSession();
  // 每一轮消息单独生成 turn id，因此不能把聊天窗口 id 当作本轮 thread id。
  const requestScope = scope.value ? { ...scope.value } : null;
  const scopeKey = scopeContinuityKey(requestScope);
  const latestTurn = session.turns[0];
  const samePremiseScope = Boolean(
    latestTurn
    // Turns restored after a reload intentionally omit the client-only
    // scopeKey. The server exposes a resume ticket only after it has matched
    // the turn, execution, and session fingerprints; keep checking the local
    // key whenever this client still has one.
    && (!latestTurn.scopeKey || latestTurn.scopeKey === scopeKey)
    && latestTurn.scopeFingerprint
    && (!session.scopeFingerprint || session.scopeFingerprint === latestTurn.scopeFingerprint),
  );
  // /turns exposes thread_id and scope_fingerprint only after the server has
  // verified that this suspended checkpoint still matches the session scope.
  // The backend validates the fingerprint again when it accepts the resume.
  const resumePremise = Boolean(
    latestTurn
    && ["suspended", "running"].includes(latestTurn.status)
    && latestTurn.interactionState === "awaiting_human_input"
    && latestTurn.interactionReasonCode === "premise_direction_mismatch"
    && latestTurn.clientTurnId
    && samePremiseScope
    && latestTurn.threadId
    && isPremiseConfirmationMessage(message),
  );
  const resumeTurn = resumePremise ? latestTurn : undefined;
  const id = resumeTurn?.clientTurnId ?? crypto.randomUUID();
  // Only an explicit correction for a same-session, same-scope premise pause
  // may reuse a checkpoint. Every other turn receives a fresh thread.
  const threadId = resumeTurn?.threadId ?? `${session.id}-${id}`;
  pendingMessage.value = message; question.value = ""; working.value = true; progress.value = [];
  currentNode.value = ""; nodeStartedAt.value = 0; nodeElapsed.value = 0; clarification.value = null; presentation.value = null;
  try {
    for (const mappingId of semanticMappingIds) {
      await api.confirmSemanticMapping(store.auth, mappingId, { chat_session_id: session.id, explorer_context: requestScope ?? undefined });
    }
    for (const mappingId of skippedSemanticMappingIds) {
      await api.skipSemanticMapping(store.auth, mappingId, { chat_session_id: session.id, explorer_context: requestScope ?? undefined });
    }
    // 发送四个不同生命周期的标识：thread 用于 checkpoint，session 用于跨轮记忆/历史，
    // turn 用于本轮聊天记录幂等关联。
    const result = await streamChat(store.auth, {
      message, thread_id: threadId, use_current_explorer_context: Boolean(requestScope), explorer_context: requestScope,
      metric_confirmation: metricConfirmation ?? undefined,
      chat_session_id: session.id, chat_turn_id: id, task_id: taskId,
      semantic_mapping_ids: semanticMappingIds.length ? semanticMappingIds : undefined,
      skipped_semantic_mapping_ids: skippedSemanticMappingIds.length ? skippedSemanticMappingIds : undefined,
    }, (event) => {
      if (isSseStartedEvent(event)) return;
      if (isSseNodeStartedEvent(event)) { currentNode.value = nodeText[event.node] ?? event.node; nodeStartedAt.value = Date.now(); nodeElapsed.value = 0; }
      if (isSseNodeCompletedEvent(event)) { progress.value.push(nodeText[event.node] ?? `已完成：${event.node}`); currentNode.value = ""; }
      if (isSseInvestigationProgressEvent(event)) {
        const action = event.current_action ? `；当前动作：${event.current_action}` : "";
        const stop = event.stop_reason ? `；停止原因：${formatInvestigationStopReason(event.stop_reason)}` : "";
        progress.value.push(`归因调查 ${event.completed_actions}/6，剩余预算 ${event.remaining_budget} 项${action}${stop}`);
      }
      if (event.event === "heartbeat" && nodeStartedAt.value) nodeElapsed.value = Math.round((Date.now() - nodeStartedAt.value) / 1000);
    });
    // `error` also accompanies an abstained result when a partial answer is
    // available (for example, a budget boundary). Preserve that answer and
    // use the backend's terminal classification as the source of truth.
    const terminalStatus = "terminal_status" in result ? result.terminal_status : null;
    const failed = terminalStatus === "failed"
      || (result.error === true && terminalStatus !== "abstained");
    const resultClarification = result.clarification ?? null;
    const metricCandidates = (resultClarification as { ambiguous_candidates?: unknown } | null)?.ambiguous_candidates;
    const metricChoiceIsStoredWithTurn = Array.isArray(metricCandidates) && metricCandidates.length > 0;
    // Metric candidates are rendered with the durable turn below. Keeping the
    // same payload in the page-level clarification also rendered duplicate
    // cards and duplicate buttons for one server response.
    clarification.value = metricChoiceIsStoredWithTurn ? null : resultClarification;
    if (result.clarification) clarificationQuestion.value = message;
    complexPlan.value = (result as unknown as { complex_plan?: ComplexPlan }).complex_plan ?? null;
    presentation.value = (result as { presentation?: AnalysisPresentation }).presentation ?? null;
    if (complexPlan.value) return;
    store.addTurn({ id: resumeTurn?.id ?? id, clientTurnId: id, threadId,
      message: resumeTurn?.message ?? message,
      answer: failed ? undefined : result.answer, presentation: presentation.value ?? undefined,
      error: failed
        ? (result.error_detail ?? "分析异常")
        : terminalStatus === "abstained" && !result.answer
          ? (result.error_detail ?? "本轮未能形成完整结论")
          : undefined,
      taskType: result.task_type, traceId: result.trace_id,
      createdAt: new Date().toISOString(), status: failed ? "failed" : result.event === "suspended" ? "suspended" : terminalStatus ?? "completed",
      interactionState: result.event === "suspended" ? result.interaction_state : undefined,
      interactionReasonCode: result.event === "suspended" ? result.interaction_reason_code ?? undefined : undefined,
      scopeFingerprint: result.event === "suspended" ? result.scope_fingerprint ?? undefined : undefined,
      clarification: result.clarification ?? null,
      scopeKey });
    if (failed) ElMessage.warning(result.error_detail ?? "分析过程出现异常");
  } catch (error) {
    const detail = error instanceof Error ? error.message : "分析请求失败";
    store.addTurn({ id, threadId, message, error: detail, taskType: "failed", traceId: "", createdAt: new Date().toISOString(), status: "failed" });
    ElMessage.error(detail);
  } finally {
    working.value = false; pendingMessage.value = ""; currentNode.value = "";
  }
}

async function continueCurrentTask() {
  const taskId = String(clarification.value?.task_id ?? "");
  if (!taskId || !taskClarificationMessage.value) return;
  await submit(null, taskClarificationMessage.value, taskId);
}

async function startNewTask() {
  const session = activeSession.value;
  const message = taskClarificationMessage.value;
  if (!session || !message) return;
  try {
    const task = await api.createTask(store.auth, session.id, { objective: message, scope: scope.value });
    await submit(null, message, String(task.task_id));
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "创建分析任务失败");
  }
}

function cancelMetricClarification() {
  clarification.value = null;
  question.value = "";
  clarificationInput.value = "";
  clarificationQuestion.value = "";
}
function selectMetricCandidate(item: { metric_id: string; display_name: string; mapping_id?: string }, sourceQuestion?: string) {
  const originalQuestion = (sourceQuestion ?? clarificationQuestion.value).trim();
  if (!originalQuestion) return ElMessage.warning("未找到待确认的问题，请重新输入分析需求");
  // Keep the user's original request and add the selected registered phrase.
  // The server still re-resolves and authorizes the metric from this text.
  const message = `${originalQuestion}\n请按${item.display_name}指标查询。`;
  return submit([item.metric_id], message, undefined, item.mapping_id ? [item.mapping_id] : []);
}
function selectComplexTemplate(item: { template_id: string; display_name: string }, sourceQuestion?: string) {
  const originalQuestion = (sourceQuestion ?? clarificationQuestion.value).trim();
  if (!originalQuestion) return ElMessage.warning("未找到待选择的问题，请重新输入分析需求");
  // This is an explicit semantic clarification, not a client-selected route.
  // The server parses and routes the clarified text again before creating any plan.
  return submit(null, `${originalQuestion}\n我选择${item.display_name}。`);
}
function submitMetricClarificationText() {
  const note = clarificationInput.value.trim();
  if (!note || working.value) return ElMessage.warning("请先填写确认说明");
  if (/^(终止|取消|放弃|结束)/.test(note)) return cancelMetricClarification();
  if (/只分析|仅分析|已支持/.test(note) && supportedMetricIds.value.length) {
    return submit(supportedMetricIds.value, clarificationQuestion.value, undefined, [], pendingMappingIds.value);
  }
  const candidate = clarificationCandidates.value.find((item) => note.includes(item.display_name) || note.includes(item.metric_id));
  if (candidate) return selectMetricCandidate(candidate);
  const selectedMetric = String(scope.value?.metric_name ?? "");
  if (selectedMetric) return submit([selectedMetric], clarificationQuestion.value);
  return submit(null, `${clarificationQuestion.value}\n补充确认：${note}`);
}

async function searchSellers(query: string) { sellerOptions.value = (await api.sellers(store.auth, query)).items; }
function planFocusLabel(plan: ComplexPlan | null): string {
  if (!plan?.preferences?.focus_key) return "";
  return plan.focus_options?.find(item => item.key === plan.preferences?.focus_key)?.label ?? plan.preferences.focus_key;
}
async function confirmComplexPlan() {
  const plan = complexPlan.value;
  if (!plan || working.value) return;
  working.value = true; runningPlanId.value = plan.plan_id; progress.value = []; currentNode.value = "正在执行已确认的复杂分析";
  try {
    const result = await streamComplexPlan(store.auth, plan.plan_id, (event) => {
      if (isSseNodeStartedEvent(event)) currentNode.value = nodeText[event.node] ?? event.node;
      if (isSseNodeCompletedEvent(event)) progress.value.push(nodeText[event.node] ?? `已完成：${event.node}`);
      if (isSseInvestigationProgressEvent(event)) {
        const action = event.current_action ? `；当前动作：${event.current_action}` : "";
        const stop = event.stop_reason ? `；停止原因：${formatInvestigationStopReason(event.stop_reason)}` : "";
        progress.value.push(`归因调查 ${event.completed_actions}/6，剩余预算 ${event.remaining_budget} 项${action}${stop}`);
      }
    });
    const resultPresentation = (result as unknown as { presentation?: AnalysisPresentation }).presentation;
    const cancelled = Boolean((result as unknown as { cancelled?: boolean }).cancelled);
    store.addTurn({ id: plan.chat_turn_id ?? crypto.randomUUID(), threadId: plan.chat_turn_id ?? "", message: plan.question,
      answer: result.answer, presentation: resultPresentation, taskType: result.task_type, traceId: result.trace_id, createdAt: new Date().toISOString(), status: cancelled ? "failed" : "completed" });
    complexPlan.value = null; clarification.value = null; adjustingPlan.value = false;
    if (cancelled) ElMessage.info("已取消本次复杂分析，结果不会保存为结论或行动草案");
    else if ((result as unknown as { action_created?: boolean }).action_created) ElMessage.success("分析完成，并已生成行动草案");
  } catch (error) { ElMessage.error(error instanceof Error ? error.message : "复杂分析执行失败"); }
  finally { working.value = false; runningPlanId.value = null; currentNode.value = ""; }
}

async function requestRunningPlanCancellation() {
  if (!runningPlanId.value) return;
  try {
    await api.cancelComplexPlan(store.auth, runningPlanId.value);
    ElMessage.info("已发送取消请求：当前受控查询结束后将丢弃结果。");
  } catch (error) { ElMessage.error(error instanceof Error ? error.message : "取消请求失败"); }
}

async function cancelComplexPlan() {
  if (!complexPlan.value) return;
  try { await api.cancelComplexPlan(store.auth, complexPlan.value.plan_id); complexPlan.value = null; clarification.value = null; adjustingPlan.value = false; ElMessage.info("已取消复杂分析计划"); }
  catch (error) { ElMessage.error(error instanceof Error ? error.message : "取消失败"); }
}

function openPlanAdjustment() {
  const plan = complexPlan.value;
  if (!plan) return;
  scopeForm.value = { seller_id: plan.scope.seller_id ?? null, metric_name: plan.scope.metric_name ?? "gmv", start_date: plan.params.start_date, end_date: plan.params.end_date, granularity: (plan.scope.granularity as ExplorerQuery["granularity"]) ?? "day" };
  revisionFocus.value = plan.preferences?.focus_key ?? plan.focus_options?.[0]?.key ?? "";
  revisionNote.value = plan.preferences?.note ?? "";
  adjustingPlan.value = true;
  searchSellers("");
}

async function reviseComplexPlan() {
  const plan = complexPlan.value;
  if (!plan || working.value || !scopeForm.value.seller_id) return ElMessage.warning("请选择店铺并补全日期范围");
  working.value = true;
  try {
    const preview = await api.preview(store.auth, scopeForm.value);
    if (preview.summary.status !== "ok") return ElMessage.warning("该范围没有可用数据，请调整后重试");
    complexPlan.value = await api.reviseComplexPlan(store.auth, plan.plan_id, { explorer_context: preview.context, focus_key: revisionFocus.value || undefined, note: revisionNote.value.trim() || undefined });
    adjustingPlan.value = false;
    ElMessage.success("已生成新版计划，请再次确认执行");
  } catch (error) { ElMessage.error(error instanceof Error ? error.message : "计划调整失败"); }
  finally { working.value = false; }
}
</script>

<template>
  <div class="chat-page">
    <aside class="history-sidebar">
      <div class="history-head"><div><strong>聊天记录</strong><span>经营分析助手</span></div><el-button type="primary" size="small" @click="newSession">+ 新对话</el-button></div>
      <div class="session-list">
        <button v-for="session in store.sessions" :key="session.id" class="session-row" :class="{ active: session.id === store.activeSessionId }" @click="selectSession(session.id)">
          <span class="session-title">{{ session.title }}</span><small>{{ new Date(session.updatedAt).toLocaleDateString() }}</small>
          <el-button class="delete-session" text type="danger" size="small" @click.stop="removeSession(session)">删除</el-button>
        </button>
        <el-button v-if="store.nextSessionsOffset !== null" text type="primary" class="load-more" @click="store.loadSessions()">加载更多</el-button>
      </div>
    </aside>

    <main class="conversation-panel">
      <header class="conversation-head">
        <div><h1>{{ activeSession?.title || "新对话" }}</h1><p>可基于经营数据提问、追溯并继续分析</p></div>
        <div class="chat-head-actions"><el-button size="small" @click="$router.push({ path: '/code-analysis', query: { question, session: activeSession?.id } })">上传数据分析</el-button></div>
      </header>
      <ScopeTag :scope="scope" @edit="openScopeEditor" @clear="clearScope" />
      <section ref="chatScroll" class="message-stream">
        <div v-if="!turns.length && !working" class="empty-chat"><h2>今天想分析什么？</h2><p>例如：本月 GMV 为什么下降，哪些商品或地区贡献最大？</p></div>
        <template v-for="turn in turns" :key="turn.id">
          <article class="message user-message"><div class="avatar">你</div><div class="user-message-content"><div class="bubble">{{ turn.message }}</div><div class="user-message-actions"><el-button text size="small" :icon="DocumentCopy" aria-label="复制问题" @click="copyMessage(turn.message, '问题')">复制</el-button></div></div></article>
          <article class="message assistant-message"><div class="avatar assistant-avatar">BI</div><div class="assistant-content">
            <AnalysisResultCard v-if="turn.presentation" :presentation="turn.presentation" :answer="displayAnswer(turn.answer ?? '')" :trace-id="turn.traceId" />
            <MarkdownContent v-else-if="turn.answer" class="answer-text" :content="displayAnswer(turn.answer)" />
            <el-alert v-else :title="turn.error ?? '分析未返回结论'" type="error" :closable="false" show-icon />
            <el-alert v-if="turn.interactionState === 'awaiting_human_input'" :class="turn.interactionReasonCode === 'premise_direction_mismatch' ? 'history-clarification' : undefined" :title="turn.interactionReasonCode === 'premise_direction_mismatch' ? '分析已暂停：问题前提与数据证据不一致。你可以确认后按实际数据继续。' : '当前请求已暂停，等待人工审核或后续确认。'" type="info" :closable="false" show-icon />
            <el-card v-if="turn.clarification?.status === 'complex_scope_required' && clarification?.status !== 'complex_scope_required'" class="complex-plan history-clarification" shadow="never"><template #header><strong>需要补充分析范围</strong></template><p>复杂分析需要一个已授权的店铺和日期范围。请选择范围后重新发送原问题；这条历史记录尚未创建或执行计划。</p><div class="plan-actions"><el-button type="primary" @click="openScopeEditor">选择店铺和日期范围</el-button></div></el-card>
            <el-alert v-else-if="turn.clarification" class="history-clarification" type="warning" :closable="false" show-icon>
              <template #title>{{ clarificationTitle(turn.clarification) }}</template>
              <p>{{ clarificationMessage(turn.clarification) }}</p>
              <p v-if="turnDimensionLabels(turn).length">候选维度：{{ turnDimensionLabels(turn).join("、") }}。请明确选择后重新提问；系统未自动选择。</p>
              <div v-if="turnMetricCandidates(turn).length" class="clarification-actions">
                <el-button v-for="item in turnMetricCandidates(turn)" :key="item.metric_id" size="small" @click="selectMetricCandidate(item, turn.message)">确认使用：{{ item.display_name }}</el-button>
              </div>
              <div v-if="Array.isArray(turn.clarification.candidate_templates)" class="clarification-actions">
                <el-button v-for="item in turn.clarification.candidate_templates as Array<{ template_id: string; display_name: string }>" :key="item.template_id" size="small" @click="selectComplexTemplate(item, turn.message)">选择：{{ item.display_name }}</el-button>
              </div>
            </el-alert>
            <div class="message-actions"><span v-if="turn.traceId">Trace · {{ turn.traceId.slice(0, 8) }}</span><div class="message-action-buttons"><el-button v-if="turn.answer" text size="small" :icon="DocumentCopy" aria-label="复制 AI 回复" @click="copyMessage(displayAnswer(turn.answer), 'AI 回复')">复制</el-button><el-button text type="danger" size="small" @click="removeTurn(turn)">删除此段</el-button></div></div>
          </div></article>
        </template>
        <template v-if="working"><article class="message user-message"><div class="avatar">你</div><div class="user-message-content"><div class="bubble">{{ pendingMessage }}</div><div class="user-message-actions"><el-button text size="small" :icon="DocumentCopy" aria-label="复制问题" @click="copyMessage(pendingMessage, '问题')">复制</el-button></div></div></article><article class="message assistant-message"><div class="avatar assistant-avatar">BI</div><div class="thinking"><strong>{{ currentNode || "正在分析" }}</strong><span v-if="nodeElapsed">{{ nodeElapsed }} 秒</span><p>{{ progress.length ? progress[progress.length - 1] : "正在构建分析路径…" }}</p><el-button v-if="runningPlanId" text type="danger" size="small" @click="requestRunningPlanCancellation">取消本次执行</el-button></div></article></template>
        <el-card v-if="complexPlan" class="complex-plan" shadow="never"><template #header><strong>复杂分析计划（等待确认）</strong><el-tag v-if="complexPlan.revision" size="small" class="plan-revision">版本 {{ complexPlan.revision }}</el-tag></template><p><strong>{{ complexPlan.display_name }}</strong>：{{ complexPlan.purpose }}</p><p>店铺：{{ complexPlan.scope.seller_id }} · {{ complexPlan.params.start_date }} 至 {{ complexPlan.params.end_date }}</p><p v-if="complexPlan.preferences?.focus_key">分析角度：{{ planFocusLabel(complexPlan) }}</p><p v-if="complexPlan.preferences?.note">关注假设：{{ complexPlan.preferences.note }}</p><p class="muted">{{ complexPlan.estimated_cost }}</p><ul><li v-for="item in complexPlan.limitations" :key="item">{{ item }}</li></ul><div class="plan-actions"><el-button @click="openPlanAdjustment">调整计划</el-button><el-button @click="cancelComplexPlan">取消</el-button><el-button type="primary" :loading="working" @click="confirmComplexPlan">确认执行</el-button></div></el-card>
        <el-card v-else-if="clarification?.status === 'task_relationship_required'" class="complex-plan" shadow="never"><template #header><strong>需要你决定</strong></template><p>这句话可能是在延续当前任务，也可能是一个独立的新分析。为避免把不同范围或目标混在一起，请选择处理方式。</p><p class="muted">原问题：{{ taskClarificationMessage }}</p><div class="plan-actions"><el-button type="primary" @click="continueCurrentTask">继续当前任务</el-button><el-button @click="startNewTask">新建分析任务</el-button></div></el-card>
        <el-card v-else-if="clarification?.status === 'complex_scope_required'" class="complex-plan" shadow="never"><template #header><strong>需要补充分析范围</strong></template><p>复杂分析需要一个已授权的店铺和日期范围。请选择范围后重新发送问题；当前尚未创建或执行计划。</p><div class="plan-actions"><el-button type="primary" @click="openScopeEditor">选择店铺和日期范围</el-button></div></el-card>
        <el-card v-else-if="clarification?.status === 'complex_template_required'" class="complex-plan" shadow="never"><template #header><strong>选择复杂分析模板</strong></template><p>该请求匹配多种分析方式，系统不会替你猜选。请选择后将重新校验请求，并生成等待确认的计划；此处不会直接执行。</p><div class="plan-actions"><el-button v-for="item in complexTemplateCandidates" :key="item.template_id" @click="selectComplexTemplate(item)">{{ item.display_name }}</el-button></div></el-card>
        <el-card v-else-if="clarification?.status === 'complex_analysis_unavailable'" class="complex-plan" shadow="never"><template #header><strong>复杂分析暂不可用</strong></template><p>语义路由或安全校验未通过，本轮未创建或执行复杂分析计划。你可以稍后重试，或使用普通聊天查询数据。</p></el-card>
        <el-card v-else-if="clarification?.status === 'time_intent_required'" class="complex-plan" shadow="never"><template #header><strong>确认时间分析方式</strong></template><p>{{ clarification.reason }}</p><p class="muted">原问题：{{ timeClarificationMessage }}</p><div class="plan-actions"><el-button type="primary" @click="submit(null, timeClarificationMessage + '。请按年度趋势分析。')">按年度趋势</el-button><el-button @click="submit(null, timeClarificationMessage + '。请比较第一个和最后一个年份。')">比较首末年份</el-button></div><el-input v-model="timeNote" class="time-note" type="textarea" :rows="2" placeholder="或者补充说明，例如：只比较 2016 年和 2018 年；按季度看。" /><el-button :disabled="!timeNote.trim()" @click="submit(null, timeClarificationMessage + '。补充说明：' + timeNote)">按备注继续</el-button></el-card>
        <el-alert v-else-if="clarification" class="clarification" type="warning" :closable="false" show-icon><template #title>{{ clarificationTitle(clarification, Boolean(semanticClarification)) }}</template><p v-if="semanticClarification">系统理解：{{ semanticClarification.interpretation || semanticClarification.display_name }}<span v-if="semanticClarification.dimension_label">（维度：{{ semanticClarification.dimension_label }}）</span></p><p v-if="unavailableMetrics.length">当前不可用：{{ unavailableMetrics.join('、') }}</p><p>{{ clarificationMessage(clarification) }}</p><p v-if="Array.isArray(clarification.dimension_candidates) && clarification.dimension_candidates.length">候选维度：{{ (clarification.dimension_candidates as string[]).map(dimensionLabel).join('、') }}。请明确选择后重新提问；系统未自动选择。</p><div class="clarification-actions"><el-button v-if="supportedMetricIds.length" type="primary" size="small" @click="submit(supportedMetricIds, clarificationQuestion, undefined, [], pendingMappingIds)">只分析已支持指标</el-button><el-button v-for="item in clarificationCandidates" :key="item.mapping_id || item.metric_id" size="small" @click="selectMetricCandidate(item)">确认使用：{{ item.display_name }}</el-button><el-button size="small" @click="cancelMetricClarification">终止本次分析</el-button></div><el-input v-model="clarificationInput" class="clarification-input" type="textarea" :rows="2" placeholder="补充确认，例如：只分析已支持指标，或改用某个近似指标" /><div class="clarification-actions"><el-button type="primary" size="small" :disabled="!clarificationInput.trim()" @click="submitMetricClarificationText">提交确认说明</el-button></div></el-alert>
        <el-card v-if="complexPlan && !adjustingPlan" class="complex-plan plan-adjust-entry" shadow="never"><p class="muted">可在执行前调整范围、分析角度或补充关注假设；调整后会生成新版本，必须再次确认。</p><div class="plan-actions"><el-button @click="openPlanAdjustment">调整计划</el-button></div></el-card>
        <el-card v-if="complexPlan && adjustingPlan" class="complex-plan" shadow="never"><template #header><strong>调整分析计划</strong></template><el-form label-position="top"><el-form-item label="店铺"><el-select v-model="scopeForm.seller_id" filterable remote :remote-method="searchSellers" placeholder="选择授权店铺"><el-option v-for="seller in sellerOptions" :key="seller.seller_id" :label="seller.label" :value="seller.seller_id" /></el-select></el-form-item><el-form-item label="日期范围"><el-date-picker v-model="scopeForm.start_date" value-format="YYYY-MM-DD" type="date" /><span class="date-separator">至</span><el-date-picker v-model="scopeForm.end_date" value-format="YYYY-MM-DD" type="date" /></el-form-item><el-form-item label="分析角度"><el-radio-group v-model="revisionFocus"><el-radio v-for="item in complexPlan.focus_options" :key="item.key" :value="item.key">{{ item.label }}</el-radio></el-radio-group></el-form-item><el-form-item label="关注点 / 假设"><el-input v-model="revisionNote" maxlength="280" show-word-limit type="textarea" :rows="3" placeholder="例如：优先验证是否由某品类造成。备注只影响分析重点，不改变模板计算。" /></el-form-item></el-form><div class="plan-actions"><el-button @click="adjustingPlan = false">返回计划</el-button><el-button type="primary" :loading="working" @click="reviseComplexPlan">生成新版计划</el-button></div></el-card>
        <el-button v-if="activeSession && store.nextTurnsOffset[activeSession.id] !== null" text type="primary" class="load-more" @click="store.loadTurns(activeSession.id)">加载更早记录</el-button>
      </section>

      <footer class="composer"><el-input v-model="question" type="textarea" :autosize="{ minRows: 1, maxRows: 5 }" resize="none" placeholder="输入经营问题，Enter 发送，Shift + Enter 换行" :disabled="working" @keydown.enter.exact.prevent="submit()" /><div><span>只保存问题与最终结论</span><el-button type="primary" :loading="working" :disabled="!question.trim()" @click="submit()">发送</el-button></div></footer>
    </main>
    <aside class="task-sidebar"><TaskContextCard :session-id="activeSession?.id" /></aside>

    <el-drawer v-model="scopeEditorVisible" title="修改本会话范围" size="420px" append-to-body>
      <p class="scope-editor-hint">应用后留在当前对话。新范围用于后续提问，不会改写已有回答。</p>
      <el-form label-position="top" class="scope-editor-form">
        <el-form-item label="店铺">
          <el-select v-model="scopeEditorForm.seller_id" filterable remote reserve-keyword clearable
            placeholder="搜索并选择授权店铺" :remote-method="searchSellers"
            @visible-change="(visible: boolean) => visible && searchSellers('')">
            <el-option v-for="item in sellerOptions" :key="item.seller_id"
              :label="item.label" :value="item.seller_id" />
          </el-select>
        </el-form-item>
        <el-form-item label="日期范围">
          <div class="scope-editor-dates">
            <el-date-picker v-model="scopeEditorForm.start_date" type="date" value-format="YYYY-MM-DD" placeholder="开始日期" />
            <span>至</span>
            <el-date-picker v-model="scopeEditorForm.end_date" type="date" value-format="YYYY-MM-DD" placeholder="结束日期" />
          </div>
        </el-form-item>
        <el-form-item label="指标">
          <el-select v-model="scopeEditorForm.metric_name" filterable>
            <el-option v-for="item in scopeMetricOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="时间粒度">
          <el-segmented v-model="scopeEditorForm.granularity" :options="[
            { label: '日', value: 'day' }, { label: '周', value: 'week' },
            { label: '月', value: 'month' }, { label: '季', value: 'quarter' },
            { label: '年', value: 'year' },
          ]" />
        </el-form-item>
      </el-form>
      <template #footer>
        <div class="scope-editor-actions">
          <el-button @click="scopeEditorVisible = false">取消</el-button>
          <el-button type="primary" :loading="scopeSaving" @click="applyScopeFromEditor">应用到本会话</el-button>
        </div>
      </template>
    </el-drawer>
  </div>
</template>

<style scoped>
.chat-page { height: calc(100vh - 110px); min-height: 640px; display: grid; grid-template-columns: 264px minmax(0, 1fr) 292px; border: 1px solid var(--el-border-color-lighter); border-radius: 14px; overflow: hidden; background: #fff; }
.history-sidebar { background: #f8fafc; border-right: 1px solid var(--el-border-color-lighter); display: flex; flex-direction: column; min-height: 0; }
.task-sidebar { min-width: 0; overflow-y: auto; padding: 14px; background: #fbfcfe; border-left: 1px solid var(--el-border-color-lighter); }
.history-head { display: flex; align-items: center; justify-content: space-between; padding: 18px 16px 15px; border-bottom: 1px solid var(--el-border-color-lighter); }.history-head strong { display: block; color: #182b49; }.history-head span { display: block; margin-top: 3px; font-size: 12px; color: #8491a5; }
.session-list { overflow-y: auto; padding: 8px; }.session-row { width: 100%; position: relative; padding: 11px 10px; text-align: left; border: 0; border-radius: 8px; background: transparent; color: inherit; cursor: pointer; }.session-row:hover, .session-row.active { background: #eaf2ff; }.session-title { display: block; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; font-size: 13px; }.session-row small { color: #98a2b3; font-size: 11px; }.delete-session { display: none; position: absolute; right: 5px; top: 7px; }.session-row:hover .delete-session { display: block; }
.conversation-panel { min-width: 0; display: grid; grid-template-rows: auto auto minmax(0, 1fr) auto; }.conversation-head { padding: 18px 28px; border-bottom: 1px solid var(--el-border-color-lighter); display: flex; align-items: center; justify-content: space-between; gap: 16px; }.conversation-head h1 { margin: 0; font-size: 18px; }.conversation-head p { margin: 4px 0 0; color: #8491a5; font-size: 13px; }
.message-stream { overflow-y: auto; padding: 28px clamp(18px, 6vw, 92px); background: #fff; }.message { display: flex; gap: 12px; max-width: 960px; margin: 0 auto 26px; }.avatar { flex: 0 0 30px; width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center; font-size: 12px; font-weight: 700; background: #dfe9ff; color: #3167c8; }.assistant-avatar { background: #1f6feb; color: #fff; }.bubble { max-width: min(74%, 680px); padding: 11px 14px; border-radius: 4px 14px 14px; background: #eaf2ff; white-space: pre-wrap; line-height: 1.65; }.assistant-content { flex: 1; min-width: 0; }.answer-text { padding: 4px 2px; white-space: pre-wrap; line-height: 1.8; color: #27364c; }.message-actions { display: flex; align-items: center; justify-content: space-between; margin-top: 9px; color: #98a2b3; font-size: 11px; }.thinking { padding: 14px 16px; border: 1px solid #dbe8ff; border-radius: 12px; background: #f5f9ff; color: #3965a9; }.thinking span { margin-left: 9px; color: #8ca5cb; font-size: 12px; }.thinking p { margin: 7px 0 0; font-size: 13px; }.empty-chat { max-width: 560px; margin: 18vh auto 0; text-align: center; color: #77849a; }.empty-chat h2 { color: #1d3557; font-size: 24px; }.clarification { max-width: 900px; margin: 0 auto 20px; }.load-more { display: block; margin: 8px auto; }
.composer { margin: 0; padding: 14px clamp(18px, 6vw, 92px) 18px; border-top: 1px solid var(--el-border-color-lighter); background: #fff; }.composer :deep(.el-textarea__inner) { padding: 12px 14px; border-radius: 12px; box-shadow: 0 1px 8px rgba(38, 67, 120, .08); }.composer > div:last-child { margin-top: 8px; display: flex; align-items: center; justify-content: space-between; color: #98a2b3; font-size: 12px; }.complex-plan { max-width: 900px; margin: 0 auto 20px; border-color: #b7d2ff; }.complex-plan p { line-height: 1.7; margin: 6px 0; }.complex-plan ul { color: #6b778c; padding-left: 20px; }.muted { color: #6b778c; font-size: 13px; }.plan-actions { display:flex; justify-content:flex-end; gap:8px; margin-top:14px; }
@media (max-width: 1050px) { .chat-page { grid-template-columns: 220px minmax(0, 1fr); }.task-sidebar { display: none; } }
@media (max-width: 900px) { .chat-page { height: calc(100vh - 92px); grid-template-columns: 1fr; }.history-sidebar { display: none; }.message-stream { padding: 20px; }.conversation-head { padding: 14px 18px; }.composer { padding: 12px 18px; } }
/* Keep the chat stream inside the available column. Long answers, tables and
   trace identifiers must wrap or scroll inside the stream instead of widening
   the page and forcing browser zoom-out. */
.chat-page { min-width: 0; }
.conversation-panel { min-width: 0; min-height: 0; }
.conversation-head { min-width: 0; flex-wrap: wrap; }
.conversation-head > div:first-child { min-width: 0; }
.conversation-head h1 { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.chat-head-actions { min-width: 0; display: flex; flex-wrap: wrap; gap: 8px; }
.message-stream { min-width: 0; min-height: 0; overflow-x: hidden; }
.message { width: 100%; min-width: 0; }
.bubble, .assistant-content, .answer-text, .thinking { min-width: 0; overflow-wrap: anywhere; }
.message-actions { gap: 8px; }
.scope-editor-hint { margin: 0 0 18px; color: var(--el-text-color-secondary); font-size: 13px; line-height: 1.6; }
.scope-editor-form :deep(.el-select), .scope-editor-form :deep(.el-date-editor) { width: 100%; }
.scope-editor-dates { display: flex; align-items: center; gap: 8px; width: 100%; }
.scope-editor-dates span { flex: 0 0 auto; color: var(--el-text-color-secondary); }
.scope-editor-actions { display: flex; justify-content: flex-end; gap: 8px; }
.message-stream :deep(.el-card), .message-stream :deep(.el-alert), .message-stream :deep(.el-descriptions) { max-width: 100%; min-width: 0; }
.message-stream :deep(.el-table) { max-width: 100%; overflow-x: auto; }
.chat-page { border-color: #e5eaf2; border-radius: 18px; box-shadow: 0 12px 36px rgba(24, 39, 75, .07); }
.history-sidebar { background: #fbfcfe; border-right-color: #edf0f5; }
.task-sidebar { background: #fbfcfe; border-left-color: #edf0f5; }
.history-head { padding: 20px 18px 17px; }
.history-head strong { color: #172033; font-size: 14px; }
.session-list { padding: 10px; }
.session-row { padding: 12px 11px; }
.session-row:hover, .session-row.active { background: #edf3ff; }
.conversation-head { padding: 20px 30px; background: rgba(255, 255, 255, .92); }
.conversation-head h1 { color: #172033; }
.message-stream { padding-top: 34px; padding-bottom: 34px; }
.message { margin-bottom: 30px; }
.assistant-content { max-width: 900px; }
.assistant-content :deep(.el-card) { box-shadow: none; border-color: #e8edf5; border-radius: 12px; }
.assistant-content :deep(.el-card__header) { background: #fbfcfe; }
.conclusion-card :deep(.el-card__header) { background: #f4f7ff; }
.bubble { background: #eaf1ff; color: #24416f; border-radius: 10px 4px 10px 10px; }
.answer-text { font-size: 15px; }
.message-actions { border-top: 1px solid #f0f2f6; padding-top: 8px; }
.composer { padding: 16px 30px 22px; background: rgba(255, 255, 255, .96); }
.composer :deep(.el-textarea__inner) { border: 1px solid #dfe6f0; box-shadow: 0 5px 18px rgba(38, 67, 120, .07); }
.complex-plan { border-color: #cbd9f8; background: #fbfcff; }

/* A quieter, conversation-first layout: neutral panels, readable messages,
   and a centered composer instead of nested dashboard cards. */
.chat-page {
  height: calc(100vh - 112px);
  min-height: 580px;
  grid-template-columns: 248px minmax(0, 1fr) 276px;
  border: 1px solid #e7e7e9;
  border-radius: 14px;
  background: #fff;
  box-shadow: none;
}
.history-sidebar { background: #f7f7f8; border-right-color: #e7e7e9; }
.history-head { padding: 17px 14px 13px; border-bottom-color: #e7e7e9; }
.history-head strong { color: #202123; font-size: 13px; }
.history-head span { color: #737780; }
.history-head :deep(.el-button) { border-radius: 9px; }
.session-list { padding: 8px; }
.session-row { border-radius: 9px; color: #34363a; padding: 10px; }
.session-row:hover, .session-row.active { background: #e9e9eb; }
.session-row small { color: #85878d; }
.task-sidebar { background: #fff; border-left-color: #e7e7e9; padding: 8px; }
.conversation-panel { background: #fff; }
.conversation-head { padding: 14px 20px 9px; border-bottom: 0; background: #fff; }
.conversation-head h1 { color: #202123; font-size: 16px; font-weight: 600; }
.conversation-head p { color: #777980; font-size: 12px; }
.chat-head-actions :deep(.el-button) { background: transparent; border-color: #dedfe2; color: #45474c; }
.message-stream { padding: 20px clamp(12px, 2vw, 28px); background: #fff; }
.message { max-width: 1100px; margin-bottom: 20px; gap: 0; }
.avatar { display: none; }
.user-message { justify-content: flex-end; }
.bubble {
  max-width: min(78%, 680px);
  padding: 9px 14px;
  border: 0;
  border-radius: 20px;
  background: #e8f0fe;
  color: #20385f;
  font-size: 16px;
  line-height: 1.65;
}
.assistant-content { max-width: 1100px; }
.answer-text {
  padding: 0;
  border: 0;
  border-radius: 0;
  background: transparent;
  color: #28292d;
  font-size: 15px;
  line-height: 1.65;
}
.user-message-content { max-width: min(78%, 680px); min-width: 0; }
.user-message-content .bubble { width: 100%; max-width: 100%; box-sizing: border-box; }
.user-message-actions { display: flex; justify-content: flex-end; margin-top: 2px; }
.message-action-buttons { display: flex; align-items: center; gap: 4px; }
.message-actions :deep(.el-button), .user-message-actions :deep(.el-button) { color: #777980; }
.message-actions { border-top-color: #efeff0; color: #8a8c91; }
.thinking { background: #f7f7f8; border-color: #ededee; color: #3d3f43; }
.thinking span, .muted { color: #737780; }
.empty-chat { max-width: 600px; margin-top: 18vh; color: #737780; }
.empty-chat h2 { color: #202123; font-size: 27px; font-weight: 600; letter-spacing: -.02em; }
.scope-editor-hint { color: var(--el-text-color-secondary); }
:deep(.scope-tag) {
  margin: 4px 20px 0;
  padding: 8px 11px;
  border-color: #e7e7e9;
  border-radius: 11px;
  background: #f7f7f8;
  color: #34363a;
}
:deep(.scope-tag.is-default) { border-color: #e7e7e9; background: #f7f7f8; color: #34363a; }
:deep(.scope-icon), :deep(.scope-tag.is-default .scope-icon) { background: #e4e4e7; color: #53555a; }
:deep(.scope-heading span), :deep(.scope-facts small), :deep(.scope-tag.is-default .scope-heading span), :deep(.scope-tag.is-default .scope-facts small) { color: #737780; }
.composer {
  width: min(100% - 24px, 960px);
  margin: 0 auto 10px;
  padding: 8px 12px 7px;
  border: 1px solid #d8e5fa;
  border-radius: 24px;
  background: #eff5ff;
  box-shadow: 0 1px 6px rgba(37, 99, 235, .06);
}
.composer :deep(.el-textarea__inner) {
  min-height: 40px;
  padding: 6px 2px;
  border: 0;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
  color: var(--el-text-color-primary);
  font-size: 16px;
  line-height: 1.6;
}
.composer :deep(.el-textarea__inner:focus) { box-shadow: none; }
.composer :deep(.el-textarea__inner::placeholder) { color: var(--el-text-color-placeholder); font-size: 14px; }
.composer:focus-within { border-color: #a9c3f5; box-shadow: 0 0 0 2px rgba(37, 99, 235, .08); }
.composer > div:last-child { margin-top: 1px; color: #85878d; }
.composer > div:last-child :deep(.el-button) { min-width: 72px; border-radius: 18px; }
.complex-plan { border-color: #e7e7e9; background: #fafafa; }

@media (max-width: 900px) {
  .chat-page { height: calc(100vh - 84px); min-height: 500px; }
  .chat-page { grid-template-columns: minmax(0, 1fr); }
  .history-sidebar, .task-sidebar { display: none; }
  .message-stream { padding: 20px 16px; }
  .conversation-head { padding: 14px 18px 8px; }
  :deep(.scope-tag) { margin: 4px 14px 0; }
  .composer { width: calc(100% - 20px); margin-bottom: 8px; }
}

@media (min-width: 901px) and (max-width: 1050px) {
  .chat-page { grid-template-columns: 220px minmax(0, 1fr); }
  .task-sidebar { display: none; }
}
</style>
