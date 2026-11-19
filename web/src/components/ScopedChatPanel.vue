<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { ElMessage } from "element-plus";
import { api, streamChat } from "@/api";
import { isSseStartedEvent, isSseNodeStartedEvent, isSseNodeCompletedEvent, isSseInvestigationProgressEvent } from "../generated/sse-events";
import { formatInvestigationStopReason } from "../utils/format";
import { clarificationMessage, clarificationTitle } from "../utils/clarificationCopy";
import { sellerScopeChatCarrier, sellerScopePayloadExpired } from "../utils/sellerScopeCredential";
import AnalysisResultCard from "@/components/AnalysisResultCard.vue";
import { useAppStore } from "@/stores/app";
import type { AnalysisPresentation, SignedSellerScopePayload } from "@/types";

const props = defineProps<{ sellerScopePayload?: SignedSellerScopePayload | null }>();
const store = useAppStore();
const question = ref(""); const progress = ref<string[]>([]); const traceId = ref(""); const working = ref(false);
const copyingTrace = ref(false);
const currentNode = ref(""); const nodeElapsed = ref(0); const nodeStartedAt = ref(0); const clarification = ref<Record<string, unknown> | null>(null); const presentation = ref<AnalysisPresentation | null>(null);
const clarificationInput = ref("");
const progressPanel = ref<HTMLElement | null>(null);
const turns = computed(() => store.activeSession?.turns ?? []);
const clarificationCandidates = computed(() => Array.isArray(clarification.value?.ambiguous_candidates) ? clarification.value.ambiguous_candidates as Array<{ metric_id: string; display_name: string; mapping_id?: string }> : []);
const pendingMappingIds = computed(() => clarificationCandidates.value.map(item => item.mapping_id).filter((id): id is string => Boolean(id)));
const supportedMetricIds = computed(() => Array.isArray(clarification.value?.confirmed_metrics) ? clarification.value.confirmed_metrics as string[] : []);
const unavailableMetrics = computed(() => Array.isArray(clarification.value?.unavailable_metrics) ? clarification.value.unavailable_metrics as string[] : []);
function dimensionLabel(value: string): string {
  const labels: Record<string, string> = { product_category: "商品品类", product: "商品", seller: "店铺", customer_state: "客户州/地区", payment_type: "支付方式", month_position: "月内时段", customer_city: "客户城市", seller_state: "卖家州" };
  return labels[value] ?? value;
}
const clarificationDimensionLabels = computed(() => {
  const values = clarification.value?.dimension_candidates;
  return Array.isArray(values) ? values.filter((item): item is string => typeof item === "string").map(dimensionLabel) : [];
});
const semanticClarification = computed(() => clarification.value?.semantic_interpretation as { source_text?: string; display_name?: string; dimension_label?: string; interpretation?: string; requires_confirmation?: boolean } | undefined);
const latestTurn = computed(() => turns.value[0]);
const latestTurnDimensionLabels = computed(() => {
  const values = latestTurn.value?.clarification?.dimension_candidates;
  return Array.isArray(values) ? values.filter((item): item is string => typeof item === "string").map(dimensionLabel) : [];
});
const earlierTurns = computed(() => turns.value.slice(1, 6));
const latestPresentation = computed(() => presentation.value || ((latestTurn.value as any)?.presentation ?? null));
const sellerScopeIntent = computed(() => props.sellerScopePayload?.seller_scope_intent ?? null);
const sellerScopeExpired = computed(() => sellerScopePayloadExpired(props.sellerScopePayload));
const sellerScopeLabel = computed(() => {
  const intent = sellerScopeIntent.value;
  if (!intent) return "";
  const kind = intent.seller_scope_type === "all" ? "全平台 ALL"
    : intent.seller_scope_type === "selected" ? "指定商户子集" : "单店";
  const sellers = intent.seller_scope_type === "all" ? "全部已授权商户" : intent.seller_ids.join("、");
  return `${kind} · ${sellers} · ${intent.start_date} 至 ${intent.end_date}`;
});
const nodeText: Record<string, string> = { init_state: "初始化", resolve_explorer_context: "验证范围", load_session_memory: "加载记忆", triage: "识别问题", retrieve_analysis_memories: "检索历史结论", metric_definition: "读取口径", data_query: "查询数据", root_cause: "整理证据", report: "生成报告", generate_answer: "生成结论", critic: "审核结论", human_review: "等待审核", persist_trace: "保存记录", persist_memory_p0: "保存记忆" };
watch(() => progress.value.length, async () => {
  await nextTick();
  if (progressPanel.value) progressPanel.value.scrollTop = progressPanel.value.scrollHeight;
});
async function submit(metricConfirmation: string[] | null = null, semanticMappingIds: string[] = [], skippedSemanticMappingIds: string[] = []) {
  const message = question.value.trim(); const scope = store.activeSession?.scope; const sellerPayload = props.sellerScopePayload;
  if (!message || (!scope && !sellerPayload) || working.value) return;
  if (sellerPayload && sellerScopeExpired.value) return ElMessage.warning("seller 范围凭证已过期，请返回数据浏览重新校验并签发");
  if (sellerPayload && (semanticMappingIds.length || skippedSemanticMappingIds.length)) {
    return ElMessage.warning("seller 范围下暂不支持语义映射确认；请返回数据浏览重新选择指标范围");
  }
  const confirmedMetricIds = Array.isArray(metricConfirmation) ? metricConfirmation : undefined;
  // session.id 是聊天窗口级标识；id 是本轮消息标识；threadId 因此每次提交都会变化。
  // 受控范围面板与普通聊天面板遵守同一套 checkpoint、记忆和聊天历史生命周期。
  const session = store.activeSession ?? store.createSession(); const id = crypto.randomUUID(); const threadId = `${session.id}-${id}`; const startedAt = new Date().toISOString();
  working.value = true; progress.value = []; currentNode.value = ""; nodeElapsed.value = 0; nodeStartedAt.value = 0; clarification.value = null; clarificationInput.value = ""; presentation.value = null;
  try {
    for (const mappingId of semanticMappingIds) {
      await api.confirmSemanticMapping(store.auth, mappingId, { chat_session_id: session.id, explorer_context: scope ? { ...scope } : undefined });
    }
    for (const mappingId of skippedSemanticMappingIds) {
      await api.skipSemanticMapping(store.auth, mappingId, { chat_session_id: session.id, explorer_context: scope ? { ...scope } : undefined });
    }
    // thread_id 只关联本次图执行；chat_session_id 是跨轮会话的唯一标识。
    let scopeCarrier;
    try {
      scopeCarrier = sellerScopeChatCarrier(sellerPayload, scope);
    } catch {
      working.value = false;
      return ElMessage.warning("seller 范围凭证已过期，请返回数据浏览重新校验并签发");
    }
    const result = await streamChat(store.auth, {
      message, thread_id: threadId,
      // The signed seller scope is a distinct carrier and cannot be combined
      // with either legacy Explorer field.
      ...scopeCarrier,
      metric_confirmation: confirmedMetricIds,
      semantic_mapping_ids: semanticMappingIds.length ? semanticMappingIds : undefined,
      skipped_semantic_mapping_ids: skippedSemanticMappingIds.length ? skippedSemanticMappingIds : undefined,
      chat_session_id: session.id, chat_turn_id: id,
    }, (event) => {
      if (isSseStartedEvent(event) && event.trace_id) traceId.value = event.trace_id;
      if (isSseNodeStartedEvent(event)) { currentNode.value = nodeText[event.node] ?? event.node; nodeStartedAt.value = Date.now(); nodeElapsed.value = 0; }
      if (isSseNodeCompletedEvent(event)) { progress.value.push(nodeText[event.node] ?? event.node); nodeElapsed.value = 0; }
      if (isSseInvestigationProgressEvent(event)) {
        const action = event.current_action ? `，当前动作 ${event.current_action}` : "";
        const stop = event.stop_reason ? `，停止原因 ${formatInvestigationStopReason(event.stop_reason)}` : "";
        progress.value.push(`归因调查 ${event.completed_actions}/6，剩余 ${event.remaining_budget} 项${action}${stop}`);
      }
      if (event.event === "heartbeat") { if (nodeStartedAt.value > 0) nodeElapsed.value = Math.round((Date.now() - nodeStartedAt.value) / 1000); }
    });
    const failed = result.error === true; clarification.value = result.clarification ?? null;
    presentation.value = (result as any).presentation ?? null;
    traceId.value = result.trace_id; currentNode.value = "";
    store.addTurn({ id, threadId, message, answer: failed ? undefined : result.answer, presentation: presentation.value ?? undefined, error: failed ? (result.error_detail ?? "图执行异常") : undefined, taskType: result.task_type, traceId: result.trace_id, createdAt: startedAt, status: failed ? "failed" : result.event === "suspended" ? "suspended" : "completed", interactionState: result.event === "suspended" ? result.interaction_state : undefined, clarification: result.clarification ?? null });
    if (failed) ElMessage.warning(result.error_detail ?? "分析过程中出现异常，部分数据不可用。");
  } catch (error) { const detail = error instanceof Error ? error.message : "分析请求失败"; store.addTurn({ id, threadId, message, error: detail, taskType: "failed", traceId: "", createdAt: startedAt, status: "failed" }); ElMessage.error(detail); }
  finally { working.value = false; }
}
function viewTrace(tid: string) { traceId.value = tid; }
async function copyTraceId(): Promise<void> {
  const value = (latestTurn.value?.traceId || traceId.value).trim();
  if (!value || copyingTrace.value) return;
  try {
    copyingTrace.value = true;
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = value;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      if (!document.execCommand("copy")) throw new Error("clipboard unavailable");
      textarea.remove();
    }
    ElMessage.success("Trace 编号已复制");
  } catch {
    ElMessage.warning("复制失败，请手动选择 Trace 编号");
  } finally {
    copyingTrace.value = false;
  }
}
function cancelClarification() { clarification.value = null; question.value = ""; }
function submitClarificationText() {
  const note = clarificationInput.value.trim();
  if (!note || working.value) return ElMessage.warning("请先填写确认说明");
  if (/^(终止|取消|放弃|结束)/.test(note)) return cancelClarification();
  if (/只分析|仅分析|已支持/.test(note) && supportedMetricIds.value.length) {
    return submit(supportedMetricIds.value, [], pendingMappingIds.value);
  }
  const candidate = clarificationCandidates.value.find((item) => note.includes(item.display_name) || note.includes(item.metric_id));
  if (candidate) return submit([candidate.metric_id], candidate.mapping_id ? [candidate.mapping_id] : []);
  // The explorer scope already contains a server-validated metric ID. When
  // the user presses the confirmation button without a candidate card, send
  // that ID explicitly instead of appending more natural-language text and
  // sending the planner through the same ambiguity path again.
  const selectedMetric = String(store.activeSession?.scope?.metric_name ?? "");
  if (selectedMetric) return submit([selectedMetric]);
  question.value = `${question.value.trim()}\n补充确认：${note}`;
  return submit();
}
</script>
<template>
  <el-card shadow="never" class="scoped-chat-panel"><template #header><div class="panel-title"><span>向当前范围的 Agent 提问</span><el-tag type="success">范围已锁定</el-tag></div></template>
    <p v-if="sellerScopeIntent" class="panel-hint">{{ sellerScopeLabel }}。问题中的文字不能修改商户集合或日期；服务端按已签发 scope payload 授权。</p>
    <p v-else class="panel-hint">提问时数据仍保留在左侧。Agent 只能访问当前店铺与时间范围内的受控数据。</p>
    <el-alert v-if="sellerScopeIntent?.seller_scope_type === 'selected'" title="未归属 seller 行来自整个日期范围，单独展示且不进入 selected total。" type="info" :closable="false" show-icon />
    <el-alert v-if="sellerScopeIntent?.seller_scope_type === 'all'" :title="sellerScopeIntent.include_unattributed ? '未归属 seller 行来自整个日期范围，并按显式选择计入 ALL total。' : '未归属 seller 行来自整个日期范围，默认不进入 ALL total。'" type="info" :closable="false" show-icon />
    <el-alert v-if="sellerScopeIntent" title="order_count 的 seller 分组是 distinct orders per seller，可能重叠、不能相加；total 是独立计算的 distinct 订单并集。" type="info" :closable="false" show-icon />
    <el-alert v-if="sellerScopeExpired" title="seller 范围凭证已过期，请返回数据浏览重新校验。" type="warning" :closable="false" show-icon />
    <el-input v-model="question" type="textarea" :rows="5" placeholder="例如：GMV 的趋势如何？按商品品类拆解。" :disabled="sellerScopeExpired" @keydown.ctrl.enter="submit()" />
    <div class="toolbar"><span>Ctrl / ⌘ + Enter 提交</span><el-button type="primary" :loading="working" :disabled="sellerScopeExpired" @click="submit()">开始分析</el-button></div>
    <el-alert v-if="clarification" type="warning" :closable="false" show-icon style="margin-top:12px">
      <template #title>{{ clarificationTitle(clarification, Boolean(semanticClarification)) }}</template>
      <p v-if="semanticClarification">系统理解：{{ semanticClarification.interpretation || semanticClarification.display_name }}<span v-if="semanticClarification.dimension_label">（维度：{{ semanticClarification.dimension_label }}）</span></p>
      <p v-if="unavailableMetrics.length">当前不可用：{{ unavailableMetrics.join("、") }}</p>
      <p>{{ clarificationMessage(clarification, "未执行数据查询，请确认要分析的指标。") }}</p>
      <p v-if="clarificationDimensionLabels.length">候选维度：{{ clarificationDimensionLabels.join("、") }}。系统未自动选择，请在问题中明确一个维度。</p>
      <el-space wrap>
        <el-button v-if="supportedMetricIds.length" type="primary" size="small" @click="submit(supportedMetricIds, [], pendingMappingIds)">只分析已支持指标</el-button>
        <el-button v-for="candidate in clarificationCandidates" :key="candidate.mapping_id || candidate.metric_id" size="small" @click="submit([candidate.metric_id], candidate.mapping_id ? [candidate.mapping_id] : [])">确认使用：{{ candidate.display_name }}</el-button>
        <el-button size="small" @click="cancelClarification">终止本次分析</el-button>
      </el-space>
      <el-input v-model="clarificationInput" class="clarification-input" type="textarea" :rows="2" placeholder="补充确认，例如：只分析已支持指标，或改用店铺平均延迟送达天数" />
      <div class="clarification-actions"><el-button size="small" type="primary" :disabled="!clarificationInput.trim()" @click="submitClarificationText">提交确认说明</el-button></div>
    </el-alert>
    <el-divider>执行过程</el-divider>
    <template v-if="progress.length || currentNode">
      <div v-if="currentNode" class="stage-badge"><el-tag type="warning" effect="dark" class="running-tag">{{ currentNode }}<span v-if="nodeElapsed > 0">  · {{ nodeElapsed }}s</span></el-tag></div>
      <div ref="progressPanel" class="progress-scroll"><el-timeline><el-timeline-item v-for="(item, index) in progress" :key="`${index}-${item}`">{{ item }}</el-timeline-item></el-timeline></div>
      <div v-if="progress.length" class="stage-badge"><el-tag type="info" effect="plain" size="small">已完成 {{ progress.length }} 个节点</el-tag></div>
    </template>
    <el-empty v-else description="等待分析请求" :image-size="48" />
    <el-divider>最终结论</el-divider><el-empty v-if="!latestTurn && !presentation" description="结论将在这里显示" :image-size="48" />
    <div class="analysis-scroll"><el-alert v-if="latestTurn?.interactionState === 'awaiting_human_input'" title="当前请求已暂停，等待人工审核或后续确认。" type="info" :closable="false" show-icon /><el-alert v-if="latestTurn?.clarification" type="warning" :closable="false" show-icon><template #title>{{ clarificationTitle(latestTurn.clarification) }}</template><p>{{ clarificationMessage(latestTurn.clarification) }}</p><p v-if="latestTurnDimensionLabels.length">候选维度：{{ latestTurnDimensionLabels.join("、") }}。请明确一个维度后重新提问。</p></el-alert><AnalysisResultCard v-if="latestPresentation || latestTurn?.answer" :presentation="latestPresentation" :answer="latestTurn?.answer ?? ''" :trace-id="traceId" @view-trace="viewTrace" />
    <template v-else-if="latestTurn?.error"><p class="turn-question">{{ latestTurn.message }}</p><el-alert :title="latestTurn.error" type="error" show-icon :closable="false" /></template></div>
    <div v-if="latestTurn && (latestTurn.traceId || traceId)" class="trace-reference">
      <span>查看本次 Trace：</span>
      <code class="trace-reference-id" role="button" tabindex="0" title="点击查看 Trace" @click="viewTrace(latestTurn.traceId || traceId)" @keydown.enter="viewTrace(latestTurn.traceId || traceId)">{{ latestTurn.traceId || traceId }}</code>
      <el-button text type="primary" size="small" :loading="copyingTrace" @click="copyTraceId">复制</el-button>
    </div>
    <template v-if="earlierTurns.length"><el-divider>本会话历史</el-divider><div v-for="turn in earlierTurns" :key="turn.id" class="history-turn"><strong>{{ turn.message }}</strong><p>{{ turn.answer ?? `请求失败：${turn.error ?? '未知错误'}` }}</p></div></template>
  </el-card>
</template>

<style scoped>
.progress-scroll { max-height: 92px; overflow-y: auto; overscroll-behavior: contain; padding-right: 4px; }
.progress-scroll :deep(.el-timeline) { margin: 0; padding-left: 8px; }
.progress-scroll :deep(.el-timeline-item) { padding-bottom: 10px; }
.analysis-scroll { max-height: min(68vh, 760px); overflow-y: auto; overscroll-behavior: contain; padding-right: 6px; }
.clarification-input { margin-top: 12px; }
.clarification-actions { margin-top: 8px; text-align: right; }
.trace-reference { display: flex; align-items: center; gap: 4px; margin-top: 10px; flex-wrap: wrap; font-size: 12px; color: var(--el-text-color-secondary); }
.trace-reference-id { user-select: text; cursor: pointer; color: var(--el-color-primary); font-family: monospace; word-break: break-all; }
</style>
