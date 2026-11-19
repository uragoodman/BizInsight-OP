<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { api } from "@/api";
import { useAppStore } from "@/stores/app";
import MarkdownReport from "@/components/MarkdownReport.vue";
import TrendChart from "@/components/TrendChart.vue";
import ContributionChart from "@/components/ContributionChart.vue";
import type { MetricDefinition, SellerOption } from "@/types";

type ReportScopeType = "platform" | "seller";

const store = useAppStore();
const metrics = ref<MetricDefinition[]>([]);
const reports = ref<Array<Record<string, any>>>([]);
const selected = ref<Record<string, any> | null>(null);
const loading = ref(false);
const reportView = ref<"preview" | "edit">("preview");
const sellers = ref<SellerOption[]>([]);
const allowedScopes = ref<ReportScopeType[]>(["platform", "seller"]);
let sellerSearchTimer: number | undefined;
let reportUiRestored = false;

function reportUiKey() {
  return `bizinsight.reports.ui.v1.${store.profile.id}`;
}
function saveReportUiState() {
  localStorage.setItem(reportUiKey(), JSON.stringify({
    selectedReportId: selected.value?.report_id ?? null,
    selectedVersion: selected.value?.version ?? null,
    form: form.value,
  }));
}
function restoreReportUiState() {
  try {
    const raw = JSON.parse(localStorage.getItem(reportUiKey()) ?? "null");
    if (!raw || typeof raw !== "object") return null;
    if (raw.form && typeof raw.form === "object") {
      const saved = raw.form as Record<string, unknown>;
      if (saved.start_date) form.value.start_date = String(saved.start_date);
      if (saved.end_date) form.value.end_date = String(saved.end_date);
      if (saved.granularity === "week" || saved.granularity === "month") form.value.granularity = saved.granularity;
      if (saved.scope_type === "platform" || saved.scope_type === "seller") form.value.scope_type = saved.scope_type;
      if (saved.seller_id === null || typeof saved.seller_id === "string") form.value.seller_id = saved.seller_id as string | null;
      if (Array.isArray(saved.metric_names)) form.value.metric_names = saved.metric_names.map(String);
    }
    return typeof raw.selectedReportId === "string" ? raw.selectedReportId : null;
  } catch {
    return null;
  }
}

const form = ref({
  start_date: "2018-07-30",
  end_date: "2018-08-05",
  granularity: "week" as "week" | "month",
  scope_type: "platform" as ReportScopeType,
  seller_id: null as string | null,
  metric_names: ["gmv", "order_count", "avg_order_value"],
  include_charts: true,
});

function parseIsoDate(value: string) {
  const [year, month, day] = value.split("-").map(Number);
  return new Date(year, month - 1, day);
}
function formatIsoDate(value: Date) {
  return `${value.getFullYear()}-${String(value.getMonth() + 1).padStart(2, "0")}-${String(value.getDate()).padStart(2, "0")}`;
}
function addDays(value: Date, days: number) {
  const result = new Date(value);
  result.setDate(result.getDate() + days);
  return result;
}
function monthBounds(value: Date) {
  return {
    start: new Date(value.getFullYear(), value.getMonth(), 1),
    end: new Date(value.getFullYear(), value.getMonth() + 1, 0),
  };
}
function weekBounds(value: Date) {
  const monday = addDays(value, -((value.getDay() + 6) % 7));
  return { start: monday, end: addDays(monday, 6) };
}

function alignPeriod(value: string) {
  if (!value) return;
  const bounds =
    form.value.granularity === "week"
      ? weekBounds(parseIsoDate(value))
      : monthBounds(parseIsoDate(value));
  form.value.start_date = formatIsoDate(bounds.start);
  form.value.end_date = formatIsoDate(bounds.end);
}
function onStartDateChange() {
  alignPeriod(form.value.start_date);
}
function onEndDateChange() {
  alignPeriod(form.value.end_date);
}
function onGranularityChange() {
  alignPeriod(form.value.start_date);
}

function reportPayload(
  report: Record<string, any> | null,
): Record<string, any> | null {
  if (!report) return null;
  const value = report.report_json ?? report.report;
  if (typeof value === "string") {
    try {
      return JSON.parse(value) as Record<string, any>;
    } catch {
      return null;
    }
  }
  return value && typeof value === "object"
    ? (value as Record<string, any>)
    : null;
}
function reportScopeLabel(report: Record<string, any>) {
  const scope = reportPayload(report)?.scope;
  return scope?.scope_type === "seller"
    ? `单店 · ${scope.seller_label ?? scope.seller_id}`
    : "全平台";
}
const reportDataCharts = computed<
  Array<{ chart: Record<string, any>; index: number }>
>(() => {
  const charts = reportPayload(selected.value)?.charts;
  return Array.isArray(charts)
    ? charts
        .map((chart, index) => ({ chart, index }))
        .filter(
          ({ chart }) =>
            chart?.status === "ok" &&
            Array.isArray(chart?.data_points) &&
            chart.data_points.length,
        )
    : [];
});
const previewContent = computed(() =>
  String(selected.value?.content ?? "").replace(
    /\n## 图表\n[\s\S]*?(?=\n## |$)/,
    "",
  ),
);
const isSellerScope = computed(() => form.value.scope_type === "seller");

// ---- AI 解读（独立洞察制品，附着于 report_version）----
const insights = ref<Array<Record<string, any>>>([]);
const insightGenerating = ref(false);
const insightForm = ref({
  angle: "overview" as
    "overview" | "contribution" | "anomaly" | "target" | "risk",
  audience: "operations" as "executive" | "operations" | "technical",
  style: "concise" as "formal" | "concise" | "conversational",
  max_insights: 5,
  min_words: 40,
  max_words: 200,
  include_recommendations: true,
  focus_metrics: [] as string[],
});
const latestInsight = computed(() => insights.value[0] ?? null);
const latestInsightItems = computed(() => {
  const raw = latestInsight.value?.insights_json;
  return Array.isArray(raw) ? raw : [];
});
const insightGeneratingNow = computed(
  () => latestInsight.value?.status === "generating",
);
const insightStage = ref("");
let insightAbort: AbortController | undefined;
async function startInsightSse(insightVersion: number) {
  if (!selected.value) return;
  insightAbort?.abort();
  const controller = new AbortController();
  insightAbort = controller;
  try {
    await api.streamInsightProgress(
      store.auth,
      String(selected.value.report_id),
      Number(selected.value.version),
      insightVersion,
      (event: Record<string, unknown>) => {
        if (event.event === "stage")
          insightStage.value = String(event.message ?? event.stage ?? "");
        else if (
          event.event === "completed" ||
          event.event === "failed" ||
          event.event === "timeout"
        )
          controller.abort();
      },
      controller.signal,
    );
  } catch {
    // SSE 断开/被中止 → 依赖 2s 轮询兜底
  }
}
let insightPollTimer: number | undefined;
// 2s 轮询兜底：SSE 失效/断开时仍能拉到终态，避免卡在"生成中"。
function scheduleInsightPoll() {
  window.clearTimeout(insightPollTimer);
  if (!selected.value) return;
  insightPollTimer = window.setTimeout(async () => {
    await loadInsights();
    if (insightGeneratingNow.value) scheduleInsightPoll();
  }, 2000);
}
async function loadInsights() {
  if (!selected.value) return;
  try {
    insights.value = await api.reportInsights(
      store.auth,
      String(selected.value.report_id),
      Number(selected.value.version),
    );
  } catch {
    insights.value = [];
  }
  if (insightGeneratingNow.value) scheduleInsightPoll();
}
// 生成 AI 解读：POST 秒回 generating 占位 → 开 SSE 看阶段进度 → 2s 轮询兜底直到终态。
async function generateInsight() {
  if (!selected.value) return;
  insightGenerating.value = true;
  insightStage.value = "";
  try {
    const result = await api.createReportInsights(
      store.auth,
      String(selected.value.report_id),
      Number(selected.value.version),
      { ...insightForm.value },
    );
    ElMessage.success("AI 解读已开始生成");
    startInsightSse(Number(result.insight_version));
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "AI 解读生成失败");
  } finally {
    insightGenerating.value = false;
    await loadInsights();
  }
}
async function reviewInsight(action: "approve" | "reject") {
  const item = latestInsight.value;
  if (!item || !selected.value) return;
  const { value } = await ElMessageBox.prompt(
    action === "reject" ? "请输入驳回原因" : "审核备注（可留空）",
    "提交洞察审核",
    { inputValue: "", inputPattern: /.*/, confirmButtonText: "提交" },
  );
  try {
    await api.reviewReportInsight(
      store.auth,
      String(selected.value.report_id),
      Number(selected.value.version),
      Number(item.insight_version),
      {
        insight_version: item.insight_version,
        action,
        actor: store.profile.id,
        reason: value,
      },
    );
    ElMessage.success("洞察审核已提交");
  } catch (error) {
    if (error !== "cancel")
      ElMessage.error(error instanceof Error ? error.message : "洞察审核失败");
  } finally {
    await loadInsights();
  }
}
const attributionEvidenceCount = computed(() => {
  const raw = latestInsight.value?.attribution_evidence;
  return raw && typeof raw === "object" ? Object.keys(raw).length : 0;
});
const attributionEvidenceRows = computed(() => {
  const raw = latestInsight.value?.attribution_evidence;
  if (!raw || typeof raw !== "object") return [];
  return Object.entries(raw as Record<string, Record<string, any>>).map(
    ([evidenceId, record]) => {
      const typed = record?.investigation_evidence ?? record;
      return {
        evidenceId,
        kind: record?.kind ?? typed?.evidence_type ?? "attribution",
        metric: record?.metric_name ?? typed?.metric_id ?? "指标",
        definition: metrics.value.find((item) => item.metric_name === (record?.metric_name ?? typed?.metric_id))?.description ?? "",
        unit: metrics.value.find((item) => item.metric_name === (record?.metric_name ?? typed?.metric_id))?.unit ?? typed?.unit ?? "",
        method: typed?.method_id ?? record?.display_method ?? "未记录",
        claim: typed?.claim ?? record?.reason ?? "",
        values: typed?.values ?? record,
        comparison: typed?.comparison ?? {},
        dataQuality: typed?.data_quality ?? {},
        limitations: typed?.limitations ?? [],
      };
    },
  );
});
function prettyEvidence(value: unknown) {
  return JSON.stringify(value, null, 2);
}

async function loadMetrics(scope = form.value.scope_type) {
  metrics.value = await api.metrics(store.auth, scope);
  const supported = new Set(
    metrics.value
      .filter((metric) => metric.available !== false)
      .map((metric) => metric.metric_name),
  );
  const retained = form.value.metric_names.filter((name) =>
    supported.has(name),
  );
  if (retained.length !== form.value.metric_names.length)
    ElMessage.info("已移除当前范围不支持的指标");
  form.value.metric_names = retained.length
    ? retained
    : metrics.value.slice(0, 3).map((metric) => metric.metric_name);
}
function searchSellers(keyword: string) {
  window.clearTimeout(sellerSearchTimer);
  sellerSearchTimer = window.setTimeout(async () => {
    try {
      sellers.value = (await api.sellers(store.auth, keyword)).items;
    } catch (error) {
      ElMessage.error(error instanceof Error ? error.message : "店铺搜索失败");
    }
  }, 250);
}
async function onScopeChange() {
  if (!isSellerScope.value) form.value.seller_id = null;
  await loadMetrics();
  if (isSellerScope.value && !form.value.seller_id) searchSellers("");
}
async function load() {
  try {
    const savedReportId = reportUiRestored ? null : restoreReportUiState();
    reportUiRestored = true;
    const [capabilities, reportList] = await Promise.all([
      api.reportCapabilities(store.auth),
      api.reports(store.auth),
    ]);
    allowedScopes.value = capabilities.allowed_scopes;
    reports.value = reportList;
    if (capabilities.locked_seller) {
      form.value.scope_type = "seller";
      form.value.seller_id = capabilities.locked_seller.seller_id;
      sellers.value = [capabilities.locked_seller];
    }
    await loadMetrics();
    const restoreId = savedReportId && reportList.some((item) => item.report_id === savedReportId)
      ? savedReportId
      : reportList[0]?.report_id;
    if (restoreId) await select(String(restoreId));
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "读取报告失败");
  }
}
async function select(id: string) {
  try {
    selected.value = (await api.report(store.auth, id)) as Record<string, any>;
    reportView.value = "preview";
    await loadInsights();
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "读取报告失败");
  }
}
async function create() {
  if (!form.value.metric_names.length)
    return ElMessage.warning("至少选择一个指标");
  if (isSellerScope.value && !form.value.seller_id)
    return ElMessage.warning("请选择一家店铺");
  loading.value = true;
  try {
    const result = await api.createReport(store.auth, form.value);
    const createdReportId = String((result.report as any)?.report_id ?? "");
    selected.value = {
      ...result,
      report_id: createdReportId,
      report_json: result.report,
      content: result.content,
      version: result.version,
      status: result.status,
    };
    saveReportUiState();
    reportView.value = "preview";
    ElMessage.success("报告草稿已生成");
    await load();
    if (createdReportId && selected.value?.report_id !== createdReportId) {
      await select(createdReportId);
    }
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "生成失败");
  } finally {
    loading.value = false;
  }
}
async function review(action: "approve" | "edit" | "reject") {
  if (!selected.value) return;
  const { value } = await ElMessageBox.prompt(
    action === "reject" ? "请输入驳回原因" : "审核备注（可留空）",
    "提交审核",
    { inputValue: "", inputPattern: /.*/, confirmButtonText: "提交" },
  );
  try {
    await api.reviewReport(store.auth, String(selected.value.report_id), {
      version: selected.value.version,
      action,
      actor: store.profile.id,
      reason: value,
      ...(action === "edit" ? { edited_content: selected.value.content } : {}),
    });
    await select(String(selected.value.report_id));
    await load();
  } catch (error) {
    if (error !== "cancel")
      ElMessage.error(error instanceof Error ? error.message : "审核失败");
  }
}
async function publish() {
  if (!selected.value) return;
  try {
    await ElMessageBox.confirm("发布后将产生业务事件，确定继续？", "发布报告", {
      type: "warning",
    });
    await api.publishReport(store.auth, String(selected.value.report_id), {
      version: selected.value.version,
      actor: store.profile.id,
    });
    await select(String(selected.value.report_id));
    await load();
  } catch (error) {
    if (error !== "cancel")
      ElMessage.error(error instanceof Error ? error.message : "发布失败");
  }
}

async function exportSelectedReport() {
  if (!selected.value || selected.value.status !== "published") return;
  try {
    const blob = await api.exportReport(store.auth, String(selected.value.report_id), Number(selected.value.version));
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${String(selected.value.report_id)}.md`;
    anchor.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "导出报告失败");
  }
}

onMounted(load);
watch(form, saveReportUiState, { deep: true });
watch(
  () => [selected.value?.report_id, selected.value?.version],
  saveReportUiState,
);
watch(() => [selected.value?.report_id, selected.value?.version], loadInsights);
onBeforeUnmount(() => {
  window.clearTimeout(sellerSearchTimer);
  window.clearTimeout(insightPollTimer);
  insightAbort?.abort();
});
</script>

<template>
  <div class="page">
    <div class="page-title">
      <div>
        <h1>周期报告</h1>
        <p>按全平台或单店范围生成、审核并发布经营报告。</p>
      </div>
      <el-button @click="load">刷新列表</el-button>
    </div>
    <el-row :gutter="16">
      <el-col :lg="9">
        <el-card shadow="never"
          ><template #header>生成报告草稿</template>
          <el-form label-position="top">
            <el-form-item label="报告范围"
              ><el-radio-group
                v-model="form.scope_type"
                :disabled="allowedScopes.length === 1"
                @change="onScopeChange"
                ><el-radio
                  v-if="allowedScopes.includes('platform')"
                  value="platform"
                  >全平台</el-radio
                ><el-radio
                  v-if="allowedScopes.includes('seller')"
                  value="seller"
                  >单店</el-radio
                ></el-radio-group
              ></el-form-item
            >
            <el-form-item v-if="isSellerScope" label="店铺"
              ><el-select
                v-model="form.seller_id"
                filterable
                remote
                reserve-keyword
                :disabled="allowedScopes.length === 1"
                placeholder="搜索店铺 ID 或选择店铺"
                :remote-method="searchSellers"
                @visible-change="
                  (visible: boolean) => visible && searchSellers('')
                "
                ><el-option
                  v-for="seller in sellers"
                  :key="seller.seller_id"
                  :label="`${seller.label} · ${seller.order_count ?? 0} 单`"
                  :value="seller.seller_id"
              /></el-select>
              <p class="field-hint">
                店铺名称来自城市、州和匿名化 ID；完整 ID 会写入报告口径。
              </p></el-form-item
            >
            <el-form-item label="起止日期"
              ><el-date-picker
                v-model="form.start_date"
                type="date"
                value-format="YYYY-MM-DD"
                @change="onStartDateChange" /><span class="date-separator"
                >至</span
              ><el-date-picker
                v-model="form.end_date"
                type="date"
                value-format="YYYY-MM-DD"
                @change="onEndDateChange"
            /></el-form-item>
            <p class="period-hint">
              周报固定为周一至周日；月报固定为每月 1
              日至月末。选择任意日期后会自动对齐。
            </p>
            <el-form-item label="周期粒度"
              ><el-radio-group
                v-model="form.granularity"
                @change="onGranularityChange"
                ><el-radio value="week">周报</el-radio
                ><el-radio value="month">月报</el-radio></el-radio-group
              ></el-form-item
            >
            <el-form-item label="指标"
              ><el-select v-model="form.metric_names" multiple
                ><el-option
                  v-for="metric in metrics"
                  :key="metric.metric_name"
                  :disabled="metric.available === false"
                  :label="
                    metric.display_name
                      ? `${metric.display_name} (${metric.metric_name})${metric.available === false ? ' · 当前范围不可用' : ''}`
                      : metric.metric_name
                  "
                  :value="metric.metric_name" /></el-select
            ></el-form-item>
            <el-checkbox v-model="form.include_charts"
              >生成趋势与贡献图</el-checkbox
            ><el-button
              class="block-button"
              type="primary"
              :loading="loading"
              @click="create"
              >生成报告草稿</el-button
            >
          </el-form>
        </el-card>
        <el-card shadow="never" class="section"
          ><template #header>报告列表</template
          ><el-empty v-if="!reports.length" description="暂无报告" /><el-menu
            v-else
            @select="select"
            ><el-menu-item
              v-for="report in reports"
              :key="report.report_id"
              :index="report.report_id"
              ><span
                >{{ reportScopeLabel(report) }} · {{ report.report_id }} · v{{
                  report.version
                }}</span
              ></el-menu-item
            ></el-menu
          ></el-card
        >
      </el-col>
      <el-col :lg="15"
        ><el-empty
          v-if="!selected"
          description="选择或生成报告后，在此查看与审核。"
        /><el-card v-else shadow="never"
          ><template #header
            ><div class="card-header">
              <span
                >{{ reportScopeLabel(selected) }} · {{ selected.report_id }} ·
                v{{ selected.version }} · {{ selected.status }}</span
              >
              <div>
                <el-button
                  size="small"
                  :type="reportView === 'preview' ? 'primary' : 'default'"
                  @click="reportView = 'preview'"
                  >预览</el-button
                ><el-button
                  v-if="selected.status === 'draft'"
                  size="small"
                  :type="reportView === 'edit' ? 'primary' : 'default'"
                  @click="reportView = 'edit'"
                  >编辑 Markdown</el-button
                ><el-button
                  v-if="selected.status === 'draft'"
                  size="small"
                  @click="review('approve')"
                  >审核通过</el-button
                ><el-button
                  v-if="selected.status === 'draft'"
                  size="small"
                  @click="review('edit')"
                  >保存编辑并通过</el-button
                ><el-button
                  v-if="selected.status === 'draft'"
                  size="small"
                  type="danger"
                  @click="review('reject')"
                  >驳回</el-button
                ><el-button
                  v-if="selected.status === 'approved'"
                  size="small"
                  type="success"
                  @click="publish"
                  >发布</el-button
                ><el-button
                  v-if="selected.status === 'published'"
                  size="small"
                  type="primary"
                  @click="exportSelectedReport"
                  >导出 Markdown</el-button
                >
              </div>
            </div></template
          ><el-input
            v-if="reportView === 'edit'"
            v-model="selected.content"
            type="textarea"
            :rows="28"
          /><template v-else
            ><MarkdownReport :content="previewContent" />
            <section v-if="reportDataCharts.length" class="report-charts">
              <h2>图表</h2>
              <el-row :gutter="16">
                <el-col v-for="item in reportDataCharts" :key="`data-${item.index}`" :xs="24" :md="12">
                  <div v-if="item.chart.chart_type === 'trend'" class="report-chart">
                    <TrendChart :data="item.chart.data_points" :metric="item.chart.metric_name || '指标趋势'" :unit="item.chart.unit || ''" />
                    <div class="chart-evidence">证据：{{ (item.chart.evidence_ids || []).join('、') }}</div>
                  </div><div v-else class="report-chart"><ContributionChart :data="item.chart.data_points" :metric="item.chart.metric_name || '变化贡献'" /><div class="chart-evidence">证据：{{ (item.chart.evidence_ids || []).join('、') }}</div></div>
                </el-col>
              </el-row>
            </section></template
          >
          <section v-if="reportView === 'preview'" class="report-insights">
            <div class="insights-header">
              <h2>AI 解读</h2>
              <div v-if="selected.status !== 'published'" class="insights-form">
                <el-select
                  v-model="insightForm.angle"
                  size="small"
                  style="width: 118px"
                  ><el-option label="整体概览" value="overview" /><el-option
                    label="变化归因"
                    value="contribution" /><el-option
                    label="异常聚焦"
                    value="anomaly" /><el-option
                    label="目标差距"
                    value="target" /><el-option
                    label="风险建议"
                    value="risk" /></el-select
                ><el-select
                  v-model="insightForm.audience"
                  size="small"
                  style="width: 96px"
                  ><el-option label="管理层" value="executive" /><el-option
                    label="运营"
                    value="operations" /><el-option
                    label="技术"
                    value="technical" /></el-select
                ><el-select
                  v-model="insightForm.style"
                  size="small"
                  style="width: 88px"
                  ><el-option label="正式" value="formal" /><el-option
                    label="简洁"
                    value="concise" /><el-option
                    label="口语"
                    value="conversational" /></el-select
                ><el-button
                  size="small"
                  type="primary"
                  :loading="insightGenerating"
                  @click="generateInsight"
                  >生成 AI 解读</el-button
                ><el-popover placement="bottom-end" :width="340" trigger="click"
                  ><template #reference
                    ><el-button size="small">高级参数</el-button></template
                  >
                  <div class="insight-config">
                    <el-form label-position="top" size="small"
                      ><el-form-item label="最多洞察条数"
                        ><el-input-number
                          v-model="insightForm.max_insights"
                          :min="1"
                          :max="10" /></el-form-item
                      ><el-form-item label="每条字数区间"
                        ><div class="word-range">
                          <el-input-number
                            v-model="insightForm.min_words"
                            :min="1"
                            :max="500"
                          />
                          至
                          <el-input-number
                            v-model="insightForm.max_words"
                            :min="1"
                            :max="500"
                          /></div></el-form-item
                      ><el-form-item label="重点指标（留空=全部）"
                        ><el-select
                          v-model="insightForm.focus_metrics"
                          multiple
                          style="width: 100%"
                          ><el-option
                            v-for="metric in metrics"
                            :key="metric.metric_name"
                            :value="metric.metric_name"
                            :label="
                              metric.display_name
                                ? `${metric.display_name} (${metric.metric_name})`
                                : metric.metric_name
                            " /></el-select></el-form-item
                      ><el-form-item
                        ><el-checkbox
                          v-model="insightForm.include_recommendations"
                          >包含建议</el-checkbox
                        ></el-form-item
                      ></el-form
                    >
                  </div></el-popover
                >
              </div>
              <el-tag v-else size="small" type="info"
                >已发布 · 洞察已钉死在发布版本</el-tag
              >
            </div>
            <p class="insights-hint">
              AI
              解读为独立洞察制品：每条事实性结论必须引用报告证据（evidence_id），经服务端白名单与
              Critic 校验后方可进入审核链路。
            </p>
            <el-empty
              v-if="!insights.length"
              description="暂无 AI 解读，点击上方按钮生成。"
            /><template v-else
              ><div class="insight-meta">
                <el-tag
                  size="small"
                  :type="
                    latestInsight?.status === 'rejected'
                      ? 'danger'
                      : latestInsight?.status === 'approved'
                        ? 'success'
                        : latestInsight?.status === 'pending_review'
                          ? 'warning'
                          : 'info'
                  "
                  >{{ latestInsight?.status }}</el-tag
                ><el-tag
                  v-if="latestInsight?.critic_verdict"
                  size="small"
                  :type="
                    latestInsight?.critic_verdict === 'approved'
                      ? 'success'
                      : 'danger'
                  "
                  >Critic: {{ latestInsight?.critic_verdict }}</el-tag
                ><span class="insight-version"
                  >insight v{{ latestInsight?.insight_version }} ·
                  {{ latestInsight?.model }}</span
                ><span
                  v-if="attributionEvidenceCount"
                  class="insight-attribution"
                  >归因补查 {{ attributionEvidenceCount }} 条证据</span
                >
                <div
                  v-if="
                    latestInsight &&
                    (latestInsight.status === 'draft' ||
                      latestInsight.status === 'pending_review')
                  "
                  class="insight-review-buttons"
                >
                  <el-button
                    size="small"
                    type="success"
                    @click="reviewInsight('approve')"
                    >审核通过</el-button
                  ><el-button
                    size="small"
                    type="danger"
                    @click="reviewInsight('reject')"
                    >驳回</el-button
                  >
                </div>
              </div>
              <p
                v-if="latestInsight?.review_reason"
                class="insight-review-reason"
              >
                审核备注：{{ latestInsight?.review_reason }} ·
                {{ latestInsight?.reviewed_by }}
              </p>
              <el-card
                v-if="attributionEvidenceRows.length"
                shadow="never"
                class="attribution-evidence-card"
              >
                <template #header>归因拆解证据（统计贡献，不代表因果）</template>
                <el-collapse>
                  <el-collapse-item
                    v-for="row in attributionEvidenceRows"
                    :key="row.evidenceId"
                    :name="row.evidenceId"
                    :title="`${row.metric} · ${row.kind} · ${row.method}`"
                  >
                    <p v-if="row.claim">{{ row.claim }}</p>
                    <p v-if="row.definition" class="attribution-limitations">指标口径（{{ row.unit }}）：{{ row.definition }}</p>
                    <div class="attribution-evidence-grid">
                      <div><strong>拆解值、分子分母或组内统计</strong><pre>{{ prettyEvidence(row.values) }}</pre></div>
                      <div><strong>比较范围</strong><pre>{{ prettyEvidence(row.comparison) }}</pre></div>
                      <div v-if="Object.keys(row.dataQuality).length"><strong>数据质量与对账状态</strong><pre>{{ prettyEvidence(row.dataQuality) }}</pre></div>
                    </div>
                    <p v-if="row.limitations.length" class="attribution-limitations">限制：{{ row.limitations.join("；") }}</p>
                    <small>证据 ID：{{ row.evidenceId }}</small>
                  </el-collapse-item>
                </el-collapse>
              </el-card>
              <el-skeleton v-if="insightGeneratingNow" :rows="3" animated />
              <p v-if="insightGeneratingNow" class="insight-generating-hint">
                正在生成 AI 解读：{{ insightStage || "准备中" }}…（SSE
                实时进度，完成后自动刷新）
              </p>
              <template v-if="!insightGeneratingNow"
                ><div
                  v-for="(item, index) in latestInsightItems"
                  :key="index"
                  class="insight-card"
                >
                  <div class="insight-title">{{ item.title }}</div>
                  <p class="insight-finding">{{ item.finding }}</p>
                  <div class="insight-tags">
                    <el-tag size="small" type="info">{{
                      item.statement_type
                    }}</el-tag
                    ><el-tag
                      v-if="item.confidence != null"
                      size="small"
                      type="warning"
                      >置信度 {{ Math.round(item.confidence * 100) }}%</el-tag
                    >
                  </div>
                  <div
                    v-if="item.recommendation"
                    class="insight-recommendation"
                  >
                    建议：{{ item.recommendation }}
                  </div>
                  <div
                    v-if="item.evidence_ids?.length"
                    class="insight-evidence"
                  >
                    <span>证据：</span
                    ><el-tag
                      v-for="eid in item.evidence_ids"
                      :key="eid"
                      size="small"
                      type="success"
                      class="evidence-tag"
                      >{{ eid }}</el-tag
                    >
                  </div>
                </div>
                <el-alert
                  v-if="latestInsight?.status === 'rejected'"
                  type="error"
                  :closable="false"
                  title="本次洞察未通过校验，未进入可审核链路"
                  ><ul>
                    <li
                      v-for="(err, i) in latestInsight?.validation_errors ?? []"
                      :key="i"
                    >
                      {{ err }}
                    </li>
                  </ul></el-alert
                ></template
              ></template
            >
          </section></el-card
        ></el-col
      >
    </el-row>
  </div>
</template>

<style scoped>
.period-hint,
.field-hint {
  margin: -8px 0 8px;
  color: var(--el-text-color-secondary);
  font-size: 12px;
  line-height: 1.5;
}
.field-hint {
  margin: 6px 0 0;
}
.report-charts {
  margin-top: 28px;
  padding-top: 4px;
  border-top: 1px solid var(--el-border-color-lighter);
}
.report-charts h2 {
  margin: 20px 0 12px;
  font-size: 18px;
}
.report-chart {
  margin: 0 0 16px;
  padding: 12px;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 6px;
  background: var(--el-fill-color-blank);
}
.report-insights {
  margin-top: 28px;
  padding-top: 4px;
  border-top: 1px solid var(--el-border-color-lighter);
}
.report-insights h2 {
  margin: 0 0 12px;
  font-size: 18px;
}
.insights-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.insights-form {
  display: flex;
  gap: 8px;
}
.insights-hint {
  margin: 8px 0 12px;
  color: var(--el-text-color-secondary);
  font-size: 12px;
  line-height: 1.6;
}
.insight-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}
.insight-version {
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
.insight-attribution {
  color: var(--el-color-primary);
  font-size: 12px;
}
.attribution-evidence-card { margin: 12px 0; }
.attribution-evidence-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.attribution-evidence-grid pre { max-height: 260px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; padding: 8px; background: var(--el-fill-color-light); border-radius: 4px; font-size: 11px; }
.attribution-limitations { color: var(--el-text-color-secondary); font-size: 12px; }
.insight-review-buttons {
  margin-left: auto;
  display: flex;
  gap: 6px;
}
.insight-review-reason {
  margin: 0 0 10px;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
.insight-generating-hint {
  margin: 10px 0 0;
  color: var(--el-text-color-secondary);
  font-size: 13px;
}
.insight-config {
  padding: 4px;
}
.word-range {
  display: flex;
  align-items: center;
  gap: 6px;
}
.insight-card {
  margin-bottom: 14px;
  padding: 12px;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 6px;
  background: var(--el-fill-color-blank);
}
.insight-title {
  font-weight: 600;
  margin-bottom: 6px;
}
.insight-finding {
  margin: 0 0 8px;
  line-height: 1.7;
  white-space: pre-wrap;
}
.insight-tags {
  display: flex;
  gap: 6px;
  margin-bottom: 6px;
}
.insight-recommendation {
  margin-bottom: 6px;
  color: var(--el-color-primary);
  font-size: 13px;
}
.insight-evidence {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  align-items: center;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.evidence-tag {
  margin-right: 2px;
}
</style>
