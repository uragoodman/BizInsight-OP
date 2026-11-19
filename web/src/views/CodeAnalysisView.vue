<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { ElMessage, ElMessageBox } from "element-plus";
import { api, streamCodeAnalysisJob } from "@/api";
import { useAppStore } from "@/stores/app";
import type {
  CodeAnalysisArtifact,
  CodeAnalysisEvent,
  CodeAnalysisFile,
  CodeAnalysisJob,
  CodeAnalysisJobSummary,
  CodeAnalysisStatus,
} from "@/types";

const store = useAppStore();
const route = useRoute();
const uploaded = ref<CodeAnalysisFile | null>(null);
const continuedFileId = ref<string | null>(null);
const continuationParentJobId = ref<string | null>(null);
const question = ref(String(route.query.question ?? ""));
const jobs = ref<CodeAnalysisJobSummary[]>([]);
const selected = ref<CodeAnalysisJob | null>(null);
const conversationJobs = ref<CodeAnalysisJob[]>([]);
const loading = ref(false);
const listLoading = ref(false);
const detailLoading = ref(false);
const conversationLoading = ref(false);
const revising = ref(false);
const revisedQuestion = ref("");
const chartUrls = ref<Record<string, string>>({});
const chartErrors = ref<Record<string, string>>({});
const jobDateFilter = ref<"all" | "7d" | "30d">("30d");
const jobPage = ref(1);
const totalJobs = ref(0);
const jobPageSize = 10;

const terminalStatuses = new Set<CodeAnalysisStatus>(["completed", "failed", "cancelled", "expired", "superseded"]);
const progressByStatus: Record<CodeAnalysisStatus, number> = {
  awaiting_confirmation: 0,
  queued: 15,
  generating: 35,
  validating: 60,
  running: 80,
  completed: 100,
  failed: 100,
  cancelled: 100,
  expired: 100,
  superseded: 100,
};
const statusLabels: Record<CodeAnalysisStatus, string> = {
  awaiting_confirmation: "待确认",
  queued: "排队中",
  generating: "生成代码",
  validating: "安全校验",
  running: "隔离执行",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
  expired: "已过期",
  superseded: "已被修订",
};

const chartArtifacts = computed(() => selected.value?.artifacts?.filter((artifact) =>
  artifact.kind === "chart"
  && (artifact.metadata?.content_type === "image/png" || artifactName(artifact).toLowerCase().endsWith(".png"))
) ?? []);
const limitations = computed(() => Array.isArray(selected.value?.result_data?.limitations) ? selected.value.result_data.limitations : []);
const evidence = computed(() => Array.isArray(selected.value?.result_data?.evidence) ? selected.value.result_data.evidence : []);
const recommendations = computed(() => Array.isArray(selected.value?.result_data?.recommendations) ? selected.value.result_data.recommendations : []);
const conclusion = computed(() => {
  const result = selected.value?.result_data;
  if (typeof result?.conclusion === "string" && result.conclusion.trim()) return result.conclusion.trim();
  return "该历史任务未返回独立的文字结论；结构化摘要和证据不能替代分析文本。";
});
const resultContractMissing = computed(() =>
  selected.value?.status === "completed"
  && !!selected.value.result_data
  && !(typeof selected.value.result_data.conclusion === "string" && selected.value.result_data.conclusion.trim())
);
const resourceLimits = computed(() =>
  selected.value?.resource_limits ?? selected.value?.plan_data.resource_limits
);
const activeFileId = computed(() => uploaded.value?.file_id ?? continuedFileId.value);
const continuingWithCurrentFile = computed(() =>
  !!continuedFileId.value && !uploaded.value
);
const canContinueCurrentFile = computed(() =>
  selected.value?.status === "completed" && !!selected.value.file_id
);

let streamController: AbortController | null = null;
let watchedJobId: string | null = null;
let chartLoadVersion = 0;
const now = ref(Date.now());
let elapsedTimer: number | null = null;

function statusLabel(status: CodeAnalysisStatus): string {
  return statusLabels[status] ?? status;
}

function statusTagType(status: CodeAnalysisStatus): "success" | "warning" | "info" | "danger" {
  if (status === "completed") return "success";
  if (status === "failed" || status === "expired") return "danger";
  if (status === "awaiting_confirmation") return "warning";
  return "info";
}

const progressPercent = computed(() => {
  if (!selected.value) return 0;
  return progressByStatus[selected.value.status] ?? 0;
});
const progressIndeterminate = computed(() =>
  selected.value?.status === "generating" || selected.value?.status === "running"
);
const progressStatus = computed<"success" | "exception" | "warning" | undefined>(() => {
  if (selected.value?.status === "completed") return "success";
  if (["failed", "expired"].includes(selected.value?.status ?? "")) return "exception";
  if (selected.value?.status === "cancelled" || selected.value?.status === "superseded") return "warning";
  return undefined;
});
const canDeleteSelected = computed(() =>
  !!selected.value
  && terminalStatuses.has(selected.value.status)
  && !selected.value.chat_session_id
);
const elapsedMs = computed(() => {
  if (!selected.value) return 0;
  const startedAt = Date.parse(selected.value.started_at ?? selected.value.created_at);
  if (!Number.isFinite(startedAt)) return 0;
  const endAt = terminalStatuses.has(selected.value.status)
    ? Date.parse(selected.value.completed_at ?? selected.value.updated_at)
    : now.value;
  return Math.max(0, (Number.isFinite(endAt) ? endAt : now.value) - startedAt);
});

function formatElapsed(milliseconds: number): string {
  const totalSeconds = Math.floor(milliseconds / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) return `${hours}时 ${String(minutes).padStart(2, "0")}分`;
  if (minutes > 0) return `${minutes}分 ${String(seconds).padStart(2, "0")}秒`;
  return `${seconds}秒`;
}

function upsertSummary(job: CodeAnalysisJob | CodeAnalysisJobSummary): void {
  const summary: CodeAnalysisJobSummary = {
    job_id: job.job_id,
    question: job.question,
    status: job.status,
    file_id: job.file_id,
    error_code: job.error_code,
    created_at: job.created_at,
    updated_at: job.updated_at,
    completed_at: job.completed_at,
  };
  const index = jobs.value.findIndex((item) => item.job_id === job.job_id);
  if (index < 0) jobs.value.unshift(summary);
  else jobs.value[index] = summary;
}

function applyJob(job: CodeAnalysisJob): void {
  if (!selected.value || selected.value.job_id === job.job_id) selected.value = job;
  upsertSummary(job);
  const conversationIndex = conversationJobs.value.findIndex((item) => item.job_id === job.job_id);
  if (conversationIndex >= 0) conversationJobs.value[conversationIndex] = job;
}

function applyStreamEvent(event: CodeAnalysisEvent): void {
  if (event.event === "status") {
    const item = jobs.value.find((job) => job.job_id === event.job_id);
    if (item) {
      item.status = event.status;
      item.error_code = event.error_code;
    }
    if (selected.value?.job_id === event.job_id) {
      selected.value = { ...selected.value, status: event.status, error_code: event.error_code };
    }
  } else if ("job" in event && event.job) {
    applyJob(event.job);
  }
}

function stopWatching(): void {
  streamController?.abort();
  streamController = null;
  watchedJobId = null;
}

function startWatching(job: CodeAnalysisJob): void {
  if (terminalStatuses.has(job.status) || job.status === "awaiting_confirmation") {
    if (watchedJobId === job.job_id) stopWatching();
    return;
  }
  if (streamController && watchedJobId === job.job_id) return;
  stopWatching();
  const controller = new AbortController();
  streamController = controller;
  watchedJobId = job.job_id;
  void streamCodeAnalysisJob(store.auth, job.job_id, applyStreamEvent, controller.signal)
    .then(async (terminalJob) => {
      if (terminalJob) applyJob(terminalJob);
      else if (!controller.signal.aborted && selected.value?.job_id === job.job_id) {
        applyJob(await api.codeAnalysisJob(store.auth, job.job_id));
      }
    })
    .catch(async (error: unknown) => {
      if (controller.signal.aborted) return;
      try {
        if (selected.value?.job_id === job.job_id) applyJob(await api.codeAnalysisJob(store.auth, job.job_id));
      } catch {
        // The stream error below remains the most actionable message.
      }
      ElMessage.warning(error instanceof Error ? error.message : "实时任务状态连接已中断，请刷新重试");
    })
    .finally(() => {
      if (streamController === controller) {
        streamController = null;
        watchedJobId = null;
      }
    });
}

async function onFile(file: { raw?: File }): Promise<void> {
  if (!file.raw) return;
  loading.value = true;
  uploaded.value = null;
  continuedFileId.value = null;
  continuationParentJobId.value = null;
  try {
    uploaded.value = await api.uploadCodeAnalysisFile(store.auth, file.raw);
    ElMessage.success("文件已校验，等待创建分析计划");
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "上传失败");
  } finally {
    loading.value = false;
  }
}

function continueWithCurrentFile(): void {
  if (!canContinueCurrentFile.value || !selected.value?.file_id) return;
  uploaded.value = null;
  continuedFileId.value = selected.value.file_id;
  continuationParentJobId.value = selected.value.job_id;
  question.value = "";
  ElMessage.info("已选择当前文件，可以继续提出新的分析问题");
}

async function create(): Promise<void> {
  if (!activeFileId.value || !question.value.trim()) return;
  loading.value = true;
  try {
    const session = typeof route.query.session === "string"
      ? route.query.session
      : selected.value?.chat_session_id ?? undefined;
    const sourceTurn = typeof route.query.turn === "string" ? route.query.turn : undefined;
    const job = await api.createCodeAnalysisJob(store.auth, {
      file_id: activeFileId.value,
      question: question.value,
      chat_session_id: session,
      source_turn_id: sourceTurn,
      parent_job_id: continuationParentJobId.value ?? undefined,
    });
    selected.value = job;
    continuedFileId.value = job.file_id;
    continuationParentJobId.value = null;
    revising.value = false;
    upsertSummary(job);
    ElMessage.success("已创建待确认的隔离分析计划");
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "创建计划失败");
  } finally {
    loading.value = false;
  }
}

async function confirm(job: CodeAnalysisJob): Promise<void> {
  detailLoading.value = true;
  try {
    const queued = await api.confirmCodeAnalysisJob(store.auth, job.job_id, job.plan_hash);
    applyJob(queued);
    startWatching(queued);
    ElMessage.success("任务已入队，状态和结果将实时更新");
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "确认失败");
    try {
      applyJob(await api.codeAnalysisJob(store.auth, job.job_id));
    } catch {
      // Preserve the confirmation error when the refresh also fails.
    }
  } finally {
    detailLoading.value = false;
  }
}

function beginRevision(job: CodeAnalysisJob): void {
  revisedQuestion.value = job.question;
  revising.value = true;
}

async function revise(job: CodeAnalysisJob): Promise<void> {
  if (!revisedQuestion.value.trim() || revisedQuestion.value.trim() === job.question) {
    ElMessage.warning("请先修改分析问题");
    return;
  }
  detailLoading.value = true;
  try {
    const revised = await api.reviseCodeAnalysisJob(store.auth, job.job_id, revisedQuestion.value.trim());
    const previous = jobs.value.find((item) => item.job_id === job.job_id);
    if (previous) previous.status = "superseded";
    selected.value = revised;
    upsertSummary(revised);
    revising.value = false;
    ElMessage.success(`已生成第 ${revised.revision} 版计划，请重新确认`);
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "调整计划失败");
  } finally {
    detailLoading.value = false;
  }
}

async function cancel(job: CodeAnalysisJob): Promise<void> {
  detailLoading.value = true;
  try {
    await api.cancelCodeAnalysisJob(store.auth, job.job_id);
    const refreshed = await api.codeAnalysisJob(store.auth, job.job_id);
    applyJob(refreshed);
    startWatching(refreshed);
    ElMessage.success(refreshed.status === "cancelled" ? "任务已取消" : "已提交取消请求");
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "取消失败");
  } finally {
    detailLoading.value = false;
  }
}

async function removeJob(job: CodeAnalysisJob): Promise<void> {
  if (!canDeleteSelected.value) return;
  try {
    await ElMessageBox.confirm(
      "删除后将移除该任务及其分析产物，且无法恢复。建议只清理重复或失败记录。",
      "删除分析任务",
      { type: "warning", confirmButtonText: "删除", cancelButtonText: "保留" },
    );
    await api.deleteCodeAnalysisJob(store.auth, job.job_id);
    stopWatching();
    selected.value = null;
    conversationJobs.value = conversationJobs.value.filter((item) => item.job_id !== job.job_id);
    ElMessage.success("分析任务已删除");
    await load(jobPage.value);
  } catch (error) {
    if (error === "cancel" || error === "close") return;
    ElMessage.error(error instanceof Error ? error.message : "删除分析任务失败");
  }
}

async function choose(jobId: string): Promise<void> {
  detailLoading.value = true;
  revising.value = false;
  stopWatching();
  try {
    const job = await api.codeAnalysisJob(store.auth, jobId);
    selected.value = job;
    upsertSummary(job);
    startWatching(job);
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "加载任务详情失败");
  } finally {
    detailLoading.value = false;
  }
}

function createdAfterForFilter(): string | undefined {
  if (jobDateFilter.value === "all") return undefined;
  const days = jobDateFilter.value === "7d" ? 7 : 30;
  return new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
}

async function load(page = jobPage.value): Promise<void> {
  listLoading.value = true;
  const selectedJobId = selected.value?.job_id;
  try {
    const session = typeof route.query.session === "string" ? route.query.session : undefined;
    const result = await api.codeAnalysisJobs(
      store.auth,
      (page - 1) * jobPageSize,
      jobPageSize,
      session,
      createdAfterForFilter(),
    );
    jobs.value = result.items;
    totalJobs.value = result.total;
    jobPage.value = page;
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "加载任务失败");
    return;
  } finally {
    listLoading.value = false;
  }
  if (selectedJobId && jobs.value.some((item) => item.job_id === selectedJobId)) await choose(selectedJobId);
  else if (selectedJobId) {
    selected.value = null;
    stopWatching();
  }
}

async function changeJobDateFilter(): Promise<void> {
  jobPage.value = 1;
  await load(1);
}

async function changeJobPage(page: number): Promise<void> {
  await load(page);
}

async function loadConversation(fileId: string): Promise<void> {
  conversationLoading.value = true;
  try {
    const result = await api.codeAnalysisJobs(store.auth, 0, 50, undefined, undefined, fileId);
    const details = await Promise.all(result.items.map((item) => api.codeAnalysisJob(store.auth, item.job_id)));
    conversationJobs.value = details.sort((left, right) =>
      Date.parse(left.created_at) - Date.parse(right.created_at),
    );
  } catch (error) {
    conversationJobs.value = [];
    ElMessage.warning(error instanceof Error ? error.message : "无法加载当前文件的分析记录");
  } finally {
    conversationLoading.value = false;
  }
}

function formatConversationDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function artifactName(artifact: CodeAnalysisArtifact): string {
  if (artifact.metadata?.name) return artifact.metadata.name;
  const defaults: Record<string, string> = {
    candidate_code: "analysis.py",
    validation: "validation.json",
    runner_log: "runner.log",
    result: "result.json",
    chart: "chart",
  };
  return defaults[artifact.kind] ?? artifact.kind;
}

async function downloadArtifact(artifact: CodeAnalysisArtifact): Promise<void> {
  if (!selected.value) return;
  try {
    const blob = await api.codeAnalysisArtifact(store.auth, selected.value.job_id, artifact.artifact_id);
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = artifactName(artifact);
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "下载产物失败");
  }
}

function clearChartUrls(): void {
  Object.values(chartUrls.value).forEach((url) => URL.revokeObjectURL(url));
  chartUrls.value = {};
  chartErrors.value = {};
}

async function loadChartPreviews(job: CodeAnalysisJob): Promise<void> {
  const version = ++chartLoadVersion;
  clearChartUrls();
  const charts = job.artifacts?.filter((artifact) =>
    artifact.kind === "chart"
    && (artifact.metadata?.content_type === "image/png" || artifactName(artifact).toLowerCase().endsWith(".png"))
  ) ?? [];
  const loaded = await Promise.all(charts.map(async (artifact) => {
    try {
      const blob = await api.codeAnalysisArtifact(store.auth, job.job_id, artifact.artifact_id);
      return { artifactId: artifact.artifact_id, url: URL.createObjectURL(blob), error: "" };
    } catch (error) {
      return { artifactId: artifact.artifact_id, url: "", error: error instanceof Error ? error.message : "图表加载失败" };
    }
  }));
  if (version !== chartLoadVersion || selected.value?.job_id !== job.job_id) {
    loaded.forEach((item) => { if (item.url) URL.revokeObjectURL(item.url); });
    return;
  }
  chartUrls.value = Object.fromEntries(loaded.filter((item) => item.url).map((item) => [item.artifactId, item.url]));
  chartErrors.value = Object.fromEntries(loaded.filter((item) => item.error).map((item) => [item.artifactId, item.error]));
}

watch(
  () => selected.value
    ? `${selected.value.job_id}:${selected.value.status}:${selected.value.updated_at}:${selected.value.artifacts?.map((item) => item.artifact_id).join(",") ?? ""}`
    : "",
  () => {
    if (selected.value) void loadChartPreviews(selected.value);
    else clearChartUrls();
  },
);

watch(
  () => selected.value?.file_id,
  (fileId) => {
    if (fileId) void loadConversation(fileId);
    else conversationJobs.value = [];
  },
);

function displayValue(value: unknown): string {
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

onMounted(() => {
  void load(1);
  elapsedTimer = window.setInterval(() => { now.value = Date.now(); }, 1000);
});
onBeforeUnmount(() => {
  stopWatching();
  if (elapsedTimer !== null) window.clearInterval(elapsedTimer);
  chartLoadVersion += 1;
  clearChartUrls();
});
</script>

<template>
  <div class="page code-page">
    <div class="page-title">
      <div>
        <h1>数据分析任务</h1>
        <p>上传数据后，系统在隔离环境生成并执行候选代码。不会访问业务数据库，原始文件默认 24 小时后清理。</p>
      </div>
      <el-button @click="load(jobPage)">刷新</el-button>
    </div>

    <el-card shadow="never" class="create">
      <template #header>新建受控分析</template>
      <div class="file-actions">
        <el-upload :auto-upload="false" :show-file-list="true" accept=".csv,.xlsx,.parquet" :on-change="onFile">
          <el-button :loading="loading">选择 CSV / Excel / Parquet</el-button>
        </el-upload>
        <el-button
          v-if="canContinueCurrentFile"
          :type="continuingWithCurrentFile ? 'primary' : 'default'"
          @click="continueWithCurrentFile"
        >
          基于当前文件继续聊
        </el-button>
      </div>
      <div v-if="uploaded" class="contract">
        <strong>{{ uploaded.filename }}</strong>
        · {{ uploaded.file_type.toUpperCase() }}
        · {{ uploaded.size_bytes.toLocaleString() }} bytes
        <br>
        <span>
          数据契约：{{ uploaded.data_contract.rows ?? "将在隔离环境中读取" }} 行，
          {{ uploaded.data_contract.columns?.length ?? 0 }} 列；原始文件保留至
          {{ new Date(uploaded.expires_at).toLocaleString() }}
        </span>
      </div>
      <div v-else-if="continuingWithCurrentFile" class="contract reused-file">
        <strong>当前分析文件</strong>
        · 已复用，不需要重新上传
        <br>
        <span>文件 ID：<code>{{ continuedFileId }}</code>，文件仍在保留期内时可以继续创建分析任务。</span>
      </div>
      <el-input
        v-model="question"
        type="textarea"
        :rows="3"
        maxlength="4000"
        show-word-limit
        :placeholder="continuingWithCurrentFile ? '继续追问当前文件，例如：解释刚才的结果，比较客单价最高的月份' : '例如：分析销售额趋势、异常月份以及可能的影响因素'"
      />
      <div class="actions">
        <el-button type="primary" :disabled="!activeFileId || !question.trim()" :loading="loading" @click="create">生成分析计划</el-button>
      </div>
    </el-card>

    <el-row :gutter="16" class="tasks">
      <el-col :lg="8">
        <el-card v-loading="listLoading" shadow="never">
          <template #header>
            <div class="task-center-head">
              <span>任务中心</span>
              <el-select v-model="jobDateFilter" size="small" class="job-date-filter" @change="changeJobDateFilter">
                <el-option label="最近 30 天" value="30d" />
                <el-option label="最近 7 天" value="7d" />
                <el-option label="全部时间" value="all" />
              </el-select>
            </div>
          </template>
          <el-empty v-if="!jobs.length" description="暂无分析任务" />
          <el-menu v-else :default-active="selected?.job_id" @select="choose">
            <el-menu-item v-for="job in jobs" :key="job.job_id" :index="job.job_id">
              <span class="job-question">{{ job.question.slice(0, 28) }}</span>
              <el-tag size="small" :type="statusTagType(job.status)">{{ statusLabel(job.status) }}</el-tag>
            </el-menu-item>
          </el-menu>
          <el-pagination
            v-if="totalJobs > jobPageSize"
            class="job-pagination"
            background
            small
            layout="prev, pager, next"
            :current-page="jobPage"
            :page-size="jobPageSize"
            :total="totalJobs"
            @current-change="changeJobPage"
          />
        </el-card>
      </el-col>

      <el-col :lg="16">
        <el-empty v-if="!selected" description="选择一个任务查看计划、结果和审计信息" />
        <el-card v-else v-loading="detailLoading" shadow="never">
          <template #header>
            <div class="detail-head">
              <span>隔离分析计划 · 第 {{ selected.revision }} 版</span>
              <div class="detail-head-actions">
                <el-button v-if="canContinueCurrentFile" size="small" type="primary" plain @click="continueWithCurrentFile">基于当前文件继续聊</el-button>
                <el-button v-if="canDeleteSelected" size="small" type="danger" plain @click="removeJob(selected)">删除任务</el-button>
                <el-tag :type="statusTagType(selected.status)">{{ statusLabel(selected.status) }}</el-tag>
              </div>
            </div>
          </template>

          <p><strong>问题：</strong>{{ selected.question }}</p>
          <p><strong>确认保护：</strong>候选代码只会在无网络、无业务凭据的一次性容器中运行。</p>
          <div class="plan-summary">
            <div><span>数据文件</span><code>{{ selected.plan_data.file_id }}</code></div>
            <div v-if="selected.parent_job_id"><span>上一版本</span><code>{{ selected.parent_job_id }}</code></div>
            <div><span>保留时长</span>{{ selected.plan_data.retention_hours }} 小时</div>
            <div>
              <span>预期输出</span>
              <el-tag v-for="output in selected.plan_data.expected_output" :key="output" size="small">{{ output }}</el-tag>
            </div>
            <div v-if="resourceLimits">
              <span>执行上限</span>
              <span>
                {{ resourceLimits.timeout_seconds ?? "—" }} 秒 ·
                {{ resourceLimits.memory_mb ?? "—" }} MB 内存 ·
                {{ resourceLimits.cpu_cores ?? "—" }} CPU ·
                {{ resourceLimits.pids ?? "—" }} 进程
              </span>
            </div>
          </div>

          <section v-if="conversationJobs.length" v-loading="conversationLoading" class="conversation-panel">
            <div class="conversation-panel-head">
              <div>
                <h3>当前文件分析对话</h3>
                <span>同一个文件的每次追问都会生成独立的隔离任务，按时间顺序保留在这里。</span>
              </div>
              <el-tag size="small">{{ conversationJobs.length }} 次分析</el-tag>
            </div>
            <div class="conversation-stream">
              <article
                v-for="conversationJob in conversationJobs"
                :key="conversationJob.job_id"
                class="conversation-item"
                :class="{ active: conversationJob.job_id === selected.job_id }"
              >
                <div class="conversation-question bubble">
                  <div class="bubble-meta">
                    <strong>问题</strong>
                    <span>{{ formatConversationDate(conversationJob.created_at) }}</span>
                  </div>
                  <p>{{ conversationJob.question }}</p>
                </div>
                <div class="conversation-answer bubble">
                  <div class="bubble-meta">
                    <strong>隔离分析计划 · 第 {{ conversationJob.revision }} 版</strong>
                    <el-tag size="small" :type="statusTagType(conversationJob.status)">{{ statusLabel(conversationJob.status) }}</el-tag>
                  </div>
                  <p v-if="conversationJob.result_data?.conclusion">{{ conversationJob.result_data.conclusion }}</p>
                  <p v-else-if="conversationJob.error_code" class="conversation-error">任务未完成：{{ conversationJob.error_code }}</p>
                  <p v-else class="conversation-muted">计划已生成，等待确认或正在处理中。</p>
                  <el-button
                    v-if="conversationJob.job_id !== selected.job_id"
                    size="small"
                    text
                    type="primary"
                    @click="choose(conversationJob.job_id)"
                  >查看详情</el-button>
                </div>
              </article>
            </div>
          </section>

          <section v-if="selected.status !== 'awaiting_confirmation'" class="progress-panel">
            <div class="progress-meta">
              <span>执行进度：{{ statusLabel(selected.status) }}</span>
              <span>{{ formatElapsed(elapsedMs) }}</span>
            </div>
            <el-progress
              :percentage="progressPercent"
              :status="progressStatus"
              :indeterminate="progressIndeterminate"
              :duration="2"
              :stroke-width="12"
            />
            <p v-if="progressIndeterminate" class="progress-hint">当前阶段正在处理，耗时取决于模型生成和隔离环境执行。</p>
          </section>

          <el-alert
            v-if="selected.status === 'awaiting_confirmation'"
            title="此任务尚未执行。请核对问题、数据文件和预期输出；确认后才会生成候选代码并进入隔离队列。"
            type="warning"
            :closable="false"
          />
          <div v-if="selected.status === 'awaiting_confirmation'" class="actions">
            <el-button type="primary" :loading="detailLoading" @click="confirm(selected)">确认执行</el-button>
            <el-button @click="beginRevision(selected)">调整计划</el-button>
            <el-button @click="cancel(selected)">取消</el-button>
          </div>
          <div v-if="revising && selected.status === 'awaiting_confirmation'" class="revision-editor">
            <el-input v-model="revisedQuestion" type="textarea" :rows="3" maxlength="4000" show-word-limit />
            <div class="actions">
              <el-button type="primary" :loading="detailLoading" @click="revise(selected)">生成新版本</el-button>
              <el-button @click="revising = false">返回</el-button>
            </div>
          </div>
          <div v-else-if="!terminalStatuses.has(selected.status)" class="actions">
            <el-button :loading="detailLoading" @click="cancel(selected)">请求取消</el-button>
          </div>

          <el-alert
            v-if="selected.error_code"
            :title="`任务未完成：${selected.error_code}`"
            type="error"
            :closable="false"
            class="result-section"
          />

          <template v-if="selected.result_data">
            <el-alert
              v-if="resultContractMissing"
              title="该已完成任务缺少强制文字结论，属于旧版或异常结果；结构化卡片不能替代分析结论。"
              type="error"
              :closable="false"
              class="result-section"
            />
            <section class="result-section">
              <h3>最终结论</h3>
              <p class="conclusion-text">{{ conclusion }}</p>
            </section>

            <section v-if="selected.result_data.method" class="result-section">
              <h3>分析方法</h3>
              <p v-if="typeof selected.result_data.method === 'string'" class="narrative-text">{{ selected.result_data.method }}</p>
              <pre v-else>{{ JSON.stringify(selected.result_data.method, null, 2) }}</pre>
            </section>

            <details v-if="evidence.length" class="result-section result-details evidence-details">
              <summary>关键证据（{{ evidence.length }} 条，点击展开）</summary>
              <div class="evidence-list">
                <article v-for="(item, index) in evidence" :key="index">{{ displayValue(item) }}</article>
              </div>
            </details>

            <section v-if="recommendations.length" class="result-section">
              <h3>建议</h3>
              <ul><li v-for="item in recommendations" :key="item">{{ item }}</li></ul>
            </section>

            <section v-if="chartArtifacts.length" class="result-section">
              <h3>分析图表</h3>
              <div class="chart-grid">
                <article v-for="artifact in chartArtifacts" :key="artifact.artifact_id" class="chart-card">
                  <strong>{{ artifactName(artifact) }}</strong>
                  <img v-if="chartUrls[artifact.artifact_id]" :src="chartUrls[artifact.artifact_id]" :alt="artifactName(artifact)">
                  <el-alert v-else-if="chartErrors[artifact.artifact_id]" :title="chartErrors[artifact.artifact_id]" type="error" :closable="false" />
                  <el-skeleton v-else :rows="4" animated />
                  <el-button size="small" @click="downloadArtifact(artifact)">下载图表</el-button>
                </article>
              </div>
            </section>

            <section v-if="limitations.length" class="result-section">
              <h3>限制说明</h3>
              <ul><li v-for="item in limitations" :key="item">{{ item }}</li></ul>
            </section>

            <section v-if="selected.result_data.reproducibility" class="result-section">
              <h3>复现说明</h3>
              <p v-if="typeof selected.result_data.reproducibility === 'string'" class="narrative-text">{{ selected.result_data.reproducibility }}</p>
              <pre v-else>{{ JSON.stringify(selected.result_data.reproducibility, null, 2) }}</pre>
            </section>

            <details v-if="selected.result_data.summary" class="result-section auxiliary-details">
              <summary>查看结构化摘要</summary>
              <pre>{{ JSON.stringify(selected.result_data.summary, null, 2) }}</pre>
            </details>

            <details class="result-section auxiliary-details">
              <summary>查看完整 JSON 明细</summary>
              <pre>{{ JSON.stringify(selected.result_data, null, 2) }}</pre>
            </details>
          </template>

          <section v-if="selected.artifacts?.length" class="result-section">
            <h3>可下载产物</h3>
            <div class="artifact-list">
              <div v-for="artifact in selected.artifacts" :key="artifact.artifact_id">
                <span><strong>{{ artifactName(artifact) }}</strong> · {{ artifact.kind }} · {{ artifact.size_bytes.toLocaleString() }} bytes</span>
                <el-button size="small" @click="downloadArtifact(artifact)">下载</el-button>
              </div>
            </div>
          </section>

          <section v-if="selected.latest_audit" class="result-section">
            <h3>可复现审计</h3>
            <dl class="audit-grid">
              <div><dt>生成服务</dt><dd>{{ selected.latest_audit.provider }}</dd></div>
              <div><dt>实际模型</dt><dd>{{ selected.latest_audit.generation_model ?? "未记录" }}</dd></div>
              <div><dt>服务版本</dt><dd>{{ selected.latest_audit.provider_version ?? "未记录" }}</dd></div>
              <div><dt>候选 / 修复</dt><dd>{{ selected.latest_audit.candidate_attempts ?? 0 }} / {{ selected.latest_audit.repair_attempts ?? 0 }}</dd></div>
              <div><dt>执行耗时</dt><dd>{{ selected.latest_audit.duration_ms }} ms</dd></div>
              <div><dt>Trace</dt><dd>{{ selected.latest_audit.trace_id ?? "—" }}</dd></div>
              <div v-if="selected.latest_audit.provider_request_id"><dt>Provider 请求 ID</dt><dd>{{ selected.latest_audit.provider_request_id }}</dd></div>
              <div v-if="selected.latest_audit.error_phase"><dt>失败阶段</dt><dd>{{ selected.latest_audit.error_phase }}</dd></div>
              <div v-if="selected.latest_audit.error_http_status"><dt>HTTP 状态</dt><dd>{{ selected.latest_audit.error_http_status }}</dd></div>
            </dl>
            <el-alert
              v-if="selected.latest_audit.error_detail"
              :title="selected.latest_audit.error_detail"
              type="error"
              :closable="false"
              class="audit-error"
            />
            <details v-if="selected.latest_audit.error_diagnostics && Object.keys(selected.latest_audit.error_diagnostics).length" class="audit-diagnostics">
              <summary>安全诊断字段</summary>
              <pre>{{ JSON.stringify(selected.latest_audit.error_diagnostics, null, 2) }}</pre>
            </details>
          </section>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.create { max-width: 980px; }
.file-actions { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
.contract { margin: 12px 0; color: var(--el-text-color-regular); }
.contract span { font-size: 13px; }
.reused-file { padding: 10px 12px; border-radius: 8px; background: var(--el-color-primary-light-9); }
.actions { margin-top: 12px; display: flex; gap: 8px; }
.tasks { margin-top: 16px; }
.task-center-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.job-date-filter { width: 118px; }
.job-pagination { justify-content: center; margin-top: 14px; }
.detail-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.detail-head-actions { display: flex; align-items: center; gap: 8px; }
.job-question { flex: 1; overflow: hidden; text-overflow: ellipsis; }
.plan-summary { display: grid; gap: 8px; margin: 14px 0; padding: 12px; border-radius: 8px; background: var(--el-fill-color-light); }
.plan-summary > div { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }
.plan-summary span:first-child { min-width: 72px; color: var(--el-text-color-secondary); }
.conversation-panel { margin: 18px 0; padding: 14px; border: 1px solid var(--el-border-color-light); border-radius: 10px; background: var(--el-fill-color-blank); }
.conversation-panel-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 12px; }
.conversation-panel-head h3 { margin: 0 0 4px; }
.conversation-panel-head span { color: var(--el-text-color-secondary); font-size: 12px; }
.conversation-stream { max-height: 620px; overflow-y: auto; padding: 4px 10px 4px 18px; border-left: 2px solid var(--el-color-primary-light-7); }
.conversation-item { position: relative; display: grid; gap: 8px; margin: 0 0 18px; }
.conversation-item::before { content: ""; position: absolute; left: -25px; top: 12px; width: 10px; height: 10px; border: 2px solid var(--el-color-primary); border-radius: 50%; background: var(--el-bg-color); }
.conversation-item.active::before { background: var(--el-color-primary); box-shadow: 0 0 0 4px var(--el-color-primary-light-8); }
.bubble { padding: 12px 14px; border-radius: 10px; }
.bubble p { margin: 8px 0 0; white-space: pre-wrap; line-height: 1.65; }
.conversation-question { background: var(--el-color-primary-light-9); }
.conversation-answer { background: var(--el-fill-color-light); }
.bubble-meta { display: flex; align-items: center; justify-content: space-between; gap: 8px; color: var(--el-text-color-secondary); font-size: 12px; }
.bubble-meta strong { color: var(--el-text-color-primary); font-size: 13px; }
.conversation-muted { color: var(--el-text-color-secondary); }
.conversation-error { color: var(--el-color-danger); }
.revision-editor { margin-top: 14px; padding: 12px; border: 1px solid var(--el-border-color); border-radius: 8px; }
.progress-panel { margin: 16px 0; padding: 12px; border-radius: 8px; background: var(--el-fill-color-light); }
.progress-meta { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 8px; color: var(--el-text-color-regular); font-size: 13px; }
.progress-hint { margin: 8px 0 0; color: var(--el-text-color-secondary); font-size: 12px; }
.result-section { margin-top: 22px; }
.result-section h3 { margin-bottom: 10px; }
.conclusion-text { margin: 0; padding: 18px; border-left: 4px solid var(--el-color-primary); border-radius: 6px; background: var(--el-color-primary-light-9); color: var(--el-text-color-primary); font-size: 17px; font-weight: 600; line-height: 1.8; white-space: pre-wrap; }
.narrative-text { line-height: 1.75; white-space: pre-wrap; }
.evidence-list { display: grid; gap: 8px; }
.evidence-list article { padding: 10px 12px; border-radius: 6px; background: var(--el-fill-color-light); white-space: pre-wrap; overflow-wrap: anywhere; }
.result-details { padding: 12px 14px; border: 1px solid var(--el-border-color-light); border-radius: 8px; background: var(--el-fill-color-blank); }
.result-details summary { cursor: pointer; color: var(--el-color-primary); font-weight: 600; }
.result-details[open] summary { margin-bottom: 12px; }
.auxiliary-details summary { cursor: pointer; color: var(--el-color-primary); }
pre { max-height: 360px; overflow: auto; background: #f6f8fb; padding: 12px; border-radius: 6px; white-space: pre-wrap; }
.chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.chart-card { display: grid; gap: 10px; padding: 12px; border: 1px solid var(--el-border-color-light); border-radius: 8px; }
.chart-card img { width: 100%; min-height: 180px; max-height: 420px; object-fit: contain; background: white; }
.artifact-list { display: grid; gap: 8px; }
.artifact-list > div { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 8px 0; border-bottom: 1px solid var(--el-border-color-lighter); }
.audit-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 0; }
.audit-grid > div { padding: 10px; background: var(--el-fill-color-light); border-radius: 6px; }
.audit-grid dt { color: var(--el-text-color-secondary); font-size: 12px; }
.audit-grid dd { margin: 4px 0 0; overflow-wrap: anywhere; }
.audit-diagnostics { margin-top: 10px; padding: 10px 12px; border: 1px solid var(--el-border-color-light); border-radius: 6px; background: var(--el-fill-color-light); }
.audit-diagnostics summary { cursor: pointer; color: var(--el-color-primary); font-size: 13px; }
</style>
