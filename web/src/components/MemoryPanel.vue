<script setup lang="ts">
import { ref, watch } from "vue";
import { ElMessage } from "element-plus";
import { api } from "@/api";
import { useAppStore } from "@/stores/app";
import type { MemoryConsolidationJob, MemoryItem } from "@/types";

defineProps<{ status: string }>();
const store = useAppStore();
const items = ref<MemoryItem[]>([]);
const jobs = ref<MemoryConsolidationJob[]>([]);

async function refresh() {
  try {
    const [active, pending, queued] = await Promise.all([api.memories(store.auth), api.memories(store.auth, "pending_review"), api.memoryConsolidationJobs(store.auth)]);
    items.value = [...pending.items, ...active.items];
    jobs.value = queued.items;
  } catch { items.value = []; }
}
async function clearSession() {
  const id = store.activeSession?.id;
  if (!id) return;
  try { await api.clearSessionMemory(store.auth, id); ElMessage.success("已清除本会话记忆"); }
  catch { ElMessage.error("清除会话记忆失败"); }
}
async function revoke(id: string) {
  try { await api.revokeMemory(store.auth, id); await refresh(); ElMessage.success("已撤销此条审核结论"); }
  catch { ElMessage.error("撤销记忆失败"); }
}
async function requestConsolidation(memory: MemoryItem) {
  if (!memory.scope_fingerprint) return;
  try {
    const job = await api.createMemoryConsolidationJob(store.auth, { scope_fingerprint: memory.scope_fingerprint, actor: store.auth.actor });
    ElMessage.success(`已创建记忆合并提案任务 ${job.job_id}`);
  } catch { ElMessage.error("合并提案任务创建失败"); }
}
async function review(memory: MemoryItem, action: "approve_independent" | "reject" | "supersede") {
  try {
    const payload: { action: "approve_independent" | "reject" | "supersede"; actor: string; old_memory_id?: string } = { action, actor: store.auth.actor };
    if (action === "supersede") {
      const candidates = (await api.memorySupersedeCandidates(store.auth, memory.memory_id)).items;
      const selected = candidates.length === 1 ? candidates[0]?.memory_id : window.prompt("输入要取代的 active Memory ID", candidates[0]?.memory_id || "");
      if (!selected) return;
      payload.old_memory_id = selected;
    }
    await api.reviewMemory(store.auth, memory.memory_id, payload);
    await refresh();
    ElMessage.success(action === "reject" ? "已驳回候选记忆" : "记忆审核完成");
  } catch { ElMessage.error("记忆审核失败"); }
}
watch(() => store.activeSessionId, refresh, { immediate: true });
</script>

<template>
  <el-card shadow="never" class="section memory-panel">
    <template #header>
      <div class="panel-title">
        <span>记忆状态</span>
        <span><el-button text type="primary" @click="refresh">刷新</el-button><el-button text type="danger" @click="clearSession">清除本会话记忆</el-button></span>
      </div>
    </template>
    <p>{{ status || "当前会话尚未写入受控记忆" }}</p>
    <p v-if="jobs[0]" class="memory-job">最新合并任务：{{ jobs[0].status }}<span v-if="jobs[0].error_code">（{{ jobs[0].error_code }}）</span></p>
    <el-empty v-if="!items.length" description="暂无可复用的审核结论" :image-size="42" />
    <div v-for="memory in items.slice(0, 6)" :key="memory.memory_id" class="history-turn">
      <strong>{{ memory.insight_text }}</strong>
      <p>{{ memory.start_date }} 至 {{ memory.end_date }} · {{ memory.status }} · Trace: {{ memory.source_trace_id }}</p>
      <template v-if="memory.status === 'pending_review'">
        <el-button text type="primary" @click="review(memory, 'approve_independent')">批准独立保留</el-button>
        <el-button text type="warning" @click="review(memory, 'supersede')">取代旧结论</el-button>
        <el-button text type="danger" @click="review(memory, 'reject')">驳回</el-button>
      </template>
      <template v-else-if="memory.status === 'active'">
        <el-button text type="primary" @click="requestConsolidation(memory)">生成合并提案</el-button>
        <el-button text type="danger" @click="revoke(memory.memory_id)">撤销</el-button>
      </template>
    </div>
  </el-card>
</template>
