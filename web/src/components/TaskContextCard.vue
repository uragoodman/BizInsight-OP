<script setup lang="ts">
import { ref, watch } from "vue";
import { api } from "@/api";
import { useAppStore } from "@/stores/app";

const props = defineProps<{ sessionId?: string }>();
const store = useAppStore();
const data = ref<Record<string, any> | null>(null);

const taskStatusLabels: Record<string, string> = {
  draft: "草稿",
  awaiting_scope: "等待补充范围",
  clarification_required: "等待澄清",
  awaiting_confirmation: "等待确认",
  queued: "等待执行",
  running: "进行中",
  completed: "已完成",
  abstained: "已安全停止",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
  superseded: "已被替代",
};

async function refresh() {
  if (!props.sessionId) return;
  data.value = await api.taskContext(store.auth, props.sessionId);
}

function taskStatusLabel(status: string) {
  return taskStatusLabels[status] ?? status;
}

watch(() => props.sessionId, () => refresh().catch(() => data.value = null), { immediate: true });
</script>

<template>
  <el-card v-if="sessionId" class="task-context-card" shadow="never">
    <template #header>
      <div class="task-context-header">
        <div>
          <strong>任务与上下文</strong>
          <small>查看任务状态和 Agent 延续对话时参考的信息</small>
        </div>
        <el-button link type="primary" @click="refresh">刷新</el-button>
      </div>
    </template>

    <div v-if="!data?.current_task" class="task-empty">本会话暂无当前任务</div>
    <template v-else>
      <el-tag>{{ taskStatusLabel(data.current_task.status) }}</el-tag>
      <p><strong>{{ data.current_task.title }}</strong></p>
      <p class="muted">{{ data.current_task.objective }}</p>

      <el-tabs>
        <el-tab-pane label="本轮（L0）">
          <p>任务状态：{{ taskStatusLabel(data.memory?.l0?.status || "idle") }}。只用于当前任务。</p>
        </el-tab-pane>
        <el-tab-pane label="会话摘要（L1）">
          <p>{{ data.memory?.l1?.summary || "尚未形成会话摘要" }}</p>
          <small>用于同一会话后续提问的上下文延续。</small>
        </el-tab-pane>
        <el-tab-pane label="历史结论（L2）">
          <el-empty v-if="!data.memory?.l2?.items?.length" description="暂无已审核的历史结论" :image-size="24" />
          <p v-for="item in data.memory?.l2?.items?.slice(0, 3)" :key="item.memory_id">
            {{ item.insight_text }}
          </p>
          <small v-if="data.memory?.l2?.items?.length">仅供参考，不作为本轮数值依据。</small>
        </el-tab-pane>
      </el-tabs>
    </template>
  </el-card>
</template>

<style scoped>
.task-context-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 6px; }
.task-context-header > div { min-width: 0; }
.task-context-header strong, .task-context-header small { display: block; }
.task-context-header strong { font-size: 13px; }
.task-context-header small { margin-top: 4px; color: #737780; font-size: 10px; font-weight: 400; line-height: 1.45; }
.task-context-card p { overflow-wrap: anywhere; line-height: 1.55; }
.task-context-card > :deep(.el-card__header) { padding: 12px; }
.task-context-card > :deep(.el-card__body) { padding: 12px; }
.task-context-card :deep(.el-tabs__item) { padding: 0 7px; font-size: 11px; }
.task-context-card :deep(.el-tabs__content) { font-size: 12px; }
.task-context-card small { color: #737780; line-height: 1.5; }
</style>
