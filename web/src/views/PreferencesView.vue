<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { api } from "@/api";
import { useAppStore } from "@/stores/app";
import type { ChatSessionPage, PreferenceHistoryItem, PreferenceItem } from "@/types";

const store = useAppStore();
const items = ref<PreferenceItem[]>([]);
const sessions = ref<ChatSessionPage["items"]>([]);
const selectedSessionId = ref("");
const personalizationEnabled = ref(true);
const loading = ref(false);
const saving = ref(false);
const editorVisible = ref(false);
const editing = ref<PreferenceItem | null>(null);
const historyVisible = ref(false);
const historyItems = ref<PreferenceHistoryItem[]>([]);
const historyTitle = ref("");
const memoryType = ref<PreferenceItem["memory_type"]>("presentation");
const memoryKey = ref("response_style");
const valueText = ref('{\n  "style": "balanced"\n}');

const typeLabels: Record<PreferenceItem["memory_type"], string> = {
  presentation: "呈现偏好",
  analysis_method: "分析方法偏好",
  scope_default: "默认分析范围",
};
const keyChoices = computed(() => {
  if (memoryType.value === "presentation") return [
    { key: "response_style", label: "回答风格" },
    { key: "report_style", label: "报告格式" },
    { key: "response_language", label: "回答语言" },
  ];
  if (memoryType.value === "analysis_method") return [
    { key: "analysis_method", label: "分析方法" },
    { key: "analysis_steps", label: "分析步骤" },
  ];
  return [
    { key: "default_period", label: "默认日期区间" },
    { key: "default_date", label: "默认日期" },
  ];
});
const valueHint = computed(() => {
  if (memoryType.value === "presentation") {
    if (memoryKey.value === "response_style") return '{ "style": "concise | balanced | detailed" }';
    if (memoryKey.value === "report_style") return '{ "format": "table | bullets | narrative" }';
    return '{ "language": "zh-CN | zh-TW | en-US" }';
  }
  if (memoryType.value === "analysis_method") return '{ "metric": "gmv", "dimension": "product_category" }';
  return memoryKey.value === "default_period"
    ? '{ "start_date": "2026-01", "end_date": "2026-03" }'
    : '{ "date": "2026-03" }';
});

function newIdempotencyKey() { return crypto.randomUUID(); }
function statusLabel(item: PreferenceItem) {
  const labels: Record<string, string> = {
    active: "已保存", pending_create: "等待保存", pending_update: "等待更新",
    pending_delete: "等待删除", failed: "失败，可重试", superseded: "已替换",
  };
  return item.overlay?.status === "failed" ? "当前会话暂存失败" : labels[item.status] ?? item.status;
}
function parseValue(): Record<string, unknown> {
  const parsed: unknown = JSON.parse(valueText.value);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("偏好值必须是 JSON 对象");
  return parsed as Record<string, unknown>;
}
function exampleValue(type: PreferenceItem["memory_type"], key: string): string {
  if (type === "presentation") {
    const value = key === "response_style" ? { style: "balanced" }
      : key === "report_style" ? { format: "table" } : { language: "zh-CN" };
    return JSON.stringify(value, null, 2);
  }
  if (type === "analysis_method") return JSON.stringify(
    key === "analysis_steps" ? { metric: "gmv", operation: "compare" } : { metric: "gmv", dimension: "product_category" }, null, 2,
  );
  const currentMonth = new Date().toISOString().slice(0, 7);
  return JSON.stringify(key === "default_period"
    ? { start_date: currentMonth, end_date: currentMonth }
    : { date: currentMonth }, null, 2);
}
function changeMemoryType(type: PreferenceItem["memory_type"]) {
  memoryKey.value = type === "presentation" ? "response_style"
    : type === "analysis_method" ? "analysis_method" : "default_period";
  valueText.value = exampleValue(type, memoryKey.value);
}
function changeMemoryKey(key: string) {
  memoryKey.value = key;
  valueText.value = exampleValue(memoryType.value, key);
}
function startCreate() {
  editing.value = null;
  memoryType.value = "presentation";
  memoryKey.value = "response_style";
  valueText.value = exampleValue("presentation", "response_style");
  editorVisible.value = true;
}
function startEdit(item: PreferenceItem) {
  editing.value = item;
  memoryType.value = item.memory_type;
  memoryKey.value = item.memory_key;
  valueText.value = JSON.stringify(item.value ?? {}, null, 2);
  editorVisible.value = true;
}
async function refresh() {
  if (!store.bearerToken) return;
  loading.value = true;
  try {
    const [preferenceResult, sessionResult] = await Promise.all([
      api.preferences(store.auth, selectedSessionId.value || undefined),
      api.chatSessions(store.auth, 0, 50),
    ]);
    items.value = preferenceResult.items;
    personalizationEnabled.value = preferenceResult.personalization_enabled;
    sessions.value = sessionResult.items;
    if (!selectedSessionId.value || !sessions.value.some((session) => session.id === selectedSessionId.value)) {
      selectedSessionId.value = sessions.value[0]?.id ?? "";
      if (selectedSessionId.value) {
        const refreshed = await api.preferences(store.auth, selectedSessionId.value);
        items.value = refreshed.items;
        personalizationEnabled.value = refreshed.personalization_enabled;
      }
    }
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "加载偏好失败");
  } finally { loading.value = false; }
}
async function createSession() {
  try {
    const session = await api.createChatSession(store.auth, "偏好设置");
    selectedSessionId.value = session.id;
    await refresh();
    ElMessage.success("已创建可用于偏好操作的会话");
  } catch (error) { ElMessage.error(error instanceof Error ? error.message : "创建会话失败"); }
}
async function savePreference() {
  if (!selectedSessionId.value) { ElMessage.warning("请先创建或选择会话"); return; }
  let value: Record<string, unknown>;
  try { value = parseValue(); }
  catch (error) { ElMessage.error(error instanceof Error ? error.message : "偏好 JSON 格式无效"); return; }
  const action = editing.value ? "更新" : "保存";
  try {
    await ElMessageBox.confirm(
      `确认${action}“${memoryKey.value}”偏好吗？此内容会用于后续对话的个性化呈现。`,
      `确认${action}偏好`, { type: "warning", confirmButtonText: `确认${action}`, cancelButtonText: "取消" },
    );
  } catch { return; }
  saving.value = true;
  try {
    if (editing.value) {
      await api.updatePreference(store.auth, editing.value.registry_id, {
        value, chat_session_id: selectedSessionId.value,
        expected_version: editing.value.version, idempotency_key: newIdempotencyKey(), confirmed: true,
      });
    } else {
      await api.createPreference(store.auth, {
        memory_key: memoryKey.value, memory_type: memoryType.value, value,
        chat_session_id: selectedSessionId.value, idempotency_key: newIdempotencyKey(), confirmed: true,
      });
    }
    editorVisible.value = false;
    await refresh();
    ElMessage.success("操作已排队；完成对账后才会显示为已保存");
  } catch (error) { ElMessage.error(error instanceof Error ? error.message : "偏好操作失败"); }
  finally { saving.value = false; }
}
async function removePreference(item: PreferenceItem) {
  try {
    await ElMessageBox.confirm(`确认删除“${item.memory_key}”偏好吗？删除将在后台完成对账。`, "删除偏好", { type: "warning", confirmButtonText: "确认删除", cancelButtonText: "取消" });
    await api.deletePreference(store.auth, item.registry_id, {
      expected_version: item.version, idempotency_key: newIdempotencyKey(), confirmed: true,
    });
    await refresh();
    ElMessage.success("删除已排队；对账完成后该偏好才会彻底移除");
  } catch (error) {
    if (error !== "cancel" && error !== "close") ElMessage.error(error instanceof Error ? error.message : "删除偏好失败");
  }
}
async function retry(item: PreferenceItem) {
  try { await api.retryPreference(store.auth, item.registry_id); await refresh(); ElMessage.success("已重新排队"); }
  catch (error) { ElMessage.error(error instanceof Error ? error.message : "重试失败"); }
}
async function togglePersonalization(value: boolean | string | number) {
  const enabled = Boolean(value);
  try {
    await ElMessageBox.confirm(
      enabled ? "开启后，后续回答可以读取你已保存的偏好。" : "关闭后，系统将停止在后续回答中使用偏好。已保存内容不会自动删除。",
      enabled ? "开启个性化" : "关闭个性化", { type: "warning", confirmButtonText: "确认", cancelButtonText: "取消" },
    );
    const result = await api.setPersonalization(store.auth, enabled);
    personalizationEnabled.value = result.personalization_enabled;
    ElMessage.success(enabled ? "已开启个性化" : "已关闭个性化");
  } catch (error) {
    personalizationEnabled.value = !enabled;
    if (error !== "cancel" && error !== "close") ElMessage.error(error instanceof Error ? error.message : "设置失败");
  }
}
async function showHistory(item: PreferenceItem) {
  try {
    const result = await api.preferenceHistory(store.auth, item.registry_id);
    historyItems.value = result.items;
    historyTitle.value = item.memory_key;
    historyVisible.value = true;
  } catch (error) { ElMessage.error(error instanceof Error ? error.message : "读取历史失败"); }
}
async function clearAll() {
  try {
    const { value } = await ElMessageBox.prompt('输入 DELETE_ALL_PREFERENCES 以确认删除所有偏好。', "删除全部偏好", {
      type: "warning", confirmButtonText: "继续", cancelButtonText: "取消", inputPlaceholder: "DELETE_ALL_PREFERENCES",
      inputValidator: (input) => input === "DELETE_ALL_PREFERENCES" || "确认短语不匹配",
    });
    if (value !== "DELETE_ALL_PREFERENCES") return;
    const result = await api.deleteAllPreferences(store.auth, newIdempotencyKey());
    await refresh();
    ElMessage.warning(`已排队删除 ${result.queued_count} 项；${result.unfinished.length} 项未能排队。`);
  } catch (error) {
    if (error !== "cancel" && error !== "close") ElMessage.error(error instanceof Error ? error.message : "删除全部偏好失败");
  }
}
function formatDate(value?: string | null) { return value ? new Date(value).toLocaleString() : "—"; }
onMounted(refresh);
</script>

<template>
  <div class="page narrow">
    <div class="page-title"><div><h1>记忆与偏好</h1><p>这里只管理你明确保存的偏好。普通聊天不会自动写入长期记忆；新建、修改和删除均需确认。</p></div></div>
    <el-card shadow="never" class="section">
      <div class="preference-toolbar">
        <div><strong>个性化使用</strong><p class="muted">关闭后保留偏好记录，但后续回答不会使用它们。</p></div>
        <el-switch :model-value="personalizationEnabled" active-text="开启" inactive-text="关闭" @change="togglePersonalization" />
      </div>
      <el-alert v-if="!personalizationEnabled" class="section" type="info" :closable="false" title="个性化已关闭；已有偏好不会参与后续回答。" />
    </el-card>

    <el-card shadow="never" class="section">
      <template #header>
        <div class="preference-toolbar"><strong>已保存的偏好</strong><div>
          <el-button @click="refresh" :loading="loading">刷新</el-button>
          <el-button type="primary" :disabled="!personalizationEnabled" @click="startCreate">添加偏好</el-button>
          <el-button type="danger" plain @click="clearAll">删除全部</el-button>
        </div></div>
      </template>
      <el-form-item label="当前会话（用于本轮即时生效）">
        <el-select v-model="selectedSessionId" placeholder="选择聊天会话" style="width: min(100%, 520px)" @change="refresh">
          <el-option v-for="session in sessions" :key="session.id" :label="`${session.title} · ${session.turn_count} 轮`" :value="session.id" />
        </el-select>
        <el-button class="session-create" @click="createSession">新建会话</el-button>
      </el-form-item>
      <el-alert v-if="!sessions.length" type="info" :closable="false" title="偏好写入需要一个持久会话，用于同会话即时生效。点击“新建会话”即可开始。" />
      <el-empty v-if="!loading && !items.length" description="还没有保存的偏好" :image-size="56" />
      <div v-for="item in items" :key="item.registry_id" class="preference-item">
        <div class="preference-item-main">
          <div class="preference-item-title"><strong>{{ item.memory_key }}</strong><el-tag size="small">{{ typeLabels[item.memory_type] }}</el-tag><el-tag size="small" :type="item.status === 'active' ? 'success' : item.status === 'failed' ? 'danger' : 'warning'">{{ statusLabel(item) }}</el-tag></div>
          <code>{{ JSON.stringify(item.value ?? {}) }}</code>
          <p class="muted">范围：{{ item.scope.seller_id ? `店铺 ${item.scope.seller_id}` : "个人" }} · 更新于 {{ formatDate(item.updated_at) }}<span v-if="item.requires_confirmation"> · 已过期，需重新确认</span></p>
          <p v-if="item.overlay?.error_code" class="error-text">当前会话暂存错误：{{ item.overlay.error_code }}</p>
        </div>
        <div class="preference-actions">
          <el-button link @click="showHistory(item)">历史</el-button>
          <el-button v-if="item.status === 'active'" link type="primary" @click="startEdit(item)">修改</el-button>
          <el-button v-if="item.status === 'failed'" link type="warning" @click="retry(item)">重试</el-button>
          <el-button v-if="item.status !== 'pending_delete' && item.status !== 'superseded'" link type="danger" @click="removePreference(item)">删除</el-button>
        </div>
      </div>
    </el-card>

    <el-dialog v-model="editorVisible" :title="editing ? '修改偏好' : '添加偏好'" width="560px">
      <el-form label-position="top">
        <el-form-item label="偏好类型"><el-select :model-value="memoryType" :disabled="Boolean(editing)" @change="changeMemoryType"><el-option v-for="(label, key) in typeLabels" :key="key" :label="label" :value="key" /></el-select></el-form-item>
        <el-form-item label="偏好字段"><el-select :model-value="memoryKey" :disabled="Boolean(editing)" @change="changeMemoryKey"><el-option v-for="choice in keyChoices" :key="choice.key" :label="choice.label" :value="choice.key" /></el-select></el-form-item>
        <el-form-item label="值（JSON 对象）"><el-input v-model="valueText" type="textarea" :rows="6" spellcheck="false" /><p class="muted">示例：{{ valueHint }}。服务器会再次校验允许字段和值。</p></el-form-item>
        <el-alert v-if="!personalizationEnabled" type="warning" :closable="false" title="个性化已关闭，当前不能新增或更新偏好。删除仍然可用。" />
      </el-form>
      <template #footer><el-button @click="editorVisible = false">取消</el-button><el-button type="primary" :disabled="!personalizationEnabled" :loading="saving" @click="savePreference">{{ editing ? "确认修改" : "确认保存" }}</el-button></template>
    </el-dialog>

    <el-dialog v-model="historyVisible" :title="`${historyTitle} · 操作历史`" width="680px">
      <el-table :data="historyItems" size="small"><el-table-column prop="created_at" label="时间" min-width="170"><template #default="scope">{{ formatDate(scope.row.created_at) }}</template></el-table-column><el-table-column prop="event_type" label="操作" /><el-table-column prop="status" label="状态" /><el-table-column prop="reason_code" label="原因" /></el-table>
      <el-empty v-if="!historyItems.length" description="暂无操作记录" />
    </el-dialog>
  </div>
</template>

<style scoped>
.preference-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
.preference-toolbar p { margin: 6px 0 0; }
.preference-item { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; padding: 16px 0; border-top: 1px solid var(--el-border-color-lighter); }
.preference-item-title { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }
.preference-item code { display: block; white-space: pre-wrap; overflow-wrap: anywhere; color: var(--el-text-color-regular); }
.preference-item p { margin: 8px 0 0; }
.preference-actions { display: flex; flex-shrink: 0; }
.session-create { margin-left: 8px; }
.error-text { color: var(--el-color-danger); }
@media (max-width: 680px) { .preference-toolbar, .preference-item { align-items: flex-start; flex-direction: column; } .preference-actions { flex-wrap: wrap; } }
</style>
