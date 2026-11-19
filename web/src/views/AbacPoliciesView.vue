<script setup lang="ts">
import { onMounted, ref } from "vue";
import { ElMessage } from "element-plus";
import { api, ApiError } from "@/api";
import { useAppStore } from "@/stores/app";

const store = useAppStore();
const items = ref<Array<Record<string, unknown>>>([]);
const policyId = ref("");
const status = ref<"active" | "disabled">("active");
const ruleText = ref('{"role":"analyst","agent":"data_query","allow":["query_gmv"],"deny":[],"conditions":{"scope_types":["seller"]}}');
const loading = ref(false);
const policyState = ref<"loading" | "ready" | "disabled" | "error">("loading");

async function load() {
  try {
    items.value = (await api.abacPolicies(store.auth)).items;
    policyState.value = "ready";
  } catch (error) {
    if (error instanceof ApiError && error.status === 403) {
      policyState.value = "disabled";
      return;
    }
    policyState.value = "error";
    throw error;
  }
}
async function save() {
  loading.value = true;
  try {
    if (!policyId.value.trim()) throw new Error("请输入策略 ID");
    const rule = JSON.parse(ruleText.value) as Record<string, unknown>;
    await api.saveAbacPolicy(store.auth, policyId.value.trim(), { rule, status: status.value });
    await load();
    ElMessage.success("ABAC 策略已保存");
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "策略保存失败");
  } finally { loading.value = false; }
}
async function remove(id: string) {
  await api.deleteAbacPolicy(store.auth, id);
  await load();
  ElMessage.success("策略已删除");
}
onMounted(() => load().catch((error) => ElMessage.error(error instanceof Error ? error.message : "加载策略失败")));
</script>

<template>
  <div class="page narrow">
    <div class="page-title"><div><h1>ABAC 策略</h1><p>仅管理员可以编辑租户策略。策略由服务端校验，未命中默认拒绝，显式拒绝优先。</p></div></div>
    <el-alert v-if="policyState === 'loading'" class="section" type="info" :closable="false" title="正在检查策略管理状态…" />
    <el-alert v-else-if="policyState === 'disabled'" class="section" type="info" :closable="false" title="当前服务未启用租户数据库 ABAC 策略管理。普通分析仍使用 RBAC 工具基线；此页面不会读取或修改策略。" />
    <el-alert v-else-if="policyState === 'error'" class="section" type="warning" :closable="false" title="暂时无法读取策略管理状态，请稍后刷新。" />
    <template v-else>
    <el-alert type="info" :closable="false" title="策略只能声明角色、工具、允许/拒绝集合和受控条件，不允许执行任意代码。" />
    <el-card class="section" shadow="never">
      <el-form label-position="top">
        <el-form-item label="策略 ID"><el-input v-model="policyId" placeholder="例如 analyst-data-query" /></el-form-item>
        <el-form-item label="规则 JSON"><el-input v-model="ruleText" type="textarea" :rows="7" /></el-form-item>
        <el-form-item label="状态"><el-select v-model="status"><el-option label="启用" value="active" /><el-option label="停用" value="disabled" /></el-select></el-form-item>
        <el-button type="primary" :loading="loading" @click="save">保存策略</el-button>
      </el-form>
    </el-card>
    <el-card class="section" shadow="never">
      <template #header><strong>当前策略</strong><el-button link type="primary" @click="load">刷新</el-button></template>
      <el-table v-if="items.length" :data="items" size="small"><el-table-column prop="policy_id" label="策略 ID" /><el-table-column prop="policy_version" label="版本" width="90" /><el-table-column prop="status" label="状态" width="90" /><el-table-column label="操作" width="90"><template #default="scope"><el-button link type="danger" @click="remove(String(scope.row.policy_id))">删除</el-button></template></el-table-column></el-table>
      <el-empty v-else description="当前租户没有数据库策略" />
    </el-card>
    </template>
  </div>
</template>
