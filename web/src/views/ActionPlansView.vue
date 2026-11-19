<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { api } from "@/api";
import { useAppStore } from "@/stores/app";
import type { ActionPlan } from "@/types";

const store = useAppStore(); const plans = ref<ActionPlan[]>([]); const selected = ref<ActionPlan | null>(null); const text = ref(""); const loading = ref(false); const next = ref<number | null>(0); const canManage = computed(() => store.account?.permissions.includes("analyze") ?? false);
async function load(reset = false) {
  if (reset) { plans.value=[]; next.value=0; }
  if (next.value === null) return;
  loading.value=true;
  try {
    const response = await api.actionPlans(store.auth,next.value) as unknown;
    const page = response as { items?: ActionPlan[]; next_offset?: number | null };
    const items = Array.isArray(response) ? response : page.items;
    if (!Array.isArray(items)) throw new Error("行动计划列表返回格式异常，请刷新后端后重试。");
    plans.value.push(...items);
    next.value = Array.isArray(response) ? null : (page.next_offset ?? null);
    if (!selected.value && plans.value[0]) await select(plans.value[0]);
  } catch (e) { ElMessage.error(e instanceof Error ? e.message : "加载行动计划失败"); }
  finally { loading.value=false; }
}
async function select(plan: ActionPlan) { selected.value=await api.actionPlan(store.auth,plan.action_plan_id); text.value=JSON.stringify(selected.value.content,null,2); }
async function save() { if (!selected.value) return; try { const content=JSON.parse(text.value); selected.value=await api.updateActionPlan(store.auth,selected.value.action_plan_id,{title:selected.value.title,content,status:selected.value.status}); const i=plans.value.findIndex(p=>p.action_plan_id===selected.value!.action_plan_id); if(i>=0) plans.value[i]=selected.value; ElMessage.success("经营建议已保存"); } catch(e) { ElMessage.error(e instanceof Error ? e.message : "内容必须是合法 JSON"); } }
async function remove() { if (!selected.value) return; try { await ElMessageBox.confirm("删除后无法恢复该行动草案。","删除行动草案",{type:"warning"}); await api.deleteActionPlan(store.auth,selected.value.action_plan_id); plans.value=plans.value.filter(p=>p.action_plan_id!==selected.value!.action_plan_id); selected.value=null; text.value=""; } catch { /* cancelled */ } }
onMounted(()=>load(true));
</script>
<template><div class="page"><div class="page-title"><div><h1>经营建议</h1><p>这里保存复杂分析生成的、带有证据和限制条件的经营建议草案；系统不会自动执行业务动作。</p></div><el-button @click="load(true)">刷新</el-button></div><el-row :gutter="16"><el-col :lg="8"><el-card shadow="never"><el-empty v-if="!plans.length&&!loading" description="暂无经营建议"/><el-menu v-else @select="(id: string) => { const plan = plans.find(p => p.action_plan_id === id); if (plan) select(plan); }"><el-menu-item v-for="plan in plans" :key="plan.action_plan_id" :index="plan.action_plan_id">{{ plan.title }} · {{ plan.status }}</el-menu-item></el-menu><el-button v-if="next!==null" text type="primary" :loading="loading" @click="load()">加载更多</el-button></el-card></el-col><el-col :lg="16"><el-empty v-if="!selected" description="选择一条经营建议查看详情"/><el-card v-else shadow="never"><template #header><div class="head"><span>证据化经营建议 · v{{ selected.version }}<small v-if="!canManage"> · 只读</small></span><div v-if="canManage"><el-button size="small" @click="save">保存</el-button><el-button size="small" type="danger" @click="remove">删除</el-button></div></div></template><el-form label-position="top"><el-form-item label="标题"><el-input v-model="selected.title" :readonly="!canManage"/></el-form-item><el-form-item label="状态"><el-select v-model="selected.status" :disabled="!canManage"><el-option label="草案" value="draft"/><el-option label="已采纳" value="accepted"/><el-option label="已归档" value="archived"/></el-select></el-form-item><el-form-item label="内容（可编辑 JSON）"><el-input v-model="text" type="textarea" :rows="18" :readonly="!canManage"/></el-form-item></el-form></el-card></el-col></el-row></div></template>
<style scoped>.head{display:flex;justify-content:space-between;align-items:center}.head>div{display:flex;gap:8px}</style>
