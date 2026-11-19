<script setup lang="ts">
// 连接与权限视图：登录/退出、首次运行创建平台管理员、LLM 连接测试、Token 用量展示。
import { onMounted, ref } from "vue";
import { ElMessage } from "element-plus";
import { api } from "@/api";
import { useAppStore } from "@/stores/app";

const store = useAppStore();
// 表单状态：邮箱/密码/管理员显示名/Bootstrap Token；result 为提示文案，loading/initializing 控制按钮态。
const email = ref(""); const password = ref(""); const displayName = ref("平台管理员"); const bootstrapToken = ref(""); const result = ref(""); const loading = ref(false); const initializing = ref(false); const usage = ref<Awaited<ReturnType<typeof api.usage>> | null>(null);
// 加载近 30 天 Token 用量（未登录时跳过）。
async function loadUsage() { if (store.bearerToken) usage.value = await api.usage(store.auth); }
// 登录：调用 store.login，成功后刷新账户信息并清空密码。
async function signIn() { loading.value = true; try { await store.login(email.value, password.value); await store.refreshAccount(); ElMessage.success("登录成功"); password.value = ""; } catch (error) { result.value = error instanceof Error ? error.message : "登录失败"; } finally { loading.value = false; } }
// 退出当前登录。
async function signOut() { await store.logout(); ElMessage.success("已退出当前会话"); }
// 首次运行初始化：用一次性 Bootstrap Token 创建平台管理员并自动登录。
async function bootstrap() { initializing.value = true; try { await api.bootstrap(bootstrapToken.value, email.value, password.value, displayName.value); await store.login(email.value, password.value); await store.refreshAccount(); bootstrapToken.value = ""; ElMessage.success("平台管理员已创建并登录"); } catch (error) { result.value = error instanceof Error ? error.message : "初始化失败"; } finally { initializing.value = false; } }
// 测试 LLM 连接：展示连通性、回显与延迟。
async function test() { loading.value = true; try { const response = await api.testLlm(store.auth); result.value = response.status === "ok" ? `连接成功：${response.reply ?? "ok"}（${response.latency_ms ?? "-"} ms）` : response.detail ?? "连接失败"; } catch (error) { result.value = error instanceof Error ? error.message : "连接测试失败"; } finally { loading.value = false; } }
// 挂载时刷新账户信息与用量（失败静默，不打断页面渲染）。
onMounted(async () => { await store.refreshAccount().catch(() => undefined); await loadUsage().catch(() => undefined); });
</script>

<template>
  <div class="page narrow"><div class="page-title"><div><h1>账户与权限</h1><p>登录后，系统会按你的组织、角色和店铺授权范围限制数据、报告与分析任务。</p></div></div>
    <el-card v-if="!store.bearerToken" shadow="never"><el-alert type="info" :closable="false" title="请先登录后使用数据、报告与质量仪表盘。首次运行且尚未创建管理员时，展开下方初始化区域。" /><el-form class="section" label-position="top" @submit.prevent="signIn"><el-form-item label="邮箱"><el-input v-model="email" autocomplete="email" /></el-form-item><el-form-item label="密码"><el-input v-model="password" type="password" show-password autocomplete="current-password" @keyup.enter="signIn" /></el-form-item><el-button type="primary" :loading="loading" @click="signIn">登录</el-button></el-form><el-collapse class="section"><el-collapse-item title="首次运行：创建平台管理员" name="bootstrap"><el-alert type="warning" :closable="false" title="仅在系统尚未创建任何账号时使用。Bootstrap Token 来自本机 .env，不会保存到浏览器。" /><el-form class="section" label-position="top"><el-form-item label="管理员显示名"><el-input v-model="displayName" /></el-form-item><el-form-item label="Bootstrap Token"><el-input v-model="bootstrapToken" type="password" show-password autocomplete="off" /></el-form-item><el-button type="warning" :loading="initializing" @click="bootstrap">创建并登录</el-button></el-form></el-collapse-item></el-collapse><el-alert v-if="result" class="section" :title="result" type="error" :closable="false" /></el-card>
    <template v-else><el-card shadow="never"><el-descriptions :column="1" border><el-descriptions-item label="当前用户">{{ store.profile.displayName }}</el-descriptions-item><el-descriptions-item label="组织">{{ store.account?.tenantName ?? "正在读取…" }}</el-descriptions-item><el-descriptions-item label="角色">{{ store.account?.role ?? "-" }}</el-descriptions-item><el-descriptions-item label="可用店铺"><span v-if="store.account?.sellerGrants?.length">{{ store.account.sellerGrants.map((item) => item.seller_id.slice(0, 8)).join("、") }}</span><span v-else>全平台（仅平台运营管理员）</span></el-descriptions-item><el-descriptions-item label="功能权限">{{ store.account?.permissions?.join("、") ?? "-" }}</el-descriptions-item></el-descriptions><el-space class="section"><el-button @click="store.refreshAccount">刷新权限</el-button><el-button type="primary" :loading="loading" @click="test">测试 LLM 连接</el-button><el-button type="danger" plain @click="signOut">退出登录</el-button></el-space><el-alert v-if="result" :title="result" :type="result.includes('成功') ? 'success' : 'error'" :closable="false" /></el-card></template>
    <el-card v-if="store.bearerToken" class="section" shadow="never">
      <template #header><strong>Token 用量（近 30 天）</strong><el-button link type="primary" @click="loadUsage">刷新</el-button></template>
      <el-empty v-if="!usage || usage.summary.calls === 0" description="暂无供应商返回的 Token 用量；规则降级回答不会计入。" />
      <template v-else>
        <el-row :gutter="12"><el-col :span="8"><el-statistic title="总 Token" :value="usage.summary.total_tokens" /></el-col><el-col :span="8"><el-statistic title="输入 / 输出" :value="`${usage.summary.input_tokens} / ${usage.summary.output_tokens}`" /></el-col><el-col :span="8"><el-statistic title="调用次数" :value="usage.summary.calls" /></el-col></el-row>
        <p class="muted">费用按当前配置单价估算；未配置单价时显示 0，不代表模型免费。</p>
        <el-table :data="usage.items" size="small" max-height="260"><el-table-column prop="created_at" label="时间" width="180" /><el-table-column prop="operation" label="调用" width="90" /><el-table-column prop="model" label="模型" min-width="130" /><el-table-column prop="input_tokens" label="输入" width="80" /><el-table-column prop="output_tokens" label="输出" width="80" /><el-table-column prop="total_tokens" label="合计" width="80" /></el-table>
      </template>
    </el-card>
  </div>
</template>
