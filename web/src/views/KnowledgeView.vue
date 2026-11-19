<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { ElMessage } from "element-plus";
import { api } from "@/api"; import { useAppStore } from "@/stores/app";
const store = useAppStore(); const data = ref<Record<string, any> | null>(null); const indexing = ref(false);
const canManageKnowledge = computed(() => store.account?.permissions.includes("manage_knowledge") ?? false);
async function load() { try { data.value = await api.knowledge(store.auth) as Record<string, any>; } catch (error) { ElMessage.error(error instanceof Error ? error.message : "读取知识库失败"); } }
async function index() { const collection = String(data.value?.collection ?? "").trim(); if (!collection) return ElMessage.error("无法读取知识库目标集合"); indexing.value = true; try { const result = await api.indexKnowledge(store.auth, collection); ElMessage.success(`索引完成：${String(result.status ?? "ok")}`); await load(); } catch (error) { ElMessage.error(error instanceof Error ? error.message : "索引失败"); } finally { indexing.value = false; } }
onMounted(load);
</script>
<template><div class="page"><div class="page-title"><div><h1>知识库</h1><p>查看 RAG 文档清单与索引版本；索引更新将由后端执行。</p></div><div><el-button @click="load">刷新</el-button><el-button v-if="canManageKnowledge" type="primary" :loading="indexing" :disabled="!data?.collection" @click="index">更新索引</el-button></div></div><el-row :gutter="16"><el-col :md="8"><el-card shadow="never" class="summary-card"><span>文档数量</span><strong>{{ data?.total ?? '-' }}</strong><small>索引版本：{{ data?.index_version ?? '-' }}</small></el-card></el-col><el-col :md="16"><el-card shadow="never"><template #header>知识库根目录</template>{{ data?.knowledge_root ?? '正在读取…' }}</el-card></el-col></el-row><el-card shadow="never" class="section"><template #header>文档列表</template><el-empty v-if="data && !data.documents?.length" description="暂无知识文档" /><el-table v-else :data="data?.documents ?? []" stripe><el-table-column prop="title" label="标题" /><el-table-column prop="doc_type" label="类型" /><el-table-column prop="metric_name" label="关联指标" /><el-table-column prop="source_path" label="来源路径" /><el-table-column prop="index_version" label="索引版本" /></el-table></el-card></div></template>
