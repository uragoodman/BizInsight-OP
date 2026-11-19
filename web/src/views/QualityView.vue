<script setup lang="ts">
// 质量仪表盘视图：展示版本化的质量证据——Golden、pytest 回归、性能基准、
// Trace 分代统计与已保存快照；仅管理员（manage_usage 权限）可访问。
import { computed, onMounted, ref, watch } from "vue";
import { ElMessage } from "element-plus";
import { api } from "@/api";
import { useAppStore } from "@/stores/app";
import { useRouter } from "vue-router";

const store = useAppStore();
const router = useRouter();
// 数据状态：report 为后端质量快照；selectedBucket 为当前选中的 Trace 分代桶。
const report = ref<any>(null); const evaluationRuns = ref<any[]>([]); const loading = ref(false); const selectedBucket = ref("");
// 派生状态：latest 最新快照；bucketOptions 全部分桶；traces 当前桶的统计（未选中时用默认桶）。
const latest = computed(() => report.value?.latest);
const bucketOptions = computed(() => latest.value?.traces?.buckets ?? []);
const traces = computed(() => bucketOptions.value.find((item: any) => item.bucket === selectedBucket.value) ?? latest.value?.traces);
const golden = computed(() => latest.value?.golden); const bench = computed(() => latest.value?.benchmark); const tests = computed(() => latest.value?.tests);
const insights = computed(() => latest.value?.insights);
const collectionChecks = computed(() => {
  const trace = traces.value; const insight = insights.value;
  return [
    { name: "流程完成率 / Critic 覆盖", ready: Boolean(trace?.completed_traces), detail: trace?.traces ? `${trace.completed_traces ?? 0}/${trace.traces} 条 Trace 已进入终态` : "没有可统计的 Trace", evidence: "完整 Agent 请求，并走到 Critic 终态", path: "/chat", action: "去 Agent 问答" },
    { name: "图执行 p95", ready: Boolean(trace?.graph_latency_ms?.samples), detail: trace?.graph_latency_ms?.samples ? `${trace.graph_latency_ms.samples} 个耗时样本` : "Trace 中没有 node_timings", evidence: "包含节点耗时的完整 Trace", path: "/chat", action: "生成新 Trace" },
    { name: "洞察生成指标", ready: Boolean(insight?.samples), detail: insight?.samples ? `${insight.samples} 份终态洞察` : "尚无终态洞察", evidence: "报告 + AI 解读 + 终态结果", path: "/reports", action: "去分析报告" },
  ];
});
function openCollectionPath(path: string) { router.push(path); }
// 分桶展示名：legacy 桶特殊标注，其余显示分代 + 工作流版本 + Trace 条数。
function bucketLabel(item: any) { return item?.bucket === "legacy" ? "legacy（无版本历史 Trace）" : `${item?.dimensions?.trace_kind ?? "unclassified"} · ${item?.dimensions?.generation} · ${item?.traces} 条`; }
// 加载质量数据，并初始化默认选中的分代桶。
async function load() { loading.value = true; try { report.value = await api.quality(store.auth); evaluationRuns.value = (await api.evaluations(store.auth)).items; if (!selectedBucket.value) selectedBucket.value = latest.value?.traces?.selected_bucket ?? ""; } catch (e) { ElMessage.error(e instanceof Error ? e.message : "无法读取质量证据"); } finally { loading.value = false; } }
// 手动触发后端快照捕获，成功后重新加载。
async function capture() { loading.value = true; try { await api.captureQuality(store.auth); await load(); ElMessage.success("已保存当前版本快照"); } catch (e) { ElMessage.error(e instanceof Error ? e.message : "保存失败"); } finally { loading.value = false; } }
// 分桶列表变化时，保证选中项仍然有效（如数据刷新后旧桶消失）。
watch(bucketOptions, (items) => { if (items.length && !items.some((item: any) => item.bucket === selectedBucket.value)) selectedBucket.value = items[0].bucket; });
// 挂载即加载质量数据。
onMounted(load);
</script>
<template>
  <div class="page">
    <div class="page-title"><div><h1>项目质量仪表盘</h1><p>只展示可复现证据；历史 Trace 与当前版本分桶，不混合统计。</p></div><el-button type="primary" :loading="loading" @click="capture">保存当前快照</el-button></div>
    <el-alert v-if="!store.account?.permissions?.includes('manage_usage')" type="warning" :closable="false" title="该页面仅向管理员开放。" />
    <el-card v-if="latest" class="section"><template #header>Trace 统计分代</template><el-select v-model="selectedBucket" style="min-width: 360px"><el-option v-for="item in bucketOptions" :key="item.bucket" :value="item.bucket" :label="bucketLabel(item)" /></el-select><p class="hint">每个桶仅比较相同的 Trace generation、工作流、Prompt、RAG 发布版与权限策略。legacy 只作历史排查，不用于当前版本简历指标。</p></el-card>
    <el-card class="section collection-status"><template #header><div class="collection-header"><span>采集状态与处理建议</span><el-button size="small" @click="load">重新读取</el-button></div></template><el-table :data="collectionChecks" size="small"><el-table-column prop="name" label="指标" width="210" /><el-table-column label="状态" width="100"><template #default="scope"><el-tag :type="scope.row.ready ? 'success' : 'warning'">{{ scope.row.ready ? '已采集' : '待采集' }}</el-tag></template></el-table-column><el-table-column prop="detail" label="当前情况" min-width="220" /><el-table-column prop="evidence" label="需要的证据" min-width="250" /><el-table-column label="处理"><template #default="scope"><el-button v-if="!scope.row.ready" link type="primary" @click="openCollectionPath(scope.row.path)">{{ scope.row.action }}</el-button><span v-else class="muted">已具备样本</span></template></el-table-column></el-table><p class="collection-note">“保存当前快照”只会保存当前已生成的证据，不会自动运行评测或生成 AI 洞察。</p></el-card>
    <el-row v-if="latest" :gutter="16" class="section">
      <el-col :md="6"><el-card class="summary-card"><span>Golden 验收</span><strong>{{ golden ? `${golden.passed}/${golden.cases}` : "未采集" }}</strong><small>{{ golden ? `${(golden.overall_rate * 100).toFixed(1)}% · ${golden.commit_sha}` : "运行 evaluation.runner 后刷新" }}</small></el-card></el-col>
      <el-col :md="6"><el-card class="summary-card"><span>pytest 回归</span><strong>{{ tests?.exit_code && tests.exit_code !== 0 ? "采集失败" : tests?.passed == null ? "未采集" : `${tests.passed}/${tests.executed}` }}</strong><small>{{ tests?.exit_code && tests.exit_code !== 0 ? "查看 test-last-failure.json 后重跑" : tests?.duration_seconds == null ? "运行 capture_quality_tests" : `${tests.duration_seconds}s · ${tests.commit_sha}` }}</small></el-card></el-col>
      <el-col :md="6"><el-card class="summary-card"><span>流程完成率</span><strong>{{ traces?.completion_rate == null ? "未采集" : `${(traces.completion_rate * 100).toFixed(1)}%` }}</strong><small>{{ traces?.traces ?? 0 }} 条 · Critic 覆盖 {{ traces?.critic_coverage_rate == null ? "—" : `${(traces.critic_coverage_rate * 100).toFixed(1)}%` }}</small></el-card></el-col>
      <el-col :md="6"><el-card class="summary-card"><span>图执行 p95</span><strong>{{ traces?.graph_latency_ms?.p95 == null ? "未采集" : `${traces.graph_latency_ms.p95} ms` }}</strong><small>{{ traces?.graph_latency_ms?.samples ?? 0 }} 个有节点耗时的样本</small></el-card></el-col>
    </el-row>
    <el-row v-if="latest" :gutter="16" class="section">
      <el-col :md="6"><el-card class="summary-card"><span>洞察生成成功率</span><strong>{{ insights?.success_rate == null ? "未采集" : `${(insights.success_rate * 100).toFixed(1)}%` }}</strong><small>{{ insights?.samples ?? 0 }} 份终态洞察 · 待复核 {{ insights?.pending_review_count ?? 0 }}</small></el-card></el-col>
      <el-col :md="6"><el-card class="summary-card"><span>证据白名单拒绝率</span><strong>{{ insights?.whitelist_reject_rate == null ? "未采集" : `${(insights.whitelist_reject_rate * 100).toFixed(1)}%` }}</strong><small>rejected · uncited_evidence（反幻觉）</small></el-card></el-col>
      <el-col :md="6"><el-card class="summary-card"><span>Critic 驳回率</span><strong>{{ insights?.critic_reject_rate == null ? "未采集" : `${(insights.critic_reject_rate * 100).toFixed(1)}%` }}</strong><small>rejected · critic_verdict=rejected</small></el-card></el-col>
      <el-col :md="6"><el-card class="summary-card"><span>洞察生成 p95</span><strong>{{ insights?.latency_ms?.p95 == null ? "未采集" : `${insights.latency_ms.p95} ms` }}</strong><small>{{ insights?.latency_ms?.samples ?? 0 }} 个样本 · token/份 {{ insights?.tokens?.avg_per_insight ?? "—" }} · 成本/份 {{ insights?.cost?.avg_per_insight ?? "—" }}</small></el-card></el-col>
    </el-row>
    <el-row :gutter="16" class="section">
      <el-col :md="12"><el-card><template #header>Critic 裁决与结构化原因</template><el-descriptions :column="1" border><el-descriptions-item label="Critic 放行率">{{ traces?.critic_accept_rate == null ? "未采集" : `${(traces.critic_accept_rate * 100).toFixed(1)}%` }}</el-descriptions-item><el-descriptions-item label="Critic 驳回率">{{ traces?.critic_reject_rate == null ? "未采集" : `${(traces.critic_reject_rate * 100).toFixed(1)}%` }}</el-descriptions-item><el-descriptions-item label="请求返工率">{{ traces?.critic_revision_rate == null ? "未采集" : `${(traces.critic_revision_rate * 100).toFixed(1)}%` }}</el-descriptions-item><el-descriptions-item label="裁决分布"><el-tag v-for="(count, verdict) in traces?.critic_verdicts" :key="String(verdict)" class="quality-tag">{{ verdict }}: {{ count }}</el-tag></el-descriptions-item><el-descriptions-item label="原因汇总"><el-tag v-for="(count, reason) in traces?.reason_counts" :key="String(reason)" type="warning" class="quality-tag">{{ reason }}: {{ count }}</el-tag><span v-if="!Object.keys(traces?.reason_counts ?? {}).length">—</span></el-descriptions-item><el-descriptions-item label="采集时间">{{ latest?.captured_at }}</el-descriptions-item></el-descriptions></el-card></el-col>
      <el-col :md="12"><el-card><template #header>性能基准</template><el-empty v-if="!bench" description="未采集真实 PostgreSQL 基准。运行基准脚本后写入 data/quality/benchmark-latest.json。" /><pre v-else class="trace">{{ JSON.stringify(bench, null, 2) }}</pre></el-card></el-col>
    </el-row>
    <el-card class="section"><template #header>Evaluation Evidence</template><el-table :data="evaluationRuns" size="small"><el-table-column prop="created_at" label="时间" width="190" /><el-table-column prop="subject_type" label="对象" width="140" /><el-table-column prop="subject_id" label="候选/主体" /><el-table-column prop="status" label="运行" width="110" /><el-table-column prop="gate_status" label="门禁" width="110" /><el-table-column prop="commit_sha" label="提交" width="110" /><el-table-column label="原因"><template #default="scope">{{ (scope.row.gate_reasons ?? []).join('; ') || '—' }}</template></el-table-column></el-table></el-card>
    <el-card class="section"><template #header>已保存的版本快照（最近 20 条）</template><el-table :data="report?.history ?? []" size="small"><el-table-column prop="captured_at" label="采集时间" width="220" /><el-table-column prop="commit_sha" label="提交" width="100" /><el-table-column label="Golden"><template #default="scope">{{ scope.row.golden ? `${scope.row.golden.passed}/${scope.row.golden.cases}` : "—" }}</template></el-table-column><el-table-column label="Trace"><template #default="scope">{{ scope.row.traces?.traces ?? 0 }}</template></el-table-column><el-table-column label="p95(ms)"><template #default="scope">{{ scope.row.traces?.graph_latency_ms?.p95 ?? "—" }}</template></el-table-column></el-table></el-card>
  </div>
</template>
<style scoped>.quality-tag { margin: 2px; }.hint { color: var(--el-text-color-secondary); font-size: 13px; margin: 10px 0 0; }.collection-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }.collection-note { color: var(--el-text-color-secondary); font-size: 13px; margin: 12px 0 0; }.muted { color: var(--el-text-color-secondary); font-size: 13px; }</style>
