<script setup lang="ts">
import type { AnalysisPresentation } from "@/types";
import { ElMessage } from "element-plus";
import { computed, ref } from "vue";
import FunnelChart from "./FunnelChart.vue";
import JointMetricCard from "./JointMetricCard.vue";
import RiskIndicator from "./RiskIndicator.vue";
import TemplateCard from "./TemplateCard.vue";
import MarkdownContent from "./MarkdownContent.vue";
import { unitSuffix } from "@/utils/metricUnits";
import { formatAnalysisDimensions } from "@/utils/semanticInterpretation";

const props = defineProps<{
  presentation: AnalysisPresentation | null;
  answer: string;
  traceId?: string;
}>();

const hasFacts = computed(() => (props.presentation?.facts?.length ?? 0) > 0);
const hasChanges = computed(() => (props.presentation?.changes?.length ?? 0) > 0);
const hasContributions = computed(() => (props.presentation?.contributions?.length ?? 0) > 0);
const hasFunnel = computed(() => (props.presentation?.funnel?.length ?? 0) > 0);
const hasJointClusters = computed(() => (props.presentation?.joint_clusters?.length ?? 0) > 0);
const hasReviewSamples = computed(() => props.presentation?.review_samples != null);
const hasRisk = computed(() => props.presentation?.risk != null);
const hasTemplateAnalyses = computed(() => (props.presentation?.template_analyses?.length ?? 0) > 0);
const hasRecommendations = computed(() => (props.presentation?.recommendations?.length ?? 0) > 0);
const hasHypothesisChain = computed(() => Object.keys(props.presentation?.hypothesis_chains_by_metric ?? {}).some(k => ((props.presentation?.hypothesis_chains_by_metric ?? {})[k]?.length ?? 0) > 0));
const hasLimitations = computed(() => (props.presentation?.limitations?.length ?? 0) > 0);
const hasEvidence = computed(() => (props.presentation?.evidence?.length ?? 0) > 0);
const isCompleted = computed(() => props.presentation?.status === "completed");
const isFailed = computed(() => props.presentation?.status === "failed");
const isClarification = computed(() => props.presentation?.status === "clarification_required");
const isUnsupported = computed(() => props.presentation?.status === "unsupported");
const semanticInterpretation = computed(() => props.presentation?.semantic_interpretation ?? null);
const analysisDimensionText = computed(() => formatAnalysisDimensions(semanticInterpretation.value));
const copyingTrace = ref(false);

function deltaColor(ratio: number | null | undefined): string {
  if (ratio == null) return "";
  return ratio >= 0 ? "var(--el-color-danger)" : "var(--el-color-success)";
}

function formatNumber(v: number | null | undefined): string {
  if (v == null || v === undefined) return "-";
  return Number(v.toFixed(2)).toLocaleString();
}

function formatPct(v: number | null | undefined): string {
  if (v == null || v === undefined) return "";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

function shortEvidenceId(eid: string): string {
  return eid.length > 16 ? eid.slice(0, 16) + "…" : eid;
}

async function copyTraceId(): Promise<void> {
  const traceId = props.traceId?.trim();
  if (!traceId || copyingTrace.value) return;
  try {
    copyingTrace.value = true;
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(traceId);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = traceId;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      if (!document.execCommand("copy")) throw new Error("clipboard unavailable");
      textarea.remove();
    }
    ElMessage.success("Trace 编号已复制");
  } catch {
    ElMessage.warning("复制失败，请手动选择 Trace 编号");
  } finally {
    copyingTrace.value = false;
  }
}

const emit = defineEmits<{
  (e: "viewTrace", traceId: string): void;
}>();
</script>

<template>
  <div class="analysis-result">
    <el-card v-if="semanticInterpretation" shadow="never" class="semantic-card">
      <template #header><span class="card-title">本次分析口径</span></template>
      <div class="semantic-row"><span class="semantic-label">原始表达</span><span>“{{ semanticInterpretation.source_text }}”</span></div>
      <div class="semantic-row"><span class="semantic-label">系统理解</span><span>{{ semanticInterpretation.interpretation || semanticInterpretation.display_name }}</span></div>
      <div class="semantic-row"><span class="semantic-label">指标</span><span>{{ semanticInterpretation.display_name || semanticInterpretation.metric_id }}</span></div>
      <div v-if="analysisDimensionText" class="semantic-row"><span class="semantic-label">分析维度</span><span>{{ analysisDimensionText }}</span></div>
      <el-tag v-if="isUnsupported" type="info" size="small">未执行数据查询</el-tag>
      <el-tag v-else-if="semanticInterpretation.requires_confirmation" type="warning" size="small">等待确认，不会执行查询</el-tag>
      <el-tag v-else-if="isClarification" type="warning" size="small">等待补充信息，不会执行查询</el-tag>
      <el-tag v-else type="success" size="small">已按此口径执行</el-tag>
    </el-card>
    <!-- 状态卡 -->
    <el-alert
      v-if="isFailed"
      title="分析异常"
      :description="answer || '分析过程出现错误。'"
      type="error"
      show-icon
      :closable="false"
    />
    <el-alert
      v-else-if="isClarification"
      title="需要确认分析指标"
      type="warning"
      show-icon
      :closable="false"
    >
      <template #default>
        <MarkdownContent :content="answer" />
      </template>
    </el-alert>
    <el-alert
      v-else-if="isUnsupported"
      title="当前查询暂不支持"
      :description="answer || '当前问题无法在锁定范围内执行。'"
      type="info"
      show-icon
      :closable="false"
    />

    <!-- 已完成的分析结论 -->
    <template v-if="isCompleted || (!isFailed && !isClarification && !isUnsupported)">
      <!-- Text is the primary conclusion; cards below are supporting evidence. -->
      <el-card v-if="answer" shadow="never" class="conclusion-card">
        <template #header><span class="card-title">分析结论</span></template>
        <MarkdownContent class="conclusion-text" :content="answer" />
      </el-card>

      <el-alert
        v-if="presentation?.plan_preferences?.focus_key || presentation?.plan_preferences?.note"
        type="info"
        :closable="false"
        class="limitation-box"
      >
        <template #title>本次人工关注点</template>
        <div v-if="presentation?.plan_preferences?.focus_key">分析角度：{{ presentation.plan_preferences.focus_key }}</div>
        <div v-if="presentation?.plan_preferences?.note">用户备注：{{ presentation.plan_preferences.note }}</div>
      </el-alert>

      <el-descriptions v-if="presentation?.execution_governance?.execution_id" :column="1" size="small" border class="scope-box">
        <el-descriptions-item label="执行治理">白名单模板 · 已审计</el-descriptions-item>
        <el-descriptions-item label="执行耗时">{{ presentation.execution_governance.duration_ms ?? 0 }} ms</el-descriptions-item>
        <el-descriptions-item label="输入指纹">{{ presentation.execution_governance.input_fingerprint?.slice(0, 16) }}…</el-descriptions-item>
        <el-descriptions-item label="模板版本">{{ presentation.execution_governance.template_code_hash?.slice(0, 16) }}…</el-descriptions-item>
      </el-descriptions>

      <!-- Scope 摘要 -->
      <el-descriptions v-if="presentation?.scope" :column="1" size="small" border class="scope-box">
        <el-descriptions-item label="店铺" v-if="presentation.scope.seller_id">{{ presentation.scope.seller_id }}</el-descriptions-item>
        <el-descriptions-item label="分析期间">{{ presentation.scope.start_date }} ~ {{ presentation.scope.end_date }}</el-descriptions-item>
        <el-descriptions-item label="粒度" v-if="presentation.scope.granularity">{{ presentation.scope.granularity }}</el-descriptions-item>
      </el-descriptions>

      <!-- 事实卡 -->
      <el-card v-if="hasFacts" shadow="never" class="card-section">
        <template #header><span class="card-title">📊 事实</span></template>
        <div v-for="fact in presentation!.facts!" :key="fact.metric_name" class="fact-row">
          <div class="fact-label">{{ fact.display_name || fact.metric_name }}</div>
          <div class="fact-value">{{ formatNumber(fact.value) }} {{ fact.unit }}</div>
          <div class="fact-date">{{ fact.date_or_range }}</div>
        </div>
      </el-card>

      <!-- 变化卡 -->
      <el-card v-if="hasChanges" shadow="never" class="card-section">
        <template #header><span class="card-title">📈 变化</span></template>
        <div v-for="chg in presentation!.changes!" :key="chg.metric_name" class="change-card">
          <div class="change-header">{{ chg.display_name || chg.metric_name }}</div>
          <div class="change-grid">
            <div class="change-item">
              <span class="change-label">当前期</span>
              <span class="change-value">{{ chg.current_period }}: {{ formatNumber(chg.current_value) }}{{ unitSuffix(chg.unit) }}</span>
            </div>
            <div class="change-item">
              <span class="change-label">基线期</span>
              <span class="change-value">{{ chg.baseline_period }}: {{ formatNumber(chg.baseline_value) }}{{ unitSuffix(chg.unit) }}</span>
            </div>
            <div class="change-item">
              <span class="change-label">差值</span>
              <span class="change-value" :style="{ color: deltaColor(chg.delta_value) }">{{ formatNumber(chg.delta_value) }}{{ unitSuffix(chg.change_unit) }}</span>
            </div>
            <div class="change-item">
              <span class="change-label">变化率</span>
              <span class="change-value" :style="{ color: deltaColor(chg.delta_ratio) }">{{ chg.delta_ratio == null ? "" : `${chg.delta_ratio >= 0 ? "+" : ""}${chg.delta_ratio.toFixed(2)}${unitSuffix(chg.delta_ratio_unit || "%")}` }}</span>
            </div>
          </div>
        </div>
      </el-card>

      <!-- 贡献卡 -->
      <el-card v-if="hasContributions" shadow="never" class="card-section">
        <template #header><span class="card-title">🔍 维度贡献度</span></template>
        <div v-for="contrib in presentation!.contributions!" :key="contrib.dimension">
          <div class="contrib-dim">
            {{ contrib.metric_name || "指标" }} · {{ contrib.dimension }}
            <span v-if="contrib.metric_definition" class="contrib-method">指标口径（v{{ contrib.definition_version }}，指标单位 {{ unitSuffix(contrib.unit) }}；贡献单位 {{ unitSuffix(contrib.contribution_unit || contrib.unit) }}）：{{ contrib.metric_definition }}</span>
            <span v-if="contrib.method === 'proportional_overlap_v1'" class="contrib-method">按品类分摊的订单贡献（每个订单按涉及品类数均分，分摊值为订单等价量；描述性拆解，非因果结论）</span>
            <span v-else class="contrib-method">区间变化贡献（{{ contrib.method || "additive_change_v1" }}；描述性拆解，非因果结论）</span>
            <span v-if="contrib.baseline_date || contrib.current_date" class="contrib-method">比较日期：{{ contrib.baseline_date || "—" }} → {{ contrib.current_date || "—" }}</span>
          </div>
          <div v-if="contrib.total_delta != null" class="contrib-reconciliation">
            总变化 {{ formatNumber(contrib.total_delta) }}{{ unitSuffix(contrib.contribution_unit || contrib.unit) }}
            <template v-if="contrib.residual_value != null"> · 余项 {{ formatNumber(contrib.residual_value) }}（{{ contrib.residual_type }}，{{ contrib.residual_explainable ? "可解释" : "未解释" }}<template v-if="contrib.residual_tolerance != null">，容差 {{ formatNumber(contrib.residual_tolerance) }}</template>）</template>
          </div>
          <el-table :data="contrib.items" size="small" stripe>
            <el-table-column prop="dimension_value" label="维度值" />
            <el-table-column prop="baseline_value" label="基期值" width="110">
              <template #default="{ row }">{{ row.baseline_value == null ? "—" : `${formatNumber(row.baseline_value)}${unitSuffix(contrib.unit)}` }}</template>
            </el-table-column>
            <el-table-column prop="current_value" label="当前值" width="110">
              <template #default="{ row }">{{ row.current_value == null ? "—" : `${formatNumber(row.current_value)}${unitSuffix(contrib.unit)}` }}</template>
            </el-table-column>
            <el-table-column prop="delta_value" label="变化量" width="120">
              <template #default="{ row }">
                <span :style="{ color: deltaColor(row.delta_value) }">{{ formatNumber(row.delta_value) }}{{ unitSuffix(contrib.contribution_unit || contrib.unit) }}</span>
              </template>
            </el-table-column>
            <el-table-column label="组内／构成效应" min-width="220">
              <template #default="{ row }">
                <span v-if="row.components?.length">
                  <span v-for="(effect, effectIndex) in row.components" :key="effectIndex">
                    <span v-if="effectIndex > 0">；</span>{{ effect.component_type }} {{ formatNumber(effect.value) }}{{ unitSuffix(effect.unit) }}
                  </span>
                </span>
                <span v-else>—</span>
              </template>
            </el-table-column>
            <el-table-column prop="contribution_ratio" label="贡献率" width="120">
              <template #default="{ row }">
                <span :style="{ color: deltaColor(row.contribution_ratio) }">{{ formatPct(row.contribution_ratio) }}</span>
              </template>
            </el-table-column>
          </el-table>
          <div v-if="contrib.residual_note" class="contrib-reconciliation">{{ contrib.residual_note }}</div>
        </div>
      </el-card>

      <!-- 漏斗图 -->
      <el-card v-if="hasFunnel" shadow="never" class="card-section">
        <template #header><span class="card-title">🔻 订单漏斗</span></template>
        <FunnelChart :funnel-data="presentation?.funnel ?? []" />
      </el-card>

      <!-- 联合分析 -->
      <el-card v-if="hasJointClusters" shadow="never" class="card-section">
        <template #header><span class="card-title">📊 履约·评价·销量 联合分析</span></template>
        <JointMetricCard
          v-for="(cluster, i) in (presentation?.joint_clusters ?? [])"
          :key="i"
          :cluster="cluster"
        />
      </el-card>

      <!-- 评论文本样本 -->
      <el-card v-if="hasReviewSamples" shadow="never" class="card-section">
        <template #header><span class="card-title">💬 评论文本</span></template>
        <div v-if="presentation?.review_samples?.summary" class="review-summary">
          {{ presentation.review_samples.summary }}
        </div>
        <div v-if="presentation?.review_samples?.comment_rate != null" class="review-stats">
          含文本率: {{ presentation.review_samples.comment_rate }}%
          <span v-if="presentation?.review_samples?.avg_length">
            · 平均长度: {{ presentation.review_samples.avg_length }} 字符
          </span>
        </div>
        <div v-for="s in (presentation?.review_samples?.samples ?? [])" :key="s.order_id_abbr" class="review-sample">
          <el-tag :type="s.score <= 2 ? 'danger' : 'success'" size="small">{{ s.score }} 分</el-tag>
          <span class="sample-text">{{ s.comment_excerpt }}</span>
          <div class="sample-meta">{{ s.purchase_date }}</div>
        </div>
      </el-card>

      <!-- 推理链 -->
      <template v-if="hasHypothesisChain">
        <template v-for="(chain, metricId) in (presentation?.hypothesis_chains_by_metric ?? {})">
          <el-card v-if="chain && chain.length" :key="metricId" shadow="never" class="card-section">
            <template #header><span class="card-title">🔗 推理链</span><el-tag size="small" class="chain-metric-tag">{{ metricId }}</el-tag></template>
            <el-timeline>
              <el-timeline-item v-for="step in chain" :key="step.step"
                :type="step.verdict === 'supported' ? 'success' : step.verdict === 'refuted' ? 'danger' : 'warning'"
                :hollow="true">
                <div><strong>{{ step.hypothesis }}</strong></div>
                <div class="chain-detail">查询: {{ step.purpose }} | 裁决: {{ step.verdict }}</div>
                <div class="chain-detail" v-if="step.rationale">说明: {{ step.rationale }}</div>
              </el-timeline-item>
            </el-timeline>
            <div class="causal-disclaimer">⚠️ 贡献分析不等同于因果证明。维度贡献表明相关关系，不能独立确立因果关系。</div>
          </el-card>
        </template>
      </template>

      <!-- 建议卡 -->
      <el-card v-if="hasRecommendations" shadow="never" class="card-section">
        <template #header><span class="card-title">💡 建议</span></template>
        <ul class="recom-list">
          <li v-for="(rec, i) in presentation!.recommendations!" :key="i">{{ rec.text }}</li>
        </ul>
      </el-card>

      <!-- 风险评分 -->
      <el-card v-if="hasRisk" shadow="never" class="card-section">
        <template #header>
          <span class="card-title">
            <el-tooltip content="综合差评率、延迟率、运费率三个维度的 0-100 风险分。分数越高风险越大：60+ 高风险（红）、30-59 中风险（黄）、30 以下低风险（绿）。" placement="top">
              ⚠️ 履约-评价风险评分
            </el-tooltip>
          </span>
        </template>
        <RiskIndicator :risk="presentation?.risk ?? { score: 0, level: 'low' }" />
      </el-card>

      <!-- 高级分析模板 -->
      <el-card v-if="hasTemplateAnalyses" shadow="never" class="card-section">
        <template #header><span class="card-title">🔬 高级分析</span></template>
        <TemplateCard
          v-for="(tpl, i) in (presentation?.template_analyses ?? [])"
          :key="i"
          :card="tpl"
        />
      </el-card>

      <!-- 局限 -->
      <el-alert v-if="hasLimitations" :title="presentation!.limitations!.join('；')" type="warning" show-icon :closable="false" class="limitation-box" />

      <!-- 证据摘要 -->
      <el-card v-if="hasEvidence" shadow="never" class="card-section evidence-box">
        <template #header><span class="card-title">📎 证据</span></template>
        <div v-for="ref in presentation!.evidence!" :key="ref.evidence_id" class="evidence-row">
          <div class="evidence-id">{{ shortEvidenceId(ref.evidence_id) }}</div>
          <el-tag size="small" type="info">{{ ref.source_type }}</el-tag>
          <span class="evidence-summary" v-if="ref.summary">{{ ref.summary }}</span>
        </div>
      </el-card>

      <!-- Trace 入口 -->
      <div v-if="traceId" class="trace-link">
        <span class="trace-label">查看本次 Trace：</span>
        <code class="trace-id" role="button" tabindex="0" title="点击查看 Trace" @click="emit('viewTrace', traceId)" @keydown.enter="emit('viewTrace', traceId)">{{ traceId }}</code>
        <el-button text type="primary" size="small" :loading="copyingTrace" @click="copyTraceId">复制</el-button>
      </div>
    </template>
  </div>
</template>

<style scoped>
.analysis-result { width: 100%; }
.semantic-card { margin-bottom: 12px; border-left: 4px solid var(--el-color-warning); }
.semantic-row { display: flex; gap: 10px; padding: 3px 0; line-height: 1.6; }
.semantic-label { min-width: 64px; color: var(--el-text-color-secondary); }
.scope-box { margin-bottom: 12px; }
.card-section { margin-bottom: 12px; }
.conclusion-card { margin-bottom: 12px; border-left: 4px solid var(--el-color-primary); }
.conclusion-text { line-height: 1.8; color: var(--el-text-color-primary); font-size: 14px; }
.card-title { font-weight: 600; font-size: 14px; }
.fact-row { display: flex; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--el-border-color-light); }
.fact-row:last-child { border-bottom: none; }
.fact-label { flex: 1; font-weight: 500; }
.fact-value { margin: 0 12px; font-size: 18px; font-weight: 700; color: var(--el-color-primary); }
.fact-date { color: var(--el-text-color-secondary); font-size: 12px; }
.change-card { padding: 8px 0; border-bottom: 1px solid var(--el-border-color-light); }
.change-card:last-child { border-bottom: none; }
.change-header { font-weight: 500; margin-bottom: 4px; }
.change-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 16px; }
.change-item { display: flex; justify-content: space-between; padding: 2px 0; }
.change-label { color: var(--el-text-color-secondary); font-size: 12px; }
.change-value { font-weight: 500; }
.contrib-dim { font-weight: 500; margin-bottom: 4px; color: var(--el-text-color-secondary); }
.contrib-method, .contrib-reconciliation { display: block; margin-top: 4px; color: var(--el-text-color-secondary); font-size: 12px; font-weight: 400; }
.recom-list { margin: 0; padding-left: 20px; }
.recom-list li { margin-bottom: 4px; }
.limitation-box { margin-bottom: 12px; }
.fallback-text { white-space: pre-wrap; line-height: 1.6; }
.evidence-box { margin-bottom: 12px; }
.evidence-row { display: flex; align-items: center; gap: 8px; padding: 4px 0; }
.evidence-id { font-family: monospace; font-size: 11px; color: var(--el-text-color-secondary); }
.evidence-summary { font-size: 12px; color: var(--el-text-color-regular); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.trace-link { margin-top: 8px; }
.trace-label { margin-right: 4px; }
.trace-id { user-select: text; cursor: pointer; font-family: monospace; font-size: 12px; color: var(--el-color-primary); word-break: break-all; }
.chain-detail { font-size: 12px; color: var(--el-text-color-secondary); margin-top: 2px; }
.causal-disclaimer { font-size: 12px; color: var(--el-text-color-secondary); margin-top: 8px; font-style: italic; }
.chain-metric-tag { margin-left: 8px; }
</style>
