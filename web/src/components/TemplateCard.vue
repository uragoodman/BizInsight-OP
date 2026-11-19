<script setup lang="ts">
import type { AnalysisTemplateCard } from "@/types";

defineProps<{ card: AnalysisTemplateCard }>();

function fmt(v: number | null | undefined): string {
  if (v == null || v === undefined) return "-";
  return Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 });
}
</script>

<template>
  <div class="template-card">
    <div class="tpl-header">{{ card.display_name || card.template_name }}</div>

    <!-- Cohort: retention matrix -->
    <template v-if="card.template_name === 'cohort_retention'">
      <div class="tpl-intro">客户留存率矩阵（按首次购买月份分组）</div>
      <div class="cohort-matrix">
        <div class="matrix-row header-row">
          <span class="matrix-cell label-cell">获客月</span>
          <span class="matrix-cell label-cell">客户数</span>
          <span v-for="p in ((card.result as any).cohorts?.[0]?.retention ?? [])"
                :key="p.period" class="matrix-cell header-cell">{{ p.period }}</span>
        </div>
        <div v-for="c in (card.result as any).cohorts ?? []" :key="c.cohort_label" class="matrix-row">
          <span class="matrix-cell label-cell">{{ c.cohort_label.slice(0, 7) }}</span>
          <span class="matrix-cell">{{ c.customer_count }}</span>
          <span v-for="r in c.retention" :key="r.period"
                class="matrix-cell rate-cell"
                :style="{ background: `rgba(64,158,255,${Math.max(0.05, (r.rate ?? 0) / 100)})` }">
            {{ (r.rate ?? 0).toFixed(1) }}%
          </span>
        </div>
      </div>
      <div class="tpl-meta" v-if="(card.result as any).total_customers">
        样本: {{ (card.result as any).total_customers }} 客户
      </div>
    </template>

    <!-- RFM: segment distribution -->
    <template v-if="card.template_name === 'rfm_segmentation'">
      <div class="tpl-intro">客户价值分群</div>
      <div class="segment-list">
        <div v-for="s in (card.result as any).segments ?? []" :key="s.segment_name" class="segment-row">
          <span class="segment-name">{{ s.segment_name }}</span>
          <span class="segment-count">{{ s.customer_count }} 人</span>
          <span class="segment-pct">{{ s.pct }}%</span>
          <span class="segment-value">均值 R$ {{ fmt(s.avg_value) }}</span>
        </div>
      </div>
      <div class="tpl-meta" v-if="(card.result as any).total_customers">
        总计: {{ (card.result as any).total_customers }} 客户
      </div>
    </template>

    <!-- Pareto: cumulative contribution -->
    <template v-if="card.template_name === 'pareto_analysis'">
      <div class="tpl-intro">累计贡献占比 — 维度: {{ (card.result as any).dimension }}</div>
      <div class="pareto-table">
        <div class="pareto-row header-row">
          <span class="pareto-cell rank-cell">#</span>
          <span class="pareto-cell name-cell">名称</span>
          <span class="pareto-cell value-cell">值</span>
          <span class="pareto-cell pct-cell">累计%</span>
        </div>
        <div v-for="item in (card.result as any).items ?? []" :key="item.rank" class="pareto-row"
             :class="{ cutoff: (card.result as any).p80_cutoff && item.rank <= (card.result as any).p80_cutoff.entity_count }">
          <span class="pareto-cell rank-cell">{{ item.rank }}</span>
          <span class="pareto-cell name-cell">{{ item.name }}</span>
          <span class="pareto-cell value-cell">{{ fmt(item.value) }}</span>
          <span class="pareto-cell pct-cell">{{ item.cumulative_pct }}%</span>
        </div>
      </div>
      <div v-if="(card.result as any).p80_cutoff" class="p80-note">
        前 {{ (card.result as any).p80_cutoff.entity_count }} 项
        ({{ (card.result as any).p80_cutoff.entity_pct }}%) 覆盖 80% 的总额
      </div>
    </template>

    <!-- Correlation: delay vs review -->
    <template v-if="card.template_name === 'delay_review_correlation'">
      <div class="tpl-intro">延迟送达 vs 差评率</div>
      <div class="corr-grid">
        <div class="corr-card on-time">
          <div class="corr-label">准时订单</div>
          <div class="corr-count">{{ (card.result as any).on_time?.order_count }} 单</div>
          <div class="corr-rate">差评率 {{ (card.result as any).on_time?.bad_review_rate }}%</div>
          <div class="corr-score">均分 {{ (card.result as any).on_time?.avg_score }}</div>
        </div>
        <div class="corr-card delayed">
          <div class="corr-label">延迟订单</div>
          <div class="corr-count">{{ (card.result as any).delayed?.order_count }} 单</div>
          <div class="corr-rate">差评率 {{ (card.result as any).delayed?.bad_review_rate }}%</div>
          <div class="corr-score">均分 {{ (card.result as any).delayed?.avg_score }}</div>
        </div>
      </div>
      <div v-if="(card.result as any).lift != null" class="lift-note">
        Lift = {{ (card.result as any).lift }}x — 延迟订单的差评率是准时的
        {{ (card.result as any).lift >= 2 ? '显著更高' : '略高' }}
      </div>
      <div v-if="(card.result as any).chi_square?.p_value != null" class="chi-note">
        卡方检验 p={{ (card.result as any).chi_square.p_value }}
        ({{ (card.result as any).chi_square.p_value < 0.05 ? '统计显著' : '不显著' }})
      </div>
      <div class="tpl-limitation">⚠️ 相关性不等于因果关系。</div>
    </template>
    <template v-if="!['cohort_retention','rfm_segmentation','pareto_analysis','delay_review_correlation'].includes(card.template_name)">
      <el-scrollbar max-height="320px"><pre class="package-result">{{ JSON.stringify(card.result, null, 2) }}</pre></el-scrollbar>
    </template>
  </div>
</template>

<style scoped>
.template-card { padding: 4px 0; }
.tpl-header { font-weight: 600; font-size: 14px; margin-bottom: 8px; }
.tpl-intro { font-size: 13px; color: var(--el-text-color-secondary); margin-bottom: 8px; }
.tpl-meta { font-size: 12px; color: var(--el-text-color-secondary); margin-top: 8px; }
.tpl-limitation { font-size: 11px; color: var(--el-text-color-secondary); font-style: italic; margin-top: 6px; }

.cohort-matrix { overflow-x: auto; font-size: 12px; }
.matrix-row { display: flex; gap: 2px; margin-bottom: 2px; }
.matrix-cell { min-width: 50px; padding: 4px 6px; text-align: center; white-space: nowrap; }
.label-cell { font-weight: 500; min-width: 70px; text-align: left; }
.header-cell { font-weight: 500; background: var(--el-fill-color-light); }
.rate-cell { border-radius: 3px; }

.segment-list { font-size: 13px; }
.segment-row { display: flex; gap: 12px; padding: 6px 0; border-bottom: 1px solid var(--el-border-color-light); }
.segment-name { flex: 1; font-weight: 500; }
.segment-count { width: 60px; text-align: right; color: var(--el-text-color-secondary); }
.segment-pct { width: 50px; text-align: right; }
.segment-value { width: 80px; text-align: right; color: var(--el-color-primary); }

.pareto-table { font-size: 12px; }
.pareto-row { display: flex; gap: 4px; padding: 3px 0; }
.pareto-row.cutoff { background: rgba(64,158,255,.06); }
.pareto-cell { white-space: nowrap; }
.rank-cell { width: 24px; text-align: right; }
.name-cell { flex: 1; overflow: hidden; text-overflow: ellipsis; }
.value-cell { width: 80px; text-align: right; }
.pct-cell { width: 60px; text-align: right; }
.p80-note { font-size: 12px; color: var(--el-color-primary); margin-top: 6px; font-weight: 500; }

.corr-grid { display: flex; gap: 12px; margin-bottom: 8px; }
.corr-card { flex: 1; padding: 10px; border-radius: 6px; text-align: center; }
.corr-card.on-time { background: rgba(64,158,255,.06); }
.corr-card.delayed { background: rgba(245,108,108,.06); }
.corr-label { font-size: 12px; color: var(--el-text-color-secondary); }
.corr-count { font-size: 18px; font-weight: 700; margin: 4px 0; }
.corr-rate { font-size: 13px; }
.corr-score { font-size: 12px; color: var(--el-text-color-secondary); }
.lift-note { font-size: 13px; font-weight: 500; margin-bottom: 4px; }
.chi-note { font-size: 12px; color: var(--el-text-color-secondary); }
.package-result { white-space: pre-wrap; font-size: 12px; line-height: 1.5; color: var(--el-text-color-regular); margin: 0; }
</style>
