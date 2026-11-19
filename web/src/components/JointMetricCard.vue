<script setup lang="ts">
import type { JointAnalysisCluster } from "@/types";

defineProps<{
  cluster: JointAnalysisCluster;
}>();

function fmt(v: number | null | undefined, unit: string): string {
  if (v == null || v === undefined) return "-";
  const n = Number(v);
  if (unit === "%") return n.toFixed(1) + "%";
  if (unit === "BRL" || unit === "BRL/order") return "R$ " + n.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (unit === "score") return n.toFixed(2);
  return n.toLocaleString();
}

function deltaColor(v: number | null | undefined): string {
  if (v == null) return "";
  return v >= 0 ? "var(--el-color-danger)" : "var(--el-color-success)";
}

function sign(v: number | null | undefined): string {
  if (v == null || v === undefined) return "";
  return v >= 0 ? "+" : "";
}
</script>

<template>
  <div class="joint-cluster">
    <div class="cluster-header" v-if="cluster.question">
      📊 {{ cluster.question }}
    </div>
    <div class="cluster-meta" v-if="cluster.seller_id || cluster.period">
      <span v-if="cluster.seller_id" class="meta-item">店铺: {{ cluster.seller_id }}</span>
      <span v-if="cluster.period" class="meta-item">期间: {{ cluster.period }}</span>
      <span class="meta-item">粒度: {{ cluster.grain }}</span>
      <span class="meta-item">口径: {{ cluster.allocation_policy }}</span>
    </div>
    <div class="metric-grid">
      <div v-for="m in cluster.metrics" :key="m.metric_name" class="metric-tile">
        <div class="metric-value">{{ fmt(m.value, m.unit || "") }}</div>
        <div class="metric-label">{{ m.display_name || m.metric_name }}</div>
        <div v-if="m.comparison" class="metric-delta"
          :style="{ color: deltaColor(m.comparison.delta_value) }">
          {{ sign(m.comparison.delta_value) }}{{ fmt(m.comparison.delta_value, m.unit || "") }}
          ({{ sign(m.comparison.delta_ratio) }}{{ m.comparison.delta_ratio?.toFixed(1) }}%)
        </div>
      </div>
    </div>
    <div v-if="cluster.assessment" class="cluster-assessment">
      📝 {{ cluster.assessment }}
    </div>
    <div v-if="cluster.sample_size" class="cluster-note">
      样本: {{ cluster.sample_size }} 单
    </div>
    <div v-if="cluster.association_limitation" class="cluster-limitation">
      ⚠️ {{ cluster.association_limitation }}
    </div>
  </div>
</template>

<style scoped>
.joint-cluster { padding: 4px 0; }
.cluster-header { font-weight: 600; font-size: 14px; margin-bottom: 6px; }
.cluster-meta { display: flex; flex-wrap: wrap; gap: 8px; font-size: 12px; color: var(--el-text-color-secondary); margin-bottom: 10px; }
.meta-item { background: var(--el-fill-color-light); padding: 2px 6px; border-radius: 4px; }
.metric-grid { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 8px; }
.metric-tile { flex: 1; min-width: 100px; background: var(--el-fill-color-lighter); border-radius: 8px; padding: 10px 12px; text-align: center; }
.metric-value { font-size: 20px; font-weight: 700; color: var(--el-color-primary); }
.metric-label { font-size: 12px; color: var(--el-text-color-secondary); margin-top: 2px; }
.metric-delta { font-size: 12px; font-weight: 500; margin-top: 2px; }
.cluster-assessment { font-size: 13px; margin-bottom: 4px; }
.cluster-note { font-size: 12px; color: var(--el-text-color-secondary); margin-bottom: 2px; }
.cluster-limitation { font-size: 11px; color: var(--el-text-color-secondary); font-style: italic; }
</style>
