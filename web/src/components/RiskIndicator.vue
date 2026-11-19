<script setup lang="ts">
import type { RiskAssessment } from "@/types";

defineProps<{ risk: RiskAssessment }>();

function color(level: string): string {
  if (level === "high") return "var(--el-color-danger)";
  if (level === "medium") return "var(--el-color-warning)";
  return "var(--el-color-success)";
}
function label(level: string): string {
  if (level === "high") return "高风险";
  if (level === "medium") return "中风险";
  return "低风险";
}
</script>

<template>
  <div class="risk-indicator">
    <div class="risk-header">
      <el-tag :type="color(risk.level)" size="large">{{ label(risk.level) }}</el-tag>
      <span class="risk-score">{{ risk.score }} / 100</span>
    </div>
    <el-progress
      :percentage="risk.score"
      :color="color(risk.level)"
      :stroke-width="24"
      :text-inside="true"
    />
    <div v-if="risk.dimensions?.length" class="risk-dims">
      <div v-for="d in risk.dimensions" :key="d.name" class="risk-dim">
        <span class="dim-name">{{ d.name }}</span>
        <span class="dim-value">{{ d.value }}%</span>
        <span class="dim-weight">权重 {{ (d.weight * 100).toFixed(0) }}%</span>
      </div>
    </div>
    <div v-if="risk.sample_size" class="risk-sample">样本: {{ risk.sample_size }} 单</div>
  </div>
</template>

<style scoped>
.risk-indicator { padding: 4px 0; }
.risk-header { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.risk-score { font-size: 24px; font-weight: 700; }
.risk-dims { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px; }
.risk-dim { flex: 1; min-width: 100px; background: var(--el-fill-color-lighter); border-radius: 6px; padding: 8px 10px; text-align: center; }
.dim-name { font-size: 12px; color: var(--el-text-color-secondary); display: block; }
.dim-value { font-size: 18px; font-weight: 600; display: block; }
.dim-weight { font-size: 11px; color: var(--el-text-color-secondary); }
.risk-sample { font-size: 12px; color: var(--el-text-color-secondary); margin-top: 8px; }
</style>
