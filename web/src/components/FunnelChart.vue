<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import * as echarts from "echarts";
import "echarts/theme/v5";
import type { FunnelCard, FunnelStage } from "@/types";

const props = defineProps<{
  funnelData: FunnelCard[];
}>();

const chartRoot = ref<HTMLDivElement>();
let chart: echarts.ECharts | undefined;

function render() {
  if (!chartRoot.value) return;
  chart?.dispose();
  chart = echarts.init(chartRoot.value, "v5");

  const firstDay = props.funnelData[0];
  if (!firstDay || !firstDay.stages.length) return;

  // The API returns one funnel per purchase day. The card represents the
  // entire locked scope, so aggregate each stage rather than rendering only
  // the final day's (often one-order) funnel.
  const stages = firstDay.stages.map((stage, index) => ({
    label: stage.label,
    count: props.funnelData.reduce(
      (sum, day) => sum + Number(day.stages[index]?.count ?? 0),
      0,
    ),
  }));
  const totalOrders = stages[0]?.count ?? 0;

  chart.setOption({
    tooltip: {
      trigger: "item",
      formatter: (params: { name?: string; value?: number; data?: { rate?: number } }) =>
        `${params.name}: ${params.value?.toLocaleString()} (${params.data?.rate ?? 0}%)`,
    },
    series: [
      {
        type: "funnel",
        left: "10%",
        right: "10%",
        top: 40,
        bottom: 20,
        minSize: "10%",
        maxSize: "100%",
        sort: "none",
        gap: 4,
        label: {
          show: true,
          position: "inside",
          formatter: (params: { name?: string; value?: number; data?: { rate?: number } }) =>
            `${params.name}\n${params.value?.toLocaleString()} (${params.data?.rate ?? 0}%)`,
          fontSize: 13,
        },
        itemStyle: { borderColor: "#fff", borderWidth: 2 },
        emphasis: { label: { fontSize: 15 } },
        data: stages.map((s: FunnelStage) => ({
          name: s.label,
          value: s.count,
          rate: totalOrders > 0 ? Number((s.count * 100 / totalOrders).toFixed(1)) : 0,
        })),
      },
    ],
  });
}

watch(
  () => props.funnelData,
  async () => {
    await nextTick();
    render();
  },
  { deep: true }
);

// The card is commonly mounted after the completed analysis payload already
// exists. A watcher alone does not run for that initial value, leaving the
// ECharts container blank despite valid funnel data.
onMounted(render);

onBeforeUnmount(() => chart?.dispose());
</script>

<template>
  <div ref="chartRoot" class="funnel-chart" />
</template>

<style scoped>
.funnel-chart { width: 100%; height: 360px; }
</style>
