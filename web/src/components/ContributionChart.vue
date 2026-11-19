<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import * as echarts from "echarts";
import "echarts/theme/v5";
import { formatMetricValue } from "@/utils/format";

const props = defineProps<{ data: Array<{ dimension: string; value: number | null }>; metric: string }>();
const chartRoot = ref<HTMLDivElement>();
let chart: echarts.ECharts | undefined;
let resizeObserver: ResizeObserver | undefined;

function render() {
  chart?.dispose();
  if (!chartRoot.value) return;
  chart = echarts.init(chartRoot.value, "v5");
  const data = props.data.filter((item) => item.value != null).slice(0, 10).reverse() as Array<{ dimension: string; value: number }>;
  chart.setOption({
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, formatter: (params: Array<{ name?: string; value?: number }>) => `${params[0]?.name ?? ""}: ${formatMetricValue(Number(params[0]?.value), "", false)}` },
    grid: { left: 120, right: 24, top: 20, bottom: 28 },
    xAxis: { type: "value" },
    yAxis: { type: "category", data: data.map((item) => item.dimension) },
    series: [{ name: props.metric, type: "bar", data: data.map((item) => item.value), itemStyle: { color: "#6b7fd7" } }],
  });
}

watch(() => props.data, async () => { await nextTick(); render(); }, { deep: true, flush: "post" });
onMounted(() => { render(); if (chartRoot.value) { resizeObserver = new ResizeObserver(() => chart?.resize()); resizeObserver.observe(chartRoot.value); } });
onBeforeUnmount(() => { resizeObserver?.disconnect(); chart?.dispose(); });
</script>

<template><div v-if="data.length" ref="chartRoot" class="contribution-chart" /><el-empty v-else description="当前范围没有可展示的贡献数据" :image-size="72" /></template>

<style scoped>.contribution-chart { width: 100%; height: 330px; }</style>
