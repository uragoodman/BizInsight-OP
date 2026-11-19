<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import * as echarts from "echarts";
import "echarts/theme/v5";
import { formatMetricValue } from "@/utils/format";
const props = defineProps<{ data: Array<{ bucket: string; value: number | null }>; unit: string; metric: string }>();
const chartRoot = ref<HTMLDivElement>();
let chart: echarts.ECharts | undefined;
let resizeObserver: ResizeObserver | undefined;

function render() {
  chart?.dispose();
  chart = undefined;
  if (!chartRoot.value) return;
  chart = echarts.init(chartRoot.value, "v5");
  const validData = props.data.filter((item) => item.value != null) as Array<{ bucket: string; value: number }>;
  const isSinglePoint = validData.length === 1;
  chart.setOption({
    tooltip: { trigger: "axis", formatter: (params: Array<{ axisValueLabel?: string; seriesName?: string; data?: unknown }>) => {
      const point = params[0];
      return `${point?.axisValueLabel ?? ""}<br/>${point?.seriesName ?? props.metric}: ${formatMetricValue(Number(point?.data), props.unit)}`;
    } },
    grid: { left: 68, right: 24, top: 42, bottom: 44 },
    xAxis: { type: "category", data: validData.map((item) => item.bucket), axisLabel: { rotate: validData.length > 8 ? 28 : 0 } },
    yAxis: { type: "value", name: props.unit, axisLabel: { formatter: (value: number) => formatMetricValue(value, props.unit, false) } },
    series: [{
      name: props.metric,
      type: isSinglePoint ? "scatter" : "line",
      smooth: !isSinglePoint,
      showSymbol: true,
      symbol: "circle",
      symbolSize: isSinglePoint ? 18 : 8,
      label: isSinglePoint ? { show: true, position: "top", formatter: ({ value }: { value: unknown }) => formatMetricValue(Number(value), props.unit) } : { show: false },
      areaStyle: isSinglePoint ? undefined : { color: "rgba(64,158,255,.16)" },
      data: validData.map((item) => item.value),
      itemStyle: { color: "#409eff" },
    }],
    graphic: isSinglePoint ? [{ type: "text", left: "center", top: 8, style: { text: "当前范围仅 1 个时间点", fill: "#7d899d", fontSize: 12 } }] : [],
  });
}
watch(() => props.data, async () => { await nextTick(); render(); }, { deep: true, flush: "post" });
onMounted(() => {
  render();
  if (chartRoot.value) {
    resizeObserver = new ResizeObserver(() => chart?.resize());
    resizeObserver.observe(chartRoot.value);
  }
});
onBeforeUnmount(() => { resizeObserver?.disconnect(); chart?.dispose(); });
</script>
<template>
  <div v-if="data.length" ref="chartRoot" class="trend-chart" />
  <el-empty v-else description="当前范围没有可展示的趋势数据" :image-size="72" />
</template>

<style scoped>
.trend-chart { width: 100%; height: 330px; }
</style>
