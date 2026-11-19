<script setup lang="ts">
import { computed } from "vue";
import type { ExplorerQuery } from "@/types";

const props = defineProps<{ scope: ExplorerQuery | null }>();
defineEmits<{ edit: []; clear: [] }>();
const metricLabel: Record<string, string> = { gmv: "GMV", order_count: "订单量", avg_order_value: "客单价", customer_count: "客户数", freight_value: "运费" };
const granularityLabel: Record<string, string> = { day: "日", week: "周", month: "月" };
const hasScope = computed(() => Boolean(props.scope));
</script>

<template>
  <section class="scope-tag" :class="{ 'is-default': !hasScope }" aria-label="当前分析范围">
    <div class="scope-icon">{{ hasScope ? "✓" : "i" }}</div>
    <div class="scope-body">
      <div class="scope-heading"><strong>{{ hasScope ? "本会话分析范围" : "尚未设置会话范围" }}</strong><span>{{ hasScope ? "店铺与指标由服务端校验；日期作为后续提问的默认范围" : "选择范围后，会应用于本会话后续提问" }}</span></div>
      <div class="scope-facts">
        <span><small>店铺</small>{{ props.scope?.seller_id || "未指定" }}</span>
        <span><small>时间</small>{{ props.scope ? `${props.scope.start_date} 至 ${props.scope.end_date}` : "由每轮问题确定" }}</span>
        <span><small>指标</small>{{ props.scope ? (metricLabel[props.scope.metric_name] || props.scope.metric_name) : "由每轮问题确定" }}</span>
        <span><small>粒度</small>{{ props.scope ? (granularityLabel[props.scope.granularity] || props.scope.granularity) : "由每轮问题确定" }}</span>
      </div>
      <div class="scope-actions"><el-button text type="primary" @click="$emit('edit')">{{ hasScope ? "修改范围" : "设置范围" }}</el-button><el-button v-if="hasScope" text type="danger" @click="$emit('clear')">清除范围</el-button></div>
    </div>
  </section>
</template>

<style scoped>
.scope-tag { display: flex; gap: 12px; margin: 12px 30px 0; padding: 13px 15px; border: 1px solid #b7dfc5; border-radius: 10px; background: #f0f9f2; color: #245b35; }
.scope-tag.is-default { border-color: #f0d58a; background: #fff9e8; color: #795b13; }
.scope-icon { flex: 0 0 24px; width: 24px; height: 24px; display: grid; place-items: center; margin-top: 2px; border-radius: 50%; background: #2e9b57; color: #fff; font-size: 13px; font-weight: 700; }
.scope-tag.is-default .scope-icon { background: #d99a16; }
.scope-body { min-width: 0; flex: 1; }.scope-heading { display: flex; align-items: baseline; gap: 10px; }.scope-heading strong { color: inherit; font-size: 13px; }.scope-heading span { color: #687b6d; font-size: 11px; }.scope-tag.is-default .scope-heading span { color: #94772f; }
.scope-facts { display: flex; flex-wrap: wrap; gap: 6px 18px; margin-top: 7px; }.scope-facts span { color: inherit; font-size: 12px; }.scope-facts small { margin-right: 5px; color: #687b6d; }.scope-tag.is-default .scope-facts small { color: #94772f; }.scope-actions { display: flex; gap: 4px; margin-top: 5px; }
@media (max-width: 900px) { .scope-tag { margin: 10px 18px 0; }.scope-heading { display: block; }.scope-heading span { display: block; margin-top: 2px; } }
</style>
