<script setup lang="ts">
import { computed } from "vue";
import { clarificationMessage, clarificationTitle } from "../utils/clarificationCopy";

const props = defineProps<{ detail: Record<string, unknown> | null }>();
const emit = defineEmits<{ confirm: [metricId: string] }>();
const candidates = computed(() => Array.isArray(props.detail?.ambiguous_candidates)
  ? props.detail?.ambiguous_candidates as Array<{ metric_id: string; display_name: string }>
  : []);
const semantic = computed(() => props.detail?.semantic_interpretation as { interpretation?: string; display_name?: string; dimension_label?: string } | undefined);
</script>

<template>
  <el-alert v-if="detail" type="warning" :closable="false" show-icon class="metric-clarification">
    <template #title>{{ clarificationTitle(detail, Boolean(semantic)) }}</template>
    <p v-if="semantic">系统理解：{{ semantic.interpretation || semantic.display_name }}<span v-if="semantic.dimension_label">（维度：{{ semantic.dimension_label }}）</span></p>
    <p>{{ clarificationMessage(detail, "未执行数据查询，请确认要分析的指标。") }}</p>
    <el-space wrap>
      <el-button v-for="candidate in candidates" :key="candidate.metric_id" size="small" @click="emit('confirm', candidate.metric_id)">
        确认：{{ candidate.display_name }}
      </el-button>
    </el-space>
  </el-alert>
</template>
