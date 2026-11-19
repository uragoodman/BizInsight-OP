<script setup lang="ts">
import { QuestionFilled } from "@element-plus/icons-vue";

defineProps<{ text: string }>();
</script>

<template>
  <el-tooltip :content="text" placement="right" :show-after="180" :hide-after="0" popper-class="page-help-popper">
    <el-icon class="page-help" role="img" aria-label="页面用途说明"><QuestionFilled /></el-icon>
  </el-tooltip>
</template>

<style scoped>
.page-help {
  margin-left: 6px;
  color: var(--el-color-info);
  font-size: 16px;
  vertical-align: middle;
  cursor: help;
}
</style>
