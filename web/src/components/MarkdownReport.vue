<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{ content: string }>();

type Block = { type: "heading" | "paragraph" | "quote" | "list"; level?: number; text?: string; items?: string[] };

function inline(text: string) {
  const escaped = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\"/g, "&quot;").replace(/'/g, "&#39;");
  return escaped.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`(.+?)`/g, "<code>$1</code>");
}

function parse(content: string): Block[] {
  const blocks: Block[] = [];
  let list: string[] = [];
  const flushList = () => { if (list.length) blocks.push({ type: "list", items: list }); list = []; };
  for (const raw of content.replace(/\r\n/g, "\n").split("\n")) {
    const line = raw.trim();
    if (!line) { flushList(); continue; }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) { flushList(); blocks.push({ type: "heading", level: heading[1].length, text: inline(heading[2]) }); continue; }
    if (line.startsWith("- ")) { list.push(inline(line.slice(2))); continue; }
    flushList();
    if (line.startsWith("> ")) blocks.push({ type: "quote", text: inline(line.slice(2)) });
    else blocks.push({ type: "paragraph", text: inline(line) });
  }
  flushList();
  return blocks;
}

const blocks = computed(() => parse(props.content));
</script>

<template>
  <article class="markdown-report">
    <template v-for="(block, index) in blocks" :key="index">
      <h1 v-if="block.type === 'heading' && block.level === 1" v-html="block.text" />
      <h2 v-else-if="block.type === 'heading' && block.level === 2" v-html="block.text" />
      <h3 v-else-if="block.type === 'heading'" v-html="block.text" />
      <ul v-else-if="block.type === 'list'"><li v-for="(item, itemIndex) in block.items" :key="itemIndex" v-html="item" /></ul>
      <blockquote v-else-if="block.type === 'quote'" v-html="block.text" />
      <p v-else v-html="block.text" />
    </template>
  </article>
</template>

<style scoped>
.markdown-report { color: var(--el-text-color-primary); font-size: 14px; line-height: 1.75; }
.markdown-report :deep(h1) { margin: 0 0 20px; font-size: 24px; line-height: 1.35; }
.markdown-report :deep(h2) { margin: 28px 0 12px; padding-bottom: 8px; border-bottom: 1px solid var(--el-border-color-lighter); font-size: 18px; }
.markdown-report :deep(h3) { margin: 20px 0 8px; font-size: 16px; }
.markdown-report :deep(p) { margin: 8px 0; }
.markdown-report :deep(ul) { margin: 8px 0; padding-left: 22px; }
.markdown-report :deep(li) { margin: 5px 0; }
.markdown-report :deep(blockquote) { margin: 18px 0; padding: 10px 14px; border-left: 3px solid var(--el-color-primary); background: var(--el-fill-color-light); color: var(--el-text-color-secondary); }
.markdown-report :deep(code) { padding: 1px 4px; border-radius: 3px; background: var(--el-fill-color-light); }
</style>
