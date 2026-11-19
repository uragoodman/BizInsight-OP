<script setup lang="ts">
import { computed } from "vue";
import DOMPurify from "dompurify";
import MarkdownIt from "markdown-it";

const props = defineProps<{ content: string }>();

const markdown = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
});

const defaultLinkOpen = markdown.renderer.rules.link_open;
markdown.renderer.rules.link_open = (tokens, index, options, env, self) => {
  const token = tokens[index];
  token.attrSet("target", "_blank");
  token.attrSet("rel", "noopener noreferrer nofollow");
  return defaultLinkOpen
    ? defaultLinkOpen(tokens, index, options, env, self)
    : self.renderToken(tokens, index, options);
};

const defaultTableOpen = markdown.renderer.rules.table_open;
markdown.renderer.rules.table_open = (tokens, index, options, env, self) => {
  const table = defaultTableOpen
    ? defaultTableOpen(tokens, index, options, env, self)
    : self.renderToken(tokens, index, options);
  return `<div class="markdown-table-wrap">\n${table}`;
};

const defaultTableClose = markdown.renderer.rules.table_close;
markdown.renderer.rules.table_close = (tokens, index, options, env, self) => {
  const table = defaultTableClose
    ? defaultTableClose(tokens, index, options, env, self)
    : self.renderToken(tokens, index, options);
  return `${table}</div>\n`;
};

const rendered = computed(() => DOMPurify.sanitize(markdown.render(props.content ?? "")));
</script>

<template>
  <article class="markdown-content" v-html="rendered" />
</template>

<style scoped>
.markdown-content { min-width: 0; color: var(--el-text-color-regular); font-size: 15px; line-height: 1.65; overflow-wrap: anywhere; white-space: normal; }
.markdown-content :deep(:first-child) { margin-top: 0; }
.markdown-content :deep(:last-child) { margin-bottom: 0; }
.markdown-content :deep(h1), .markdown-content :deep(h2), .markdown-content :deep(h3), .markdown-content :deep(h4) { margin: 1.15em 0 .55em; color: var(--el-text-color-primary); line-height: 1.4; }
.markdown-content :deep(h1) { font-size: 1.45em; }
.markdown-content :deep(h2) { padding-bottom: .3em; border-bottom: 1px solid #e8edf5; font-size: 1.25em; }
.markdown-content :deep(h3), .markdown-content :deep(h4) { font-size: 1.08em; }
.markdown-content :deep(p) { margin: .15em 0; }
.markdown-content :deep(ul), .markdown-content :deep(ol) { margin: .25em 0; padding-left: 1.6em; }
.markdown-content :deep(li + li) { margin-top: .25em; }
.markdown-content :deep(blockquote) { margin: .8em 0; padding: .3em 1em; border-left: 3px solid #9bb8e8; color: var(--el-text-color-secondary); background: #f7f9fc; }
.markdown-content :deep(pre) { max-width: 100%; overflow-x: auto; padding: 12px 14px; border-radius: 8px; background: #f3f5f8; line-height: 1.55; }
.markdown-content :deep(code) { padding: .1em .35em; border-radius: 4px; background: #f0f3f7; font-size: .9em; }
.markdown-content :deep(pre code) { padding: 0; background: transparent; font-size: .9em; }
.markdown-content :deep(.markdown-table-wrap) { width: 100%; max-width: 100%; overflow-x: auto; }
.markdown-content :deep(table) { display: table; width: max-content; min-width: 100%; border-collapse: collapse; font-size: .92em; }
.markdown-content :deep(th), .markdown-content :deep(td) { padding: 7px 10px; border: 1px solid #dfe5ee; text-align: left; white-space: normal; }
.markdown-content :deep(th) { background: #f5f7fa; }
.markdown-content :deep(a) { color: var(--el-color-primary); text-decoration: underline; text-underline-offset: 2px; }
.markdown-content :deep(hr) { border: 0; border-top: 1px solid #e5eaf1; }
</style>
