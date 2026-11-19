<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import * as Icons from "@element-plus/icons-vue";
import { api } from "@/api";
import { navigation } from "@/router";
import { useAppStore } from "@/stores/app";
import PageHelp from "@/components/PageHelp.vue";

const store = useAppStore(); const router = useRouter(); const healthy = ref<"loading" | "ok" | "error">("loading");
const darkMode = ref(false);
const iconFor = (name: string) => Icons[name as keyof typeof Icons];
function navigate(path: string) { void router.push(path); }
function applyTheme(): void { document.documentElement.classList.toggle("dark", darkMode.value); localStorage.setItem("bizinsight.theme", darkMode.value ? "dark" : "light"); }
function toggleTheme(): void { darkMode.value = !darkMode.value; applyTheme(); }
onMounted(async () => { darkMode.value = localStorage.getItem("bizinsight.theme") === "dark"; applyTheme(); try { healthy.value = (await api.health(store.auth)).status === "ok" ? "ok" : "error"; } catch { healthy.value = "error"; } });
</script>

<template>
  <el-container class="app-shell">
    <el-aside width="216px" class="side"><div class="brand"><span>BI</span><div><strong>BizInsight</strong><small>受控分析工作台</small></div></div>
      <el-menu :default-active="$route.path" class="menu">
        <el-menu-item v-for="item in navigation" :key="item.path" :index="item.path" @click="navigate(item.path)"><el-icon><component :is="iconFor(item.icon)" /></el-icon><span>{{ item.label }}</span><span class="nav-help" @click.stop><PageHelp :text="item.help" /></span></el-menu-item>
        <el-sub-menu v-if="store.account?.permissions.includes('manage_members')" index="/admin-more">
          <template #title><el-icon><component :is="iconFor('MoreFilled')" /></el-icon><span>更多</span></template>
          <el-menu-item index="/abac-policies"><el-icon><component :is="iconFor('Lock')" /></el-icon><span>ABAC 策略</span></el-menu-item>
        </el-sub-menu>
      </el-menu>
      <div class="side-footer"><el-tag :type="healthy === 'ok' ? 'success' : healthy === 'error' ? 'danger' : 'info'" effect="plain">{{ healthy === 'ok' ? '服务正常' : healthy === 'error' ? '服务不可用' : '检查服务…' }}</el-tag><small>{{ store.profile.displayName }}</small></div>
    </el-aside>
    <el-container><el-header class="header"><div><strong>{{ String($route.meta.title ?? navigation.find((item) => item.path === $route.path)?.label ?? 'BizInsight') }}</strong><span>经营数据由服务端校验，Agent 结论附带 Trace。</span></div><div class="header-actions"><el-button text class="theme-toggle" :aria-label="darkMode ? '切换浅色模式' : '切换深色模式'" @click="toggleTheme">{{ darkMode ? '☀️' : '🌙' }} <span>{{ darkMode ? '浅色' : '深色' }}</span></el-button><el-button text @click="router.push('/settings')"><el-icon><Setting /></el-icon>连接设置</el-button></div></el-header>
      <el-main class="main"><router-view /></el-main>
    </el-container>
  </el-container>
</template>
