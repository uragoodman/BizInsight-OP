// ============================================================
// 前端应用入口
// 作用：创建 Vue 应用，安装 Pinia 状态管理、Element Plus 与路由，
//       并在路由跳转前做「登录守卫」——未登录只允许停留在连接与权限页。
// ============================================================
import { createApp } from "vue";
import { createPinia } from "pinia";
import ElementPlus from "element-plus";
import { ElMessage } from "element-plus";
import "element-plus/dist/index.css";
import "./styles.css";
import App from "./App.vue";
import router from "./router";
import { useAppStore } from "./stores/app";
import { AUTH_EXPIRED_EVENT } from "./api";

const app = createApp(App);
const pinia = createPinia();
app.use(pinia);
const initialStore = useAppStore(pinia);
window.addEventListener(AUTH_EXPIRED_EVENT, () => {
  const store = useAppStore(pinia);
  const wasAuthenticated = Boolean(store.bearerToken);
  store.clearAuth();
  if (wasAuthenticated && router.currentRoute.value.path !== "/settings") {
    ElMessage.warning("登录会话已过期，请重新登录");
    void router.push({ path: "/settings", query: { reason: "session-expired" } });
  }
});
if (initialStore.bearerToken && window.location.pathname !== "/settings") {
  void initialStore.refreshAccount().catch(() => undefined);
}
// 登录守卫：除 /settings 外，未持有 bearerToken 时强制跳转到设置页完成登录/初始化。
router.beforeEach((to) => {
  const store = useAppStore(pinia);
  if (to.path !== "/settings" && !store.ensureSessionValid()) return "/settings";
  const requiredPermission = to.meta.requiredPermission;
  if (requiredPermission && !store.account?.permissions.includes(requiredPermission)) return "/chat";
  if (to.meta.adminOnly && !store.account?.permissions.includes("manage_members")) return "/chat";
  return true;
});
app.use(router).use(ElementPlus).mount("#app");
