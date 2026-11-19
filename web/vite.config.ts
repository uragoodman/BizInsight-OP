// ============================================================
// Vite 构建/开发服务器配置
// 作用：配置 Vue 插件、@ 别名、本地开发服务器端口与后端 API 代理。
// 说明：开发时前端跑在 15299 端口，/chat、/auth、/api 等前缀代理到
//       FastAPI（127.0.0.1:18080）；生产环境由 FastAPI 直接托管构建产物。
// ============================================================
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// 页面导航和 API 使用了相同的前缀（例如 /code-analysis）。
// 浏览器刷新页面时必须留在 Vite，让 SPA 回退到开发版 index.html；
// 只有非 HTML 请求才转发到 FastAPI。
const apiProxy = {
  target: process.env.VITE_API_PROXY_TARGET ?? "http://127.0.0.1:18080",
  bypass(req: { headers: { accept?: string } }) {
    if (req.headers.accept?.includes("text/html")) return "/index.html";
  },
};

export default defineConfig({
  plugins: [vue()],
  test: {
    // Python 测试运行可能在 web/ 下留下 Windows 无法访问的缓存目录，
    // 它不是 Vitest 源码目录，因此从测试发现中排除。
    // Python test runs can leave this Windows-inaccessible cache under web/.
    // It is never a Vitest source directory, so exclude it from discovery.
    exclude: ["**/node_modules/**", "**/dist/**", "**/.pytest_cache/**", "**/tests/e2e/**"]
  },
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  server: {
    // 固定本地地址，但端口选在 Windows 常见保留段 4710-6160 之外；可用 VITE_DEV_PORT 覆盖。
    // Keep a stable local URL, but use a port outside Windows' commonly
    // reserved 4710-6160 range. Override with VITE_DEV_PORT if needed.
    host: "127.0.0.1",
    port: Number(process.env.VITE_DEV_PORT ?? 15299),
    strictPort: true,
    proxy: {
      // 本地开发时的后端 API 代理；生产环境仍由 FastAPI 自身托管。
      // Local development API. Production remains served by FastAPI itself.
      "/chat": apiProxy,
      "/analysis": apiProxy,
      "/data-explorer": apiProxy,
      "/knowledge": apiProxy,
      "/traces": apiProxy,
      "/health": apiProxy,
      "/metrics": apiProxy,
      "/auth": apiProxy,
      "/memory": apiProxy,
      "/memories": apiProxy,
      "/complex-analysis": apiProxy,
      "/code-analysis": apiProxy,
      "/api": apiProxy
    }
  }
} as any);
