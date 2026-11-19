import { defineConfig } from "@playwright/test";
import { existsSync } from "node:fs";

const port = process.env.E2E_FRONTEND_PORT ?? "15300";
const baseURL = process.env.E2E_BASE_URL ?? `http://127.0.0.1:${port}`;
const chrome = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
  ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";

export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: process.env.E2E_SUITE === "expanded"
    ? "expanded-matrix.spec.ts"
    : process.env.E2E_SUITE === "history"
      ? ["ambiguous-dimension-history.spec.ts", "premise-history-resume.spec.ts"]
      : "full-chain.spec.ts",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 15 * 60_000,
  expect: { timeout: 30_000 },
  outputDir: process.env.E2E_ARTIFACT_DIR ?? "./test-results/full-chain",
  reporter: [
    ["list"],
    ["json", { outputFile: process.env.E2E_JSON_REPORT ?? "./test-results/full-chain.json" }],
  ],
  use: {
    baseURL,
    actionTimeout: 30_000,
    navigationTimeout: 45_000,
    browserName: "chromium",
    headless: true,
    launchOptions: {
      ...(existsSync(chrome) ? { executablePath: chrome } : {}),
      args: ["--no-sandbox"],
    },
    // Auth headers and bearer tokens appear in Playwright traces; the suite
    // records redacted workflow events and screenshots instead.
    trace: "off",
    screenshot: "off",
    video: "off",
  },
  webServer: {
    command: `npm run dev -- --host 127.0.0.1 --port ${port} --strictPort`,
    url: baseURL,
    reuseExistingServer: false,
    timeout: 60_000,
    env: {
      VITE_API_PROXY_TARGET: process.env.E2E_API_TARGET ?? "http://127.0.0.1:28081",
      VITE_DEV_PORT: port,
    },
  },
});
