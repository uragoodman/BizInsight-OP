import { defineConfig } from "@playwright/test";

/**
 * Minimal browser-level config for the SSE disconnect/reconnect acceptance.
 *
 * It intentionally has no global setup/teardown and no webServer: the
 * verifier script starts the real API and points the browser at it.  The test
 * only needs a Chromium binary, which is the system Chrome configured below.
 */
export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  timeout: 180_000,
  expect: { timeout: 15_000 },
  reporter: [["list"]],
  use: {
    browserName: "chromium",
    headless: true,
    launchOptions: {
      executablePath:
        process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ??
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
      args: ["--no-sandbox"],
    },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "off",
  },
});
