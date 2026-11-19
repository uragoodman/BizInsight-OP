import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
test("seller user completes the main browser workflows against the running Compose stack", async ({ page }) => {
  test.setTimeout(12 * 60_000);
  const account = JSON.parse(readFileSync(path.join(here, ".e2e-account.json"), "utf8")) as {
    email: string;
    password: string;
    seller_id: string;
  };
  const apiFailures: string[] = [];
  page.on("response", (response) => {
    if (response.status() >= 500 && new URL(response.url()).origin === new URL(page.url()).origin) {
      apiFailures.push(`${response.status()} ${response.url()}`);
    }
  });

  await page.goto("/settings");
  await page.locator('input[autocomplete="email"]').fill(account.email);
  await page.locator('input[autocomplete="current-password"]').fill(account.password);
  await page.getByRole("button", { name: "登录", exact: true }).click();
  await expect(page.locator(".side-footer").getByText("Playwright E2E", { exact: true })).toBeVisible();

  // Explorer: authorized seller lookup -> real PostgreSQL preview -> locked chat scope.
  await page.goto("/explorer");
  const sellerSelect = page.locator(".filter-grid .el-select").first();
  await sellerSelect.click();
  await page.locator(".el-select-dropdown__item:visible").first().click();
  await page.getByRole("button", { name: "查询数据", exact: true }).click();
  await expect(page.getByText("指标趋势", { exact: true })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("向当前范围的 Agent 提问", { exact: true })).toBeVisible();

  // Inventory import is deliberately scoped to a unique, synthetic observation.
  const batchId = `pw-${Date.now()}`;
  const productId = `pw-product-${Date.now()}`;
  const batch = {
    source_system: "playwright-e2e",
    source_batch_id: batchId,
    rows: [{
      seller_id: account.seller_id,
      warehouse_id: "pw-warehouse",
      product_id: productId,
      sku: "PW-E2E",
      snapshot_at: "2018-08-31T23:59:59+00:00",
      snapshot_semantics: "daily_close",
      on_hand_quantity: 17,
      reserved_quantity: 4,
      available_quantity: 13,
      quantity_unit: "unit",
    }],
  };
  await page.getByLabel("选择库存快照 JSON 批次").setInputFiles({
    name: "inventory-e2e.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(batch)),
  });
  await page.getByRole("button", { name: "校验并追加库存批次" }).click();
  const inventoryResponsePromise = page.waitForResponse((response) => response.url().includes("/inventory-snapshots/ingest"));
  await page.getByRole("button", { name: "确认追加" }).click();
  const inventoryResponse = await inventoryResponsePromise;
  expect(inventoryResponse.ok(), await inventoryResponse.text()).toBeTruthy();
  await expect(page.getByText(batchId, { exact: true })).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "查询库存快照" }).click();
  await expect(page.getByText(productId, { exact: true })).toBeVisible({ timeout: 30_000 });

  // Scoped chat uses the exact scope that was previewed above, via the real SSE endpoint.
  const scopedQuestion = page.locator(".scoped-chat-panel textarea");
  await scopedQuestion.fill("GMV 合计是多少？请结合当前范围给出结果。");
  await page.getByRole("button", { name: "开始分析", exact: true }).click();
  await expect(page.locator(".trace-reference-id")).toBeVisible({ timeout: 180_000 });
  await expect(page.locator(".analysis-result-card, .analysis-scroll")).toContainText(/.+/, { timeout: 180_000 });

  // Reports: generate from the live data, review, publish, then verify the export path.
  await page.goto("/reports");
  await page.getByRole("button", { name: "生成报告草稿" }).click();
  await expect(page.locator(".card-header")).toContainText("draft", { timeout: 60_000 });
  await page.locator(".card-header").getByRole("button", { name: "审核通过" }).click();
  await page.getByRole("button", { name: "提交" }).click();
  await expect(page.locator(".card-header")).toContainText("approved", { timeout: 30_000 });
  await page.getByRole("button", { name: "发布", exact: true }).click();
  await page.getByRole("button", { name: "OK", exact: true }).click();
  await expect(page.locator(".card-header")).toContainText("published", { timeout: 30_000 });
  const exportDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出 Markdown" }).click();
  expect((await exportDownload).suggestedFilename()).toMatch(/\.md$/);

  // Metrics: create a uniquely versioned target and remove it from the same UI.
  await page.goto("/metrics");
  await page.getByRole("tab", { name: "目标管理" }).click();
  const version = `pw-${Date.now()}`;
  await page.getByPlaceholder("例如 v1、2026Q3").fill(version);
  await page.locator(".target-form .el-input-number input").fill("12345");
  await page.getByRole("button", { name: "保存目标" }).click();
  await expect(page.getByText(version, { exact: true })).toBeVisible({ timeout: 30_000 });
  const targetRow = page.locator("tr").filter({ hasText: version });
  await targetRow.getByRole("button", { name: "删除" }).click();
  await page.getByRole("button", { name: "OK", exact: true }).click();
  await expect(page.getByText(version, { exact: true })).toHaveCount(0);

  // Knowledge page reads the live Qdrant-backed inventory and runs the real index task.
  await page.goto("/knowledge");
  await expect(page.getByText("文档列表", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "更新索引" }).click();
  await expect(page.getByText(/索引完成/)).toBeVisible({ timeout: 180_000 });

  // Remaining large operational views must load with authenticated state and actual backend data.
  await page.goto("/quality");
  await expect(page.getByText("项目质量仪表盘", { exact: true })).toBeVisible();
  await expect(page.getByText("采集状态与处理建议", { exact: true })).toBeVisible({ timeout: 30_000 });
  await page.goto("/action-plans");
  await expect(page.getByRole("heading", { name: "经营建议", exact: true })).toBeVisible();

  // Code Agent: upload -> plan -> explicit confirmation -> isolated worker -> result -> cleanup.
  await page.goto("/code-analysis");
  await page.locator('input[type="file"]').first().setInputFiles({
    name: "playwright-sales.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("month,revenue\n2025-01,100\n2025-02,125\n2025-03,160\n"),
  });
  await expect(page.getByText("playwright-sales.csv", { exact: true })).toBeVisible({ timeout: 30_000 });
  const codeQuestion = "计算 revenue 的总和，并概括月度变化。";
  await page.locator(".create textarea").fill(codeQuestion);
  await page.getByRole("button", { name: "生成分析计划" }).click();
  await expect(page.getByRole("button", { name: "确认执行" })).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "确认执行" }).click();
  await expect(page.locator(".conclusion-text")).toBeVisible({ timeout: 8 * 60_000 });
  await expect(page.locator(".conclusion-text")).toContainText(/.+/);
  await page.getByRole("button", { name: "删除任务" }).click();
  await page.getByRole("button", { name: "删除", exact: true }).last().click();
  await expect(page.locator(".tasks .el-menu-item").filter({ hasText: codeQuestion })).toHaveCount(0, { timeout: 30_000 });

  expect(apiFailures, `Unexpected API 5xx responses: ${apiFailures.join("; ")}`).toEqual([]);
});
