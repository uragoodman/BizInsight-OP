import { expect, test } from "@playwright/test";

const email = process.env.E2E_PLATFORM_EMAIL ?? "";
const password = process.env.E2E_PLATFORM_PASSWORD ?? "";
const displayName = process.env.E2E_PLATFORM_DISPLAY_NAME ?? "";

function parseEvents(body: string): Array<Record<string, any>> {
  return body.split(/\r?\n\r?\n/).flatMap((block) => {
    const data = block.split(/\r?\n/).find((line) => line.startsWith("data:"))?.slice(5).trim();
    if (!data) return [];
    try { return [JSON.parse(data) as Record<string, any>]; } catch { return []; }
  });
}

function valuesForKey(value: unknown, keyName: string, output: unknown[] = []): unknown[] {
  if (!value || typeof value !== "object") return output;
  if (Array.isArray(value)) {
    for (const item of value) valuesForKey(item, keyName, output);
    return output;
  }
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    if (key === keyName) output.push(child);
    valuesForKey(child, keyName, output);
  }
  return output;
}

test("ambiguous dimensions pause before data access and survive a browser refresh", async ({ page }, testInfo) => {
  if (!email || !password || !displayName) throw new Error("History suite runner did not provide isolated platform E2E credentials");
  await page.goto("/settings");
  await page.locator('input[autocomplete="email"]').fill(email);
  await page.locator('input[autocomplete="current-password"]').fill(password);
  await page.getByRole("button", { name: "登录", exact: true }).click();
  await expect(page.locator(".side-footer").getByText(displayName, { exact: true })).toBeVisible();

  await page.goto("/chat");
  await page.getByRole("button", { name: "+ 新对话", exact: true }).click();
  const question = "请比较订单量在2018年7月和2018年8月的统计贡献，按店铺和商品品类展示。";
  const responsePromise = page.waitForResponse((response) =>
    response.request().method() === "POST" && new URL(response.url()).pathname.endsWith("/chat/stream"),
  );
  await page.locator('textarea[placeholder="输入经营问题，Enter 发送，Shift + Enter 换行"]').fill(question);
  await page.getByRole("button", { name: "发送", exact: true }).click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const events = parseEvents(await response.text());
  const terminal = [...events].reverse().find((event) => ["completed", "suspended", "failed"].includes(event.event));
  expect(terminal?.event).toBe("suspended");
  expect(terminal?.clarification?.reason_code).toBe("ambiguous_dimension");
  expect(terminal?.clarification?.dimension_candidates).toEqual(expect.arrayContaining(["seller", "product_category"]));
  expect(events.filter((event) => event.event === "node_started").map((event) => event.node)).not.toContain("data_query");
  await expect(page.locator(".clarification")).toContainText("需要明确分组维度");
  await expect(page.locator(".clarification")).toContainText("候选维度：");
  await expect(page.locator(".clarification")).toContainText("店铺");
  await expect(page.locator(".clarification")).toContainText("商品品类");

  const storedAuth = await page.evaluate(() => JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}") as { bearerToken?: string });
  expect(storedAuth.bearerToken).toBeTruthy();
  const traceResponse = await page.request.get(`/traces/${encodeURIComponent(String(terminal?.trace_id ?? ""))}`, {
    headers: { Authorization: `Bearer ${storedAuth.bearerToken}` },
  });
  expect(traceResponse.ok()).toBeTruthy();
  const trace = await traceResponse.json() as Record<string, any>;
  const snapshot = trace.state_snapshot ?? {};
  expect(snapshot.trace_id).toBe(terminal?.trace_id);
  expect(snapshot.query_results ?? []).toEqual([]);
  const toolNames = valuesForKey(snapshot.tool_calls ?? [], "tool_name").filter((value): value is string => typeof value === "string");
  expect(toolNames.filter((name) => /^(query_|registry_|breakdown_|compare_)/.test(name))).toEqual([]);

  await page.reload();
  await expect(page.locator(".history-clarification")).toContainText("需要明确分组维度");
  await expect(page.locator(".history-clarification")).toContainText("候选维度：");
  await expect(page.locator(".history-clarification")).toContainText("店铺");
  await expect(page.locator(".history-clarification")).toContainText("商品品类");
  await testInfo.attach("browser-diagnostics.json", {
    body: Buffer.from(JSON.stringify({ test: testInfo.title, traceIds: [terminal?.trace_id] })),
    contentType: "application/json",
  });
});
