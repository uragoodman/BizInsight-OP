import { expect, test, type Page, type TestInfo } from "@playwright/test";

type Account = { email: string; password: string; display_name: string; seller_id?: string };
type SseEvent = Record<string, any>;
type CapturedSse = { url: string; status: number; body: string; ready: boolean; error?: string };
type HttpError = { method: string; url: string; status: number; resource_type: string; test_title: string; classification: string };
type Diagnostics = {
  api5xx: string[];
  consoleErrors: string[];
  pageErrors: string[];
  httpErrors: HttpError[];
  faviconRequests: Array<{ method: string; url: string; resource_type: string }>;
  traceIds: string[];
  planIds: string[];
  clientTurnIds: string[];
  sessionIds: string[];
};

declare global {
  interface Window {
    __fullChainSseResponses?: CapturedSse[];
  }
}

function account(role: "platform" | "seller" | "seller_analyst"): Account {
  const prefix = role === "platform" ? "E2E_PLATFORM" : role === "seller" ? "E2E_SELLER" : "E2E_SELLER_ANALYST";
  const email = process.env[`${prefix}_EMAIL`] ?? "";
  const password = process.env[`${prefix}_PASSWORD`] ?? "";
  const display_name = process.env[`${prefix}_DISPLAY_NAME`] ?? "";
  const seller_id = process.env.E2E_SELLER_ID;
  if (!email || !password || !display_name || (role !== "platform" && !seller_id)) {
    throw new Error(`Missing isolated ${role} E2E account environment`);
  }
  return { email, password, display_name, seller_id };
}

function parseSse(body: string): SseEvent[] {
  const events: SseEvent[] = [];
  for (const block of body.split(/\r?\n\r?\n/)) {
    const data = block.split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("\n");
    if (!data) continue;
    try { events.push(JSON.parse(data) as SseEvent); } catch { /* Ignore non-JSON SSE comments. */ }
  }
  return events;
}

async function sseMark(page: Page): Promise<number> {
  return page.evaluate(() => window.__fullChainSseResponses?.length ?? 0);
}

async function capturedSseAfter(page: Page, mark: number, path: string): Promise<CapturedSse> {
  await page.waitForFunction(({ start, expectedPath }) => {
    const records = window.__fullChainSseResponses ?? [];
    return records.slice(start).some((record) => record.ready && record.url.includes(expectedPath));
  }, { start: mark, expectedPath: path }, { timeout: 12 * 60_000 });
  return page.evaluate(({ start, expectedPath }) => {
    const records = window.__fullChainSseResponses ?? [];
    const record = records.slice(start).find((candidate) => candidate.ready && candidate.url.includes(expectedPath));
    if (!record) throw new Error(`SSE body was not captured for ${expectedPath}`);
    return record;
  }, { start: mark, expectedPath: path });
}

async function installSseCapture(page: Page): Promise<void> {
  await page.addInitScript(() => {
    if (window.__fullChainSseResponses) return;
    window.__fullChainSseResponses = [];
    const originalFetch = window.fetch.bind(window);
    window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const response = await originalFetch(input, init);
      const requestUrl = input instanceof Request ? input.url : String(input);
      const requestMethod = String(init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
      const path = new URL(requestUrl, window.location.href).pathname;
      const isWorkflowSse = requestMethod === "POST" && (
        path.endsWith("/chat/stream") || /\/complex-analysis\/plans\/[^/]+\/confirm$/.test(path)
      );
      if (isWorkflowSse) {
        const record: CapturedSse = { url: path, status: response.status, body: "", ready: false };
        window.__fullChainSseResponses!.push(record);
        void response.clone().text().then((body) => {
          record.body = body;
          record.ready = true;
        }).catch((error: unknown) => {
          record.error = error instanceof Error ? error.name : "capture_failed";
          record.ready = true;
        });
      }
      return response;
    };
  });
}

async function login(page: Page, who: "platform" | "seller" | "seller_analyst"): Promise<void> {
  const identity = account(who);
  await page.goto("/settings");
  await expect(page.getByRole("heading", { name: "账户与权限" })).toBeVisible();
  await page.locator('input[autocomplete="email"]').fill(identity.email);
  await page.locator('input[autocomplete="current-password"]').fill(identity.password);
  await page.getByRole("button", { name: "登录", exact: true }).click();
  await expect(page.locator(".side-footer").getByText(identity.display_name, { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "账户与权限" })).toBeVisible();
}

async function sendChat(page: Page, message: string): Promise<{ responseStatus: number; events: SseEvent[] }> {
  const mark = await sseMark(page);
  const responsePromise = page.waitForResponse((response) =>
    response.url().includes("/chat/stream") && response.request().method() === "POST",
  );
  await page.locator('textarea[placeholder="输入经营问题，Enter 发送，Shift + Enter 换行"]').fill(message);
  await page.getByRole("button", { name: "发送", exact: true }).click();
  const response = await responsePromise;
  await expect(page.locator(".thinking")).toHaveCount(0, { timeout: 12 * 60_000 });
  await expect(page.locator(".message.assistant-message").last()).toBeVisible();
  const captured = await capturedSseAfter(page, mark, "/chat/stream");
  if (captured.error) throw new Error(`Failed to capture real SSE stream: ${captured.error}`);
  return { responseStatus: response.status(), events: parseSse(captured.body) };
}

function completed(events: SseEvent[]): SseEvent {
  const event = [...events].reverse().find((item) => item.event === "completed");
  expect(event, `SSE terminal event missing; events=${events.map((item) => item.event).join(",")}`).toBeTruthy();
  return event!;
}

function terminal(events: SseEvent[]): SseEvent {
  const event = [...events].reverse().find((item) => ["completed", "suspended", "failed"].includes(item.event));
  expect(event, `SSE terminal event missing; events=${events.map((item) => item.event).join(",")}`).toBeTruthy();
  return event!;
}

async function selectAuthorizedSellerAndPreview(page: Page): Promise<void> {
  const sellerId = account("seller").seller_id!;
  await page.goto("/explorer");
  await expect(page.getByRole("heading", { name: "电商数据浏览" })).toBeVisible();
  const sellerSelect = page.locator(".filter-grid .el-select").first();
  await sellerSelect.click();
  await expect(page.locator(".el-select-dropdown__item:visible").first()).toBeVisible();
  await page.locator(".el-select-dropdown__item:visible").first().click();

  const dateInputs = page.locator(".filter-grid .el-date-editor input");
  await dateInputs.nth(0).fill("2018-01-01");
  await dateInputs.nth(0).press("Enter");
  await dateInputs.nth(1).fill("2018-08-31");
  await dateInputs.nth(1).press("Enter");

  await page.getByRole("button", { name: "查询数据", exact: true }).click();
  await expect(page.getByText("指标趋势", { exact: true })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("向当前范围的 Agent 提问", { exact: true })).toBeVisible();
  await expect(page.locator("body")).toContainText(sellerId.slice(0, 8));
}

test.beforeEach(async ({ page }, testInfo) => {
  testInfo.setTimeout(15 * 60_000);
  await installSseCapture(page);
  const diagnostics: Diagnostics = {
    api5xx: [], consoleErrors: [], pageErrors: [], httpErrors: [], faviconRequests: [],
    traceIds: [], planIds: [], clientTurnIds: [], sessionIds: [],
  };
  (page as any).__e2eDiagnostics = diagnostics;
  const safeUrl = (raw: string): URL => {
    const url = new URL(raw);
    for (const key of [...url.searchParams.keys()]) {
      if (/(token|secret|password|authorization|api[_-]?key)/i.test(key)) url.searchParams.delete(key);
    }
    return url;
  };
  const recordHttpError = (entry: HttpError) => {
    if (!diagnostics.httpErrors.some((item) =>
      item.method === entry.method && item.url === entry.url && item.status === entry.status,
    )) diagnostics.httpErrors.push(entry);
  };
  page.on("response", (response) => {
    const status = response.status();
    if (status < 400) return;
    const request = response.request();
    const url = safeUrl(response.url());
    const resourceType = request.resourceType();
    const pathname = url.pathname;
    const classification = status === 403 && pathname.endsWith("/api/quality")
      ? "expected_permission_403"
      : status === 403 && pathname.endsWith("/auth/abac/policies")
        ? "expected_abac_disabled_403"
      : status === 404 && (pathname === "/favicon.ico" || resourceType === "favicon")
        ? "benign_favicon"
        : status === 404
          ? "unclassified_404"
          : "http_error";
    recordHttpError({
      method: request.method(),
      url: `${url.origin}${pathname}${url.search}`,
      status,
      resource_type: resourceType,
      test_title: testInfo.title,
      classification,
    });
    if (status >= 500 && (url.origin === "http://127.0.0.1" || url.origin === "http://localhost")) {
      diagnostics.api5xx.push(`${status} ${url.origin}${pathname}`);
    }
  });
  page.on("console", (message) => {
    if (message.type() === "error") {
      const location = message.location().url;
      diagnostics.consoleErrors.push(`${message.text().slice(0, 500)}${location ? ` [${location}]` : ""}`);
    }
  });
  page.on("pageerror", (error) => diagnostics.pageErrors.push(error.message.slice(0, 500)));
  page.on("request", (request) => {
    const requestUrl = safeUrl(request.url());
    if (requestUrl.pathname === "/favicon.ico") {
      diagnostics.faviconRequests.push({
        method: request.method(),
        url: `${requestUrl.origin}${requestUrl.pathname}`,
        resource_type: request.resourceType(),
      });
    }
    if (request.method() !== "POST" || !requestUrl.pathname.endsWith("/chat/stream")) return;
    try {
      const payload = JSON.parse(request.postData() ?? "{}") as { chat_turn_id?: string; chat_session_id?: string };
      if (payload.chat_turn_id) diagnostics.clientTurnIds.push(payload.chat_turn_id);
      if (payload.chat_session_id) diagnostics.sessionIds.push(payload.chat_session_id);
    } catch { /* Do not retain raw request bodies; IDs are audit metadata only. */ }
  });
  // Chromium's browser-generated favicon request may not surface through the
  // page response event. CDP supplies method, URL, resource type, and status
  // without recording headers, bodies, or query secrets.
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Network.enable");
  const networkRequests = new Map<string, { method: string; url: string }>();
  cdp.on("Network.requestWillBeSent", (event) => {
    const url = safeUrl(event.request.url);
    networkRequests.set(event.requestId, { method: event.request.method, url: `${url.origin}${url.pathname}${url.search}` });
    if (url.pathname === "/favicon.ico" && !diagnostics.faviconRequests.some((item) => item.url === `${url.origin}${url.pathname}`)) {
      diagnostics.faviconRequests.push({
        method: event.request.method,
        url: `${url.origin}${url.pathname}`,
        resource_type: "pending_response",
      });
    }
  });
  cdp.on("Network.responseReceived", (event) => {
    if (event.response.status < 400) return;
    const request = networkRequests.get(event.requestId);
    if (!request) return;
    const url = safeUrl(request.url);
    const resourceType = String(event.type || "unobserved");
    const classification = event.response.status === 403 && url.pathname.endsWith("/api/quality")
      ? "expected_permission_403"
      : event.response.status === 403 && url.pathname.endsWith("/auth/abac/policies")
        ? "expected_abac_disabled_403"
        : event.response.status === 404 && (url.pathname === "/favicon.ico" || resourceType.toLowerCase() === "favicon")
          ? "benign_favicon"
          : event.response.status === 404
            ? "unclassified_404"
            : "http_error";
    const safeUrlValue = `${url.origin}${url.pathname}${url.search}`;
    recordHttpError({
      method: request.method,
      url: safeUrlValue,
      status: event.response.status,
      resource_type: resourceType,
      test_title: testInfo.title,
      classification,
    });
    if (url.pathname === "/favicon.ico") {
      const existing = diagnostics.faviconRequests.find((item) => item.url === `${url.origin}${url.pathname}`);
      if (existing) existing.resource_type = resourceType;
    }
  });
});

test.afterEach(async ({ page }, testInfo) => {
  const diagnostics = (page as any).__e2eDiagnostics as Diagnostics | undefined;
  if (diagnostics) {
    const captures = await page.evaluate(() => window.__fullChainSseResponses ?? []);
    const workflowEvents = captures.flatMap((capture) => parseSse(capture.body));
    diagnostics.traceIds = [...new Set(workflowEvents.map((event) => String(event.trace_id ?? "")).filter(Boolean))];
    diagnostics.planIds = [...new Set(workflowEvents.map((event) => String(event.complex_plan?.plan_id ?? "")).filter(Boolean))];
    await testInfo.attach("browser-diagnostics.json", {
      body: Buffer.from(JSON.stringify({
        test: testInfo.title,
        url: page.url(),
        api5xx: diagnostics.api5xx,
        consoleErrors: diagnostics.consoleErrors,
        pageErrors: diagnostics.pageErrors,
        httpErrors: diagnostics.httpErrors,
        faviconRequests: diagnostics.faviconRequests,
        traceIds: diagnostics.traceIds,
        planIds: diagnostics.planIds,
        clientTurnIds: [...new Set(diagnostics.clientTurnIds)],
        sessionIds: [...new Set(diagnostics.sessionIds)],
      }, null, 2)),
      contentType: "application/json",
    });
    expect(diagnostics.api5xx, `API 5xx: ${diagnostics.api5xx.join("; ")}`).toEqual([]);
    expect(diagnostics.pageErrors, `Unhandled browser errors: ${diagnostics.pageErrors.join("; ")}`).toEqual([]);
  }
  const passwordInput = page.locator('input[type="password"]');
  if (!(await passwordInput.isVisible().catch(() => false))) {
    await testInfo.attach("final-browser-screen.png", {
      body: await page.screenshot({ fullPage: true, animations: "disabled" }),
      contentType: "image/png",
    });
  }
});

test("platform chat: definition, real metric query, and unsupported comparison fail closed", async ({ page }, testInfo) => {
  await login(page, "platform");
  await page.goto("/chat");

  const definition = await sendChat(page, "GMV 是什么？请说明指标口径。");
  expect(definition.responseStatus).toBe(200);
  const definitionResult = completed(definition.events);
  expect(definitionResult.error).not.toBe(true);
  expect(String(definitionResult.answer ?? "")).toMatch(/GMV|成交|商品/);

  await page.getByRole("button", { name: /新对话/ }).click();
  const query = await sendChat(page, "按商品品类查看 2016-10-18 的 GMV。");
  expect(query.responseStatus).toBe(200);
  const queryResult = completed(query.events);
  expect(queryResult.error).not.toBe(true);
  expect(String(queryResult.trace_id ?? "")).not.toBe("");
  expect(String(queryResult.answer ?? "")).toMatch(/GMV|品类|商品|金额/);
  await expect(page.locator(".message-actions").last()).toContainText("Trace");

  await page.getByRole("button", { name: /新对话/ }).click();
  const composite = await sendChat(page, "按商品品类查看 2016-10-18 GMV，并比较支付方式。");
  expect(composite.responseStatus).toBe(200);
  const compositeResult = terminal(composite.events);
  expect(compositeResult.event).toBe("suspended");
  const clarify = compositeResult.clarification ?? {};
  expect(compositeResult.error).not.toBe(true);
  expect(clarify.reason_code ?? clarify.reason ?? compositeResult.contract?.uncertainty_reason)
    .toBe("comparison_dimension_not_supported");
  const completedNodes = composite.events.filter((item) => item.event === "node_completed").map((item) => item.node);
  expect(completedNodes).not.toContain("data_query");

  await testInfo.attach("platform-chat-workflow-events.json", {
    body: Buffer.from(JSON.stringify({
      definition: definition.events.map((item) => item.event),
      metric_query: query.events.map((item) => item.node ?? item.event),
      unsupported_comparison: composite.events.map((item) => item.node ?? item.event),
      clarification_reason: clarify.reason_code ?? clarify.reason,
      query_trace_id: queryResult.trace_id,
    }, null, 2)),
    contentType: "application/json",
  });
});

test("platform chat: authorized two-axis GMV query renders both dimensions", async ({ page }, testInfo) => {
  await login(page, "platform");
  await page.goto("/chat");

  const result = await sendChat(page, "查询 2018-08-20 的 GMV，按商品品类和客户州交叉汇总。");
  expect(result.responseStatus).toBe(200);
  const final = completed(result.events);
  expect(final.error).not.toBe(true);
  expect(String(final.trace_id ?? "")).not.toBe("");
  expect(String(final.answer ?? "").trim().length).toBeGreaterThan(0);
  const nodes = result.events.filter((item) => item.event === "node_completed").map((item) => String(item.node));
  expect(nodes).toContain("data_query");

  const resultCard = page.locator(".message.assistant-message .analysis-result").last();
  await expect(resultCard).toContainText(/商品品类/);
  await expect(resultCard).toContainText(/客户州\/地区/);
  const cardText = await resultCard.innerText();
  expect(cardText).toMatch(/客户州\/地区\s*×\s*商品品类|商品品类\s*×\s*客户州\/地区/);
  await expect(page.locator(".message-actions").last()).toContainText("Trace");

  await testInfo.attach("multi-axis-query-evidence.json", {
    body: Buffer.from(JSON.stringify({
      trace_id: final.trace_id,
      task_type: final.task_type,
      nodes,
      semantic_dimensions: final.presentation?.semantic_interpretation?.dimensions ?? null,
      answer_nonempty: Boolean(String(final.answer ?? "").trim()),
      rendered_both_axes: true,
    }, null, 2)),
    contentType: "application/json",
  });
});

test("platform anomaly question reaches a traced terminal result", async ({ page }, testInfo) => {
  await login(page, "platform");
  await page.goto("/chat");
  const result = await sendChat(page, "为什么 2018-08-20 的 GMV 下降了？请只根据查询到的证据回答；没有足够证据就明确说明。");
  expect(result.responseStatus).toBe(200);
  const final = terminal(result.events);
  // The synthetic fixture shows an increase on this date. A premise check
  // must suspend and ask for confirmation instead of inventing a decline cause.
  expect(final.event).toBe("suspended");
  expect(final.trace_id).toBeTruthy();
  expect(String(final.answer ?? "").trim().length).toBeGreaterThan(0);
  const nodes = result.events.filter((item) => item.event === "node_completed").map((item) => String(item.node));
  expect(nodes).toContain("triage");
  expect(nodes).toContain("premise_check");
  expect(nodes).not.toContain("root_cause");
  expect(String(final.answer)).toMatch(/前提|上升/);
  await testInfo.attach("anomaly-workflow-events.json", {
    body: Buffer.from(JSON.stringify({ trace_id: final.trace_id, task_type: final.task_type, nodes, answer: final.answer }, null, 2)),
    contentType: "application/json",
  });
});

test("platform operational tabs load read only and preserve the configured ABAC-disabled state", async ({ page }) => {
  await login(page, "platform");
  const routes: Array<[string, string]> = [
    ["/metrics", "指标与目标"],
    ["/reports", "周期报告"],
    ["/quality", "项目质量仪表盘"],
    ["/knowledge", "知识库"],
    ["/preferences", "记忆与偏好"],
    ["/action-plans", "经营建议"],
    ["/code-analysis", "数据分析任务"],
    ["/abac-policies", "ABAC 策略"],
    ["/settings", "账户与权限"],
  ];
  for (const [route, heading] of routes) {
    await page.goto(route);
    await expect(page.getByRole("heading", { name: heading, exact: true }), `${route} should render ${heading}`).toBeVisible();
  }
  await page.goto("/abac-policies");
  await expect(page.getByRole("alert").filter({ hasText: /当前服务未启用租户数据库 ABAC 策略管理/ })).toBeVisible();
  await page.goto("/knowledge");
  await expect(page.getByText("文档列表", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "更新索引" })).toBeEnabled();
  // Deliberately do not press the index button: the E2E shares a read-only local Qdrant service.
});

test("seller Explorer reads its granted range and scoped chat returns a trace", async ({ page }, testInfo) => {
  await login(page, "seller");
  await selectAuthorizedSellerAndPreview(page);
  const identity = account("seller");
  const sellerOptions = page.locator(".el-select-dropdown__item:visible");
  await page.locator(".filter-grid .el-select").first().click();
  await expect(sellerOptions.first()).toBeVisible();
  const sellerOptionText = (await sellerOptions.allTextContents()).join("\n");
  expect(sellerOptionText).not.toContain("seed_seller_02");
  await page.keyboard.press("Escape");
  const question = page.locator(".scoped-chat-panel textarea");
  await question.fill("请按当前店铺和日期范围汇总 GMV，给出查询口径和数值。");
  const scopedMark = await sseMark(page);
  const scopedResponse = page.waitForResponse((response) =>
    response.url().includes("/chat/stream") && response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "开始分析", exact: true }).click();
  const response = await scopedResponse;
  expect(response.status()).toBe(200);
  const scopedCapture = await capturedSseAfter(page, scopedMark, "/chat/stream");
  const events = parseSse(scopedCapture.body);
  const final = completed(events);
  expect(final.error).not.toBe(true);
  expect(final.trace_id).toBeTruthy();
  await expect(page.locator(".trace-reference-id")).toBeVisible({ timeout: 12 * 60_000 });
  await expect(page.locator(".analysis-result-card, .analysis-scroll")).toContainText(/GMV|销售|成交|金额/, { timeout: 12 * 60_000 });
  await testInfo.attach("seller-scoped-query.json", {
    body: Buffer.from(JSON.stringify({ seller_prefix: identity.seller_id!.slice(0, 8), trace_id: final.trace_id, task_type: final.task_type, events: events.map((item) => item.node ?? item.event) }, null, 2)),
    contentType: "application/json",
  });
});

test("seller complex analysis requires template choice and executes a confirmed RFM plan", async ({ page }, testInfo) => {
  await login(page, "seller");
  await selectAuthorizedSellerAndPreview(page);
  await page.goto("/chat");

  const retention = await sendChat(page, "客户留存分析");
  const retentionResult = terminal(retention.events);
  expect(retentionResult.event).toBe("suspended");
  expect(retentionResult.clarification?.status).toBe("complex_template_required");
  await expect(page.getByText("选择复杂分析模板", { exact: true })).toBeVisible();
  const selectedTemplateMark = await sseMark(page);
  await page.getByRole("button", { name: "选择：Cohort 留存分析", exact: true }).click();
  await expect(page.locator(".complex-plan").filter({ hasText: "复杂分析计划（等待确认）" })).toBeVisible({ timeout: 60_000 });
  const selectedTemplateDiagnostic = await capturedSseAfter(page, selectedTemplateMark, "/chat/stream");
  const selectedTemplateEvents = parseSse(selectedTemplateDiagnostic.body);
  expect(selectedTemplateEvents.some((item) => item.event === "suspended")).toBeTruthy();
  expect(selectedTemplateEvents.some((item) => item.node === "data_query" && item.event === "node_completed")).toBeFalsy();
  const selectedPlan = selectedTemplateEvents
    .map((item) => item.complex_plan as { plan_id?: string; template_name?: string } | undefined)
    .find((item) => item?.template_name === "cohort_retention" && item.plan_id);
  expect(selectedPlan?.plan_id, "selected cohort plan should be present in the suspended response").toBeTruthy();
  const cancelledSelectedPlan = await page.evaluate(async (planId) => {
    const saved = JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}");
    const response = await fetch(`/complex-analysis/plans/${encodeURIComponent(planId)}/cancel`, {
      method: "POST",
      headers: { Authorization: `Bearer ${saved.bearerToken ?? ""}` },
    });
    return { status: response.status, body: await response.json() };
  }, selectedPlan!.plan_id!);
  expect(cancelledSelectedPlan.status).toBe(200);
  expect(cancelledSelectedPlan.body.cancelled).toBe(true);
  await testInfo.attach("selected-template-cancellation.json", {
    body: Buffer.from(JSON.stringify({
      template_name: selectedPlan!.template_name,
      plan_id: selectedPlan!.plan_id,
      plan_cancelled: cancelledSelectedPlan.body.cancelled,
      cancellation_status: cancelledSelectedPlan.status,
    }, null, 2)),
    contentType: "application/json",
  });
  // The plan was cancelled through the authenticated API above. That API call
  // does not create or update the chat's clarification control, so don't click it here.
  await expect(page.getByRole("alert").filter({ hasText: "需要确认分析口径" })).toHaveCount(0);

  await page.getByRole("button", { name: /新对话/ }).click();
  const planMark = await sseMark(page);
  const planResponsePromise = page.waitForResponse((response) =>
    response.url().includes("/chat/stream") && response.request().method() === "POST",
  );
  await page.locator('textarea[placeholder="输入经营问题，Enter 发送，Shift + Enter 换行"]').fill(
    "按最近购买、购买频次和消费金额三个依据做 RFM 客户分群",
  );
  await page.getByRole("button", { name: "发送", exact: true }).click();
  const planResponse = await planResponsePromise;
  const planCapture = await capturedSseAfter(page, planMark, "/chat/stream");
  const planEvents = parseSse(planCapture.body);
  const planResult = terminal(planEvents);
  expect(planResponse.status()).toBe(200);
  expect(planResult.event).toBe("suspended");
  expect(planResult.clarification?.status).toBe("complex_scope_required");
  const scopeCard = page.locator(".complex-plan").filter({ hasText: "需要补充分析范围" });
  await expect(scopeCard).toBeVisible();

  const scopeSellerSearch = page.waitForResponse((response) =>
    response.url().includes("/data-explorer/sellers?query=") && response.request().method() === "GET",
  );
  await scopeCard.locator(".el-select input").first().fill(account("seller").seller_id!);
  const sellerSearchResponse = await scopeSellerSearch;
  expect(sellerSearchResponse.status()).toBe(200);
  const sellerSearchBody = await sellerSearchResponse.json() as { items: Array<{ seller_id: string }> };
  expect(sellerSearchBody.items.map((item) => item.seller_id)).toEqual([account("seller").seller_id]);
  const sellerOptions = page.getByRole("option");
  await expect(sellerOptions).toHaveCount(1);
  await sellerOptions.first().click();

  const scopeDates = scopeCard.locator(".el-date-editor input");
  await scopeDates.nth(0).fill("2018-01-01");
  await scopeDates.nth(0).press("Enter");
  await scopeDates.nth(1).fill("2018-08-31");
  await scopeDates.nth(1).press("Enter");
  const createPlanResponse = page.waitForResponse((response) =>
    response.url().includes("/complex-analysis/plans") && response.request().method() === "POST",
  );
  await scopeCard.getByRole("button", { name: "校验并生成计划", exact: true }).click();
  const createdPlanResponse = await createPlanResponse;
  expect([200, 201]).toContain(createdPlanResponse.status());
  const createdPlan = await createdPlanResponse.json() as { plan_id: string; template_name?: string };
  expect(createdPlan.plan_id).toBeTruthy();
  await testInfo.attach("seller-rfm-plan-events.json", {
    body: Buffer.from(JSON.stringify({
      scope_required_events: planEvents.map((item) => item.node ?? item.event),
      scope_required_reason: planResult.clarification?.status,
      selected_seller_count: sellerSearchBody.items.length,
      created_template: createdPlan.template_name,
      plan_id: createdPlan.plan_id,
    }, null, 2)),
    contentType: "application/json",
  });
  await expect(page.getByText("复杂分析计划（等待确认）", { exact: true })).toBeVisible();

  const executeMark = await sseMark(page);
  const executePromise = page.waitForResponse((response) =>
    response.url().includes("/complex-analysis/plans/") && response.url().endsWith("/confirm"),
  );
  await page.getByRole("button", { name: "确认执行", exact: true }).click();
  const executionResponse = await executePromise;
  const executeCapture = await capturedSseAfter(page, executeMark, "/complex-analysis/plans/");
  const executionEvents = parseSse(executeCapture.body);
  expect(executionResponse.status()).toBe(200);
  expect(executionEvents.some((item) => item.event === "failed")).toBeFalsy();
  expect(executionEvents.some((item) => item.event === "completed")).toBeTruthy();
  await expect(page.locator(".message-actions").last()).toContainText("Trace", { timeout: 60_000 });
  await expect(page.locator(".message.assistant-message").last()).toBeVisible();
  await testInfo.attach("complex-analysis.json", {
    body: Buffer.from(JSON.stringify({
      ambiguity_candidates: retentionResult.clarification?.candidates?.map((item: any) => item.template_name),
      selected_template: "cohort_retention",
      executed_template: createdPlan.template_name,
      plan_id: createdPlan.plan_id,
      execution_events: executionEvents.map((item) => item.event),
      trace_id: executionEvents.find((item) => item.event === "completed")?.trace_id,
    }, null, 2)),
    contentType: "application/json",
  });
});

test("seller target and preference workflows use real authenticated APIs", async ({ page }) => {
  await login(page, "seller");
  await page.goto("/metrics");
  await page.getByRole("tab", { name: "目标管理" }).click();
  const version = `fullchain-${Date.now()}`;
  await page.getByPlaceholder("例如 v1、2026Q3").fill(version);
  await page.locator(".target-form .el-input-number input").fill("12345");
  const targetResponsePromise = page.waitForResponse((response) =>
    response.url().includes("/analysis/targets/gmv") && response.request().method() === "PUT",
  );
  await page.getByRole("button", { name: "保存目标", exact: true }).click();
  const targetResponse = await targetResponsePromise;
  expect(targetResponse.status()).toBe(200);
  await expect(page.getByText(version, { exact: true })).toBeVisible();
  const row = page.locator("tr").filter({ hasText: version });
  await row.getByRole("button", { name: "删除" }).click();
  await page.getByRole("button", { name: "OK", exact: true }).click();
  await expect(page.getByText(version, { exact: true })).toHaveCount(0);

  await page.goto("/preferences");
  await expect(page.getByRole("heading", { name: "记忆与偏好" })).toBeVisible();
  const personalization = page.getByRole("switch");
  await expect(personalization).toHaveAttribute("aria-checked", /true|false/);
  if ((await personalization.getAttribute("aria-checked")) === "false") {
    const enabledPromise = page.waitForResponse((response) =>
      response.url().includes("/memory/preferences/settings") && response.request().method() === "PUT",
    );
    await personalization.click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await page.getByRole("button", { name: "确认", exact: true }).click();
    expect((await enabledPromise).status()).toBe(200);
    await expect(personalization).toHaveAttribute("aria-checked", "true");
  }
  await page.getByRole("button", { name: "新建会话", exact: true }).click();
  await expect(page.getByText("偏好写入需要一个持久会话", { exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "添加偏好", exact: true }).click();
  await expect(page.locator(".el-dialog").filter({ hasText: "值（JSON 对象）" })).toBeVisible();
  await page.locator(".el-dialog").filter({ hasText: "值（JSON 对象）" }).getByRole("button", { name: "确认保存", exact: true }).click();
  await expect(page.locator(".el-message-box")).toBeVisible();
  const preferenceResponsePromise = page.waitForResponse((response) =>
    response.url().includes("/memory/preferences") && response.request().method() === "POST",
  );
  await page.locator(".el-message-box").getByRole("button", { name: "确认保存", exact: true }).click();
  const preferenceResponse = await preferenceResponsePromise;
  expect(preferenceResponse.status()).toBe(202);
  const createdPreference = page.locator(".preference-item").filter({ hasText: "response_style" });
  await expect(createdPreference).toBeVisible();
  await expect(createdPreference).toContainText(/等待保存|已保存/);
});

test("platform report lifecycle creates, reviews, publishes, and exports an artifact", async ({ page }) => {
  await login(page, "platform");
  await page.goto("/reports");
  const createPromise = page.waitForResponse((response) =>
    response.url().includes("/analysis/reports") && response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "生成报告草稿", exact: true }).click();
  const createResponse = await createPromise;
  expect(createResponse.status()).toBe(200);
  await expect(page.locator(".card-header")).toContainText("draft", { timeout: 90_000 });
  await page.locator(".card-header").getByRole("button", { name: "审核通过" }).click();
  await page.getByRole("button", { name: "提交", exact: true }).click();
  await expect(page.locator(".card-header")).toContainText("approved", { timeout: 30_000 });
  await page.getByRole("button", { name: "发布", exact: true }).click();
  await page.getByRole("button", { name: "OK", exact: true }).click();
  await expect(page.locator(".card-header")).toContainText("published", { timeout: 30_000 });
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出 Markdown", exact: true }).click();
  expect((await downloadPromise).suggestedFilename()).toMatch(/\.md$/);
});

test("seller receives 403 from the platform-only quality API", async ({ page }) => {
  await login(page, "seller_analyst");
  const status = await page.evaluate(async () => {
    const saved = JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}");
    const response = await fetch("/api/quality", {
      headers: { Authorization: `Bearer ${saved.bearerToken ?? ""}` },
    });
    return response.status;
  });
  expect(status).toBe(403);
});

test("seller is redirected away from the platform-only quality route", async ({ page }) => {
  await login(page, "seller_analyst");
  const qualityApiRequests: string[] = [];
  page.on("request", (request) => {
    if (new URL(request.url()).pathname.replace(/\/$/, "").endsWith("/api/quality")) {
      qualityApiRequests.push(request.url());
    }
  });
  await page.goto("/quality");
  await expect(page).toHaveURL(/\/chat$/);
  expect(qualityApiRequests).toEqual([]);
});
