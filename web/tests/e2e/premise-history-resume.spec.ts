import { expect, test, type Page, type Request } from "@playwright/test";

type SseEvent = Record<string, any>;
type TurnsPayload = { items: Array<Record<string, any>> };
type CapturedStream = { status: number; path: string; body: string; ready: boolean };

declare global {
  interface Window { __historySse?: CapturedStream[] }
}

const email = process.env.E2E_SELLER_EMAIL ?? "";
const password = process.env.E2E_SELLER_PASSWORD ?? "";
const displayName = process.env.E2E_SELLER_DISPLAY_NAME ?? "";
const sellerId = process.env.E2E_SELLER_ID ?? "";
const platformEmail = process.env.E2E_PLATFORM_EMAIL ?? "";
const platformPassword = process.env.E2E_PLATFORM_PASSWORD ?? "";
const platformDisplayName = process.env.E2E_PLATFORM_DISPLAY_NAME ?? "";

function requireAccount(): void {
  if (!email || !password || !displayName || !sellerId) {
    throw new Error("History suite runner did not provide isolated seller E2E credentials");
  }
}

function parseSse(body: string): SseEvent[] {
  return body.split(/\r?\n\r?\n/).flatMap((block) => {
    const data = block.split(/\r?\n/).find((line) => line.startsWith("data:"))?.slice(5).trim();
    if (!data) return [];
    try { return [JSON.parse(data) as SseEvent]; } catch { return []; }
  });
}

function streamRequest(request: Request): boolean {
  return request.method() === "POST" && new URL(request.url()).pathname.endsWith("/chat/stream");
}

async function installSseCapture(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const host = window as Window & { __historySse?: CapturedStream[] };
    host.__historySse = [];
    const originalFetch = window.fetch.bind(window);
    window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const response = await originalFetch(input, init);
      const requestUrl = input instanceof Request ? input.url : String(input);
      const method = String(init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
      const path = new URL(requestUrl, window.location.href).pathname;
      if (method === "POST" && path.endsWith("/chat/stream")) {
        const record: CapturedStream = { status: response.status, path, body: "", ready: false };
        host.__historySse!.push(record);
        void response.clone().text().then((body) => {
          record.body = body;
          record.ready = true;
        }).catch(() => { record.ready = true; });
      }
      return response;
    };
  });
}

async function streamMark(page: Page): Promise<number> {
  return page.evaluate(() => (window as Window & { __historySse?: CapturedStream[] }).__historySse?.length ?? 0);
}

async function capturedStream(page: Page, start: number): Promise<CapturedStream> {
  await page.waitForFunction((from) => {
    const records = (window as Window & { __historySse?: CapturedStream[] }).__historySse ?? [];
    return records.slice(from).some((record) => record.ready);
  }, start, { timeout: 12 * 60_000 });
  return page.evaluate((from) => {
    const records = (window as Window & { __historySse?: CapturedStream[] }).__historySse ?? [];
    const record = records.slice(from).find((item) => item.ready);
    if (!record) throw new Error("No completed SSE capture found");
    return record;
  }, start);
}

async function readTurns(page: Page, sessionId: string): Promise<TurnsPayload> {
  const token = await page.evaluate(() => JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}").bearerToken as string);
  expect(token).toBeTruthy();
  const response = await page.request.get(`/chat/sessions/${encodeURIComponent(sessionId)}/turns?limit=20`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  expect(response.ok()).toBeTruthy();
  return await response.json() as TurnsPayload;
}

test("premise pause survives reload and resumes the same durable turn and execution", async ({ page }, testInfo) => {
  requireAccount();
  await installSseCapture(page);
  const http5xx: string[] = [];
  page.on("response", (response) => {
    if (response.status() >= 500) http5xx.push(`${response.status()} ${response.url()}`);
  });

  await page.goto("/settings");
  await page.locator('input[autocomplete="email"]').fill(email);
  await page.locator('input[autocomplete="current-password"]').fill(password);
  await page.getByRole("button", { name: "登录", exact: true }).click();
  await expect(page.locator(".side-footer").getByText(displayName, { exact: true })).toBeVisible();

  await page.goto("/chat");
  await page.getByRole("button", { name: "+ 新对话", exact: true }).click();
  await page.getByRole("button", { name: "设置范围", exact: true }).click();
  const sellerSearch = page.getByRole("combobox", { name: "店铺" });
  await sellerSearch.fill(sellerId);
  await page.getByRole("option").filter({ hasText: sellerId.slice(0, 8) }).click();
  await page.getByRole("button", { name: "应用到本会话", exact: true }).click();
  await expect(page.locator(".scope-tag")).toContainText(sellerId);

  const originalQuestion = "为什么 2018-08-21 的 GMV 下降了？请只根据查询证据回答；没有足够证据就明确说明。";
  const initialRequestPromise = page.waitForRequest(streamRequest);
  const initialStreamMark = await streamMark(page);
  await page.locator('textarea[placeholder="输入经营问题，Enter 发送，Shift + Enter 换行"]').fill(originalQuestion);
  await page.getByRole("button", { name: "发送", exact: true }).click();
  const initialRequest = await initialRequestPromise;
  const initialPayload = initialRequest.postDataJSON() as Record<string, any>;
  const initialStream = await capturedStream(page, initialStreamMark);
  expect(initialStream.status).toBe(200);
  const initialEvents = parseSse(initialStream.body);
  const suspended = [...initialEvents].reverse().find((event) => event.event === "suspended");
  expect(suspended, JSON.stringify(initialEvents.map((event) => event.event))).toBeTruthy();
  expect(suspended!.interaction_reason_code).toBe("premise_direction_mismatch");
  expect(suspended!.interaction_state).toBe("awaiting_human_input");
  expect(suspended!.trace_id).toBeTruthy();

  const beforeReload = await readTurns(page, String(initialPayload.chat_session_id));
  expect(beforeReload.items).toHaveLength(1);
  const pausedTurn = beforeReload.items[0];
  expect(pausedTurn.status).toMatch(/^(running|suspended)$/);
  expect(pausedTurn.interaction_state).toBe("awaiting_human_input");
  expect(pausedTurn.interaction_reason_code).toBe("premise_direction_mismatch");
  expect(pausedTurn.client_turn_id).toBe(initialPayload.chat_turn_id);
  expect(pausedTurn.thread_id).toBeTruthy();
  expect(pausedTurn.scope_fingerprint).toBeTruthy();

  await page.reload();
  await expect(page.locator(".history-clarification")).toBeVisible();
  await expect(page.locator(".history-clarification")).toContainText(/前提|方向|实际/);

  const resumeRequestPromise = page.waitForRequest(streamRequest);
  const resumeStreamMark = await streamMark(page);
  await page.locator('textarea[placeholder="输入经营问题，Enter 发送，Shift + Enter 换行"]').fill("按实际数据纠正方向");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  const resumeRequest = await resumeRequestPromise;
  const resumePayload = resumeRequest.postDataJSON() as Record<string, any>;
  const resumeStream = await capturedStream(page, resumeStreamMark);
  expect(resumeStream.status).toBe(200);
  expect(resumePayload.chat_session_id).toBe(initialPayload.chat_session_id);
  expect(resumePayload.chat_turn_id).toBe(pausedTurn.client_turn_id);
  expect(resumePayload.thread_id).toBe(pausedTurn.thread_id);
  const resumedEvents = parseSse(resumeStream.body);
  const completed = [...resumedEvents].reverse().find((event) => event.event === "completed");
  expect(completed, JSON.stringify(resumedEvents.map((event) => event.event))).toBeTruthy();
  expect(completed!.trace_id).toBeTruthy();
  expect(String(completed!.answer ?? "")).toMatch(/上升|没有下降|不支持下降/);
  expect(String(completed!.answer ?? "")).not.toMatch(/是否继续|请确认是否/);

  const afterResume = await readTurns(page, String(initialPayload.chat_session_id));
  expect(afterResume.items).toHaveLength(1);
  expect(afterResume.items[0].id).toBe(pausedTurn.id);
  expect(afterResume.items[0].status).toBe("completed");
  expect(afterResume.items[0].interaction_state).not.toBe("awaiting_human_input");
  await expect(page.locator(".thinking")).toHaveCount(0);
  expect(http5xx).toEqual([]);

  await testInfo.attach("history-resume-evidence.json", {
    body: Buffer.from(JSON.stringify({
      test: testInfo.title,
      session_id: initialPayload.chat_session_id,
      turn_id: pausedTurn.id,
      client_turn_id: pausedTurn.client_turn_id,
      thread_id: pausedTurn.thread_id,
      initial_trace_id: suspended!.trace_id,
      resumed_trace_id: completed!.trace_id,
      terminal_status: afterResume.items[0].status,
    }, null, 2)),
    contentType: "application/json",
  });
});

test("historical metric candidates create a fresh authorized turn after reload", async ({ page }, testInfo) => {
  if (!platformEmail || !platformPassword || !platformDisplayName) {
    throw new Error("History suite runner did not provide isolated platform E2E credentials");
  }
  await installSseCapture(page);
  await page.goto("/settings");
  await page.locator('input[autocomplete="email"]').fill(platformEmail);
  await page.locator('input[autocomplete="current-password"]').fill(platformPassword);
  await page.getByRole("button", { name: "登录", exact: true }).click();
  await expect(page.locator(".side-footer").getByText(platformDisplayName, { exact: true })).toBeVisible();

  await page.goto("/chat");
  await page.getByRole("button", { name: "+ 新对话", exact: true }).click();

  const question = "请查看 2018-08-20 的经营表现。";
  const initialRequestPromise = page.waitForRequest(streamRequest);
  const initialStreamMark = await streamMark(page);
  await page.locator('textarea[placeholder="输入经营问题，Enter 发送，Shift + Enter 换行"]').fill(question);
  await page.getByRole("button", { name: "发送", exact: true }).click();
  const initialRequest = await initialRequestPromise;
  const initialPayload = initialRequest.postDataJSON() as Record<string, any>;
  const initialStream = await capturedStream(page, initialStreamMark);
  expect(initialStream.status).toBe(200);
  const initialEvents = parseSse(initialStream.body);
  const clarification = [...initialEvents].reverse().find((event) => event.event === "suspended");
  expect(clarification, JSON.stringify(initialEvents.map((event) => event.event))).toBeTruthy();
  expect(clarification!.clarification?.ambiguous_candidates?.length ?? 0).toBeGreaterThan(0);
  const firstCandidate = clarification!.clarification.ambiguous_candidates[0] as { metric_id: string; display_name: string };
  expect(firstCandidate.metric_id).toBeTruthy();
  expect(firstCandidate.display_name).toBeTruthy();

  const beforeReload = await readTurns(page, String(initialPayload.chat_session_id));
  expect(beforeReload.items).toHaveLength(1);
  const historicalTurn = beforeReload.items[0];
  expect(historicalTurn.clarification?.ambiguous_candidates).toEqual(expect.arrayContaining([
    expect.objectContaining({ metric_id: firstCandidate.metric_id, display_name: firstCandidate.display_name }),
  ]));
  await page.reload();
  const candidateButton = page.getByRole("button", { name: `确认使用：${firstCandidate.display_name}`, exact: true });
  await expect(candidateButton).toBeVisible();

  const nextRequestPromise = page.waitForRequest(streamRequest);
  const nextStreamMark = await streamMark(page);
  await candidateButton.click();
  const nextRequest = await nextRequestPromise;
  const nextPayload = nextRequest.postDataJSON() as Record<string, any>;
  const nextStream = await capturedStream(page, nextStreamMark);
  expect(nextStream.status).toBe(200);
  expect(nextPayload.chat_session_id).toBe(initialPayload.chat_session_id);
  expect(nextPayload.chat_turn_id).not.toBe(initialPayload.chat_turn_id);
  expect(nextPayload.thread_id).not.toBe(initialPayload.thread_id);
  expect(nextPayload.metric_confirmation).toContain(firstCandidate.metric_id);
  const nextEvents = parseSse(nextStream.body);
  const terminal = [...nextEvents].reverse().find((event) => ["completed", "suspended", "failed"].includes(event.event));
  expect(terminal).toBeTruthy();
  expect(terminal!.event).not.toBe("failed");
  expect(terminal!.trace_id).toBeTruthy();
  const afterConfirmation = await readTurns(page, String(initialPayload.chat_session_id));
  expect(afterConfirmation.items).toHaveLength(2);
  expect(afterConfirmation.items.map((turn) => turn.id)).toContain(historicalTurn.id);
  expect(afterConfirmation.items.map((turn) => turn.client_turn_id)).not.toContain(initialPayload.chat_turn_id);
  const newTurn = afterConfirmation.items.find((turn) => turn.id !== historicalTurn.id);
  expect(newTurn).toBeTruthy();

  const storedAuth = await page.evaluate(() => JSON.parse(localStorage.getItem("bizinsight.auth.v1") ?? "{}") as { bearerToken?: string });
  expect(storedAuth.bearerToken).toBeTruthy();
  const initialTraceResponse = await page.request.get(`/traces/${encodeURIComponent(String(clarification!.trace_id))}`, {
    headers: { Authorization: `Bearer ${storedAuth.bearerToken}` },
  });
  const confirmedTraceResponse = await page.request.get(`/traces/${encodeURIComponent(String(terminal!.trace_id))}`, {
    headers: { Authorization: `Bearer ${storedAuth.bearerToken}` },
  });
  expect(initialTraceResponse.ok()).toBeTruthy();
  expect(confirmedTraceResponse.ok()).toBeTruthy();
  const initialTrace = await initialTraceResponse.json() as Record<string, any>;
  const confirmedTrace = await confirmedTraceResponse.json() as Record<string, any>;
  const initialState = initialTrace.state_snapshot ?? {};
  const confirmedState = confirmedTrace.state_snapshot ?? {};
  expect(initialState.trace_id).toBe(clarification!.trace_id);
  expect(initialState.thread_id).toBe(initialPayload.thread_id);
  expect(initialState.chat_turn_id).toBe(initialPayload.chat_turn_id);
  expect(confirmedState.trace_id).toBe(terminal!.trace_id);
  expect(confirmedState.thread_id).toBe(nextPayload.thread_id);
  expect(confirmedState.chat_turn_id).toBe(nextPayload.chat_turn_id);
  expect(confirmedState.metric_confirmation).toContain(firstCandidate.metric_id);
  expect(confirmedState.metric_planning_decision?.status).toBe("confirmed");
  expect(confirmedState.metric_planning_decision?.confirmed_metrics).toContain(firstCandidate.metric_id);

  await testInfo.attach("history-metric-evidence.json", {
    body: Buffer.from(JSON.stringify({
      test: testInfo.title,
      session_id: initialPayload.chat_session_id,
      old_turn_id: historicalTurn.id,
      new_turn_id: newTurn!.id,
      old_client_turn_id: initialPayload.chat_turn_id,
      new_client_turn_id: nextPayload.chat_turn_id,
      old_trace_id: clarification!.trace_id,
      new_trace_id: terminal!.trace_id,
      confirmed_metric_id: firstCandidate.metric_id,
      old_thread_id: initialPayload.thread_id,
      new_thread_id: nextPayload.thread_id,
    }, null, 2)),
    contentType: "application/json",
  });
});
