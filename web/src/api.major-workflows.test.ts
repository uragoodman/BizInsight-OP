import { afterEach, describe, expect, it, vi } from "vitest";
import { api, streamChat, type ApiAuth } from "./api";

const auth: ApiAuth = { actor: "api-test-user", analysisToken: "", bearerToken: "test-bearer" };

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status, headers: { "Content-Type": "application/json" } });
}

function sseResponse(events: unknown[]): Response {
  const body = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("");
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

afterEach(() => vi.unstubAllGlobals());

describe("frontend API contracts for major workflows", () => {
  it("auth: sends login credentials to the login API", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ access_token: "token", profile: { user_id: "u1" } })));
    await api.login("user@example.invalid", "test-password");
    expect(fetch).toHaveBeenCalledWith("/auth/login", expect.objectContaining({ method: "POST" }));
  });

  it("auth: requests the profile with the session bearer", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ user_id: "u1" })));
    await api.me(auth);
    expect(fetch).toHaveBeenCalledWith("/auth/me", expect.objectContaining({ headers: expect.objectContaining({ Authorization: "Bearer test-bearer" }) }));
  });

  it("auth: sends logout with the session bearer", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ logged_out: true })));
    await api.logout(auth);
    expect(fetch).toHaveBeenCalledWith("/auth/logout", expect.objectContaining({ method: "POST" }));
  });

  it("explorer: loads dataset time bounds", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ min_date: "2017-01-01", max_date: "2018-08-31" })));
    await api.timeBounds(auth);
    expect(fetch).toHaveBeenCalledWith("/data-explorer/time-bounds", expect.any(Object));
  });

  it("explorer: resolves a selected period preset", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ start_date: "2018-08-01", end_date: "2018-08-31" })));
    await api.resolveTimeRange(auth, { kind: "month", year: 2018, value: 8 });
    expect(fetch).toHaveBeenCalledWith("/data-explorer/time-range/resolve", expect.objectContaining({ method: "POST" }));
  });

  it("explorer: posts the metric and locked date scope", async () => {
    const query = { seller_id: "seller-1", metric_name: "gmv", start_date: "2018-08-01", end_date: "2018-08-31", granularity: "month" as const };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ context: query, points: [] })));
    await api.preview(auth, query);
    expect(fetch).toHaveBeenCalledWith("/data-explorer/preview", expect.objectContaining({ method: "POST", body: JSON.stringify(query) }));
  });

  it("chat: consumes a completed SSE answer and sends the scoped request", async () => {
    const result = { task_type: "metric_query", answer: "GMV 已查询", trace_id: "trace-1" };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse([
      { event: "started", trace_id: "trace-1", sequence_number: 0 },
      { event: "completed", ...result, sequence_number: 1 },
    ])));
    const payload = { message: "查询月度 GMV", use_current_explorer_context: true, explorer_context: { seller_id: "seller-1" } };
    await expect(streamChat(auth, payload, () => undefined)).resolves.toMatchObject({ trace_id: "trace-1" });
    expect(fetch).toHaveBeenCalledWith("/chat/stream", expect.objectContaining({ method: "POST", body: JSON.stringify(payload) }));
  });

  it("chat: returns a suspended clarification as a terminal result", async () => {
    const clarification = {
      status: "clarification_required",
      reason_code: "comparison_dimension_not_supported",
      reason: "暂不支持同时按多个维度比较。",
    };
    const onEvent = vi.fn();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse([
      { event: "suspended", task_type: "clarification", answer: "请调整问题", trace_id: "trace-suspended", interaction_state: "awaiting_metric_confirmation", clarification, sequence_number: 1 },
    ])));

    await expect(streamChat(auth, { message: "按品类查看 GMV 并比较支付方式" }, onEvent)).resolves.toMatchObject({
      event: "suspended",
      trace_id: "trace-suspended",
      interaction_state: "awaiting_metric_confirmation",
      clarification,
      error: false,
      error_detail: null,
    });
    expect(onEvent).toHaveBeenCalledWith(expect.objectContaining({ event: "suspended", clarification }));
  });

  it("chat: preserves a pending complex plan carried by a suspended event", async () => {
    const complexPlan = {
      plan_id: "plan-1",
      template_name: "cohort_retention",
      status: "pending_confirmation",
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse([
      {
        event: "suspended",
        task_type: "complex_analysis",
        answer: "已生成复杂分析计划，请确认后执行。",
        trace_id: "trace-plan",
        interaction_state: "awaiting_metric_confirmation",
        clarification: { status: "confirmation_required", reason_code: "plan_confirmation_required" },
        complex_plan: complexPlan,
        sequence_number: 1,
      },
    ])));

    await expect(streamChat(auth, { message: "Cohort 留存分析" }, () => undefined)).resolves.toMatchObject({
      event: "suspended",
      complex_plan: complexPlan,
      clarification: { status: "confirmation_required" },
    });
  });

  it("chat: preserves a human-review suspension without calling it completed", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse([
      { event: "suspended", task_type: "metric_query", answer: "等待人工审核", trace_id: "trace-review", interaction_state: "awaiting_human_input", sequence_number: 1 },
    ])));

    await expect(streamChat(auth, { message: "query" }, () => undefined)).resolves.toMatchObject({
      event: "suspended",
      interaction_state: "awaiting_human_input",
      answer: "等待人工审核",
      clarification: null,
    });
  });

  it("chat: surfaces a terminal SSE failure instead of hanging", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse([{ event: "failed", detail: "scope denied", sequence_number: 1 }])));
    await expect(streamChat(auth, { message: "query" }, () => undefined)).rejects.toThrow("scope denied");
  });

  it("chat: posts an explicit semantic-mapping confirmation", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ status: "confirmed", mapping_id: "map-1" })));
    await api.confirmSemanticMapping(auth, "map/1", { chat_session_id: "session-1" });
    expect(fetch).toHaveBeenCalledWith("/chat/semantic-mappings/map%2F1/confirm", expect.objectContaining({ method: "POST" }));
  });

  it("reports: creates a report through the report API", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ report: { report_id: "r1" }, status: "draft" })));
    await api.createReport(auth, { metric_names: ["gmv"] });
    expect(fetch).toHaveBeenCalledWith("/analysis/reports", expect.objectContaining({ method: "POST" }));
  });

  it("reports: reviews and publishes the same report resource", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ status: "published" })));
    await api.reviewReport(auth, "r/1", { action: "approve", version: 1 });
    await api.publishReport(auth, "r/1", { version: 1 });
    expect(fetch).toHaveBeenNthCalledWith(1, "/analysis/reports/r%2F1/review", expect.objectContaining({ method: "POST" }));
    expect(fetch).toHaveBeenNthCalledWith(2, "/analysis/reports/r%2F1/publish", expect.objectContaining({ method: "POST" }));
  });

  it("reports: reads generated insight artifacts", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse([{ insight_version: 1, status: "draft" }])));
    await api.reportInsights(auth, "r1", 2);
    expect(fetch).toHaveBeenCalledWith("/analysis/reports/r1/versions/2/insights", expect.any(Object));
  });

  it("complex analysis: creates a confirmation-gated plan", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ plan_id: "p1", status: "awaiting_confirmation" })));
    await api.createComplexPlan(auth, { question: "compare periods", explorer_context: {} });
    expect(fetch).toHaveBeenCalledWith("/complex-analysis/plans", expect.objectContaining({ method: "POST" }));
  });

  it("complex analysis: revises a plan without changing endpoint identity", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ plan_id: "p1", revision: 2 })));
    await api.reviseComplexPlan(auth, "p/1", { note: "focus category" });
    expect(fetch).toHaveBeenCalledWith("/complex-analysis/plans/p%2F1/revise", expect.objectContaining({ method: "POST" }));
  });

  it("complex analysis: supports cancellation", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ cancelled: true })));
    await api.cancelComplexPlan(auth, "p1");
    expect(fetch).toHaveBeenCalledWith("/complex-analysis/plans/p1/cancel", expect.objectContaining({ method: "POST" }));
  });

  it("knowledge: lists documents without exposing content by default", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ total: 1, documents: [{ doc_id: "d1" }] })));
    await api.knowledge(auth);
    expect(fetch).toHaveBeenCalledWith("/knowledge/documents", expect.any(Object));
  });

  it("knowledge: requests an index build with a configured collection", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ status: "skipped" })));
    await api.indexKnowledge(auth, "bizinsight knowledge");
    expect(fetch).toHaveBeenCalledWith("/knowledge/index?collection=bizinsight%20knowledge", expect.objectContaining({ method: "POST" }));
  });

  it("knowledge: reaches the model connectivity check", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ status: "ok" })));
    await api.testLlm(auth);
    expect(fetch).toHaveBeenCalledWith("/chat/test-llm", expect.objectContaining({ method: "POST" }));
  });
});
