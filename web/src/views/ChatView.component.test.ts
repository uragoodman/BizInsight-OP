import { createRenderer, nextTick, ssrContextKey } from "vue";
import { afterEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  streamChat: vi.fn(),
  createComplexPlan: vi.fn(),
  warning: vi.fn(),
  store: null as any,
}));

vi.mock("@/api", () => ({ api: { createComplexPlan: mocks.createComplexPlan }, streamChat: mocks.streamChat, streamComplexPlan: vi.fn() }));
vi.mock("@/stores/app", () => ({ useAppStore: () => mocks.store }));
vi.mock("@/components/AnalysisResultCard.vue", () => ({ default: { template: "<div />" } }));
vi.mock("@/components/MarkdownContent.vue", () => ({ default: { template: "<div />" } }));
vi.mock("@/components/ScopeTag.vue", () => ({ default: { template: "<div />" } }));
vi.mock("@/components/TaskContextCard.vue", () => ({ default: { template: "<div />" } }));
vi.mock("element-plus", () => ({ ElMessage: { warning: mocks.warning, error: vi.fn(), success: vi.fn() }, ElMessageBox: {} }));
vi.mock("@element-plus/icons-vue", () => ({ DocumentCopy: { template: "<i />" } }));

import ChatView from "@/views/ChatView.vue";
import { clarificationMessage, clarificationTitle } from "@/utils/clarificationCopy";

type HostNode = {
  type: string;
  props: Record<string, any>;
  children: HostNode[];
  parent: HostNode | null;
  text: string;
  scrollTop: number;
};

function hostNode(type: string, text = ""): HostNode {
  return { type, props: {}, children: [], parent: null, text, scrollTop: 0 };
}

const renderer = createRenderer<HostNode, HostNode>({
  createElement: (type) => hostNode(type),
  createText: (text) => hostNode("#text", text),
  createComment: (text) => hostNode("#comment", text),
  setText: (node, text) => { node.text = text; },
  setElementText: (node, text) => { node.text = text; node.children = []; },
  patchProp: (node, key, _previous, value) => { node.props[key] = value; },
  insert: (node, parent, anchor) => {
    node.parent = parent;
    const existing = parent.children.indexOf(node);
    if (existing >= 0) parent.children.splice(existing, 1);
    const index = anchor ? parent.children.indexOf(anchor) : -1;
    parent.children.splice(index < 0 ? parent.children.length : index, 0, node);
  },
  remove: (node) => {
    if (!node.parent) return;
    const index = node.parent.children.indexOf(node);
    if (index >= 0) node.parent.children.splice(index, 1);
    node.parent = null;
  },
  parentNode: (node) => node.parent,
  nextSibling: (node) => node.parent?.children[node.parent.children.indexOf(node) + 1] ?? null,
});

afterEach(() => {
  vi.clearAllMocks();
  mocks.store = null;
});

describe("ChatView Phase 0 complex-analysis boundary", () => {
  it("submits ordinary chat without a template-choice field or plan creation path", async () => {
    const session = { id: "session-1", scope: null, turns: [] };
    mocks.store = {
      auth: { actor: "admin-1", analysisToken: "", bearerToken: "bearer" },
      activeSession: session,
      createSession: vi.fn(() => session),
      addTurn: vi.fn(),
      loadSessions: vi.fn().mockResolvedValue(undefined),
    };
    mocks.streamChat.mockResolvedValue({
      event: "completed",
      task_type: "metric_query",
      answer: "GMV 为 100。",
      trace_id: "trace-1",
    });

    const root = hostNode("root");
    const app = renderer.createApp(ChatView);
    app.config.warnHandler = (message) => {
      if (!message.includes("missing template or render function")) throw new Error(message);
    };
    app.provide(ssrContextKey, { modules: new Set<string>() });
    app.mount(root);
    await nextTick();
    const setupState = (app as any)._instance?.setupState;
    expect(setupState?.selectComplexTemplate).toBeTypeOf("function");
    expect(setupState?.completeComplexScope).toBeUndefined();
    expect(setupState?.scopeRequest).toBeUndefined();
    setupState.question = "请查看 GMV";
    await nextTick();
    await setupState.submit();

    expect(mocks.streamChat).toHaveBeenCalledTimes(1);
    expect(mocks.streamChat.mock.calls[0][1]).toMatchObject({
      message: "请查看 GMV",
    });
    expect(mocks.streamChat.mock.calls[0][1]).not.toHaveProperty("complex_template_confirmation");
    app.unmount();
  });

  it("reparses a user's template choice through chat without sending a template ID", async () => {
    mocks.store = {
      auth: { actor: "admin-1", analysisToken: "", bearerToken: "bearer" },
      activeSession: { id: "session-1", scope: null, turns: [] },
      createSession: vi.fn(),
      addTurn: vi.fn(),
      loadSessions: vi.fn().mockResolvedValue(undefined),
    };
    mocks.streamChat.mockResolvedValue({
      event: "completed",
      task_type: "complex_analysis",
      answer: "已生成待确认计划。",
      trace_id: "trace-choice",
      complex_plan: { plan_id: "plan-1", template_name: "cohort_retention" },
    });
    const root = hostNode("root");
    const app = renderer.createApp(ChatView);
    app.config.warnHandler = (message) => {
      if (!message.includes("missing template or render function")) throw new Error(message);
    };
    app.provide(ssrContextKey, { modules: new Set<string>() });
    app.mount(root);
    await nextTick();
    const setupState = (app as any)._instance?.setupState;
    expect(setupState?.selectComplexTemplate).toBeTypeOf("function");
    await setupState.selectComplexTemplate(
      { template_id: "cohort_retention", display_name: "Cohort 留存分析" },
      "客户留存分析",
    );
    expect(mocks.streamChat).toHaveBeenCalledTimes(1);
    expect(mocks.streamChat.mock.calls[0][1].message).toBe(
      "客户留存分析\n我选择Cohort 留存分析。",
    );
    expect(mocks.streamChat.mock.calls[0][1]).not.toHaveProperty("template_id");
    expect(mocks.createComplexPlan).not.toHaveBeenCalled();
    app.unmount();
  });
});

describe("ChatView premise checkpoint resume", () => {
  it("reuses only the pending same-scope thread for an explicit premise correction", async () => {
    const session: { id: string; scope: null; scopeFingerprint?: string; turns: any[] } = {
      id: "session-premise", scope: null, turns: [],
    };
    mocks.store = {
      auth: { actor: "admin-1", analysisToken: "", bearerToken: "bearer" },
      activeSession: session,
      createSession: vi.fn(() => session),
      addTurn: vi.fn((turn) => {
        const existing = session.turns.findIndex((item) => item.id === turn.id);
        if (existing >= 0) session.turns.splice(existing, 1, turn);
        else session.turns.unshift(turn);
      }),
      loadSessions: vi.fn().mockResolvedValue(undefined),
    };
    mocks.streamChat
      .mockResolvedValueOnce({
        event: "suspended", task_type: "anomaly_analysis", answer: "",
        trace_id: "trace-premise", interaction_state: "awaiting_human_input",
        interaction_reason_code: "premise_direction_mismatch", scope_fingerprint: "server-scope",
      })
      .mockResolvedValueOnce({
        event: "completed", task_type: "anomaly_analysis", answer: "按实际数据继续分析。",
        trace_id: "trace-premise-resumed",
      })
      .mockResolvedValueOnce({
        event: "completed", task_type: "metric_query", answer: "GMV 为 100。",
        trace_id: "trace-new-topic",
      });

    const root = hostNode("root");
    const app = renderer.createApp(ChatView);
    app.config.warnHandler = (message) => {
      if (!message.includes("missing template or render function")) throw new Error(message);
    };
    app.provide(ssrContextKey, { modules: new Set<string>() });
    app.mount(root);
    await nextTick();
    const setupState = (app as any)._instance?.setupState;
    if (!setupState?.submit) throw new Error("ChatView submit handler was not exposed");

    setupState.question = "为什么 2018-08-20 的 GMV 下降了？";
    await setupState.submit();
    const firstThread = mocks.streamChat.mock.calls[0][1].thread_id;
    const firstTurnId = mocks.streamChat.mock.calls[0][1].chat_turn_id;
    // The initial suspended SSE event carries the server-validated recovery ticket.
    session.scopeFingerprint = "server-scope";
    expect(session.turns[0]).toMatchObject({
      status: "suspended",
      interactionState: "awaiting_human_input",
      interactionReasonCode: "premise_direction_mismatch",
      scopeFingerprint: "server-scope",
    });

    setupState.question = "我刚才的下降前提可能错了，请按实际数据纠正方向。";
    await setupState.submit();
    expect(mocks.streamChat.mock.calls[1][1].thread_id).toBe(firstThread);
    expect(mocks.streamChat.mock.calls[1][1].chat_turn_id).toBe(firstTurnId);
    expect(session.turns).toHaveLength(1);

    setupState.question = "请查看 2018-08-20 的 GMV。";
    await setupState.submit();
    expect(mocks.streamChat.mock.calls[2][1].thread_id).not.toBe(firstThread);
    app.unmount();
  });

  it("does not resume when the server omits the recovery ticket for a changed scope", async () => {
    const originalScope = {
      seller_id: "seller-a", metric_name: "gmv", start_date: "2018-08-01",
      end_date: "2018-08-31", granularity: "day",
    };
    const changedScope = { ...originalScope, seller_id: "seller-b" };
    const pendingTurn = {
      id: "turn-pending", threadId: "", message: "为什么 GMV 下降？",
      taskType: "anomaly_analysis", traceId: "trace-pending", createdAt: "now",
      status: "suspended", interactionState: "awaiting_human_input",
      // /turns omits its recovery-only fields when current scope differs.
      scopeKey: JSON.stringify([originalScope.seller_id, originalScope.metric_name, originalScope.start_date, originalScope.end_date, originalScope.granularity]),
    };
    const session = { id: "session-scope", scope: changedScope, turns: [pendingTurn] };
    mocks.store = {
      auth: { actor: "seller-1", analysisToken: "", bearerToken: "bearer" },
      activeSession: session,
      createSession: vi.fn(() => session),
      addTurn: vi.fn((turn) => session.turns.unshift(turn)),
      loadSessions: vi.fn().mockResolvedValue(undefined),
    };
    mocks.streamChat.mockResolvedValue({
      event: "completed", task_type: "metric_query", answer: "GMV 为 100。", trace_id: "trace-new",
    });

    const root = hostNode("root");
    const app = renderer.createApp(ChatView);
    app.config.warnHandler = (message) => {
      if (!message.includes("missing template or render function")) throw new Error(message);
    };
    app.provide(ssrContextKey, { modules: new Set<string>() });
    app.mount(root);
    await nextTick();
    const setupState = (app as any)._instance?.setupState;
    setupState.question = "请按实际情况继续分析。";
    await setupState.submit();

    expect(mocks.streamChat.mock.calls[0][1].thread_id).not.toBe("thread-pending");
    app.unmount();
  });

  it("rejects a resume ticket when its server scope fingerprint differs from the session", async () => {
    const scope = {
      seller_id: "seller-a", metric_name: "gmv", start_date: "2018-08-01",
      end_date: "2018-08-31", granularity: "day",
    };
    const session = {
      id: "session-server-scope", scope, scopeFingerprint: "current-scope",
      turns: [{
        id: "turn-paused", clientTurnId: "client-paused", threadId: "thread-paused",
        message: "为什么 GMV 下降？", taskType: "anomaly_analysis", traceId: "trace-paused",
        createdAt: "now", status: "running", interactionState: "awaiting_human_input",
        interactionReasonCode: "premise_direction_mismatch",
        scopeKey: JSON.stringify([scope.seller_id, scope.metric_name, scope.start_date, scope.end_date, scope.granularity]),
      scopeFingerprint: "server-issued-scope",
      }],
    };
    mocks.store = {
      auth: { actor: "seller-1", analysisToken: "", bearerToken: "bearer" },
      activeSession: session, createSession: vi.fn(() => session), addTurn: vi.fn(),
      loadSessions: vi.fn().mockResolvedValue(undefined),
    };
    mocks.streamChat.mockResolvedValue({
      event: "completed", task_type: "metric_query", answer: "GMV 为 100。", trace_id: "trace-new",
    });
    const root = hostNode("root");
    const app = renderer.createApp(ChatView);
    app.config.warnHandler = (message) => {
      if (!message.includes("missing template or render function")) throw new Error(message);
    };
    app.provide(ssrContextKey, { modules: new Set<string>() });
    app.mount(root);
    await nextTick();
    const setupState = (app as any)._instance?.setupState;
    setupState.question = "按实际情况继续分析。";
    await setupState.submit();

    expect(mocks.streamChat.mock.calls[0][1].thread_id).not.toBe("thread-paused");
    expect(mocks.streamChat.mock.calls[0][1].chat_turn_id).not.toBe("client-paused");
    app.unmount();
  });

  it("resumes a server-validated history ticket when the client-only scope key is absent", async () => {
    const scope = {
      seller_id: "seller-a", metric_name: "gmv", start_date: "2018-08-01",
      end_date: "2018-08-31", granularity: "day",
    };
    const session = {
      id: "session-restored-scope", scope, scopeFingerprint: "server-scope",
      turns: [{
        id: "turn-restored", clientTurnId: "client-restored", threadId: "thread-restored",
        message: "为什么 GMV 下降？", taskType: "anomaly_analysis", traceId: "trace-restored",
        createdAt: "now", status: "running", interactionState: "awaiting_human_input",
        interactionReasonCode: "premise_direction_mismatch", scopeFingerprint: "server-scope",
        // /turns returns only the server-validated fingerprint ticket; scopeKey
        // is a frontend-only continuity key and is absent after reload.
      }],
    };
    mocks.store = {
      auth: { actor: "seller-1", analysisToken: "", bearerToken: "bearer" },
      activeSession: session, createSession: vi.fn(() => session), addTurn: vi.fn(),
      loadSessions: vi.fn().mockResolvedValue(undefined),
    };
    mocks.streamChat.mockResolvedValue({
      event: "completed", task_type: "anomaly_analysis", answer: "GMV 实际上升。", trace_id: "trace-resumed",
    });

    const root = hostNode("root");
    const app = renderer.createApp(ChatView);
    app.config.warnHandler = (message) => {
      if (!message.includes("missing template or render function")) throw new Error(message);
    };
    app.provide(ssrContextKey, { modules: new Set<string>() });
    app.mount(root);
    await nextTick();
    const setupState = (app as any)._instance?.setupState;
    setupState.question = "按实际数据纠正方向";
    await setupState.submit();

    expect(mocks.streamChat.mock.calls[0][1].thread_id).toBe("thread-restored");
    expect(mocks.streamChat.mock.calls[0][1].chat_turn_id).toBe("client-restored");
    app.unmount();
  });
});

describe("ChatView persisted clarification readback", () => {
  it("restores the pending clarification data and its localized copy for history", async () => {
    const session = {
      id: "session-history-clarification", scope: null,
      turns: [{
        id: "turn-pending", threadId: "", message: "按店铺和商品品类拆解订单量",
        answer: "请求中有多个可能的维度匹配。请明确选择一个维度。",
        taskType: "clarification", traceId: "trace-pending", createdAt: "now",
        status: "running", interactionState: "awaiting_metric_confirmation",
        clarification: {
          status: "clarification_required", reason_code: "ambiguous_dimension",
          missing_slots: ["dimension"], dimension_candidates: ["seller", "product_category"],
        },
      }],
    };
    mocks.store = {
      auth: { actor: "admin-1", analysisToken: "", bearerToken: "bearer" },
      activeSession: session,
      createSession: vi.fn(() => session), addTurn: vi.fn(),
      loadSessions: vi.fn().mockResolvedValue(undefined),
    };
    const root = hostNode("root");
    const app = renderer.createApp(ChatView);
    app.config.warnHandler = (message) => {
      if (!message.includes("missing template or render function")) throw new Error(message);
    };
    app.provide(ssrContextKey, { modules: new Set<string>() });
    app.mount(root);
    await nextTick();

    const setupState = (app as any)._instance?.setupState;
    expect(setupState.turns[0].clarification.dimension_candidates).toEqual(["seller", "product_category"]);
    expect(clarificationTitle(setupState.turns[0].clarification)).toBe("需要明确分组维度");
    expect(clarificationMessage(setupState.turns[0].clarification)).toContain("没有自动选择");
    expect(session.turns[0].status).toBe("running");
    app.unmount();
  });

  it("re-submits the original historical question when a persisted metric candidate is selected", async () => {
    const session = {
      id: "session-history-metric", scope: null,
      turns: [{
        id: "turn-metric", threadId: "", message: "请比较订单量和成交额",
        answer: "问题中有多个可能的指标匹配，请选择一个指标。",
        taskType: "clarification", traceId: "trace-metric", createdAt: "now",
        status: "running", interactionState: "awaiting_metric_confirmation",
        clarification: {
          status: "clarification_required", reason_code: "ambiguous_metric",
          ambiguous_candidates: [{ metric_id: "gmv", display_name: "成交总额" }],
        },
      }],
    };
    mocks.store = {
      auth: { actor: "admin-1", analysisToken: "", bearerToken: "bearer" },
      activeSession: session,
      createSession: vi.fn(() => session), addTurn: vi.fn(),
      loadSessions: vi.fn().mockResolvedValue(undefined),
    };
    mocks.streamChat.mockResolvedValue({
      event: "completed", task_type: "metric_query", answer: "已完成。", trace_id: "trace-next",
    });
    const root = hostNode("root");
    const app = renderer.createApp(ChatView);
    app.config.warnHandler = (message) => {
      if (!message.includes("missing template or render function")) throw new Error(message);
    };
    app.provide(ssrContextKey, { modules: new Set<string>() });
    app.mount(root);
    await nextTick();
    const setupState = (app as any)._instance?.setupState;

    await setupState.selectMetricCandidate(
      { metric_id: "gmv", display_name: "成交总额" },
      session.turns[0].message,
    );

    expect(mocks.streamChat.mock.calls[0][1]).toMatchObject({
      message: "请比较订单量和成交额\n请按成交总额指标查询。",
      metric_confirmation: ["gmv"],
    });
    app.unmount();
  });
});

describe("ChatView metric clarification selection", () => {
  it("resubmits the retained request with the selected exact registry phrase", async () => {
    const session = { id: "session-metric-clarification", scope: null, turns: [] as any[] };
    mocks.store = {
      auth: { actor: "admin-1", analysisToken: "", bearerToken: "bearer" },
      activeSession: session,
      createSession: vi.fn(() => session),
      addTurn: vi.fn(),
      loadSessions: vi.fn().mockResolvedValue(undefined),
    };
    mocks.streamChat.mockResolvedValue({
      event: "completed", task_type: "metric_query", answer: "GMV 为 100。", trace_id: "trace-confirmed-metric",
    });

    const root = hostNode("root");
    const app = renderer.createApp(ChatView);
    app.config.warnHandler = (message) => {
      if (!message.includes("missing template or render function")) throw new Error(message);
    };
    app.provide(ssrContextKey, { modules: new Set<string>() });
    app.mount(root);
    await nextTick();
    const setupState = (app as any)._instance?.setupState;
    if (!setupState?.selectMetricCandidate) throw new Error("ChatView metric selection handler was not exposed");

    setupState.clarificationQuestion = "请查看 2018-08-20 的经营表现。";
    setupState.clarification = {
      status: "clarification_required",
      ambiguous_candidates: [{ metric_id: "gmv", display_name: "成交总额" }],
    };
    await setupState.selectMetricCandidate({ metric_id: "gmv", display_name: "成交总额" });

    const request = mocks.streamChat.mock.calls[0][1];
    expect(request.message).toBe("请查看 2018-08-20 的经营表现。\n请按成交总额指标查询。");
    expect(request.metric_confirmation).toEqual(["gmv"]);
    app.unmount();
  });
});
