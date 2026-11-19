import { createRenderer, nextTick, ssrContextKey } from "vue";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ExplorerQuery, SignedSellerScopePayload } from "@/types";

const mocks = vi.hoisted(() => ({
  streamChat: vi.fn(),
  warning: vi.fn(),
  store: null as any,
}));

vi.mock("@/api", () => ({ api: {}, streamChat: mocks.streamChat }));
vi.mock("@/stores/app", () => ({ useAppStore: () => mocks.store }));
vi.mock("@/components/AnalysisResultCard.vue", () => ({ default: { template: "<div />" } }));
vi.mock("element-plus", () => ({ ElMessage: { warning: mocks.warning, error: vi.fn(), success: vi.fn() } }));

import ScopedChatPanel from "@/components/ScopedChatPanel.vue";

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
  nextSibling: (node) => {
    if (!node.parent) return null;
    return node.parent.children[node.parent.children.indexOf(node) + 1] ?? null;
  },
});

const legacyScope: ExplorerQuery = {
  seller_id: "legacy-seller", metric_name: "gmv", start_date: "2018-01-01", end_date: "2018-01-31", granularity: "day",
};

function sellerPayload(expiresAt: string): SignedSellerScopePayload {
  return {
    seller_scope_intent: {
      seller_scope_type: "selected", seller_ids: ["seller-a"], start_date: "2018-01-01", end_date: "2018-01-31",
      metric_ids: ["gmv"], dimensions: ["seller"], granularity: "day", include_unattributed: false,
    },
    provenance: { source: "explorer_selector", token: "signed-token", expires_at: expiresAt },
  };
}

async function mountPanel(payload: SignedSellerScopePayload) {
  const store = {
    auth: { actor: "user-1", analysisToken: "", bearerToken: "bearer" },
    activeSession: { id: "session-1", scope: legacyScope, turns: [] },
    createSession: vi.fn(),
    addTurn: vi.fn(),
  };
  mocks.store = store;
  const root = hostNode("root");
  const app = renderer.createApp(ScopedChatPanel, { sellerScopePayload: payload });
  // Vitest transforms this scoped SFC into its SSR render variant; this test
  // invokes the real setup handler through Vue's renderer without a DOM.
  app.config.warnHandler = (message) => {
    if (!message.includes("missing template or render function")) throw new Error(message);
  };
  app.provide(ssrContextKey, { modules: new Set<string>() });
  app.mount(root);
  await nextTick();
  const setupState = (app as any)._instance?.setupState;
  if (!setupState?.submit) throw new Error("ScopedChatPanel setup handler was not exposed");
  setupState.question = "按 seller 拆解 GMV";
  await nextTick();
  return { app, root, store, submit: setupState.submit, setupState };
}

afterEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
  mocks.store = null;
});

describe("ScopedChatPanel seller-scope component", () => {
  it("submits the signed seller carrier without legacy Explorer fields", async () => {
    vi.setSystemTime(new Date("2026-11-06T00:00:00Z"));
    mocks.streamChat.mockResolvedValue({ answer: "ok", error: false, trace_id: "trace-1", task_type: "metric_query" });
    const signed = sellerPayload("2026-11-06T00:01:00Z");
    const { app, submit } = await mountPanel(signed);

    await submit();

    const request = mocks.streamChat.mock.calls[0][1];
    expect(request.seller_scope_payload).toBe(signed);
    expect(request).not.toHaveProperty("explorer_context");
    expect(request).not.toHaveProperty("use_current_explorer_context");
    app.unmount();
  });

  it("blocks a token that expires after the component is mounted", async () => {
    vi.setSystemTime(new Date("2026-11-06T00:00:00Z"));
    const { app, submit, setupState } = await mountPanel(sellerPayload("2026-11-06T00:00:01Z"));
    vi.setSystemTime(new Date("2026-11-06T00:00:02Z"));

    await submit();

    expect(mocks.streamChat).not.toHaveBeenCalled();
    expect(mocks.warning).toHaveBeenCalledWith(expect.stringContaining("已过期"));
    expect(setupState.working).toBe(false);
    app.unmount();
  });
});
