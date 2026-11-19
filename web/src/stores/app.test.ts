import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { api } from "@/api";
import { useAppStore } from "./app";
import type { DebugTurn, ExplorerPreview } from "@/types";

vi.mock("@/api", () => ({
  AUTH_EXPIRED_EVENT: "bizinsight:auth-expired",
  api: {
    login: vi.fn(),
    me: vi.fn(),
    logout: vi.fn(),
    chatSessions: vi.fn(),
    chatTurns: vi.fn(),
    deleteChatSession: vi.fn(),
    deleteChatTurn: vi.fn(),
  },
}));

class MemoryStorage implements Storage {
  private store = new Map<string, string>();
  get length() { return this.store.size; }
  clear() { this.store.clear(); }
  getItem(key: string) { return this.store.get(key) ?? null; }
  key(index: number) { return [...this.store.keys()][index] ?? null; }
  removeItem(key: string) { this.store.delete(key); }
  setItem(key: string, value: string) { this.store.set(key, String(value)); }
}

const sessionItem = (id: string, title = `会话-${id}`) => ({
  id, title, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z", scope: null, turn_count: 0,
});
const turnItem = (id: string) => ({
  id, message: `msg-${id}`, answer: "", task_type: "metric_definition", trace_id: "", created_at: "2026-01-01T00:00:00Z",
});
const turn = (id: string, message: string): DebugTurn => ({
  id, threadId: "", message, answer: "", taskType: "metric_definition", traceId: "", createdAt: "2026-01-01T00:00:00Z", status: "completed",
});
const previewFixture: ExplorerPreview = {
  context: { seller_id: "seller-1", metric_name: "gmv", start_date: "2018-08-01", end_date: "2018-08-10", granularity: "day", locked: true, scope: "platform" },
  summary: { metric_name: "gmv", value: 10, unit: "BRL", status: "ok" },
  series: [], table: { columns: [], rows: [] },
  lineage: { source_table: "analytics.fact_orders", query_time: "server", latest_data_date: "2018-08-10" },
};

beforeEach(() => {
  vi.stubGlobal("localStorage", new MemoryStorage());
  setActivePinia(createPinia());
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("app store: sessions", () => {
  it("creates a session, makes it active, and persists it", () => {
    const store = useAppStore();
    const session = store.createSession();
    expect(store.sessions).toHaveLength(1);
    expect(store.activeSessionId).toBe(session.id);
    expect(session.title).toBe("新的调试会话");
    expect(localStorage.getItem("bizinsight.debug-sessions.v1")).toContain(session.id);
  });

  it("titles a session from the first turn, truncated to 28 chars", () => {
    const store = useAppStore();
    store.addTurn(turn("t1", "x".repeat(50)));
    expect(store.activeSession?.title).toBe("x".repeat(28));
  });

  it("overwrites a turn with the same id instead of duplicating it", () => {
    const store = useAppStore();
    store.addTurn(turn("t1", "first"));
    store.addTurn(turn("t1", "second"));
    expect(store.activeSession?.turns).toHaveLength(1);
    expect(store.activeSession?.turns[0].message).toBe("second");
  });

  it("keeps the most recently updated session first", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
    const store = useAppStore();
    store.createSession();
    const older = store.activeSessionId;
    vi.setSystemTime(new Date("2026-01-02T00:00:00Z"));
    store.createSession();
    expect(store.sessions[0].id).toBe(store.activeSessionId);
    store.activeSessionId = older;
    vi.setSystemTime(new Date("2026-01-03T00:00:00Z"));
    store.addTurn(turn("t1", "touch older"));
    expect(store.sessions[0].id).toBe(older);
  });

  it("stores a copy of the scope so later mutations of the caller object stay isolated", () => {
    const store = useAppStore();
    const scope = { seller_id: "s1", metric_name: "gmv", start_date: "2018-08-01", end_date: "2018-08-10", granularity: "day" as const };
    store.setActiveSessionScope(scope);
    scope.seller_id = "mutated";
    expect(store.activeSession?.scope?.seller_id).toBe("s1");
  });

  it("clears the explorer preview", () => {
    const store = useAppStore();
    store.preview = previewFixture;
    store.clearPreview();
    expect(store.preview).toBeNull();
  });
});

describe("app store: pagination", () => {
  it("preserves a fresh session created while the server history reset is in flight", async () => {
    let resolvePage!: (page: Awaited<ReturnType<typeof api.chatSessions>>) => void;
    vi.mocked(api.chatSessions).mockImplementationOnce(() => new Promise((resolve) => { resolvePage = resolve; }));
    const store = useAppStore();
    const staleLocalSession = store.createSession();
    const loading = store.loadSessions(true);
    const freshSession = store.createSession();

    resolvePage({ items: [sessionItem("server-session")], next_offset: null });
    await loading;

    expect(store.sessions.map((session) => session.id)).toEqual([freshSession.id, "server-session"]);
    expect(store.sessions.some((session) => session.id === staleLocalSession.id)).toBe(false);
    expect(store.activeSessionId).toBe(freshSession.id);
  });

  it("loads sessions and deduplicates across pages", async () => {
    vi.mocked(api.chatSessions)
      .mockResolvedValueOnce({ items: [sessionItem("s1")], next_offset: 0 })
      .mockResolvedValueOnce({ items: [sessionItem("s1"), sessionItem("s2")], next_offset: null });
    const store = useAppStore();
    await store.loadSessions(true);
    await store.loadSessions(false);
    expect(store.sessions.map((s) => s.id)).toEqual(["s1", "s2"]);
  });

  it("restores the server-computed session scope fingerprint", async () => {
    vi.mocked(api.chatSessions).mockResolvedValue({
      items: [{
        ...sessionItem("s1"),
        scope_fingerprint: "b".repeat(64),
      }],
      next_offset: null,
    });
    const store = useAppStore();
    await store.loadSessions(true);
    expect(store.sessions[0].scopeFingerprint).toBe("b".repeat(64));
  });

  it("loads turns appended without duplicating already-known ids", async () => {
    vi.mocked(api.chatTurns)
      .mockResolvedValueOnce({ items: [turnItem("a")], next_offset: 1 })
      .mockResolvedValueOnce({ items: [turnItem("a"), turnItem("b")], next_offset: null });
    const store = useAppStore();
    store.createSession();
    const id = store.activeSessionId;
    await store.loadTurns(id, true);
    await store.loadTurns(id, false);
    expect(store.activeSession?.turns.map((t) => t.id)).toEqual(["a", "b"]);
  });

  it("restores final trace id and analysis presentation from chat history", async () => {
    const presentation = {
      status: "completed",
      facts: [{ metric_name: "gmv", value: 73.76, date_or_range: "2018-08-22" }],
    } as DebugTurn["presentation"];
    vi.mocked(api.chatTurns).mockResolvedValue({
      items: [{
        ...turnItem("completed"),
        trace_id: "final-trace",
        clarification: { status: "complex_scope_required", reason_code: "scope_required" },
        presentation,
      }],
      next_offset: null,
    });
    const store = useAppStore();
    store.createSession();
    await store.loadTurns(store.activeSessionId, true);

    expect(store.activeSession?.turns[0].traceId).toBe("final-trace");
    expect(store.activeSession?.turns[0].clarification?.status).toBe("complex_scope_required");
    expect(store.activeSession?.turns[0].presentation?.facts?.[0].date_or_range).toBe("2018-08-22");
  });

  it("restores the safe premise checkpoint fields from server history", async () => {
    vi.mocked(api.chatTurns).mockResolvedValue({
      items: [{
        ...turnItem("paused"),
        status: "running",
        client_turn_id: "client-paused",
        interaction_state: "awaiting_human_input",
        interaction_reason_code: "premise_direction_mismatch",
        thread_id: "turn:durable-checkpoint",
        scope_key: '["seller-1","gmv","2018-08-01","2018-08-31","day"]',
        scope_fingerprint: "a".repeat(64),
        clarification: { status: "clarification_required", reason_code: "ambiguous_metric" },
      }],
      next_offset: null,
    });
    const store = useAppStore();
    store.createSession();
    await store.loadTurns(store.activeSessionId, true);

    expect(store.activeSession?.turns[0]).toMatchObject({
      id: "paused",
      status: "running",
      interactionState: "awaiting_human_input",
      interactionReasonCode: "premise_direction_mismatch",
      threadId: "turn:durable-checkpoint",
      clientTurnId: "client-paused",
      scopeKey: '["seller-1","gmv","2018-08-01","2018-08-31","day"]',
      scopeFingerprint: "a".repeat(64),
      clarification: { reason_code: "ambiguous_metric" },
    });
  });

  it("replaces turns on reset and keeps the active session valid", async () => {
    vi.mocked(api.chatTurns).mockResolvedValue({ items: [turnItem("a"), turnItem("b")], next_offset: null });
    const store = useAppStore();
    store.createSession();
    const id = store.activeSessionId;
    await store.loadTurns(id, true);
    vi.mocked(api.chatTurns).mockResolvedValue({ items: [turnItem("c")], next_offset: null });
    await store.loadTurns(id, true);
    expect(store.activeSession?.turns.map((t) => t.id)).toEqual(["c"]);
  });
});

describe("app store: deletion and auth", () => {
  it("switches the active session to the first remaining after deletion", async () => {
    vi.mocked(api.deleteChatSession).mockResolvedValue({ deleted: true });
    const store = useAppStore();
    store.createSession();
    store.createSession();
    const remaining = store.sessions[1].id;
    await store.deleteSession(store.sessions[0].id);
    expect(store.sessions).toHaveLength(1);
    expect(store.activeSessionId).toBe(remaining);
  });

  it("removes a turn from its session", async () => {
    vi.mocked(api.deleteChatTurn).mockResolvedValue({ deleted: true });
    const store = useAppStore();
    store.addTurn(turn("t1", "one"));
    store.addTurn(turn("t2", "two"));
    await store.deleteTurn(store.activeSessionId, "t1");
    expect(store.activeSession?.turns.map((t) => t.id)).toEqual(["t2"]);
  });

  it("login stores the bearer token and account and persists auth", async () => {
    vi.mocked(api.login).mockResolvedValue({
      access_token: "access-token",
      expires_at: "2026-01-02T00:00:00Z",
      profile: { user_id: "u1", display_name: "分析师", tenant_name: "租户", role: "admin", permissions: [] },
    });
    const store = useAppStore();
    await store.login("analyst@example.com", "secret");
    expect(store.bearerToken).toBe("access-token");
    expect(store.account?.role).toBe("admin");
    expect(localStorage.getItem("bizinsight.auth.v1")).toContain("access-token");
  });

  it("clears the bearer token on a 401 from /auth/me", async () => {
    vi.mocked(api.me).mockRejectedValue({ status: 401 });
    const store = useAppStore();
    store.bearerToken = "stale-token";
    await expect(store.refreshAccount()).rejects.toThrow();
    expect(store.bearerToken).toBe("");
    expect(store.account).toBeNull();
  });

  it("logout notifies the backend and clears local auth state", async () => {
    vi.mocked(api.logout).mockResolvedValue({ logged_out: true });
    const store = useAppStore();
    store.bearerToken = "token";
    await store.logout();
    expect(api.logout).toHaveBeenCalled();
    expect(store.bearerToken).toBe("");
    expect(localStorage.getItem("bizinsight.auth.v1")).toBeNull();
  });
});
