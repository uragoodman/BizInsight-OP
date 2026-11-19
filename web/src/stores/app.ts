// ============================================================
// 全局应用状态（Pinia store）
// 作用：集中管理调试会话列表、当前活动会话、认证身份（bearerToken/account）、
//       数据浏览预览，以及这些状态在 localStorage 的持久化与后端 API 的同步。
// ============================================================
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import type { ApiAuth } from "@/api";
import { api, AUTH_EXPIRED_EVENT } from "@/api";
import type { DebugSession, DebugTurn, ExplorerPreview, ExplorerQuery } from "@/types";

// localStorage 持久化键：调试身份、调试会话、登录令牌。
const PROFILE_KEY = "bizinsight.debug-profile.v1";
const SESSIONS_KEY = "bizinsight.debug-sessions.v1";
const AUTH_KEY = "bizinsight.auth.v1";
// 登录后的账户信息（组织/角色/权限/店铺授权），用于前端展示与权限提示。
type Account = { tenantName: string; role: string; permissions: string[]; sellerGrants: Array<{ seller_id: string; start_date?: string | null; end_date?: string | null }> };
// 持久化到 localStorage 的认证快照。
type StoredAuth = { bearerToken: string; account: Account | null; expiresAt?: string | null };
// 生成唯一 ID（优先 crypto.randomUUID，降级到时间戳 + 随机数）。
const uid = (prefix: string) => `${prefix}-${crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`}`;
// 安全读取 localStorage 的 JSON；解析失败时回退到默认值。
const read = <T>(key: string, fallback: T): T => { try { return (JSON.parse(localStorage.getItem(key) ?? "") as T) ?? fallback; } catch { return fallback; } };

function jwtExpiresAt(token: string): string | null {
  try {
    const payload = token.split(".")[1];
    if (!payload) return null;
    const normalized = payload.replace(/-/g, "+").replace(/_/g, "/");
    const claims = JSON.parse(atob(normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "="))) as { exp?: unknown };
    return typeof claims.exp === "number" ? new Date(claims.exp * 1000).toISOString() : null;
  } catch {
    return null;
  }
}

export const useAppStore = defineStore("app", () => {
  const profile = ref(read<{ id: string; displayName: string; createdAt: string } | null>(PROFILE_KEY, null) ?? { id: uid("debug-user"), displayName: "本地分析员", createdAt: new Date().toISOString() });
  const persisted = read<{ activeSessionId?: string; sessions?: DebugSession[] }>(SESSIONS_KEY, {});
  const sessions = ref<DebugSession[]>(persisted.sessions ?? []);
  const activeSessionId = ref(persisted.activeSessionId ?? "");
  const storedAuth = read<StoredAuth | null>(AUTH_KEY, null);
  const analysisToken = ref("");
  // bearerToken 特意只在本浏览器/源内持久化：普通刷新不会登出本地控制台用户。
  // 登出或 /auth/me 返回 401 时都会清空它（见 clearAuth）。
  const bearerToken = ref(storedAuth?.bearerToken ?? "");
  const authExpiresAt = ref<string | null>(storedAuth?.expiresAt ?? jwtExpiresAt(bearerToken.value));
  const account = ref<Account | null>(storedAuth?.account ?? null);
  const preview = ref<ExplorerPreview | null>(null);
  const nextSessionsOffset = ref<number | null>(0);
  const nextTurnsOffset = ref<Record<string, number | null>>({});
  const auth = computed<ApiAuth>(() => ({ actor: profile.value.id, analysisToken: analysisToken.value, bearerToken: bearerToken.value }));
  const activeSession = computed(() => sessions.value.find((session) => session.id === activeSessionId.value));
  function saveSessions() { localStorage.setItem(PROFILE_KEY, JSON.stringify(profile.value)); localStorage.setItem(SESSIONS_KEY, JSON.stringify({ activeSessionId: activeSessionId.value, sessions: sessions.value.slice(0, 20).map((session) => ({ ...session, turns: session.turns.slice(0, 40) })) })); }
  // 把认证状态写入 localStorage（有令牌则存，无则清除该键）。
  function saveAuth() { if (bearerToken.value) localStorage.setItem(AUTH_KEY, JSON.stringify({ bearerToken: bearerToken.value, account: account.value, expiresAt: authExpiresAt.value })); else localStorage.removeItem(AUTH_KEY); }
  // 清空本地认证状态并同步删除持久化记录。
  function clearAuth() { bearerToken.value = ""; authExpiresAt.value = null; account.value = null; saveAuth(); }
  function isSessionValid(): boolean {
    if (!bearerToken.value) return false;
    if (!authExpiresAt.value) return true;
    const expiresAt = Date.parse(authExpiresAt.value);
    return !Number.isFinite(expiresAt) || expiresAt > Date.now();
  }
  function expireSession(): void {
    if (!bearerToken.value) return;
    clearAuth();
    if (typeof window !== "undefined") window.dispatchEvent(new CustomEvent(AUTH_EXPIRED_EVENT));
  }
  function ensureSessionValid(): boolean {
    if (isSessionValid()) return true;
    expireSession();
    return false;
  }
  // 新建一个空的调试会话并置为活动会话。
  function createSession() { const now = new Date().toISOString(); const session: DebugSession = { id: uid("session"), title: "新的调试会话", createdAt: now, updatedAt: now, turns: [], scope: null }; sessions.value.unshift(session); activeSessionId.value = session.id; saveSessions(); return session; }
  // 追加或覆盖一条对话轮次；首条轮次用其内容作为会话标题。
  function addTurn(turn: DebugTurn) { const session = activeSession.value ?? createSession(); const index = session.turns.findIndex((item) => item.id === turn.id); if (index >= 0) session.turns[index] = turn; else session.turns.unshift(turn); if (session.turns.length === 1) session.title = turn.message.slice(0, 28) || session.title; session.updatedAt = new Date().toISOString(); sessions.value.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)); saveSessions(); }
  // 本地设置数据浏览范围，保留同步 setter 供纯前端状态操作使用。
  function setActiveSessionScope(scope: ExplorerQuery | null) {
    const session = activeSession.value ?? createSession();
    session.scope = scope ? { ...scope } : null;
    session.scopeFingerprint = undefined;
    session.updatedAt = new Date().toISOString();
    saveSessions();
  }
  // 保存到服务端，确保切换页面或刷新后仍使用这次选择的范围。
  async function saveActiveSessionScope(scope: ExplorerQuery | null) {
    const session = activeSession.value ?? createSession();
    const previousScope = session.scope ? { ...session.scope } : null;
    const previousScopeFingerprint = session.scopeFingerprint;
    const previousUpdatedAt = session.updatedAt;
    setActiveSessionScope(scope);
    try {
      const persisted = await api.updateChatSessionScope(auth.value, session.id, session.scope ?? null);
      session.scope = persisted.scope ? { ...persisted.scope } : null;
      session.scopeFingerprint = persisted.scope_fingerprint ?? undefined;
      session.updatedAt = persisted.updated_at;
      saveSessions();
    } catch (error) {
      session.scope = previousScope;
      session.scopeFingerprint = previousScopeFingerprint;
      session.updatedAt = previousUpdatedAt;
      saveSessions();
      throw error;
    }
  }
  // 清空数据浏览预览。
  function clearPreview() { preview.value = null; }
  // 登录：换取 bearerToken，更新本地身份 profile 与账户信息，并持久化认证状态。
  async function login(email: string, password: string) { const result = await api.login(email, password); bearerToken.value = result.access_token; authExpiresAt.value = result.expires_at ?? jwtExpiresAt(result.access_token); profile.value = { id: result.profile.user_id, displayName: result.profile.display_name, createdAt: new Date().toISOString() }; account.value = { tenantName: result.profile.tenant_name, role: result.profile.role, permissions: result.profile.permissions, sellerGrants: [] }; saveSessions(); saveAuth(); }
  // 刷新账户信息：重新请求 /auth/me；返回 401 视为登录失效并清空认证。
  async function refreshAccount() { if (!ensureSessionValid()) throw new Error("登录会话已过期，请重新登录"); try { const result = await api.me(auth.value); profile.value = { id: result.user_id, displayName: result.display_name, createdAt: profile.value.createdAt }; account.value = { tenantName: result.tenant_name, role: result.role, permissions: result.permissions, sellerGrants: result.seller_grants }; saveSessions(); saveAuth(); } catch (error) { const status = typeof error === "object" && error && "status" in error ? Number((error as { status?: number }).status) : 0; if (status === 401) clearAuth(); throw error; } }
  // 退出登录：通知后端注销（失败也忽略），并清空本地认证状态。
  async function logout() { if (bearerToken.value) await api.logout(auth.value).catch(() => undefined); clearAuth(); }
  function mapTurn(item: { id: string; client_turn_id?: string; message: string; answer: string | null; task_type: string; trace_id: string; status?: string; interaction_state?: DebugTurn["interactionState"] | "none"; interaction_reason_code?: DebugTurn["interactionReasonCode"]; thread_id?: string; scope_key?: string; scope_fingerprint?: string; clarification?: DebugTurn["clarification"]; presentation?: DebugTurn["presentation"] | null; created_at: string }): DebugTurn { return { id: item.id, threadId: item.thread_id ?? "", clientTurnId: item.client_turn_id, message: item.message, answer: item.answer ?? undefined, taskType: item.task_type, traceId: item.trace_id, createdAt: item.created_at, status: item.status ?? "completed", interactionState: item.interaction_state && item.interaction_state !== "none" ? item.interaction_state : undefined, interactionReasonCode: item.interaction_reason_code, clarification: item.clarification ?? null, presentation: item.presentation ?? undefined, scopeKey: item.scope_key, scopeFingerprint: item.scope_fingerprint }; }
  async function loadSessions(reset = false) {
    const offset = reset ? 0 : nextSessionsOffset.value;
    if (offset === null) return;
    // A reset fetch can overlap with the user creating a fresh local session
    // (for example, while ChatView's initial history request is still in
    // flight). Keep only sessions created after this request began; older
    // local-only sessions are still replaced by the server's authoritative
    // history below.
    const sessionIdsAtRequestStart = reset ? new Set(sessions.value.map((session) => session.id)) : null;
    const page = await api.chatSessions(auth.value, offset);
    const loaded = page.items.map((item) => ({ id: item.id, title: item.title, createdAt: item.created_at, updatedAt: item.updated_at, turns: [], scope: item.scope ?? null, scopeFingerprint: item.scope_fingerprint ?? undefined }));
    if (reset) {
      const loadedIds = new Set(loaded.map((session) => session.id));
      const createdDuringRequest = sessions.value.filter((session) =>
        !sessionIdsAtRequestStart!.has(session.id) && !loadedIds.has(session.id),
      );
      sessions.value = [...createdDuringRequest, ...loaded];
    } else {
      sessions.value = [...sessions.value.filter((s) => !loaded.some((x) => x.id === s.id)), ...loaded];
    }
    nextSessionsOffset.value = page.next_offset;
    if (!sessions.value.some((session) => session.id === activeSessionId.value)) activeSessionId.value = sessions.value[0]?.id ?? "";
    // The server is authoritative for persisted chat history. Persist the
    // reconciled list so stale local-only sessions cannot reappear after
    // leaving and returning to the chat page.
    if (reset) saveSessions();
  }
  async function loadTurns(sessionId: string, reset = false) {
    const offset = reset ? 0 : nextTurnsOffset.value[sessionId] ?? 0;
    if (offset === null) return;
    const page = await api.chatTurns(auth.value, sessionId, offset);
    const session = sessions.value.find((item) => item.id === sessionId);
    if (!session) return;
    const loaded = page.items.map(mapTurn);
    session.turns = reset ? loaded : [...session.turns, ...loaded.filter((turn) => !session.turns.some((old) => old.id === turn.id))];
    nextTurnsOffset.value[sessionId] = page.next_offset;
  }
  async function deleteSession(sessionId: string) { await api.deleteChatSession(auth.value, sessionId); sessions.value = sessions.value.filter((s) => s.id !== sessionId); if (activeSessionId.value === sessionId) activeSessionId.value = sessions.value[0]?.id ?? ""; }
  async function deleteTurn(sessionId: string, turnId: string) { await api.deleteChatTurn(auth.value, sessionId, turnId); const session = sessions.value.find((item) => item.id === sessionId); if (session) session.turns = session.turns.filter((turn) => turn.id !== turnId); }
  return { profile, account, sessions, activeSessionId, activeSession, analysisToken, bearerToken, preview, auth, createSession, addTurn, saveSessions, setActiveSessionScope, saveActiveSessionScope, clearPreview, login, logout, refreshAccount, clearAuth, ensureSessionValid, isSessionValid, nextSessionsOffset, nextTurnsOffset, loadSessions, loadTurns, deleteSession, deleteTurn };
});
