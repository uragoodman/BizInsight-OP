import { effectScope, nextTick, ref } from "vue";
import { afterEach, describe, expect, it } from "vitest";
import type { ExplorerQuery, SignedSellerScopePayload } from "@/types";
import { sellerScopeChatCarrier, sellerScopePayloadExpired, useSellerScopeCredential } from "./sellerScopeCredential";

const payload = (expiresAt = "2099-01-01T00:00:00Z"): SignedSellerScopePayload => ({
  seller_scope_intent: {
    seller_scope_type: "all", seller_ids: [], start_date: "2018-01-01", end_date: "2018-01-31",
    metric_ids: ["gmv"], dimensions: ["seller"], granularity: "day", include_unattributed: false,
  },
  provenance: { source: "explorer_selector", token: "signed-token", expires_at: expiresAt },
});

const legacyScope: ExplorerQuery = {
  seller_id: "seller-1", metric_name: "gmv", start_date: "2018-01-01", end_date: "2018-01-31", granularity: "day",
};

describe("seller scope credential behavior", () => {
  const scopes: Array<ReturnType<typeof effectScope>> = [];
  afterEach(() => { scopes.splice(0).forEach((scope) => scope.stop()); });

  it("clears the signed token when granularity changes", async () => {
    const granularity = ref("day");
    const sessionId = ref<string | null>("session-1");
    const scope = effectScope(); scopes.push(scope);
    const credential = scope.run(() => useSellerScopeCredential(() => [granularity.value], sessionId))!;
    credential.bindToSession(sessionId.value);
    credential.setPayload(payload());
    granularity.value = "month";
    await nextTick();
    expect(credential.payload.value).toBeNull();
  });

  it("clears the signed token when ALL confirmation is cancelled", async () => {
    const allConfirmed = ref(true);
    const sessionId = ref<string | null>("session-1");
    const scope = effectScope(); scopes.push(scope);
    const credential = scope.run(() => useSellerScopeCredential(() => [allConfirmed.value], sessionId))!;
    credential.bindToSession(sessionId.value);
    credential.setPayload(payload());
    allConfirmed.value = false;
    await nextTick();
    expect(credential.payload.value).toBeNull();
  });

  it("clears the signed token when the active chat session changes", async () => {
    const sessionId = ref<string | null>("session-1");
    const scope = effectScope(); scopes.push(scope);
    const credential = scope.run(() => useSellerScopeCredential(() => [], sessionId))!;
    credential.bindToSession(sessionId.value);
    credential.setPayload(payload());
    sessionId.value = "session-2";
    await nextTick();
    expect(credential.payload.value).toBeNull();
  });

  it("rejects expired tokens at submit carrier construction", () => {
    const expired = payload("2000-01-01T00:00:00Z");
    expect(sellerScopePayloadExpired(expired, Date.parse("2020-01-01T00:00:00Z"))).toBe(true);
    expect(() => sellerScopeChatCarrier(expired, legacyScope, Date.parse("2020-01-01T00:00:00Z")))
      .toThrow("seller_scope_expired");
  });

  it("sends either the seller payload or legacy Explorer scope, never both", () => {
    const signed = payload();
    expect(sellerScopeChatCarrier(signed, legacyScope)).toEqual({ seller_scope_payload: signed });
    expect(sellerScopeChatCarrier(null, legacyScope)).toEqual({
      use_current_explorer_context: true, explorer_context: legacyScope,
    });
  });
});
