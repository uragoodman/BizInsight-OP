import { ref, watch, type Ref } from "vue";
import type { ExplorerQuery, SignedSellerScopePayload } from "@/types";

export type SellerChatScopeCarrier =
  | { seller_scope_payload: SignedSellerScopePayload }
  | { use_current_explorer_context: boolean; explorer_context: ExplorerQuery | undefined };

export function sellerScopePayloadExpired(
  payload: SignedSellerScopePayload | null | undefined,
  now = Date.now(),
): boolean {
  return Boolean(payload && Date.parse(payload.provenance.expires_at) <= now);
}

export function sellerScopeChatCarrier(
  payload: SignedSellerScopePayload | null | undefined,
  legacyScope: ExplorerQuery | null | undefined,
  now = Date.now(),
): SellerChatScopeCarrier {
  if (payload) {
    if (sellerScopePayloadExpired(payload, now)) throw new Error("seller_scope_expired");
    return { seller_scope_payload: payload };
  }
  return {
    use_current_explorer_context: Boolean(legacyScope),
    explorer_context: legacyScope ? { ...legacyScope } : undefined,
  };
}

export function useSellerScopeCredential(
  scopeInputs: () => readonly unknown[],
  activeSessionId: Ref<string | null | undefined>,
) {
  const payload = ref<SignedSellerScopePayload | null>(null);
  const sessionId = ref<string | null>(null);
  const invalidate = () => {
    payload.value = null;
    sessionId.value = null;
  };

  watch(scopeInputs, invalidate);
  watch(activeSessionId, (currentId) => {
    if (payload.value && currentId !== sessionId.value) invalidate();
  });

  function bindToSession(currentId: string | null | undefined) {
    sessionId.value = currentId ?? null;
  }

  function setPayload(value: SignedSellerScopePayload) {
    payload.value = value;
  }

  return { payload, sessionId, bindToSession, setPayload, invalidate };
}
