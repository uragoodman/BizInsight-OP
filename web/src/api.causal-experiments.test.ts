import { afterEach, describe, expect, it, vi } from "vitest";
import { api, type ApiAuth } from "./api";

const auth: ApiAuth = { actor: "experiment-user", analysisToken: "", bearerToken: "signed-token" };
const scope = { seller_id: "seller-1", start_date: "2026-01-01", end_date: "2026-01-31" };

afterEach(() => vi.unstubAllGlobals());

describe("randomized experiment API client", () => {
  it("lists server-authorized verified experiments for the selected scope", async () => {
    const result = { items: [], truncated: false, scope };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(result), { status: 200 })));
    await expect(api.eligibleCausalExperiments(auth, scope)).resolves.toEqual(result);
    expect(fetch).toHaveBeenCalledWith("/causal-experiments/eligible", expect.objectContaining({
      method: "POST",
      body: JSON.stringify(scope),
      headers: expect.objectContaining({ Authorization: "Bearer signed-token" }),
    }));
  });

  it("submits only the selected opaque experiment handle and scope", async () => {
    const payload = { ...scope, experiment_id: "registered-exp-1" };
    const result = { status: "ok", causal_effect_estimated: true, effect: 2.5 };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(result), { status: 200 })));
    await expect(api.estimateCausalExperiment(auth, payload)).resolves.toEqual(result);
    const [, init] = vi.mocked(fetch).mock.calls[0];
    expect(init?.body).toBe(JSON.stringify(payload));
    expect(init?.body).not.toContain("outcome_metric_id");
  });

  it("preserves server-side authorization failures for the UI", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "tool not authorized" }), { status: 403 })));
    await expect(api.eligibleCausalExperiments(auth, scope)).rejects.toMatchObject({ status: 403, message: "tool not authorized" });
  });
});
