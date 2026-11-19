import { afterEach, describe, expect, it, vi } from "vitest";
import { api, type ApiAuth } from "./api";
import type { InventorySnapshotBatchInput, InventorySnapshotBatchResult, InventorySnapshotResult } from "./types";

const auth: ApiAuth = {
  actor: "inventory-viewer",
  analysisToken: "",
  bearerToken: "signed-token",
};

afterEach(() => vi.unstubAllGlobals());

describe("inventory snapshot API", () => {
  it("posts the scoped query with authenticated headers and returns observed coverage metadata", async () => {
    const payload = {
      seller_id: "seller-1",
      start_date: "2026-09-01",
      end_date: "2026-09-25",
      as_of: "2026-09-25T23:59:59+00:00",
      quantity_basis: "available" as const,
      limit: 500,
    };
    const result: InventorySnapshotResult = {
      status: "ok",
      seller_id: "seller-1",
      as_of: payload.as_of,
      quantity_basis: "available",
      observed_product_warehouse_count: 1,
      truncated: false,
      items: [{
        warehouse_id: "warehouse-1",
        product_id: "product-1",
        snapshot_at: payload.as_of,
        snapshot_semantics: "daily_close",
        quantity: 12,
        quantity_unit: "unit",
        source_system: "erp",
        measurement_status: "observed",
      }],
      coverage_status: "unknown_without_authoritative_product_warehouse_catalog",
      missing_policy: "missing is not zero",
      source_table: "analytics.inventory_snapshots",
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(result), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })));

    await expect(api.inventorySnapshot(auth, payload)).resolves.toEqual(result);
    expect(fetch).toHaveBeenCalledWith(
      "/data-explorer/inventory-snapshots/query",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ Authorization: "Bearer signed-token" }),
        body: JSON.stringify(payload),
      }),
    );
  });

  it("posts an owner-submitted batch to the bounded ingestion endpoint", async () => {
    const batch: InventorySnapshotBatchInput = {
      source_system: "erp",
      source_batch_id: "batch-2026-09-25",
      rows: [{
        seller_id: "seller-1",
        warehouse_id: "warehouse-1",
        product_id: "product-1",
        snapshot_at: "2026-09-25T23:59:59+00:00",
        snapshot_semantics: "daily_close",
        on_hand_quantity: 12,
        available_quantity: 10,
        quantity_unit: "unit",
      }],
    };
    const result: InventorySnapshotBatchResult = {
      status: "ok",
      seller_id: "seller-1",
      source_system: "erp",
      source_batch_id: "batch-2026-09-25",
      received_rows: 1,
      inserted_rows: 1,
      duplicate_rows_ignored: 0,
      snapshot_semantics: ["daily_close"],
      quantity_units: ["unit"],
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(result), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })));

    await expect(api.ingestInventorySnapshotBatch(auth, batch)).resolves.toEqual(result);
    expect(fetch).toHaveBeenCalledWith(
      "/data-explorer/inventory-snapshots/ingest",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ Authorization: "Bearer signed-token" }),
        body: JSON.stringify(batch),
      }),
    );
  });

  it("preserves an API permission failure instead of presenting a successful empty snapshot", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Permission denied" }), {
      status: 403,
      headers: { "Content-Type": "application/json" },
    })));

    await expect(api.inventorySnapshot(auth, {
      seller_id: "seller-1",
      start_date: "2026-09-01",
      end_date: "2026-09-25",
      as_of: "2026-09-25T23:59:59+00:00",
      quantity_basis: "available",
      limit: 100,
    })).rejects.toEqual(expect.objectContaining({ message: "Permission denied", status: 403 }));
  });
});
