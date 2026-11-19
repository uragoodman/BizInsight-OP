import { describe, expect, it } from "vitest";
import { canSubmitExplorerScope, validateExplorerQuery } from "./scope";
import { formatInvestigationStopReason, formatMetricValue } from "./format";
import type { ExplorerPreview } from "@/types";

const preview: ExplorerPreview = { context: { seller_id: "seller-1", metric_name: "gmv", start_date: "2018-08-01", end_date: "2018-08-10", granularity: "day", locked: true, scope: "platform" }, summary: { metric_name: "gmv", value: 10, unit: "BRL", status: "ok" }, series: [], table: { columns: [], rows: [] }, lineage: { source_table: "analytics.fact_order_items", query_time: "server", latest_data_date: "2018-08-10" } };

describe("analysis scope", () => {
  it("rejects missing seller, reversed dates, and oversized ranges", () => {
    expect(validateExplorerQuery({ seller_id: null, metric_name: "gmv", start_date: "2018-08-01", end_date: "2018-08-10", granularity: "day" })).toContain("店铺");
    expect(validateExplorerQuery({ seller_id: "s", metric_name: "gmv", start_date: "2018-08-10", end_date: "2018-08-01", granularity: "day" })).toContain("结束日期");
    expect(validateExplorerQuery({ seller_id: "s", metric_name: "gmv", start_date: "2017-01-01", end_date: "2018-01-03", granularity: "day" })).toContain("366");
  });
  it("requires an explicit user choice and a successful locked server preview", () => {
    expect(canSubmitExplorerScope(preview, false)).toBe(false);
    expect(canSubmitExplorerScope({ ...preview, summary: { ...preview.summary, status: "not_found" } }, true)).toBe(false);
    expect(canSubmitExplorerScope(preview, true)).toBe(true);
  });
  it("accepts an exactly-366-day range and fails closed without consent or lock", () => {
    expect(validateExplorerQuery({ seller_id: "s", metric_name: "gmv", start_date: "2017-01-01", end_date: "2018-01-02", granularity: "day" })).toBeNull();
    expect(canSubmitExplorerScope(null, true)).toBe(false);
    expect(canSubmitExplorerScope({ ...preview, context: { ...preview.context, locked: false } }, true)).toBe(false);
  });
});

describe("metric display formatting", () => {
  it("keeps BRL and BRL/order values at two decimal places everywhere", () => {
    expect(formatMetricValue(52, "BRL/order")).toBe("52.00 BRL/order");
    expect(formatMetricValue(52.5, "BRL/order")).toBe("52.50 BRL/order");
    expect(formatMetricValue(52.678, "BRL")).toBe("52.68 BRL");
    expect(formatMetricValue(12, "orders")).toBe("12 orders");
  });
  it("renders null, undefined, and non-finite values as a dash", () => {
    expect(formatMetricValue(null, "BRL")).toBe("-");
    expect(formatMetricValue(undefined, "BRL")).toBe("-");
    expect(formatMetricValue(Number.NaN)).toBe("-");
  });
  it("supports negatives, thousands separators, and unit suppression", () => {
    expect(formatMetricValue(-52.5, "BRL")).toBe("-52.50 BRL");
    expect(formatMetricValue(1000, "orders")).toBe("1,000 orders");
    expect(formatMetricValue(52, "BRL", false)).toBe("52.00");
  });
});

describe("investigation stop reason formatting", () => {
  it("translates structured backend codes and legacy values for chat progress", () => {
    expect(formatInvestigationStopReason("CAPABILITY_GAP")).toBe("当前数据能力不足，无法继续深入");
    expect(formatInvestigationStopReason("BUDGET_EXHAUSTED")).toBe("已达到本次调查步数上限");
    expect(formatInvestigationStopReason("insufficient_evidence")).toBe("证据不足，无法得出可靠结论");
  });
  it("preserves unknown diagnostics and omits empty values", () => {
    expect(formatInvestigationStopReason("FUTURE_CODE")).toBe("FUTURE_CODE");
    expect(formatInvestigationStopReason(null)).toBe("");
  });
});
