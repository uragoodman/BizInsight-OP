import { describe, expect, it } from "vitest";
import { formatAnalysisDimensions } from "./semanticInterpretation";

describe("formatAnalysisDimensions", () => {
  it("shows every cross-analysis axis with readable labels", () => {
    expect(formatAnalysisDimensions({ dimensions: ["customer_state", "product_category"] }))
      .toBe("客户州/地区 × 商品品类");
  });

  it("keeps the existing single-dimension presentation as a fallback", () => {
    expect(formatAnalysisDimensions({ dimension: "product_category", dimension_label: "商品品类" }))
      .toBe("商品品类");
  });
});
