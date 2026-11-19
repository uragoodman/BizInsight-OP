import { describe, expect, it } from "vitest";
import { clarificationMessage, clarificationTitle } from "./clarificationCopy";

describe("clarification copy", () => {
  it("asks for the missing contribution dimension", () => {
    const detail = { reason_code: "dimension_required" };
    expect(clarificationTitle(detail)).toBe("需要补充统计贡献维度");
    expect(clarificationMessage(detail)).toContain("按哪个维度展示统计贡献");
    expect(clarificationMessage(detail)).toContain("不会执行数据查询");
  });

  it("localizes the unsupported composite comparison code", () => {
    const detail = { reason: "comparison_dimension_not_supported", reason_code: "comparison_dimension_not_supported" };
    expect(clarificationTitle(detail)).toBe("复合比较暂不支持");
    expect(clarificationMessage(detail)).toContain("商品品类");
    expect(clarificationMessage(detail)).toContain("支付方式");
    expect(clarificationMessage(detail)).not.toContain("comparison_dimension_not_supported");
  });

  it("keeps readable legacy reasons and suppresses unknown machine codes", () => {
    expect(clarificationMessage({ reason: "请明确日期范围。" })).toBe("请明确日期范围。");
    expect(clarificationMessage({ reason_code: "new_internal_code", reason: "new_internal_code" })).toBe("未执行数据查询，请补充确认分析口径。");
  });
});
