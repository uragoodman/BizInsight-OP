import { describe, expect, it } from "vitest";
import { unitLabel, unitSuffix } from "./metricUnits";

describe("registered comparison unit labels", () => {
  it("keeps percentage levels and relative rates distinct from percentage-point deltas", () => {
    expect(unitLabel("%")).toBe("%");
    expect(unitLabel("percentage_points")).toBe("百分点");
    expect(unitSuffix("percentage_points")).toBe(" 百分点");
  });

  it("renders registered non-ratio units and leaves unknown units unguessed", () => {
    expect(unitSuffix("BRL")).toBe(" 雷亚尔");
    expect(unitSuffix("orders")).toBe(" 单");
    expect(unitSuffix("BRL/order")).toBe(" 雷亚尔/单");
    expect(unitSuffix(null)).toBe("");
    expect(unitSuffix("custom-unit")).toBe(" custom-unit");
  });
});
