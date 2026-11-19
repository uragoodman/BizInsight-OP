import type { SemanticInterpretation } from "@/types";

const DIMENSION_LABELS: Record<string, string> = {
  product_category: "商品品类",
  customer_state: "客户州/地区",
  customer_city: "客户城市",
  seller_state: "卖家州/地区",
  payment_type: "支付方式",
  month_position: "月内时段",
  day_of_week: "星期",
  hour_of_day: "小时",
  quarter_of_year: "季度",
  half_of_year: "半年",
  seller: "店铺",
  seller_id: "店铺",
};

export function formatAnalysisDimensions(semantic: SemanticInterpretation | null | undefined): string {
  const dimensions = semantic?.dimensions?.filter(Boolean) ?? [];
  if (dimensions.length) {
    return dimensions.map((dimension) => DIMENSION_LABELS[dimension] ?? dimension).join(" × ");
  }
  return semantic?.dimension_label || semantic?.dimension || "";
}
