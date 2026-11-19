const DISPLAY_UNITS: Record<string, string> = {
  "%": "%",
  percentage_points: "百分点",
  BRL: "雷亚尔",
  "BRL/order": "雷亚尔/单",
  "items/order": "件/单",
  customers: "人",
  orders: "单",
  days: "天",
  hours: "小时",
  score: "分",
};

export function unitLabel(unit?: string | null): string {
  if (!unit) return "";
  return DISPLAY_UNITS[unit] ?? unit;
}

export function unitSuffix(unit?: string | null): string {
  const label = unitLabel(unit);
  return label ? ` ${label}` : "";
}
