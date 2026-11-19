/** Keep currency values visually comparable across cards, tables, and charts. */
const investigationStopReasonLabels: Record<string, string> = {
  ENOUGH_EVIDENCE: "证据已足够，调查结束",
  NO_HIGH_VALUE_ACTION: "没有更有价值的可执行调查项",
  CAPABILITY_GAP: "当前数据能力不足，无法继续深入",
  DATA_INSUFFICIENT: "数据不足，无法可靠判断",
  BUDGET_EXHAUSTED: "已达到本次调查步数上限",
  USER_SCOPE_LIMIT: "已达到分析范围或下钻限制",
  AUTHORIZATION_BLOCKED: "权限校验未通过，已停止查询",
  INVALID_DATA: "数据校验未通过，已停止分析",
  EXECUTION_FAILED: "查询执行失败",
  COMPLETED: "调查结束",
  completed: "调查结束",
  insufficient_evidence: "证据不足，无法得出可靠结论",
  budget_exhausted: "已达到本次调查步数上限",
  policy: "安全策略阻止了此次查询",
};

/** Translate stable backend stop codes without hiding unknown diagnostic values. */
export function formatInvestigationStopReason(code: string | null | undefined): string {
  if (!code) return "";
  return investigationStopReasonLabels[code] ?? code;
}

export function formatMetricValue(value: number | null | undefined, unit = "", includeUnit = true): string {
  if (value === null || value === undefined) return "-";
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return "-";

  const isCurrency = unit === "BRL" || unit.startsWith("BRL/");
  const formatted = new Intl.NumberFormat("zh-CN", {
    minimumFractionDigits: isCurrency ? 2 : 0,
    maximumFractionDigits: 2,
  }).format(numericValue);

  return includeUnit && unit ? `${formatted} ${unit}` : formatted;
}
