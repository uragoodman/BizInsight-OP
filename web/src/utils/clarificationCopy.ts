export type ClarificationDetail = Record<string, unknown> | null | undefined;

const CODE_COPY: Record<string, { title: string; message: string }> = {
  dimension_required: {
    title: "需要补充统计贡献维度",
    message: "请明确要按哪个维度展示统计贡献。系统不会自行选择维度，也不会执行数据查询。",
  },
  ambiguous_dimension: {
    title: "需要明确分组维度",
    message: "问题中包含多个可能的分组维度。请在问题中明确只按一个维度拆解；系统没有自动选择，也没有执行数据查询。",
  },
  date_outside_locked_scope: {
    title: "请求日期超出当前范围",
    message: "比较期间超出本会话已锁定的日期范围。系统不会自动扩大范围；请先修改会话范围，再重新提问。",
  },
  comparison_dimension_not_supported: {
    title: "复合比较暂不支持",
    message: "当前问题同时要求按商品品类分组并比较支付方式；系统暂不支持一次查询完成这项复合比较。请改为单独查询，或先在数据浏览中选择一个维度。",
  },
};

const DEFAULT_TITLE = "需要确认分析口径";
const DEFAULT_MESSAGE = "未执行数据查询，请补充确认分析口径。";

function textField(detail: ClarificationDetail, field: string): string {
  const value = detail?.[field];
  return typeof value === "string" ? value.trim() : "";
}

function reasonCode(detail: ClarificationDetail): string {
  const explicit = textField(detail, "reason_code");
  if (explicit) return explicit;
  const legacyReason = textField(detail, "reason");
  return CODE_COPY[legacyReason] ? legacyReason : "";
}

export function clarificationTitle(detail: ClarificationDetail, hasSemanticContext = false): string {
  const code = reasonCode(detail);
  if (CODE_COPY[code]) return CODE_COPY[code].title;
  if (code === "unsupported_metric") return "当前范围缺少部分指标";
  if (code === "unsupported_dimension") return "当前维度暂不支持";
  if (hasSemanticContext) return "请确认系统对问题的理解";
  return DEFAULT_TITLE;
}

export function clarificationMessage(detail: ClarificationDetail, fallback = DEFAULT_MESSAGE): string {
  const code = reasonCode(detail);
  if (CODE_COPY[code]) return CODE_COPY[code].message;
  const reason = textField(detail, "reason");
  // Older payloads sometimes put an opaque code in `reason`. Never show that
  // code as prose when no localized mapping is available.
  if (!reason || /^[a-z][a-z0-9_]*$/.test(reason)) return fallback;
  return reason;
}
