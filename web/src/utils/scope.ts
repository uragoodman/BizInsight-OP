import type { ExplorerPreview, ExplorerQuery } from "@/types";

export const DEFAULT_ANALYSIS_RANGE = {
  start_date: "2018-08-01",
  end_date: "2018-08-31",
} as const;

/** Browser-side convenience only: the server remains the authority for scope validation. */
export function validateExplorerQuery(query: ExplorerQuery): string | null {
  if (!query.seller_id) return "请选择店铺。";
  if (!query.start_date || !query.end_date) return "请选择开始和结束日期。";
  if (query.end_date < query.start_date) return "结束日期不能早于开始日期。";
  const days = (Date.parse(`${query.end_date}T00:00:00Z`) - Date.parse(`${query.start_date}T00:00:00Z`)) / 86_400_000;
  if (days > 366) return "日期范围不能超过 366 天。";
  return null;
}

export function canSubmitExplorerScope(preview: ExplorerPreview | null, explicitConsent: boolean): boolean {
  return Boolean(explicitConsent && preview?.summary.status === "ok" && preview.context.locked);
}
