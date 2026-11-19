"""Best-effort, tenant-isolated accounting for provider-reported LLM usage."""
from __future__ import annotations

from datetime import datetime, timezone
from secrets import token_hex
from typing import Any

from sqlalchemy import text

from app.data.database import engine


def record_usage(*, tenant_id: str, user_id: str, trace_id: str | None,
                 chat_session_id: str | None, operation: str, model: str,
                 input_tokens: int, output_tokens: int, estimated_cost: float,
                 task_id: str | None = None, route_profile: str = "standard",
                 provider_id: str | None = None, fallback_from: str | None = None,
                 evaluation_run_id: str | None = None) -> None:
    """Store one provider-reported call without making analysis unavailable.

    `provider_id` 记录实际命中的提供商；`fallback_from` 记录因可恢复故障切走的主提供商，
    供质量仪表盘按 provider/回退次数分桶。If accounting storage is temporarily
    unavailable, the trace/runtime budget still contains the call; user analysis
    must not fail merely due to billing telemetry.
    """
    if not tenant_id or not user_id or input_tokens < 0 or output_tokens < 0:
        return
    if input_tokens + output_tokens == 0:
        return  # Provider did not supply usage; never invent token precision.
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT set_config('app.tenant_id', :value, true)"), {"value": tenant_id})
            conn.execute(text("SELECT set_config('app.user_id', :value, true)"), {"value": user_id})
            conn.execute(text("""INSERT INTO governance.llm_usage_events
              (event_id,tenant_id,user_id,trace_id,chat_session_id,task_id,evaluation_run_id,operation,model,route_profile,
               provider_id,fallback_from,input_tokens,output_tokens,total_tokens,estimated_cost,usage_source)
              VALUES(:event_id,:tenant_id,:user_id,:trace_id,:chat_session_id,:task_id,:evaluation_run_id,:operation,:model,:route_profile,
                     :provider_id,:fallback_from,:input_tokens,:output_tokens,:total_tokens,:estimated_cost,'provider_reported')"""), {
                "event_id": f"llu_{token_hex(12)}", "tenant_id": tenant_id, "user_id": user_id,
                "trace_id": trace_id, "chat_session_id": chat_session_id, "operation": operation[:32],
                "task_id":task_id,"evaluation_run_id": evaluation_run_id, "route_profile":route_profile,
                "provider_id": (provider_id or "")[:32], "fallback_from": (fallback_from or "")[:32],
                "model": model[:160], "input_tokens": input_tokens, "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens, "estimated_cost": max(0.0, estimated_cost),
            })
    except Exception:
        return


def usage_overview(*, tenant_id: str, user_id: str, days: int = 30, limit: int = 20) -> dict[str, Any]:
    days = max(1, min(int(days), 365)); limit = max(1, min(int(limit), 100))
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', :value, true)"), {"value": tenant_id})
        conn.execute(text("SELECT set_config('app.user_id', :value, true)"), {"value": user_id})
        summary = conn.execute(text("""SELECT COALESCE(sum(input_tokens),0) AS input_tokens,
          COALESCE(sum(output_tokens),0) AS output_tokens, COALESCE(sum(total_tokens),0) AS total_tokens,
          COALESCE(sum(estimated_cost),0) AS estimated_cost, count(*) AS calls
          FROM governance.llm_usage_events WHERE tenant_id=:tenant AND user_id=:user
          AND created_at >= now() - (:days || ' days')::interval"""), {"tenant": tenant_id, "user": user_id, "days": days}).mappings().one()
        items = conn.execute(text("""SELECT event_id,trace_id,chat_session_id,operation,model,input_tokens,
          output_tokens,total_tokens,estimated_cost,usage_source,created_at
          FROM governance.llm_usage_events WHERE tenant_id=:tenant AND user_id=:user
          ORDER BY created_at DESC LIMIT :limit"""), {"tenant": tenant_id, "user": user_id, "limit": limit}).mappings().all()
    return {"period_days": days, "summary": dict(summary), "items": [dict(item) for item in items]}

def quota_allows(tenant_id: str, user_id: str) -> bool:
    """Fail closed only when an explicitly configured user quota is exhausted."""
    from app.core.config import get_settings
    s=get_settings()
    if not any((s.llm_user_daily_token_limit,s.llm_user_monthly_token_limit,s.llm_user_daily_cost_limit,s.llm_user_monthly_cost_limit)): return True
    try:
        daily=usage_overview(tenant_id=tenant_id,user_id=user_id,days=1,limit=1)["summary"]
        monthly=usage_overview(tenant_id=tenant_id,user_id=user_id,days=31,limit=1)["summary"]
        return not ((s.llm_user_daily_token_limit and int(daily["total_tokens"])>=s.llm_user_daily_token_limit) or (s.llm_user_monthly_token_limit and int(monthly["total_tokens"])>=s.llm_user_monthly_token_limit) or (s.llm_user_daily_cost_limit and float(daily["estimated_cost"])>=s.llm_user_daily_cost_limit) or (s.llm_user_monthly_cost_limit and float(monthly["estimated_cost"])>=s.llm_user_monthly_cost_limit))
    except Exception: return True
