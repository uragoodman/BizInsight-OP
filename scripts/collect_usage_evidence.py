"""Export provider-reported per-trace token/cost evidence for one tenant user."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from sqlalchemy import text

from app.data.database import engine

ROOT = Path(__file__).resolve().parents[1]


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * ratio + 0.999999) - 1))]


def collect(tenant_id: str, user_id: str, days: int = 30) -> dict:
    """Aggregate only provider-reported events; zero samples is unavailable."""
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', :id, true)"), {"id": tenant_id})
        conn.execute(text("SELECT set_config('app.user_id', :id, true)"), {"id": user_id})
        rows = conn.execute(text("""SELECT COALESCE(trace_id, event_id) AS trace_id,
              sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens,
              sum(total_tokens) AS total_tokens, sum(estimated_cost) AS estimated_cost
            FROM governance.llm_usage_events
            WHERE tenant_id=:tenant AND user_id=:user AND usage_source='provider_reported'
              AND created_at >= now() - (:days || ' days')::interval
            GROUP BY COALESCE(trace_id, event_id)"""), {"tenant": tenant_id, "user": user_id, "days": max(1, min(days, 365))}).mappings().all()
    traces = [dict(row) for row in rows]
    if not traces:
        return {"status": "unavailable", "reason": "no provider-reported usage events", "period_days": days, "traces": 0}
    def summary(key: str) -> dict:
        values = [float(row[key] or 0) for row in traces]
        return {"p50": round(median(values), 3), "p95": round(_percentile(values, 0.95) or 0, 3), "mean": round(sum(values) / len(values), 3), "total": round(sum(values), 3)}
    return {"status": "available", "period_days": days, "traces": len(traces), "input_tokens": summary("input_tokens"), "output_tokens": summary("output_tokens"), "total_tokens": summary("total_tokens"), "estimated_cost": summary("estimated_cost")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", required=True); parser.add_argument("--user", required=True)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    result = {"captured_at": datetime.now(timezone.utc).isoformat(), "tenant_id": args.tenant, "user_id": args.user, **collect(args.tenant, args.user, args.days)}
    path = ROOT / "data" / "quality" / f"usage-{args.tenant}-{args.user}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
