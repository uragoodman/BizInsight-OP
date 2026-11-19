"""RLS tenant metric latency gate for a seeded PostgreSQL database."""
from __future__ import annotations
import os
from statistics import quantiles
from time import perf_counter

from app.data.repositories.tenant_metrics import query_seller_metric, query_seller_trend
from app.tenancy import SELLER_SCOPE, TenantContext


def _p95(samples: list[float]) -> float:
    return quantiles(samples, n=20)[-1] if len(samples) >= 20 else max(samples)


def main() -> None:
    context = TenantContext("seller-tenant-1", "seller-user-1", "analyst", SELLER_SCOPE, "seller-1")
    samples=[]
    for _ in range(20):
        started=perf_counter(); query_seller_metric(context,"sales_gmv","2018-08-01","2018-08-31"); samples.append((perf_counter()-started)*1000)
    trend=[]
    for _ in range(20):
        started=perf_counter(); query_seller_trend(context,"sales_gmv","2018-08-01","2018-08-31"); trend.append((perf_counter()-started)*1000)
    limit=float(os.getenv("PHASE7_TENANT_METRIC_P95_MS","5000")); result={"metric_p95_ms":round(_p95(samples),2),"trend_p95_ms":round(_p95(trend),2),"limit_ms":limit}
    print(result)
    if max(result["metric_p95_ms"],result["trend_p95_ms"]) > limit: raise SystemExit("Phase 7 tenant performance gate failed")


if __name__ == "__main__": main()
