"""多租户越权隔离压力测试（受控链路真实拦截验证）。

对受控链路构造多类越权尝试，验证全部被应用层 scope 校验拦截，并如实记录
PostgreSQL RLS 在本连接（postgres superuser）下的边界——superuser 默认绕过 RLS，
因此本系统实际生效的主防线是应用层锁定的 ControlledAnalysisScope + SQL 强制过滤。

用法：python -m scripts.pressure_test_tenancy_isolation
"""
from __future__ import annotations

import json
from datetime import date
from typing import Callable

from sqlalchemy import text

from app.data.database import engine
from app.data.repositories.explorer import (
    get_accessible_seller,
    preview_explorer,
    query_scoped_metric,
)
from app.mcp.client import validate_periods
from app.schemas.explorer import ControlledAnalysisScope, ExplorerQuery
from app.tenancy import SELLER_SCOPE, SellerGrant, TenantContext

# 种子数据中真实存在且有数据的 seller；B 为非授权 seller。
SELLER_A = "6560211a19b47992c3666cc44a7e94c0"
SELLER_B = "6560211a00000000000000000000000000"


def _seller_ctx(seller_id: str = SELLER_A, start: str | None = None, end: str | None = None) -> TenantContext:
    grants = [SellerGrant(seller_id=seller_id, start_date=start, end_date=end)] if start else []
    return TenantContext(
        tenant_id="t-a", user_id="u-a", role="analyst",
        scope_type=SELLER_SCOPE, seller_id=seller_id, seller_grants=grants,
    )


def _scope(
    seller_id: str,
    start: str = "2018-08-01",
    end: str = "2018-08-31",
    metrics: tuple[str, ...] = ("gmv",),
    max_span: int = 3650,
) -> ControlledAnalysisScope:
    return ControlledAnalysisScope(
        seller_id=seller_id,
        allowed_metrics=metrics,
        default_start_date=date.fromisoformat(start),
        default_end_date=date.fromisoformat(end),
        metric_name="gmv",
        granularity="day",
        max_date_span_days=max_span,
    )


def _rls_visibility() -> dict[str, object]:
    """记录 RLS 在本连接的实测行为：superuser 默认绕过 RLS。"""
    with engine.connect() as connection:
        connection.execute(text("SELECT set_config('app.tenant_id', 'dv-a-1598be', true)"))
        connection.execute(text("SELECT set_config('app.scope_type', 'seller', true)"))
        rows = connection.execute(text(
            "SELECT COUNT(*) AS n, COUNT(DISTINCT tenant_id) AS tenants FROM analytics.tenant_fact_order_items"
        )).mappings().one()
    return {"visible_rows": rows["n"], "visible_tenants": rows["tenants"], "expected_tenant_rows": 1}


CASES: list[tuple[str, Callable[[], object]]] = [
    ("跨店铺直接访问", lambda: get_accessible_seller(_seller_ctx(), SELLER_B)),
    (
        "跨店铺预览查询",
        lambda: preview_explorer(
            _seller_ctx(),
            ExplorerQuery(seller_id=SELLER_B, metric_name="gmv", start_date=date(2018, 8, 1), end_date=date(2018, 8, 31)),
        ),
    ),
    (
        "日期超出授权",
        lambda: preview_explorer(
            _seller_ctx(start="2018-08-01", end="2018-08-31"),
            ExplorerQuery(seller_id=SELLER_A, metric_name="gmv", start_date=date(2018, 9, 1), end_date=date(2018, 9, 30)),
        ),
    ),
    ("指标越权", lambda: query_scoped_metric(_scope(SELLER_A, metrics=("gmv",)), "order_count")),
    ("查询范围超限", lambda: validate_periods(
        [{"start_date": date(2018, 1, 1), "end_date": date(2018, 12, 31)}],
        _scope(SELLER_A, max_span=90),
    )),
]


def main() -> None:
    results = []
    blocked = 0
    for name, attempt in CASES:
        try:
            outcome = attempt()
        except ValueError as exc:
            blocked += 1
            results.append({"case": name, "blocked": True, "reason": str(exc)})
            continue
        if isinstance(outcome, str) and outcome:  # validate_periods 返回拒绝原因
            blocked += 1
            results.append({"case": name, "blocked": True, "reason": outcome})
        else:
            results.append({"case": name, "blocked": False, "reason": None})

    # 数据隔离正面验证：受控路径返回授权 seller 的指标值。
    value = query_scoped_metric(_scope(SELLER_A), "gmv")
    isolation = {"metric": "gmv", "seller_id": SELLER_A, "status": value.get("status"), "value": value.get("value")}

    report = {
        "cases": results,
        "blocked": blocked,
        "total": len(CASES),
        "interception_rate": round(blocked / len(CASES), 4),
        "isolation_positive": isolation,
        "rls_note": _rls_visibility(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if blocked != len(CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
