"""可信的数据浏览查询：按店铺、时间和指标生成统一的预览快照。

本模块与 app/mcp/client.py 中的 query_scoped_* 函数共同构成"受控查询路径"：
- 用户在数据浏览界面筛选一次（店铺 + 日期范围 + 指标 + 粒度）
- 服务端在 preview_explorer 中重新查库，生成快照
- controlled_scope 将快照锁定为 ControlledAnalysisScope
- 后续 Agent 只能在这个范围内查数，不能跨店铺、超日期

关键设计决策：
1. 原始数据通过 raw.order_items JOIN analytics.fact_orders 获取——不通过
   daily_metrics 聚合表，因为聚合表只有日粒度，缺少按周/月/季/年分桶的灵活性。
2. 派生时间维度（周几、时段、季度、上半年/下半年）在 SQL 中实时计算，
   不需要给数据库加预计算字段。由于是实时计算，也只支持事实表的 5 个核心指标
   （GMV / 订单量 / 客单价 / 客户数 / 运费）。
3. 数据边界以实际查询源表为准，不与全局 daily_metrics 边界混用。
   see app/core/time_expression.py 中的 data_time_bounds() 对比。
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

from sqlalchemy import text

from app.data.database import engine
from app.data.rls import canonical_seller_query_transaction, seller_query_transaction
from app.data.repositories.order_allocation import (
    ALLOCATION_METHOD,
    ALLOCATION_NOTE,
    allocated_items,
    paired_allocated_items,
    require_reconciled,
)
from app.schemas.explorer import (
    CONTROLLED_ANALYSIS_DEFAULT_DIMENSIONS,
    ControlledAnalysisScope,
    ExplorerQuery,
)
from app.tenancy import (
    SELLER_SCOPE,
    TenantContext,
    activate_tenant_context,
    current_tenant_context,
)

# ============================================================================
# FROM / METRIC / BUCKET 三要素
#
# _FROM_CLAUSE：所有受控查询的基表 = raw.order_items × analytics.fact_orders
#   order_items 提供 seller_id（用于店铺 RLS）、price、freight_value；
#   fact_orders 提供 purchase_date、customer_unique_id、item_count、payment_value。
#   两者通过 order_id 关联。
#
# metric_sql()：参数化指标表达式。调用方必须传入与 _FROM_CLAUSE 一致的别名，
#   禁止在 CTE 外层复用——必须与 _FROM_CLAUSE 在同一查询层使用。
#
# _BUCKET_SQL：按粒度截断 purchase_date 的 date_trunc 表达式。
#   日 = 原始日期，周 = ISO 周一，月/季/年 = 自然边界。
# ============================================================================
_FROM_CLAUSE = """
    FROM analytics.v_delivered_order_items oi
    JOIN analytics.v_delivered_orders fo ON oi.order_id = fo.order_id
"""
_CORE_QUERY_SOURCE = "analytics.v_delivered_order_items + analytics.v_delivered_orders"
_CORE_QUERY_LINEAGE = "analytics.fact_order_items + analytics.fact_orders"


def metric_sql(metric_name: str, oi: str = "oi", fo: str = "fo") -> str:
    """返回给定指标的 SQL 聚合表达式，使用指定的表别名。

    注意：返回的表达式依赖 *oi* 和 *fo* 表别名在调用处的查询层中存在。
    禁止在 CTE 外层使用——必须与 ``_FROM_CLAUSE`` 在同一查询层。
    """
    _TEMPLATES = {
        "gmv": f"COALESCE(SUM({oi}.price + {oi}.freight_value), 0)",
        "order_count": f"COUNT(DISTINCT {oi}.order_id)",
        "avg_order_value": (
            f"CASE WHEN COUNT(DISTINCT {oi}.order_id) = 0 THEN 0 "
            f"ELSE ROUND((SUM({oi}.price + {oi}.freight_value) "
            f"/ COUNT(DISTINCT {oi}.order_id))::numeric, 2) END"
        ),
        "customer_count": f"COUNT(DISTINCT {fo}.customer_unique_id)",
        "freight_value": f"COALESCE(SUM({oi}.freight_value), 0)",
    }
    return _TEMPLATES[metric_name]
_UNITS = {"gmv": "BRL", "order_count": "orders", "avg_order_value": "BRL/order", "customer_count": "customers", "freight_value": "BRL", "payment_value": "BRL", "installment_order_share": "%", "avg_approval_lead_hours": "hours", "avg_dispatch_lead_hours": "hours", "avg_delivery_transit_hours": "hours", "on_time_delivery_rate": "%", "review_score_avg": "score", "bad_review_rate": "%", "seller_order_bad_review_rate": "%", "seller_items_per_order": "items/order", "seller_freight_to_gmv_rate": "%", "order_approval_rate": "%", "order_shipment_rate": "%", "order_final_delivery_rate": "%", "review_comment_rate": "%"}
_DOMAIN_METRICS = {"payment_value", "installment_order_share", "avg_approval_lead_hours", "avg_dispatch_lead_hours", "avg_delivery_transit_hours", "on_time_delivery_rate", "review_score_avg", "bad_review_rate", "seller_order_bad_review_rate", "seller_items_per_order", "seller_freight_to_gmv_rate", "order_approval_rate", "order_shipment_rate", "order_final_delivery_rate", "review_comment_rate"}
_BUCKET_SQL = {
    "day": "fo.purchase_date",
    "week": "date_trunc('week', fo.purchase_date)::date",
    "month": "date_trunc('month', fo.purchase_date)::date",
    "quarter": "date_trunc('quarter', fo.purchase_date)::date",
    "year": "date_trunc('year', fo.purchase_date)::date",
}

# ---------------------------------------------------------------------------
# 派生时间维度的 SQL 表达式 — 实时计算而非预存字段
#
# 这些维度不是数据库的列，而是在查询时通过 CASE WHEN 或 EXTRACT 实时计算的。
# 好处：加维度不需要改表结构、不需要重建 ETL，数据始终与源表一致。
# 代价：每天用同一订单表查不同 "天" 的表达式可能命中不同的索引分支，
#       query_time 的 HOUR 表达式涉及 timestamp，当前是逐行计算的——数据量
#       (10 万行量级) 尚可接受。
#
# 注意 month_position 不在本字典中。它由独立的 breakdown_scoped_month_position()
# 处理，以保证 月初 → 月中 → 月末 的固定排序不被通用路径的 value-sort 覆盖。
# ---------------------------------------------------------------------------
_TIME_DIMENSION_SQL = {
    "day_of_week": "CASE EXTRACT(ISODOW FROM fo.purchase_date) WHEN 1 THEN '周一' WHEN 2 THEN '周二' WHEN 3 THEN '周三' WHEN 4 THEN '周四' WHEN 5 THEN '周五' WHEN 6 THEN '周六' ELSE '周日' END",
    # CHR(58) 用于生成 ':'。SQLAlchemy 会把 ':00' 字面量误识别为命名绑定参数，
    # 所以不能直接写 '|| '':00' ''，只能用 CHR(58) 绕过。
    "hour_of_day": "LPAD(EXTRACT(HOUR FROM fo.order_purchase_timestamp)::text, 2, '0') || CHR(58) || '00'",
    "quarter_of_year": "'Q' || EXTRACT(QUARTER FROM fo.purchase_date)::text",
    "half_of_year": "CASE WHEN EXTRACT(MONTH FROM fo.purchase_date) <= 6 THEN '上半年' ELSE '下半年' END",
}


def _seller_label(seller_id: str, city: object | None, state: object | None) -> str:
    location = " · ".join(part for part in (str(city) if city else "", str(state) if state else "") if part)
    return f"{location}｜{seller_id[:8]}…" if location else seller_id


def _seller_rows(query: str, limit: int) -> list[dict[str, object]]:
    pattern = f"%{query.strip()}%"
    with engine.connect() as connection:
        rows = connection.execute(text(f"""
            SELECT oi.seller_id, s.seller_city, s.seller_state,
                   COUNT(DISTINCT oi.order_id) AS order_count,
                   MIN(fo.purchase_date) AS first_date, MAX(fo.purchase_date) AS last_date
            {_FROM_CLAUSE}
            LEFT JOIN raw.sellers s ON s.seller_id = oi.seller_id
            WHERE oi.seller_id ILIKE :pattern
            GROUP BY oi.seller_id, s.seller_city, s.seller_state
            ORDER BY order_count DESC, oi.seller_id ASC LIMIT :limit
        """), {"pattern": pattern, "limit": min(max(limit, 1), 50)}).mappings().all()
    return [{
        "seller_id": str(row["seller_id"]),
        "label": _seller_label(str(row["seller_id"]), row["seller_city"], row["seller_state"]),
        "city": row["seller_city"], "state": row["seller_state"],
        "order_count": int(row["order_count"] or 0),
        "first_date": str(row["first_date"]), "last_date": str(row["last_date"]),
    } for row in rows]


def search_sellers(context: TenantContext, query: str = "", limit: int = 20) -> list[dict[str, object]]:
    """返回当前身份可浏览的店铺，不暴露订单级数据。

    卖家范围查询的入口——平台管理员可以看到所有店铺，卖家只能看到自己的店。
    这是一个安全的"店铺模糊搜索"，不允许跨租户枚举。

    参数：
        context: 当前租户上下文（含身份类型和卖家 ID）。
        query: 店铺 ID 的部分匹配（任意位置，不区分大小写），空字符串返回所有店铺。
        limit: 返回条数上限，强制限制在 1~50。

    返回：
        按订单量降序排列的店铺列表，每项包含 seller_id、label、order_count、
        first_date、last_date。last_date 会被后续 preview_explorer 用于覆盖提示。
    """

    if context.scope_type == SELLER_SCOPE:
        allowed = context.allowed_seller_ids
        rows = []
        for seller_id in allowed:
            rows.extend(_seller_rows(seller_id, 1))
        return rows[:min(max(limit, 1), 50)]
    return _seller_rows(query, limit)


def get_accessible_seller(context: TenantContext, seller_id: str) -> dict[str, object]:
    """Resolve one seller through the same authorization boundary as search."""
    if context.scope_type == SELLER_SCOPE and seller_id not in context.allowed_seller_ids:
        raise ValueError("Seller tenant cannot access another seller")
    rows = _seller_rows(seller_id, 50)
    seller = next((item for item in rows if item["seller_id"] == seller_id), None)
    if seller is None:
        raise ValueError("Seller does not exist or has no reportable data")
    return seller


def _effective_query(context: TenantContext, query: ExplorerQuery) -> ExplorerQuery:
    """根据租户身份应用默认卖家（如果有），防止跨租户选店。"""
    if context.scope_type == SELLER_SCOPE:
        seller_id = query.seller_id or context.seller_id
        if not seller_id:
            raise ValueError("Select one of your authorized sellers")
        grant = context.seller_grant(seller_id)
        if grant is None:
            raise ValueError("You are not authorized for this seller")
        grant_start = date.fromisoformat(grant.start_date) if grant.start_date else None
        grant_end = date.fromisoformat(grant.end_date) if grant.end_date else None
        if grant_start and query.start_date < grant_start:
            raise ValueError("Requested start date is before your seller data entitlement")
        if grant_end and query.end_date > grant_end:
            raise ValueError("Requested end date is after your seller data entitlement")
        return query.model_copy(update={"seller_id": seller_id})
    if not query.seller_id:
        raise ValueError("seller_id is required for platform data browsing")
    return query


def preview_explorer(context: TenantContext, query: ExplorerQuery) -> dict[str, object]:
    """Run preview under the server-authenticated context passed by the API."""
    active = current_tenant_context()
    if active is not None:
        if (active.tenant_id, active.user_id) != (context.tenant_id, context.user_id):
            raise PermissionError("Explorer context does not match the active authenticated identity")
        return _preview_explorer_with_context(context, query)
    with activate_tenant_context(context):
        return _preview_explorer_with_context(context, query)


def _preview_explorer_with_context(context: TenantContext, query: ExplorerQuery) -> dict[str, object]:
    """在服务端重新查询店铺事实表，返回用于 UI 和 Agent 证据的同一快照。

    这是数据浏览页的核心查询函数，职责是：
    1. 验证卖家身份和请求参数（_effective_query）
    2. 查店铺的指标汇总值（summary = 整个时间范围的聚合）
    3. 查按粒度分桶的时间序列（series = 逐 bucket 的值）
    4. 查该店铺在事实表中的可用时间范围（freshness）
    5. 构造 coverage 字段，与实际数据边界对比，标记是否被截断

    整条路径执行一次事务，快照的一致性由同一个 connection 保证。
    返回的 snapshot 会被 controlled_scope() 锁定为 ControlledAnalysisScope，
    后续 Agent 不能超出此范围。

    数据边界问题：
        以前这里用 app.core.time_expression.data_time_bounds() 取 global 边界
        （查询 analytics.daily_metrics — 截至 2018-10-17）。
        但本函数实际查询的是 order_items JOIN fact_orders — 截至 2018-08-29。
        两个源表的覆盖范围不一致，不能混用。
        修复方式：freshness 查询 MIN/MAX 就用实际 join 的同一张表。

    时间范围重叠校验：
        freshness 查的不是全局 min/max，而是"这个店铺在这个表上有数据的区间"。
        如果客户端请求的日期与该区间没有重叠（例如请求 2020 年但店铺 2018 年后
        就没数据了），直接抛 ValueError，不让后续查询返回空结果给用户造成困惑。
    """

    effective = _effective_query(context, query)

    # 数据浏览页的可见性、店铺实现和能力均以指标注册中心为准。这样新增
    # 指标时，YAML 声明和受控 handler 成为同一份契约，而不是前端列表与
    # preview SQL 各维护一套白名单。
    from app.metrics.registry import get_registry

    scope_dict = effective.model_dump()
    scope_dict["default_start_date"] = scope_dict.pop("start_date")
    scope_dict["default_end_date"] = scope_dict.pop("end_date")
    scope = ControlledAnalysisScope.model_validate(scope_dict)
    registry = get_registry()
    definition = registry.get(effective.metric_name)
    if "explorer" not in definition.surfaces:
        raise ValueError(f"Metric {effective.metric_name!r} is not available in data explorer")
    if not registry.supports(effective.metric_name, "value", scope="seller") or not registry.supports(
        effective.metric_name, "trend", scope="seller"
    ):
        raise ValueError(f"Metric {effective.metric_name!r} has no seller preview implementation")

    summary = registry.execute(
        effective.metric_name, operation="value", scope="seller", seller_scope=scope,
    )
    trend = registry.execute(
        effective.metric_name, operation="trend", scope="seller", seller_scope=scope,
        start_date=effective.start_date, end_date=effective.end_date,
        granularity=effective.granularity,
    )
    points = trend.get("points", [])

    # 检查实际数据边界，修复时间预设与交付数据范围落差问题。
    # daily_metrics 最大日期到 2018-10-17，但 Explorer 实际查的
    # v_delivered_orders 只到 2018-08-29。当请求日期在此范围之外
    # 时，标记截断而不是无声返回空趋势。
    data_max: date | None = None
    data_min: date | None = None
    if not points or summary.get("value") == 0:
        from sqlalchemy import text
        from app.data.database import engine
        with seller_query_transaction(engine, scope) as conn:
            row = conn.execute(text(
                "SELECT MIN(purchase_date) AS first, MAX(purchase_date) AS last "
                "FROM analytics.v_delivered_order_items "
                "WHERE seller_id=:sid"
            ), {"sid": effective.seller_id}).mappings().one()
            data_min = row["first"]
            data_max = row["last"]

    truncated = False
    actual_start = effective.start_date
    actual_end = effective.end_date
    if data_max is not None and effective.end_date > data_max:
        actual_end = data_max
        truncated = True
    if data_min is not None and effective.start_date < data_min:
        actual_start = data_min
        truncated = True
    if truncated:
        # 在截断后的范围内重查 trend，避免前端拿到 0 数据点
        trend = registry.execute(
            effective.metric_name, operation="trend", scope="seller",
            seller_scope=scope,
            start_date=max(effective.start_date, data_min) if data_min else effective.start_date,
            end_date=min(effective.end_date, data_max) if data_max else effective.end_date,
            granularity=effective.granularity,
        )
        points = trend.get("points", [])

    return {
        "context": effective.model_dump(mode="json") | {"locked": True, "scope": context.scope_type},
        "summary": {"metric_name": effective.metric_name, "value": summary["value"], "unit": definition.unit, "status": summary["status"]},
        "series": points,
        "table": {"columns": [{"key": "bucket", "label": "时间", "kind": "date"}, {"key": "value", "label": effective.metric_name, "kind": "number", "unit": definition.unit}], "rows": points},
        "lineage": {"source_table": summary["source_table"], "query_time": "server", "latest_data_date": str(actual_end), "note": summary.get("lineage_note", definition.implementations["seller"].lineage_description)},
        "coverage": {"requested_start": effective.start_date.isoformat(), "requested_end": effective.end_date.isoformat(), "start_date": actual_start.isoformat(), "end_date": actual_end.isoformat(), "data_max_date": str(data_max or actual_end), "truncated": truncated},
    }

def controlled_scope(snapshot: dict[str, object]) -> ControlledAnalysisScope:
    """从服务端生成的可信预览快照中提取锁定范围。

    只信任 preview_explorer 生成的 snapshot，不信任浏览器传来的 payload。
    Agent 只能在这个锁定的 seller_id/start_date/end_date/metric_name/granularity
    范围内查数。scope 是不可变的——Agent 不能通过自然语言修改这些约束。

    对比：
        explorer 的 resolve_explorer_context 节点会生成 snapshot，
        然后调用本函数创建 scope，再写入 state["controlled_analysis_scope"]。
        后续 data_query 看到 scope 就走 query_scoped_business_data，不走全局
        query_business_data。
    """

    context = dict(snapshot["context"])
    payload = {
        "seller_id": context["seller_id"], "default_start_date": context["start_date"],
        "default_end_date": context["end_date"], "metric_name": context["metric_name"],
        "granularity": context["granularity"],
    }
    # Keep the historical default scope stable with seller execution disabled.
    # Only this trusted server issuer may opt the dimension into a scope.
    from app.core.config import get_settings
    if get_settings().seller_dimension_enabled:
        payload["allowed_dimensions"] = (
            *CONTROLLED_ANALYSIS_DEFAULT_DIMENSIONS,
            "seller",
        )
    return ControlledAnalysisScope.model_validate(payload)


def _scope_params(scope: ControlledAnalysisScope) -> dict[str, object]:
    """锁定范围的 SQL 参数化模板，所有受控查询共用。"""
    return {"seller_id": scope.seller_id, "start_date": scope.default_start_date, "end_date": scope.default_end_date}


def breakdown_by_seller(scope: object, metric_name: str, limit: int = 10) -> dict[str, object]:
    """Execute a seller breakdown only inside a server-resolved CanonicalScope."""
    from app.core.config import get_settings
    from app.scope.canonical import CanonicalScope
    from app.auth.context import AuthorizationDenied, build_authorization_context

    if not isinstance(scope, CanonicalScope):
        raise PermissionError("A server-resolved CanonicalScope is required")
    if not get_settings().seller_dimension_enabled:
        raise PermissionError("Seller dimension execution is disabled")
    if scope.seller_scope_type in {"selected", "all"} and not get_settings().seller_multi_scope_enabled:
        raise PermissionError("Multi-seller scope execution is disabled")
    if len(scope.seller_ids) > get_settings().seller_scope_max_selected_sellers:
        raise PermissionError("Seller scope exceeds the configured seller count limit")
    if metric_name not in {"gmv", "order_count"} or metric_name not in scope.metric_ids:
        raise ValueError("Seller breakdown is only registered for locked gmv/order_count metrics")
    if "seller" not in scope.dimensions:
        raise PermissionError("Seller dimension is outside the locked scope")
    if scope.include_unattributed and scope.seller_scope_type != "all":
        raise PermissionError("Only ALL scope may include unattributed rows")

    tenant = current_tenant_context()
    if tenant is None:
        raise PermissionError("Trusted tenant context is required for seller breakdown")
    if scope.include_unattributed and not tenant.is_platform_admin:
        raise PermissionError("Including unattributed rows requires a platform administrator")
    if (scope.end_date - scope.start_date).days > get_settings().seller_scope_max_date_span_days:
        raise ValueError("Seller scope date range exceeds the configured scope limit")
    authz = build_authorization_context(
        tenant,
        metric_scope_type="seller" if tenant.scope_type == "seller" else "platform",
    )
    authz.require_action("analyze")
    authz.require_metrics((metric_name,))
    authz.require_dimensions(scope.dimensions)
    if scope.authorization_fingerprint != authz.authz_digest:
        from app.auth.enforcement import StaleAuthorization
        raise StaleAuthorization("seller breakdown authorization changed")
    if scope.permission_version != authz.permission_version or scope.policy_set_version != authz.policy_set_version:
        from app.auth.enforcement import StaleAuthorization
        raise StaleAuthorization("seller breakdown permission or policy version changed")
    if scope.seller_scope_type == "all":
        if not tenant.is_platform_admin:
            raise PermissionError("ALL scope requires a platform administrator")
    else:
        authz.require_sellers(scope.seller_ids)
        for seller_id in scope.seller_ids:
            grant = authz.seller_grant(seller_id)
            if tenant.is_platform_admin:
                if scope.seller_scope_type == "selected" and grant is None:
                    raise PermissionError(f"Selected seller {seller_id!r} has no explicit seller grant")
                if grant is not None:
                    grant_start = date.fromisoformat(grant.start_date) if grant.start_date else None
                    grant_end = date.fromisoformat(grant.end_date) if grant.end_date else None
                    if (grant_start and scope.start_date < grant_start) or (
                        grant_end and scope.end_date > grant_end
                    ):
                        raise PermissionError("Requested date range exceeds seller grant")
            else:
                authz.require_date_range(seller_id, scope.start_date, scope.end_date)

    from app.metrics.registry import get_registry
    registry = get_registry()
    if not registry.supports(metric_name, "breakdown", scope="platform") or not registry.is_dimension_allowed(
        metric_name, "seller", scope="platform"
    ):
        raise ValueError(f"Seller breakdown is not enabled for {metric_name}")
    semantic = registry.get(metric_name).implementations["platform"].dimension_semantics.get("seller")

    predicates: list[str] = ["fo.purchase_date BETWEEN :start_date AND :end_date"]
    params: dict[str, object] = {
        "start_date": scope.start_date,
        "end_date": scope.end_date,
    }
    if scope.seller_scope_type in {"single", "selected"}:
        predicates.append("oi.seller_id = ANY(:seller_ids)")
        params["seller_ids"] = list(scope.seller_ids)
    else:
        predicates.append("(oi.seller_id IS NOT NULL OR :include_unattributed = TRUE)")
        params["include_unattributed"] = scope.include_unattributed
    where_sql = " AND ".join(predicates)
    total_where = where_sql
    if scope.seller_scope_type == "all" and not scope.include_unattributed:
        total_where += " AND oi.seller_id IS NOT NULL"
    metric_expression = metric_sql(metric_name)
    group_params = dict(params)
    total_params = dict(params)
    unknown_params = {
        "start_date": scope.start_date,
        "end_date": scope.end_date,
    }
    cap = min(max(int(limit), 1), 100)

    with canonical_seller_query_transaction(engine, scope) as connection:
        items = connection.execute(text(f"""
            SELECT oi.seller_id::text AS seller_id,
                   {metric_expression} AS value,
                   COUNT(DISTINCT oi.order_id) AS group_order_count
            {_FROM_CLAUSE}
            WHERE {where_sql} AND oi.seller_id IS NOT NULL
            GROUP BY oi.seller_id
            ORDER BY value DESC NULLS LAST, oi.seller_id ASC
            LIMIT :limit
        """), group_params | {"limit": cap}).mappings().all()
        total = connection.execute(text(f"""
            SELECT COALESCE(SUM(oi.price + oi.freight_value), 0) AS gmv,
                   COUNT(DISTINCT oi.order_id) AS order_count
            {_FROM_CLAUSE}
            WHERE {total_where}
        """), total_params).mappings().one()
        unattributed = connection.execute(text(f"""
            SELECT COUNT(*) AS row_count,
                   COALESCE(SUM(oi.price + oi.freight_value), 0) AS gmv,
                   COUNT(DISTINCT oi.order_id) AS order_count
            {_FROM_CLAUSE}
            WHERE fo.purchase_date BETWEEN :start_date AND :end_date
              AND oi.seller_id IS NULL
        """), unknown_params).mappings().one()

    total_gmv = float(total["gmv"] or 0)
    total_orders = int(total["order_count"] or 0)
    total_value = total_gmv if metric_name == "gmv" else total_orders
    unknown = None
    if not tenant.is_platform_admin:
        unknown = {
            "gmv": None,
            "order_count": None,
            "reason": "outside_single_seller_scope",
            "scope": "outside_single_seller_scope",
        }
    elif int(unattributed["row_count"] or 0):
        unknown = {
            "gmv": float(unattributed["gmv"] or 0),
            "order_count": int(unattributed["order_count"] or 0),
            "reason": "seller_id_is_null; order_count may overlap seller-attributed orders and is not additive",
            "scope": "date_range_all_unattributed",
        }
    return {
        "metric_name": metric_name,
        "dimension": "seller",
        "items": [
            {
                "seller_id": str(row["seller_id"]),
                "value": float(row["value"] or 0) if metric_name == "gmv" else int(row["value"] or 0),
                "order_count": int(row["group_order_count"] or 0),
            }
            for row in items
        ],
        "total": {
            "value": total_value,
            "gmv": total_gmv,
            "order_count": total_orders,
            "total_includes_unattributed": bool(scope.include_unattributed),
        },
        "unattributed": unknown,
        "non_additive": bool(semantic is not None and not semantic.additive),
        "semantics": semantic.semantics if semantic else None,
        "group_order_count_non_additive": True,
        "group_order_count_semantics": "distinct_orders_per_seller",
        "seller_scope_type": scope.seller_scope_type,
        "seller_ids": list(scope.seller_ids) if scope.seller_scope_type != "all" else None,
        "all_sellers": scope.seller_scope_type == "all",
        "query_scope_fingerprint": scope.query_scope_fingerprint,
        "message": (
            "单店范围只有一个 seller 分组，不构成商户间对比。"
            if scope.seller_scope_type == "single" else None
        ),
        "source_table": _CORE_QUERY_SOURCE,
        "status": (
            "ok"
            if items or total_gmv or total_orders or (
                tenant.is_platform_admin and int(unattributed["row_count"] or 0)
            )
            else "not_found"
        ),
    }


# ============================================================================
# 受控查询函数集
#
# 这三个函数（query_scoped_metric / query_scoped_trend / breakdown_scoped_*）
# 是 Agent 在锁定范围内能调用的全部工具。它们都：
# 1. 校验 metric_name 是否在 scope.allowed_metrics 中
# 2. 从 _FROM_CLAUSE 实时取数
# 3. 由 scope_params 注入 seller_id / start_date / end_date，不信任请求参数
# 4. 返回统一结构的 dict（上层调用链在 client.py 中拼装成 DataQueryAgentResult）
#
# query_scoped_business_data (client.py) 根据 task_type 路由到这三个函数。
# 它们在行为上等价于 breakdown_by_dimension 在 scope 内的版本。
# ============================================================================

def _scoped_domain_definition(metric_name: str) -> tuple[str, str, str]:
    """返回支付/履约指标的受控数据源、表达式和下单日期列。

    支付金额只使用 ``allocated_payment_value``，所以多卖家订单不会把整笔
    支付重复计入每个店铺。店铺履约视图已排除多卖家订单，保证该限制可追溯。
    """
    if metric_name == "payment_value":
        return ("analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)", "SUM(p.allocated_payment_value)", "o.purchase_date")
    if metric_name == "installment_order_share":
        return ("analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)", "SUM(CASE WHEN p.payment_installments > 1 THEN p.allocated_payment_value ELSE 0 END) * 100.0 / NULLIF(SUM(p.allocated_payment_value), 0)", "o.purchase_date")
    if metric_name == "review_score_avg":
        # 一个订单可有多件商品。先按 seller_id + order_id 去重，避免同一条
        # 订单评论因商品行 JOIN 而被重复计入平均分。
        return (
            """(
                SELECT DISTINCT oi.seller_id, oi.order_id, fo.purchase_date
                FROM analytics.v_delivered_order_items oi
                JOIN analytics.v_delivered_orders fo USING(order_id)
            ) seller_orders
            JOIN raw.order_reviews r USING(order_id)""",
            "AVG(r.review_score)",
            "seller_orders.purchase_date",
        )
    if metric_name == "seller_order_bad_review_rate":
        return ("""(SELECT DISTINCT oi.seller_id, oi.order_id, fo.purchase_date FROM analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id)) seller_orders JOIN raw.order_reviews r USING(order_id)""", "COUNT(*) FILTER (WHERE r.review_score <= 2) * 100.0 / NULLIF(COUNT(r.review_score), 0)", "seller_orders.purchase_date")
    if metric_name == "bad_review_rate":
        # 同 seller_order_bad_review_rate，供 Explorer 预览使用；下单日期归一.
        return ("""(SELECT DISTINCT oi.seller_id, oi.order_id, fo.purchase_date FROM analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id)) seller_orders JOIN raw.order_reviews r USING(order_id)""", "COUNT(*) FILTER (WHERE r.review_score <= 2) * 100.0 / NULLIF(COUNT(r.review_score), 0)", "seller_orders.purchase_date")
    if metric_name == "seller_items_per_order":
        return ("""(SELECT oi.seller_id, oi.order_id, fo.purchase_date, COUNT(*) AS seller_item_count FROM analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id) GROUP BY oi.seller_id, oi.order_id, fo.purchase_date) seller_orders""", "AVG(seller_item_count)", "seller_orders.purchase_date")
    if metric_name == "seller_freight_to_gmv_rate":
        return ("analytics.v_delivered_order_items oi JOIN analytics.v_delivered_orders fo USING(order_id)", "SUM(oi.freight_value) * 100.0 / NULLIF(SUM(oi.price + oi.freight_value), 0)", "fo.purchase_date")
    # Phase 6 M1b: 订单漏斗（按 seller 过滤）
    if metric_name == "order_approval_rate":
        return ("""(
            SELECT DISTINCT oi.seller_id, o.order_id,
                   o.order_purchase_timestamp AS purchase_date,
                   o.order_approved_at
            FROM raw.order_items oi JOIN raw.orders o USING(order_id)
        ) seller_funnel""",
        "COUNT(*) FILTER (WHERE order_approved_at IS NOT NULL) * 100.0 / NULLIF(COUNT(*), 0)",
        "seller_funnel.purchase_date")
    if metric_name == "order_shipment_rate":
        return ("""(
            SELECT DISTINCT oi.seller_id, o.order_id,
                   o.order_purchase_timestamp AS purchase_date,
                   o.order_delivered_carrier_date
            FROM raw.order_items oi JOIN raw.orders o USING(order_id)
        ) seller_funnel""",
        "COUNT(*) FILTER (WHERE order_delivered_carrier_date IS NOT NULL) * 100.0 / NULLIF(COUNT(*), 0)",
        "seller_funnel.purchase_date")
    if metric_name == "order_final_delivery_rate":
        return ("""(
            SELECT DISTINCT oi.seller_id, o.order_id,
                   o.order_purchase_timestamp AS purchase_date,
                   o.order_status
            FROM raw.order_items oi JOIN raw.orders o USING(order_id)
        ) seller_funnel""",
        "COUNT(*) FILTER (WHERE order_status = 'delivered') * 100.0 / NULLIF(COUNT(*), 0)",
        "seller_funnel.purchase_date")
    # Phase 6 M3: 含文本评论率
    if metric_name == "review_comment_rate":
        return ("""(
            SELECT DISTINCT oi.seller_id, r.order_id,
                   fo.purchase_date, r.review_comment_message
            FROM analytics.v_delivered_order_items oi
            JOIN analytics.v_delivered_orders fo USING(order_id)
            LEFT JOIN raw.order_reviews r USING(order_id)
        ) seller_reviews""",
        "COUNT(*) FILTER (WHERE review_comment_message IS NOT NULL"
        " AND TRIM(review_comment_message) != '')"
        " * 100.0 / NULLIF(COUNT(*), 0)",
        "seller_reviews.purchase_date")
    expressions = {
        "avg_approval_lead_hours": "AVG(approval_lead_hours)",
        "avg_dispatch_lead_hours": "AVG(dispatch_lead_hours)",
        "avg_delivery_transit_hours": "AVG(delivery_transit_hours)",
        "on_time_delivery_rate": "COUNT(*) FILTER (WHERE is_on_time) * 100.0 / NULLIF(COUNT(*) FILTER (WHERE is_on_time IS NOT NULL), 0)",
    }
    return ("analytics.v_seller_fulfillment", expressions[metric_name], "purchase_date")


def _inclusive_date_range_predicate(date_column: str) -> str:
    """Build a half-open SQL range for a user-facing inclusive calendar range.

    A DATE end bound is midnight at the start of that day when compared with a
    timestamp. Advancing the bound by one day and using ``<`` includes the
    entire requested end date while excluding midnight of the following day.
    The caller keeps the original inclusive dates for locked-scope checks.
    """
    return f"{date_column} >= :start_date AND {date_column} < :end_date_exclusive"


def _query_scoped_domain(scope: ControlledAnalysisScope, metric_name: str, *, metric_date: date | None = None) -> dict[str, object]:
    source, expression, date_column = _scoped_domain_definition(metric_name)
    params = _scope_params(scope)
    if metric_date is not None:
        params["metric_date"] = metric_date
        date_predicate = f"{date_column} = :metric_date"
    else:
        date_predicate = f"{date_column} BETWEEN :start_date AND :end_date"
    seller_column = (
        "p.seller_id" if metric_name in {"payment_value", "installment_order_share"}
        else "seller_orders.seller_id" if metric_name in {"review_score_avg", "bad_review_rate", "seller_order_bad_review_rate", "seller_items_per_order"}
        else "seller_funnel.seller_id" if metric_name in {"order_approval_rate", "order_shipment_rate", "order_final_delivery_rate"}
        else "seller_reviews.seller_id" if metric_name in {"review_comment_rate"}
        else "oi.seller_id" if metric_name == "seller_freight_to_gmv_rate"
        else "seller_id"
    )
    with seller_query_transaction(
        engine, scope,
        start_date=metric_date or scope.default_start_date,
        end_date=metric_date or scope.default_end_date,
    ) as connection:
        row = connection.execute(text(f"SELECT {expression} AS value FROM {source} WHERE {seller_column}=:seller_id AND {date_predicate}"), params).mappings().one()
    return {"metric_name": metric_name, "metric_date": (metric_date.isoformat() if metric_date else f"{scope.default_start_date} 至 {scope.default_end_date}"),
            "value": float(row["value"] or 0), "start_date": (metric_date or scope.default_start_date).isoformat(),
            "end_date": (metric_date or scope.default_end_date).isoformat(), "source_table": source,
            "lineage_note": (
                "支付按商品金额加运费比例分摊"
                if metric_name in {"payment_value", "installment_order_share"}
                else "同一订单的多件商品仅计一条评论；多卖家订单评论归属给参与店铺"
                if metric_name == "review_score_avg"
                else "店铺履约已排除多卖家订单"
            ), "status": "ok"}


def breakdown_scoped_payment_type(scope: ControlledAnalysisScope, metric_name: str) -> dict[str, object]:
    """按支付方式拆解店铺范围内的分摊支付金额。

    这个维度只对支付金额开放；返回的 share 是金额占比而非订单数占比。
    """
    if metric_name != "payment_value" or "payment_type" not in scope.allowed_dimensions:
        raise ValueError("payment_type is only available for payment_value in the locked scope")
    with seller_query_transaction(engine, scope) as connection:
        rows = connection.execute(text("""
            SELECT COALESCE(p.payment_type, 'UNKNOWN') AS dimension_value,
                   SUM(p.allocated_payment_value) AS value
            FROM analytics.fact_order_payments p
            JOIN analytics.fact_orders o USING(order_id)
            WHERE p.seller_id=:seller_id AND o.purchase_date BETWEEN :start_date AND :end_date
            GROUP BY COALESCE(p.payment_type, 'UNKNOWN') ORDER BY value DESC
        """), _scope_params(scope)).mappings().all()
    total = sum(float(row["value"] or 0) for row in rows)
    return {"metric_name": metric_name, "dimension": "payment_type", "items": [
        {"dimension_value": str(row["dimension_value"]), "value": float(row["value"] or 0), "share": round(float(row["value"] or 0) * 100 / total, 2) if total else None}
        for row in rows], "source_table": "analytics.fact_order_payments", "lineage_note": "支付按商品金额加运费比例分摊", "status": "ok" if rows else "not_found"}

def query_scoped_metric(scope: ControlledAnalysisScope, metric_name: str,
                         *, metric_date: date | str | None = None) -> dict[str, object]:
    """在锁定范围内查询指标值。

    参数：
        metric_date: 可选，指定单日查询。为 None 时返回完整时间范围上的汇总值。
    """

    if isinstance(metric_date, str):
        try:
            metric_date = date.fromisoformat(metric_date)
        except ValueError as exc:
            raise ValueError("metric_date must use YYYY-MM-DD format") from exc
    if metric_name not in scope.allowed_metrics:
        raise ValueError(f"Metric {metric_name!r} is not allowed by the locked analysis scope")
    # Seller-expanded metrics are registered in the same metric catalog but
    # have dedicated fixed SQL handlers.  Route them before the generic
    # explorer SQL map; otherwise a valid metric such as
    # ``seller_avg_freight_per_order`` raises a misleading KeyError.
    from app.data.repositories import seller_extensions
    if seller_extensions.supports(metric_name):
        return seller_extensions.query(scope, metric_name, metric_date)
    if metric_name in _DOMAIN_METRICS:
        return _query_scoped_domain(scope, metric_name, metric_date=metric_date)
    params = _scope_params(scope)
    where_date = "fo.purchase_date = :metric_date"
    if metric_date:
        params["metric_date"] = metric_date
    else:
        where_date = "fo.purchase_date BETWEEN :start_date AND :end_date"
    with seller_query_transaction(
        engine, scope,
        start_date=metric_date or scope.default_start_date,
        end_date=metric_date or scope.default_end_date,
    ) as connection:
        row = connection.execute(text(
            f"SELECT {metric_sql(metric_name)} AS value {_FROM_CLAUSE} "
            "WHERE oi.seller_id=:seller_id AND " + where_date
        ), params).mappings().one()
    display_date = metric_date.isoformat() if metric_date else f"{scope.default_start_date.isoformat()} 至 {scope.default_end_date.isoformat()}"
    return {"metric_name": metric_name, "metric_date": display_date, "value": float(row["value"] or 0),
            "start_date": (metric_date or scope.default_start_date).isoformat(),
            "end_date": (metric_date or scope.default_end_date).isoformat(),
            # source_table names the objects read by this SQL; source_lineage
            # names the underlying facts exposed through those delivered views.
            "source_table": _CORE_QUERY_SOURCE,
            "source_lineage": _CORE_QUERY_LINEAGE,
            "status": "ok"}


def query_scoped_statistics(
    scope: ControlledAnalysisScope,
    metric_name: str,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    granularity: str | None = None,
    requested: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Return descriptive statistics over scoped metric time buckets."""

    from app.metrics.statistics import summarize_points

    effective_start = start_date or scope.default_start_date
    effective_end = end_date or scope.default_end_date
    trend_scope = scope.model_copy(update={"granularity": granularity or scope.granularity})
    trend = query_scoped_trend(
        trend_scope,
        metric_name,
        start_date=effective_start,
        end_date=effective_end,
    )
    summary = summarize_points(trend.get("points", []), requested=requested)
    return {
        "metric_name": metric_name,
        "start_date": effective_start.isoformat(),
        "end_date": effective_end.isoformat(),
        "granularity": trend_scope.granularity,
        "statistics": summary["statistics"],
        "count": summary["count"],
        "min_bucket": summary["min_bucket"],
        "max_bucket": summary["max_bucket"],
        "source_table": trend.get("source_table", ""),
        "status": summary["status"],
    }


def query_scoped_order_freight_distribution(
    scope: ControlledAnalysisScope,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, object]:
    """Summarize per-order freight, not the distribution of daily averages.

    This first entity-level statistics contract is deliberately narrow: one
    observation is a delivered seller-order, with freight summed across that
    seller's item rows. Zero freight is retained; orders with no non-NULL
    freight row are excluded and reported separately. No order identifier is
    returned.
    """
    from app.tenancy import current_tenant_context
    from app.data.rls import set_local_context

    metric_name = "seller_avg_freight_per_order"
    if metric_name not in scope.allowed_metrics:
        raise PermissionError("Order freight distribution is outside the locked metric scope")
    effective_start = start_date or scope.default_start_date
    effective_end = end_date or scope.default_end_date
    if effective_end < effective_start:
        raise ValueError("end_date must not be earlier than start_date")
    if effective_start < scope.default_start_date or effective_end > scope.default_end_date:
        raise ValueError("Order freight distribution date range is outside the locked scope")
    tenant = current_tenant_context()
    if tenant is None:
        raise PermissionError("Trusted tenant context is required for entity-level statistics")

    statement = text("""
        WITH per_order AS (
            SELECT oi.order_id,
                   SUM(oi.freight_value) AS order_freight,
                   COUNT(oi.freight_value) AS freight_rows,
                   COUNT(*) AS item_rows
            FROM analytics.v_delivered_order_items oi
            JOIN analytics.v_delivered_orders fo USING (order_id)
            WHERE oi.seller_id = :seller_id
              AND fo.purchase_date BETWEEN :start_date AND :end_date
            GROUP BY oi.order_id
        )
        SELECT COUNT(*) FILTER (WHERE freight_rows = item_rows) AS sample_size,
               COUNT(*) FILTER (WHERE freight_rows != item_rows) AS excluded_missing,
               AVG(order_freight) FILTER (WHERE freight_rows = item_rows) AS mean,
               STDDEV_POP(order_freight) FILTER (WHERE freight_rows = item_rows) AS std,
               MIN(order_freight) FILTER (WHERE freight_rows = item_rows) AS min,
               percentile_cont(0.25) WITHIN GROUP (ORDER BY order_freight)
                   FILTER (WHERE freight_rows = item_rows) AS q1,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY order_freight)
                   FILTER (WHERE freight_rows = item_rows) AS median,
               percentile_cont(0.75) WITHIN GROUP (ORDER BY order_freight)
                   FILTER (WHERE freight_rows = item_rows) AS q3,
               MAX(order_freight) FILTER (WHERE freight_rows = item_rows) AS max,
               COUNT(*) FILTER (WHERE freight_rows = item_rows AND order_freight = 0) AS zero_freight_orders
        FROM per_order
    """)
    with engine.begin() as connection:
        set_local_context(
            connection, tenant, scope_type="seller",
            seller_access={scope.seller_id: (effective_start, effective_end)},
        )
        row = connection.execute(statement, {
            "seller_id": scope.seller_id,
            "start_date": effective_start,
            "end_date": effective_end,
        }).mappings().one()
    numeric_names = ("mean", "std", "min", "q1", "median", "q3", "max")
    stats = {
        name: (float(row[name]) if row[name] is not None else None)
        for name in numeric_names
    }
    if any(value is not None and not math.isfinite(value) for value in stats.values()):
        raise ValueError("Database returned a non-finite entity statistic")
    sample_size = int(row["sample_size"] or 0)
    return {
        "metric_name": metric_name,
        "basis": "order_entity",
        "entity": "delivered_seller_order",
        "start_date": effective_start.isoformat(),
        "end_date": effective_end.isoformat(),
        "sample_size": sample_size,
        "statistics": stats,
        "zero_freight_orders": int(row["zero_freight_orders"] or 0),
        "excluded_missing_freight_orders": int(row["excluded_missing"] or 0),
        "source_table": "analytics.v_delivered_order_items + analytics.v_delivered_orders",
        "population_definition": "Delivered seller orders whose every item-row freight value is non-null; zero-freight orders are retained. The date filter uses purchase_date, and each order's freight is the sum of its seller item-row freight values.",
        "status": "ok" if sample_size else "not_found",
    }


def query_scoped_delivery_review_association(
    scope: ControlledAnalysisScope,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    pair_id: str,
) -> dict[str, object]:
    """Execute one server-registered association pair in the locked seller scope."""
    from app.tenancy import current_tenant_context
    from app.data.rls import set_local_context
    from app.analysis.association_catalog import get_association_pair

    pair = get_association_pair(pair_id)
    if pair is None:
        raise ValueError("Association pair is not registered")
    metric_id = pair.target_metric_id
    if metric_id not in scope.allowed_metrics:
        raise PermissionError("The target metric for this association is outside the locked scope")
    start = start_date or scope.default_start_date
    end = end_date or scope.default_end_date
    if end < start:
        raise ValueError("end_date must not be earlier than start_date")
    if start < scope.default_start_date or end > scope.default_end_date:
        raise ValueError("Association date range is outside the locked scope")
    tenant = current_tenant_context()
    if tenant is None:
        raise PermissionError("Trusted tenant context is required for association analysis")

    from app.auth.context import AuthorizationDenied, build_authorization_context

    authz = build_authorization_context(
        tenant, allowed_metric_ids=scope.allowed_metrics, metric_scope_type="seller",
    )
    try:
        authz.require_sellers([scope.seller_id])
        authz.require_metrics([metric_id])
        authz.require_date_range(scope.seller_id, start, end)
        authz.require_tool("query_scoped_registered_association")
    except AuthorizationDenied as exc:
        raise PermissionError(str(exc)) from exc

    if pair.kind == "continuous":
        return _query_continuous_review_association(
            scope, pair=pair, start=start, end=end, tenant=tenant,
        )
    if pair.kind != "binary" or pair.query_handler != "delivery_timing_bad_review":
        raise ValueError("Association query handler is not registered")
    from app.analysis.association_stats import summarize_binary_association

    statement = text("""
        SELECT
          COUNT(DISTINCT order_id) AS population_orders,
          COUNT(DISTINCT order_id) FILTER (WHERE review_score IS NOT NULL AND is_on_time IS NOT NULL) AS eligible_orders,
          COUNT(DISTINCT order_id) FILTER (WHERE review_score IS NOT NULL AND is_on_time IS TRUE) AS on_time_n,
          COUNT(DISTINCT order_id) FILTER (WHERE review_score IS NOT NULL AND is_on_time IS TRUE AND review_score <= 2) AS on_time_bad,
          COUNT(DISTINCT order_id) FILTER (WHERE review_score IS NOT NULL AND is_on_time IS FALSE) AS delayed_n,
          COUNT(DISTINCT order_id) FILTER (WHERE review_score IS NOT NULL AND is_on_time IS FALSE AND review_score <= 2) AS delayed_bad
        FROM analytics.v_order_unified
        WHERE seller_id = :seller_id
          AND purchase_date BETWEEN :start_date AND :end_date
          AND is_single_seller_order
    """)
    with engine.begin() as connection:
        set_local_context(
            connection, tenant, scope_type="seller",
            seller_access={scope.seller_id: (start, end)},
        )
        row = connection.execute(statement, {
            "seller_id": scope.seller_id, "start_date": start, "end_date": end,
        }).mappings().one()
    population = int(row["population_orders"] or 0)
    eligible = int(row["eligible_orders"] or 0)
    on_time_n = int(row["on_time_n"] or 0)
    delayed_n = int(row["delayed_n"] or 0)
    stats = summarize_binary_association(
        exposed_events=int(row["delayed_bad"] or 0), exposed_total=delayed_n,
        comparison_events=int(row["on_time_bad"] or 0), comparison_total=on_time_n,
        min_total=50,
    )
    return {
        "pair_id": pair_id,
        "target_metric": metric_id,
        "exposure": pair.exposure_label,
        "outcome": pair.outcome_label,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "analysis_contract": pair.evidence_contract(),
        "population_orders": population,
        "eligible_orders": eligible,
        "excluded_missing_pair_fields": max(0, population - eligible),
        **stats,
        "source_table": pair.source_table,
        "missing_policy": pair.exclusion_policy,
        "limitations": list(pair.limitations),
    }


def _query_continuous_review_association(
    scope: ControlledAnalysisScope, *, pair: Any, start: date, end: date, tenant: Any,
) -> dict[str, object]:
    """Run a registered order-level continuous exposure/rating Spearman pair."""
    from app.analysis.association_stats import summarize_spearman_association
    from app.data.rls import set_local_context

    # SQL identifiers must remain server-owned. The pair registry may select
    # only one of these fixed source columns; no model/client string is
    # interpolated into the statement.
    exposure_columns = {
        "delivery_hours_review_score": "delivery_transit_hours",
        "order_freight_review_score": "freight_sum",
        "order_item_line_count_review_score": "item_count",
        "order_merchandise_value_review_score": "price_sum",
    }
    exposure_column = exposure_columns.get(pair.query_handler)
    if exposure_column is None:
        raise ValueError("Association query handler is not registered")

    counts_sql = text("""
        SELECT COUNT(DISTINCT order_id) AS population_orders,
               COUNT(DISTINCT order_id) FILTER (
                 WHERE {exposure_column} IS NOT NULL
                   AND {exposure_column} >= 0
                   AND review_score IS NOT NULL
               ) AS eligible_orders
        FROM analytics.v_order_unified
        WHERE seller_id = :seller_id
          AND purchase_date BETWEEN :start_date AND :end_date
          AND is_single_seller_order
    """.format(exposure_column=exposure_column))
    rows_sql = text("""
        SELECT {exposure_column} AS exposure_value, review_score
        FROM analytics.v_order_unified
        WHERE seller_id = :seller_id
          AND purchase_date BETWEEN :start_date AND :end_date
          AND is_single_seller_order
          AND {exposure_column} IS NOT NULL
          AND {exposure_column} >= 0
          AND review_score IS NOT NULL
        ORDER BY purchase_date, {exposure_column}, review_score
        LIMIT :row_limit
    """.format(exposure_column=exposure_column))
    params = {"seller_id": scope.seller_id, "start_date": start, "end_date": end}
    with engine.begin() as connection:
        set_local_context(
            connection, tenant, scope_type="seller",
            seller_access={scope.seller_id: (start, end)},
        )
        counts = connection.execute(counts_sql, params).mappings().one()
        population = int(counts["population_orders"] or 0)
        eligible = int(counts["eligible_orders"] or 0)
        if eligible > scope.max_rows:
            return {
                "pair_id": pair.pair_id,
                "target_metric": pair.target_metric_id,
                "exposure": pair.exposure_label,
                "outcome": pair.outcome_label,
                "analysis_contract": pair.evidence_contract(),
                "method_id": pair.method_id,
                "source_table": pair.source_table,
                "status": "over_limit",
                "population_orders": population,
                "eligible_orders": eligible,
                "excluded_missing_pair_fields": population - eligible,
                "missing_policy": "No estimate is produced from truncated or partial observations.",
                "row_limit": scope.max_rows,
                "limitations": ["Eligible observations exceed the bounded in-memory rank-correlation limit; no partial estimate is reported."],
            }
        rows = connection.execute(rows_sql, {**params, "row_limit": scope.max_rows + 1}).mappings().all()
    if len(rows) != eligible:
        raise ValueError("Association query count and observation rows do not reconcile")
    summary = summarize_spearman_association(
        [(float(row["exposure_value"]), float(row["review_score"])) for row in rows],
        population_n=population,
        missing_n=max(0, population - eligible),
        min_total=pair.minimum_sample_size,
    )
    return {
        "pair_id": pair.pair_id,
        "target_metric": pair.target_metric_id,
        "exposure": pair.exposure_label,
        "outcome": pair.outcome_label,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "population_orders": population,
        "eligible_orders": eligible,
        "excluded_missing_pair_fields": max(0, population - eligible),
        **summary,
        "analysis_contract": pair.evidence_contract(),
        "source_table": pair.source_table,
        "missing_policy": pair.exclusion_policy,
        "limitations": list(pair.limitations) + [
            "Order identifiers are not returned.",
        ],
    }
def compare_scoped_metric(
    scope: ControlledAnalysisScope,
    metric_name: str,
    *,
    current_start: date,
    current_end: date,
    baseline_start: date,
    baseline_end: date,
) -> dict[str, object]:
    """Compare two periods — the caller (query_scoped_business_data) is responsible
    for SecurityScope and QueryPolicy enforcement; this function only runs SQL.
    """
    if metric_name not in scope.allowed_metrics:
        raise ValueError(f"Metric {metric_name!r} is not allowed by the locked analysis scope")

    current_scope = scope.model_copy(update={"default_start_date": current_start, "default_end_date": current_end})
    baseline_scope = scope.model_copy(update={"default_start_date": baseline_start, "default_end_date": baseline_end})
    current = query_scoped_metric(current_scope, metric_name)
    baseline = query_scoped_metric(baseline_scope, metric_name)
    current_value = current.get("value")
    baseline_value = baseline.get("value")
    delta_value = None
    delta_ratio = None
    if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
        delta_value = round(float(current_value) - float(baseline_value), 2)
        if baseline_value != 0:
            delta_ratio = round(delta_value * 100.0 / float(baseline_value), 2)

    return {
        "metric_name": metric_name,
        "left_date": current_start.isoformat(),
        "left_end_date": current_end.isoformat(),
        "right_date": baseline_start.isoformat(),
        "right_end_date": baseline_end.isoformat(),
        "left_value": current_value,
        "right_value": baseline_value,
        "delta_value": delta_value,
        "delta_ratio": delta_ratio,
        "source_table": current["source_table"],
        "lineage_note": current.get("lineage_note"),
        "status": "ok" if current.get("status") == "ok" or baseline.get("status") == "ok" else "not_found",
    }


def query_scoped_trend(scope: ControlledAnalysisScope, metric_name: str,
                        *, start_date: date | None = None, end_date: date | None = None) -> dict[str, object]:
    """在锁定范围内按 scope.granularity 分桶查询指标趋势。

    参数：
        start_date / end_date: 可选，在 scope 范围内进一步缩小查询窗口。
                               为 None 时使用 scope 的完整范围。
    """

    if metric_name not in scope.allowed_metrics:
        raise ValueError(f"Metric {metric_name!r} is not allowed by the locked analysis scope")
    # Keep trend and point queries on the same seller-extension handler path.
    # This is also the path used by metric_statistics, which summarizes the
    # returned scoped points without changing the metric definition.
    from app.data.repositories import seller_extensions
    if seller_extensions.supports(metric_name):
        return seller_extensions.trend(scope, metric_name, start_date, end_date)
    if metric_name in _DOMAIN_METRICS:
        source, expression, date_column = _scoped_domain_definition(metric_name)
        seller_column = (
            "p.seller_id" if metric_name in {"payment_value", "installment_order_share"}
            else "seller_orders.seller_id" if metric_name in {"review_score_avg", "seller_order_bad_review_rate", "seller_items_per_order"}
            else "oi.seller_id" if metric_name == "seller_freight_to_gmv_rate"
            else "seller_id"
        )
        params = _scope_params(scope)
        selected_start = start_date or scope.default_start_date
        selected_end = end_date or scope.default_end_date
        params["start_date"] = selected_start
        if date_column == "seller_funnel.purchase_date":
            # The three seller-funnel definitions expose a timestamp. Convert
            # the inclusive calendar end to a half-open timestamp boundary.
            params.pop("end_date", None)
            params["end_date_exclusive"] = selected_end + timedelta(days=1)
            date_predicate = _inclusive_date_range_predicate(date_column)
        else:
            # Keep DATE-backed domain trends on their existing inclusive path.
            params["end_date"] = selected_end
            date_predicate = f"{date_column} BETWEEN :start_date AND :end_date"
        bucket = f"date_trunc('{scope.granularity}', {date_column})::date"
        with seller_query_transaction(
            engine, scope,
            start_date=selected_start,
            end_date=selected_end,
        ) as connection:
            rows = connection.execute(text(f"SELECT {bucket} AS bucket, {expression} AS value FROM {source} WHERE {seller_column}=:seller_id AND {date_predicate} GROUP BY {bucket} ORDER BY bucket"), params).mappings().all()
        return {"metric_name": metric_name, "start_date": str(selected_start), "end_date": str(selected_end), "granularity": scope.granularity,
                "points": [{"bucket": str(row["bucket"]), "value": float(row["value"] or 0)} for row in rows], "source_table": source,
                "lineage_note": (
                    "支付按商品金额加运费比例分摊"
                    if metric_name in {"payment_value", "installment_order_share"}
                    else "同一订单的多件商品仅计一条评论；多卖家订单评论归属给参与店铺"
                    if metric_name == "review_score_avg"
                    else "店铺履约已排除多卖家订单"
                ), "status": "ok" if rows else "not_found"}
    bucket_sql = _BUCKET_SQL[scope.granularity]
    params = _scope_params(scope)
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    display_start = (start_date or scope.default_start_date).isoformat()
    display_end = (end_date or scope.default_end_date).isoformat()
    with seller_query_transaction(
        engine, scope,
        start_date=start_date or scope.default_start_date,
        end_date=end_date or scope.default_end_date,
    ) as connection:
        rows = connection.execute(text(f"""
            SELECT {bucket_sql} AS bucket, {metric_sql(metric_name)} AS value
            {_FROM_CLAUSE}
            WHERE oi.seller_id=:seller_id AND fo.purchase_date BETWEEN :start_date AND :end_date
            GROUP BY {bucket_sql} ORDER BY bucket
        """), params).mappings().all()
    return {"metric_name": metric_name, "start_date": display_start, "end_date": display_end,
            "granularity": scope.granularity, "points": [{"bucket": str(row["bucket"]), "value": float(row["value"] or 0)} for row in rows],
            "source_table": "raw.order_items + analytics.fact_orders", "status": "ok" if rows else "not_found"}


def breakdown_scoped_product_category(scope: ControlledAnalysisScope, metric_name: str, limit: int = 10) -> dict[str, object]:
    """在锁定范围内按商品品类拆解指标。

    品类拆解依赖 `product_category_name_english` 列，它在 `analytics.v_product_category`
    视图中（`raw.products` + `raw.product_category_name_translation` 的 JOIN）。
    LEFT JOIN 保证了如果没有 product_id 或 category 为 NULL 的订单仍会归入 'UNKNOWN' 类别。
    """

    if metric_name not in scope.allowed_metrics:
        raise ValueError(f"Metric {metric_name!r} is not allowed by the locked analysis scope")
    with seller_query_transaction(engine, scope) as connection:
        rows = connection.execute(text(f"""
            WITH grouped AS (
                SELECT COALESCE(cat.product_category_name_english, 'UNKNOWN') AS dimension_value,
                       {metric_sql(metric_name)} AS value
                {_FROM_CLAUSE}
                LEFT JOIN analytics.v_product_category cat ON oi.product_id = cat.product_id
                WHERE oi.seller_id=:seller_id AND fo.purchase_date BETWEEN :start_date AND :end_date
                GROUP BY COALESCE(cat.product_category_name_english, 'UNKNOWN')
            )
            SELECT dimension_value, value, SUM(value) OVER () AS population_total
            FROM grouped
            ORDER BY value DESC NULLS LAST LIMIT :limit
        """), _scope_params(scope) | {"limit": min(max(limit, 1), 100)}).mappings().all()
    values = [(str(row["dimension_value"]), float(row["value"] or 0)) for row in rows]
    total = sum(value for _, value in values)
    return {"metric_name": metric_name, "metric_date": f"{scope.default_start_date.isoformat()} 至 {scope.default_end_date.isoformat()}",
            "dimension": "product_category", "items": [{"dimension_value": name, "value": value, "share": round(value * 100 / total, 2) if total else None} for name, value in values],
            "population_total": float(rows[0].get("population_total", total) or 0) if rows else 0.0,
            "source_table": "raw.order_items + analytics.v_product_category + analytics.fact_orders", "status": "ok" if values else "not_found"}


def breakdown_scoped_order_count_allocated(
    scope: ControlledAnalysisScope,
    *,
    limit: int = 10,
    dimension: str = "product_category",
) -> dict[str, object]:
    """在锁定店铺范围内按品类比例分摊去重订单量。

    一个订单在该店铺范围内涉及 k 个品类时，每个品类分 1/k 个订单；各品类
    分摊值之和等于该店铺同口径去重订单数。品类集合只取本店铺的商品行，
    不会因为订单同时属于其他店铺而改变分摊比例。
    """

    if dimension != "product_category":
        raise ValueError(f"Unsupported order allocation dimension: {dimension}")
    if "order_count" not in scope.allowed_metrics:
        raise ValueError("Metric 'order_count' is not allowed by the locked analysis scope")
    with seller_query_transaction(engine, scope) as connection:
        rows = _scoped_order_count_allocation_rows(
            connection,
            seller_id=scope.seller_id,
            start_date=scope.default_start_date,
            end_date=scope.default_end_date,
            label="seller breakdown",
        )
    return {
        "metric_name": "order_count",
        "metric_date": f"{scope.default_start_date.isoformat()} 至 {scope.default_end_date.isoformat()}",
        "dimension": dimension,
        "items": allocated_items(rows[0], total=rows[1], limit=limit),
        "population_total": round(rows[1], 6),
        "source_table": "analytics.v_delivered_order_items + analytics.v_product_category",
        "status": "ok" if rows[0] else "not_found",
        "method": ALLOCATION_METHOD,
        "allocation_method": ALLOCATION_METHOD,
        "allocation_note": ALLOCATION_NOTE,
    }


def compare_scoped_order_count_allocation(
    scope: ControlledAnalysisScope,
    current_start: date,
    current_end: date,
    baseline_start: date,
    baseline_end: date,
    *,
    limit: int = 10,
) -> dict[str, object]:
    """Compare seller-scoped distinct order counts with proportional allocation.

    Both periods must already lie inside the locked Explorer range; the
    seller-scope transaction revalidates that boundary before either query.
    """

    if "order_count" not in scope.allowed_metrics:
        raise ValueError("Metric 'order_count' is not allowed by the locked analysis scope")
    for start, end in ((current_start, current_end), (baseline_start, baseline_end)):
        if start < scope.default_start_date or end > scope.default_end_date or end < start:
            raise PermissionError("Order allocation period exceeds the locked analysis scope")
    with seller_query_transaction(engine, scope) as connection:
        current_rows, current_total = _scoped_order_count_allocation_rows(
            connection,
            seller_id=scope.seller_id,
            start_date=current_start,
            end_date=current_end,
            label="seller current period",
        )
        baseline_rows, baseline_total = _scoped_order_count_allocation_rows(
            connection,
            seller_id=scope.seller_id,
            start_date=baseline_start,
            end_date=baseline_end,
            label="seller baseline period",
        )
    current_items, baseline_items = paired_allocated_items(
        current_rows, baseline_rows,
        current_total=current_total, baseline_total=baseline_total, limit=limit,
    )
    return {
        "metric_name": "order_count",
        "dimension": "product_category",
        "current_date": current_end.isoformat(),
        "baseline_date": baseline_end.isoformat(),
        "current_period": {"start_date": current_start.isoformat(), "end_date": current_end.isoformat()},
        "baseline_period": {"start_date": baseline_start.isoformat(), "end_date": baseline_end.isoformat()},
        "current_items": current_items,
        "baseline_items": baseline_items,
        "current_total": round(current_total, 6),
        "baseline_total": round(baseline_total, 6),
        "total_delta": round(current_total - baseline_total, 6),
        "source_table": "analytics.v_delivered_order_items + analytics.v_product_category",
        "status": "ok" if current_rows or baseline_rows else "not_found",
        "method": ALLOCATION_METHOD,
        "allocation_method": ALLOCATION_METHOD,
        "allocation_note": ALLOCATION_NOTE,
    }


def _scoped_order_count_allocation_rows(
    connection: Any,
    *,
    seller_id: str,
    start_date: date,
    end_date: date,
    label: str,
) -> tuple[list[tuple[str, float]], float]:
    rows = connection.execute(text(f"""
        WITH order_dimension AS (
            SELECT DISTINCT
                oi.order_id,
                COALESCE(cat.product_category_name_english, 'UNKNOWN') AS dimension_value
            {_FROM_CLAUSE}
            LEFT JOIN analytics.v_product_category cat ON oi.product_id = cat.product_id
            WHERE oi.seller_id=:seller_id
              AND fo.purchase_date BETWEEN :start_date AND :end_date
        ),
        counted AS (
            SELECT
                order_id,
                dimension_value,
                COUNT(*) OVER (PARTITION BY order_id) AS dimension_count
            FROM order_dimension
        ),
        grouped AS (
            SELECT dimension_value, SUM(1.0 / dimension_count) AS allocated_value
            FROM counted
            GROUP BY dimension_value
        ),
        total AS (
            SELECT COUNT(DISTINCT order_id) AS total_orders FROM order_dimension
        )
        SELECT g.dimension_value, g.allocated_value, t.total_orders
        FROM grouped g CROSS JOIN total t
        ORDER BY g.allocated_value DESC, g.dimension_value ASC
    """), {
        "seller_id": seller_id,
        "start_date": start_date,
        "end_date": end_date,
    }).mappings().all()
    values = [
        (str(row["dimension_value"]), float(row["allocated_value"] or 0))
        for row in rows
    ]
    total = float(rows[0]["total_orders"] or 0) if rows else 0.0
    require_reconciled(values, total, label=label)
    return values, total


def breakdown_scoped_month_position(
    scope: ControlledAnalysisScope, metric_name: str, *, product_category_filter: str | None = None,
) -> dict[str, object]:
    """在锁定范围内按月初/月中/月末分桶拆解指标。

    这是 month_position 派生维度的受控查询实现，复用与品类拆解相同的安全约束
    （同表、同 FROM 子句、同 seller_id/date 范围）。使用独立的 breakdown_scoped_
    month_position 而不是通用 _TIME_DIMENSION_SQL 路径，是因为：
    - 月初/月中/月末需要固定排序（1~10 日 → 月初，11~20 → 月中，21~月末）。
    - _TIME_DIMENSION_SQL 中的通用路径按 SQL 引擎的原生 GROUP BY 序（字典序或
      值序）输出，不会按 月初→月中→月末 排。

    计算口径：
      以 fo.purchase_date（下单日期）为基准，而不是 order_delivered_customer_date。
      这意味着"月末下单、次月初送达"的订单仍归月末。与 daily_metrics 保持一致。

    参数校验：
      metric_name 限制在 scope.allowed_metrics（5 个核心指标）。
    """

    if metric_name not in scope.allowed_metrics:
        raise ValueError(f"Metric {metric_name!r} is not allowed by the locked analysis scope")
    category_join = "LEFT JOIN analytics.v_product_category cat ON oi.product_id = cat.product_id" if product_category_filter is not None else ""
    category_predicate = "AND COALESCE(cat.product_category_name_english, 'UNKNOWN')=:product_category_filter" if product_category_filter is not None else ""
    parameters = _scope_params(scope)
    if product_category_filter is not None:
        parameters["product_category_filter"] = product_category_filter
    with seller_query_transaction(engine, scope) as connection:
        rows = connection.execute(text(f"""
            WITH grouped AS (
                SELECT
                    CASE
                        WHEN EXTRACT(DAY FROM fo.purchase_date) BETWEEN 1 AND 10 THEN '月初(1-10日)'
                        WHEN EXTRACT(DAY FROM fo.purchase_date) BETWEEN 11 AND 20 THEN '月中(11-20日)'
                        ELSE '月末(21-31日)'
                    END AS dimension_value,
                    {metric_sql(metric_name)} AS value
                {_FROM_CLAUSE}
                {category_join}
                WHERE oi.seller_id=:seller_id
                  AND fo.purchase_date BETWEEN :start_date AND :end_date
                  {category_predicate}
                GROUP BY dimension_value
            )
            SELECT dimension_value, value, SUM(value) OVER () AS population_total
            FROM grouped
        """), parameters).mappings().all()
    # 固定顺序：月初 → 月中 → 月末
    segment_order = {"月初(1-10日)": 0, "月中(11-20日)": 1, "月末(21-31日)": 2}
    values = sorted(
        [(str(row["dimension_value"]), float(row["value"] or 0)) for row in rows],
        key=lambda item: segment_order.get(item[0], 99),
    )
    total = sum(value for _, value in values)
    return {
        "metric_name": metric_name,
        "metric_date": f"{scope.default_start_date.isoformat()} 至 {scope.default_end_date.isoformat()}",
        "dimension": "month_position",
        "items": [{"dimension_value": name, "value": value, "share": round(value * 100 / total, 2) if total else None} for name, value in values],
        "population_total": float(rows[0].get("population_total", total) or 0) if rows else 0.0,
        "source_table": "raw.order_items + analytics.fact_orders" + (" + analytics.v_product_category" if product_category_filter is not None else ""),
        "status": "ok" if values else "not_found",
    }


def breakdown_scoped_time_dimension(scope: ControlledAnalysisScope, metric_name: str, dimension: str) -> dict[str, object]:
    """在锁定范围内按派生时间维度（周几/时段/季度/半年）执行拆解。

    这个函数是 _TIME_DIMENSION_SQL 中 4 个派生维度的统一入口。
    与 breakdown_scoped_product_category 共享相同的 _FROM_CLAUSE 和 metric_sql()。

    注意 month_position 不经过此路径——它由独立的 breakdown_scoped_month_position 处理。

    排序规则：
        周几：周一→周日；上半年下半年：上半年→下半年；季度：Q1→Q4；时段：字母序（00:00→23:00）。
        这些排序是硬编码在 ranks 字典里的，因为 SQL 的 GROUP BY 不会按语义序输出。
        ranks 未命中的值（如未知的时段标签）会排到末尾（99）。

    数据口径（source_table 标注）：
        时段拆解使用 fo.order_purchase_timestamp（不含时区信息的数据库本地时间）。
        如果源数据的 timestamp 是 UTC 但卖家的业务时间是 BRT，此处不会做时区转换。
        这一点在返回的 source_table 中显式标注了 "(hour uses order creation time)"。
    """

    if metric_name not in scope.allowed_metrics or dimension not in scope.allowed_dimensions:
        raise ValueError("Metric or dimension is not allowed by the locked analysis scope")
    expression = _TIME_DIMENSION_SQL.get(dimension)
    if expression is None:
        if dimension == "month_position":
            return breakdown_scoped_month_position(scope, metric_name)
        raise ValueError(f"Unsupported time dimension {dimension!r}")
    with seller_query_transaction(engine, scope) as connection:
        rows = connection.execute(text(f"""
            SELECT {expression} AS dimension_value, {metric_sql(metric_name)} AS value
            {_FROM_CLAUSE}
            WHERE oi.seller_id=:seller_id AND fo.purchase_date BETWEEN :start_date AND :end_date
            GROUP BY {expression}
        """), _scope_params(scope)).mappings().all()
    ranks = {"周一": 1, "周二": 2, "周三": 3, "周四": 4, "周五": 5, "周六": 6, "周日": 7,
             "上半年": 1, "下半年": 2, "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    values = sorted([(str(row["dimension_value"]), float(row["value"] or 0)) for row in rows], key=lambda x: ranks.get(x[0], int(x[0][:2]) if x[0][:2].isdigit() else 99))
    total = sum(value for _, value in values)
    return {"metric_name": metric_name, "metric_date": f"{scope.default_start_date.isoformat()} 至 {scope.default_end_date.isoformat()}",
            "dimension": dimension, "items": [{"dimension_value": name, "value": value, "share": round(value * 100 / total, 2) if total else None} for name, value in values],
            "source_table": "raw.order_items + analytics.fact_orders (hour uses order creation time)", "status": "ok" if values else "not_found"}


def query_scoped_order_funnel(scope: ControlledAnalysisScope) -> dict[str, object]:
    """查询店铺级别的订单阶段漏斗数据。

    返回按天聚合的漏斗阶段计数（全部订单 → 已审批 → 已发货 → 已送达）。
    阶段由时间戳推导（approved_at / carrier_at / delivered_customer_at），
    不是 order_status 快照。  ``delivered_missing_timestamp`` 标记
    order_status='delivered' 但签收时间为空的数据质量问题。
    """
    if "order_approval_rate" not in scope.allowed_metrics:
        raise ValueError("Order funnel is not allowed by the locked analysis scope")
    params = _scope_params(scope)
    with seller_query_transaction(engine, scope) as connection:
        rows = connection.execute(text(f"""
            SELECT DATE(o.order_purchase_timestamp) AS purchase_date,
                   COUNT(DISTINCT o.order_id) AS total_orders,
                   COUNT(DISTINCT o.order_id) FILTER (WHERE o.order_approved_at IS NOT NULL)
                       AS stage_approved,
                   COUNT(DISTINCT o.order_id) FILTER (WHERE o.order_delivered_carrier_date IS NOT NULL)
                       AS stage_shipped,
                   COUNT(DISTINCT o.order_id) FILTER (WHERE o.order_delivered_customer_date IS NOT NULL)
                       AS stage_delivered,
                   COUNT(DISTINCT o.order_id) FILTER (WHERE o.order_status = 'delivered'
                       AND o.order_delivered_customer_date IS NULL)
                       AS delivered_missing_timestamp,
                   COUNT(DISTINCT o.order_id) FILTER (WHERE o.order_status IN ('canceled','unavailable'))
                       AS final_lost
            FROM raw.orders o
            JOIN raw.order_items oi ON oi.order_id = o.order_id
            WHERE oi.seller_id = :seller_id
              AND DATE(o.order_purchase_timestamp) BETWEEN :start_date AND :end_date
            GROUP BY DATE(o.order_purchase_timestamp)
            ORDER BY purchase_date
        """), params).mappings().all()
    has_data_quality_issue = False
    funnel_data = []
    for row in rows:
        date_str = str(row["purchase_date"])
        total = int(row["total_orders"] or 0)
        delivered = int(row["stage_delivered"] or 0)
        missing_ts = int(row["delivered_missing_timestamp"] or 0)
        if missing_ts > 0:
            has_data_quality_issue = True
        stages = [
            {"label": "全部订单", "count": total},
            {"label": "已审批", "count": int(row["stage_approved"] or 0)},
            {"label": "已发货", "count": int(row["stage_shipped"] or 0)},
            {"label": "已送达", "count": delivered},
        ]
        for s in stages[1:]:
            s["rate"] = round(s["count"] * 100.0 / total, 1) if total > 0 else None
        funnel_data.append({"date": date_str, "stages": stages})
    return {
        "metric_name": "order_funnel",
        "start_date": scope.default_start_date.isoformat(),
        "end_date": scope.default_end_date.isoformat(),
        "funnel_data": funnel_data,
        "source_table": "raw.orders + raw.order_items",
        "data_quality_note": (
            f"{sum(int(r['delivered_missing_timestamp'] or 0) for r in rows)} 笔订单状态为 delivered 但缺少签收时间戳"
            if has_data_quality_issue else ""
        ),
        "status": "ok" if funnel_data else "not_found",
    }
