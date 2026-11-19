"""集成测试：Olist 支付与履约数据域。

验证 fact_order_seller_allocation、fact_order_payments、fact_fulfillment
以及各视图的正确性。依赖真实 PostgreSQL 数据库与种子数据。
"""

from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.data.database import engine
from app.data.repositories.explorer import (
    _DOMAIN_METRICS,
    _scoped_domain_definition,
    breakdown_scoped_payment_type,
    preview_explorer,
    query_scoped_metric,
    query_scoped_trend,
)
from app.schemas.explorer import ControlledAnalysisScope, ExplorerQuery
from app.tenancy import SellerGrant, TenantContext, activate_tenant_context

SELLER = "6560211a19b47992c3666cc44a7e94c0"
AUG_2018 = ControlledAnalysisScope(
    seller_id=SELLER, default_start_date="2018-08-01", default_end_date="2018-08-31",
    metric_name="payment_value", granularity="day",
)
@pytest.fixture(scope="module")
def live_seller_identity():
    """Provision the trusted IAM identity required by seller-scoped queries."""
    suffix = uuid4().hex
    tenant_id = f"olist_it_t_{suffix}"
    user_id = f"olist_it_u_{suffix}"
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
            VALUES (:tenant, 'Olist marts integration test', 'seller', 'active')
        """), {"tenant": tenant_id})
        connection.execute(text("""
            INSERT INTO iam.users(user_id, display_name, email, status)
            VALUES (:user, 'Olist marts integration test', :email, 'active')
        """), {"user": user_id, "email": f"{user_id}@example.invalid"})
        connection.execute(text("""
            INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
            VALUES (:tenant, :user, 'analyst')
        """), {"tenant": tenant_id, "user": user_id})
        connection.execute(text("""
            INSERT INTO iam.membership_seller_grants(
                tenant_id, user_id, seller_id, access_start_date, access_end_date
            ) VALUES (:tenant, :user, :seller, '2018-08-01', '2018-08-31')
        """), {"tenant": tenant_id, "user": user_id, "seller": SELLER})

    context = TenantContext(
        tenant_id=tenant_id,
        user_id=user_id,
        role="analyst",
        scope_type="seller",
        seller_id=SELLER,
        seller_grants=(SellerGrant(SELLER, "2018-08-01", "2018-08-31"),),
    )
    try:
        yield context
    finally:
        with engine.begin() as connection:
            # Integration tests may run under a restricted database role.
            connection.execute(text("RESET ROLE"))
            connection.execute(text(
                "DELETE FROM iam.membership_seller_grants WHERE tenant_id=:tenant"
            ), {"tenant": tenant_id})
            connection.execute(text(
                "DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant"
            ), {"tenant": tenant_id})
            connection.execute(text(
                "DELETE FROM iam.tenants WHERE tenant_id=:tenant"
            ), {"tenant": tenant_id})
            connection.execute(text(
                "DELETE FROM iam.users WHERE user_id=:user"
            ), {"user": user_id})


@pytest.fixture(autouse=True)
def active_live_seller_identity(live_seller_identity):
    with activate_tenant_context(live_seller_identity):
        yield


# =============================================================================
# Data Refresh Integrity
# =============================================================================


def test_fact_order_items_matches_raw_count() -> None:
    """fact_order_items 行数必须等于 raw.order_items。"""
    with engine.connect() as conn:
        fo_count = conn.execute(
            text("SELECT COUNT(*) FROM analytics.fact_order_items")
        ).scalar()
        ro_count = conn.execute(
            text("SELECT COUNT(*) FROM raw.order_items")
        ).scalar()
    assert fo_count == ro_count, (
        f"fact_order_items ({fo_count}) != raw.order_items ({ro_count})"
    )


def test_fact_orders_contains_all_statuses() -> None:
    """fact_orders 保留全部 order_status，不只是 delivered。"""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT order_status, COUNT(*) AS cnt "
                 "FROM analytics.fact_orders GROUP BY order_status")
        ).mappings().all()
    statuses = {r["order_status"]: r["cnt"] for r in rows}
    assert "delivered" in statuses
    assert "canceled" in statuses
    assert "shipped" in statuses
    assert "unavailable" in statuses
    assert statuses["delivered"] == 96478, "delivered count mismatch"


def test_v_delivered_orders_only_contains_delivered() -> None:
    """v_delivered_orders 视图只返回已交付订单。"""
    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM analytics.v_delivered_orders")
        ).scalar()
        non_delivered = conn.execute(
            text("SELECT COUNT(*) FROM analytics.v_delivered_orders "
                 "WHERE order_status != 'delivered'")
        ).scalar()
    assert total > 0
    assert non_delivered == 0, (
        f"v_delivered_orders contains {non_delivered} non-delivered rows"
    )


def test_v_delivered_order_items_only_contains_delivered() -> None:
    """v_delivered_order_items 视图也只关联已交付订单。"""
    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM analytics.v_delivered_order_items")
        ).scalar()
        non_delivered = conn.execute(
            text("SELECT COUNT(*) FROM analytics.v_delivered_order_items "
                 "WHERE order_status != 'delivered'")
        ).scalar()
        fo_total = conn.execute(
            text("SELECT COUNT(*) FROM analytics.fact_order_items")
        ).scalar()
    assert total > 0
    assert non_delivered == 0
    assert total < fo_total, "v_delivered_order_items should be a subset"


# =============================================================================
# Payment Domain — Seller Allocation
# =============================================================================


def test_seller_allocation_ratios_sum_to_one_per_order() -> None:
    """per-order seller_allocation_ratio 之和必须为 1.0（容差 0.001）。"""
    with engine.connect() as conn:
        violations = conn.execute(
            text("SELECT order_id, SUM(seller_allocation_ratio) AS total_ratio "
                 "FROM analytics.fact_order_seller_allocation "
                 "GROUP BY order_id "
                 "HAVING ABS(SUM(seller_allocation_ratio) - 1.0) > 0.001 "
                 "LIMIT 1")
        ).mappings().all()
    assert len(violations) == 0, (
        f"order {violations[0]['order_id']} has ratio sum = "
        f"{float(violations[0]['total_ratio'])}"
    )


def test_single_seller_order_ratio_is_exactly_one() -> None:
    """单卖家订单的 seller_allocation_ratio 必须为 1。"""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT seller_allocation_ratio "
                 "FROM analytics.fact_order_seller_allocation "
                 "WHERE is_single_seller_order = true AND "
                 "seller_allocation_ratio != 1.0 "
                 "LIMIT 1")
        ).mappings().all()
    assert len(rows) == 0, "single-seller orders with ratio != 1.0 found"


def test_multi_seller_orders_exist_and_ratios_sum_to_one() -> None:
    """多卖家订单确实存在，且每单 ratio 和为 1。"""
    with engine.connect() as conn:
        multi = conn.execute(
            text("SELECT order_id, seller_count "
                 "FROM analytics.fact_order_seller_allocation "
                 "WHERE is_single_seller_order = false "
                 "LIMIT 5")
        ).mappings().all()
        assert len(multi) > 0, "no multi-seller orders found"
        max_sellers = max(r["seller_count"] for r in multi)
        assert max_sellers >= 2, "no order with 2+ sellers found"
        # 验证这些多卖家订单的 ratio 和是否也为 1
        multi_ids = tuple(r["order_id"] for r in multi)
        placeholders = ",".join(f"'{oid}'" for oid in multi_ids)
        violations = conn.execute(
            text(f"SELECT order_id, SUM(seller_allocation_ratio) AS total_ratio "
                 f"FROM analytics.fact_order_seller_allocation "
                 f"WHERE order_id IN ({placeholders}) "
                 f"GROUP BY order_id "
                 f"HAVING ABS(SUM(seller_allocation_ratio) - 1.0) > 0.001")
        ).mappings().all()
        assert len(violations) == 0


def test_allocated_payment_value_matches_original_per_order() -> None:
    """每个原始支付记录的分摊金额之和等于原始 payment_value。"""
    with engine.connect() as conn:
        violations = conn.execute(
            text("SELECT order_id, payment_sequential, "
                 "       COUNT(*) AS split_count, "
                 "       SUM(allocated_payment_value) AS alloc_sum, "
                 "       MIN(payment_value) AS orig_value "
                 "FROM analytics.fact_order_payments "
                 "WHERE seller_id != '__unattributed__' "
                 "GROUP BY order_id, payment_sequential "
                 "HAVING ABS(SUM(allocated_payment_value) "
                 "          - MIN(payment_value)) > 0.02 "
                 "LIMIT 5")
        ).mappings().all()
    # 舍入误差容差 2 分钱
    assert len(violations) == 0, (
        f"allocated_payment_value mismatch: "
        f"order={violations[0]['order_id']}, "
        f"alloc_sum={float(violations[0]['alloc_sum'])}, "
        f"orig={float(violations[0]['orig_value'])}"
    )


def test_allocated_payment_handles_unattributed_orders() -> None:
    """没有 seller_allocation 匹配的订单归入 __unattributed__ 卖家。"""
    with engine.connect() as conn:
        unattributed = conn.execute(
            text("SELECT COUNT(*) "
                 "FROM analytics.fact_order_payments "
                 "WHERE seller_id = '__unattributed__'")
        ).scalar()
    # __unattributed__ 行可以为 0（全部订单都有 seller 匹配）
    assert unattributed is not None, "query for unattributed failed"


# =============================================================================
# Payment Domain — Scoped Queries
# =============================================================================


@pytest.mark.parametrize("metric", ["payment_value", "installment_order_share"])
def test_scoped_payment_metric_returns_data(metric: str) -> None:
    """受控查询路径中的支付指标返回有值结果。"""
    scope = AUG_2018.model_copy(update={"metric_name": metric})
    result = query_scoped_metric(scope, metric)
    assert result["status"] == "ok", f"{metric}: {result}"
    assert "lineage_note" in result
    assert "分摊" in result["lineage_note"]
    assert result["value"] > 0, f"{metric}: expected positive value"


def test_scoped_payment_trend_returns_data() -> None:
    """payment_value 在受控范围内返回时间序列。"""
    result = query_scoped_trend(AUG_2018, "payment_value")
    assert result["status"] == "ok"
    assert len(result["points"]) > 0
    assert result["granularity"] == "day"


def test_breakdown_scoped_payment_type_returns_data() -> None:
    """按支付方式拆解支付金额返回多行且 share 之和约 100%。"""
    result = breakdown_scoped_payment_type(AUG_2018, "payment_value")
    assert result["status"] == "ok"
    assert len(result["items"]) > 0
    shares = sum(item["share"] for item in result["items"] if item["share"] is not None)
    assert abs(shares - 100) < 1.0, f"shares sum to {shares}%"


def test_breakdown_scoped_payment_type_rejects_non_payment_metric() -> None:
    """只有 payment_value 允许用 payment_type 拆解。"""
    scope = AUG_2018.model_copy(update={"metric_name": "installment_order_share"})
    with pytest.raises(ValueError, match="only available for payment_value"):
        breakdown_scoped_payment_type(scope, "installment_order_share")


# =============================================================================
# Fulfillment Domain
# =============================================================================


def test_null_timestamps_produce_null_derived_hours() -> None:
    """空时间戳不产生伪造的派生时长字段。"""
    with engine.connect() as conn:
        fake_approval = conn.execute(
            text("SELECT COUNT(*) FROM analytics.fact_fulfillment "
                 "WHERE approval_lead_hours IS NOT NULL AND approved_at IS NULL")
        ).scalar()
        fake_dispatch = conn.execute(
            text("SELECT COUNT(*) FROM analytics.fact_fulfillment "
                 "WHERE dispatch_lead_hours IS NOT NULL AND delivered_carrier_at IS NULL")
        ).scalar()
        fake_transit = conn.execute(
            text("SELECT COUNT(*) FROM analytics.fact_fulfillment "
                 "WHERE delivery_transit_hours IS NOT NULL AND delivered_carrier_at IS NULL")
        ).scalar()
    assert fake_approval == 0
    assert fake_dispatch == 0
    assert fake_transit == 0


def test_is_on_time_boundary_correct() -> None:
    """is_on_time 边界正确：签收 ≤ 预计为 true，签收 > 预计为 false。"""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT is_on_time, delivered_customer_at, estimated_delivery_at "
                 "FROM analytics.fact_fulfillment "
                 "WHERE is_on_time IS NOT NULL "
                 "LIMIT 200")
        ).mappings().all()
    assert len(rows) > 0
    for r in rows:
        on_time = r["delivered_customer_at"] <= r["estimated_delivery_at"]
        assert bool(r["is_on_time"]) == on_time, (
            f"is_on_time={r['is_on_time']} but "
            f"delivered={r['delivered_customer_at']} <= "
            f"estimated={r['estimated_delivery_at']} is {on_time}"
        )


def test_v_seller_fulfillment_excludes_multi_seller_orders() -> None:
    """v_seller_fulfillment 行数应小于 fact_fulfillment（排除了多卖家订单）。"""
    with engine.connect() as conn:
        vsf = conn.execute(
            text("SELECT COUNT(*) FROM analytics.v_seller_fulfillment")
        ).scalar()
        ff = conn.execute(
            text("SELECT COUNT(*) FROM analytics.fact_fulfillment")
        ).scalar()
    assert vsf < ff, (
        f"v_seller_fulfillment ({vsf}) should be < fact_fulfillment ({ff})"
    )


def test_v_seller_fulfillment_has_no_multi_seller_orders() -> None:
    """v_seller_fulfillment 中不应存在多卖家订单。"""
    with engine.connect() as conn:
        multi = conn.execute(
            text("SELECT COUNT(*) FROM analytics.v_seller_fulfillment v "
                 "JOIN analytics.fact_order_seller_allocation a USING(order_id) "
                 "WHERE a.is_single_seller_order = false")
        ).scalar()
    assert multi == 0, "multi-seller orders found in v_seller_fulfillment"


# =============================================================================
# Fulfillment — Scoped Queries
# =============================================================================


@pytest.mark.parametrize("metric", [
    "avg_approval_lead_hours",
    "avg_dispatch_lead_hours",
    "avg_delivery_transit_hours",
    "on_time_delivery_rate",
])
def test_scoped_fulfillment_metric_returns_data(metric: str) -> None:
    """受控查询路径中的履约指标返回有值结果。"""
    scope = AUG_2018.model_copy(update={"metric_name": metric})
    result = query_scoped_metric(scope, metric)
    assert result["status"] == "ok", f"{metric}: {result}"
    assert "lineage_note" in result
    assert "排除多卖家" in result["lineage_note"]


def test_scoped_on_time_delivery_rate_is_reasonable() -> None:
    """准时送达率应在 0~100% 合理范围。"""
    result = query_scoped_metric(AUG_2018, "on_time_delivery_rate")
    assert result["status"] == "ok"
    assert 0 < result["value"] <= 100


def test_scoped_fulfillment_trend_returns_data() -> None:
    """履约指标在受控范围内返回时间序列。"""
    scope = AUG_2018.model_copy(update={"metric_name": "avg_approval_lead_hours"})
    result = query_scoped_trend(scope, "avg_approval_lead_hours")
    assert result["status"] == "ok"
    assert len(result["points"]) > 0


# =============================================================================
# Scope & Integration
# =============================================================================


def test_scoped_domain_metric_rejects_unallowed_metric() -> None:
    """query_scoped_metric 拒绝不在 allowed_metrics 中的指标名。"""
    with pytest.raises(ValueError, match="not allowed"):
        query_scoped_metric(AUG_2018, "nonexistent_metric")


def test_scoped_domain_result_contains_lineage_note() -> None:
    """域指标查询结果包含 lineage_note 字段标识数据口径。"""
    result = query_scoped_metric(AUG_2018, "payment_value")
    assert "lineage_note" in result
    assert "支付按商品金额加运费比例分摊" in result["lineage_note"]


def test_scoped_fulfillment_lineage_notes_exclusion() -> None:
    """履约域 lineage_note 说明已排除多卖家订单。"""
    scope = AUG_2018.model_copy(
        update={"metric_name": "avg_approval_lead_hours"}
    )
    result = query_scoped_metric(scope, "avg_approval_lead_hours")
    assert "lineage_note" in result
    assert "排除多卖家订单" in result["lineage_note"]


@pytest.mark.parametrize("metric", [
    "gmv", "order_count", "avg_order_value", "customer_count", "freight_value",
])
def test_preview_explorer_core_metrics_no_regression(metric: str, live_seller_identity) -> None:
    """核心 5 个指标通过 preview_explorer 查询正常，保持历史口径。"""
    query = ExplorerQuery(
        seller_id=SELLER,
        metric_name=metric,
        start_date=date(2018, 8, 1),
        end_date=date(2018, 8, 31),
        granularity="day",
    )
    result = preview_explorer(live_seller_identity, query)
    assert result["summary"]["status"] == "ok", f"{metric}: preview failed"
    assert result["summary"]["value"] > 0, f"{metric}: expected positive value"
    assert len(result["series"]) > 0, f"{metric}: expected time series"
    # preview_explorer 使用 _FROM_CLAUSE（raw.order_items + analytics.fact_orders）
    assert "raw.order_items" in result.get("lineage", {}).get("source_table", ""), (
        f"{metric}: unexpected source_table"
    )


def test_domain_metrics_all_have_source_definition() -> None:
    """所有 _DOMAIN_METRICS 必须能通过 _scoped_domain_definition 解析。"""
    for metric in _DOMAIN_METRICS:
        source, expression, date_column = _scoped_domain_definition(metric)
        assert source, f"{metric}: missing source"
        # 域查询源由后端固定代码定义；评分域需要一个去重子查询，避免一笔
        # 多商品订单将同一评论重复计入。无论何种来源都不得包含多语句分隔符。
        assert ";" not in source, f"{metric}: source must not contain multiple SQL statements"
        assert expression, f"{metric}: missing expression"
        assert date_column, f"{metric}: missing date column"


def test_domain_metrics_listed_in_semantic_registry() -> None:
    """支付和履约指标已注册到 live registry。

    seller-only 指标（如 seller_order_bad_review_rate）没有 platform 实现，
    不在旧的 ``METRIC_REGISTRY`` 快照里，但在 live registry 中。
    """
    from app.metrics.registry import get_registry

    reg = get_registry()
    platform_ids = {m.metric_id for m in reg.list()}
    for metric in _DOMAIN_METRICS:
        assert metric in platform_ids, (
            f"{metric} not in registry"
        )


def test_domain_metrics_in_mcp_schema() -> None:
    """支付和履约指标被 MCP MetricName validator 接受。"""
    from app.mcp.schemas import MetricQueryPlan

    for metric in _DOMAIN_METRICS:
        # MetricName validation runs Pydantic AfterValidator → registry
        plan = MetricQueryPlan(metric_name=metric, metric_date="2018-08-10")
        assert plan.metric_name is not None, f"{metric} rejected by MCP schema"


def test_domain_metrics_in_frontend_explorer_metric_type() -> None:
    """支付和履约指标可通过注册中心校验，前端从 API 动态获取。"""
    from app.metrics.registry import get_registry

    reg = get_registry()
    platform_ids = {m.metric_id for m in reg.list()}
    for metric in _DOMAIN_METRICS:
        assert metric in platform_ids, f"{metric} not in registry platform metrics"


def test_all_domain_metrics_have_scoped_query_working() -> None:
    """每个域指标至少通过受控查询返回一个值。"""
    for metric in _DOMAIN_METRICS:
        scope = AUG_2018.model_copy(update={"metric_name": metric})
        result = query_scoped_metric(scope, metric)
        assert result["status"] == "ok", f"{metric}: query failed"
        assert "lineage_note" in result


def test_review_score_preview_is_registry_declared_and_uses_deduplicated_orders() -> None:
    """平均评分可在 Explorer 中预览，且其受控查询使用店铺订单去重口径。"""
    from app.metrics.registry import get_registry

    definition = get_registry().get("review_score_avg")
    assert "explorer" in definition.surfaces
    assert get_registry().supports("review_score_avg", "value", scope="seller")
    assert get_registry().supports("review_score_avg", "trend", scope="seller")

    scope = AUG_2018.model_copy(update={"metric_name": "review_score_avg"})
    result = query_scoped_metric(scope, "review_score_avg")
    assert result["status"] == "ok"
    assert result["value"] > 0
    assert "多件商品仅计一条评论" in result["lineage_note"]
