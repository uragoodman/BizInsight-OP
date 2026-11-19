"""集成测试：受控月内分桶 + 真实数据库。

标记为 @pytest.mark.integration，有 DB 时自动跑，无 DB 时跳过。
"""

import pytest
from uuid import uuid4

from app.schemas.explorer import ControlledAnalysisScope
from app.data.repositories.explorer import breakdown_scoped_month_position, breakdown_scoped_product_category

SELLER = "6560211a19b47992c3666cc44a7e94c0"
AUG_2018 = ControlledAnalysisScope(
    seller_id=SELLER, default_start_date="2018-08-01", default_end_date="2018-08-31",
    metric_name="order_count", granularity="day",
)


@pytest.fixture(scope="module")
def live_seller_identity():
    """Create an isolated IAM principal for real-database scope tests."""
    from sqlalchemy import text
    from app.data.database import engine
    from app.tenancy import SellerGrant, TenantContext

    suffix = uuid4().hex
    tenant_id = f"scoped_it_t_{suffix}"
    user_id = f"scoped_it_u_{suffix}"
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
            VALUES (:tenant, 'scoped query integration test', 'seller', 'active')
        """), {"tenant": tenant_id})
        connection.execute(text("""
            INSERT INTO iam.users(user_id, display_name, email, status)
            VALUES (:user, 'scoped query integration test', :email, 'active')
        """), {"user": user_id, "email": f"{user_id}@example.invalid"})
        connection.execute(text("""
            INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
            VALUES (:tenant, :user, 'analyst')
        """), {"tenant": tenant_id, "user": user_id})
        connection.execute(text("""
            INSERT INTO iam.membership_seller_grants(
              tenant_id, user_id, seller_id, access_start_date, access_end_date
            ) VALUES (:tenant, :user, :seller, '2018-07-01', '2018-08-31')
        """), {"tenant": tenant_id, "user": user_id, "seller": SELLER})
    context = TenantContext(
        tenant_id=tenant_id,
        user_id=user_id,
        role="analyst",
        scope_type="seller",
        seller_id=SELLER,
        seller_grants=(SellerGrant(SELLER, "2018-07-01", "2018-08-31"),),
    )
    try:
        yield context
    finally:
        with engine.begin() as connection:
            connection.execute(text("RESET ROLE"))
            connection.execute(text("DELETE FROM iam.membership_seller_grants WHERE tenant_id=:tenant"), {"tenant": tenant_id})
            connection.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant"), {"tenant": tenant_id})
            connection.execute(text("DELETE FROM iam.tenants WHERE tenant_id=:tenant"), {"tenant": tenant_id})
            connection.execute(text("DELETE FROM iam.users WHERE user_id=:user"), {"user": user_id})


@pytest.fixture(autouse=True)
def active_live_seller_identity(live_seller_identity):
    from app.tenancy import activate_tenant_context

    with activate_tenant_context(live_seller_identity):
        yield


def test_db_reachable() -> None:
    """前提检查：数据库必须可连接且有种子数据。"""
    from app.data.database import engine
    from sqlalchemy import text
    with engine.connect() as conn:
        row = conn.execute(text("SELECT COUNT(*) AS cnt FROM raw.order_items")).mappings().one()
    assert row["cnt"] > 0, "seed data required"


@pytest.mark.parametrize("metric", [
    "order_count", "gmv", "avg_order_value", "customer_count", "freight_value",
])
def test_all_metrics_have_three_position_buckets(metric: str) -> None:
    """全部 5 个受控指标都返回月初/月中/月末三段。"""
    scope = AUG_2018.model_copy(update={"metric_name": metric})
    result = breakdown_scoped_month_position(scope, metric)
    assert result["status"] == "ok", f"{metric}: {result}"
    items = result["items"]
    assert len(items) == 3, f"{metric}: expected 3 buckets, got {len(items)}"
    # 固定排序
    assert items[0]["dimension_value"].startswith("月初")
    assert items[1]["dimension_value"].startswith("月中")
    assert items[2]["dimension_value"].startswith("月末")
    # 数据总量 > 0（有种子数据的卖家）
    total = sum(i["value"] for i in items)
    assert total > 0, f"{metric}: total must be > 0"
    # share 加起来约 100%
    shares = sum(i["share"] for i in items)
    assert abs(shares - 100) < 1.0, f"{metric}: shares sum to {shares}%"


def test_scoped_month_position_dates_are_within_scope() -> None:
    """日期绝不能超出锁定范围。"""
    result = breakdown_scoped_month_position(AUG_2018, "order_count")
    assert result["metric_date"] == "2018-08-01 至 2018-08-31", result["metric_date"]


def test_scoped_category_evidence_can_be_safely_drilled_into_month_position() -> None:
    categories = breakdown_scoped_product_category(AUG_2018, "order_count", limit=1)
    assert categories["status"] == "ok", categories
    selected_category = categories["items"][0]["dimension_value"]
    result = breakdown_scoped_month_position(
        AUG_2018, "order_count", product_category_filter=selected_category,
    )
    assert result["status"] == "ok", result
    assert result["source_table"].endswith("analytics.v_product_category")
    assert sum(item["value"] for item in result["items"]) > 0


def test_full_chain_via_query_scoped_business_data(live_seller_identity) -> None:
    """通过 query_scoped_business_data 验证完整链路。"""
    from unittest.mock import patch
    from app.mcp.schemas import MetricQueryPlan
    from app.metrics.authorization import authorize_metric
    from app.mcp.client import query_scoped_business_data

    scope_payload = AUG_2018.model_dump(mode="json")
    authorized = authorize_metric(
        "order_count",
        provenance_type="explorer_selection",
        provenance_id="integration-selection",
        scope="seller",
        scope_data={
            **scope_payload,
            "tenant_id": None,
            "user_id": None,
            "permissions": None,
        },
    )
    plan = MetricQueryPlan(
        query_type="metric_breakdown", metric_name="order_count",
        authorized_metric=authorized,
        dimension="month_position", metric_date="2018-08-01",
        tool_name="breakdown_scoped_month_position",
    )
    with patch("app.mcp.client.resolve_expression") as mock_resolve:
        mock_resolve.return_value = None
        from app.tenancy import activate_tenant_context
        _scope_activation = activate_tenant_context(live_seller_identity)
        _scope_activation.__enter__()
        result = query_scoped_business_data(
            "评估本月销售量在月初、月中、月末的销量比较",
            AUG_2018.model_dump(mode="json"),
            plan=plan,
        )
        _scope_activation.__exit__(None, None, None)
    assert result["tool_name"] == "breakdown_scoped_month_position"
    assert result["result"]["status"] == "ok"
    items = result["result"]["breakdown_result"]["items"]
    assert len(items) == 3
    assert items[0]["value"] > 0


@pytest.mark.parametrize("scope_start,scope_end,question", [
    ("2018-08-01", "2018-08-31", "月初/月中/月末分布"),
    ("2018-07-01", "2018-07-31", "月初/月中/月末分布"),
])
def test_different_months_resolve_correctly(scope_start, scope_end, question) -> None:
    """不同锁定月份都正确返回对应月内的三段分桶。"""
    from datetime import date
    start = date.fromisoformat(scope_start)
    end = date.fromisoformat(scope_end)
    scope = AUG_2018.model_copy(update={
        "default_start_date": start, "default_end_date": end,
        "seller_id": SELLER,
    })
    result = breakdown_scoped_month_position(scope, "order_count")
    assert result["status"] == "ok"
    assert scope_start in result["metric_date"]
