"""P1/P2/P3 店铺扩展指标的真实 Registry 执行回归。"""

from datetime import date
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.data.database import engine
from app.metrics.registry import get_registry
from app.schemas.explorer import ControlledAnalysisScope
from app.tenancy import SellerGrant, TenantContext, activate_tenant_context


P1_P2_P3_METRICS = {
    "seller_item_gross_value", "seller_avg_item_price", "seller_free_shipping_order_rate",
    "seller_avg_freight_per_order", "seller_avg_payment_installments",
    "seller_credit_card_payment_share", "seller_boleto_payment_share",
    "seller_active_product_count", "seller_active_category_count",
    "seller_top1_product_gross_share", "seller_top3_category_gross_share", "seller_new_product_gross_share",
    "seller_historical_returning_customer_count", "seller_historical_returning_customer_rate",
    "seller_new_customer_count", "seller_new_customer_rate",
    "seller_single_item_order_product_review_score_avg", "seller_single_item_order_product_bad_review_rate",
    "seller_product_attributed_review_score_avg", "seller_product_attributed_bad_review_rate",
    "seller_avg_early_delivery_days", "seller_avg_late_delivery_days",
    "seller_avg_delivery_estimate_abs_error_days", "seller_p90_order_to_delivery_days",
    "seller_missing_delivery_timestamp_rate",
}


@pytest.fixture
def seller_metric_identity():
    suffix = uuid4().hex
    tenant_id = f"seller_metrics_t_{suffix}"
    user_id = f"seller_metrics_u_{suffix}"
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
            VALUES (:tenant, 'seller metrics integration test', 'seller', 'active')
        """), {"tenant": tenant_id})
        connection.execute(text("""
            INSERT INTO iam.users(user_id, display_name, email, status)
            VALUES (:user, 'seller metrics integration test', :email, 'active')
        """), {"user": user_id, "email": f"{user_id}@example.invalid"})
        connection.execute(text("""
            INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
            VALUES (:tenant, :user, 'analyst')
        """), {"tenant": tenant_id, "user": user_id})
        connection.execute(text("""
            INSERT INTO iam.membership_seller_grants(
              tenant_id, user_id, seller_id, access_start_date, access_end_date
            ) VALUES (:tenant, :user, :seller, '2018-08-01', '2018-08-31')
        """), {"tenant": tenant_id, "user": user_id, "seller": SELLER_ID})
    context = TenantContext(
        tenant_id=tenant_id,
        user_id=user_id,
        role="analyst",
        scope_type="seller",
        seller_id=SELLER_ID,
        seller_grants=(SellerGrant(SELLER_ID, "2018-08-01", "2018-08-31"),),
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


SELLER_ID = "6560211a19b47992c3666cc44a7e94c0"


def test_expanded_metrics_are_registered_and_execute_against_olist(seller_metric_identity) -> None:
    with engine.connect() as connection:
        seller_id = connection.execute(text("""
            SELECT seller_id FROM analytics.v_delivered_order_items
            GROUP BY seller_id ORDER BY COUNT(*) DESC LIMIT 1
        """)).scalar_one()
    assert seller_id == SELLER_ID
    scope = ControlledAnalysisScope(
        seller_id=seller_id, default_start_date=date(2018, 8, 1), default_end_date=date(2018, 8, 31),
        metric_name="gmv", granularity="day",
    )
    registry = get_registry()
    assert P1_P2_P3_METRICS <= {metric.metric_id for metric in registry.list(scope="seller")}
    with activate_tenant_context(seller_metric_identity):
        for metric_id in P1_P2_P3_METRICS:
            value = registry.execute(metric_id, scope="seller", seller_scope=scope)
            trend = registry.execute(metric_id, "trend", scope="seller", seller_scope=scope,
                                     start_date=scope.default_start_date, end_date=scope.default_end_date)
            assert value["status"] == "ok"
            assert isinstance(value["value"], float)
            assert trend["points"]
