"""Isolated, authorized acceptance for all registered seller value metrics."""

from __future__ import annotations

import json
import os
import re
from datetime import date
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url

from app.data.repositories import explorer
from app.metrics.registry import get_registry
from app.schemas.explorer import ControlledAnalysisScope
from app.tenancy import SellerGrant, TenantContext, activate_tenant_context
from test_support.database import bind_test_engine


SELLER_ID = "seller-target"
OTHER_SELLER_ID = "seller-other"
START_DATE = date(2024, 1, 1)
END_DATE = date(2024, 1, 2)
QUERY_DATE = date(2024, 1, 2)
BREAKDOWN_DATE = date(2024, 1, 3)
CORE_SALES_METRICS = {
    "gmv", "avg_order_value", "customer_count", "order_count", "freight_value",
}
CORE_QUERY_SOURCE = "analytics.v_delivered_order_items + analytics.v_delivered_orders"
CORE_QUERY_LINEAGE = "analytics.fact_order_items + analytics.fact_orders"

# Hand-calculated from the isolated facts below; units are independently
# specified so registry metadata drift is visible instead of self-validating.
EXPECTED = {
    "customer_count": (3.0, "customers"),
    "repeat_purchase_rate": (33.33, "%"),
    "on_time_delivery_rate": (66.67, "%"),
    "avg_approval_lead_hours": (3.0, "hours"),
    "avg_dispatch_lead_hours": (6.0, "hours"),
    "avg_delivery_transit_hours": (24.0, "hours"),
    "order_approval_rate": (75.0, "%"),
    "order_shipment_rate": (50.0, "%"),
    "order_final_delivery_rate": (100.0, "%"),
    "payment_value": (490.0, "BRL"),
    "installment_order_share": (82.65, "%"),
    "review_score_avg": (3.0, "score"),
    "bad_review_rate": (50.0, "%"),
    "review_comment_rate": (50.0, "%"),
    "gmv": (540.0, "BRL"),
    "order_count": (4.0, "orders"),
    "avg_order_value": (135.0, "BRL/order"),
    "freight_value": (40.0, "BRL"),
    "seller_free_shipping_order_rate": (25.0, "%"),
    "seller_avg_early_delivery_days": (2.0, "days"),
    "seller_avg_late_delivery_days": (1.0, "days"),
    "seller_avg_delivery_estimate_abs_error_days": (1.0, "days"),
    "seller_p90_order_to_delivery_days": (7.4, "days"),
    "seller_missing_delivery_timestamp_rate": (25.0, "%"),
    "seller_single_item_order_product_review_score_avg": (10 / 3, "score"),
    "seller_single_item_order_product_bad_review_rate": (100 / 3, "%"),
    "seller_product_attributed_review_score_avg": (2.8, "score"),
    "seller_product_attributed_bad_review_rate": (60.0, "%"),
    "seller_historical_returning_customer_count": (1.0, "customers"),
    "seller_historical_returning_customer_rate": (33.33, "%"),
    "seller_new_customer_count": (2.0, "customers"),
    "seller_new_customer_rate": (66.67, "%"),
    "seller_top1_product_gross_share": (60.0, "%"),
    "seller_top3_category_gross_share": (90.0, "%"),
    "seller_new_product_gross_share": (40.0, "%"),
    "seller_item_gross_value": (500.0, "BRL"),
    "seller_avg_item_price": (100.0, "BRL/item"),
    "seller_avg_freight_per_order": (10.0, "BRL/order"),
    "seller_avg_payment_installments": (1115 / 490, "installments"),
    "seller_credit_card_payment_share": (40500 / 490, "%"),
    "seller_boleto_payment_share": (8500 / 490, "%"),
    "seller_active_product_count": (4.0, "products"),
    "seller_active_category_count": (4.0, "categories"),
    "seller_order_bad_review_rate": (50.0, "%"),
    "seller_items_per_order": (1.25, "items/order"),
    "seller_freight_to_gmv_rate": (4000 / 540, "%"),
}


@pytest.fixture(scope="module")
def isolated_seller_db():
    admin_url_text = os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL")
    if not admin_url_text:
        pytest.skip("Set ISOLATED_TEST_POSTGRES_ADMIN_URL to an isolated-test PostgreSQL maintenance URL")
    admin_url = make_url(admin_url_text)
    database_name = f"seller_value_acceptance_{uuid4().hex}"
    admin_engine = create_engine(
        admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT",
        pool_pre_ping=True, connect_args={"connect_timeout": 5},
    )
    isolated_engine = None
    try:
        with admin_engine.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{database_name}"'))
        isolated_engine = create_engine(
            admin_url.set(database=database_name), pool_pre_ping=True,
            connect_args={"connect_timeout": 5},
        )
        with isolated_engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA iam"))
            conn.execute(text("CREATE SCHEMA analytics"))
            conn.execute(text("CREATE SCHEMA raw"))
            conn.execute(text("""
                CREATE TABLE iam.tenants (
                    tenant_id text PRIMARY KEY, scope_type text NOT NULL, status text NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE iam.users (user_id text PRIMARY KEY, status text NOT NULL)
            """))
            conn.execute(text("""
                CREATE TABLE iam.tenant_memberships (
                    tenant_id text NOT NULL, user_id text NOT NULL, role text NOT NULL,
                    revoked_at timestamptz
                )
            """))
            conn.execute(text("""
                CREATE TABLE iam.membership_seller_grants (
                    tenant_id text NOT NULL, user_id text NOT NULL, seller_id text NOT NULL,
                    access_start_date date, access_end_date date, revoked_at timestamptz
                )
            """))
            conn.execute(text("""
                CREATE TABLE analytics.v_delivered_order_items (
                    seller_id text, order_id text NOT NULL, product_id text NOT NULL,
                    purchase_date date NOT NULL, price numeric NOT NULL, freight_value numeric NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE analytics.v_delivered_orders (
                    order_id text PRIMARY KEY, customer_unique_id text, purchase_date date NOT NULL,
                    item_count integer, order_status text NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE analytics.fact_orders (order_id text PRIMARY KEY, purchase_date date NOT NULL)
            """))
            conn.execute(text("""
                CREATE TABLE analytics.fact_order_payments (
                    seller_id text NOT NULL, order_id text NOT NULL, payment_installments integer,
                    allocated_payment_value numeric NOT NULL, payment_type text
                )
            """))
            conn.execute(text("""
                CREATE TABLE analytics.v_product_category (
                    product_id text PRIMARY KEY, product_category_name_english text
                )
            """))
            conn.execute(text("""
                CREATE TABLE analytics.v_seller_fulfillment (
                    seller_id text NOT NULL, order_id text NOT NULL, purchase_date date NOT NULL,
                    order_status text NOT NULL, purchase_at timestamp,
                    delivered_customer_at timestamp, estimated_delivery_at timestamp,
                    approval_lead_hours numeric, dispatch_lead_hours numeric,
                    delivery_transit_hours numeric, is_on_time boolean
                )
            """))
            conn.execute(text("""
                CREATE TABLE raw.order_items (
                    seller_id text NOT NULL, order_id text NOT NULL, order_item_id integer NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE raw.orders (
                    order_id text PRIMARY KEY, order_purchase_timestamp timestamp NOT NULL,
                    order_approved_at timestamp, order_delivered_carrier_date timestamp,
                    order_status text NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE raw.order_reviews (
                    order_id text NOT NULL, review_score integer, review_comment_message text
                )
            """))

            conn.execute(text("""
                INSERT INTO iam.tenants VALUES ('tenant-seller-value', 'seller', 'active')
            """))
            conn.execute(text("""
                INSERT INTO iam.users VALUES ('user-seller-value', 'active')
            """))
            conn.execute(text("""
                INSERT INTO iam.tenant_memberships VALUES ('tenant-seller-value', 'user-seller-value', 'analyst', NULL)
            """))
            conn.execute(text("""
                INSERT INTO iam.membership_seller_grants VALUES
                ('tenant-seller-value', 'user-seller-value', 'seller-target', '2024-01-01', '2024-01-02', NULL)
            """))
            conn.execute(text("""
                INSERT INTO iam.tenants VALUES ('tenant-platform-breakdown', 'platform', 'active')
            """))
            conn.execute(text("INSERT INTO iam.users VALUES ('user-platform-breakdown', 'active')"))
            conn.execute(text("""
                INSERT INTO iam.tenant_memberships VALUES
                ('tenant-platform-breakdown', 'user-platform-breakdown', 'admin', NULL)
            """))
            conn.execute(text("""
                INSERT INTO iam.membership_seller_grants VALUES
                ('tenant-platform-breakdown', 'user-platform-breakdown', 'A', '2024-01-01', '2024-01-31', NULL),
                ('tenant-platform-breakdown', 'user-platform-breakdown', 'B', '2024-01-01', '2024-01-31', NULL),
                ('tenant-platform-breakdown', 'user-platform-breakdown', 'C', '2024-01-01', '2024-01-31', NULL)
            """))

            # Historical target-seller purchase makes c1 a returning customer
            # and product p1 a non-new product in the measured day.
            conn.execute(text("""
                INSERT INTO analytics.v_delivered_orders VALUES
                    ('h1', 'c1', '2024-01-01', 1, 'delivered'),
                    ('t1', 'c1', '2024-01-02', 2, 'delivered'),
                    ('t2', 'c2', '2024-01-02', 1, 'delivered'),
                    ('t3', 'c3', '2024-01-02', 1, 'delivered'),
                    ('t4', 'c1', '2024-01-02', 1, 'delivered'),
                    ('other_shared', 'other-c1', '2024-01-02', 1, 'delivered'),
                    ('other_only', 'other-c2', '2024-01-02', 1, 'delivered'),
                    ('O1', 'fixture-c1', '2024-01-03', 2, 'delivered'),
                    ('O2', 'fixture-c2', '2024-01-03', 1, 'delivered'),
                    ('O3', 'fixture-c3', '2024-01-03', 1, 'delivered'),
                    ('O4', 'fixture-c4', '2024-01-03', 1, 'delivered'),
                    ('O6', 'fixture-c6', '2024-01-03', 2, 'delivered')
            """))
            conn.execute(text("""
                INSERT INTO analytics.v_delivered_order_items VALUES
                    ('seller-target', 'h1', 'p1', '2024-01-01', 20, 2),
                    ('seller-target', 't1', 'p1', '2024-01-02', 100, 10),
                    ('seller-target', 't1', 'p2', '2024-01-02', 50, 5),
                    ('seller-target', 't2', 'p1', '2024-01-02', 200, 20),
                    ('seller-target', 't3', 'p3', '2024-01-02', 50, 0),
                    ('seller-target', 't4', 'p4', '2024-01-02', 100, 5),
                    ('seller-other', 't2', 'other-p1', '2024-01-02', 1000, 100),
                    ('seller-other', 'other_only', 'other-p2', '2024-01-02', 5000, 500),
                    ('A', 'O1', 'fixture-a1', '2024-01-03', 100, 0),
                    ('B', 'O1', 'fixture-b1', '2024-01-03', 200, 0),
                    ('A', 'O2', 'fixture-a2', '2024-01-03', 150, 0),
                    ('C', 'O3', 'fixture-c3', '2024-01-03', 300, 0),
                    (NULL, 'O4', 'fixture-u4', '2024-01-03', 50, 0),
                    ('A', 'O6', 'fixture-a6', '2024-01-03', 60, 0),
                    (NULL, 'O6', 'fixture-u6', '2024-01-03', 40, 0)
            """))
            conn.execute(text("""
                INSERT INTO analytics.fact_orders VALUES
                    ('h1', '2024-01-01'), ('t1', '2024-01-02'), ('t2', '2024-01-02'),
                    ('t3', '2024-01-02'), ('t4', '2024-01-02'),
                    ('other_shared', '2024-01-02'), ('other_only', '2024-01-02')
            """))
            conn.execute(text("""
                INSERT INTO analytics.fact_order_payments VALUES
                    ('seller-target', 't1', 2, 80, 'credit_card'),
                    ('seller-target', 't1', 1, 35, 'boleto'),
                    ('seller-target', 't2', 3, 220, 'credit_card'),
                    ('seller-target', 't3', 1, 50, 'boleto'),
                    ('seller-target', 't4', 2, 105, 'credit_card'),
                    ('seller-target', 'h1', 1, 22, 'boleto'),
                    ('seller-other', 't2', 12, 300, 'credit_card'),
                    ('seller-other', 'other_only', 10, 5000, 'credit_card')
            """))
            conn.execute(text("""
                INSERT INTO analytics.v_product_category VALUES
                    ('p1', 'cat_a'), ('p2', 'cat_b'), ('p3', 'cat_c'), ('p4', 'cat_d'),
                    ('other-p1', 'other_cat'), ('other-p2', 'other_cat')
            """))
            conn.execute(text("""
                INSERT INTO analytics.v_seller_fulfillment VALUES
                    ('seller-target', 't1', '2024-01-02', 'delivered', '2024-01-02', '2024-01-07', '2024-01-09', 1, 4, 20, TRUE),
                    ('seller-target', 't2', '2024-01-02', 'delivered', '2024-01-02', '2024-01-10', '2024-01-09', 3, 6, 40, FALSE),
                    ('seller-target', 't3', '2024-01-02', 'delivered', '2024-01-02', '2024-01-06', '2024-01-06', 5, 8, 12, TRUE),
                    ('seller-target', 't4', '2024-01-02', 'delivered', '2024-01-02', NULL, '2024-01-10', NULL, NULL, NULL, NULL),
                    ('seller-other', 'other_only', '2024-01-02', 'delivered', '2024-01-02', '2024-02-01', '2024-01-10', 100, 100, 100, FALSE)
            """))
            conn.execute(text("""
                INSERT INTO raw.orders VALUES
                    ('t1', '2024-01-02 00:00:00', '2024-01-02 01:00:00', '2024-01-02 05:00:00', 'delivered'),
                    ('t2', '2024-01-02 00:00:00', '2024-01-02 03:00:00', '2024-01-02 06:00:00', 'delivered'),
                    ('t3', '2024-01-02 00:00:00', '2024-01-02 05:00:00', NULL, 'delivered'),
                    ('t4', '2024-01-02 00:00:00', NULL, NULL, 'delivered'),
                    ('other_shared', '2024-01-02 00:00:00', '2024-01-02 01:00:00', '2024-01-02 02:00:00', 'delivered'),
                    ('other_only', '2024-01-02 00:00:00', '2024-01-02 01:00:00', '2024-01-02 02:00:00', 'delivered')
            """))
            conn.execute(text("""
                INSERT INTO raw.order_items VALUES
                    ('seller-target', 't1', 1), ('seller-target', 't1', 2),
                    ('seller-target', 't2', 1), ('seller-target', 't3', 1), ('seller-target', 't4', 1),
                    ('seller-other', 't2', 1), ('seller-other', 'other_only', 1)
            """))
            conn.execute(text("""
                INSERT INTO raw.order_reviews VALUES
                    ('t1', 2, 'poor but useful'), ('t2', 5, '   '),
                    ('t3', 4, NULL), ('t4', 1, 'delivery issue')
            """))

        statements: list[tuple[str, object]] = []

        @event.listens_for(isolated_engine, "before_cursor_execute")
        def capture_sql(_conn, _cursor, statement, parameters, _context, _many):
            statements.append((statement, parameters))

        yield isolated_engine, statements
    finally:
        if isolated_engine is not None:
            isolated_engine.dispose()
        try:
            with admin_engine.connect() as conn:
                conn.execute(text(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)'))
        finally:
            admin_engine.dispose()


@pytest.fixture(scope="module")
def seller_identity(isolated_seller_db):
    return TenantContext(
        tenant_id="tenant-seller-value", user_id="user-seller-value", role="analyst",
        scope_type="seller", seller_id=SELLER_ID,
        seller_grants=(SellerGrant(SELLER_ID, "2024-01-01", "2024-01-02"),),
    )


def test_registry_seller_value_inventory_matches_46_hand_calculated_cases():
    registry = get_registry()
    registered = {
        metric.metric_id
        for metric in registry.list(scope="seller")
        if "value" in metric.implementations["seller"].capabilities
    }
    assert len(registered) == 46
    assert registered == set(EXPECTED)


@pytest.mark.parametrize("metric_id", sorted(CORE_SALES_METRICS))
def test_core_seller_source_metadata_separates_query_objects_and_lineage(metric_id):
    implementation = get_registry().get(metric_id).implementations["seller"]
    assert implementation.lineage_source == CORE_QUERY_LINEAGE
    assert "底层事实血缘" in implementation.lineage_description
    assert "analytics.v_delivered_order_items" in implementation.lineage_description
    assert "analytics.v_delivered_orders" in implementation.lineage_description


@pytest.mark.parametrize("metric_id", sorted(EXPECTED))
def test_seller_value_through_authorized_registry_query(
    monkeypatch, isolated_seller_db, seller_identity, metric_id
):
    isolated_engine, statements = isolated_seller_db
    bind_test_engine(monkeypatch, isolated_engine)
    registry = get_registry()
    definition = registry.get(metric_id)
    scope = ControlledAnalysisScope(
        seller_id=SELLER_ID,
        default_start_date=QUERY_DATE,
        default_end_date=QUERY_DATE,
        metric_name=metric_id,
        granularity="day",
        # This is a server-built locked scope covering the registry's seller
        # value capabilities, including the report/chat-only repeat rate.
        allowed_metrics=tuple(sorted(EXPECTED)),
    )
    start = len(statements)
    exception = None
    try:
        with activate_tenant_context(seller_identity):
            actual = registry.execute(
                metric_id,
                operation="value",
                scope="seller",
                seller_scope=scope,
                metric_date=QUERY_DATE,
            )
    except Exception as exc:  # Preserve current business failures for diagnosis.
        exception = f"{type(exc).__name__}: {exc}"
        actual = {
            "metric_name": metric_id,
            "value": None,
            "source_table": None,
            "status": "exception",
        }

    trace = statements[start:]
    authorization_sql = [
        statement.lower()
        for statement, _ in trace
        if " from iam." in statement.lower()
    ]
    query_trace = [
        (statement, params)
        for statement, params in trace
        if isinstance(params, dict) and params.get("seller_id") == SELLER_ID
        and ("metric_date" in params or "start_date" in params)
    ]
    actual_source_tokens = re.findall(
        r"\b(?:analytics|raw)\.[a-z_][a-z_0-9]*\b",
        str(actual.get("source_table") or "").lower(),
    )
    query_sql = " ".join(statement.lower() for statement, _ in query_trace)
    expected_value, expected_unit = EXPECTED[metric_id]
    expected_numeric = float(expected_value)
    checks = {
        "value": actual.get("value") is not None
        and float(actual["value"]) == pytest.approx(expected_numeric, abs=0.01),
        "date_filter": any(
            params.get("metric_date") == QUERY_DATE
            or (
                params.get("start_date") == QUERY_DATE
                and params.get("end_date") == QUERY_DATE
            )
            for _, params in query_trace
        ),
        "seller_filter": any(
            params.get("seller_id") == SELLER_ID for _, params in query_trace
        ),
        "source": bool(actual.get("source_table")) and bool(query_trace)
        and all(token in query_sql for token in actual_source_tokens),
        "tenant_seller_authorization": all(
            any(table in statement for statement in authorization_sql)
            for table in (
                "iam.tenants", "iam.users", "iam.tenant_memberships",
                "iam.membership_seller_grants",
            )
        ),
        "status": actual.get("status") == "ok",
        "registered_unit": definition.unit == expected_unit,
    }
    lineage = definition.implementations["seller"].lineage_source
    lineage_tokens = re.findall(
        r"\b(?:analytics|raw)\.[a-z_][a-z_0-9]*\b", lineage.lower()
    )
    if metric_id in CORE_SALES_METRICS:
        # source_table is the direct FROM/JOIN object; source_lineage follows
        # the delivered-view definitions to their underlying fact tables.
        source_semantics = {
            "direct_query_objects": actual.get("source_table") == CORE_QUERY_SOURCE
            and all(token in query_sql for token in re.findall(
                r"\b(?:analytics|raw)\.[a-z_][a-z_0-9]*\b",
                CORE_QUERY_SOURCE.lower(),
            )),
            "underlying_lineage": actual.get("source_lineage") == CORE_QUERY_LINEAGE
            and lineage == CORE_QUERY_LINEAGE,
        }
        lineage_mismatch = []
    else:
        source_semantics = {}
        lineage_mismatch = [token for token in lineage_tokens if token not in query_sql]
    checks["source_semantics"] = all(source_semantics.values()) if source_semantics else True
    report = {
        "metric_id": metric_id,
        "expected": {"value": expected_numeric, "unit": expected_unit, "date": QUERY_DATE.isoformat()},
        "actual": actual,
        "registered_unit": definition.unit,
        "registered_lineage_source": lineage,
        "source_semantics": source_semantics,
        "lineage_tokens_missing_from_query": lineage_mismatch,
        "checks": checks,
        "exception": exception,
        "sql": [statement for statement, _ in query_trace],
        "sql_parameters": [params for _, params in query_trace],
        "authorization_queries": authorization_sql,
    }
    print("SELLER_VALUE_RESULT " + json.dumps(report, ensure_ascii=False, sort_keys=True, default=str))
    failures = [name for name, passed in checks.items() if not passed]
    assert not failures, f"{metric_id}: failed checks={failures}; report={report!r}"


def test_authorized_seller_breakdown_postgres_totals_and_unattributed(isolated_seller_db, monkeypatch):
    from app.auth.context import build_authorization_context
    from app.core.config import Settings
    from app.scope.seller_scope import SellerScopeIntent, resolve_seller_scope

    isolated_engine, _statements = isolated_seller_db
    bind_test_engine(monkeypatch, isolated_engine)
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(seller_dimension_enabled=True, seller_multi_scope_enabled=True),
    )
    admin = TenantContext(
        tenant_id="tenant-platform-breakdown", user_id="user-platform-breakdown",
        role="admin", scope_type="platform",
        seller_grants=(
            SellerGrant("A", "2024-01-01", "2024-01-31"),
            SellerGrant("B", "2024-01-01", "2024-01-31"),
            SellerGrant("C", "2024-01-01", "2024-01-31"),
        ),
    )
    authz = build_authorization_context(admin, metric_scope_type="platform")

    def run(scope_type, seller_ids=(), *, include_unattributed=False, metric="gmv", limit=10):
        intent = SellerScopeIntent(
            seller_scope_type=scope_type, seller_ids=tuple(seller_ids),
            start_date=BREAKDOWN_DATE, end_date=BREAKDOWN_DATE,
            metric_ids=("gmv", "order_count"), dimensions=("seller",),
            include_unattributed=include_unattributed,
        )
        scope = resolve_seller_scope(intent, tenant=admin, authz=authz)
        with activate_tenant_context(admin):
            return explorer.breakdown_by_seller(scope, metric, limit=limit)

    selected = run("selected", ("A", "B"), metric="order_count")
    assert {item["seller_id"]: item["value"] for item in selected["items"]} == {"A": 3, "B": 1}
    assert selected["total"]["gmv"] == 510
    assert selected["total"]["order_count"] == 3
    assert selected["non_additive"] is True
    assert selected["unattributed"] == {
        "gmv": 90.0, "order_count": 2,
        "reason": "seller_id_is_null; order_count may overlap seller-attributed orders and is not additive",
        "scope": "date_range_all_unattributed",
    }

    all_authorized = run("all", metric="gmv")
    assert {item["seller_id"]: item["value"] for item in all_authorized["items"]} == {
        "A": 310.0, "B": 200.0, "C": 300.0,
    }
    assert all_authorized["total"]["gmv"] == 810
    assert all_authorized["total"]["order_count"] == 4
    assert all_authorized["total"]["total_includes_unattributed"] is False

    all_including_unknown = run("all", include_unattributed=True, metric="order_count")
    assert all_including_unknown["total"]["gmv"] == 900
    assert all_including_unknown["total"]["order_count"] == 5
    assert all_including_unknown["total"]["total_includes_unattributed"] is True

    selected_a = run("selected", ("A",))
    assert {item["seller_id"] for item in selected_a["items"]} == {"A"}
    assert selected_a["total"]["gmv"] == 310
    # The total is calculated independently from the top-N item list.
    truncated = run("selected", ("A", "B"), limit=1)
    assert len(truncated["items"]) == 1
    assert truncated["total"]["gmv"] == selected["total"]["gmv"]
    single = run("single", ("A",))
    assert len(single["items"]) == 1
    assert "只有一个 seller 分组" in single["message"]
    assert single["total"]["gmv"] == 310

    from app.scope.canonical import ScopeError
    with pytest.raises(ScopeError, match="no explicit seller grant"):
        run("selected", ("ungranted",))
