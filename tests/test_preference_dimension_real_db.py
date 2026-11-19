"""Opt-in PostgreSQL proof that a validated dimension preference changes results."""

from __future__ import annotations

import os
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


@pytest.fixture
def isolated_preference_db(monkeypatch):
    admin_url_text = os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL")
    if not admin_url_text:
        pytest.skip("Set ISOLATED_TEST_POSTGRES_ADMIN_URL for the real PostgreSQL preference experiment")

    from app.data.repositories import daily_metrics

    admin_url = make_url(admin_url_text)
    database_name = f"pref_dim_{uuid4().hex}"
    admin_engine = create_engine(
        admin_url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        connect_args={"connect_timeout": 5},
    )
    database_engine = None
    try:
        with admin_engine.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{database_name}"'))
        database_engine = create_engine(
            admin_url.set(database=database_name),
            connect_args={"connect_timeout": 5},
        )
        with database_engine.begin() as connection:
            connection.execute(text("CREATE SCHEMA analytics"))
            connection.execute(text("CREATE SCHEMA raw"))
            connection.execute(text("""
                CREATE TABLE analytics.daily_metrics (
                    metric_date date PRIMARY KEY,
                    gmv numeric(12, 2) NOT NULL,
                    order_count integer NOT NULL,
                    avg_order_value numeric(12, 2) NOT NULL
                )
            """))
            connection.execute(text("""
                CREATE TABLE analytics.fact_orders (
                    order_id text PRIMARY KEY,
                    customer_unique_id text,
                    purchase_date date NOT NULL
                )
            """))
            connection.execute(text("""
                CREATE TABLE raw.orders (
                    order_id text PRIMARY KEY,
                    customer_id text,
                    order_purchase_timestamp timestamp NOT NULL,
                    order_delivered_customer_date timestamp,
                    order_estimated_delivery_date timestamp,
                    order_status text NOT NULL
                )
            """))
            connection.execute(text("""
                CREATE TABLE raw.order_items (
                    order_id text NOT NULL,
                    product_id text NOT NULL,
                    seller_id text,
                    price numeric(12, 2) NOT NULL,
                    freight_value numeric(12, 2) NOT NULL
                )
            """))
            connection.execute(text("CREATE TABLE raw.customers (customer_id text PRIMARY KEY, customer_unique_id text)"))
            connection.execute(text("CREATE TABLE raw.sellers (seller_id text PRIMARY KEY, seller_state text)"))
            connection.execute(text("CREATE TABLE raw.products (product_id text PRIMARY KEY, product_category_name text)"))
            connection.execute(text("CREATE TABLE raw.order_reviews (order_id text, review_score integer)"))

            connection.execute(text("""
                INSERT INTO analytics.daily_metrics VALUES ('2024-01-10', 385, 3, 128.33)
            """))
            connection.execute(text("""
                INSERT INTO raw.customers VALUES
                    ('c1', 'customer-1'), ('c2', 'customer-2'), ('c3', 'customer-3')
            """))
            connection.execute(text("""
                INSERT INTO raw.sellers VALUES ('s1', 'SP'), ('s2', 'RJ'), ('s3', 'MG')
            """))
            connection.execute(text("""
                INSERT INTO raw.products VALUES
                    ('p1', 'books'), ('p2', 'books'), ('p3', 'toys')
            """))
            connection.execute(text("""
                INSERT INTO raw.orders VALUES
                    ('o1', 'c1', '2024-01-10 09:00:00', '2024-01-11 09:00:00', '2024-01-12 09:00:00', 'delivered'),
                    ('o2', 'c2', '2024-01-10 10:00:00', '2024-01-11 10:00:00', '2024-01-12 10:00:00', 'delivered'),
                    ('o3', 'c3', '2024-01-10 11:00:00', '2024-01-12 11:00:00', '2024-01-12 12:00:00', 'delivered')
            """))
            connection.execute(text("""
                INSERT INTO raw.order_items VALUES
                    ('o1', 'p1', 's1', 100, 10),
                    ('o2', 'p2', 's2', 50, 5),
                    ('o3', 'p3', 's3', 200, 20)
            """))
            connection.execute(text("""
                INSERT INTO analytics.fact_orders VALUES
                    ('o1', 'customer-1', '2024-01-10'),
                    ('o2', 'customer-2', '2024-01-10'),
                    ('o3', 'customer-3', '2024-01-10')
            """))

        monkeypatch.setattr(daily_metrics, "engine", database_engine)
        yield database_engine
    finally:
        if database_engine is not None:
            database_engine.dispose()
        with admin_engine.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)'))
        admin_engine.dispose()


def test_validated_preference_changes_real_query_result_to_category_breakdown(
    isolated_preference_db, monkeypatch,
) -> None:
    from app.agents.data_query_agent import plan_confirmed_metric_queries
    from app.context.compiler import compile_turn_context
    from app.mcp.client import query_business_data
    from app.tenancy.context import TenantContext, activate_tenant_context

    class PreferenceProvider:
        def search(self, *, need, state):
            return [{
                "memory_key": "analysis_style",
                "memory_type": "analysis_method",
                "value": {"dimension": "product_category"},
                "confidence": 0.93,
                "source_ref": "isolated-db-preference-fixture",
            }]

    # Force the planner's deterministic exact-alias path; no external LLM is used.
    monkeypatch.setattr("app.llm.client.llm_extract", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.mcp.client.resolve_expression", lambda *_args, **_kwargs: None)
    fake_authz = SimpleNamespace(
        authz_digest="isolated-test-authz",
        permission_version="permission-v1",
        policy_set_version="policy-v1",
        authz_ref="isolated-test-authz-ref",
        require_action=lambda *_args, **_kwargs: None,
        require_metrics=lambda *_args, **_kwargs: None,
        require_dimensions=lambda *_args, **_kwargs: None,
        require_tool=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "app.auth.context.build_authorization_context",
        lambda *_args, **_kwargs: fake_authz,
    )

    scope_data = {
        "tenant_id": "tenant-test",
        "user_id": "user-test",
        "permissions": "admin",
        "scope_type": "platform",
    }
    context = TenantContext(
        tenant_id="tenant-test", user_id="user-test", role="admin", scope_type="platform",
    )

    def run_query(*, with_preference: bool):
        question = "按我平时的分析方式查询 2024-01-10 的 GMV"
        state = compile_turn_context({
            "user_message": question,
            "current_query_scope_fingerprint": "isolated-scope",
            "scope_type": "platform",
            **({"preference_provider": PreferenceProvider()} if with_preference else {}),
            "node_history": [],
        })
        compilation = state["context_compilation"]
        batch = plan_confirmed_metric_queries(
            question,
            scope="platform",
            allowed_metrics={"gmv"},
            scope_data=scope_data,
            operation_intent=compilation["operation_intent"],
            request_contract=compilation["request_contract"],
        )
        assert batch.decision.status == "confirmed", batch.decision
        with activate_tenant_context(context):
            response = query_business_data(
                "metric_query",
                question,
                tenant_id="tenant-test",
                role="admin",
                plan=batch.plans[0],
            )
        assert response["result"]["status"] == "ok", response["result"]
        return batch.plans[0], response["result"]

    scalar_plan, scalar_result = run_query(with_preference=False)
    breakdown_plan, breakdown_result = run_query(with_preference=True)

    assert scalar_plan.query_type == "metric_value"
    assert scalar_result["metric_result"]["value"] == 385.0
    assert breakdown_plan.query_type == "metric_breakdown"
    assert breakdown_plan.dimension == "product_category"
    assert breakdown_result["breakdown_result"]["dimension"] == "product_category"
    assert {
        item["dimension_value"]: item["value"]
        for item in breakdown_result["breakdown_result"]["items"]
    } == {"books": 165.0, "toys": 220.0}
    assert sum(item["value"] for item in breakdown_result["breakdown_result"]["items"]) == scalar_result["metric_result"]["value"]
