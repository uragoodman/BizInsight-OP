"""Isolated acceptance checks for all registered platform period metrics.

The test uses a disposable PostgreSQL database created only when the caller
explicitly supplies ISOLATED_TEST_POSTGRES_ADMIN_URL (a maintenance DB URL).
It never connects to the configured business database.
"""

from __future__ import annotations

import os
import json
import re
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url

from app.data.repositories import daily_metrics
from app.metrics.registry import get_registry


PERIOD_START = date(2024, 1, 1)
PERIOD_END = date(2024, 1, 2)
PERIOD_EXPECTED = {
    # Existing five cases, retained once and now routed through MetricRegistry.
    "payment_value": (1000.00, "analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)"),
    "installment_order_share": (18.00, "analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)"),
    "avg_approval_lead_hours": (7.50, "analytics.fact_fulfillment"),
    "avg_dispatch_lead_hours": (9.00, "analytics.fact_fulfillment"),
    "avg_delivery_transit_hours": (18.00, "analytics.fact_fulfillment"),
    # New cases: values are hand-calculated from the fixture below.
    "avg_delivery_lead_time": (0.88, "raw.orders"),
    "avg_items_per_order": (1.50, "analytics.fact_orders"),
    "avg_order_value": (255.00, "analytics.fact_orders"),
    "bad_review_rate": (50.00, "raw.orders+raw.order_reviews"),
    "cancel_rate": (20.00, "raw.orders"),
    "customer_count": (3.00, "analytics.fact_orders"),
    "delivery_delay_rate": (50.00, "raw.orders"),
    "freight_rate": (1.96, "analytics.fact_orders"),
    "freight_value": (20.00, "analytics.fact_orders"),
    "gmv": (1020.00, "analytics.fact_orders"),
    "on_time_delivery_rate": (50.00, "raw.orders"),
    "order_count": (4.00, "analytics.fact_orders"),
    "repeat_customer_count": (1.00, "analytics.fact_orders"),
    "repeat_purchase_rate": (33.33, "analytics.fact_orders"),
    "review_score_avg": (3.00, "raw.orders+raw.order_reviews"),
}


@pytest.fixture(scope="module")
def isolated_metric_db():
    admin_url_text = os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL")
    if not admin_url_text:
        pytest.skip("Set ISOLATED_TEST_POSTGRES_ADMIN_URL to an isolated-test PostgreSQL maintenance URL")

    admin_url = make_url(admin_url_text)
    database_name = f"metric_period_acceptance_{uuid4().hex}"
    admin_engine = create_engine(
        admin_url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )
    isolated_engine = None
    try:
        with admin_engine.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{database_name}"'))

        isolated_engine = create_engine(
            admin_url.set(database=database_name),
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5},
        )
        with isolated_engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA analytics"))
            conn.execute(text("CREATE SCHEMA raw"))
            conn.execute(text("""
                CREATE TABLE analytics.fact_orders (
                    order_id text PRIMARY KEY,
                    purchase_date date NOT NULL,
                    customer_unique_id text,
                    item_gross_value numeric(12, 2) NOT NULL,
                    freight_value numeric(12, 2) NOT NULL,
                    item_count integer NOT NULL,
                    order_status text NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE analytics.fact_order_payments (
                    order_id text NOT NULL,
                    payment_value numeric(12, 2) NOT NULL,
                    allocated_payment_value numeric(12, 2) NOT NULL,
                    payment_installments integer NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE analytics.fact_fulfillment (
                    order_id text PRIMARY KEY,
                    purchase_date date NOT NULL,
                    approval_lead_hours numeric(12, 2),
                    dispatch_lead_hours numeric(12, 2),
                    delivery_transit_hours numeric(12, 2)
                )
            """))
            # These tables let the current fallback branch execute so the test
            # can expose its result instead of failing on a missing relation.
            conn.execute(text("""
                CREATE TABLE raw.orders (
                    order_id text PRIMARY KEY,
                    order_purchase_timestamp timestamp NOT NULL,
                    order_status text NOT NULL,
                    order_delivered_customer_date timestamp,
                    order_estimated_delivery_date timestamp
                )
            """))
            conn.execute(text("""
                CREATE TABLE raw.order_reviews (
                    order_id text NOT NULL,
                    review_score integer NOT NULL
                )
            """))

            conn.execute(text("""
                INSERT INTO analytics.fact_orders(
                    order_id, purchase_date, customer_unique_id, item_gross_value,
                    freight_value, item_count, order_status
                ) VALUES
                    ('p1', '2024-01-01', 'c1', 100, 10, 1, 'delivered'),
                    ('p2', '2024-01-02', 'c1', 30, 5, 3, 'delivered'),
                    ('p3', '2024-01-02', 'c2', 20, 4, 1, 'delivered'),
                    ('p4', '2024-01-02', 'c3', 850, 1, 1, 'delivered'),
                    ('xcancel', '2024-01-02', NULL, 0, 0, 0, 'canceled'),
                    ('p_before', '2023-12-31', 'c1', 9999, 999, 1, 'delivered'),
                    ('p_after', '2024-01-03', 'c9', 9999, 999, 1, 'delivered'),
                    ('p_zero', '2024-02-01', 'c10', 0, 0, 1, 'delivered')
            """))
            conn.execute(text("""
                INSERT INTO analytics.fact_order_payments(order_id, payment_value, allocated_payment_value, payment_installments) VALUES
                    ('p1', 90, 90, 3), ('p2', 10, 10, 1),
                    ('p3', 90, 90, 3), ('p4', 810, 810, 1),
                    ('p_before', 7000, 7000, 8), ('p_after', 9000, 9000, 10),
                    ('p_zero', 0, 0, 3)
            """))
            # Four in-range observations across two dates make AVG(AVG(day))
            # differ from AVG(all facts); outside-range rows test inclusive bounds.
            conn.execute(text("""
                INSERT INTO analytics.fact_fulfillment(
                    order_id, purchase_date, approval_lead_hours,
                    dispatch_lead_hours, delivery_transit_hours
                ) VALUES
                    ('f1', '2024-01-01', 0, 0, 0),
                    ('f2', '2024-01-02', 10, 12, 24),
                    ('f3', '2024-01-02', 10, 12, 24),
                    ('f4', '2024-01-02', 10, 12, 24),
                    ('f_before', '2023-12-31', 1000, 1000, 1000),
                    ('f_after', '2024-01-03', 2000, 2000, 2000)
            """))
            conn.execute(text("""
                INSERT INTO raw.orders(
                    order_id, order_purchase_timestamp, order_status,
                    order_delivered_customer_date, order_estimated_delivery_date
                ) VALUES
                    ('p1', '2024-01-01 00:00:00', 'delivered', '2024-01-02 00:00:00', '2024-01-02 01:00:00'),
                    ('p2', '2024-01-02 00:00:00', 'delivered', '2024-01-02 12:00:00', '2024-01-02 12:00:00'),
                    ('p3', '2024-01-02 12:00:00', 'delivered', '2024-01-03 12:00:00', '2024-01-03 10:00:00'),
                    ('p4', '2024-01-02 23:59:59', 'delivered', '2024-01-04 00:00:00', '2024-01-03 00:00:00'),
                    ('xcancel', '2024-01-02 20:00:00', 'canceled', NULL, NULL),
                    ('raw_before', '2023-12-31 23:59:59', 'delivered', '2024-01-01 23:59:59', '2024-01-02 23:59:59'),
                    ('raw_after', '2024-01-03 00:00:00', 'delivered', '2024-01-04 00:00:00', '2024-01-05 00:00:00')
            """))
            conn.execute(text("""
                INSERT INTO raw.order_reviews(order_id, review_score) VALUES
                    ('p1', 1), ('p2', 5), ('p3', 4), ('p4', 2)
            """))

        statements: list[str] = []

        @event.listens_for(isolated_engine, "before_cursor_execute")
        def capture_sql(_conn, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)

        cases: list[dict] = []
        yield isolated_engine, statements, cases
        from _artifact_safety import isolated_output_path
        result_path = isolated_output_path("PLATFORM_PERIOD_RESULT_PATH")
        if result_path:
            target = Path(result_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({
                "period_metric_count": len(PERIOD_EXPECTED),
                "cases": cases,
            }, indent=2), encoding="utf-8")
    finally:
        if isolated_engine is not None:
            isolated_engine.dispose()
        try:
            with admin_engine.connect() as conn:
                conn.execute(text(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)'))
        finally:
            admin_engine.dispose()


def test_platform_period_registry_inventory_exactly_20():
    registry = get_registry()
    registry.load()
    actual = {
        metric_id
        for metric_id, definition in registry._metrics.items()
        if "platform" in definition.implementations
        and "period" in definition.implementations["platform"].capabilities
    }
    assert actual == set(PERIOD_EXPECTED)
    assert len(actual) == 20


@pytest.mark.parametrize(
    ("metric_name", "expected", "expected_source"),
    [(metric_id, *PERIOD_EXPECTED[metric_id]) for metric_id in sorted(PERIOD_EXPECTED)],
)
def test_platform_metric_period_uses_real_source_bounds_and_aggregation(
    monkeypatch, isolated_metric_db, metric_name, expected, expected_source
):
    isolated_engine, statements, cases = isolated_metric_db
    monkeypatch.setattr(daily_metrics, "engine", isolated_engine)
    statement_start = len(statements)
    registry = get_registry()
    registry.load()

    actual = registry.execute(
        metric_name,
        operation="period",
        scope="platform",
        start_date=PERIOD_START,
        end_date=PERIOD_END,
    )
    executed_statements = statements[statement_start:]
    relation_pattern = re.compile(r"\b(?:analytics|raw)\.[a-z_][a-z_0-9]*\b", re.I)
    source_relations = sorted(set(relation_pattern.findall(actual.get("source_table", ""))))
    expected_source_relations = sorted(set(relation_pattern.findall(expected_source)))
    sql_relations = sorted({relation for sql in executed_statements for relation in relation_pattern.findall(sql)})
    registered_lineage = registry.get_implementation(metric_name, "platform").lineage_source
    lineage_text = registered_lineage if isinstance(registered_lineage, str) else " ".join(registered_lineage)
    lineage_relations = sorted(set(relation_pattern.findall(lineage_text)))
    checks = {
        "numeric_value": actual.get("value") == expected,
        "status": actual.get("status") == "ok",
        "start_boundary": actual.get("start_date") == PERIOD_START.isoformat(),
        "end_boundary": actual.get("end_date") == PERIOD_END.isoformat(),
        "one_actual_query": len(executed_statements) == 1,
        "expected_source_matches_sql": set(expected_source_relations) == set(sql_relations),
        "reported_source_matches_sql": set(source_relations) == set(sql_relations),
        # ``source_table`` is the direct query object, while Registry
        # ``lineage_source`` may name an underlying fact/view lineage.  Those
        # are different contracts: direct-source correctness is established by
        # the two checks above against the executed SQL.  Do not call a valid
        # view-backed query a lineage failure solely because its SQL relation
        # differs from the registered lineage label.
        "registered_lineage_declared": bool(lineage_relations),
    }
    cases.append({
        "metric": metric_name,
        "expected": expected,
        "expected_source": expected_source,
        "actual": actual,
        "sql": executed_statements,
        "checks": checks,
        "reported_source_relations": source_relations,
        "expected_source_relations": expected_source_relations,
        "executed_sql_relations": sql_relations,
        "registered_lineage": list(registered_lineage),
        "registered_lineage_relations": lineage_relations,
        "numeric_status": "pass" if checks["numeric_value"] and checks["status"] else "fail",
        "returned_source_status": "pass" if checks["reported_source_matches_sql"] else "fail",
        "registry_lineage_status": "recorded" if checks["registered_lineage_declared"] else "missing",
    })
    failures = [name for name, passed in checks.items() if not passed]
    assert not failures, (
        f"{metric_name}: failed checks={failures}; expected={expected}; actual={actual!r}; "
        f"SQL={executed_statements!r}; registry_lineage={registered_lineage!r}"
    )


@pytest.mark.parametrize(
    ("metric_name", "expected_source"),
    [
        ("payment_value", "analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)"),
        ("installment_order_share", "analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)"),
        ("avg_approval_lead_hours", "analytics.fact_fulfillment"),
        ("avg_dispatch_lead_hours", "analytics.fact_fulfillment"),
        ("avg_delivery_transit_hours", "analytics.fact_fulfillment"),
    ],
)
def test_platform_period_with_no_facts_is_not_found(
    monkeypatch, isolated_metric_db, metric_name, expected_source
):
    isolated_engine, _statements, _cases = isolated_metric_db
    monkeypatch.setattr(daily_metrics, "engine", isolated_engine)

    actual = daily_metrics.query_metric_period(metric_name, "2025-01-01", "2025-01-02")

    assert actual == {
        "metric_name": metric_name,
        "start_date": "2025-01-01",
        "end_date": "2025-01-02",
        "value": None,
        "source_table": expected_source,
        "status": "not_found",
    }


def test_installment_share_zero_period_denominator_returns_no_ratio(
    monkeypatch, isolated_metric_db
):
    isolated_engine, _statements, _cases = isolated_metric_db
    monkeypatch.setattr(daily_metrics, "engine", isolated_engine)

    actual = daily_metrics.query_metric_period(
        "installment_order_share", "2024-02-01", "2024-02-01"
    )

    assert actual == {
        "metric_name": "installment_order_share",
        "start_date": "2024-02-01",
        "end_date": "2024-02-01",
        "value": None,
        "source_table": "analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)",
        "status": "not_found",
    }


def test_registered_metric_without_period_implementation_is_not_reported_as_not_found():
    # order_approval_rate is registered for value/trend, but not period.
    with pytest.raises(NotImplementedError, match="Period query is not implemented"):
        daily_metrics.query_metric_period("order_approval_rate", "2024-01-01", "2024-01-02")
