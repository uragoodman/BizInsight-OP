"""Isolated acceptance checks for registered platform point comparisons."""

from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url

from app.data.repositories import daily_metrics
from app.metrics.registry import get_registry


DATE_LEFT = date(2024, 1, 1)
DATE_RIGHT = date(2024, 1, 2)
DATE_ZERO = date(2024, 1, 3)
DATE_MISSING = date(2024, 1, 4)
DATE_ZERO_INSTALLMENT_AND_REPEAT = date(2024, 1, 5)
DATE_ZERO_DAILY_AGGREGATE = date(2024, 1, 6)

# These values are independently calculated from the fixture rows below.
NORMAL_VALUES = {
    "avg_approval_lead_hours": (3.00, 8.00),
    "avg_delivery_lead_time": (2.00, 4.00),
    "avg_delivery_transit_hours": (10.00, 22.00),
    "avg_dispatch_lead_hours": (6.00, 12.00),
    "avg_items_per_order": (2.00, 6.00),
    "avg_order_value": (100.00, 200.00),
    "bad_review_rate": (50.00, 66.67),
    "cancel_rate": (33.33, 25.00),
    "customer_count": (2.00, 3.00),
    "delivery_delay_rate": (50.00, 66.67),
    "freight_rate": (10.00, 5.00),
    "freight_value": (20.00, 30.00),
    "gmv": (200.00, 600.00),
    "installment_order_share": (50.00, 66.67),
    "on_time_delivery_rate": (50.00, 33.33),
    "order_count": (2.00, 3.00),
    "payment_value": (200.00, 600.00),
    "repeat_customer_count": (1.00, 1.00),
    "repeat_purchase_rate": (50.00, 33.33),
    "review_score_avg": (3.00, 1.67),
}

ZERO_BASE_DATE = {
    "avg_approval_lead_hours": DATE_ZERO,
    "avg_delivery_lead_time": DATE_ZERO,
    "avg_delivery_transit_hours": DATE_ZERO,
    "avg_dispatch_lead_hours": DATE_ZERO,
    "avg_order_value": DATE_ZERO_DAILY_AGGREGATE,
    "bad_review_rate": DATE_ZERO,
    "cancel_rate": DATE_ZERO,
    "customer_count": DATE_ZERO,
    "delivery_delay_rate": DATE_ZERO,
    "freight_rate": DATE_ZERO_INSTALLMENT_AND_REPEAT,
    "freight_value": DATE_ZERO,
    "gmv": DATE_ZERO_DAILY_AGGREGATE,
    "installment_order_share": DATE_ZERO_INSTALLMENT_AND_REPEAT,
    "on_time_delivery_rate": DATE_ZERO_INSTALLMENT_AND_REPEAT,
    "order_count": DATE_ZERO_DAILY_AGGREGATE,
    "payment_value": DATE_ZERO,
    "repeat_customer_count": DATE_ZERO,
    "repeat_purchase_rate": DATE_ZERO_INSTALLMENT_AND_REPEAT,
}

DIRECT_SOURCE = {
    "avg_approval_lead_hours": "analytics.fact_fulfillment",
    "avg_delivery_lead_time": "raw.orders",
    "avg_delivery_transit_hours": "analytics.fact_fulfillment",
    "avg_dispatch_lead_hours": "analytics.fact_fulfillment",
    "avg_items_per_order": "analytics.v_delivered_orders",
    "avg_order_value": "analytics.daily_metrics",
    "bad_review_rate": "raw.order_reviews",
    "cancel_rate": "raw.orders",
    "customer_count": "analytics.v_delivered_orders",
    "delivery_delay_rate": "raw.orders",
    "freight_rate": "analytics.fact_orders",
    "freight_value": "analytics.v_delivered_orders",
    "gmv": "analytics.daily_metrics",
    "installment_order_share": "analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)",
    "on_time_delivery_rate": "raw.orders",
    "order_count": "analytics.daily_metrics",
    "payment_value": "analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)",
    "repeat_customer_count": "analytics.v_delivered_orders",
    "repeat_purchase_rate": "analytics.v_delivered_orders",
    "review_score_avg": "raw.order_reviews",
}
EXPECTED_LINEAGE = {
    "avg_items_per_order": "analytics.fact_orders",
    "repeat_customer_count": "analytics.fact_orders",
    "repeat_purchase_rate": "analytics.fact_orders",
}

METRICS = tuple(sorted(NORMAL_VALUES))
COUNT_METRICS = {"customer_count", "repeat_customer_count"}
RELATION_RE = re.compile(r"\b(?:analytics|raw)\.[a-z_][a-z_0-9]*\b", re.I)


def _delta(left: float | None, right: float | None) -> tuple[float | None, float | None]:
    if left is None or right is None:
        return None, None
    delta = round(right - left, 2)
    # Delta ratio is relative change against left_date, not percentage-point change.
    ratio = round(delta * 100.0 / left, 2) if left != 0 else None
    return delta, ratio


@pytest.fixture(scope="module")
def isolated_compare_db():
    admin_url_text = os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL")
    if not admin_url_text:
        pytest.skip("Set ISOLATED_TEST_POSTGRES_ADMIN_URL to an isolated-test PostgreSQL maintenance URL")
    admin_url = make_url(admin_url_text)
    database_name = f"metric_compare_acceptance_{uuid4().hex}"
    admin_engine = create_engine(
        admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT", pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )
    test_engine = None
    try:
        with admin_engine.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{database_name}"'))
        test_engine = create_engine(admin_url.set(database=database_name), pool_pre_ping=True,
                                    connect_args={"connect_timeout": 5})
        with test_engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA analytics"))
            conn.execute(text("CREATE SCHEMA raw"))
            conn.execute(text("""
                CREATE TABLE analytics.daily_metrics (
                    metric_date date PRIMARY KEY, gmv numeric(12,2), order_count integer,
                    avg_order_value numeric(12,2)
                )
            """))
            conn.execute(text("""
                CREATE TABLE analytics.fact_orders (
                    order_id text PRIMARY KEY, purchase_date date NOT NULL,
                    customer_unique_id text, item_gross_value numeric(12,2) NOT NULL,
                    freight_value numeric(12,2) NOT NULL, item_count integer NOT NULL,
                    order_status text NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE VIEW analytics.v_delivered_orders AS
                SELECT * FROM analytics.fact_orders WHERE order_status = 'delivered'
            """))
            conn.execute(text("""
                CREATE TABLE analytics.fact_order_payments (
                    order_id text NOT NULL, payment_value numeric(12,2) NOT NULL,
                    allocated_payment_value numeric(12,2) NOT NULL,
                    payment_installments integer NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE analytics.fact_fulfillment (
                    order_id text PRIMARY KEY, purchase_date date NOT NULL,
                    approval_lead_hours numeric(12,2), dispatch_lead_hours numeric(12,2),
                    delivery_transit_hours numeric(12,2)
                )
            """))
            conn.execute(text("""
                CREATE TABLE raw.orders (
                    order_id text PRIMARY KEY, order_purchase_timestamp timestamp NOT NULL,
                    order_status text NOT NULL, order_delivered_customer_date timestamp,
                    order_estimated_delivery_date timestamp
                )
            """))
            conn.execute(text("""
                CREATE TABLE raw.order_reviews (order_id text NOT NULL, review_score integer NOT NULL)
            """))

            conn.execute(text("""
                INSERT INTO analytics.daily_metrics VALUES
                    ('2024-01-01', 200, 2, 100),
                    ('2024-01-02', 600, 3, 200),
                    ('2024-01-06', 0, 0, 0)
            """))
            conn.execute(text("""
                INSERT INTO analytics.fact_orders VALUES
                    ('hist', '2023-12-31', 'c1', 10, 0, 1, 'delivered'),
                    ('a1', '2024-01-01', 'c1', 90, 10, 1, 'delivered'),
                    ('a2', '2024-01-01', 'c2', 90, 10, 3, 'delivered'),
                    ('b1', '2024-01-02', 'c1', 190, 10, 4, 'delivered'),
                    ('b2', '2024-01-02', 'c3', 190, 10, 6, 'delivered'),
                    ('b3', '2024-01-02', 'c4', 190, 10, 8, 'delivered'),
                    ('z3', '2024-01-03', NULL, 100, 0, 1, 'delivered'),
                    ('z5', '2024-01-05', 'c5', 100, 0, 2, 'delivered')
            """))
            conn.execute(text("""
                INSERT INTO analytics.fact_order_payments VALUES
                    ('a1', 100, 100, 2), ('a2', 100, 100, 1),
                    ('b1', 200, 200, 3), ('b2', 200, 200, 3), ('b3', 200, 200, 1),
                    ('z3', 0, 0, 1), ('z5', 100, 100, 1)
            """))
            conn.execute(text("""
                INSERT INTO analytics.fact_fulfillment VALUES
                    ('a1', '2024-01-01', 2, 4, 8), ('a2', '2024-01-01', 4, 8, 12),
                    ('b1', '2024-01-02', 6, 10, 20), ('b2', '2024-01-02', 8, 12, 22),
                    ('b3', '2024-01-02', 10, 14, 24),
                    ('z3', '2024-01-03', 0, 0, 0), ('z5', '2024-01-05', 0, 0, 0)
            """))
            conn.execute(text("""
                INSERT INTO raw.orders VALUES
                    ('a1', '2024-01-01 00:00:00', 'delivered', '2024-01-02 00:00:00', '2024-01-02 00:00:00'),
                    ('a2', '2024-01-01 12:00:00', 'delivered', '2024-01-04 12:00:00', '2024-01-03 12:00:00'),
                    ('a3', '2024-01-01 20:00:00', 'canceled', NULL, NULL),
                    ('b1', '2024-01-02 00:00:00', 'delivered', '2024-01-04 00:00:00', '2024-01-04 00:00:00'),
                    ('b2', '2024-01-02 08:00:00', 'delivered', '2024-01-06 08:00:00', '2024-01-05 08:00:00'),
                    ('b3', '2024-01-02 16:00:00', 'delivered', '2024-01-08 16:00:00', '2024-01-07 16:00:00'),
                    ('b4', '2024-01-02 20:00:00', 'canceled', NULL, NULL),
                    ('z3', '2024-01-03 00:00:00', 'delivered', '2024-01-03 00:00:00', '2024-01-03 00:00:00'),
                    ('z5', '2024-01-05 00:00:00', 'delivered', '2024-01-07 00:00:00', '2024-01-06 00:00:00')
            """))
            conn.execute(text("""
                INSERT INTO raw.order_reviews VALUES
                    ('a1', 2), ('a2', 4), ('b1', 1), ('b2', 1), ('b3', 3), ('z3', 5), ('z5', 5)
            """))

        statements: list[str] = []
        @event.listens_for(test_engine, "before_cursor_execute")
        def capture_sql(_conn, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)
        cases: list[dict] = []
        yield test_engine, statements, cases
        from _artifact_safety import isolated_output_path
        result_path = isolated_output_path("PLATFORM_COMPARE_RESULT_PATH")
        if result_path:
            path = Path(result_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"platform_compare_metric_count": len(METRICS), "cases": cases},
                                       indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        if test_engine is not None:
            test_engine.dispose()
        try:
            with admin_engine.connect() as conn:
                conn.execute(text(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)'))
        finally:
            admin_engine.dispose()


def _catalog():
    registry = get_registry()
    registry.load()
    return registry


def test_platform_compare_registry_inventory_exactly_20():
    registry = _catalog()
    actual = {d.metric_id for d in registry.list("platform")
              if "compare" in d.implementations["platform"].capabilities}
    assert actual == set(METRICS)
    assert len(actual) == 20


@pytest.mark.parametrize("metric_name", METRICS)
@pytest.mark.parametrize("scenario", ["normal", "swapped", "zero_baseline", "right_missing", "left_missing"])
def test_platform_compare_public_registry_semantics(
    monkeypatch, isolated_compare_db, metric_name, scenario
):
    engine, statements, cases = isolated_compare_db
    monkeypatch.setattr(daily_metrics, "engine", engine)
    registry = _catalog()
    expected_left_value, expected_right_value = NORMAL_VALUES[metric_name]
    if scenario == "normal":
        left_date, right_date = DATE_LEFT, DATE_RIGHT
        expected_left, expected_right = expected_left_value, expected_right_value
    elif scenario == "swapped":
        left_date, right_date = DATE_RIGHT, DATE_LEFT
        expected_left, expected_right = expected_right_value, expected_left_value
    elif scenario == "zero_baseline":
        if metric_name not in ZERO_BASE_DATE:
            cases.append({
                "metric": metric_name,
                "scenario": scenario,
                "status": "not_applicable",
                "reason": (
                    "A valid domain value cannot be zero: item count per delivered order is positive; "
                    "review scores are on a positive 1-5 scale."
                ),
            })
            pytest.skip("No valid zero value exists for this metric under its domain contract")
        left_date, right_date = ZERO_BASE_DATE[metric_name], DATE_RIGHT
        expected_left, expected_right = 0.00, expected_right_value
    elif scenario == "right_missing":
        left_date, right_date = DATE_LEFT, DATE_MISSING
        expected_left = expected_left_value
        # DATE_MISSING is a known empty date in this isolated fixture. Counts
        # therefore expect zero; this makes no claim about production coverage.
        expected_right = 0.00 if metric_name in COUNT_METRICS else None
    else:
        left_date, right_date = DATE_MISSING, DATE_LEFT
        expected_left = 0.00 if metric_name in COUNT_METRICS else None
        expected_right = expected_left_value

    before = len(statements)
    actual = registry.execute(metric_name, operation="compare", scope="platform",
                              left_date=left_date, right_date=right_date)
    sql_trace = statements[before:]
    expected_delta, expected_ratio = _delta(expected_left, expected_right)
    source = actual.get("source_table", "")
    source_relations = sorted(set(RELATION_RE.findall(source)))
    sql_relations = sorted({rel for sql in sql_trace for rel in RELATION_RE.findall(sql)})
    registered_lineage = registry.get_implementation(metric_name, "platform").lineage_source
    lineage_text = registered_lineage if isinstance(registered_lineage, str) else " ".join(registered_lineage)
    lineage_relations = sorted(set(RELATION_RE.findall(lineage_text)))
    expected_source_relations = sorted(set(RELATION_RE.findall(DIRECT_SOURCE[metric_name])))
    both_present = expected_left is not None and expected_right is not None
    expected_status = "ok" if both_present else "not_found"
    checks = {
        "left_date": actual.get("left_date") == left_date.isoformat(),
        "right_date": actual.get("right_date") == right_date.isoformat(),
        "left_value": actual.get("left_value") == expected_left,
        "right_value": actual.get("right_value") == expected_right,
        "right_minus_left_delta": actual.get("delta_value") == expected_delta,
        "relative_change_from_left": actual.get("delta_ratio") == expected_ratio,
        "compare_status_contract": actual.get("status") == expected_status,
        "missing_operand_null_deltas": (
            actual.get("delta_value") is None and actual.get("delta_ratio") is None
            if not both_present else None
        ),
        "zero_baseline_delta_valid_ratio_null": (
            actual.get("delta_value") == expected_delta and actual.get("delta_ratio") is None
            if both_present and expected_left == 0 else None
        ),
        "reported_source_matches_expected": actual.get("source_table") == DIRECT_SOURCE[metric_name],
        "reported_source_relations_present_in_sql": set(source_relations).issubset(set(sql_relations)),
        "target_lineage_matches_registry_and_expected": (
            actual.get("source_lineage") == EXPECTED_LINEAGE[metric_name]
            and lineage_text == EXPECTED_LINEAGE[metric_name]
            if metric_name in EXPECTED_LINEAGE else None
        ),
    }
    source_status = (
        "pass" if checks["reported_source_matches_expected"] and checks["reported_source_relations_present_in_sql"]
        else "direct_source_mismatch"
    )
    record = {
        "metric": metric_name, "scenario": scenario,
        "expectation_revision": (
            "In this isolated fixture only, DATE_MISSING is defined as no inserted business rows. "
            "COUNT-based customer_count and repeat_customer_count therefore expect 0; non-count metrics retain None. "
            "This corrects the acceptance expectation, not business code, and does not prove production completeness."
            if scenario in {"right_missing", "left_missing"} and metric_name in COUNT_METRICS else None
        ),
        "left_date": left_date.isoformat(), "right_date": right_date.isoformat(),
        "expected": {"left_value": expected_left, "right_value": expected_right,
                     "delta_value": expected_delta, "delta_ratio": expected_ratio},
        "actual": actual, "checks": checks, "sql": sql_trace,
        "numeric_status_groups": {
            "values_and_deltas": {
                key: checks[key] for key in (
                    "left_value", "right_value", "right_minus_left_delta", "relative_change_from_left",
                )
            },
            "status": {
                key: checks[key] for key in (
                    "compare_status_contract", "missing_operand_null_deltas",
                    "zero_baseline_delta_valid_ratio_null",
                )
            },
        },
        "reported_source_relations": source_relations,
        "expected_source": DIRECT_SOURCE[metric_name],
        "source_lineage": actual.get("source_lineage"),
        "expected_lineage": EXPECTED_LINEAGE.get(metric_name),
        "expected_source_relations": expected_source_relations,
        "executed_sql_relations": sql_relations,
        "registered_lineage": registered_lineage,
        "registered_lineage_relations": lineage_relations,
        "source_status": source_status,
        "lineage_exact_relation_coverage": set(sql_relations).issubset(set(lineage_relations)),
        "comparison_complete": actual.get("left_value") is not None and actual.get("right_value") is not None,
    }
    cases.append(record)

    failures = [name for name, passed in checks.items() if passed is False]
    assert not failures, (
        f"{metric_name}/{scenario}: failed={failures}; expected={record['expected']}; "
        f"actual={actual!r}; source={source!r}; SQL={sql_trace!r}"
    )
