"""Isolated PostgreSQL acceptance for every registered platform daily trend."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url

from app.data.repositories import daily_metrics
from app.metrics.registry import get_registry


START = date(2024, 5, 2)
END = date(2024, 5, 4)
DATES = [date(2024, 5, day) for day in (2, 3, 4)]

# Values are independently calculated from the rows seeded below.  The source
# label is the handler's declared result label; SQL relations are recorded from
# the actual execution trace and evaluated separately.
EXPECTED = {
    "customer_count": ([2, 3, 4], "analytics.v_delivered_orders"),
    "repeat_customer_count": ([0, 2, 3], "analytics.fact_orders"),
    "repeat_purchase_rate": ([0, 66.67, 75.0], "analytics.fact_orders"),
    "avg_items_per_order": ([1.5, 2.0, 2.5], "analytics.fact_orders"),
    "on_time_delivery_rate": ([50.0, 66.67, 75.0], "raw.orders"),
    "delivery_delay_rate": ([50.0, 33.33, 25.0], "raw.orders"),
    "avg_delivery_lead_time": ([2.0, 3.0, 4.0], "raw.orders"),
    "avg_approval_lead_hours": ([2.0, 4.0, 6.0], "analytics.fact_fulfillment"),
    "avg_dispatch_lead_hours": ([3.0, 6.0, 9.0], "analytics.fact_fulfillment"),
    "avg_delivery_transit_hours": ([4.0, 8.0, 12.0], "analytics.fact_fulfillment"),
    "order_approval_rate": ([50.0, 66.67, 75.0], "analytics.v_order_cohort_funnel"),
    "order_shipment_rate": ([25.0, 50.0, 62.5], "analytics.v_order_cohort_funnel"),
    "order_final_delivery_rate": ([25.0, 33.33, 50.0], "analytics.v_order_cohort_funnel"),
    "payment_value": ([30.0, 90.0, 180.0], "analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)"),
    "installment_order_share": ([33.33, 66.67, 50.0], "analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)"),
    "review_score_avg": ([3.0, 4.0, 3.75], "raw.order_reviews"),
    "bad_review_rate": ([50.0, 0.0, 25.0], "raw.order_reviews"),
    "cancel_rate": ([33.33, 25.0, 20.0], "raw.orders"),
    "review_comment_rate": ([50.0, 66.67, 75.0], "raw.orders o LEFT JOIN raw.order_reviews r ON r.order_id = o.order_id"),
    "gmv": ([33.0, 108.0, 180.0], "analytics.daily_metrics"),
    "order_count": ([2.0, 3.0, 4.0], "analytics.daily_metrics"),
    "avg_order_value": ([16.5, 36.0, 45.0], "analytics.daily_metrics"),
    "freight_value": ([3.0, 8.0, 20.0], "analytics.v_delivered_orders"),
    "freight_rate": ([9.09, 7.41, 11.11], "analytics.fact_orders"),
}
EXPECTED_SQL_RELATIONS = {
    "customer_count": ("analytics.v_delivered_orders",),
    "repeat_customer_count": ("analytics.v_delivered_orders",),
    "repeat_purchase_rate": ("analytics.v_delivered_orders",),
    "avg_items_per_order": ("analytics.v_delivered_orders",),
    "on_time_delivery_rate": ("raw.orders",),
    "delivery_delay_rate": ("raw.orders",),
    "avg_delivery_lead_time": ("raw.orders",),
    "avg_approval_lead_hours": ("analytics.fact_fulfillment",),
    "avg_dispatch_lead_hours": ("analytics.fact_fulfillment",),
    "avg_delivery_transit_hours": ("analytics.fact_fulfillment",),
    "order_approval_rate": ("analytics.v_order_cohort_funnel",),
    "order_shipment_rate": ("analytics.v_order_cohort_funnel",),
    "order_final_delivery_rate": ("analytics.v_order_cohort_funnel",),
    "payment_value": ("analytics.fact_order_payments", "analytics.fact_orders"),
    "installment_order_share": ("analytics.fact_order_payments", "analytics.fact_orders"),
    "review_score_avg": ("raw.orders", "raw.order_reviews"),
    "bad_review_rate": ("raw.orders", "raw.order_reviews"),
    "cancel_rate": ("raw.orders",),
    "review_comment_rate": ("raw.orders", "raw.order_reviews"),
    "gmv": ("analytics.daily_metrics",),
    "order_count": ("analytics.daily_metrics",),
    "avg_order_value": ("analytics.daily_metrics",),
    "freight_value": ("analytics.v_delivered_orders",),
    "freight_rate": ("analytics.fact_orders",),
}


def _relations(sql_statements: list[str]) -> list[str]:
    normalized = " ".join(sql_statements).lower()
    known = (
        "analytics.daily_metrics",
        "analytics.fact_order_payments",
        "analytics.fact_fulfillment",
        "analytics.fact_orders",
        "analytics.v_delivered_orders",
        "analytics.v_order_cohort_funnel",
        "raw.order_reviews",
        "raw.orders",
    )
    return [relation for relation in known if relation in normalized]


def _lineage_covered_by_trace(lineage: str, actual_relations: list[str]) -> bool:
    # '+' means all inputs participate; '/' denotes alternative/conditional
    # lineage and only one listed relation needs to be used by this handler.
    for required_group in lineage.split("+"):
        alternatives = [item.strip().lower() for item in required_group.split("/") if item.strip()]
        if alternatives and not any(item in actual_relations for item in alternatives):
            return False
    return True


@pytest.fixture(scope="module")
def isolated_trend_db(request):
    admin_url_text = os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL")
    if not admin_url_text:
        pytest.skip("Set ISOLATED_TEST_POSTGRES_ADMIN_URL to an isolated-test PostgreSQL maintenance URL")
    admin_url = make_url(admin_url_text)
    database_name = f"platform_trend_acceptance_{uuid4().hex}"
    admin_engine = create_engine(admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT", pool_pre_ping=True)
    engine = None
    statements: list[str] = []
    sql_events: list[dict] = []
    cases: list[dict] = []
    try:
        with admin_engine.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{database_name}"'))
        engine = create_engine(admin_url.set(database=database_name), pool_pre_ping=True)
        with engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA analytics"))
            conn.execute(text("CREATE SCHEMA raw"))
            conn.execute(text("CREATE TABLE analytics.daily_metrics (metric_date date PRIMARY KEY, gmv numeric, order_count bigint, avg_order_value numeric)"))
            conn.execute(text("CREATE TABLE analytics.fact_orders (order_id text PRIMARY KEY, customer_unique_id text, item_count int NOT NULL, item_gross_value numeric NOT NULL, freight_value numeric NOT NULL, purchase_date date NOT NULL, order_status text NOT NULL)"))
            conn.execute(text("CREATE VIEW analytics.v_delivered_orders AS SELECT * FROM analytics.fact_orders WHERE order_status='delivered'"))
            conn.execute(text("CREATE TABLE analytics.fact_order_payments (order_id text NOT NULL, payment_value numeric NOT NULL, allocated_payment_value numeric NOT NULL, payment_installments int NOT NULL)"))
            conn.execute(text("CREATE TABLE analytics.fact_fulfillment (order_id text PRIMARY KEY, purchase_date date NOT NULL, approval_lead_hours numeric, dispatch_lead_hours numeric, delivery_transit_hours numeric)"))
            conn.execute(text("CREATE TABLE analytics.v_order_cohort_funnel (purchase_date date NOT NULL, stage_approved int NOT NULL, stage_shipped int NOT NULL, final_delivered int NOT NULL, total_orders int NOT NULL)"))
            conn.execute(text("CREATE TABLE raw.orders (order_id text PRIMARY KEY, order_purchase_timestamp timestamp NOT NULL, order_status text NOT NULL, order_delivered_customer_date timestamp, order_estimated_delivery_date timestamp)"))
            conn.execute(text("CREATE TABLE raw.order_reviews (order_id text NOT NULL, review_score int, review_comment_message text)"))

            conn.execute(text("""INSERT INTO analytics.daily_metrics VALUES
                ('2024-05-01',99999,99,1010),('2024-05-02',33,2,16.5),
                ('2024-05-03',108,3,36),('2024-05-04',180,4,45),('2024-05-05',88888,88,1000)"""))

            # Delivered orders are the shared analytic facts. Every day has
            # distinct customers (2/3/4), different item averages, freight,
            # and a day-varying freight-to-GMV ratio. Earlier dates establish
            # repeat-customer history; May 4 is range noise.
            fact_rows = []
            customer_days = [
                [("a", 1, 10, 1), ("b", 2, 20, 2)],
                [("a", 1, 20, 2), ("b", 2, 30, 3), ("c", 3, 50, 3)],
                [("a", 1, 20, 3), ("b", 2, 30, 4), ("c", 3, 50, 5), ("d", 4, 60, 8)],
            ]
            payments = []
            fulfillment = []
            raw_orders = []
            reviews = []
            for day_index, rows in enumerate(customer_days, start=1):
                current = date(2024, 5, day_index + 1)
                for order_index, (customer, items, gross, freight) in enumerate(rows, start=1):
                    order_id = f"d{day_index}o{order_index}"
                    fact_rows.append((order_id, customer, items, gross, freight, current, "delivered"))
                    # Per day totals: payment 30/90/180; installment shares
                    # use payment-weighted amounts, not mean of order rates.
                    if day_index == 1:
                        payment_rows = [(20, 1), (10, 3)] if order_index == 1 else []
                    elif day_index == 2:
                        payment_rows = [(30, 1)] if order_index == 1 else [(60, 3)] if order_index == 2 else []
                    else:
                        payment_rows = [(90, 1)] if order_index == 1 else [(90, 3)] if order_index == 2 else []
                    payments.extend((order_id, value, installments) for value, installments in payment_rows)
                    fulfillment.append((order_id, current, day_index * 2, day_index * 3, day_index * 4))
                    # Lead time and delivery rates: D1 1/2 on-time, D2 2/3,
                    # D3 3/4; average lead time exactly 2/3/4 days.
                    purchase = f"{current.isoformat()} 08:00:00"
                    delivered_date = current.fromordinal(current.toordinal() + day_index + 1)
                    delivered = f"{delivered_date.isoformat()} 08:00:00"
                    estimated_date = delivered_date
                    if order_index == len(rows):
                        estimated_date = delivered_date.fromordinal(delivered_date.toordinal() - 1)
                    estimated = f"{estimated_date.isoformat()} 08:00:00"
                    raw_orders.append((order_id, purchase, "delivered", delivered, estimated))
                    score = ([1, 5][order_index - 1] if day_index == 1 else [4, 4, 4][order_index - 1] if day_index == 2 else [2, 4, 5, 4][order_index - 1])
                    comment = "text" if order_index <= day_index else "   "
                    reviews.append((order_id, score, comment))
                # Add one canceled order each day.
                canceled_count = 1 if day_index < 3 else 1
                for canceled_index in range(canceled_count):
                    order_id = f"d{day_index}c{canceled_index}"
                    raw_orders.append((order_id, f"{current.isoformat()} 20:00:00", "canceled", None, None))
            # Prior delivered orders establish two returning customers on May 2.
            fact_rows.extend([
                ("prior_x", "x", 1, 1, 1, date(2024, 5, 1), "delivered"),
                ("prior_y", "y", 1, 1, 1, date(2024, 5, 1), "delivered"),
            ])
            for order_id, current, *_ in fact_rows:
                if order_id.startswith("d"):
                    pass
            conn.execute(text("INSERT INTO analytics.fact_orders VALUES (:id,:customer,:items,:gross,:freight,:day,:status)"), [dict(id=r[0], customer=r[1], items=r[2], gross=r[3], freight=r[4], day=r[5], status=r[6]) for r in fact_rows])
            conn.execute(text("INSERT INTO analytics.fact_order_payments VALUES (:id,:value,:allocated,:installments)"), [dict(id=r[0], value=r[1], allocated=r[1], installments=r[2]) for r in payments])
            conn.execute(text("INSERT INTO analytics.fact_fulfillment VALUES (:id,:day,:approval,:dispatch,:transit)"), [dict(id=r[0], day=r[1], approval=r[2], dispatch=r[3], transit=r[4]) for r in fulfillment])
            conn.execute(text("INSERT INTO analytics.v_order_cohort_funnel VALUES ('2024-05-02',2,1,1,4),('2024-05-03',4,3,2,6),('2024-05-04',6,5,4,8),('2024-05-01',999,999,999,1000),('2024-05-05',888,888,888,900)"))
            conn.execute(text("INSERT INTO raw.orders VALUES (:id,:purchase,:status,:delivered,:estimated)"), [dict(id=r[0], purchase=r[1], status=r[2], delivered=r[3], estimated=r[4]) for r in raw_orders])
            conn.execute(text("INSERT INTO raw.order_reviews VALUES (:id,:score,:comment)"), [dict(id=r[0], score=r[1], comment=r[2]) for r in reviews])

        @event.listens_for(engine, "before_cursor_execute")
        def capture_sql(_conn, _cursor, statement, parameters, _context, _many):
            statements.append(statement)
            sql_events.append({"statement": statement, "parameters": parameters})

        yield engine, statements, sql_events, cases
    finally:
        from _artifact_safety import isolated_output_path
        output_path = isolated_output_path("PLATFORM_TREND_RESULT_PATH")
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            inventory = [m for m in get_registry().list(scope="platform") if "trend" in m.implementations["platform"].capabilities]
            Path(output_path).write_text(json.dumps({"scope": "platform", "operation": "daily trend", "inventory": [{"metric_id": m.metric_id, "unit": m.unit, "lineage_source": m.implementations["platform"].lineage_source} for m in inventory], "inventory_count": len(inventory), "cases": cases}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if engine is not None:
            engine.dispose()
        with admin_engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)'))
        admin_engine.dispose()


def test_registered_platform_trend_inventory_is_explicit():
    registry = get_registry()
    actual = {m.metric_id for m in registry.list(scope="platform") if "trend" in m.implementations["platform"].capabilities}
    assert actual == set(EXPECTED)


@pytest.mark.parametrize("metric_id", sorted(EXPECTED))
def test_platform_daily_trend_against_independent_expected_series(monkeypatch, isolated_trend_db, metric_id):
    registry = get_registry()
    engine, statements, sql_events, cases = isolated_trend_db
    monkeypatch.setattr(daily_metrics, "engine", engine)
    definition = registry.get(metric_id)
    start_at = len(statements)
    event_start = len(sql_events)
    try:
        actual = registry.execute(metric_id, operation="trend", scope="platform", start_date=START, end_date=END, granularity="day")
    except Exception as exc:
        actual = {"metric_name": metric_id, "status": "exception", "exception": f"{type(exc).__name__}: {exc}"}
    sql = statements[start_at:]
    trace = sql_events[event_start:]
    expected_values, expected_source = EXPECTED[metric_id]
    actual_points = actual.get("points", [])
    values = [point.get("value") for point in actual_points]
    expected_dates = [item.isoformat() for item in DATES]
    actual_dates = [point.get("bucket") for point in actual_points]
    numeric_checks = {
        "values": len(values) == 3 and all(value is not None and value == pytest.approx(expected, abs=0.01) for value, expected in zip(values, expected_values)),
        "date_range": actual.get("start_date") == START.isoformat() and actual.get("end_date") == END.isoformat(),
        "daily_buckets_and_order": actual_dates == expected_dates,
        "unit_from_registry": bool(definition.unit),
        "status": actual.get("status") == "ok",
    }
    normalized_sql = " ".join(sql).lower()
    actual_relations = _relations(sql)
    expected_relations = EXPECTED_SQL_RELATIONS[metric_id]
    declared_lineage = definition.implementations["platform"].lineage_source
    metadata_checks = {
        "result_source_label_matches_handler": actual.get("source_table") == expected_source,
        "sql_trace_has_expected_relations": all(relation in actual_relations for relation in expected_relations),
        "registered_lineage_covered_by_trace": _lineage_covered_by_trace(declared_lineage, actual_relations),
    }
    case = {
        "metric_id": metric_id,
        "unit": definition.unit,
        "expected": {"dates": expected_dates, "values": expected_values, "source_label": expected_source},
        "actual": actual,
        "actual_sql_relations": actual_relations,
        "registered_lineage_source": definition.implementations["platform"].lineage_source,
        "sql_trace": trace,
        "numeric_checks": numeric_checks,
        "source_metadata_checks": metadata_checks,
        "numeric_pass": all(numeric_checks.values()),
        "source_metadata_pass": all(metadata_checks.values()),
    }
    cases.append(case)
    print("PLATFORM_TREND_RESULT " + json.dumps(case, ensure_ascii=False, sort_keys=True, default=str))
    assert all(numeric_checks.values()), case
