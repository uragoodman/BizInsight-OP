"""Isolated single-point acceptance for every registered platform value metric."""

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


QUERY_DATE = "2024-01-02"
UNIT_DISPLAY = {
    "%": "%", "BRL": "雷亚尔", "BRL/order": "雷亚尔/单",
    "items/order": "件/单", "customers": "人", "orders": "单",
    "days": "天", "hours": "小时", "score": "分",
}

# Values are hand-calculated from the rows inserted below. SQL sources mirror
# the platform handler lineage; the SQL trace is checked independently from
# the source_table returned by the handler.
EXPECTED = {
    "gmv": {"value": 1300.00, "unit": "BRL", "source": "analytics.daily_metrics", "sql_sources": ("analytics.daily_metrics",)},
    "order_count": {"value": 5.00, "unit": "orders", "source": "analytics.daily_metrics", "sql_sources": ("analytics.daily_metrics",)},
    "avg_order_value": {"value": 260.00, "unit": "BRL/order", "source": "analytics.daily_metrics", "sql_sources": ("analytics.daily_metrics",)},
    "customer_count": {"value": 3.00, "unit": "customers", "source": "analytics.v_delivered_orders", "sql_sources": ("analytics.v_delivered_orders",)},
    "review_score_avg": {"value": 3.25, "unit": "score", "source": "raw.order_reviews", "sql_sources": ("raw.orders", "raw.order_reviews")},
    "freight_value": {"value": 160.00, "unit": "BRL", "source": "analytics.v_delivered_orders", "sql_sources": ("analytics.v_delivered_orders",)},
    "repeat_customer_count": {"value": 2.00, "unit": "customers", "source": "analytics.v_delivered_orders", "source_lineage": "analytics.fact_orders", "sql_sources": ("analytics.v_delivered_orders",)},
    "repeat_purchase_rate": {"value": 66.67, "unit": "%", "source": "analytics.v_delivered_orders", "source_lineage": "analytics.fact_orders", "sql_sources": ("analytics.v_delivered_orders",)},
    "payment_value": {"value": 1300.00, "unit": "BRL", "source": "analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)", "sql_sources": ("analytics.fact_order_payments", "analytics.fact_orders")},
    "installment_order_share": {"value": 13.85, "unit": "%", "source": "analytics.fact_order_payments p JOIN analytics.fact_orders o USING(order_id)", "sql_sources": ("analytics.fact_order_payments", "analytics.fact_orders")},
    "cancel_rate": {"value": 20.00, "unit": "%", "source": "raw.orders", "sql_sources": ("raw.orders",)},
    "delivery_delay_rate": {"value": 33.33, "unit": "%", "source": "raw.orders", "sql_sources": ("raw.orders",)},
    "avg_items_per_order": {"value": 4.00, "unit": "items/order", "source": "analytics.v_delivered_orders", "source_lineage": "analytics.fact_orders", "sql_sources": ("analytics.v_delivered_orders",)},
    "avg_delivery_lead_time": {"value": 3.00, "unit": "days", "source": "raw.orders", "sql_sources": ("raw.orders",)},
    "on_time_delivery_rate": {"value": 66.67, "unit": "%", "source": "raw.orders", "sql_sources": ("raw.orders",)},
    "freight_rate": {"value": 12.31, "unit": "%", "source": "analytics.fact_orders", "sql_sources": ("analytics.fact_orders",)},
    "bad_review_rate": {"value": 25.00, "unit": "%", "source": "raw.order_reviews", "sql_sources": ("raw.orders", "raw.order_reviews")},
    "avg_approval_lead_hours": {"value": 7.40, "unit": "hours", "source": "analytics.fact_fulfillment", "sql_sources": ("analytics.fact_fulfillment",)},
    "avg_dispatch_lead_hours": {"value": 8.40, "unit": "hours", "source": "analytics.fact_fulfillment", "sql_sources": ("analytics.fact_fulfillment",)},
    "avg_delivery_transit_hours": {"value": 16.60, "unit": "hours", "source": "analytics.fact_fulfillment", "sql_sources": ("analytics.fact_fulfillment",)},
    "order_approval_rate": {"value": 86.36, "unit": "%", "source": "analytics.v_order_cohort_funnel", "sql_sources": ("analytics.v_order_cohort_funnel",)},
    "order_shipment_rate": {"value": 50.00, "unit": "%", "source": "analytics.v_order_cohort_funnel", "sql_sources": ("analytics.v_order_cohort_funnel",)},
    "order_final_delivery_rate": {"value": 36.36, "unit": "%", "source": "analytics.v_order_cohort_funnel", "sql_sources": ("analytics.v_order_cohort_funnel",)},
    "review_comment_rate": {"value": 50.00, "unit": "%", "source": "raw.orders o LEFT JOIN raw.order_reviews r ON r.order_id = o.order_id", "sql_sources": ("raw.orders", "raw.order_reviews")},
}


@pytest.fixture(scope="module")
def isolated_metric_db():
    admin_url_text = os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL")
    if not admin_url_text:
        pytest.skip("Set ISOLATED_TEST_POSTGRES_ADMIN_URL to an isolated-test PostgreSQL maintenance URL")

    admin_url = make_url(admin_url_text)
    database_name = f"platform_value_acceptance_{uuid4().hex}"
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
                CREATE TABLE analytics.daily_metrics (
                    metric_date date PRIMARY KEY,
                    gmv numeric(14, 2), order_count bigint,
                    avg_order_value numeric(14, 2)
                )
            """))
            conn.execute(text("""
                CREATE TABLE analytics.fact_orders (
                    order_id text PRIMARY KEY,
                    customer_unique_id text,
                    item_count integer NOT NULL,
                    item_gross_value numeric(14, 2) NOT NULL,
                    freight_value numeric(14, 2) NOT NULL,
                    purchase_date date NOT NULL,
                    order_status text NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE VIEW analytics.v_delivered_orders AS
                SELECT * FROM analytics.fact_orders WHERE order_status = 'delivered'
            """))
            conn.execute(text("""
                CREATE TABLE analytics.fact_order_payments (
                    order_id text NOT NULL,
                    payment_value numeric(14, 2) NOT NULL,
                    allocated_payment_value numeric(14, 2) NOT NULL,
                    payment_installments integer NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE analytics.fact_fulfillment (
                    order_id text PRIMARY KEY,
                    purchase_date date NOT NULL,
                    approval_lead_hours numeric(14, 2),
                    dispatch_lead_hours numeric(14, 2),
                    delivery_transit_hours numeric(14, 2)
                )
            """))
            conn.execute(text("""
                CREATE TABLE analytics.v_order_cohort_funnel (
                    purchase_date date NOT NULL,
                    stage_approved integer NOT NULL,
                    stage_shipped integer NOT NULL,
                    final_delivered integer NOT NULL,
                    total_orders integer NOT NULL
                )
            """))
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
                    review_score integer,
                    review_comment_message text
                )
            """))

            conn.execute(text("""
                INSERT INTO analytics.daily_metrics VALUES
                    ('2024-01-01', 90000, 90, 1000),
                    ('2024-01-02', 1300, 5, 260),
                    ('2024-01-03', 80000, 80, 1000)
            """))
            conn.execute(text("""
                INSERT INTO analytics.fact_orders VALUES
                    ('hist_a', 'cust_a', 2, 180, 20, '2024-01-01', 'delivered'),
                    ('hist_c', 'cust_c', 1, 90, 10, '2024-01-01', 'delivered'),
                    ('o1', 'cust_a', 1, 90, 10, '2024-01-02', 'delivered'),
                    ('o2', 'cust_a', 3, 810, 90, '2024-01-02', 'delivered'),
                    ('o3', 'cust_b', 2, 80, 20, '2024-01-02', 'delivered'),
                    ('o4', 'cust_c', 6, 70, 30, '2024-01-02', 'delivered'),
                    ('o5', NULL, 8, 90, 10, '2024-01-02', 'delivered'),
                    ('after', 'cust_z', 99, 9000, 1000, '2024-01-03', 'delivered')
            """))
            conn.execute(text("""
                INSERT INTO analytics.fact_order_payments VALUES
                    -- o1's first payment appears twice at seller-allocation grain;
                    -- the raw 90 repeats while its allocated pieces total 90.
                    ('o1', 90, 60, 3), ('o1', 90, 30, 3), ('o1', 10, 10, 1),
                    ('o2', 90, 90, 3), ('o2', 810, 810, 1),
                    ('o3', 100, 100, 1), ('o4', 100, 100, 1), ('o5', 100, 100, 1),
                    ('hist_a', 5000, 5000, 10), ('after', 7000, 7000, 12)
            """))
            conn.execute(text("""
                INSERT INTO analytics.fact_fulfillment VALUES
                    ('f1', '2024-01-02', 1, 2, 3),
                    ('f2', '2024-01-02', 9, 10, 20),
                    ('f3', '2024-01-02', 9, 10, 20),
                    ('f4', '2024-01-02', 9, 10, 20),
                    ('f5', '2024-01-02', 9, 10, 20),
                    ('f_before', '2024-01-01', 1000, 1000, 1000),
                    ('f_after', '2024-01-03', 2000, 2000, 2000)
            """))
            conn.execute(text("""
                INSERT INTO analytics.v_order_cohort_funnel VALUES
                    ('2024-01-02', 1, 1, 0, 2),
                    ('2024-01-02', 18, 10, 8, 20),
                    ('2024-02-01', 0, 0, 0, 0),
                    ('2024-01-01', 900, 800, 700, 1000),
                    ('2024-01-03', 800, 700, 600, 900)
            """))
            conn.execute(text("""
                INSERT INTO raw.orders VALUES
                    ('raw_before', '2024-01-01 23:59:59', 'canceled', NULL, NULL),
                    ('o1', '2024-01-02 08:00:00', 'canceled', NULL, NULL),
                    ('o2', '2024-01-02 09:00:00', 'delivered', '2024-01-03 09:00:00', '2024-01-03 08:00:00'),
                    ('o3', '2024-01-02 10:00:00', 'delivered', '2024-01-04 10:00:00', '2024-01-05 10:00:00'),
                    ('o4', '2024-01-02 11:00:00', 'delivered', '2024-01-06 11:00:00', '2024-01-06 11:00:00'),
                    ('o5', '2024-01-02 12:00:00', 'delivered', '2024-01-07 12:00:00', NULL),
                    ('edge_a', '2024-03-01 08:00:00', 'delivered', NULL, NULL),
                    ('edge_b', '2024-03-01 09:00:00', 'delivered', NULL, NULL),
                    ('edge_c', '2024-03-01 10:00:00', 'delivered', NULL, NULL),
                    ('edge_unreviewed', '2024-03-01 11:00:00', 'delivered', NULL, NULL),
                    ('no_review', '2024-03-02 08:00:00', 'delivered', NULL, NULL),
                    ('raw_after', '2024-01-03 00:00:00', 'canceled', NULL, NULL)
            """))
            conn.execute(text("""
                INSERT INTO raw.order_reviews VALUES
                    ('o1', 1, 'poor'), ('o2', 3, '   '),
                    ('o3', 4, NULL), ('o4', 5, 'clear description'),
                    ('edge_a', NULL, 'comment without a score'),
                    ('edge_a', 4, ''),
                    ('edge_b', 2, '   '),
                    ('edge_c', NULL, NULL),
                    ('edge_c', 5, 'second comment')
            """))

        statements: list[str] = []

        @event.listens_for(isolated_engine, "before_cursor_execute")
        def capture_sql(_conn, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)

        yield isolated_engine, statements
    finally:
        if isolated_engine is not None:
            isolated_engine.dispose()
        try:
            with admin_engine.connect() as conn:
                conn.execute(text(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)'))
        finally:
            admin_engine.dispose()


def test_registry_contains_exactly_the_24_platform_value_metrics():
    registry = get_registry()
    actual_ids = {
        metric.metric_id
        for metric in registry.list(scope="platform")
        if "value" in metric.implementations["platform"].capabilities
    }
    assert actual_ids == set(EXPECTED)


def test_every_registered_platform_value_metric_has_a_single_point_execution_path():
    registry = get_registry()
    platform_value_ids = {
        metric.metric_id
        for metric in registry.list(scope="platform")
        if "value" in metric.implementations["platform"].capabilities
    }
    # query_metric dispatches aggregated metrics through
    # _query_from_aggregated_table and every other metric through the direct map.
    executable_ids = daily_metrics.AGGREGATED_METRICS | set(
        daily_metrics.DIRECT_QUERY_HANDLERS
    )
    assert platform_value_ids <= executable_ids
    assert platform_value_ids == set(EXPECTED)


@pytest.mark.parametrize("metric_id", sorted(EXPECTED))
def test_platform_single_point_value_via_registry_dispatch(
    monkeypatch, isolated_metric_db, metric_id
):
    registry = get_registry()
    definition = registry.get(metric_id)
    isolated_engine, statements = isolated_metric_db
    monkeypatch.setattr(daily_metrics, "engine", isolated_engine)
    statement_start = len(statements)

    try:
        actual = registry.execute(
            metric_id,
            operation="value",
            scope="platform",
            metric_date=date.fromisoformat(QUERY_DATE),
        )
    except Exception as exc:  # Preserve dispatch failures as per-metric evidence.
        actual = {
            "metric_name": metric_id,
            "metric_date": None,
            "value": None,
            "source_table": None,
            "status": "exception",
            "exception": f"{type(exc).__name__}: {exc}",
        }
    executed_sql = statements[statement_start:]
    expected = EXPECTED[metric_id]
    normalized_sql = " ".join(" ".join(executed_sql).lower().split())
    checks = {
        "registry_unit": definition.unit == expected["unit"],
        "date": actual.get("metric_date") == QUERY_DATE,
        "source_label": actual.get("source_table") == expected["source"],
        "source_lineage": (
            actual.get("source_lineage") == expected["source_lineage"]
            if "source_lineage" in expected else True
        ),
        "sql_sources": all(source.lower() in normalized_sql for source in expected["sql_sources"]),
        "status": actual.get("status") == "ok",
        "value": actual.get("value") is not None
        and actual.get("value") == pytest.approx(expected["value"], abs=0.01),
    }
    report = {
        "metric_id": metric_id,
        "expected": expected,
        "actual": actual,
        "unit_from_registry": definition.unit,
        "unit_in_result": actual.get("unit"),
        "source_lineage": actual.get("source_lineage"),
        "sql": executed_sql,
        "checks": checks,
    }
    print("PLATFORM_VALUE_RESULT " + json.dumps(report, ensure_ascii=False, sort_keys=True, default=str))
    failures = [check for check, passed in checks.items() if not passed]
    assert not failures, f"{metric_id}: failed checks={failures}; report={report!r}"


@pytest.mark.parametrize(
    "metric_id",
    [
        "order_approval_rate",
        "order_shipment_rate",
        "order_final_delivery_rate",
        "review_comment_rate",
    ],
)
def test_newly_dispatched_metrics_return_not_found_for_empty_day(
    monkeypatch, isolated_metric_db, metric_id
):
    registry = get_registry()
    isolated_engine, statements = isolated_metric_db
    monkeypatch.setattr(daily_metrics, "engine", isolated_engine)
    statement_start = len(statements)

    actual = registry.execute(
        metric_id,
        operation="value",
        scope="platform",
        metric_date=date(2024, 1, 4),
    )
    executed_sql = statements[statement_start:]
    expected = EXPECTED[metric_id]
    normalized_sql = " ".join(" ".join(executed_sql).lower().split())
    report = {
        "metric_id": metric_id,
        "expected": {"value": None, "status": "not_found"},
        "actual": actual,
        "sql": executed_sql,
    }
    print("PLATFORM_EMPTY_RESULT " + json.dumps(report, ensure_ascii=False, sort_keys=True, default=str))
    assert actual.get("value") is None, report
    assert actual.get("status") == "not_found", report
    assert actual.get("metric_date") == "2024-01-04", report
    assert actual.get("source_table") == expected["source"], report
    assert all(source.lower() in normalized_sql for source in expected["sql_sources"]), report


@pytest.mark.parametrize(
    "metric_id",
    ["order_approval_rate", "order_shipment_rate", "order_final_delivery_rate"],
)
def test_funnel_rates_return_not_found_when_denominator_is_zero(
    monkeypatch, isolated_metric_db, metric_id
):
    registry = get_registry()
    isolated_engine, statements = isolated_metric_db
    monkeypatch.setattr(daily_metrics, "engine", isolated_engine)
    statement_start = len(statements)

    actual = registry.execute(
        metric_id,
        operation="value",
        scope="platform",
        metric_date=date(2024, 2, 1),
    )
    report = {
        "metric_id": metric_id,
        "expected": {"value": None, "status": "not_found"},
        "actual": actual,
        "sql": statements[statement_start:],
    }
    print("PLATFORM_ZERO_DENOMINATOR_RESULT " + json.dumps(report, ensure_ascii=False, sort_keys=True, default=str))
    assert actual.get("value") is None, report
    assert actual.get("status") == "not_found", report
    assert actual.get("metric_date") == "2024-02-01", report


def test_review_comment_rate_counts_review_records_and_excludes_orders_without_reviews(
    monkeypatch, isolated_metric_db
):
    registry = get_registry()
    isolated_engine, statements = isolated_metric_db
    monkeypatch.setattr(daily_metrics, "engine", isolated_engine)
    statement_start = len(statements)

    actual = registry.execute(
        "review_comment_rate",
        operation="value",
        scope="platform",
        metric_date=date(2024, 3, 1),
    )
    executed_sql = statements[statement_start:]
    normalized_sql = " ".join(" ".join(executed_sql).lower().split())
    report = {
        "metric_id": "review_comment_rate",
        "expected": {"value": 40.0, "status": "ok"},
        "actual": actual,
        "sql": executed_sql,
    }
    print("PLATFORM_REVIEW_COMMENT_BOUNDARY " + json.dumps(report, ensure_ascii=False, sort_keys=True, default=str))
    assert actual.get("value") == pytest.approx(40.0), report
    assert actual.get("status") == "ok", report
    assert actual.get("metric_date") == "2024-03-01", report
    assert actual.get("source_table") == EXPECTED["review_comment_rate"]["source"], report
    assert "count(r.order_id)" in normalized_sql, report
    assert "date(o.order_purchase_timestamp)" in normalized_sql, report


def test_review_comment_rate_is_not_found_when_day_has_orders_but_no_reviews(
    monkeypatch, isolated_metric_db
):
    registry = get_registry()
    isolated_engine, statements = isolated_metric_db
    monkeypatch.setattr(daily_metrics, "engine", isolated_engine)
    statement_start = len(statements)

    actual = registry.execute(
        "review_comment_rate",
        operation="value",
        scope="platform",
        metric_date=date(2024, 3, 2),
    )
    report = {
        "metric_id": "review_comment_rate",
        "expected": {"value": None, "status": "not_found"},
        "actual": actual,
        "sql": statements[statement_start:],
    }
    print("PLATFORM_REVIEW_COMMENT_NO_REVIEWS " + json.dumps(report, ensure_ascii=False, sort_keys=True, default=str))
    assert actual.get("value") is None, report
    assert actual.get("status") == "not_found", report
    assert actual.get("metric_date") == "2024-03-02", report


def _assert_business_checks(metric_id: str, checks: dict[str, bool]) -> None:
    failed_checks = [name for name, passed in checks.items() if not passed]
    assert not failed_checks, f"{metric_id}: failed business checks={failed_checks}"


def test_full_workflow_business_check_assertion_rejects_failures():
    all_true = {"plan_matches_request": True, "evidence_is_bound": True}
    assert _assert_business_checks("all_checks_true", all_true) is None

    one_false = {"plan_matches_request": True, "evidence_is_bound": False}
    with pytest.raises(
        AssertionError,
        match=r"counterexample_metric: failed business checks=\['evidence_is_bound'\]",
    ):
        _assert_business_checks("counterexample_metric", one_false)


@pytest.mark.parametrize("metric_id", sorted(EXPECTED))
def test_full_chat_workflow_for_each_platform_value_metric(
    monkeypatch, isolated_metric_db, metric_id,
):
    """Observe each point-value task through the real chat workflow entry.

    This is an acceptance recorder: business checks are serialized per case
    rather than asserted here, so pytest collection success cannot be confused
    with a passing business outcome.
    """
    from app.graph import workflow
    from app.core import time_expression
    from app.graph.nodes import triage as triage_module
    from app.llm import client as llm_client
    from app.mcp import client as mcp_client
    from app.metrics.registry import get_registry
    from app.tenancy import TenantContext

    registry = get_registry()
    definition = registry.get(metric_id)
    isolated_engine, statements = isolated_metric_db
    monkeypatch.setattr(daily_metrics, "engine", isolated_engine)
    monkeypatch.setattr(time_expression, "engine", isolated_engine)
    monkeypatch.setattr(triage_module, "llm_classify", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_client, "llm_extract", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_client, "llm_chat", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mcp_client,
        "call_local_mcp_tool",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("isolated full-workflow test blocks MCP fallback")
        ),
    )

    captured: dict = {}
    monkeypatch.setattr(workflow, "write_trace", lambda _trace_id, state: captured.update(state))
    monkeypatch.setattr(workflow, "_persist_chat_turn", lambda *args, **kwargs: None)

    # Each question is metric-specific and date-specific; only the registry
    # display label varies, so the test does not inject a plan or metric ID.
    question = f"请查询统计日期 {QUERY_DATE} 的“{definition.display_name}”指标值。"
    request_trace_id = f"platform-value-workflow-{uuid4().hex}"
    sql_start = len(statements)
    workflow_error = None
    result: dict = {}
    try:
        result = workflow.run_chat_workflow(
            question,
            tenant_context=TenantContext(
                tenant_id="platform", user_id=f"value-acceptance-{uuid4().hex[:8]}",
                role="admin", scope_type="platform",
            ),
            memory_enabled=False,
            trace_id=request_trace_id,
        )
    except Exception as exc:  # record unexpected workflow failures per metric
        workflow_error = f"{type(exc).__name__}: {exc}"

    executed_sql = [" ".join(sql.split()) for sql in statements[sql_start:]]
    metric_sql = [
        sql for sql in executed_sql
        if "SELECT MIN(purchase_date) AS first_date, MAX(purchase_date) AS last_date" not in sql
    ]
    plans = ((captured.get("metric_query_batch_plan") or {}).get("plans") or [])
    query_items = captured.get("query_results") or []
    query_item = query_items[0] if query_items else {}
    inner_result = query_item.get("result") or {}
    metric_result = inner_result.get("metric_result") or {}
    expected = EXPECTED[metric_id]
    selected_plan = plans[0] if len(plans) == 1 else {}
    expected_sources_found = all(
        source.lower() in " ".join(metric_sql).lower()
        for source in expected["sql_sources"]
    )

    evidence_bundles = captured.get("evidence_bundles") or []
    evidence_id = query_item.get("evidence_id")
    presentation = (result or {}).get("presentation") or {}
    facts = presentation.get("facts") or []
    if not evidence_id and facts:
        fact_ids = facts[0].get("evidence_ids") or []
        evidence_id = fact_ids[0] if fact_ids else None
    matching_evidence = next(
        (bundle for bundle in evidence_bundles if bundle.get("evidence_id") == evidence_id),
        None,
    )
    evidence_content = (matching_evidence or {}).get("content") or {}
    evidence_result = evidence_content.get("result") or {}
    critic_review = ((captured.get("critic_result") or {}).get("data_review") or {})
    answer = str((result or {}).get("answer") or "")

    checks = {
        "selected_metric_operation_date": (
            selected_plan.get("metric_name") == metric_id
            and selected_plan.get("query_type") == "metric_value"
            and selected_plan.get("metric_date") == QUERY_DATE
        ),
        "actual_metric_sql_and_expected_source": bool(metric_sql) and expected_sources_found,
        "tool_result_matches_manual_value_date_source_status": (
            inner_result.get("status") == "ok"
            and metric_result.get("metric_name") == metric_id
            and metric_result.get("metric_date") == QUERY_DATE
            and metric_result.get("source_table") == expected["source"]
            and metric_result.get("value") is not None
            and abs(float(metric_result["value"]) - float(expected["value"])) <= 0.01
        ),
        "tool_result_unit_matches_registry": metric_result.get("unit") == expected["unit"],
        "fact_unit_matches_registry": bool(facts) and facts[0].get("unit") == expected["unit"],
        "evidence_matches_result_and_plan": (
            matching_evidence is not None
            and evidence_result == inner_result
            and (
                evidence_content.get("metric_id") == metric_id
                or (evidence_result.get("parsed") or {}).get("metric_name") == metric_id
            )
            and (evidence_result.get("parsed") or {}).get("query_type") == "metric_value"
            and (evidence_result.get("parsed") or {}).get("metric_date") == QUERY_DATE
        ),
        "evidence_bound_to_request_id": bool(
            matching_evidence and evidence_content.get("request_id") == query_item.get("request_id")
        ),
        "critic_approved": critic_review.get("status") == "approved",
        "final_answer_has_expected_value_and_date": (
            f"{float(expected['value']):.2f}" in answer and QUERY_DATE in answer
        ),
        "final_answer_unit_matches_registry": (
            bool(facts)
            and facts[0].get("unit") == expected["unit"]
            and UNIT_DISPLAY.get(expected["unit"], expected["unit"]) in answer
        ),
        "final_task_completed": presentation.get("status") == "completed",
    }
    report = {
        "metric_id": metric_id,
        "question": question,
        "expected": expected,
        "selected_plan": selected_plan,
        "request_id": query_item.get("request_id"),
        "tool_name": query_item.get("tool_name"),
        "tool_result": inner_result,
        "evidence_id": evidence_id,
        "evidence_content": evidence_content,
        "critic_review": critic_review,
        "presentation": presentation,
        "answer": answer,
        "workflow_error": workflow_error,
        "tool_call_count": len(captured.get("tool_calls") or []),
        "tool_calls": captured.get("tool_calls") or [],
        "sql_attempt_count": len(metric_sql),
        "sql_attempts": metric_sql,
        "all_sql_statements": executed_sql,
        "checks": checks,
        "tool_query_correct": all(checks[k] for k in (
            "selected_metric_operation_date",
            "actual_metric_sql_and_expected_source",
            "tool_result_matches_manual_value_date_source_status",
            "tool_result_unit_matches_registry",
        )),
        "full_task_pass": all(checks.values()),
    }
    from _artifact_safety import isolated_output_path
    trace_dir = isolated_output_path("PLATFORM_VALUE_WORKFLOW_TRACE_DIR")
    if trace_dir:
        trace_path = Path(trace_dir) / f"{metric_id}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps({
            **report,
            "workflow_state": captured,
            "workflow_result": result,
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("PLATFORM_VALUE_WORKFLOW_RESULT " + json.dumps(report, ensure_ascii=False, sort_keys=True, default=str))
    assert trace_dir is None or (Path(trace_dir) / f"{metric_id}.json").exists()
    _assert_business_checks(metric_id, checks)
