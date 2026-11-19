"""Non-GMV diagnostic development acceptance with real isolated SQL facts."""
import json
import ipaddress
import os
import socket
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError

from test_platform_metric_trend_acceptance import isolated_trend_db  # noqa: F401

CASES = [
    ("order_dimension", "order_count", "订单量", "下降", "dimension"),
    ("freight_dimension", "freight_value", "运费总额", "下降", "dimension"),
    ("aov_ratio", "avg_order_value", "客单价", "上升", "ratio"),
    ("freight_ratio", "freight_rate", "运费率", "上升", "ratio"),
    ("ratio_missing", "avg_order_value", "客单价", "上升", "missing"),
    ("order_conflict", "order_count", "订单量", "下降", "conflict"),
    # Registered symmetric ratio paths: isolate one factor, make both factors
    # move in opposing directions, and exercise their domain boundaries.
    ("aov_numerator_only", "avg_order_value", "客单价", "上升", "ratio"),
    ("aov_denominator_only", "avg_order_value", "客单价", "下降", "ratio"),
    ("aov_offsetting_effects", "avg_order_value", "客单价", "上升", "ratio"),
    ("freight_numerator_only", "freight_rate", "运费率", "上升", "ratio"),
    ("freight_denominator_only", "freight_rate", "运费率", "下降", "ratio"),
    ("freight_offsetting_effects", "freight_rate", "运费率", "上升", "ratio"),
    ("aov_zero_denominator", "avg_order_value", "客单价", "上升", "zero_denominator"),
    ("freight_zero_denominator", "freight_rate", "运费率", "上升", "zero_denominator"),
    ("freight_missing", "freight_rate", "运费率", "上升", "missing"),
    ("aov_conflict", "avg_order_value", "客单价", "上升", "ratio_conflict"),
    ("freight_conflict", "freight_rate", "运费率", "上升", "ratio_conflict"),
    ("aov_execution_exception", "avg_order_value", "avg_order_value", "up", "execution_exception"),
    # Registered additive dimensions: distributed changes, opposing group
    # effects, and a Top-N result whose omitted mass must remain as Other.
    ("order_dispersed", "order_count", "订单量", "下降", "dimension"),
    ("freight_dispersed", "freight_value", "运费总额", "下降", "dimension"),
    ("order_offset", "order_count", "订单量", "下降", "dimension"),
    ("freight_offset", "freight_value", "运费总额", "下降", "dimension"),
    ("order_partial_coverage", "order_count", "订单量", "下降", "dimension"),
    ("freight_partial_coverage", "freight_value", "运费总额", "下降", "dimension"),
    ("order_requested_product_category", "order_count", "order_count", "\u4e0b\u964d", "dimension"),
    ("order_unauthorized_customer_state", "order_count", "order_count", "\u4e0b\u964d", "authorization_denied_dimension"),
    ("order_unsupported_dimension", "order_count", "订单量", "下降", "unsupported_dimension"),
    ("freight_unsupported_dimension", "freight_value", "运费总额", "下降", "unsupported_dimension"),
    ("score_unsupported", "review_score_avg", "平均评分", "下降", "unsupported"),
]

# Independently hand-calculated from the seeded facts.  These values are
# intentionally literal acceptance oracles; do not replace with calls to the
# production attribution helpers.
RATIO_ORACLES = {
    "aov_ratio": {
        "baseline_value": 10.0, "current_value": 20.0, "total_delta": 10.0,
        "numerator_effect": 8 / 3, "denominator_effect": 22 / 3,
    },
    "freight_ratio": {
        "baseline_value": 10.0, "current_value": 20.0, "total_delta": 10.0,
        "numerator_effect": 8 / 3, "denominator_effect": 22 / 3,
    },
    "aov_numerator_only": {
        "baseline_value": 10.0, "current_value": 12.0, "total_delta": 2.0,
        "numerator_effect": 2.0, "denominator_effect": 0.0,
    },
    "aov_denominator_only": {
        "baseline_value": 10.0, "current_value": 100 / 12, "total_delta": -5 / 3,
        "numerator_effect": 0.0, "denominator_effect": -5 / 3,
    },
    "aov_offsetting_effects": {
        "baseline_value": 10.0, "current_value": 120 / 11, "total_delta": 120 / 11 - 10,
        "numerator_effect": 21 / 11, "denominator_effect": -1.0,
    },
    "freight_numerator_only": {
        "baseline_value": 10.0, "current_value": 20.0, "total_delta": 10.0,
        "numerator_effect": 10.0, "denominator_effect": 0.0,
    },
    "freight_denominator_only": {
        "baseline_value": 10.0, "current_value": 8.0, "total_delta": -2.0,
        "numerator_effect": 0.0, "denominator_effect": -2.0,
    },
    "freight_offsetting_effects": {
        "baseline_value": 10.0, "current_value": 1200 / 110, "total_delta": 1200 / 110 - 10,
        "numerator_effect": 21 / 11, "denominator_effect": -1.0,
    },
}

DIMENSION_ORACLES = {
    "order_dimension": {"total_delta": -4.0, "deltas": {"A": -4.0, "B": 0.0}},
    "freight_dimension": {"total_delta": -4.0, "deltas": {"A": -4.0, "B": 0.0}},
    "order_dispersed": {"total_delta": -4.0, "deltas": {"A": -2.0, "B": -2.0}},
    "freight_dispersed": {"total_delta": -4.0, "deltas": {"A": -2.0, "B": -2.0}},
    "order_offset": {"total_delta": -2.0, "deltas": {"A": -3.0, "B": 1.0}},
    "freight_offset": {"total_delta": -2.0, "deltas": {"A": -3.0, "B": 1.0}},
    "order_partial_coverage": {
        "total_delta": -12.0, "each_returned_group_delta": -1.0,
        "returned_group_count": 10, "Other": -2.0,
    },
    "freight_partial_coverage": {
        "total_delta": -12.0, "each_returned_group_delta": -1.0,
        "returned_group_count": 10, "Other": -2.0,
    },
}


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch):
    """Allow only loopback for the isolated test DB; reject external sockets."""
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def is_loopback(address):
        host = address[0] if isinstance(address, tuple) else address
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except (TypeError, ValueError):
            return False

    def blocked_connect(sock, address):
        if is_loopback(address):
            return original_connect(sock, address)
        raise AssertionError("external network is blocked in offline diagnostic acceptance")

    def blocked_create_connection(address, *args, **kwargs):
        if is_loopback(address):
            return original_create_connection(address, *args, **kwargs)
        raise AssertionError("external network is blocked in offline diagnostic acceptance")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    monkeypatch.setattr(socket, "create_connection", blocked_create_connection)


def seed_facts(engine, case_id, *, start_date=date(2024, 4, 28)):
    """One source of facts: six prior days, then current day; independent oracle."""
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE analytics.fact_orders ADD COLUMN IF NOT EXISTS customer_state text"))
        conn.execute(text("ALTER TABLE analytics.fact_orders ADD COLUMN IF NOT EXISTS customer_city text"))
        conn.execute(text("ALTER TABLE raw.orders ADD COLUMN IF NOT EXISTS customer_id text"))
        conn.execute(text("CREATE TABLE IF NOT EXISTS raw.customers (customer_id text, customer_unique_id text, customer_state text, customer_city text)"))
        conn.execute(text("CREATE TABLE IF NOT EXISTS raw.sellers (seller_id text, seller_state text)"))
        conn.execute(text("CREATE TABLE IF NOT EXISTS raw.products (product_id text, product_category_name text)"))
        conn.execute(text("CREATE TABLE IF NOT EXISTS raw.order_items (order_id text, product_id text, seller_id text, price numeric, freight_value numeric)"))
        for table in ("raw.customers", "raw.sellers", "raw.products", "raw.order_items"):
            conn.execute(text(f"DELETE FROM {table}"))
        conn.execute(text("INSERT INTO raw.sellers VALUES ('seller','SP')"))
        conn.execute(text("INSERT INTO raw.products VALUES ('A','A'),('B','B')"))
        if case_id in {"order_partial_coverage", "freight_partial_coverage"}:
            conn.execute(text("INSERT INTO raw.products SELECT 'G' || lpad(i::text,2,'0'), 'G' || lpad(i::text,2,'0') FROM generate_series(0,11) AS i"))
        for table in ("analytics.daily_metrics", "analytics.fact_orders", "raw.orders", "raw.order_reviews", "analytics.fact_order_payments", "analytics.fact_fulfillment"):
            conn.execute(text(f"DELETE FROM {table}"))
        for offset in range(7):
            day = start_date + timedelta(days=offset)
            current = offset == 6
            count = 6 if current else 10
            if case_id in {"aov_denominator_only", "freight_denominator_only", "freight_offsetting_effects", "freight_numerator_only"} and current:
                count = 12 if case_id == "aov_denominator_only" else 10
            if case_id in {"aov_numerator_only", "aov_offsetting_effects"} and current:
                count = 10 if case_id == "aov_numerator_only" else 11
            if case_id in {"order_offset", "freight_offset"} and current:
                count = 8
            if case_id in {"order_partial_coverage", "freight_partial_coverage"}:
                count = 12 if current else 24 if offset == 5 else 10
            if case_id == "aov_zero_denominator" and current:
                count = 0
            if case_id == "freight_zero_denominator" and current:
                count = 1
            if case_id == "freight_missing" and offset == 5:
                count = 0

            gross, freight = (9, 1)
            if current and case_id in {"aov_ratio", "ratio_missing", "aov_conflict", "aov_execution_exception"}:
                gross, freight = 18, 2
            elif current and case_id in {"freight_ratio", "freight_conflict"}:
                gross, freight = 8, 2
            elif current and case_id == "aov_numerator_only":
                gross, freight = 11, 1
            elif current and case_id == "aov_denominator_only":
                gross, freight = (100 / 12) - 1, 1
            elif current and case_id == "aov_offsetting_effects":
                gross, freight = (120 / 11) - 1, 1
            elif current and case_id == "freight_numerator_only":
                gross, freight = 8, 2
            elif current and case_id == "freight_denominator_only":
                gross, freight = 11.5, 1
            elif current and case_id == "freight_offsetting_effects":
                gross, freight = 9.8, 1.2
            elif current and case_id == "freight_zero_denominator":
                gross, freight = 0, 0

            for i in range(count):
                order = f"{day}-{i}"
                if case_id in {"order_partial_coverage", "freight_partial_coverage"} and (current or offset == 5):
                    group = f"G{i // 2:02d}" if not current else f"G{i:02d}"
                elif case_id in {"order_dispersed", "freight_dispersed"}:
                    split = 3 if current else 5
                    group = "A" if i < split else "B"
                elif case_id in {"order_offset", "freight_offset"}:
                    split = 5 if current else 8
                    group = "A" if i < split else "B"
                else:
                    group = "B" if i < 2 else "A"
                conn.execute(text("""INSERT INTO analytics.fact_orders
                    (order_id,customer_unique_id,item_count,item_gross_value,freight_value,purchase_date,order_status,customer_state,customer_city)
                    VALUES (:id,:id,1,:gross,:freight,:day,'delivered',:group,:group)"""),
                    dict(id=order, gross=gross, freight=freight, day=day, group=group))
                conn.execute(text("INSERT INTO raw.orders (order_id,order_purchase_timestamp,order_status,order_delivered_customer_date,order_estimated_delivery_date,customer_id) VALUES (:id,:day,'delivered',:day,:day,:id)"), dict(id=order, day=day))
                conn.execute(text("INSERT INTO raw.customers VALUES (:id,:id,:group,:group)"), dict(id=order, group=group))
                conn.execute(text("INSERT INTO raw.order_items VALUES (:id,:group,'seller',:gross,:freight)"), dict(id=order, group=group, gross=gross, freight=freight))
                conn.execute(text("INSERT INTO raw.order_reviews VALUES (:id,:score,'text')"), dict(id=order, score=2 if current else 4))
            gmv = count * (gross + freight)
            if current and case_id == "freight_conflict":
                # Deliberately inconsistent cross-evidence: trend freight_rate
                # and freight_value come from fact_orders, while registered
                # GMV support is sourced from this report snapshot.
                gmv = 120
            aov = gmv/count if count else None
            if current and case_id == "aov_conflict":
                # The target report snapshot disagrees with GMV / order_count.
                aov = 18
            conn.execute(text("INSERT INTO analytics.daily_metrics VALUES (:day,:gmv,:count,:aov)"),
                         dict(day=day, gmv=gmv, count=count + (1 if current and case_id == "order_conflict" else 0), aov=aov))
        if case_id == "ratio_missing":
            # Data absence, not zero: a missing target trend point on baseline.
            conn.execute(text("UPDATE analytics.daily_metrics SET avg_order_value=NULL WHERE metric_date='2024-05-03'"))
        if case_id == "freight_missing":
            # Missing baseline facts must remain missing, rather than being
            # represented as a numeric zero ratio.
            conn.execute(text("DELETE FROM analytics.fact_orders WHERE purchase_date='2024-05-03'"))
            conn.execute(text("UPDATE analytics.daily_metrics SET gmv=NULL,order_count=NULL,avg_order_value=NULL WHERE metric_date='2024-05-03'"))


@pytest.mark.parametrize("case_id,metric,label,direction,kind", CASES, ids=[c[0] for c in CASES])
def test_non_gmv_diagnostic_workflow(monkeypatch, isolated_trend_db, tmp_path, case_id, metric, label, direction, kind):
    from app.graph import workflow
    from app.graph.nodes import triage
    from app.analysis import attribution_methods
    from app.llm import client as llm
    from app.agents import critic_agent
    from app.mcp import client as mcp, tools
    from app.core import time_expression
    from app.data.repositories import daily_metrics
    from app.tenancy import TenantContext

    # These are SDK/MCP-stubbed offline workflow checks. Only the loopback
    # isolated PostgreSQL fixture may open a socket; accidental Provider or
    # other external traffic must fail at the test boundary.
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def is_loopback(address):
        host = address[0] if isinstance(address, tuple) else address
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except (TypeError, ValueError):
            return False

    def guarded_connect(sock, address):
        if is_loopback(address):
            return original_connect(sock, address)
        raise AssertionError("external network blocked during offline diagnostic workflow")

    def guarded_connect_ex(sock, address):
        if is_loopback(address):
            return original_connect_ex(sock, address)
        raise AssertionError("external network blocked during offline diagnostic workflow")

    def guarded_create_connection(address, *args, **kwargs):
        if is_loopback(address):
            return original_create_connection(address, *args, **kwargs)
        raise AssertionError("external network blocked during offline diagnostic workflow")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)

    engine, statements, sql_events, _ = isolated_trend_db
    seed_facts(engine, case_id)
    monkeypatch.setattr(daily_metrics, "engine", engine)
    monkeypatch.setattr(time_expression, "engine", engine)
    monkeypatch.setattr(triage, "llm_classify", lambda *a, **k: None)
    for name in ("llm_extract", "llm_chat", "llm_agent_action", "llm_diagnostic_plan"):
        if hasattr(llm, name):
            monkeypatch.setattr(llm, name, lambda *a, **k: None)
    monkeypatch.setattr(critic_agent, "llm_extract", lambda *a, **k: None)
    if case_id == "aov_execution_exception":
        def injected_calculation_failure(**kwargs):
            raise ValueError("injected unexpected ratio calculation failure")
        monkeypatch.setattr(attribution_methods, "symmetric_ratio_attribution", injected_calculation_failure)
    async def local_transport(tool_name, arguments):
        # Only replace subprocess transport. Outer gateway authorization,
        # concrete tool argument validation and repository SQL stay real.
        allowed = {"compare_breakdown_by_dimension", "compare_breakdown_by_dimension_period",
                   "compare_freight_rate_ratio_facts_by_dimension_period", "compare_avg_order_value_ratio_facts_by_dimension_period"}
        assert tool_name in allowed, f"Unexpected legacy tool {tool_name}"
        return getattr(tools, tool_name)(**arguments)
    monkeypatch.setattr(mcp, "_call_local_mcp_tool", local_transport)
    captured = {}
    monkeypatch.setattr(workflow, "write_trace", lambda tid, state: captured.update(state))
    monkeypatch.setattr(workflow, "_persist_chat_turn", lambda *a, **k: None)
    question = f"\u4e3a\u4ec0\u4e48 2024-05-04 \u7684{label}{direction}\u4e86\uff1f\u8bf7\u68c0\u67e5\u8bc1\u636e\uff0c\u533a\u5206\u7edf\u8ba1\u5206\u89e3\u4e0e\u4e1a\u52a1\u539f\u56e0\u3002"
    requested_dimension = None
    if kind == "unsupported_dimension":
        requested_dimension = "payment_type"
        if metric == "order_count":
            question = (
                "\u8bf7\u89e3\u91ca\u5e73\u53f0\u5df2\u4ea4\u4ed8\u8ba2\u5355\u91cf\u4ece 2024-05-03 \u5230 2024-05-04 \u7684\u53d8\u5316\uff0c"
                "\u5148\u6838\u5bf9\u8d8b\u52bf\uff0c\u518d\u4e25\u683c\u6309 payment_type \u5c55\u793a\u7edf\u8ba1\u8d21\u732e\uff1b"
                "\u8be5\u7ef4\u5ea6\u5982\u679c\u4e0d\u652f\u6301\uff0c\u8bf7\u660e\u786e\u8bf4\u660e\uff0c\u4e0d\u8981\u66ff\u6362\u4e3a\u5176\u4ed6\u7ef4\u5ea6\u3002"
            )
        else:
            question = (
                "\u8bf7\u89e3\u91ca\u5e73\u53f0\u8fd0\u8d39\u603b\u989d\u4ece 2024-05-03 \u5230 2024-05-04 \u7684\u53d8\u5316\uff0c"
                "\u5148\u6838\u5bf9\u8d8b\u52bf\uff0c\u518d\u4e25\u683c\u6309 payment_type \u5c55\u793a\u7edf\u8ba1\u8d21\u732e\uff1b"
                "\u8be5\u7ef4\u5ea6\u5982\u679c\u4e0d\u652f\u6301\uff0c\u8bf7\u660e\u786e\u8bf4\u660e\uff0c\u4e0d\u8981\u66ff\u6362\u4e3a\u5176\u4ed6\u7ef4\u5ea6\u3002"
            )
    elif case_id == "order_requested_product_category":
        requested_dimension = "product_category"
        question += " Please strictly use product_category."
    elif case_id == "order_unauthorized_customer_state":
        requested_dimension = "customer_state"
        question = (
            "\u8bf7\u89e3\u91ca\u5e73\u53f0\u5df2\u4ea4\u4ed8\u8ba2\u5355\u91cf\u4ece 2024-05-03 \u5230 2024-05-04 \u7684\u53d8\u5316\uff0c"
            "\u5148\u6838\u5bf9\u8d8b\u52bf\uff0c\u518d\u6309 customer_state \u7ef4\u5ea6\u5c55\u793a\u7edf\u8ba1\u8d21\u732e\uff1b"
            "\u5e76\u8bf4\u660e\u8fd9\u4e0d\u7b49\u4e8e\u4e1a\u52a1\u56e0\u679c\uff0c\u4e0d\u8981\u66ff\u6362\u4e3a\u5176\u4ed6\u7ef4\u5ea6\u3002"
        )
    start = len(statements)
    result, error = {}, None
    try:
        result = workflow.run_chat_workflow(question, tenant_context=TenantContext(
            tenant_id="platform", user_id="non-gmv-diagnostic", role="admin", scope_type="platform",
        ), memory_enabled=False, trace_id=f"non-gmv-{case_id}-{uuid4().hex}",
            _critic_llm_review_mode="critic_llm_review_off")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    investigation = captured.get("investigation_state") or {}
    evidence = investigation.get("evidence") or []
    analysis = captured.get("analysis_result") or {}
    stop = investigation.get("stop_reason") or {}
    action_results = investigation.get("action_results") or []
    initial_result = ((captured.get("tool_result") or {}).get("result") or {})
    initial_parsed = initial_result.get("parsed") or {}
    initial_request_id = next((call.get("request_id") for call in captured.get("tool_calls", [])
                               if call.get("request_id")), None)
    bound_initial_evidence = next((bundle for bundle in captured.get("evidence_bundles", [])
        if bundle.get("trusted") is True
        and (bundle.get("content") or {}).get("request_id") == initial_request_id
        and ((bundle.get("content") or {}).get("result") or {}).get("parsed", {}).get("metric_name") == metric), None)
    early_dimension_stop = kind in {"unsupported_dimension", "authorization_denied_dimension"}
    compiled_request = (captured.get("context_compilation") or {}).get("request_contract") or {}
    compiled_comparison = compiled_request.get("comparison") or {}
    early_request_matches = (
        compiled_request.get("operation_intent") == "dimensional_statistical_contribution"
        and compiled_request.get("metric_id") == metric
        and compiled_request.get("dimension") == requested_dimension
        and compiled_comparison.get("kind") == "day_pair"
        and compiled_comparison.get("baseline_date") == "2024-05-03"
        and compiled_comparison.get("current_date") == "2024-05-04"
    ) if early_dimension_stop else False
    valid_ids = {eid for action in action_results if action.get("status") == "completed"
                 and (action.get("validation_result") or {}).get("status") == "valid"
                 for eid in action.get("evidence_ids", [])}
    decompositions = [e for e in evidence if e.get("evidence_type") == "metric_decomposition" and e.get("evidence_id") in valid_ids]
    contributions = [e for e in evidence if e.get("evidence_type") == "dimension_contribution" and e.get("evidence_id") in valid_ids]
    checks = {"normal_entry": error is None and captured.get("task_type") == "anomaly_analysis",
              "target": (investigation.get("target") or {}).get("metric_id") == metric
                  or (captured.get("premise_check") or {}).get("metric_name") == metric
                  or initial_parsed.get("metric_name") == metric,
              "sql_executed": (
                  not statements[start:]
                  if early_dimension_stop
                  else bool(statements[start:])
              ),
              "authorized_platform_context": captured.get("tenant_id") == "platform"
                  and captured.get("scope_type") == "platform"
                  and (early_dimension_stop or any(item.get("metric_id") == metric for item in captured.get("authorized_metrics", []))),
              "initial_query_plan_matches_target": (
                  early_request_matches if early_dimension_stop else initial_parsed.get("metric_name") == metric
              ),
              "initial_result_bound_to_request": (
                  bool(initial_request_id and bound_initial_evidence)
                  or (early_dimension_stop and early_request_matches and not initial_request_id and not captured.get("query_results"))
              )}
    if kind == "ratio":
        expected = RATIO_ORACLES[case_id]
        d = next((e for e in decompositions if e.get("method_id") == "symmetric_ratio_v1"), {})
        values = d.get("values") or {}
        checks["registered_decomposition"] = bool(d) and all(
            abs(float(values.get(key, -999)) - expected[key]) < .01
            for key in ("baseline_value", "current_value", "total_delta")
        )
        component_values = {
            str(component.get("component_id")): float(component.get("value", -999))
            for component in values.get("display_components") or []
        }
        checks["component_values"] = all(
            abs(component_values.get(key, -999) - expected[key]) < .01
            for key in ("numerator_effect", "denominator_effect")
        )
        checks["bounded_finish"] = stop.get("code") == "ENOUGH_EVIDENCE" and not analysis.get("abstain")
        checks["causal_limit"] = "业务原因" in analysis.get("summary", "")
        checks["clean"] = not captured.get("errors")
        checks["statistical_decomposition_not_business_cause"] = (
            bool(d) and analysis.get("causal_readiness", {}).get("status") == "not_ready"
        )
    elif kind == "dimension":
        checks["statistical_contribution"] = bool(contributions) and bool(analysis.get("contribution"))
        checks["bounded_abstention"] = analysis.get("abstain") is True and stop.get("code") != "EXECUTION_FAILED"
        checks["clean"] = not captured.get("errors")
        if case_id in DIMENSION_ORACLES:
            expected = DIMENSION_ORACLES[case_id]
            contribution = next((item for item in analysis.get("contribution", [])
                                 if item.get("metric_name") == metric), {})
            actual_items = {str(item.get("dimension_value")): float(item.get("delta_value", -999))
                            for item in contribution.get("items", [])}
            checks["hand_calculated_total_delta"] = abs(float(contribution.get("total_delta", -999)) - expected["total_delta"]) < .01
            if case_id in {"order_partial_coverage", "freight_partial_coverage"}:
                reported_groups = {key: value for key, value in actual_items.items() if key != "Other"}
                checks["hand_calculated_group_deltas"] = (
                    len(reported_groups) == expected["returned_group_count"]
                    and all(key.startswith("G") and abs(value - expected["each_returned_group_delta"]) < .01 for key, value in reported_groups.items())
                    and abs(actual_items.get("Other", -999) - expected["Other"]) < .01
                )
            else:
                checks["hand_calculated_group_deltas"] = all(
                    abs(actual_items.get(group, -999) - delta) < .01
                    for group, delta in expected["deltas"].items()
                )
            checks["statistical_not_causal"] = (
                analysis.get("causal_readiness", {}).get("status") == "not_ready"
                and analysis.get("abstain") is True
            )
    elif kind == "zero_denominator":
        checks["no_numeric_decomposition"] = not decompositions
        checks["safe_terminal_limit"] = (
            stop.get("code") in {"DATA_INSUFFICIENT", "INVALID_DATA"}
            or (captured.get("premise_check") or {}).get("status") == "unavailable"
        )
        checks["zero_is_not_reported_as_zero_ratio"] = not any(
            e.get("evidence_type") == "metric_decomposition" and
            (e.get("values") or {}).get("total_delta") is not None
            for e in evidence
        )
        checks["not_execution_failure"] = stop.get("code") != "EXECUTION_FAILED"
        checks["no_root_cause_claim"] = not analysis or not analysis.get("contribution")
    elif kind == "ratio_conflict":
        checks["conflict_rejected"] = stop.get("code") == "INVALID_DATA" and analysis.get("abstain") is True
        checks["no_numeric_decomposition"] = not decompositions
        checks["no_conflicting_contribution_claim"] = not analysis.get("contribution")
        failures = captured.get("diagnostic_data_reconciliation_failures", [])
        failure = next((item for item in failures if item.get("metric_id") == metric), {})
        checks["reconciliation_failure_recorded"] = (
            failure.get("code") == "DATA_RECONCILIATION_FAILED"
            and "Registered ratio factors disagree with the target trend." in str(failure.get("detail") or "")
        )
        checks["conflicting_input_evidence_preserved"] = bool(failure.get("input_evidence_ids")) and all(
            evidence_id in {
                item.get("evidence_id") for item in captured.get("evidence_bundles", [])
            }
            for evidence_id in failure.get("input_evidence_ids", [])
        )
        required_inputs = (
            {"avg_order_value", "gmv", "order_count"}
            if case_id == "aov_conflict"
            else {"freight_rate", "freight_value", "gmv"}
        )
        checks["target_and_factor_inputs_identified"] = required_inputs <= set(failure.get("input_metric_ids", []))
        checks["not_misclassified_as_execution_failure"] = (
            stop.get("code") == "INVALID_DATA"
            and not any(e.get("code") == "diagnostic_decomposition_execution_failed" for e in captured.get("errors", []))
        )
        checks["final_output_states_data_conflict"] = (
            "\u8bc1\u636e\u6570\u636e\u4e0d\u4e00\u81f4" in str(result.get("answer") or "")
        )
        checks["conflict_is_presented_as_partial"] = (
            (result.get("presentation") or {}).get("status") == "partial"
            and (captured.get("analysis_presentation") or {}).get("status") == "partial"
        )
        checks["no_spurious_reference_error"] = not any(
            e.get("code") == "diagnostic_evidence_reference_invalid" for e in captured.get("errors", []))
    elif kind == "execution_exception":
        checks["unexpected_calculation_error_stays_execution_failed"] = stop.get("code") == "EXECUTION_FAILED"
        checks["unexpected_error_is_recorded"] = any(
            e.get("code") == "diagnostic_decomposition_execution_failed"
            and "injected unexpected ratio calculation failure" in str(e.get("message") or "")
            for e in captured.get("errors", [])
        )
        checks["not_relabelled_as_data_conflict"] = not captured.get("diagnostic_data_reconciliation_failures")
        checks["no_spurious_reference_error"] = not any(
            e.get("code") == "diagnostic_evidence_reference_invalid" for e in captured.get("errors", []))
    elif kind == "conflict":
        checks["conflict_rejected"] = stop.get("code") == "INVALID_DATA" and analysis.get("abstain") is True
        checks["no_decomposition_finish"] = not decompositions
        checks["no_conflicting_contribution_claim"] = not analysis.get("contribution")
        checks["actual_reconciliation_conflict"] = any(
            "无法对账" in str(a.get("failure_detail") or "") for a in action_results)
        checks["no_spurious_reference_error"] = not any(
            e.get("code") == "diagnostic_evidence_reference_invalid" for e in captured.get("errors", []))
    else:
        checks["safe_abstention"] = analysis.get("abstain") is True and stop.get("code") != "EXECUTION_FAILED"
        checks["no_decomposition"] = not decompositions
        checks["clean"] = not captured.get("errors")
        if kind == "missing" and (captured.get("premise_check") or {}).get("status") == "unavailable":
            checks["target"] = (captured.get("metric_query_batch_plan") or {}).get("plans", [{}])[0].get("metric_name") == metric
            checks["safe_abstention"] = not analysis and not decompositions and "不会继续" in result.get("answer", "")
        if kind == "unsupported":
            checks["capability_limit_recorded"] = bool(captured.get("diagnostic_capability_gaps"))
            checks["no_invented_contribution"] = not analysis.get("contribution") and "validated dimension contribution" not in analysis.get("summary", "")
        if kind == "unsupported_dimension":
            queried_dimensions = {
                str((e.get("values") or {}).get("query", {}).get("dimension"))
                for e in evidence if e.get("evidence_type") == "dimension_contribution"
            }
            checks["requested_dimension_not_silently_replaced"] = (
                not queried_dimensions and stop.get("code") == "CAPABILITY_GAP"
            )
            checks["capability_limit_recorded"] = bool(captured.get("diagnostic_capability_gaps")) or stop.get("code") == "CAPABILITY_GAP"
            checks["no_invented_supported_dimension_result"] = not analysis.get("contribution")
            checks["not_execution_failure"] = stop.get("code") != "EXECUTION_FAILED"
            checks["limitation_is_terminal_and_partial"] = (
                stop.get("code") in {"CAPABILITY_GAP", "AUTHORIZATION_BLOCKED"}
                and analysis.get("abstain") is True
                and (result.get("presentation") or {}).get("status") == "partial"
            )
            checks["capability_limit_does_not_consume_action_budget"] = (
                int(investigation.get("remaining_budget", 0)) > 0
                and int(captured.get("runtime_steps", 0))
                < int((captured.get("runtime_budget") or {}).get("max_graph_steps", 0))
            )
            checks["answer_explains_requested_dimension_limit"] = (
                "\u652f\u4ed8\u65b9\u5f0f" in str(result.get("answer") or "")
                and "\u672a\u66ff\u6362" in str(result.get("answer") or "")
            )
        if kind == "authorization_denied_dimension":
            gaps = captured.get("diagnostic_capability_gaps") or []
            dimension_tool_calls = [
                call for call in captured.get("tool_calls", [])
                if call.get("tool_name") in {
                    "compare_breakdown_by_dimension",
                    "compare_breakdown_by_dimension_period",
                }
            ]
            checks["registered_but_identity_denied"] = (
                bool(gaps)
                and gaps[0].get("code") == "AUTHORIZATION_BLOCKED"
                and gaps[0].get("requested_dimension") == "customer_state"
                and stop.get("code") == "AUTHORIZATION_BLOCKED"
            )
            checks["initial_query_refused_before_tool_execution"] = (
                early_request_matches
                and not captured.get("query_results")
                and not captured.get("tool_calls")
                and not dimension_tool_calls
            )
            checks["denied_plan_keeps_requested_metric_dimension_and_dates"] = (
                early_request_matches
            )
            checks["no_requested_or_alternative_contribution"] = (
                not contributions and not analysis.get("contribution") and not dimension_tool_calls
            )
            checks["no_dimension_sql"] = not any(
                "compare_breakdown_by_dimension" in str(item.get("statement") or "")
                for item in sql_events[start:]
            )
            checks["policy_reports_only_authorized_alternative"] = (
                bool(gaps) and gaps[0].get("supported_dimensions") == ["product_category"]
            )
            checks["no_query_sql_was_needed_for_early_denial"] = not statements[start:]
    if kind in {"ratio", "dimension", "unsupported"}:
        checks["critic_passed"] = captured.get("critic_verdict") == "approved"
        checks["trajectory_passed"] = (captured.get("critic_result") or {}).get("diagnostic_review", {}).get("status") == "approved"
        checks["optional_llm_reviews_are_explicitly_ablated"] = (
            (captured.get("critic_result") or {}).get("llm_review", {}).get("status") == "evaluation_ablation"
        )
    if case_id == "order_dimension":
        queried_dimensions = {
            str((e.get("values") or {}).get("query", {}).get("dimension"))
            for e in evidence if e.get("evidence_type") == "dimension_contribution"
        }
        checks["unspecified_dimension_keeps_registered_default"] = (
            not captured.get("requested_dimensions") and queried_dimensions == {"product_category"}
        )
    if case_id == "order_requested_product_category":
        queried_dimensions = {
            str((e.get("values") or {}).get("query", {}).get("dimension"))
            for e in evidence if e.get("evidence_type") == "dimension_contribution"
        }
        contribution_calls = [
            call for call in captured.get("tool_calls", [])
            if call.get("tool_name") == "compare_breakdown_by_dimension"
        ]
        contribution_evidence = next((
            item for item in contributions
            if item.get("metric_id") == metric
            and (item.get("values") or {}).get("query", {}).get("dimension") == "product_category"
        ), {})
        contribution_values = contribution_evidence.get("values") or {}
        contribution_result = contribution_values.get("result") or {}
        checks["explicit_supported_dimension_is_preserved"] = (
            captured.get("requested_dimensions") == ["product_category"]
            and queried_dimensions == {"product_category"}
        )
        checks["no_alternative_dimension_query"] = queried_dimensions <= {"product_category"}
        checks["supported_contribution_keeps_date_roles"] = bool(contribution_calls) and all(
            (call.get("arguments") or {}).get("comparison", {}).get("current_date") == "2024-05-04"
            and (call.get("arguments") or {}).get("comparison", {}).get("baseline_date") == "2024-05-03"
            for call in contribution_calls
        )
        checks["supported_contribution_evidence_is_valid_and_bound"] = (
            bool(contribution_evidence)
            and contribution_evidence.get("evidence_id") in valid_ids
            and contribution_values.get("attribution_status") == "ok"
            and contribution_result.get("status") == "ok"
            and contribution_result.get("current_date") == "2024-05-04"
            and contribution_result.get("baseline_date") == "2024-05-03"
        )
    if kind == "unsupported_dimension":
        checks["critic_passed"] = captured.get("critic_verdict") == "approved"
        checks["trajectory_passed"] = (captured.get("critic_result") or {}).get("diagnostic_review", {}).get("status") == "approved"
        checks["critic_records_limited_outcome"] = (
            (captured.get("critic_result") or {}).get("diagnostic_review", {}).get("outcome")
            == "capability_limited"
        )
    zero_value_trace = None
    if kind == "zero_denominator":
        # Independently verify the test facts, then ask the normal business
        # entry for the value. This reaches the real registry query path and
        # verifies the system does not turn an undefined ratio into numeric 0.
        with engine.connect() as fact_conn:
            fact_row = fact_conn.execute(text("""
                SELECT COUNT(*) AS order_count,
                       COALESCE(SUM(item_gross_value + freight_value), 0) AS gmv,
                       COALESCE(SUM(freight_value), 0) AS freight_value
                FROM analytics.fact_orders WHERE purchase_date='2024-05-04'
            """)).mappings().one()
        if metric == "avg_order_value":
            fixture_zero = int(fact_row["order_count"]) == 0 and float(fact_row["gmv"]) == 0.0
        else:
            fixture_zero = int(fact_row["order_count"]) == 1 and float(fact_row["gmv"]) == 0.0 and float(fact_row["freight_value"]) == 0.0
        checks["zero_denominator_fixture_verified"] = fixture_zero
        direct_capture = {}
        monkeypatch.setattr(workflow, "write_trace", lambda tid, state: direct_capture.update(state))
        direct_start = len(statements)
        direct_question = f"请查询 2024-05-04 的{label}（{metric}）指标值，并说明数据缺失时不要把结果当作 0。"
        direct_result, direct_error = {}, None
        try:
            direct_result = workflow.run_chat_workflow(direct_question, tenant_context=TenantContext(
                tenant_id="platform", user_id="non-gmv-diagnostic", role="admin", scope_type="platform",
            ), memory_enabled=False, trace_id=f"non-gmv-{case_id}-value-{uuid4().hex}")
        except Exception as exc:
            direct_error = f"{type(exc).__name__}: {exc}"
        direct_tool_result = ((direct_capture.get("tool_result") or {}).get("result") or {})
        direct_metric_result = direct_tool_result.get("metric_result") or {}
        direct_parsed = direct_tool_result.get("parsed") or {}
        direct_request_id = next((call.get("request_id") for call in direct_capture.get("tool_calls", [])
                                  if call.get("request_id")), None)
        direct_evidence = next((bundle for bundle in direct_capture.get("evidence_bundles", [])
            if bundle.get("trusted") is True
            and (bundle.get("content") or {}).get("request_id") == direct_request_id
            and ((bundle.get("content") or {}).get("result") or {}).get("parsed", {}).get("metric_name") == metric), None)
        direct_facts = ((direct_result or {}).get("presentation") or {}).get("facts") or []
        zero_value_trace = {
            "question": direct_question, "workflow_state": direct_capture,
            "workflow_result": direct_result, "sql_events": sql_events[direct_start:],
            "fixture_fact_oracle": dict(fact_row), "error": direct_error,
        }
        checks["zero_value_workflow_normal_entry"] = direct_error is None and direct_capture.get("task_type") == "metric_query"
        checks["zero_value_workflow_selected_metric_date"] = (
            direct_parsed.get("metric_name") == metric
            and str(direct_metric_result.get("metric_date")) == "2024-05-04"
        )
        checks["zero_value_registry_result_is_not_found"] = (
            direct_metric_result.get("status") == "not_found" and direct_metric_result.get("value") is None
        )
        checks["zero_value_query_evidence_bound"] = bool(direct_request_id and direct_evidence)
        checks["zero_value_final_facts_do_not_claim_numeric_zero"] = not any(
            fact.get("metric_name") == metric and fact.get("value") == 0 for fact in direct_facts
        )
        direct_data_review = (direct_capture.get("critic_result") or {}).get("data_review") or {}
        direct_items = direct_data_review.get("items") or []
        direct_item = direct_items[0] if direct_items else {}
        checks["zero_value_critic_reviewed_as_terminal_unavailable"] = (
            direct_capture.get("critic_verdict") == "rejected"
            and direct_data_review.get("status") == "partial"
            and direct_data_review.get("batch_status") == "partial_unavailable"
            and direct_item.get("status") == "unavailable"
            and all((direct_item.get("binding_checks") or {}).values())
            and direct_capture.get("critic_revision_target") is None
        )
        checks["zero_value_not_retried_or_replanned"] = (
            len([call for call in direct_capture.get("tool_calls", []) if call.get("tool_name") == "registry_value"]) == 1
            and int((direct_capture.get("_revision_counts") or {}).get("data_query", 0)) == 1
            and not any(error.get("code") == "critic_revision_limit_exceeded"
                        for error in direct_capture.get("errors", []))
            and direct_capture.get("critic_revision_target") is None
        )
        checks["zero_value_final_status_is_partial"] = (
            ((direct_result or {}).get("presentation") or {}).get("status") == "partial"
            and ((direct_capture.get("analysis_presentation") or {}).get("status") == "partial")
        )
        checks["zero_value_final_answer_explains_unavailable"] = (
            "\u6682\u65e0\u6570\u636e" in str((direct_result or {}).get("answer") or "")
            and not any(fact.get("metric_name") == metric and fact.get("value") == 0 for fact in direct_facts)
        )
        checks["zero_value_one_metric_sql_attempt"] = sum(
            1 for item in zero_value_trace["sql_events"]
            if "FROM analytics." in item.get("statement", "")
            and str(item.get("parameters", {}).get("metric_date")) == "2024-05-04"
        ) == 1

        # A valid numeric zero and a normal nonzero remain successful values.
        # The zero case's daily order_count is a real 0 row, while the prior
        # day's GMV is positive in the same isolated fixture.
        control_traces = {}

        def run_value_control(control_metric, control_date, control_id, inject_sql_error=False):
            control_capture = {}
            monkeypatch.setattr(workflow, "write_trace", lambda tid, state: control_capture.update(state))
            sql_start = len(sql_events)
            listener = None
            if inject_sql_error:
                def listener(conn, cursor, statement, parameters, context, executemany):
                    params = parameters if isinstance(parameters, dict) else {}
                    if (
                        "FROM analytics.daily_metrics" in statement
                        and control_metric in statement
                        and str(params.get("metric_date")) == control_date
                    ):
                        raise OperationalError(statement, parameters, Exception("injected SQL execution error"))
                event.listen(engine, "before_cursor_execute", listener)
            control_result, control_error = {}, None
            try:
                control_question = (
                    f"\u8bf7\u67e5\u8be2 {control_date} \u7684 {control_metric} \u6307\u6807\u503c\u3002"
                )
                control_result = workflow.run_chat_workflow(
                    control_question,
                    tenant_context=TenantContext(
                        tenant_id="platform", user_id="non-gmv-diagnostic", role="admin", scope_type="platform",
                    ),
                    memory_enabled=False,
                    trace_id=f"non-gmv-{case_id}-{control_id}-{uuid4().hex}",
                )
            except Exception as exc:
                control_error = f"{type(exc).__name__}: {exc}"
            finally:
                if listener is not None:
                    event.remove(engine, "before_cursor_execute", listener)
            trace = {
                "workflow_state": control_capture,
                "workflow_result": control_result,
                "sql_events": sql_events[sql_start:],
                "error": control_error,
            }
            control_traces[control_id] = trace
            return trace

        if case_id == "aov_zero_denominator":
            zero_control = run_value_control("order_count", "2024-05-04", "valid_numeric_zero")
            zero_state = zero_control["workflow_state"]
            zero_result = (((zero_state.get("tool_result") or {}).get("result") or {}).get("metric_result") or {})
            zero_presentation = (zero_control["workflow_result"].get("presentation") or {})
            checks["valid_numeric_zero_stays_success"] = (
                zero_control["error"] is None
                and zero_result.get("status") == "ok" and zero_result.get("value") == 0
                and zero_state.get("critic_verdict") == "approved"
                and (zero_state.get("critic_result") or {}).get("data_review", {}).get("status") == "approved"
                and zero_presentation.get("status") == "completed"
            )

            positive_control = run_value_control("gmv", "2024-05-03", "valid_nonzero")
            positive_state = positive_control["workflow_state"]
            positive_result = (((positive_state.get("tool_result") or {}).get("result") or {}).get("metric_result") or {})
            checks["valid_nonzero_stays_success"] = (
                positive_control["error"] is None
                and positive_result.get("status") == "ok" and float(positive_result.get("value") or 0) > 0
                and positive_state.get("critic_verdict") == "approved"
                and (positive_state.get("critic_result") or {}).get("data_review", {}).get("status") == "approved"
            )

            exception_control = run_value_control("gmv", "2024-05-03", "query_exception", inject_sql_error=True)
            exception_state = exception_control["workflow_state"]
            exception_review = ((exception_state.get("critic_result") or {}).get("data_review") or {})
            exception_item = (exception_review.get("items") or [{}])[0]
            exception_calls = exception_state.get("tool_calls", [])
            checks["real_query_exception_keeps_execution_failure"] = (
                exception_control["error"] is None
                and ((exception_state.get("tool_result") or {}).get("result") or {}).get("status") == "execution_failed"
                and exception_item.get("status") == "execution_failed"
                and (((exception_state.get("tool_result") or {}).get("result") or {}).get("failure_context") or {}).get("category") == "database_error"
                and exception_state.get("metric_batch_execution", {}).get("status") != "partial_unavailable"
                and exception_item.get("status") != "unavailable"
            )
            checks["real_query_exception_not_relabelled_not_found"] = (
                exception_review.get("status") != "partial"
                and not any(item.get("status") == "unavailable" for item in exception_review.get("items", []))
                and ((exception_state.get("tool_result") or {}).get("result") or {}).get("status") != "not_found"
            )
            zero_value_trace["controls"] = control_traces
    signatures = [(c.get("tool_name"), json.dumps(c.get("arguments") or {}, sort_keys=True, default=str))
                  for c in captured.get("tool_calls", [])]
    checks["no_duplicate_tool_queries"] = len(signatures) == len(set(signatures))
    oracle = RATIO_ORACLES.get(case_id) or DIMENSION_ORACLES.get(case_id) or {
        "ratio_missing": {"baseline": "NULL (missing target fact)", "expected_outcome": "reasonable stop; no ratio decomposition"},
        "freight_missing": {"baseline": "no delivered-order facts; denominator unavailable", "expected_outcome": "reasonable stop; no ratio decomposition"},
        "aov_zero_denominator": {"current_order_count": 0, "current_gmv": 0, "expected_outcome": "reasonable stop; do not report a numeric AOV ratio"},
        "freight_zero_denominator": {"current_freight_value": 0, "current_gmv": 0, "expected_outcome": "reasonable stop; do not report a numeric freight rate"},
        "aov_conflict": {"current_gmv": 120, "current_order_count": 6, "fact_ratio": 20, "reported_target_ratio": 18, "expected_outcome": "reject inconsistent evidence"},
        "aov_execution_exception": {"same_facts_as": "aov_ratio", "expected_outcome": "EXECUTION_FAILED for injected calculation exception"},
        "freight_conflict": {"current_freight_value": 12, "fact_gmv": 60, "snapshot_gmv": 120, "fact_rate": 20, "support_ratio": 10, "expected_outcome": "reject inconsistent evidence"},
        "order_unsupported_dimension": {"requested_dimension": requested_dimension, "expected_outcome": "CAPABILITY_GAP; do not substitute another dimension"},
        "order_unauthorized_customer_state": {"requested_dimension": requested_dimension, "expected_outcome": "AUTHORIZATION_BLOCKED before contribution query; do not substitute another dimension"},
        "freight_unsupported_dimension": {"requested_dimension": requested_dimension, "expected_outcome": "CAPABILITY_GAP; do not substitute another dimension"},
    }.get(case_id)
    if decompositions:
        actual_outcome = "numeric_decomposition"
    elif contributions:
        actual_outcome = "statistical_contribution"
    elif stop.get("code") == "EXECUTION_FAILED" or error:
        actual_outcome = "execution_failure"
    else:
        actual_outcome = "reasonable_stop"
    report = dict(case_id=case_id, metric=metric, kind=kind, question=question,
                  expected_outcome=oracle, checks=checks,
                  actual_outcome=actual_outcome, passed=all(checks.values()), error=error, workflow_state=captured,
                  workflow_result=result, sql_events=sql_events[start:], zero_value_trace=zero_value_trace)
    directory = tmp_path / "non_gmv_diagnostic_traces"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{case_id}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    assert all(checks.values()), {"case": case_id, "error": error, "checks": checks, "stop": stop, "analysis": analysis}
