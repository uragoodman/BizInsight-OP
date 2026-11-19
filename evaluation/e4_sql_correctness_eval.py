# -*- coding: utf-8 -*-
"""E4.0 SQL/data-correctness seed evaluation.

Runs the active planner plus the governed query gateway against a disposable
PostgreSQL fixture. The fixture has one deterministic snapshot; gold values are
frozen in ``evaluation/datasets/e4_sql_cases_v1.jsonl``.

The script creates and drops a temporary database derived from
``ISOLATED_TEST_POSTGRES_ADMIN_URL`` or the configured PostgreSQL admin URL.
It is opt-in and never writes to the application database.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

from app.agents.data_query_agent import plan_confirmed_metric_queries
from app.context.compiler import compile_turn_context
from app.core.config import get_settings
from app.core.time_expression import resolve_expression as _resolve_expression
from app.mcp.client import query_business_data
from app.tenancy.context import TenantContext, activate_tenant_context

_TENANT_ID = "tenant-e4"
_USER_ID = "user-e4"

_SCHEMA_STATEMENTS = (
    "CREATE SCHEMA analytics",
    "CREATE SCHEMA raw",
    """
    CREATE TABLE analytics.daily_metrics (
        metric_date date PRIMARY KEY,
        gmv numeric(12, 2) NOT NULL,
        order_count integer NOT NULL,
        avg_order_value numeric(12, 2) NOT NULL
    )
    """,
    """
    CREATE TABLE analytics.fact_orders (
        order_id text PRIMARY KEY,
        customer_unique_id text,
        purchase_date date NOT NULL,
        order_status text NOT NULL DEFAULT 'delivered',
        item_gross_value numeric(12, 2) NOT NULL DEFAULT 0,
        freight_value numeric(12, 2) NOT NULL DEFAULT 0,
        item_count integer NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE TABLE raw.orders (
        order_id text PRIMARY KEY,
        customer_id text,
        order_purchase_timestamp timestamp NOT NULL,
        order_delivered_customer_date timestamp,
        order_estimated_delivery_date timestamp,
        order_status text NOT NULL
    )
    """,
    """
    CREATE TABLE raw.order_items (
        order_id text NOT NULL,
        product_id text NOT NULL,
        seller_id text,
        price numeric(12, 2) NOT NULL,
        freight_value numeric(12, 2) NOT NULL
    )
    """,
    "CREATE TABLE raw.customers (customer_id text PRIMARY KEY, customer_unique_id text)",
    "CREATE TABLE raw.sellers (seller_id text PRIMARY KEY, seller_state text)",
    "CREATE TABLE raw.products (product_id text PRIMARY KEY, product_category_name text)",
    "CREATE TABLE raw.order_reviews (order_id text, review_score integer)",
)

_SEED_STATEMENTS = (
    """
    INSERT INTO analytics.daily_metrics VALUES
        ('2024-01-09', 300.00, 2, 150.00),
        ('2024-01-10', 385.00, 3, 128.33),
        ('2024-01-11', 420.00, 3, 140.00)
    """,
    """
    INSERT INTO analytics.fact_orders
        (order_id, customer_unique_id, purchase_date, order_status,
         item_gross_value, freight_value, item_count)
    VALUES
        ('o1', 'customer-1', '2024-01-10', 'delivered', 110, 10, 1),
        ('o2', 'customer-2', '2024-01-10', 'delivered', 55, 5, 1),
        ('o3', 'customer-3', '2024-01-10', 'delivered', 220, 20, 1)
    """,
    """
    INSERT INTO raw.orders VALUES
        ('o1', 'c1', '2024-01-10 09:00:00', '2024-01-11 09:00:00', '2024-01-12 09:00:00', 'delivered'),
        ('o2', 'c2', '2024-01-10 10:00:00', '2024-01-11 10:00:00', '2024-01-12 10:00:00', 'delivered'),
        ('o3', 'c3', '2024-01-10 11:00:00', '2024-01-12 11:00:00', '2024-01-12 12:00:00', 'delivered')
    """,
    """
    INSERT INTO raw.order_items VALUES
        ('o1', 'p1', 's1', 100, 10),
        ('o2', 'p2', 's2', 50, 5),
        ('o3', 'p3', 's3', 200, 20)
    """,
    "INSERT INTO raw.customers VALUES ('c1', 'customer-1'), ('c2', 'customer-2'), ('c3', 'customer-3')",
    "INSERT INTO raw.sellers VALUES ('s1', 'SP'), ('s2', 'RJ'), ('s3', 'MG')",
    "INSERT INTO raw.products VALUES ('p1', 'books'), ('p2', 'books'), ('p3', 'toys')",
)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _admin_url(override: str | None):
    if override:
        return make_url(override).set(database="postgres")
    configured = os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL")
    if configured:
        return make_url(configured).set(database="postgres")
    return make_url(get_settings().postgres_url).set(database="postgres")


@contextmanager
def _disposable_database(
    admin_url,
    *,
    schema_statements: tuple[str, ...] | None = None,
    seed_statements: tuple[str, ...] | None = None,
) -> Iterator[Engine]:
    database_name = f"e4_eval_{uuid4().hex[:12]}"
    admin_engine = create_engine(
        admin_url,
        isolation_level="AUTOCOMMIT",
        connect_args={"connect_timeout": 5},
    )
    test_engine: Engine | None = None
    try:
        with admin_engine.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{database_name}"'))
        test_engine = create_engine(
            admin_url.set(database=database_name),
            connect_args={"connect_timeout": 5},
        )
        schema = _SCHEMA_STATEMENTS if schema_statements is None else schema_statements
        seed = _SEED_STATEMENTS if seed_statements is None else seed_statements
        with test_engine.begin() as connection:
            for statement in (*schema, *seed):
                connection.execute(text(statement))
        yield test_engine
    finally:
        if test_engine is not None:
            test_engine.dispose()
        with admin_engine.connect() as connection:
            connection.execute(
                text(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)')
            )
        admin_engine.dispose()


@contextmanager
def _fixture_runtime(
    test_engine: Engine,
    *,
    data_bounds: tuple[date, date] = (date(2024, 1, 9), date(2024, 1, 11)),
) -> Iterator[None]:
    fixture_bounds = data_bounds

    def _fake_resolve(question: str, *, data_bounds=None):
        return _resolve_expression(
            question,
            data_bounds=data_bounds or fixture_bounds,
        )

    # Repository modules import ``engine`` by value; patch every active alias.
    with patch("app.data.database.engine", test_engine), \
            patch("app.data.report_snapshot.engine", test_engine), \
            patch("app.data.repositories.daily_metrics.engine", test_engine), \
            patch("app.data.repositories.explorer.engine", test_engine), \
            patch("app.llm.client.llm_extract", lambda *_args, **_kwargs: None), \
            patch("app.agents.data_query_agent.resolve_expression", _fake_resolve), \
            patch("app.mcp.client.resolve_expression", _fake_resolve), \
            patch(
                "app.auth.context.build_authorization_context",
                lambda *_args, **_kwargs: SimpleNamespace(
                    authz_digest="e4-fixture-authz",
                    permission_version="permission-e4",
                    policy_set_version="policy-e4",
                    authz_ref="e4-fixture-authz-ref",
                    require_action=lambda *_a, **_k: None,
                    require_metrics=lambda *_a, **_k: None,
                    require_dimensions=lambda *_a, **_k: None,
                    require_tool=lambda *_a, **_k: None,
                ),
            ):
        yield


def _scope_data(case: dict[str, Any]) -> dict[str, Any]:
    # ``query_business_data`` reconstructs this payload without allowed_metrics;
    # keep the planner scope minimal so AuthorizedMetric fingerprints match.
    return {
        "tenant_id": _TENANT_ID,
        "user_id": _USER_ID,
        "permissions": "admin",
        "scope_type": "platform",
    }


def _plan(case: dict[str, Any]):
    question = str(case["question"])
    state = compile_turn_context({
        "user_message": question,
        "current_query_scope_fingerprint": "e4-fixture-scope",
        "scope_type": "platform",
        "node_history": [],
    })
    compilation = state["context_compilation"]
    return plan_confirmed_metric_queries(
        question,
        scope="platform",
        allowed_metrics=set(case["allowed_metrics"]),
        scope_data=_scope_data(case),
        operation_intent=compilation["operation_intent"],
        request_contract=compilation["request_contract"],
    )


def _execute(plan, question: str) -> dict[str, Any]:
    context = TenantContext(
        tenant_id=_TENANT_ID,
        user_id=_USER_ID,
        role="admin",
        scope_type="platform",
    )
    with activate_tenant_context(context):
        return query_business_data(
            "metric_query",
            question,
            tenant_id=_TENANT_ID,
            role="admin",
            plan=plan,
        )


def _close(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _close(a, e, tolerance) for a, e in zip(actual, expected)
        )
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _close(actual[key], value, tolerance)
            for key, value in expected.items()
        )
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(expected)) <= tolerance
    return actual == expected


def _actual_result_value(plan, result: dict[str, Any]) -> Any:
    for key in ("metric_result", "trend_result", "compare_result", "breakdown_result"):
        if isinstance(result.get(key), dict):
            payload = result[key]
            if plan.query_type == "metric_breakdown" and isinstance(payload.get("items"), list):
                payload = {
                    **payload,
                    "items": {
                        str(item.get("dimension_value")): item.get("value")
                        for item in payload["items"]
                    },
                }
            return payload
    return result


def _check_plan(plan, expected: dict[str, Any]) -> bool:
    fields = (
        "query_type", "metric_name", "metric_date", "start_date", "end_date",
        "baseline_date", "current_date", "dimension",
    )
    return all(
        getattr(plan, field, None) == expected.get(field)
        for field in fields
        if field in expected
    )


def _actual_behavior(decision_status: str, reason_code: str | None,
                     unavailable_metrics: list[str], execution_status: str | None) -> str:
    if decision_status == "confirmed":
        if execution_status == "ok":
            return "result"
        if execution_status == "not_found":
            return "empty"
        if execution_status == "unsupported":
            return "unsupported"
        if execution_status == "execution_failed":
            return "execution_error"
        return "no_result"
    if reason_code in {"unsupported_metric", "scope_rejected"} or unavailable_metrics:
        return "unauthorized"
    if decision_status == "clarification_required":
        return "clarify"
    if decision_status == "unsupported":
        return "refuse"
    return decision_status


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    batch = _plan(case)
    decision = batch.decision
    expected = case["expected"]
    details: dict[str, Any] = {
        "case_id": case["case_id"],
        "expected_behavior": case["expected_behavior"],
        "decision_status": decision.status,
        "reason_code": decision.reason_code,
        "unavailable_metrics": decision.unavailable_metrics,
        "plans": [],
        "response": None,
        "execution_status": None,
        "actual_behavior": None,
        "checks": {},
    }
    if batch.decision.status == "confirmed" and batch.plans:
        plan = batch.plans[0]
        details["plans"] = [{
            "query_type": plan.query_type,
            "metric_name": plan.metric_name,
            "metric_date": plan.metric_date,
            "start_date": plan.start_date,
            "end_date": plan.end_date,
            "baseline_date": plan.baseline_date,
            "current_date": plan.current_date,
            "dimension": plan.dimension,
        }]
        response = _execute(plan, str(case["question"]))
        result = response.get("result") if isinstance(response.get("result"), dict) else {}
        actual_payload = _actual_result_value(plan, result)
        details["response"] = {
            "server": response.get("server"),
            "tool_name": response.get("tool_name"),
            "status": result.get("status"),
            "result": actual_payload,
        }
        details["execution_status"] = result.get("status")
        details["actual_behavior"] = _actual_behavior(
            decision.status, decision.reason_code, decision.unavailable_metrics,
            details["execution_status"],
        )
        details["checks"]["plan"] = _check_plan(plan, expected)
        details["checks"]["execution"] = result.get("status") in {"ok", "not_found"}
        if case["expected_behavior"] == "result":
            expected_result = {
                key: value
                for key, value in expected.items()
                if key not in {"decision_status", "query_type"}
            }
            details["checks"]["result"] = _close(
                actual_payload, expected_result, float(case.get("tolerance", 0.01)),
            ) and result.get("status") == "ok"
        elif case["expected_behavior"] == "empty":
            expected_result = {
                key: value
                for key, value in expected.items()
                if key not in {"decision_status", "query_type"}
            }
            details["checks"]["result"] = _close(
                actual_payload, expected_result, float(case.get("tolerance", 0.01)),
            )
        else:
            details["checks"]["result"] = True
    else:
        details["actual_behavior"] = _actual_behavior(
            decision.status, decision.reason_code, decision.unavailable_metrics, None,
        )
        details["checks"]["plan"] = decision.status == expected.get("decision_status") and (
            "reason_code" not in expected or decision.reason_code == expected["reason_code"]
        )
        details["checks"]["execution"] = True
        details["checks"]["result"] = True
    details["checks"]["behavior"] = details["actual_behavior"] == case["expected_behavior"]
    details["checks"]["invalid_tool_call"] = not (
        details["checks"]["plan"]
        and details["checks"]["execution"]
        and (case["expected_behavior"] not in {"result", "empty"} or details["response"] is not None)
    )
    return details


def _summary(details: list[dict[str, Any]]) -> dict[str, Any]:
    data_cases = [row for row in details if row["expected_behavior"] in {"result", "empty"}]
    result_cases = [row for row in details if row["expected_behavior"] == "result"]
    behavior_correct = sum(1 for row in details if row["checks"]["behavior"])
    plan_correct = sum(1 for row in details if row["checks"]["plan"])
    execution_success = sum(1 for row in data_cases if row["checks"]["execution"])
    result_correct = sum(1 for row in result_cases if row["checks"]["result"])
    invalid = sum(1 for row in details if row["checks"]["invalid_tool_call"])
    return {
        "cases": len(details),
        "behavior_correct": behavior_correct,
        "behavior_correct_rate": behavior_correct / len(details) if details else None,
        "plan_correct": plan_correct,
        "plan_correct_rate": plan_correct / len(details) if details else None,
        "data_cases": len(data_cases),
        "execution_success": execution_success,
        "execution_success_rate": execution_success / len(data_cases) if data_cases else None,
        "result_cases": len(result_cases),
        "result_correct": result_correct,
        "result_correct_rate": result_correct / len(result_cases) if result_cases else None,
        "invalid_tool_calls": invalid,
        "invalid_tool_call_rate": invalid / len(details) if details else None,
        "repair_success_rate": None,
        "repair_success_note": "not exercised in E4.0 seed run",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evaluation/datasets/e4_sql_cases_v1.jsonl")
    parser.add_argument("--admin-url", default=None)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    admin_url = _admin_url(args.admin_url)
    with _disposable_database(admin_url) as test_engine:
        with _fixture_runtime(test_engine):
            details = [_run_case(case) for case in cases]
    report = {
        "schema_version": "e4-sql-correctness-v1",
        "dataset_version": "e4-sql-cases-v1",
        "snapshot": "e4-fixture-v1",
        "cases": len(details),
        "summary": _summary(details),
        "details": details,
        "limits": [
            "Eight-case E4.0 seed run against one deterministic fixture snapshot; not a production sample.",
            "The fixture is a disposable PostgreSQL database; no application data is read or written.",
            "Repair success is not exercised yet; the field is intentionally null.",
            "Result correctness uses frozen gold values with a 0.01 tolerance.",
        ],
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output + "\n", encoding="utf-8")
    return 0 if report["summary"]["behavior_correct_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
