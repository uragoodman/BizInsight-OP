"""Focused full-chain acceptance for item-wise review of metric batches."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url

from app.data.repositories import daily_metrics
from app.metrics.exceptions import CapabilityNotSupported
from app.metrics.registry import get_registry
from app.tenancy import TenantContext
from app.agents.critic_agent import review_data_query_batch, review_single_data_query
from app.mcp.client import is_platform_registry_value_plan
from app.mcp.schemas import MetricQueryPlan
from app.metrics.handler_catalog import get_handler
from app.agents.data_query_agent import VALUE_TOOL_BY_METRIC


QUERY_DATE = "2024-01-02"


@pytest.fixture(scope="module")
def isolated_batch_db():
    admin_url_text = os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL")
    if not admin_url_text:
        pytest.skip("Set ISOLATED_TEST_POSTGRES_ADMIN_URL to an isolated-test PostgreSQL maintenance URL")

    admin_url = make_url(admin_url_text)
    database_name = f"metric_batch_critic_{uuid4().hex}"
    admin_engine = create_engine(
        admin_url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )
    db_engine = None
    try:
        try:
            with admin_engine.connect() as conn:
                conn.execute(text(f'CREATE DATABASE "{database_name}"'))
        except Exception as exc:
            pytest.skip(f"Cannot create isolated PostgreSQL database ({type(exc).__name__})")
        db_engine = create_engine(
            admin_url.set(database=database_name),
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5},
        )
        with db_engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA analytics"))
            conn.execute(text("CREATE SCHEMA raw"))
            conn.execute(text("CREATE TABLE analytics.daily_metrics (metric_date date PRIMARY KEY, gmv numeric(14,2), order_count bigint, avg_order_value numeric(14,2))"))
            conn.execute(text("CREATE TABLE analytics.fact_orders (order_id text PRIMARY KEY, purchase_date date NOT NULL, order_status text NOT NULL)"))
            conn.execute(text("CREATE VIEW analytics.v_delivered_orders AS SELECT * FROM analytics.fact_orders WHERE order_status = 'delivered'"))
            conn.execute(text("CREATE TABLE analytics.fact_fulfillment (order_id text PRIMARY KEY, purchase_date date NOT NULL, approval_lead_hours numeric(14,2), dispatch_lead_hours numeric(14,2), delivery_transit_hours numeric(14,2))"))
            conn.execute(text("CREATE TABLE raw.orders (order_id text PRIMARY KEY, order_purchase_timestamp timestamp NOT NULL, order_status text NOT NULL, order_delivered_customer_date timestamp, order_estimated_delivery_date timestamp)"))
            conn.execute(text("CREATE TABLE raw.order_reviews (order_id text NOT NULL, review_score integer)"))
        yield db_engine
    finally:
        if db_engine is not None:
            db_engine.dispose()
        try:
            with admin_engine.connect() as conn:
                conn.execute(text(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)'))
        finally:
            admin_engine.dispose()


def _seed(db_engine) -> None:
    with db_engine.begin() as conn:
        conn.execute(text("TRUNCATE analytics.daily_metrics, analytics.fact_fulfillment, analytics.fact_orders, raw.orders, raw.order_reviews"))
        conn.execute(text("INSERT INTO analytics.daily_metrics VALUES (:day, 1250, 5, 250)"), {"day": QUERY_DATE})
        conn.execute(text("INSERT INTO analytics.fact_orders VALUES ('r1', :day, 'delivered'), ('r2', :day, 'delivered')"), {"day": QUERY_DATE})
        conn.execute(text("INSERT INTO raw.orders VALUES ('r1', CAST(:day AS date) + time '08:00', 'delivered', NULL, NULL), ('r2', CAST(:day AS date) + time '09:00', 'delivered', NULL, NULL)"), {"day": QUERY_DATE})
        conn.execute(text("INSERT INTO raw.order_reviews VALUES ('r1', 2), ('r2', 4)"))


def _invoke(monkeypatch, question: str, db_engine, *, unsupported_metric: str | None = None) -> tuple[dict, dict, list[str]]:
    from app.graph import workflow
    from app.core import time_expression
    from app.llm import client as llm_client
    from app.graph.nodes import triage as triage_module
    from app.mcp import client as mcp_client

    # Keep every model call unavailable while leaving the deterministic
    # registered-phrase planner and all query/critic nodes on the real path.
    monkeypatch.setattr(triage_module, "llm_classify", lambda *args, **kwargs: None)
    # data_query_agent imports llm_extract inside its functions, so replace the
    # defining module attribute used by those runtime imports.
    monkeypatch.setattr(llm_client, "llm_extract", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_client, "llm_chat", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_metrics, "engine", db_engine)
    monkeypatch.setattr(time_expression, "engine", db_engine)
    # Registry failures in these cases must never fall through to a separate
    # MCP process that could be connected to a non-test database.
    monkeypatch.setattr(mcp_client, "call_local_mcp_tool", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("isolated test blocks MCP fallback")))

    if unsupported_metric:
        registry = get_registry()
        original_execute = registry.execute

        def execute(metric_id, *args, **kwargs):
            if metric_id == unsupported_metric:
                raise CapabilityNotSupported(f"runtime capability refusal for {metric_id}")
            return original_execute(metric_id, *args, **kwargs)

        monkeypatch.setattr(registry, "execute", execute)

    captured: dict = {}
    monkeypatch.setattr(workflow, "write_trace", lambda trace_id, state: captured.update(state))
    monkeypatch.setattr(workflow, "_persist_chat_turn", lambda *args, **kwargs: None)
    sql_statements: list[str] = []

    def record_sql(_conn, _cursor, statement, _parameters, _context, _many):
        sql_statements.append(" ".join(statement.split()))

    event.listen(db_engine, "before_cursor_execute", record_sql)
    trace_id = f"batch-test-{uuid4().hex}"
    try:
        result = workflow.run_chat_workflow(
            question,
            tenant_context=TenantContext(
                tenant_id="platform", user_id=f"batch-test-{uuid4().hex[:8]}",
                role="admin", scope_type="platform",
            ),
            memory_enabled=False,
            trace_id=trace_id,
        )
    finally:
        event.remove(db_engine, "before_cursor_execute", record_sql)
        from _artifact_safety import isolated_output_path
        trace_dir = isolated_output_path("METRIC_BATCH_TRACE_DIR")
        if trace_dir:
            case_name = os.getenv("PYTEST_CURRENT_TEST", "metric_batch_unknown")
            case_name = case_name.split(" ", 1)[0].rsplit("::", 1)[-1]
            trace_path = Path(trace_dir) / f"{case_name}.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(
                json.dumps(
                    {
                        "case": case_name,
                        "trace_id": trace_id,
                        "workflow_trace": captured,
                        "metric_sql_statements": sql_statements,
                        "result": locals().get("result"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
    return result, captured, sql_statements


def _question(metrics: str) -> str:
    return f"查询 {QUERY_DATE} 的 {metrics}。"


def _metric_query_count(statements: list[str]) -> int:
    return sum(
        any(token in statement for token in (
            "SELECT metric_date, gmv, order_count, avg_order_value",
            "AVG(r.review_score)",
            "AVG(approval_lead_hours)",
        ))
        for statement in statements
    )


def _sql_attempts(statements: list[str], signature: str) -> list[str]:
    """Return execute attempts matching one metric SQL signature."""
    return [statement for statement in statements if signature in statement]


def test_platform_value_registry_dispatch_covers_metrics_without_legacy_tool_names():
    """Dispatch coverage only; actual query correctness is tested separately."""
    platform_values = [
        definition for definition in get_registry().list("platform")
        if "value" in definition.implementations["platform"].capabilities
    ]
    assert platform_values
    context = TenantContext(
        tenant_id="platform", user_id="dispatch-coverage", role="admin",
        scope_type="platform",
    )
    missing_legacy_names = []
    for definition in platform_values:
        implementation = definition.implementations["platform"]
        # Confirm the declared handler key resolves. This is not an assertion
        # that the metric's SQL/data contract is correct.
        assert get_handler(implementation.handler_key)
        plan = MetricQueryPlan(
            query_type="metric_value", metric_name=definition.metric_id,
            metric_date=QUERY_DATE, tool_name=None,
        )
        assert is_platform_registry_value_plan(plan, context)
        if definition.metric_id not in VALUE_TOOL_BY_METRIC:
            missing_legacy_names.append(definition.metric_id)
    assert missing_legacy_names


def test_batch_reviewer_binds_each_result_to_its_plan_and_evidence():
    plans = [
        {"metric_name": "gmv", "query_type": "metric_value"},
        {"metric_name": "review_score_avg", "query_type": "metric_value"},
    ]
    wrappers = []
    bundles = []
    calls = []
    for index, plan in enumerate(plans, start=1):
        request_id = f"batch:{index}"
        inner = {
            "status": "ok",
            "parsed": {
                **plan,
                "authorized_metric": {"metric_id": plan["metric_name"]},
            },
            "metric_result": {
                "metric_name": plan["metric_name"],
                "metric_date": QUERY_DATE,
                "value": 1.0,
                "source_table": "isolated.test_fixture",
            },
        }
        evidence_id = f"evidence-{index}"
        wrappers.append({
            "server": "metric-registry",
            "tool_name": "registry_value",
            "request_id": request_id,
            "evidence_id": evidence_id,
            "result": inner,
        })
        bundles.append({
            "evidence_id": evidence_id,
            "source_ref": "registry_value",
            "content": {
                "request_id": request_id,
                "metric_id": plan["metric_name"],
                "query_type": plan["query_type"],
                "result": inner,
            },
        })
        calls.append({
            "server": "metric-registry", "tool_name": "registry_value",
            "request_id": request_id, "planned_metric_id": plan["metric_name"],
            "planned_operation": plan["query_type"], "status": "ok",
        })

    reviewed = review_data_query_batch(wrappers, plans, bundles, calls)
    assert reviewed["batch_status"] == "all_success"
    assert [item["metric_id"] for item in reviewed["items"]] == ["gmv", "review_score_avg"]
    assert reviewed["all_items_reviewed"] is True

    wrappers[1]["evidence_id"] = "evidence-1"
    invalid = review_data_query_batch(wrappers, plans, bundles, calls)
    assert invalid["batch_status"] == "execution_failed"
    assert invalid["items"][1]["status"] == "failed"
    assert "matching evidence" in invalid["items"][1]["failure_reason"]


@pytest.mark.parametrize(
    ("execution_status", "expected_item_status", "expected_batch_status"),
    [
        ("not_found", "unavailable", "partial_unavailable"),
        ("unsupported", "unsupported", "partial_unavailable"),
        ("execution_failed", "execution_failed", "execution_failed"),
    ],
)
def test_batch_reviewer_preserves_terminal_item_outcomes(
    execution_status, expected_item_status, expected_batch_status,
):
    plan = {"metric_name": "avg_approval_lead_hours", "query_type": "metric_value"}
    inner = {
        "status": execution_status,
        "parsed": {
            **plan,
            "authorized_metric": {"metric_id": plan["metric_name"]},
        },
        "reason": f"test {execution_status}",
    }
    wrapper = {
        "server": "metric-registry", "tool_name": "registry_value",
        "request_id": "batch:1", "evidence_id": "evidence-1", "result": inner,
    }
    bundle = {
        "evidence_id": "evidence-1",
        "source_ref": "registry_value",
        "content": {
            "request_id": "batch:1",
            "metric_id": plan["metric_name"],
            "query_type": plan["query_type"],
            "result": inner,
        },
    }

    calls = [{
        "server": "metric-registry", "tool_name": "registry_value", "request_id": "batch:1",
        "planned_metric_id": plan["metric_name"], "planned_operation": plan["query_type"],
        "status": execution_status if execution_status in {"ok", "not_found", "unsupported"} else "failed",
    }]
    reviewed = review_data_query_batch([wrapper], [plan], [bundle], calls)
    assert reviewed["items"][0]["status"] == expected_item_status
    assert reviewed["batch_status"] == expected_batch_status
    assert reviewed["items"][0]["failure_reason"] == f"test {execution_status}"


def _single_binding_fixture():
    plan = {"metric_name": "gmv", "query_type": "metric_value"}
    inner = {
        "status": "ok",
        "parsed": {**plan, "authorized_metric": {"metric_id": "gmv"}},
        "metric_result": {
            "metric_name": "gmv", "metric_date": QUERY_DATE, "value": 12.0,
            "source_table": "analytics.daily_metrics",
        },
    }
    wrapper = {
        "server": "metric-registry", "tool_name": "registry_value",
        "request_id": "req:1", "evidence_id": "ev:1", "result": inner,
    }
    bundle = {
        "evidence_id": "ev:1", "source_ref": "registry_value",
        "content": {
            "request_id": "req:1", "metric_id": "gmv", "query_type": "metric_value", "result": inner,
        },
    }
    call = {
        "server": "metric-registry", "tool_name": "registry_value", "request_id": "req:1",
        "planned_metric_id": "gmv", "planned_operation": "metric_value", "status": "ok",
    }
    return plan, wrapper, [bundle], [call]


def test_single_reviewer_requires_full_plan_invocation_and_evidence_binding():
    plan, wrapper, bundles, calls = _single_binding_fixture()
    accepted = review_single_data_query(wrapper, plan, bundles, calls)
    assert accepted["status"] == "approved"
    assert all(accepted["items"][0]["binding_checks"].values())

    # Missing evidence association cannot pass on query success alone.
    missing = dict(wrapper, evidence_id=None)
    assert review_single_data_query(missing, plan, bundles, calls)["status"] == "rejected"

    # A different request ID cannot borrow another request's call/evidence.
    mismatched_request = dict(wrapper, request_id="req:other")
    assert review_single_data_query(mismatched_request, plan, bundles, calls)["status"] == "rejected"

    # Result metric and operation must both agree with the authorized plan.
    wrong_metric_inner = dict(wrapper["result"], parsed={**wrapper["result"]["parsed"], "metric_name": "order_count"})
    wrong_metric = dict(wrapper, result=wrong_metric_inner)
    assert review_single_data_query(wrong_metric, plan, bundles, calls)["status"] == "rejected"
    wrong_operation_inner = dict(wrapper["result"], parsed={**wrapper["result"]["parsed"], "query_type": "metric_trend"})
    wrong_operation = dict(wrapper, result=wrong_operation_inner)
    assert review_single_data_query(wrong_operation, plan, bundles, calls)["status"] == "rejected"


@pytest.mark.parametrize(
    ("execution_status", "expected_item_status"),
    [("not_found", "unavailable"), ("unsupported", "unsupported")],
)
def test_single_binding_preserves_terminal_no_data_and_controlled_refusal(execution_status, expected_item_status):
    plan, wrapper, bundles, calls = _single_binding_fixture()
    inner = dict(wrapper["result"], status=execution_status, reason="expected terminal limitation")
    wrapper = dict(wrapper, result=inner)
    bundles = [{
        **bundles[0], "content": {**bundles[0]["content"], "result": inner},
    }]
    calls = [{**calls[0], "status": execution_status}]
    reviewed = review_single_data_query(wrapper, plan, bundles, calls)
    assert reviewed["items"][0]["status"] == expected_item_status
    assert reviewed["items"][0]["binding_checks"]["evidence_matches_request_plan_result"] is True
    if execution_status == "not_found":
        assert reviewed["status"] == "partial"
        assert reviewed["batch_status"] == "partial_unavailable"
    elif execution_status == "unsupported":
        assert reviewed["status"] == "approved"


def test_single_not_found_with_invalid_evidence_binding_is_not_terminal_unavailable():
    plan, wrapper, bundles, calls = _single_binding_fixture()
    inner = dict(wrapper["result"], status="not_found", reason="no row")
    wrapper = dict(wrapper, result=inner)
    bundles = [{
        **bundles[0], "content": {**bundles[0]["content"], "result": inner},
    }]
    calls = [{**calls[0], "status": "not_found"}]

    missing_evidence = dict(wrapper, evidence_id=None)
    review = review_single_data_query(missing_evidence, plan, bundles, calls)
    assert review["status"] == "rejected"
    assert review.get("batch_status") is None
    assert review["items"][0]["status"] == "failed"


def test_full_chain_reviews_all_successful_items(monkeypatch, isolated_batch_db):
    _seed(isolated_batch_db)
    result, trace, sql = _invoke(
        monkeypatch, _question("GMV 和平均评分"), isolated_batch_db,
    )

    plans = trace["metric_query_batch_plan"]["plans"]
    reviews = trace["critic_result"]["data_review"]
    assert trace["task_type"] == "metric_query"
    assert {plan["metric_name"] for plan in plans} == {"gmv", "review_score_avg"}
    assert [item["status"] for item in reviews["items"]] == ["approved", "approved"]
    assert reviews["batch_status"] == "all_success"
    assert all(item["evidence_id"] for item in reviews["items"])
    assert trace["critic_verdict"] == "approved"
    assert result["presentation"]["status"] == "completed"
    assert "1250.00" in result["answer"] and "3.00" in result["answer"]
    metric_units = {
        item["result"]["parsed"]["metric_name"]: item["result"]["metric_result"]["unit"]
        for item in trace["query_results"]
    }
    assert metric_units == {"gmv": "BRL", "review_score_avg": "score"}
    fact_units = {
        fact["metric_name"]: fact["unit"]
        for fact in result["presentation"]["facts"]
    }
    assert fact_units == metric_units
    assert "雷亚尔" in result["answer"] and "分" in result["answer"]
    assert len(trace["tool_calls"]) == 2
    assert _metric_query_count(sql) == 2


def test_full_chain_keeps_success_when_later_metric_has_no_data(monkeypatch, isolated_batch_db):
    _seed(isolated_batch_db)
    result, trace, sql = _invoke(
        monkeypatch, _question("GMV 和平均审批时长"), isolated_batch_db,
    )

    reviews = trace["critic_result"]["data_review"]
    by_metric = {item["metric_id"]: item for item in reviews["items"]}
    assert by_metric["gmv"]["status"] == "approved"
    assert by_metric["avg_approval_lead_hours"]["status"] == "unavailable"
    assert by_metric["avg_approval_lead_hours"]["failure_reason"]
    assert reviews["batch_status"] == "partial_unavailable"
    assert trace["critic_verdict"] == "rejected"
    assert result["presentation"]["status"] == "partial"
    assert "部分指标不可用" in result["answer"]
    assert "当前范围无数据" in result["answer"]
    assert "小时" not in result["answer"]
    assert len(trace["tool_calls"]) == 2
    assert _metric_query_count(sql) == 2


def test_full_chain_records_later_query_execution_failure(monkeypatch, isolated_batch_db):
    _seed(isolated_batch_db)
    with isolated_batch_db.begin() as conn:
        conn.execute(text("DROP TABLE raw.order_reviews"))
    try:
        result, trace, sql = _invoke(
            monkeypatch, _question("GMV 和平均评分"), isolated_batch_db,
        )
    finally:
        with isolated_batch_db.begin() as conn:
            conn.execute(text("CREATE TABLE raw.order_reviews (order_id text NOT NULL, review_score integer)"))

    reviews = trace["critic_result"]["data_review"]
    by_metric = {item["metric_id"]: item for item in reviews["items"]}
    assert by_metric["gmv"]["status"] == "approved"
    assert by_metric["review_score_avg"]["status"] == "execution_failed"
    assert by_metric["review_score_avg"]["failure_reason"]
    assert reviews["batch_status"] == "execution_failed"
    assert trace["critic_verdict"] == "rejected"
    assert result["presentation"]["status"] == "failed"
    assert "执行失败" in result["answer"]
    # before_cursor_execute observes attempts, including the failed review SQL.
    # Attribute each attempt to its metric and inspect outcomes separately so
    # a duplicate retry cannot satisfy a total-count assertion.
    gmv_attempts = _sql_attempts(sql, "SELECT metric_date, gmv, order_count, avg_order_value")
    review_attempts = _sql_attempts(sql, "AVG(r.review_score) AS review_score_avg")
    assert len(gmv_attempts) == 1
    assert len(review_attempts) == 1
    assert _metric_query_count(sql) == 2

    plans = trace["metric_query_batch_plan"]["plans"]
    query_results = trace["query_results"]
    assert [plan["metric_name"] for plan in plans] == ["gmv", "review_score_avg"]
    assert [item["result"]["status"] for item in query_results] == ["ok", "execution_failed"]
    assert [item["result"]["parsed"]["metric_name"] for item in query_results] == [
        "gmv", "review_score_avg",
    ]
    calls = trace["tool_calls"]
    assert len(calls) == 2
    assert [call["status"] for call in calls] == ["ok", "failed"]
    request_ids = [item["request_id"] for item in query_results]
    assert len(set(request_ids)) == 2
    assert [call["request_id"] for call in calls] == request_ids
    assert all(item.get("evidence_id") for item in query_results)


def test_full_chain_records_controlled_unsupported_item_without_legacy_retry(monkeypatch, isolated_batch_db):
    _seed(isolated_batch_db)
    result, trace, sql = _invoke(
        monkeypatch,
        _question("GMV 和平均评分"),
        isolated_batch_db,
        unsupported_metric="review_score_avg",
    )

    reviews = trace["critic_result"]["data_review"]
    by_metric = {item["metric_id"]: item for item in reviews["items"]}
    assert by_metric["gmv"]["status"] == "approved"
    assert by_metric["review_score_avg"]["status"] == "unsupported"
    assert by_metric["review_score_avg"]["failure_reason"]
    assert reviews["batch_status"] == "partial_unavailable"
    assert trace["critic_verdict"] == "rejected"
    assert result["presentation"]["status"] == "partial"
    assert "能力不支持" in result["answer"]
    assert len(trace["tool_calls"]) == 2
    assert _metric_query_count(sql) == 1


def test_full_chain_review_is_invariant_to_metric_order(monkeypatch, isolated_batch_db):
    _seed(isolated_batch_db)
    result, trace, sql = _invoke(
        monkeypatch, _question("平均评分和 GMV"), isolated_batch_db,
    )

    plans = trace["metric_query_batch_plan"]["plans"]
    reviews = trace["critic_result"]["data_review"]
    assert [plan["metric_name"] for plan in plans] == [item["metric_id"] for item in reviews["items"]]
    assert {item["metric_id"]: item["status"] for item in reviews["items"]} == {
        "gmv": "approved", "review_score_avg": "approved",
    }
    assert reviews["batch_status"] == "all_success"
    assert result["presentation"]["status"] == "completed"
    assert len(trace["tool_calls"]) == 2
    assert _metric_query_count(sql) == 2
