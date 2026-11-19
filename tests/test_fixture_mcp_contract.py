from __future__ import annotations

from app.mcp.client import _build_arguments_from_plan
from app.mcp.schemas import MetricQueryPlan
from app.metrics.authorization import AuthorizedMetric
from evaluation.fixtures import fixture_mcp


def _authorized_trend_plan() -> MetricQueryPlan:
    return MetricQueryPlan(
        query_type="metric_trend",
        metric_name="gmv",
        start_date="2018-04-17",
        end_date="2018-04-18",
        granularity="day",
        tool_name="query_metric_trend",
        authorized_metric=AuthorizedMetric(
            metric_id="gmv",
            provenance_type="exact",
            provenance_id="mention:fixture-contract",
            scope_fingerprint="scope:fixture-contract",
            catalog_digest="catalog:fixture-contract",
        ),
    )


def test_fixture_mcp_preserves_authorized_plan_and_real_registry_envelope() -> None:
    plan = _authorized_trend_plan()
    result = fixture_mcp(
        "metric_query",
        "查询 2018-04-17 到 2018-04-18 的 gmv 趋势",
        plan=plan,
        request_id="fixture-request:1",
    )

    assert result["server"] == "metric-registry"
    assert result["tool_name"] == "registry_trend"
    assert result["arguments"] == _build_arguments_from_plan(plan)
    assert result["request_id"] == "fixture-request:1"
    assert result["idempotency_key"] == "fixture-request:1"
    assert result["result"]["status"] == "ok"
    assert result["result"]["parsed"] == plan.model_dump(mode="json")
    assert result["result"]["parsed"]["authorized_metric"]["metric_id"] == "gmv"


def test_fixture_mcp_request_ids_do_not_cross_bind_and_missing_id_is_unique() -> None:
    plan = _authorized_trend_plan()
    first = fixture_mcp(
        "metric_query", "gmv trend", plan=plan, request_id="request-a",
    )
    second = fixture_mcp(
        "metric_query", "gmv trend", plan=plan, request_id="request-b",
    )
    generated_a = fixture_mcp("metric_query", "gmv value", plan=None)
    generated_b = fixture_mcp("metric_query", "gmv value", plan=None)

    assert (first["request_id"], second["request_id"]) == ("request-a", "request-b")
    assert first["request_id"] != second["request_id"]
    assert generated_a["request_id"] and generated_b["request_id"]
    assert generated_a["request_id"] != generated_b["request_id"]
    assert "golden-request" not in {
        first["request_id"], second["request_id"],
        generated_a["request_id"], generated_b["request_id"],
    }


def test_fixture_mcp_does_not_invent_authorization_for_unbound_plan() -> None:
    plan = MetricQueryPlan(
        query_type="metric_trend",
        metric_name="gmv",
        start_date="2018-04-17",
        end_date="2018-04-18",
        granularity="day",
        tool_name="query_metric_trend",
    )
    result = fixture_mcp(
        "metric_query", "查询 2018-04-17 到 2018-04-18 的 gmv 趋势",
        plan=plan, request_id="unauthorized-fixture-request",
    )

    assert result["result"]["status"] == "unsupported"
    assert result["result"]["parsed"] == plan.model_dump(mode="json")
    assert result["result"]["parsed"]["authorized_metric"] is None
    assert "缺少服务端 AuthorizedMetric" in result["result"]["reason"]
    assert "trend_result" not in result["result"]
