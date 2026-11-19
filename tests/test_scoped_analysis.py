from datetime import date

import pytest
from unittest.mock import patch

from app.mcp.client import _resolve_scoped_compare_periods, query_business_data, query_scoped_business_data
from app.agents.data_query_agent import (
    _explicit_requested_granularity,
    _extract_dimension,
    apply_trusted_time_range,
    is_explicit_standard_metric_request,
    plan_confirmed_metric_queries,
)
from app.mcp.schemas import MetricQueryPlan
from app.agents.critic_agent import review_data_query_result
from app.metrics.authorization import authorize_metric
from app.schemas.explorer import ControlledAnalysisScope
from app.tenancy import TenantContext, activate_tenant_context


@pytest.fixture(autouse=True)
def trusted_scoped_context():
    """Scoped MCP now fail-closes when no trusted request context exists."""
    with activate_tenant_context(TenantContext(
        tenant_id="platform", user_id="scoped-test", role="admin", scope_type="platform",
    )):
        yield


SCOPE = {
    "seller_id": "seller-a", "default_start_date": "2018-08-01", "default_end_date": "2018-08-10",
    "metric_name": "gmv", "granularity": "day",
}


def test_scoped_compare_prefers_structured_dates_and_never_expands_scope():
    scope = ControlledAnalysisScope.model_validate(SCOPE)
    in_scope = MetricQueryPlan(
        query_type="metric_contribution", metric_name="gmv",
        current_date="2018-08-10", baseline_date="2018-08-09",
    )
    assert _resolve_scoped_compare_periods(scope, "compare", in_scope) == (
        date(2018, 8, 10), date(2018, 8, 10),
        date(2018, 8, 9), date(2018, 8, 9),
    )
    out_of_scope = in_scope.model_copy(update={"baseline_date": "2018-07-31"})
    assert _resolve_scoped_compare_periods(scope, "compare", out_of_scope) is None


def test_scoped_period_contribution_accepts_two_ranges_only_inside_scope():
    scope = ControlledAnalysisScope.model_validate({
        **SCOPE,
        "default_start_date": "2017-09-01",
        "default_end_date": "2018-08-31",
    })
    plan = MetricQueryPlan(
        query_type="metric_contribution", metric_name="gmv", granularity="month",
        baseline_start="2017-09-01", baseline_end="2018-03-01",
        current_start="2018-03-02", current_end="2018-08-31",
    )
    assert _resolve_scoped_compare_periods(scope, "compare", plan) == (
        date(2018, 3, 2), date(2018, 8, 31),
        date(2017, 9, 1), date(2018, 3, 1),
    )
    escaped = plan.model_copy(update={"baseline_start": "2017-08-31"})
    assert _resolve_scoped_compare_periods(scope, "compare", escaped) is None


def test_scoped_period_contribution_executes_both_full_ranges(monkeypatch):
    scope_data = SCOPE | {
        "seller_id": "seller-a", "metric_name": "gmv",
        "allowed_metrics": ["gmv"], "granularity": "month",
        "default_start_date": "2017-09-01", "default_end_date": "2018-08-31",
    }
    plan = _authorized_plan(MetricQueryPlan(
        query_type="metric_contribution", metric_name="gmv", dimension="product_category",
        granularity="month", baseline_start="2017-09-01", baseline_end="2018-03-01",
        current_start="2018-03-02", current_end="2018-08-31",
    ), scope_data)
    breakdown_calls: list[tuple[str, str]] = []

    def fake_breakdown(scoped, metric_name, limit):
        breakdown_calls.append((scoped.default_start_date.isoformat(), scoped.default_end_date.isoformat()))
        return {"items": [], "status": "ok", "source_table": "scoped"}

    monkeypatch.setattr("app.mcp.client.breakdown_scoped_product_category", fake_breakdown)
    monkeypatch.setattr(
        "app.mcp.client.compare_scoped_metric",
        lambda *_args, **kwargs: {
            "left_value": 15.0, "right_value": 10.0,
            "delta_value": 5.0, "status": "ok",
        },
    )
    result = query_scoped_business_data("compare two periods", scope_data, plan=plan)
    assert result["result"]["status"] == "ok"
    assert result["tool_name"] == "compare_breakdown_by_dimension_period"
    assert result["arguments"]["current_start"] == "2018-03-02"
    assert result["arguments"]["baseline_start"] == "2017-09-01"
    assert breakdown_calls == [("2018-03-02", "2018-08-31"), ("2017-09-01", "2018-03-01")]
    assert result["arguments"]["time_resolution"]["periods"] == [
        {"name": "current", "start_date": "2018-03-02", "end_date": "2018-08-31"},
        {"name": "baseline", "start_date": "2017-09-01", "end_date": "2018-03-01"},
    ]


def test_scoped_drilldown_executes_month_position_inside_evidence_category(monkeypatch):
    scope_data = SCOPE | {"allowed_metrics": ["gmv"], "allowed_dimensions": ["month_position"]}
    plan = _authorized_plan(MetricQueryPlan(
        query_type="metric_contribution", metric_name="gmv", dimension="month_position",
        granularity="month", baseline_start="2018-08-01", baseline_end="2018-08-05",
        current_start="2018-08-06", current_end="2018-08-10",
        filters={"product_category": "furniture"},
    ), scope_data)
    seen: list[tuple[str, str, str]] = []

    def fake_month_position(scoped, metric_name, *, product_category_filter=None):
        seen.append((scoped.default_start_date.isoformat(), metric_name, product_category_filter))
        return {"items": [{"dimension_value": "月初(1-10日)", "value": 20.0}],
                "population_total": 20.0,
                "status": "ok", "source_table": "scoped"}

    monkeypatch.setattr("app.mcp.client.apply_trusted_time_range", lambda value, _question, **_kwargs: value)
    monkeypatch.setattr("app.mcp.client.breakdown_scoped_month_position", fake_month_position)
    monkeypatch.setattr(
        "app.mcp.client.compare_scoped_metric",
        lambda *_args, **_kwargs: pytest.fail(
            "filtered month_position totals must use the same category-filtered query"
        ),
    )
    result = query_scoped_business_data("继续下钻家具品类", scope_data, plan=plan)
    assert result["result"]["status"] == "ok"
    contribution = result["result"]["contribution_result"]
    assert contribution["dimension"] == "month_position"
    assert contribution["current_total"] == 20.0
    assert contribution["baseline_total"] == 20.0
    assert contribution["total_delta"] == 0.0
    assert contribution["method"] == "additive_change_v1"
    assert result["tool_name"] == "compare_breakdown_by_dimension_period"
    assert result["execution_metadata"]["filter_applied"] == {"product_category": "furniture"}
    assert seen == [
        ("2018-08-06", "gmv", "furniture"),
        ("2018-08-01", "gmv", "furniture"),
    ]


def test_scoped_month_position_category_filter_is_parameterized(monkeypatch):
    from app.data.repositories import explorer
    from contextlib import contextmanager

    captured: dict[str, object] = {}

    class Result:
        def mappings(self):
            return self

        def all(self):
            return [{"dimension_value": "月初(1-10日)", "value": 3.0}]

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, parameters):
            captured["sql"] = str(statement)
            captured["parameters"] = dict(parameters)
            return Result()

    class Engine:
        def connect(self):
            return Connection()

    @contextmanager
    def fake_seller_transaction(_engine, _scope, **_kwargs):
        yield Connection()

    monkeypatch.setattr(explorer, "engine", Engine())
    monkeypatch.setattr(explorer, "seller_query_transaction", fake_seller_transaction)
    scope = ControlledAnalysisScope.model_validate(SCOPE)
    value = "furniture' OR 1=1 --"
    explorer.breakdown_scoped_month_position(scope, "gmv", product_category_filter=value)
    assert ":product_category_filter" in str(captured["sql"])
    assert value not in str(captured["sql"])
    assert captured["parameters"]["product_category_filter"] == value


def test_platform_period_contribution_uses_period_mcp_tool_not_point_registry(monkeypatch):
    identity_scope = {
        "tenant_id": "platform", "user_id": "scoped-test",
        "permissions": "admin", "scope_type": "platform",
    }
    authorized = authorize_metric(
        "gmv", provenance_type="exact", provenance_id="mention-gmv",
        scope="platform", scope_data=identity_scope,
    )
    plan = MetricQueryPlan(
        query_type="metric_contribution", metric_name="gmv", tool_name="compare_breakdown_by_dimension",
        dimension="product_category", granularity="month",
        baseline_start="2018-07-01", baseline_end="2018-07-31",
        current_start="2018-08-01", current_end="2018-08-31",
        authorized_metric=authorized,
    )
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr("app.mcp.client.call_local_mcp_tool", lambda name, args, **_: calls.append((name, args)) or {
        "status": "ok", "metric_name": "gmv", "dimension": "product_category",
        "current_items": [], "baseline_items": [], "current_total": 10.0,
        "baseline_total": 8.0, "total_delta": 2.0,
    })
    result = query_business_data("metric_query", "compare two monthly periods", plan=plan)
    assert calls and calls[0][0] == "compare_breakdown_by_dimension_period"
    assert calls[0][1]["current_start"] == "2018-08-01"
    assert calls[0][1]["baseline_end"] == "2018-07-31"
    assert result["tool_name"] == "compare_breakdown_by_dimension_period"
    assert result["result"]["status"] == "ok"


def test_platform_ratio_attribution_uses_dedicated_facts_mcp_tool(monkeypatch):
    identity_scope = {
        "tenant_id": "platform", "user_id": "scoped-test",
        "permissions": "admin", "scope_type": "platform",
    }
    authorized = authorize_metric(
        "freight_rate", provenance_type="exact", provenance_id="mention-freight-rate",
        scope="platform", scope_data=identity_scope,
    )
    plan = MetricQueryPlan(
        query_type="metric_contribution", metric_name="freight_rate",
        tool_name="compare_freight_rate_ratio_facts_by_dimension_period",
        dimension="month_position", granularity="month",
        baseline_start="2018-07-01", baseline_end="2018-07-31",
        current_start="2018-08-01", current_end="2018-08-31",
        authorized_metric=authorized,
    )
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "app.mcp.client.call_local_mcp_tool",
        lambda name, args, **_: calls.append((name, args)) or {
            "status": "ok", "metric_name": "freight_rate", "complete": True,
        },
    )
    result = query_business_data("metric_query", "compare monthly freight rate", plan=plan)
    assert calls and calls[0][0] == "compare_freight_rate_ratio_facts_by_dimension_period"
    assert calls[0][1] == {
        "metric_name": "freight_rate",
        "current_start": "2018-08-01", "current_end": "2018-08-31",
        "baseline_start": "2018-07-01", "baseline_end": "2018-07-31",
        "granularity": "month", "dimension": "month_position",
    }
    assert result["tool_name"] == "compare_freight_rate_ratio_facts_by_dimension_period"
    assert result["result"]["status"] == "ok"


def test_platform_average_order_value_attribution_uses_dedicated_facts_mcp_tool(monkeypatch):
    identity_scope = {
        "tenant_id": "platform", "user_id": "scoped-test",
        "permissions": "admin", "scope_type": "platform",
    }
    authorized = authorize_metric(
        "avg_order_value", provenance_type="exact", provenance_id="mention-aov",
        scope="platform", scope_data=identity_scope,
    )
    plan = MetricQueryPlan(
        query_type="metric_contribution", metric_name="avg_order_value",
        tool_name="compare_avg_order_value_ratio_facts_by_dimension_period",
        dimension="month_position", granularity="month",
        baseline_start="2018-07-01", baseline_end="2018-07-31",
        current_start="2018-08-01", current_end="2018-08-31",
        authorized_metric=authorized,
    )
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "app.mcp.client.call_local_mcp_tool",
        lambda name, args, **_: calls.append((name, args)) or {
            "status": "ok", "metric_name": "avg_order_value", "complete": True,
        },
    )

    result = query_business_data("metric_query", "比较客单价月内时段贡献", plan=plan)

    assert calls == [(
        "compare_avg_order_value_ratio_facts_by_dimension_period",
        {
            "metric_name": "avg_order_value",
            "current_start": "2018-08-01", "current_end": "2018-08-31",
            "baseline_start": "2018-07-01", "baseline_end": "2018-07-31",
            "granularity": "month", "dimension": "month_position",
        },
    )]
    assert result["tool_name"] == "compare_avg_order_value_ratio_facts_by_dimension_period"
    assert result["result"]["status"] == "ok"


@pytest.mark.parametrize(
    ("tool_name", "metric_name", "dimension"),
    [
        ("compare_freight_rate_ratio_facts_by_dimension_period", "avg_order_value", "month_position"),
        ("compare_avg_order_value_ratio_facts_by_dimension_period", "avg_order_value", "product_category"),
    ],
)
def test_platform_ratio_facts_dispatch_rejects_wrong_metric_or_dimension(
    monkeypatch, tool_name, metric_name, dimension,
):
    identity_scope = {
        "tenant_id": "platform", "user_id": "scoped-test",
        "permissions": "admin", "scope_type": "platform",
    }
    authorized = authorize_metric(
        metric_name, provenance_type="exact", provenance_id="mention-ratio",
        scope="platform", scope_data=identity_scope,
    )
    plan = MetricQueryPlan(
        query_type="metric_contribution", metric_name=metric_name,
        tool_name=tool_name, dimension=dimension, granularity="month",
        baseline_start="2018-07-01", baseline_end="2018-07-31",
        current_start="2018-08-01", current_end="2018-08-31",
        authorized_metric=authorized,
    )
    monkeypatch.setattr("app.mcp.client.call_local_mcp_tool", lambda *_a, **_k: pytest.fail("must fail before MCP"))

    result = query_business_data("metric_query", "比较指标分组贡献", plan=plan)

    assert result["result"]["status"] == "unsupported"
    assert result["tool_name"] is None


def test_ratio_facts_mcp_schema_rejects_unregistered_metric_or_dimension():
    from app.mcp.schemas import FreightRateRatioFactsArguments

    base = {
        "metric_name": "freight_rate", "current_start": "2018-08-01",
        "current_end": "2018-08-31", "baseline_start": "2018-07-01",
        "baseline_end": "2018-07-31", "granularity": "month",
        "dimension": "month_position",
    }
    assert FreightRateRatioFactsArguments.model_validate(base).dimension == "month_position"
    with pytest.raises(Exception):
        FreightRateRatioFactsArguments.model_validate({**base, "metric_name": "gmv"})
    with pytest.raises(Exception):
        FreightRateRatioFactsArguments.model_validate({**base, "dimension": "customer_city"})


def test_monthly_language_is_a_monthly_trend_not_a_point_query() -> None:
    """Chinese monthly/extrema wording must produce a trend-shaped plan."""
    question = "\u8fd9\u4e00\u5e74\u7684\u6bcf\u4e2a\u6708\u5e97\u94fa\u5e73\u5747\u6bcf\u5355\u8fd0\u8d39\u662f\u591a\u5c11\uff0c\u54ea\u4e2a\u6708\u6700\u9ad8\u54ea\u4e2a\u6708\u6700\u4f4e\uff1f"
    plan = MetricQueryPlan(
        query_type="metric_value",
        metric_name="seller_avg_freight_per_order",
    )

    normalized = apply_trusted_time_range(plan, question)

    assert normalized.query_type == "metric_trend"
    assert normalized.granularity == "month"


def test_product_noun_in_metric_does_not_request_product_breakdown() -> None:
    assert _extract_dimension(
        "\u6309\u6708\u5206\u6790\u5e97\u94fa\u5e73\u5747\u5546\u54c1\u552e\u4ef7\u7684\u53d8\u5316"
    ) is None
    assert _extract_dimension(
        "\u6309\u6708\u7edf\u8ba1\u6d3b\u8dc3\u5546\u54c1\u6570"
    ) is None
    assert _extract_dimension(
        "\u6309\u5546\u54c1\u6bd4\u8f83\u5e73\u5747\u8bc4\u5206\u6700\u9ad8\u548c\u6700\u4f4e\u7684\u5546\u54c1"
    ) == "product_category"


def test_exact_metric_operation_outranks_template_keyword() -> None:
    question = (
        "\u6309\u6708\u7edf\u8ba1\u5e97\u94fa\u5e73\u5747\u5ef6\u8bef\u9001\u8fbe\u5929\u6570，"
        "\u54ea\u4e2a\u6708\u6700\u9ad8？"
    )
    assert is_explicit_standard_metric_request(question) is True


def test_selected_explorer_metric_is_default_not_a_hard_metric_lock() -> None:
    scope = {
        "seller_id": "a" * 32,
        "default_start_date": "2017-09-01",
        "default_end_date": "2018-08-31",
        "metric_name": "seller_avg_freight_per_order",
        "allowed_metrics": ["seller_items_per_order"],
        "granularity": "day",
        "tenant_id": "platform",
        "user_id": "scoped-test",
        "scope_type": "seller",
    }
    question = (
        "\u6309\u6708\u5206\u6790\u5e97\u94fa\u5e73\u5747\u6bcf\u5355\u5546\u54c1\u6570，"
        "\u6700\u9ad8\u548c\u6700\u4f4e\u5206\u522b\u662f\u54ea\u4e2a\u6708？"
    )
    with patch("app.llm.client.llm_extract", return_value=None):
        batch = plan_confirmed_metric_queries(
            question,
            scope="seller",
            allowed_metrics={"seller_items_per_order"},
            scope_data=scope,
            default_metric="seller_avg_freight_per_order",
            explorer_selection_id="explorer:test-selection",
        )
    assert batch.decision.status == "confirmed"
    assert batch.decision.confirmed_metrics == ["seller_items_per_order"]
    assert len(batch.plans) == 1
    assert batch.plans[0].authorized_metric is not None
    assert batch.plans[0].authorized_metric.provenance_type == "exact"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("过去三年每年店铺平均每单运费是多少，哪一年最高？", "year"),
        ("本月每天店铺平均每单运费是多少？", "day"),
        ("按季度比较店铺平均每单运费", "quarter"),
        ("每周店铺平均每单运费是多少？", "week"),
    ],
)
def test_explicit_granularity_supports_all_time_buckets(question: str, expected: str) -> None:
    assert _explicit_requested_granularity(question) == expected
    normalized = apply_trusted_time_range(
        MetricQueryPlan(query_type="metric_value", metric_name="gmv"), question,
    )
    assert normalized.query_type == "metric_trend"
    assert normalized.granularity == expected


def test_monthly_scoped_query_uses_month_granularity(monkeypatch) -> None:
    """The scoped SQL handler must receive the requested monthly grain."""
    monkeypatch.setattr(
        "app.core.time_expression.data_time_bounds",
        lambda: pytest.fail("scoped time parsing must not query global dataset bounds"),
    )
    scope = SCOPE | {
        "metric_name": "seller_avg_freight_per_order",
        "allowed_metrics": ["seller_avg_freight_per_order"],
        "default_start_date": "2017-09-01",
        "default_end_date": "2018-08-31",
        "granularity": "day",
    }
    plan = _authorized_plan(
        MetricQueryPlan(
            query_type="metric_value",
            metric_name="seller_avg_freight_per_order",
        ),
        scope,
    )
    calls: list[tuple[str, str, str]] = []

    def fake_trend(scoped, metric_name, *, start_date=None, end_date=None):
        calls.append((scoped.granularity, start_date.isoformat(), end_date.isoformat()))
        return {
            "metric_name": metric_name,
            "points": [{"bucket": "2017-09-01", "value": 20.0}],
            "source_table": "seller-extension",
            "status": "ok",
        }

    monkeypatch.setattr("app.mcp.client.query_scoped_trend", fake_trend)
    question = "\u8fd9\u4e00\u5e74\u7684\u6bcf\u4e2a\u6708\u5e97\u94fa\u5e73\u5747\u6bcf\u5355\u8fd0\u8d39\u662f\u591a\u5c11\uff0c\u54ea\u4e2a\u6708\u6700\u9ad8\u54ea\u4e2a\u6708\u6700\u4f4e\uff1f"
    result = query_scoped_business_data(question, scope, plan=plan)

    assert result["result"]["status"] == "ok"
    assert result["result"]["trend_result"]["granularity"] == "month"
    assert calls == [("month", "2017-09-01", "2018-08-31")]


def _authorized_plan(plan: MetricQueryPlan, scope: dict[str, object] = SCOPE) -> MetricQueryPlan:
    normalized_scope = ControlledAnalysisScope.model_validate(scope).model_dump(mode="json")
    authorized = authorize_metric(
        plan.metric_name or str(scope["metric_name"]),
        provenance_type="explorer_selection",
        provenance_id="scoped-test-selection",
        scope="seller",
        scope_data={**normalized_scope, "tenant_id": None, "user_id": None, "permissions": None},
    )
    return plan.model_copy(update={"authorized_metric": authorized})


def test_scoped_query_defaults_to_locked_range(monkeypatch) -> None:
    monkeypatch.setattr("app.mcp.client.query_scoped_metric", lambda scope, metric: {"status": "ok", "metric_name": metric, "metric_date": "2018-08-01 至 2018-08-10", "value": 12.0, "source_table": "scoped"})
    result = query_scoped_business_data("查询 GMV", SCOPE, plan=_authorized_plan(MetricQueryPlan(metric_name="gmv")))
    assert result["tool_name"] == "query_scoped_metric"
    assert result["arguments"]["scope"]["seller_id"] == "seller-a"


def test_scoped_query_rejects_dates_outside_locked_range(monkeypatch) -> None:
    """A natural-language date cannot silently expand the confirmed session scope."""
    from app.mcp.schemas import MetricQueryPlan
    monkeypatch.setattr("app.mcp.client.query_scoped_metric", lambda *_args, **_kwargs: pytest.fail("out-of-scope query must not reach the repository"))
    result = query_scoped_business_data("查询 2018-08-11 的 GMV", SCOPE, plan=_authorized_plan(MetricQueryPlan(metric_name="gmv", metric_date="2018-08-11")))
    assert result["result"]["status"] == "unsupported"
    assert "超出当前会话锁定范围" in result["result"]["reason"]


def test_scoped_query_rejects_a_different_seller(monkeypatch) -> None:
    from app.mcp.schemas import MetricQueryPlan
    scope = SCOPE | {"seller_id": "a" * 32}
    result = query_scoped_business_data(f"查询 {'b' * 32} 的 GMV", scope, plan=_authorized_plan(MetricQueryPlan(metric_name="gmv"), scope))
    assert result["result"]["status"] == "unsupported"
    assert "店铺" in result["result"]["reason"]


def test_scoped_query_rejects_unapproved_dimension(monkeypatch) -> None:
    from app.mcp.schemas import MetricQueryPlan
    plan = _authorized_plan(MetricQueryPlan(query_type="metric_breakdown", metric_name="gmv", dimension="customer_state"))
    result = query_scoped_business_data("按州拆解 GMV", SCOPE, plan=plan)
    assert result["result"]["status"] == "unsupported"
    assert "Dimension scope is not authorized" in result["result"]["reason"]


def test_scoped_chat_example_compares_october_with_previous_month(monkeypatch) -> None:
    """Chat regression: ``10月与上月`` must execute inside the locked Sep-Oct scope."""
    scope = SCOPE | {
        "seller_id": "seller-a",
        "metric_name": "freight_value",
        "default_start_date": "2017-09-01",
        "default_end_date": "2017-10-31",
        "granularity": "month",
    }
    plan = _authorized_plan(
        MetricQueryPlan(query_type="metric_compare", metric_name="freight_value"),
        scope,
    )
    calls: list[dict] = []

    monkeypatch.setattr("app.mcp.client.apply_trusted_time_range", lambda value, _question, **_kwargs: value)
    monkeypatch.setattr(
        "app.mcp.client.compare_scoped_metric",
        lambda _scope, metric_name, **kwargs: calls.append({"metric_name": metric_name, **kwargs}) or {
            "metric_name": metric_name,
            "left_date": kwargs["current_start"].isoformat(), "left_end_date": kwargs["current_end"].isoformat(),
            "right_date": kwargs["baseline_start"].isoformat(), "right_end_date": kwargs["baseline_end"].isoformat(),
            "left_value": 120.0, "right_value": 100.0, "delta_value": 20.0, "delta_ratio": 20.0,
            "source_table": "scoped", "status": "ok",
        },
    )

    result = query_scoped_business_data("判断10月与上月的运费总额对比", scope, plan=plan)

    assert result["tool_name"] == "query_scoped_metric_compare"
    assert result["result"]["status"] == "ok"
    assert result["result"]["compare_result"]["delta_ratio"] == 20.0
    assert calls == [{
        "metric_name": "freight_value",
        "current_start": __import__("datetime").date(2017, 10, 1),
        "current_end": __import__("datetime").date(2017, 10, 31),
        "baseline_start": __import__("datetime").date(2017, 9, 1),
        "baseline_end": __import__("datetime").date(2017, 9, 30),
    }]


def test_unsupported_result_is_terminal_and_not_a_critic_retry() -> None:
    review = review_data_query_result({
        "status": "unsupported",
        "parsed": {"metric_name": "freight_value", "query_type": "metric_contribution"},
        "reason": "该能力尚未开放",
    })
    assert review["status"] == "approved"
    assert review["checks"]["terminal_unsupported"] is True


# ============================================================================
# metric_contribution（受控范围贡献度）
# ============================================================================


def test_scoped_contribution_product_category_summable(monkeypatch) -> None:
    """Summable metric with product_category contribution returns ok + contribution_result."""
    scope = SCOPE | {
        "seller_id": "seller-a",
        "metric_name": "gmv",
        "default_start_date": "2017-09-01",
        "default_end_date": "2017-10-31",
    }
    plan = _authorized_plan(MetricQueryPlan(query_type="metric_contribution", metric_name="gmv",
                           current_date="2017-10-01", baseline_date="2017-09-01",
                           dimension="product_category"), scope)

    monkeypatch.setattr("app.mcp.client.apply_trusted_time_range", lambda v, _q, **_kwargs: v)
    def fake_breakdown(_scope, metric, limit=10):
        current = _scope.default_start_date == date(2017, 10, 1)
        return {
            "metric_name": metric, "dimension": "product_category", "status": "ok",
            "items": (
                [{"dimension_value": "cat_a", "value": 120.0}, {"dimension_value": "cat_b", "value": 70.0}]
                if current else
                [{"dimension_value": "cat_a", "value": 100.0}, {"dimension_value": "cat_b", "value": 50.0}]
            ),
            "source_table": "scoped",
        }

    monkeypatch.setattr("app.mcp.client.breakdown_scoped_product_category", fake_breakdown)
    monkeypatch.setattr("app.mcp.client.compare_scoped_metric",
                        lambda _scope, m, **kw: {
                            "left_value": 190.0, "right_value": 150.0,
                            "delta_value": 40.0, "status": "ok",
                        })

    result = query_scoped_business_data("分析 GMV 品类贡献度", scope, plan=plan)
    assert result["result"]["status"] == "ok"
    contrib = result["result"]["contribution_result"]
    assert contrib is not None
    assert contrib["status"] == "ok"
    assert len(contrib["current_items"]) == 2
    assert len(contrib["baseline_items"]) == 2
    assert contrib["baseline_total"] == 150.0
    assert contrib["current_total"] == 190.0
    assert contrib["total_delta"] == 40.0
    assert contrib["method"] == "additive_change_v1"


def test_seller_order_count_product_category_contribution_uses_allocation(monkeypatch) -> None:
    """Distinct orders spanning categories are allocated 1/k, not summed."""
    scope = SCOPE | {
        "seller_id": "seller-a",
        "metric_name": "order_count",
        "allowed_metrics": ["order_count"],
        "allowed_dimensions": ["product_category"],
        "granularity": "month",
        "default_start_date": "2018-07-01",
        "default_end_date": "2018-08-31",
    }
    plan = _authorized_plan(MetricQueryPlan(
        query_type="metric_contribution",
        metric_name="order_count",
        dimension="product_category",
        granularity="month",
        baseline_start="2018-07-01",
        baseline_end="2018-07-31",
        current_start="2018-08-01",
        current_end="2018-08-31",
    ), scope)
    calls: list[tuple[object, ...]] = []

    def fake_allocation(scoped, current_start, current_end, baseline_start, baseline_end, *, limit=10):
        calls.append((current_start, current_end, baseline_start, baseline_end, limit))
        return {
            "metric_name": "order_count",
            "dimension": "product_category",
            "current_date": current_end.isoformat(),
            "baseline_date": baseline_end.isoformat(),
            "current_period": {"start_date": current_start.isoformat(), "end_date": current_end.isoformat()},
            "baseline_period": {"start_date": baseline_start.isoformat(), "end_date": baseline_end.isoformat()},
            "current_items": [{"dimension_value": "books", "value": 1.0, "share": 100.0}],
            "baseline_items": [{"dimension_value": "books", "value": 0.5, "share": 100.0}],
            "current_total": 1.0,
            "baseline_total": 0.5,
            "total_delta": 0.5,
            "source_table": "scoped",
            "status": "ok",
            "method": "proportional_overlap_v1",
            "allocation_method": "proportional_overlap_v1",
            "allocation_note": "每个订单按涉及品类数均分",
        }

    monkeypatch.setattr(
        "app.mcp.client.compare_scoped_order_count_allocation", fake_allocation,
    )

    result = query_scoped_business_data("比较7月和8月订单量的品类贡献", scope, plan=plan)

    assert result["result"]["status"] == "ok"
    assert result["tool_name"] == "compare_breakdown_by_dimension_period"
    assert result["result"]["contribution_result"]["method"] == "proportional_overlap_v1"
    assert len(calls) == 1
    assert calls[0][4] == 10


def test_scoped_contribution_rejects_non_additive(monkeypatch) -> None:
    """Ratio metrics (avg_order_value) should be rejected for contribution analysis."""
    scope = SCOPE | {
        "seller_id": "seller-a",
        "metric_name": "avg_order_value",
        "default_start_date": "2017-09-01",
        "default_end_date": "2017-10-31",
    }
    plan = _authorized_plan(MetricQueryPlan(query_type="metric_contribution", metric_name="avg_order_value",
                           current_date="2017-10-01", baseline_date="2017-09-01",
                           dimension="product_category"), scope)

    monkeypatch.setattr("app.mcp.client.apply_trusted_time_range", lambda v, _q, **_kwargs: v)

    result = query_scoped_business_data("分析客单价品类贡献度", scope, plan=plan)
    assert result["result"]["status"] == "unsupported"
    assert "比率/均值类指标" in result["result"]["reason"]


def test_scoped_contribution_fails_closed_for_unregistered_additive_capability(monkeypatch) -> None:
    """A sum metric still needs an explicit additive-attribution declaration."""
    metric_name = "seller_item_gross_value"
    scope = SCOPE | {
        "seller_id": "seller-a", "metric_name": metric_name,
        "default_start_date": "2017-09-01", "default_end_date": "2017-10-31",
    }
    plan = _authorized_plan(MetricQueryPlan(
        query_type="metric_contribution", metric_name=metric_name,
        current_date="2017-10-01", baseline_date="2017-09-01",
        dimension="product_category",
    ), scope)
    monkeypatch.setattr("app.mcp.client.apply_trusted_time_range", lambda value, *_a, **_k: value)

    result = query_scoped_business_data("分析店铺商品金额总额的品类贡献", scope, plan=plan)

    assert result["result"]["status"] == "unsupported"
    assert "未声明" in result["result"]["reason"]
    assert result["tool_name"] is None


def test_scoped_contribution_rejects_unsupported_dimension(monkeypatch) -> None:
    """Dimensions other than product_category should be rejected."""
    scope = SCOPE | {
        "seller_id": "seller-a",
        "metric_name": "gmv",
        "default_start_date": "2017-09-01",
        "default_end_date": "2017-10-31",
    }
    plan = _authorized_plan(MetricQueryPlan(query_type="metric_contribution", metric_name="gmv",
                           current_date="2017-10-01", baseline_date="2017-09-01",
                           dimension="payment_type"), scope)

    monkeypatch.setattr("app.mcp.client.apply_trusted_time_range", lambda v, _q, **_kwargs: v)

    result = query_scoped_business_data("按支付方式拆解贡献度", scope, plan=plan)
    assert result["result"]["status"] == "unsupported"
    assert "商品品类" in result["result"]["reason"]


def test_scoped_contribution_produces_correct_payload_shape(monkeypatch) -> None:
    """contribution_result payload must contain all fields that _contribution_from_evidence expects."""
    scope = SCOPE | {
        "seller_id": "seller-a",
        "metric_name": "gmv",
        "default_start_date": "2017-09-01",
        "default_end_date": "2017-10-31",
    }
    plan = _authorized_plan(MetricQueryPlan(query_type="metric_contribution", metric_name="gmv",
                           current_date="2017-10-01", baseline_date="2017-09-01",
                           dimension="product_category"), scope)

    monkeypatch.setattr("app.mcp.client.apply_trusted_time_range", lambda v, _q, **_kwargs: v)
    def fake_breakdown(_scope, m, limit=10):
        current = _scope.default_start_date == date(2017, 10, 1)
        return {
            "metric_name": m, "dimension": "product_category", "status": "ok",
            "items": [{"dimension_value": "cat_a", "value": 120.0 if current else 100.0}],
            "source_table": "scoped",
        }

    monkeypatch.setattr("app.mcp.client.breakdown_scoped_product_category", fake_breakdown)
    monkeypatch.setattr("app.mcp.client.compare_scoped_metric",
                        lambda _scope, m, **kw: {
                            "left_value": 130.0, "right_value": 110.0,
                            "delta_value": 20.0, "status": "ok",
                        })

    result = query_scoped_business_data("分析 GMV 品类贡献度", scope, plan=plan)
    contrib = result["result"]["contribution_result"]
    assert contrib["status"] == "ok"
    assert "metric_name" in contrib
    assert "dimension" in contrib
    assert "current_date" in contrib
    assert "baseline_date" in contrib
    assert "current_items" in contrib
    assert "baseline_items" in contrib
    assert contrib["baseline_total"] == 110.0
    assert contrib["current_total"] == 130.0
    assert "total_delta" in contrib
    assert contrib["method"] == "additive_change_v1"
    assert "source_table" in contrib


def test_seller_month_position_unfiltered_uses_independent_totals_and_contract_tool(monkeypatch):
    scope = SCOPE | {
        "seller_id": "seller-a", "metric_name": "gmv",
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["month_position"],
        "default_start_date": "2018-08-01", "default_end_date": "2018-08-31",
        "granularity": "month",
    }
    plan = _authorized_plan(MetricQueryPlan(
        query_type="metric_contribution", metric_name="gmv", dimension="month_position",
        current_date="2018-08-31", baseline_date="2018-08-01", granularity="month",
    ), scope)
    breakdown_calls = []

    def fake_month_position(scoped, metric, *, product_category_filter=None):
        current = scoped.default_start_date == date(2018, 8, 31)
        value = 120.0 if current else 100.0
        label = "月末(21-31日)" if current else "月初(1-10日)"
        breakdown_calls.append((scoped.default_start_date, product_category_filter))
        return {
            "metric_name": metric, "dimension": "month_position", "status": "ok",
            "items": [{"dimension_value": label, "value": value}],
            "population_total": value, "source_table": "scoped_month_position",
        }

    compare_calls = []

    def fake_compare(_scope, metric, **kwargs):
        compare_calls.append(kwargs)
        return {"left_value": 120.0, "right_value": 100.0, "delta_value": 20.0, "status": "ok"}

    monkeypatch.setattr("app.mcp.client.breakdown_scoped_month_position", fake_month_position)
    monkeypatch.setattr("app.mcp.client.compare_scoped_metric", fake_compare)

    result = query_scoped_business_data("比较8月1日和8月31日 GMV 月内位置贡献", scope, plan=plan)
    contribution = result["result"]["contribution_result"]

    assert result["tool_name"] == "compare_breakdown_by_dimension"
    assert result["arguments"]["metric_name"] == "gmv"
    assert result["arguments"]["current_date"] == "2018-08-31"
    assert result["arguments"]["baseline_date"] == "2018-08-01"
    assert result["arguments"]["dimension"] == "month_position"
    assert contribution["baseline_total"] == 100.0
    assert contribution["current_total"] == 120.0
    assert contribution["total_delta"] == 20.0
    assert contribution["method"] == "additive_change_v1"
    assert len(compare_calls) == 1
    assert len(breakdown_calls) == 2
    assert result["execution_metadata"]["controlled_handlers"] == [
        "breakdown_scoped_month_position", "compare_scoped_metric",
    ]
    from app.agents.critic_agent import review_data_query_result
    assert review_data_query_result(result["result"])["status"] == "approved"


def test_seller_month_position_filtered_uses_same_filter_total_without_unfiltered_compare(monkeypatch):
    scope = SCOPE | {
        "seller_id": "seller-a", "metric_name": "gmv",
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["month_position"],
        "default_start_date": "2018-08-01", "default_end_date": "2018-08-31",
        "granularity": "month",
    }
    plan = _authorized_plan(MetricQueryPlan(
        query_type="metric_contribution", metric_name="gmv", dimension="month_position",
        current_start="2018-08-21", current_end="2018-08-31",
        baseline_start="2018-08-01", baseline_end="2018-08-10",
        granularity="month", filters={"product_category": "furniture"},
    ), scope)
    seen_filters = []

    def fake_month_position(scoped, metric, *, product_category_filter=None):
        current = scoped.default_start_date == date(2018, 8, 21)
        value = 60.0 if current else 50.0
        seen_filters.append(product_category_filter)
        return {
            "metric_name": metric, "dimension": "month_position", "status": "ok",
            "items": [{"dimension_value": "月末(21-31日)" if current else "月初(1-10日)", "value": value}],
            "population_total": value, "source_table": "scoped_filtered_month_position",
        }

    monkeypatch.setattr("app.mcp.client.breakdown_scoped_month_position", fake_month_position)
    monkeypatch.setattr(
        "app.mcp.client.compare_scoped_metric",
        lambda *_args, **_kwargs: pytest.fail(
            "filtered month_position must not compare against an unfiltered seller total"
        ),
    )

    result = query_scoped_business_data("比较过滤品类的 GMV 月内位置贡献", scope, plan=plan)
    contribution = result["result"]["contribution_result"]

    assert result["tool_name"] == "compare_breakdown_by_dimension_period"
    assert result["arguments"]["current_start"] == "2018-08-21"
    assert result["arguments"]["baseline_start"] == "2018-08-01"
    assert result["execution_metadata"]["filter_applied"] == {"product_category": "furniture"}
    assert contribution["baseline_total"] == 50.0
    assert contribution["current_total"] == 60.0
    assert contribution["total_delta"] == 10.0
    assert seen_filters == ["furniture", "furniture"]
    from app.agents.critic_agent import review_data_query_result
    assert review_data_query_result(result["result"])["status"] == "approved"


@pytest.mark.parametrize("filtered", [False, True])
def test_seller_month_position_group_total_mismatch_fails_closed(monkeypatch, filtered):
    scope = SCOPE | {
        "seller_id": "seller-a", "metric_name": "gmv",
        "allowed_metrics": ["gmv"], "allowed_dimensions": ["month_position"],
        "default_start_date": "2018-08-01", "default_end_date": "2018-08-31",
        "granularity": "month",
    }
    updates = {
        "query_type": "metric_contribution", "metric_name": "gmv",
        "dimension": "month_position", "granularity": "month",
        "baseline_start": "2018-08-01", "baseline_end": "2018-08-10",
        "current_start": "2018-08-21", "current_end": "2018-08-31",
        "filters": {"product_category": "furniture"} if filtered else {},
    }
    if not filtered:
        updates.update({"baseline_start": None, "baseline_end": None,
                        "current_start": None, "current_end": None,
                        "baseline_date": "2018-08-01", "current_date": "2018-08-31"})
    plan = _authorized_plan(MetricQueryPlan(**updates), scope)
    monkeypatch.setattr(
        "app.mcp.client.breakdown_scoped_month_position",
        lambda scoped, metric, *, product_category_filter=None: {
            "items": [{"dimension_value": "月初(1-10日)", "value": 40.0}],
            "population_total": 50.0, "status": "ok", "source_table": "scoped",
        },
    )
    monkeypatch.setattr(
        "app.mcp.client.compare_scoped_metric",
        lambda *_args, **_kwargs: {
            "left_value": 50.0, "right_value": 50.0, "delta_value": 0.0, "status": "ok",
        },
    )

    result = query_scoped_business_data("比较月内位置 GMV 贡献", scope, plan=plan)

    assert result["result"]["status"] == "execution_failed"
    assert "分组与同口径总值无法对账" in result["result"]["reason"]
    assert result["tool_name"] is None
