from datetime import date

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from app.agents.contracts import MetricQueryPlan
from app.agents.critic_agent import review_data_query_result
from app.analysis.metrics import summarize_metric_result
from app.data.repositories import daily_metrics
from app.graph.nodes.answer import (
    _has_incomplete_comparison,
    _make_changes_from_tool,
    _summarize_metric_query,
    generate_answer,
)
from app.graph.nodes.critic import critic
from app.graph.workflow import _route_from_critic
from app.mcp.client import _result_from_payload
from app.metrics.registry import get_registry


def _comparison_payload(left, right, *, status=None):
    complete = left is not None and right is not None
    return {
        "metric_name": "gmv",
        "left_date": "2024-01-01",
        "right_date": "2024-01-02",
        "left_value": left,
        "right_value": right,
        "delta_value": round(right - left, 2) if complete else None,
        "delta_ratio": round((right - left) * 100 / left, 2) if complete and left != 0 else None,
        "source_table": "analytics.daily_metrics",
        "status": status or ("ok" if complete else "not_found"),
    }


@pytest.mark.parametrize(
    "left,right,expected_status,expected_delta,expected_ratio",
    [
        (10.0, 15.0, "ok", 5.0, 50.0),
        (None, 15.0, "not_found", None, None),
        (10.0, None, "not_found", None, None),
        (None, None, "not_found", None, None),
        (0.0, 15.0, "ok", 15.0, None),
        # A real zero repeat-customer count is a value, not a missing operand.
        (0.0, 1.0, "ok", 1.0, None),
    ],
)
def test_public_platform_compare_requires_both_values(
    monkeypatch, left, right, expected_status, expected_delta, expected_ratio
):
    values = {date(2024, 1, 1): left, date(2024, 1, 2): right}

    def fake_query_metric(metric_name, metric_date):
        normalized = daily_metrics._normalize_date(metric_date)
        value = values[normalized]
        return {
            "metric_name": metric_name,
            "metric_date": normalized.isoformat(),
            "value": value,
            "source_table": "analytics.daily_metrics",
            "status": "ok" if value is not None else "not_found",
        }

    monkeypatch.setattr(daily_metrics, "query_metric", fake_query_metric)
    result = get_registry().execute(
        "gmv", operation="compare", scope="platform",
        left_date=date(2024, 1, 1), right_date=date(2024, 1, 2),
    )
    assert result["status"] == expected_status
    assert result["left_value"] == left
    assert result["right_value"] == right
    assert result["delta_value"] == expected_delta
    assert result["delta_ratio"] == expected_ratio


@pytest.mark.parametrize("left,right", [(None, 4.0), (4.0, None), (None, None)])
def test_incomplete_compare_survives_mcp_wrapper_and_is_not_approved(left, right):
    payload = _comparison_payload(left, right)
    plan = MetricQueryPlan(
        query_type="metric_compare", metric_name="gmv",
        left_date="2024-01-01", right_date="2024-01-02",
    )
    wrapped = _result_from_payload(plan, payload).model_dump()
    assert wrapped["status"] == "not_found"
    assert wrapped["compare_result"]["status"] == "not_found"
    assert wrapped["compare_result"]["left_value"] == left
    assert wrapped["compare_result"]["right_value"] == right

    review = review_data_query_result(wrapped)
    assert review["status"] == "incomplete"
    assert review["checks"]["has_result"] is False
    assert review["checks"]["comparison_complete"] is False

    # Terminal incomplete result: persist an abstention instead of requerying.
    assert _route_from_critic({
        "critic_verdict": "rejected",
        "critic_result": {"data_review": review},
    }) == "persist_trace"


def test_incomplete_compare_is_explained_without_change_claim_or_card():
    inner = {
        "status": "not_found",
        "compare_result": _comparison_payload(12.0, None),
    }
    tool_result = {"status": "not_found", "result": inner}

    answer = _summarize_metric_query(tool_result)
    assert "无法完成比较" in answer
    assert "2024-01-01=12.00" in answer
    assert "未计算或解释变化" in answer
    assert "差值=" not in answer
    assert _has_incomplete_comparison(tool_result) is True
    assert _make_changes_from_tool(tool_result, []) == ([], [])

    answer_state = generate_answer({"tool_result": tool_result, "errors": []})
    assert answer_state["analysis_presentation"]["status"] == "failed"
    assert answer_state["analysis_presentation"]["changes"] == []
    assert "无法完成比较" in answer_state["answer"]

    summary = summarize_metric_result("compare", {"result": inner}, "")
    assert "无法完成比较" in summary
    assert "差值=" not in summary


def test_critic_rejects_incomplete_compare_without_retry_target():
    wrapped = {
        "status": "not_found",
        "parsed": {"query_type": "metric_compare", "metric_name": "gmv"},
        "compare_result": _comparison_payload(None, 4.0),
    }
    state = critic({
        "task_type": "metric_query",
        "tool_result": {"result": wrapped},
        "claims": [],
        "evidence_bundles": [],
        "errors": [],
    })
    assert state["critic_verdict"] == "rejected"
    assert state["critic_revision_target"] is None
    assert state["critic_result"]["data_review"]["status"] == "incomplete"
    assert _route_from_critic(state) == "persist_trace"


def test_complete_zero_baseline_is_approved_with_null_ratio():
    inner = {
        "status": "ok",
        "parsed": {"query_type": "metric_compare", "metric_name": "gmv"},
        "compare_result": _comparison_payload(0.0, 5.0),
    }
    review = review_data_query_result(inner)
    assert review["status"] == "approved"
    assert inner["compare_result"]["delta_value"] == 5.0
    assert inner["compare_result"]["delta_ratio"] is None


def test_repeat_customer_count_zero_is_a_valid_comparison_value(monkeypatch):
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    with engine.begin() as conn:
        conn.execute(text("ATTACH DATABASE ':memory:' AS analytics"))
        conn.execute(text("""
            CREATE TABLE analytics.fact_orders (
                order_id TEXT PRIMARY KEY, purchase_date DATE NOT NULL,
                customer_unique_id TEXT, order_status TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE VIEW analytics.v_delivered_orders AS
            SELECT * FROM analytics.fact_orders WHERE order_status = 'delivered'
        """))
        conn.execute(text("""
            INSERT INTO analytics.fact_orders VALUES
                ('prior', '2024-01-01', 'returning', 'delivered'),
                ('first', '2024-01-02', 'first_timer', 'delivered'),
                ('return', '2024-01-03', 'returning', 'delivered')
        """))
    monkeypatch.setattr(daily_metrics, "engine", engine)
    # Jan 2 has one current customer and no repeat customer; Jan 3 has one repeat.
    assert daily_metrics.query_metric("customer_count", date(2024, 1, 2))["value"] == 1.0
    assert daily_metrics.query_metric("repeat_customer_count", date(2024, 1, 2))["value"] == 0.0
    result = daily_metrics.compare_metric(
        "repeat_customer_count", date(2024, 1, 2), date(2024, 1, 3)
    )
    engine.dispose()
    assert result["status"] == "ok"
    assert result["left_value"] == 0.0
    assert result["delta_value"] == 1.0
