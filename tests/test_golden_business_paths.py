"""Regression coverage for the failing report and payment-contribution Golden cases."""

from __future__ import annotations

import pytest

from app.graph.nodes.data_query import data_query
from app.graph.nodes.report import report
from evaluation.fixtures import offline_dependencies


@pytest.mark.parametrize(
    ("query", "expected_type"),
    [
        ("生成本周经营分析报告", "weekly"),
        ("生成月度经营分析报告", "monthly"),
    ],
)
def test_report_node_builds_structured_artifact_from_offline_dependencies(query, expected_type) -> None:
    with offline_dependencies():
        state = report({
            "user_message": query,
            "tool_result": {},
            "metric_context": "GMV is gross merchandise value.",
            "audit_actor": "golden-reviewer",
            "errors": [],
            "evidence_bundles": [],
            "node_history": [],
            "_revision_counts": {},
        })

    assert state.get("periodic_report")
    assert state["periodic_report"]["report_type"] == expected_type
    assert state.get("report_id") and state.get("report_version")
    assert not any(item.get("code") == "periodic_report_unavailable" for item in state["errors"])


def test_exact_gmv_payment_contribution_uses_metric_contribution_path() -> None:
    query = "拆解 2018-08-10 与 2018-08-09 的 GMV 按支付方式贡献度"
    with offline_dependencies():
        state = data_query({"task_type": "metric_query", "user_message": query, "trace_id": "golden-payment-regression"})

    assert state["metric_query_batch_plan"]["decision"]["query_type"] == "metric_contribution"
    assert state["metric_query_batch_plan"]["plans"][0]["dimension"] == "payment_type"
    assert state["tool_result"]["tool_name"] == "compare_breakdown_by_dimension"
    assert state["tool_result"]["result"]["status"] == "ok"
    assert state["tool_result"]["result"].get("contribution_result")


def test_directional_delay_root_cause_bypasses_broad_correlation_template() -> None:
    query = (
        "2018-08-25 GMV 下降了，延迟送达率也升高了，到底为什么？"
    )
    with offline_dependencies():
        state = data_query({
            "task_type": "anomaly_analysis", "user_message": query,
            "trace_id": "golden-delay-rootcause-routing",
        })

    assert state["tool_result"]["tool_name"] == "query_metric_trend"
    assert state["tool_calls"][0]["tool_name"] == "query_metric_trend"
    assert state["tool_result"]["result"]["trend_result"]["metric_name"] == "gmv"
    assert state["tool_result"]["result"]["trend_result"]["points"][-1]["value"] == 500.0
