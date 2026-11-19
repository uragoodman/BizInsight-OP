"""阶段 4 展示层：批量结果与证据绑定回归测试。"""

from app.graph.nodes import answer as answer_node
from app.graph.nodes.answer import (
    _make_changes_from_tool,
    _make_facts_from_tool,
    _scope_sidecars_enabled,
)


def _metric_result(metric_name: str, value: float) -> dict:
    return {
        "tool_name": "query_scoped_metric",
        "result": {
            "metric_result": {
                "metric_name": metric_name,
                "value": value,
                "metric_date": "2018-08-01~2018-08-31",
            },
        },
    }


def _compare_result(metric_name: str, left_value: float, right_value: float) -> dict:
    return {
        "tool_name": "query_scoped_metric_compare",
        "result": {
            "compare_result": {
                "status": "ok",
                "metric_name": metric_name,
                "left_value": left_value,
                "right_value": right_value,
                "delta_value": left_value - right_value,
                "delta_ratio": (left_value - right_value) / right_value,
                "left_date": "2018-08-01",
                "right_date": "2018-07-01",
            },
        },
    }


def _evidence(evidence_id: str, item: dict) -> dict:
    return {
        "evidence_id": evidence_id,
        "source_type": "mcp",
        "source_ref": item["tool_name"],
        "content": {"result": item["result"]},
    }


def test_batch_fact_cards_are_not_duplicated_and_keep_per_metric_evidence() -> None:
    first = _metric_result("gmv", 100.0)
    second = _metric_result("freight_value", 20.0)
    tool_result = {**first, "batch_results": [first, second]}

    facts, evidence_ids = _make_facts_from_tool(
        tool_result,
        [_evidence("ev_gmv", first), _evidence("ev_freight", second)],
    )

    assert [fact["metric_name"] for fact in facts] == ["gmv", "freight_value"]
    assert [fact["evidence_ids"] for fact in facts] == [["ev_gmv"], ["ev_freight"]]
    assert evidence_ids == ["ev_gmv", "ev_freight"]


def test_batch_change_cards_are_not_duplicated_and_keep_per_metric_evidence() -> None:
    first = _compare_result("gmv", 100.0, 80.0)
    second = _compare_result("freight_value", 20.0, 25.0)
    tool_result = {**first, "batch_results": [first, second]}

    changes, evidence_ids = _make_changes_from_tool(
        tool_result,
        [_evidence("ev_gmv_compare", first), _evidence("ev_freight_compare", second)],
    )

    assert [change["metric_name"] for change in changes] == ["gmv", "freight_value"]
    assert [change["evidence_ids"] for change in changes] == [
        ["ev_gmv_compare"],
        ["ev_freight_compare"],
    ]
    assert evidence_ids == ["ev_gmv_compare", "ev_freight_compare"]


def test_explorer_fact_uses_data_explorer_source_reference() -> None:
    tool_result = {
        "result": {
            "explorer_result": {
                "context": {"seller_id": "seller-a", "start_date": "2018-08-01", "end_date": "2018-08-31"},
                "summary": {"metric_name": "gmv", "value": 100.0, "unit": "BRL"},
            },
        },
    }
    bundles = [{
        "evidence_id": "ev_explorer",
        "source_type": "explorer",
        "source_ref": "data-explorer",
        "content": {},
    }]

    facts, evidence_ids = _make_facts_from_tool(tool_result, bundles)

    assert facts[0]["evidence_ids"] == ["ev_explorer"]
    assert evidence_ids == ["ev_explorer"]


def test_trend_fact_uses_latest_bucket_instead_of_labeling_it_as_full_period() -> None:
    tool_result = {
        "tool_name": "query_scoped_trend",
        "result": {
            "trend_result": {
                "metric_name": "gmv",
                "start_date": "2018-08-01",
                "end_date": "2018-08-31",
                "granularity": "day",
                "points": [
                    {"bucket": "2018-08-04", "value": 108.97},
                    {"bucket": "2018-08-18", "value": 107.29},
                    {"bucket": "2018-08-22", "value": 73.76},
                ],
            },
        },
    }

    facts, _ = _make_facts_from_tool(tool_result, [])

    assert facts[0]["value"] == 73.76
    assert facts[0]["date_or_range"] == "2018-08-22"


def test_scope_sidecars_are_skipped_for_bound_metric_and_causal_requests() -> None:
    for operation_intent in (
        "ordinary_metric_query",
        "business_causal_investigation",
        "dimensional_statistical_contribution",
        "registered_numeric_decomposition",
    ):
        state = {
            "context_compilation": {
                "request_contract": {"operation_intent": operation_intent},
            },
        }
        assert _scope_sidecars_enabled(state, True) is False

    assert _scope_sidecars_enabled({"context_compilation": {}}, True) is True
    assert _scope_sidecars_enabled({}, False) is False


def test_generate_answer_adds_sparse_daily_limit_and_does_not_fetch_scope_sidecars(monkeypatch) -> None:
    from app.data.repositories import explorer, joint_analysis, review_text, risk_indicator

    sidecar_calls: list[str] = []
    for module, names in (
        (explorer, ("query_scoped_order_funnel",)),
        (joint_analysis, ("query_joint_cluster",)),
        (review_text, ("query_review_comment_stats", "query_review_samples")),
        (risk_indicator, ("query_risk_score",)),
    ):
        for name in names:
            monkeypatch.setattr(
                module,
                name,
                lambda *_args, _name=name, **_kwargs: sidecar_calls.append(_name),
            )
    monkeypatch.setattr(
        answer_node,
        "_grounded_narrative",
        lambda _state, _evidence, fallback: (fallback, "template_fallback"),
    )

    result = answer_node.generate_answer({
        "task_type": "metric_query",
        "user_message": "只总结趋势数据，并说明稀疏日期不能支持哪些结论。",
        "context_compilation": {"operation_intent": {"kind": "ordinary_metric_query"}},
        "use_current_explorer_context": True,
        "controlled_analysis_scope": {
            "seller_id": "seller-a",
            "allowed_metrics": ["gmv"],
            "allowed_dimensions": [],
            "default_start_date": "2018-08-01",
            "default_end_date": "2018-08-31",
            "metric_name": "gmv",
            "granularity": "day",
        },
        "tool_result": {
            "tool_name": "query_scoped_trend",
            "result": {
                "status": "ok",
                "trend_result": {
                    "metric_name": "gmv",
                    "start_date": "2018-08-01",
                    "end_date": "2018-08-31",
                    "granularity": "day",
                    "unit": "BRL",
                    "source_table": "analytics.daily_metrics",
                    "points": [
                        {"bucket": "2018-08-04", "value": 108.97},
                        {"bucket": "2018-08-18", "value": 107.29},
                        {"bucket": "2018-08-22", "value": 73.76},
                    ],
                },
            },
        },
        "errors": [],
        "evidence_bundles": [],
        "analysis_memories": [],
        "memory_enabled": False,
        "node_history": [],
    })

    limitation = "仅返回 3 个有数值的日期"
    assert limitation in result["answer"]
    assert any(limitation in item for item in result["analysis_presentation"]["limitations"])
    assert result["analysis_presentation"]["facts"][0]["date_or_range"] == "2018-08-22"
    assert sidecar_calls == []
    assert result["analysis_presentation"]["joint_clusters"] == []
    assert result["analysis_presentation"]["review_samples"] is None
