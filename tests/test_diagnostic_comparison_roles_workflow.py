"""Normal graph acceptance for explicit dates on a governed dimension task."""
from __future__ import annotations

from test_platform_metric_trend_acceptance import isolated_trend_db  # noqa: F401
from test_registered_numeric_decomposition_intent import _install_offline_workflow


def test_normal_graph_keeps_requested_dates_in_investigation_and_contribution(monkeypatch, isolated_trend_db):
    invoke, state, statements, _model_calls, external_attempts = _install_offline_workflow(
        monkeypatch, isolated_trend_db, "order_dimension",
    )
    response = invoke(
        "请解释平台已交付订单量从 2024-05-03 到 2024-05-04 的变化，"
        "先核对趋势，再按 product_category 展示统计贡献；统计贡献不等于业务因果。",
        "critic_llm_review_off",
    )
    target = state["investigation_state"]["target"]
    assert target["baseline_start"] == "2024-05-03"
    assert target["current_start"] == "2024-05-04"
    contributions = [item for item in state["investigation_state"]["evidence"]
                     if item["evidence_type"] == "dimension_contribution"]
    assert contributions
    for evidence in contributions:
        comparison = evidence["comparison"]["comparison"]
        assert comparison["baseline_date"] == "2024-05-03"
        assert comparison["current_date"] == "2024-05-04"
        values = evidence["values"]
        assert values["total_delta"] == -4
        components = {item["dimension_value"]: item["delta_value"] for item in values["components"]}
        assert components == {"A": -4, "B": 0}
    assert statements
    assert external_attempts == []
    assert "???" not in response["answer"]
    assert not any(row.get("code") == "diagnostic_investigation_target_changed" for row in state["errors"])
