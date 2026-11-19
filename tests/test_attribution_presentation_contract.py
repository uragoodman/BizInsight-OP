from __future__ import annotations

from app.analysis.semantic import list_metric_definitions
from app.graph.nodes.answer import _make_contributions
from app.graph.nodes.root_cause import _contribution_from_evidence
from app.metrics.registry import get_registry
from app.schemas.analysis_presentation import AnalysisPresentation


def test_chat_contribution_preserves_all_groups_and_reconciliation() -> None:
    root_cause_evidence = [{
        "purpose": "dimension_contribution:product_category",
        "result": {"result": {"contribution_result": {
            "status": "ok",
            "metric_name": "gmv",
            "current_date": "2026-02-08",
            "baseline_date": "2026-02-01",
            "total_delta": 30.0,
            "current_items": [
                {"dimension_value": "A", "value": 70.0},
                {"dimension_value": "B", "value": 40.0},
            ],
            "baseline_items": [
                {"dimension_value": "A", "value": 50.0},
                {"dimension_value": "B", "value": 30.0},
            ],
        }}}},
    ]
    findings = _contribution_from_evidence(root_cause_evidence)
    assert len(findings) == 1
    assert len(findings[0]["items"]) == 2
    assert findings[0]["total_delta"] == 30.0
    assert findings[0]["residual_value"] == 0.0

    cards, _ = _make_contributions(
        {"contribution": findings},
        [{"evidence_id": "ev-root", "source_type": "root_cause", "source_ref": "root_cause:gmv"}],
    )
    presentation = AnalysisPresentation.model_validate({"contributions": cards})
    card = presentation.contributions[0]
    assert card.metric_name == "gmv"
    assert card.metric_definition
    assert card.unit
    assert card.analysis_nature == "descriptive_contribution"
    assert [item.dimension_value for item in card.items] == ["A", "B"]
    assert sum(item.delta_value for item in card.items) == card.total_delta


def test_metric_catalog_exposes_attribution_state_for_both_scopes() -> None:
    metrics = {item["metric_name"]: item for item in list_metric_definitions("platform")}
    assert len(metrics) == len(get_registry().list())
    assert all(
        set(metric["attribution"]["support_by_scope"]) == {"platform", "seller"}
        for metric in metrics.values()
    )
    gmv = metrics["gmv"]["attribution"]["support_by_scope"]
    assert gmv["platform"]["status"] == "supported"
    assert gmv["seller"]["status"] == "supported"
    assert "additive_change_v1" in gmv["seller"]["methods"]
    unsupported = metrics["bad_review_rate"]["attribution"]["support_by_scope"]
    assert unsupported["platform"]["status"] == "unsupported"
    assert unsupported["platform"]["methods"] == []
    assert unsupported["platform"]["limitation"]
