"""Period-pair additive contribution projection tests (no database required)."""

from __future__ import annotations

from app.analysis.contribution_capabilities import registered_period_contribution_method
from app.graph.nodes import answer as answer_node
from app.metrics.registry import get_registry


def _method(metric_id: str, dimension: str, scope: str = "platform", granularity: str = "month") -> str | None:
    return registered_period_contribution_method(
        get_registry().get(metric_id),
        metric_id=metric_id,
        dimension=dimension,
        scope=scope,
        granularity=granularity,
    )


def test_gmv_category_month_uses_additive_period_method() -> None:
    assert _method("gmv", "product_category") == "additive_change_v1"
    assert _method("freight_value", "product_category") == "additive_change_v1"


def test_gmv_category_period_method_stays_scope_and_granularity_bound() -> None:
    assert _method("gmv", "product_category", scope="seller") is None
    assert _method("gmv", "product_category", granularity="week") is None
    assert _method("gmv", "customer_city") is None


def _period_additive_state():
    comparison = {
        "kind": "period_pair",
        "baseline": {"start_date": "2018-07-01", "end_date": "2018-07-31"},
        "current": {"start_date": "2018-08-01", "end_date": "2018-08-31"},
        "granularity": "month",
    }
    contract = {
        "metric_id": "gmv",
        "dimension": "product_category",
        "comparison": comparison,
    }
    payload = {
        "metric_name": "gmv",
        "dimension": "product_category",
        "baseline_period": comparison["baseline"],
        "current_period": comparison["current"],
        "baseline_items": [
            {"dimension_value": "books", "value": 100.0, "share": 100.0},
        ],
        "current_items": [
            {"dimension_value": "books", "value": 120.0, "share": 80.0},
            {"dimension_value": "health", "value": 30.0, "share": 20.0},
        ],
        "baseline_total": 100.0,
        "current_total": 150.0,
        "total_delta": 50.0,
        "status": "ok",
        "method": "additive_change_v1",
    }
    bundle = {
        "evidence_id": "ev-gmv-period",
        "trusted": True,
        "content": {
            "result": {
                "status": "ok",
                "parsed": {
                    "query_type": "metric_contribution",
                    "metric_name": "gmv",
                    "dimension": "product_category",
                    "baseline_start": "2018-07-01",
                    "baseline_end": "2018-07-31",
                    "current_start": "2018-08-01",
                    "current_end": "2018-08-31",
                    "granularity": "month",
                },
                "contribution_result": payload,
            },
        },
    }
    state = {
        "evidence_bundles": [bundle],
        "errors": [],
        "controlled_analysis_scope": {},
    }
    return state, contract, payload


def test_gmv_period_additive_projection_keeps_full_periods() -> None:
    state, contract, _ = _period_additive_state()
    cards = answer_node._validated_additive_request_cards(
        state, contract, {"ev-gmv-period"}, get_registry().get("gmv"),
    )
    assert len(cards) == 1
    card = cards[0]
    assert card["method"] == "additive_change_v1"
    assert card["baseline_period"] == {
        "start_date": "2018-07-01", "end_date": "2018-07-31", "granularity": "month",
    }
    assert card["current_period"] == {
        "start_date": "2018-08-01", "end_date": "2018-08-31", "granularity": "month",
    }
    assert card["total_delta"] == 50.0
    assert card["residual_explainable"] is True


def test_gmv_period_additive_projection_rejects_unreconciled_delta() -> None:
    state, contract, payload = _period_additive_state()
    payload["total_delta"] = 40.0
    assert answer_node._validated_additive_request_cards(
        state, contract, {"ev-gmv-period"}, get_registry().get("gmv"),
    ) == []
    assert state["contribution_projection"]["status"] == "failed"
