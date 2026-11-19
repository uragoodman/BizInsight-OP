"""Proportional order-count contribution tests (no database required)."""

from __future__ import annotations

import pytest

from app.data.repositories.order_allocation import (
    RESIDUAL_LABEL,
    allocated_items,
    paired_allocated_items,
    require_reconciled,
)
from app.graph.nodes import answer as answer_node
from app.metrics.registry import get_registry


def test_one_order_two_categories_splits_equally() -> None:
    items = allocated_items([("books", 0.5), ("health", 0.5)], total=1.0)
    assert {item["dimension_value"]: item["value"] for item in items} == {
        "books": 0.5, "health": 0.5,
    }
    assert sum(item["value"] for item in items) == 1.0


def test_three_category_order_allocates_one_third_each() -> None:
    rows = [("a", 1 / 3), ("b", 1 / 3), ("c", 1 / 3)]
    require_reconciled(rows, 1.0, label="test")
    items = allocated_items(rows, total=1.0)
    assert [item["value"] for item in items] == [1 / 3, 1 / 3, 1 / 3]
    assert sum(item["value"] for item in items) == 1.0


def test_allocation_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="does not reconcile"):
        require_reconciled([("books", 1.0), ("health", 1.0)], 1.0, label="test")


def test_pairing_folds_tail_into_explicit_residual() -> None:
    current, baseline = paired_allocated_items(
        [("a", 2.0), ("b", 1.0), ("c", 1.0)],
        [("a", 1.0), ("b", 1.0), ("c", 1.0)],
        current_total=4.0, baseline_total=3.0, limit=2,
    )
    assert [item["dimension_value"] for item in current] == ["a", "b", RESIDUAL_LABEL]
    assert current[2]["value"] == 1.0
    assert baseline[2]["value"] == 1.0
    assert sum(item["value"] for item in current) == 4.0
    assert sum(item["value"] for item in baseline) == 3.0


def test_registry_declares_proportional_method_for_order_count() -> None:
    definition = get_registry().get("order_count")
    for scope in ("platform", "seller"):
        capability = definition.attribution.for_scope(scope)
        assert "proportional_overlap_v1" in capability.method_ids
        assert "product_category" in capability.dimensions


def _projection_state() -> tuple[dict, dict, dict]:
    definition = get_registry().get("order_count")
    comparison = {
        "kind": "period_pair",
        "baseline": {"start_date": "2018-07-01", "end_date": "2018-07-31"},
        "current": {"start_date": "2018-08-01", "end_date": "2018-08-31"},
    }
    contract = {
        "metric_id": "order_count",
        "dimension": "product_category",
        "comparison": comparison,
    }
    payload = {
        "metric_name": "order_count",
        "dimension": "product_category",
        "current_date": "2018-08-31",
        "baseline_date": "2018-07-31",
        "current_period": comparison["current"],
        "baseline_period": comparison["baseline"],
        "current_items": [
            {"dimension_value": "books", "value": 0.5, "share": 50.0},
            {"dimension_value": "health", "value": 0.5, "share": 50.0},
        ],
        "baseline_items": [
            {"dimension_value": "books", "value": 1.0, "share": 100.0},
        ],
        "current_total": 1.0,
        "baseline_total": 1.0,
        "total_delta": 0.0,
        "method": "proportional_overlap_v1",
        "allocation_method": "proportional_overlap_v1",
        "allocation_note": "每个订单按涉及品类数均分",
        "status": "ok",
    }
    bundle = {
        "evidence_id": "ev-proportional",
        "trusted": True,
        "content": {
            "result": {
                "status": "ok",
                "parsed": {
                    "query_type": "metric_contribution",
                    "metric_name": "order_count",
                    "dimension": "product_category",
                },
                "contribution_result": payload,
            },
        },
    }
    state = {
        "evidence_bundles": [bundle],
        "errors": [],
    }
    return state, contract, bundle["content"]["result"]["contribution_result"]


def test_proportional_projection_builds_allocated_card() -> None:
    state, contract, _ = _projection_state()
    definition = get_registry().get("order_count")
    cards = answer_node._validated_proportional_request_cards(
        state, contract, {"ev-proportional"}, definition,
    )
    assert len(cards) == 1
    card = cards[0]
    assert card["method"] == "proportional_overlap_v1"
    assert card["aggregation"] == "registered_proportional_overlap_contribution"
    assert card["residual_value"] == 0.0
    text = answer_node._dimension_contribution_text(card)
    assert "分摊" in text and "订单等价量" in text


def test_proportional_projection_rejects_unreconciled_groups() -> None:
    state, contract, payload = _projection_state()
    payload["current_items"] = [{"dimension_value": "books", "value": 0.4}]
    definition = get_registry().get("order_count")
    assert answer_node._validated_proportional_request_cards(
        state, contract, {"ev-proportional"}, definition,
    ) == []
    assert state["contribution_projection"]["status"] == "failed"
