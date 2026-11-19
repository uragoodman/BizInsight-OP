from __future__ import annotations

import socket

import pytest

from app.graph.nodes.answer import _make_changes_from_tool
from app.schemas.analysis_presentation import AnalysisPresentation


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(*args, **kwargs):
        raise AssertionError("network access is blocked in ChangeCard unit tests")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def _compare(metric: str, left: float | None, right: float | None, delta: float | None, ratio: float | None, index: int) -> dict:
    return {
        "tool_name": "query_scoped_metric_compare",
        "result": {"compare_result": {
            "metric_name": metric,
            "left_value": left,
            "right_value": right,
            "delta_value": delta,
            "delta_ratio": ratio,
            "left_date": "2026-02-02",
            "right_date": "2026-02-01",
            "status": "ok",
        }},
        "_test_index": index,
    }


def _evidence(item: dict, index: int) -> dict:
    return {
        "evidence_id": f"ev-{index}",
        "source_type": "mcp",
        "source_ref": item["tool_name"],
        "content": {"result": item["result"]},
    }


def _make_cards(items: list[dict]) -> list[dict]:
    bundles = [_evidence(item, index) for index, item in enumerate(items)]
    tool_result = {**items[0], "batch_results": items}
    cards, _ = _make_changes_from_tool(tool_result, bundles)
    return AnalysisPresentation.model_validate({"changes": cards}).model_dump(mode="json")["changes"]


def test_registry_units_flow_through_comparison_card_and_schema() -> None:
    rows = [
        _compare("freight_rate", 12.0, 10.0, 2.0, 20.0, 0),
        _compare("gmv", 120.0, 100.0, 20.0, 20.0, 1),
        _compare("order_count", 12.0, 10.0, 2.0, 20.0, 2),
        _compare("avg_order_value", 12.0, 10.0, 2.0, 20.0, 3),
    ]
    cards = _make_cards(rows)
    assert [(card["unit"], card["change_unit"], card["delta_ratio_unit"]) for card in cards] == [
        ("%", "percentage_points", "%"),
        ("BRL", "BRL", "%"),
        ("orders", "orders", "%"),
        ("BRL/order", "BRL/order", "%"),
    ]


def test_zero_null_and_zero_baseline_keep_registered_units() -> None:
    cards = _make_cards([
        _compare("freight_rate", 0.0, 0.0, 0.0, 0.0, 0),
        _compare("freight_rate", 5.0, 0.0, 5.0, None, 1),
    ])
    assert cards[0]["current_value"] == 0.0 and cards[0]["delta_value"] == 0.0
    assert cards[0]["delta_ratio"] == 0.0
    assert cards[1]["delta_ratio"] is None
    for card in cards:
        assert card["unit"] == "%"
        assert card["change_unit"] == "percentage_points"
        assert card["delta_ratio_unit"] == "%"
    missing = _make_cards([_compare("freight_rate", None, 0.0, None, None, 0)])
    assert missing == []


def test_unknown_registry_unit_is_null_and_legacy_payload_remains_valid() -> None:
    cards = _make_cards([_compare("unknown_metric", 2.0, 1.0, 1.0, 100.0, 0)])
    assert cards[0]["unit"] is None
    assert cards[0]["change_unit"] is None
    # This is the compare operation's known relative-ratio contract, not a
    # guessed metric unit.
    assert cards[0]["delta_ratio_unit"] == "%"

    legacy = AnalysisPresentation.model_validate({"changes": [{
        "metric_name": "gmv", "current_value": 2.0, "baseline_value": 1.0,
        "delta_value": 1.0, "delta_ratio": 100.0,
    }]}).model_dump(mode="json")["changes"][0]
    assert legacy["unit"] is None
    assert legacy["change_unit"] is None
    assert legacy["delta_ratio_unit"] is None
