"""P2 tests: deterministic dimension-role classification."""

from __future__ import annotations

import json
from pathlib import Path

from app.context.compiler import compile_turn_context
from app.context.dimension_intent import RoleAssignment, aggregate_roles
from evaluation.dimension_role_audit import _load_cases, _state_for

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "datasets" / "dimension_role_cases_v1.jsonl"
P2_BASELINE = ROOT / "tests" / "fixtures" / "p2_contract_hashes.json"

SELLER_SCOPE = {
    "seller_id": "seller-1",
    "default_start_date": "2018-08-01",
    "default_end_date": "2018-08-31",
    "metric_name": "gmv",
    "granularity": "day",
    "allowed_metrics": ["gmv", "order_count", "freight_value"],
    "allowed_dimensions": ["product_category", "payment_type", "month_position"],
}


def _compile(question: str, *, scope: dict | None = None, previous=None) -> dict:
    state = {
        "user_message": question,
        "trace_id": "p2-test",
        "preference_context": {"source_needs": [], "candidate_analysis_plan": None},
    }
    if scope:
        state["controlled_analysis_scope"] = scope
        state["use_current_explorer_context"] = True
    if previous:
        state["previous_turn_context"] = previous
    return compile_turn_context(state)["context_compilation"]


def _roles(compiled: dict) -> dict[str, str]:
    assignments = [
        RoleAssignment(
            dimension=item["dimension"],
            start=item["start"],
            end=item["end"],
            role=item["role"],
            evidence=tuple(item.get("evidence") or ()),
            conflict=bool(item.get("conflict")),
        )
        for item in compiled.get("dimension_roles") or []
    ]
    return aggregate_roles(assignments)


def _core_cases() -> list[dict]:
    return [case for case in _load_cases(DATASET) if case["tier"] == "core"]


def test_original_question_uses_reference_role_for_seller() -> None:
    case = next(item for item in _core_cases() if item["case_id"] == "core-01")
    compiled = compile_turn_context(_state_for(case))["context_compilation"]
    roles = _roles(compiled)
    assert roles["seller"] == "reference"
    assert roles["product_category"] == "grouping"
    assert compiled.get("clarification_reason") == "date_outside_locked_scope"
    assert compiled["known_slots"]["dimension"]["value"] == "product_category"


def test_scope_reference_is_not_a_dimension() -> None:
    compiled = _compile("店铺范围内的 GMV", scope=SELLER_SCOPE)
    assert _roles(compiled) == {"seller": "reference"}
    assert "dimension" not in compiled["known_slots"]
    assert compiled["needs_clarification"] is False


def test_region_phrase_is_filter_not_grouping() -> None:
    compiled = _compile("华东地区的 GMV")
    assert _roles(compiled).get("customer_state") == "filter"
    assert "dimension" not in compiled["known_slots"]
    assert compiled["clarification_reason"] == "region_filter_not_supported"


def test_metric_noun_is_reference_not_dimension() -> None:
    compiled = _compile("平均每单商品数的统计贡献")
    assert _roles(compiled).get("product") == "reference"
    assert "dimension" not in compiled["known_slots"]
    assert compiled["clarification_reason"] == "dimension_required"


def test_definition_intent_does_not_create_grouping() -> None:
    compiled = _compile("GMV 按地区的口径是什么")
    assert _roles(compiled).get("customer_state") == "reference"
    assert "dimension" not in compiled["known_slots"]
    assert compiled["needs_clarification"] is False


def test_seller_grouping_still_fails_closed() -> None:
    compiled = _compile("按店铺拆解 GMV", scope=SELLER_SCOPE)
    assert _roles(compiled).get("seller") == "grouping"
    assert compiled["clarification_reason"] == "dimension_not_registered"


def test_two_grouping_dimensions_still_clarify() -> None:
    compiled = _compile("按店铺和商品品类拆解订单量", scope=SELLER_SCOPE)
    assert compiled["clarification_reason"] == "ambiguous_dimension"
    assert set(compiled["dimension_candidates"]) == {"seller", "product_category"}


def test_grouping_and_scope_cues_conflict() -> None:
    compiled = _compile("按店铺范围的 GMV", scope=SELLER_SCOPE)
    assert compiled["clarification_reason"] == "role_conflict"
    assert "dimension" not in compiled["known_slots"]


def test_platform_seller_grouping_is_not_silent() -> None:
    compiled = _compile("各店铺的 GMV")
    assert _roles(compiled).get("seller") == "grouping"
    assert compiled["clarification_reason"] == "dimension_not_registered"


def test_registered_seller_dimension_preserves_grouping_role(monkeypatch) -> None:
    from app.core.config import Settings

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(seller_dimension_enabled=True),
    )
    compiled = _compile("按店铺拆解 GMV")
    assert _roles(compiled).get("seller") == "grouping"


def test_seller_filter_role_does_not_become_a_grouping(monkeypatch) -> None:
    from app.core.config import Settings

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(seller_dimension_enabled=True),
    )
    compiled = _compile("只看店铺 seller-1 的 GMV")
    assert _roles(compiled).get("seller") == "filter"
    assert "dimension" not in compiled.get("known_slots", {})


def test_region_comparison_is_not_a_grouping_dimension() -> None:
    compiled = _compile("比较华东和华南的 GMV")
    assert "dimension" not in compiled["known_slots"]
    assert compiled["clarification_reason"] == "region_filter_not_supported"


def test_multi_turn_dimension_is_inherited_as_grouping() -> None:
    case = next(item for item in _load_cases(DATASET) if item["case_id"] == "ext-17")
    compiled = compile_turn_context(_state_for(case))["context_compilation"]
    assert _roles(compiled).get("product_category") == "grouping"


def test_p2_core_contract_hashes_match_p2_baseline() -> None:
    baseline = json.loads(P2_BASELINE.read_text(encoding="utf-8"))
    expected = {item["case_id"]: item["contract_hash"] for item in baseline["cases"]}
    mismatches = []
    for case in _core_cases():
        compiled = compile_turn_context(_state_for(case))["context_compilation"]
        actual = compiled.get("request_contract_hash")
        if actual != expected.get(case["case_id"]):
            mismatches.append((case["case_id"], expected.get(case["case_id"]), actual))
    assert mismatches == []
