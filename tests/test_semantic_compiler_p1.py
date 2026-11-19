"""P1 tests: detection/support decoupling and unified dimension support."""

from __future__ import annotations

import json
from pathlib import Path

from app.agents.data_query_agent import _capability_rejection, build_planning_catalog
from app.context.compiler import compile_turn_context
from app.context.support_resolver import resolve_support
from app.context.turn_resolver import _dimension_allowed
from app.graph.nodes.diagnostic import _requested_dimension_policy
from app.mcp.schemas import MetricPlanningDecision
from app.tenancy import TenantContext, activate_tenant_context
from evaluation.dimension_role_audit import _load_cases

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "datasets" / "dimension_role_cases_v1.jsonl"
P1_BASELINE = ROOT / "tests" / "fixtures" / "p1_contract_hashes.json"

SELLER_SCOPE = {
    "seller_id": "seller-1",
    "default_start_date": "2018-08-01",
    "default_end_date": "2018-08-31",
    "metric_name": "gmv",
    "granularity": "day",
    "allowed_metrics": ["gmv", "order_count", "freight_value"],
    "allowed_dimensions": ["product_category", "payment_type", "month_position"],
}


def _compile(question: str, *, scope: dict | None = None) -> dict:
    state = {
        "user_message": question,
        "trace_id": "p1-test",
        "preference_context": {"source_needs": [], "candidate_analysis_plan": None},
    }
    if scope:
        state["controlled_analysis_scope"] = scope
        state["use_current_explorer_context"] = True
    return compile_turn_context(state)["context_compilation"]


def _core_cases() -> list[dict]:
    return [case for case in _load_cases(DATASET) if case["tier"] == "core"]


def test_seller_grouping_is_detected_and_fails_closed() -> None:
    compiled = _compile("按店铺拆解 GMV", scope=SELLER_SCOPE)
    assert compiled["needs_clarification"] is True
    assert compiled["clarification_reason"] == "dimension_not_registered"
    assert (compiled["known_slots"].get("dimension") or {}).get("value") is None
    detection = compiled["p0_audit"]["dimension_detection"]
    assert "seller" in detection["candidates_all"]
    assert detection["business_candidates"] == ["seller"]
    assert detection["unregistered"] == ["seller"]


def test_scope_reference_is_not_turned_into_dimension() -> None:
    compiled = _compile("店铺范围内的 GMV", scope=SELLER_SCOPE)
    assert compiled["needs_clarification"] is False
    assert "dimension" not in compiled["known_slots"]
    detection = compiled["p0_audit"]["dimension_detection"]
    assert "seller" in detection["candidates_all"]
    assert detection["business_candidates"] == []


def test_contribution_request_keeps_seller_detected() -> None:
    compiled = _compile("按店铺拆解 GMV，统计贡献", scope=SELLER_SCOPE)
    assert compiled["needs_clarification"] is True
    assert compiled["clarification_reason"] == "dimension_not_registered"
    assert "seller" in compiled["p0_audit"]["dimension_detection"]["candidates_all"]


def test_registered_dimension_still_works() -> None:
    compiled = _compile("按商品品类统计 GMV")
    assert compiled["needs_clarification"] is False
    assert compiled["known_slots"]["dimension"]["value"] == "product_category"


def test_two_grouping_dimensions_still_clarify() -> None:
    compiled = _compile("按店铺和商品品类拆解订单量", scope=SELLER_SCOPE)
    assert compiled["needs_clarification"] is True
    assert compiled["clarification_reason"] == "ambiguous_dimension"
    assert set(compiled["dimension_candidates"]) == {"seller", "product_category"}


def test_original_question_uses_reference_role_for_seller() -> None:
    case = next(item for item in _core_cases() if item["case_id"] == "core-01")
    compiled = compile_turn_context(
        {
            "user_message": case["input"],
            "trace_id": "p1-original",
            "preference_context": {"source_needs": [], "candidate_analysis_plan": None},
            "controlled_analysis_scope": SELLER_SCOPE,
            "use_current_explorer_context": True,
        }
    )["context_compilation"]
    roles = {}
    for item in compiled.get("dimension_roles") or []:
        dimension = item["dimension"]
        if dimension not in roles or item["role"] == "grouping":
            roles[dimension] = item["role"]
    assert roles["seller"] == "reference"
    assert roles["product_category"] == "grouping"
    assert compiled.get("clarification_reason") is None
    assert compiled["known_slots"]["dimension"]["value"] == "product_category"


def test_platform_seller_grouping_is_not_silently_ignored() -> None:
    compiled = _compile("各店铺的 GMV")
    assert compiled["needs_clarification"] is True
    assert compiled["clarification_reason"] == "dimension_not_registered"
    assert (compiled["known_slots"].get("dimension") or {}).get("value") is None


def test_order_count_ordinary_and_contribution_differ() -> None:
    catalog = build_planning_catalog(scope="seller")
    contribution = MetricPlanningDecision(
        status="confirmed",
        confirmed_metrics=["order_count"],
        query_type="metric_contribution",
        dimension="product_category",
    )
    ordinary = MetricPlanningDecision(
        status="confirmed",
        confirmed_metrics=["order_count"],
        query_type="metric_breakdown",
        dimension="product_category",
    )
    contribution_rejection = _capability_rejection(
        contribution,
        catalog,
        question="按商品品类分析订单量贡献",
        requested_dimension="product_category",
        scope="seller",
        scope_data=SELLER_SCOPE,
    )
    ordinary_rejection = _capability_rejection(
        ordinary,
        catalog,
        question="按商品品类拆解订单量",
        requested_dimension="product_category",
        scope="seller",
        scope_data=SELLER_SCOPE,
    )
    assert contribution_rejection is None
    assert ordinary_rejection is not None
    assert "不支持按" in ordinary_rejection


def test_dimension_wrappers_agree_with_resolver() -> None:
    state = {"controlled_analysis_scope": SELLER_SCOPE}
    allowed, reason = _dimension_allowed("product_category", {"gmv"}, state)
    assert allowed is False
    assert reason == "dimension_not_supported_by_metric"

    allowed, reason = _dimension_allowed("seller", {"gmv"}, state)
    assert allowed is False
    assert reason == "dimension_not_registered"

    direct = resolve_support(
        metric_id="gmv",
        dimension="seller",
        operation="breakdown",
        scope=SELLER_SCOPE,
    )
    assert direct.executable is False
    assert direct.reason_code == "dimension_not_registered"


def test_diagnostic_policy_uses_resolver_for_seller() -> None:
    seller = TenantContext(
        tenant_id="p1-tenant",
        user_id="p1-user",
        role="analyst",
        scope_type="seller",
        seller_id="seller-1",
    )
    state = {
        "user_message": "按店铺和商品品类分析订单量贡献",
        "requested_dimensions": ["seller"],
        "controlled_analysis_scope": SELLER_SCOPE,
        "diagnostic_target_metric": "order_count",
    }
    with activate_tenant_context(seller):
        policy = _requested_dimension_policy(state)
    assert "seller" in policy["unsupported"]

    state["requested_dimensions"] = ["product_category"]
    with activate_tenant_context(seller):
        policy = _requested_dimension_policy(state)
    assert "product_category" in policy["registered"]
    assert "product_category" not in policy["unsupported"]


def test_p1_baseline_preserved_except_p2_role_changes() -> None:
    p2_behavior_change = {"core-01", "core-05", "core-06", "core-09", "core-12"}
    baseline = json.loads(P1_BASELINE.read_text(encoding="utf-8"))
    expected = {item["case_id"]: item["contract_hash"] for item in baseline["cases"]}
    mismatches = []
    for case in _core_cases():
        compiled = compile_turn_context(
            {
                "user_message": case["input"],
                "trace_id": f"p0-{case['case_id']}",
                "preference_context": {
                    "source_needs": [],
                    "candidate_analysis_plan": None,
                },
                **(
                    {
                        "controlled_analysis_scope": case["scope"],
                        "use_current_explorer_context": True,
                    }
                    if case.get("scope")
                    else {}
                ),
                **(
                    {"previous_turn_context": case["context"]["previous_turn_context"]}
                    if case.get("context")
                    else {}
                ),
            }
        )["context_compilation"]
        actual = compiled.get("request_contract_hash")
        if case["case_id"] in p2_behavior_change:
            if actual == expected.get(case["case_id"]):
                mismatches.append((case["case_id"], "expected_p2_drift_missing"))
        elif actual != expected.get(case["case_id"]):
            mismatches.append((case["case_id"], expected.get(case["case_id"]), actual))
    assert mismatches == []
