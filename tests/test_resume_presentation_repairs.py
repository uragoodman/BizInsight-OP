from __future__ import annotations

import socket

import pytest

from app.analysis.attribution_methods import (
    bind_registered_ratio_factor_identities,
    symmetric_ratio_attribution,
)
from app.graph.nodes.diagnostic import (
    _english_contribution_share_text,
    _render_unresolved_branch,
    _validated_share_text,
)
from evaluation.resume_bare_tool_loop import _registered_decomposition_card
from app.metrics.registry import get_registry


@pytest.fixture(autouse=True)
def block_network_and_database(monkeypatch):
    def deny(address, *args, **kwargs):
        raise AssertionError(f"network/database connection blocked in offline test: {address!r}")

    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)


def _ratio_response(*, identities: bool = True) -> dict:
    registry = get_registry()
    definition = registry.get("avg_order_value")
    result = symmetric_ratio_attribution(
        metric_id="avg_order_value",
        baseline_numerator=1000,
        current_numerator=1500,
        baseline_denominator=100,
        current_denominator=100,
        unit=definition.unit,
    )
    result = bind_registered_ratio_factor_identities(result, scope="platform")
    result = result.model_copy(update={"evidence_ids": ["source:gmv", "source:order_count"]})
    dumped = result.model_dump(mode="json")
    if not identities:
        dumped["factor_identities"] = []
    return {
        "result": {
            "status": "ok",
            "operation_intent": "registered_numeric_decomposition",
            "target_metric_id": "avg_order_value",
            "baseline_date": "2024-05-03",
            "current_date": "2024-05-04",
            "method_id": "symmetric_ratio_v1",
            "metric_unit": definition.unit,
            "parsed": {
                "metric_name": "avg_order_value",
                "baseline_date": "2024-05-03",
                "current_date": "2024-05-04",
                "authorized_metric": {"metric_id": "avg_order_value", "provenance_type": "exact"},
            },
            "decomposition_result": dumped,
            "source_evidence_ids": ["source:gmv", "source:order_count"],
        }
    }


def _percent_ratio_response() -> dict:
    registry = get_registry()
    definition = registry.get("freight_rate")
    result = symmetric_ratio_attribution(
        metric_id="freight_rate",
        baseline_numerator=10,
        current_numerator=20,
        baseline_denominator=100,
        current_denominator=100,
        unit=definition.unit,
    )
    result = bind_registered_ratio_factor_identities(result, scope="platform")
    result = result.model_copy(update={"evidence_ids": ["source:freight_value", "source:gmv"]})
    return {
        "result": {
            "status": "ok",
            "operation_intent": "registered_numeric_decomposition",
            "target_metric_id": "freight_rate",
            "baseline_date": "2024-05-03",
            "current_date": "2024-05-04",
            "method_id": "symmetric_ratio_v1",
            "metric_unit": definition.unit,
            "parsed": {
                "metric_name": "freight_rate",
                "baseline_date": "2024-05-03",
                "current_date": "2024-05-04",
                "authorized_metric": {"metric_id": "freight_rate", "provenance_type": "exact"},
            },
            "decomposition_result": result.model_dump(mode="json"),
            "source_evidence_ids": ["source:freight_value", "source:gmv"],
        }
    }


def test_registered_ratio_result_exposes_registered_factor_identity_and_card():
    response = _ratio_response()
    serialized = response["result"]["decomposition_result"]
    roles = {item["role_id"]: item["metric_id"] for item in serialized["factor_identities"]}
    assert roles == {"numerator_effect": "gmv", "denominator_effect": "order_count"}

    card = _registered_decomposition_card(response)
    assert card is not None
    factors = {item["role_id"]: item for item in card["factors"]}
    assert factors["numerator_effect"]["display_name"] == get_registry().get("gmv").display_name
    assert factors["denominator_effect"]["display_name"] == get_registry().get("order_count").display_name
    assert card["baseline_date"] == "2024-05-03"
    assert card["current_date"] == "2024-05-04"
    assert card["business_cause_confirmed"] is False


def test_registered_percent_decomposition_card_separates_level_and_absolute_change_units():
    card = _registered_decomposition_card(_percent_ratio_response())

    assert card["unit"] == card["metric_unit"] == "%"
    assert card["change_unit"] == card["component_unit"] == "percentage_points"
    assert {item["unit"] for item in card["factors"]} == {"percentage_points"}


@pytest.mark.parametrize(
    "mutate", ["missing", "conflicting_metric", "conflicting_display_name", "target_mismatch", "date_mismatch"],
)
def test_registered_ratio_card_fails_closed_on_missing_or_conflicting_mapping(mutate):
    response = _ratio_response(identities=mutate != "missing")
    result = response["result"]
    if mutate == "conflicting_metric":
        result["decomposition_result"]["factor_identities"][1]["metric_id"] = "gmv"
    elif mutate == "conflicting_display_name":
        result["decomposition_result"]["factor_identities"][1]["display_name"] = "成交总额"
    elif mutate == "target_mismatch":
        result["parsed"]["metric_name"] = "gmv"
    elif mutate == "date_mismatch":
        result["parsed"]["baseline_date"] = "2024-05-04"
    with pytest.raises(ValueError):
        _registered_decomposition_card(response)


def test_bare_loop_attaches_registry_named_card_without_rewriting_model_text():
    from app.tenancy import TenantContext
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop

    question = (
        "Compare avg_order_value on 2024-05-03 and 2024-05-04 using its registered numeric decomposition."
    )
    actions = iter([
        {"action": "TOOL_CALL", "tool": "registered_numeric_decomposition", "arguments": {
            "metric_id": "avg_order_value", "baseline_date": "2024-05-03", "current_date": "2024-05-04",
        }},
        {"action": "FINAL", "answer": "The ratio changed; business cause is unconfirmed.", "claims": [],
         "presentation": {"metric_id": "avg_order_value"}},
    ])
    response = _ratio_response()
    response["request_id"] = "offline-request"

    def tool_fn(**_kwargs):
        return response

    output = run_bare_tool_loop(
        question,
        tenant_context=TenantContext(
            tenant_id="platform", user_id="offline-ratio-render", role="admin", scope_type="platform",
        ),
        run_id="offline-ratio-render",
        model_fn=lambda _system, _user: next(actions),
        tool_fn=tool_fn,
    )
    assert output["errors"] == []
    assert output["final"]["answer"].startswith("The ratio changed; business cause is unconfirmed.")
    assert "成交总额" in output["final"]["answer"]
    assert "订单量" in output["final"]["answer"]
    assert "不确认业务原因" in output["final"]["answer"]
    card = output["final"]["presentation"]["metric_decomposition"]
    assert {item["display_name"] for item in card["factors"]} == {
        get_registry().get("gmv").display_name,
        get_registry().get("order_count").display_name,
    }
    assert card["change_unit"] == card["component_unit"] == "BRL/order"
    assert output["final"]["presentation"]["status"] == "completed"
    assert output["terminal_code"] == "ENOUGH_EVIDENCE"


def test_bare_loop_does_not_treat_registry_ids_as_user_facing_factor_names():
    from app.tenancy import TenantContext
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop

    response = _ratio_response()
    response["request_id"] = "offline-id-only-factor-labels"
    actions = iter([
        {"action": "TOOL_CALL", "tool": "registered_numeric_decomposition", "arguments": {
            "metric_id": "avg_order_value", "baseline_date": "2024-05-03", "current_date": "2024-05-04",
        }},
        {"action": "FINAL", "answer": (
            "avg_order_value changed; avg_order_value factors were "
            "gmv and order_count. Business cause is unconfirmed."
        ), "claims": [], "presentation": {"metric_id": "avg_order_value"}},
    ])
    output = run_bare_tool_loop(
        "Compare avg_order_value on 2024-05-03 and 2024-05-04 using its registered numeric decomposition.",
        tenant_context=TenantContext(
            tenant_id="platform", user_id="offline-factor-labels", role="admin", scope_type="platform",
        ),
        run_id="offline-factor-labels",
        model_fn=lambda _system, _user: next(actions),
        tool_fn=lambda **_kwargs: response,
    )

    answer = output["final"]["answer"]
    assert "成交总额" in answer
    assert "订单量" in answer
    assert "avg_order_value" in output["final"]["presentation"]["metric_decomposition"]["metric_id"]


def test_bare_loop_projects_percent_units_and_completion_from_validated_decomposition():
    from app.tenancy import TenantContext
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop

    response = _percent_ratio_response()
    response["request_id"] = "offline-freight-rate-request"
    actions = iter([
        {"action": "TOOL_CALL", "tool": "registered_numeric_decomposition", "arguments": {
            "metric_id": "freight_rate", "baseline_date": "2024-05-03", "current_date": "2024-05-04",
        }},
        {"action": "FINAL", "answer": "数值分解已核对，不推断业务因果。", "claims": [],
         "presentation": {"metric_id": "freight_rate"}},
    ])
    output = run_bare_tool_loop(
        "比较平台运费率并按注册方式数值分解，不推断业务因果。",
        tenant_context=TenantContext(
            tenant_id="platform", user_id="offline-rate-units", role="admin", scope_type="platform",
        ),
        run_id="offline-rate-units",
        model_fn=lambda _system, _user: next(actions),
        tool_fn=lambda **_kwargs: response,
    )

    card = output["final"]["presentation"]["metric_decomposition"]
    assert card["unit"] == card["metric_unit"] == "%"
    assert card["change_unit"] == card["component_unit"] == "percentage_points"
    assert {item["unit"] for item in card["factors"]} == {"percentage_points"}
    assert output["final"]["presentation"]["status"] == "completed"
    assert output["terminal_code"] == "ENOUGH_EVIDENCE"


def test_absolute_share_copy_preserves_direction_and_does_not_clip_over_100_percent():
    phrase = _validated_share_text(
        share=13.41, effect=-3768.41, root_delta=281.01,
        component_values=[-3768.41, 4049.42],
    )
    assert "1341.0%" in phrase
    assert "绝对贡献幅度" in phrase
    assert "方向相反" in phrase
    assert "形成数值抵消" in phrase
    assert "1341.0%" in _english_contribution_share_text(-13.41, -3768.41)
    assert "absolute net change" in _english_contribution_share_text(-13.41, -3768.41)
    assert "opposing contribution" in _english_contribution_share_text(-13.41, -3768.41)


def test_share_copy_does_not_claim_offset_without_validated_opposing_component():
    phrase = _validated_share_text(
        share=0.4, effect=4.0, root_delta=10.0, component_values=[4.0, 6.0],
    )
    assert "形成数值抵消" not in phrase
    assert "方向相反" not in phrase
    assert "相当于根目标净变化绝对值" in phrase


def test_branch_copy_distinguishes_target_factor_and_dimension():
    evidence_id = "ev-valid"
    evidence = {
        evidence_id: {
            "evidence_type": "metric_decomposition", "metric_id": "avg_order_value",
            "method_id": "symmetric_ratio_v1",
            "values": {
                "total_delta": 10.0, "factor_metrics": ["gmv", "order_count"],
                "factor_identities": [
                    {"role_id": "numerator_effect", "metric_id": "gmv", "display_name": "成交总额"},
                    {"role_id": "denominator_effect", "metric_id": "order_count", "display_name": "订单量"},
                ],
                "components": [
                    {"component_id": "numerator_effect", "value": 14.0},
                    {"component_id": "denominator_effect", "value": -4.0},
                ],
            },
        }
    }
    common = {
        "dimension_labels": {"customer_state": "顾客州/地区"},
        "validated_evidence_ids": {evidence_id}, "evidence_by_id": evidence,
        "target_metric_id": "avg_order_value", "target_metric_name": "客单价",
        "root_delta": 10.0, "numeric_decomposition_valid": True,
    }
    target = _render_unresolved_branch({
        "branch_type": "root_target", "metric_id": "avg_order_value", "status": "open",
    }, **common)
    assert "分析目标「客单价」" in target
    assert "数值分解已验证" in target
    assert "维度=" not in target

    factor = _render_unresolved_branch({
        "branch_type": "metric_effect", "metric_id": "order_count", "effect_value": -4.0,
        "effect_unit": "orders", "root_effect_share": 0.4,
        "parent_evidence_id": evidence_id, "status": "open",
    }, **common)
    assert "指标因素「订单量」" in factor
    assert "__metric_factor__" not in factor
    assert "业务原因仍未确认" in factor

    legacy_factor = _render_unresolved_branch({
        "metric_id": "order_count", "dimension": "__metric_factor__",
        "dimension_value": "order_count", "effect_value": -4.0,
        "parent_evidence_id": evidence_id, "status": "open",
    }, **common)
    assert "指标因素「订单量」" in legacy_factor
    assert "__metric_factor__" not in legacy_factor

    dimension = _render_unresolved_branch({
        "branch_type": "dimension_contribution", "metric_id": "avg_order_value",
        "dimension": "customer_state", "dimension_value": "SP", "status": "blocked",
        "effect_value": 2.0, "effect_unit": "BRL/order", "root_effect_share": 0.2,
        "parent_evidence_id": "ev-dimension",
    },
       **{**common,
          "validated_evidence_ids": {evidence_id, "ev-dimension"},
          "evidence_by_id": {
              **evidence,
              "ev-dimension": {
                  "evidence_type": "dimension_contribution", "metric_id": "avg_order_value",
                  "values": {"dimension": "customer_state", "components": [
                      {"dimension_value": "SP", "delta_value": 2.0},
                  ]},
              },
          }})
    assert "维度贡献「顾客州/地区=SP」" in dimension
    assert "受数据覆盖限制" in dimension


def test_metric_effect_copy_uses_validated_component_unit_not_target_level_unit():
    evidence_id = "ev-freight-rate-decomposition"
    evidence = {
        evidence_id: {
            "evidence_type": "metric_decomposition", "metric_id": "freight_rate",
            "method_id": "symmetric_ratio_v1",
            "values": {
                "factor_metrics": ["freight_value", "gmv"],
                "factor_identities": [
                    {"role_id": "numerator_effect", "metric_id": "freight_value", "display_name": "运费总额"},
                    {"role_id": "denominator_effect", "metric_id": "gmv", "display_name": "成交总额"},
                ],
                "display_components": [
                    {"component_id": "numerator_effect", "value": 1.25, "unit": "percentage_points"},
                    {"component_id": "denominator_effect", "value": -0.25, "unit": "percentage_points"},
                ],
            },
        }
    }
    text = _render_unresolved_branch(
        {
            "branch_type": "metric_effect", "metric_id": "freight_value",
            "effect_value": 1.25, "effect_unit": "%", "parent_evidence_id": evidence_id,
            "root_effect_share": 1.25, "status": "open",
        },
        dimension_labels={}, validated_evidence_ids={evidence_id}, evidence_by_id=evidence,
        target_metric_id="freight_rate", target_metric_name="运费率", root_delta=1.0,
        numeric_decomposition_valid=True,
    )

    assert "数值贡献 +1.25 个百分点" in text
    assert "percentage_points" not in text
    assert "数值贡献 +1.25 %" not in text


def test_chinese_answer_projection_localizes_percentage_point_units_without_changing_contract():
    from app.graph.nodes.answer import _decomposition_unit_text, _metric_decomposition_text

    decomposition = {
        "display_name": "运费率",
        "unit": "%",
        "change_unit": "percentage_points",
        "baseline_value": 16.75,
        "current_value": 16.29,
        "delta_value": -0.46,
        "baseline_period": {"start_date": "2017-07-15", "end_date": "2017-07-15"},
        "current_period": {"start_date": "2017-07-16", "end_date": "2017-07-16"},
        "factors": [{"display_name": "运费总额", "contribution": 1.91, "unit": "percentage_points"}],
        "evidence_ids": ["ev-offline"],
    }

    assert _decomposition_unit_text("percentage_points") == "个百分点"
    rendered = _metric_decomposition_text(decomposition)
    assert "个百分点" in rendered
    assert "percentage points" not in rendered
    assert decomposition["unit"] == "%"
    assert decomposition["change_unit"] == "percentage_points"


def test_invalid_branch_evidence_is_not_rendered_as_verified_contribution():
    text = _render_unresolved_branch({
        "branch_type": "metric_effect", "metric_id": "order_count", "effect_value": -4.0,
        "root_effect_share": 0.4, "parent_evidence_id": "not-validated", "status": "open",
    }, dimension_labels={}, validated_evidence_ids=set(), evidence_by_id={},
       target_metric_id="gmv", target_metric_name="成交总额", root_delta=10.0,
       numeric_decomposition_valid=False)
    assert "尚无可引用的有效贡献证据" in text
    assert "已验证数值贡献" not in text
    assert "约占根目标变化" not in text
