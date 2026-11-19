"""Regression for the real G run that shifted a two-date task back one day."""
from __future__ import annotations

import copy
import json
import socket
from pathlib import Path

import httpx
import pytest

from app.agents.diagnostic_contracts import DiagnosticStep
from app.graph.nodes import diagnostic
from app.tenancy import TenantContext, activate_tenant_context


TRACE = Path(__file__).resolve().parents[1] / (
    "docs/testing/resume_dev_pair_order_dimension_20261015_cb7bd101/traces/"
    "a3f66690-368f-4309-a166-664c06d89ba9.json"
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("network forbidden in diagnostic date regression")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)


def _state(baseline="2024-05-03", current="2024-05-04", metric="order_count"):
    return {
        "user_message": f"Explain {metric} from {baseline} to {current} by product_category",
        "tenant_id": "platform", "user_id": "date-role-regression", "user_role": "admin",
        "diagnostic_target_metric": metric,
        "requested_dimensions": ["product_category"],
        "context_compilation": {"known_slots": {
            "baseline_date": {"value": baseline}, "current_date": {"value": current},
        }},
        "runtime_budget": {"max_tool_calls": 8},
    }


@pytest.mark.parametrize("metric", ["order_count", "gmv", "freight_value"])
@pytest.mark.parametrize("baseline,current", [
    ("2024-05-03", "2024-05-04"),
    ("2024-05-01", "2024-05-04"),
    ("2024-05-04", "2024-05-01"),
])
def test_target_required_query_and_candidates_share_exact_compiled_roles(metric, baseline, current):
    state = _state(baseline, current, metric)
    context = TenantContext(
        tenant_id="platform", user_id=state["user_id"], role="admin", scope_type="platform",
    )
    with activate_tenant_context(context):
        investigation = diagnostic._ensure_graph_investigation_state(state)
        query = diagnostic._required_action_for(state, DiagnosticStep(
            step_id="dimension", kind="dimension_contribution", required=True,
            objective="Get product_category statistical contribution",
            allowed_actions=["CALL_TOOL", "FINISH", "ABSTAIN"],
            completion_condition="Validated contribution for the requested pair",
        ))
        catalog = diagnostic._candidate_tool_catalog(state)
    assert investigation.target.baseline_start.isoformat() == baseline
    assert investigation.target.current_start.isoformat() == current
    assert query.comparison.baseline_date == baseline
    assert query.comparison.current_date == current
    contributions = [action for action in catalog.values()
                     if getattr(action, "tool_name", None) == "compare_breakdown_by_dimension"]
    # Adaptive dimension candidates require supported natural periods; an
    # exact-day contribution is scheduled by the required-action boundary.
    # Do not invent extra candidate capabilities to satisfy this regression.
    assert catalog
    for action in catalog.values():
        if getattr(action, "period", None) is not None:
            assert action.period.start_date == min(baseline, current)
            assert action.period.end_date == max(baseline, current)
    for action in contributions:
        assert action.comparison.baseline_date == baseline
        assert action.comparison.current_date == current
    assert diagnostic._diagnostic_dates(state) == tuple(sorted((baseline, current)))


def test_saved_failure_first_divergence_is_removed_without_rewriting_history():
    original = json.loads(TRACE.read_text(encoding="utf-8"))["state_snapshot"]
    assert original["investigation_state"]["target"]["current_start"] == "2024-05-03"
    state = copy.deepcopy(original)
    # Re-enter the target boundary before any investigation existed, preserving
    # the actual compiler slots and initial planning/authorized query records.
    for key in ("investigation_state", "root_cause_evidence", "diagnostic_tool_results"):
        state.pop(key, None)
    state["diagnostic_investigation_action_count"] = 0
    target = diagnostic._ensure_graph_investigation_state(state).target
    assert target.baseline_start.isoformat() == "2024-05-03"
    assert target.current_start.isoformat() == "2024-05-04"
    assert diagnostic._diagnostic_dates(state) == ("2024-05-03", "2024-05-04")
    assert json.loads(TRACE.read_text(encoding="utf-8"))["state_snapshot"] == original


def test_compiler_roles_take_precedence_over_raw_text_and_legacy_single_date_stays_supported():
    state = _state()
    state["user_message"] = "old unrelated date 2020-01-01"
    assert diagnostic._diagnostic_dates(state) == ("2024-05-03", "2024-05-04")
    assert diagnostic._dates("Explain order_count on 2024-05-04") == ("2024-04-28", "2024-05-04")
    assert diagnostic._dates("current 2024-05-01 baseline 2024-05-04") == ("2024-05-01", "2024-05-04")


def test_chinese_adjacent_iso_date_does_not_use_legacy_default():
    question = "为什么2024-05-04的订单量下降了？请检查证据，区分统计贡献与业务原因。"
    assert diagnostic._dates(question) == ("2024-04-28", "2024-05-04")


def test_corrupt_compiled_roles_are_not_replaced_with_guessed_dates():
    state = _state()
    del state["context_compilation"]["known_slots"]["current_date"]
    with pytest.raises(ValueError, match="date roles are invalid"):
        diagnostic._diagnostic_dates(state)


def test_month_period_comparison_is_reused_for_target_and_required_contribution():
    state = {
        "user_message": "平台范围内，2018 年 8 月 GMV 为什么比 2018 年 7 月下降？按 product_category 拆解。",
        "tenant_id": "platform", "user_id": "month-period-regression", "user_role": "admin",
        "diagnostic_target_metric": "gmv",
        "requested_dimensions": ["product_category"],
        "metric_query_batch_plan": {"plans": [{
            "query_type": "metric_compare",
            "metric_name": "gmv",
            "authorized_metric": {"metric_id": "gmv"},
            "baseline_start": "2018-07-01",
            "baseline_end": "2018-07-31",
            "current_start": "2018-08-01",
            "current_end": "2018-08-31",
            "granularity": "month",
        }]},
        "runtime_budget": {"max_tool_calls": 8},
    }
    context = TenantContext(
        tenant_id="platform", user_id=state["user_id"], role="admin", scope_type="platform",
    )
    with activate_tenant_context(context):
        investigation = diagnostic._ensure_graph_investigation_state(state)
        action = diagnostic._required_action_for(state, DiagnosticStep(
            step_id="dimension", kind="dimension_contribution", required=True,
            objective="Get product_category statistical contribution",
            allowed_actions=["CALL_TOOL", "FINISH", "ABSTAIN"],
            completion_condition="Validated contribution for the requested periods",
        ))

    assert diagnostic._diagnostic_dates(state) == ("2018-07-01", "2018-08-31")
    assert investigation.target.baseline_start.isoformat() == "2018-07-01"
    assert investigation.target.baseline_end.isoformat() == "2018-07-31"
    assert investigation.target.current_start.isoformat() == "2018-08-01"
    assert investigation.target.current_end.isoformat() == "2018-08-31"
    assert action.tool_name == "compare_breakdown_by_dimension"
    assert action.comparison.baseline_start == "2018-07-01"
    assert action.comparison.baseline_end == "2018-07-31"
    assert action.comparison.current_start == "2018-08-01"
    assert action.comparison.current_end == "2018-08-31"
    assert action.comparison.granularity == "month"


def test_incomplete_compiled_period_pair_is_not_replaced_with_a_default_date():
    state = {
        "user_message": "平台范围内，2018 年 8 月 GMV 为什么比 2018 年 7 月下降？",
        "diagnostic_target_metric": "gmv",
        "context_compilation": {"request_contract": {
            "metric_id": "gmv",
            "comparison": {
                "kind": "period_pair",
                "baseline": {"start_date": "2018-07-01", "end_date": "2018-07-31"},
                "current": {"start_date": "2018-08-01"},
                "granularity": "month",
            },
        }},
    }

    with pytest.raises(ValueError, match="period boundaries are incomplete"):
        diagnostic._diagnostic_dates(state)


def test_saved_zero_contribution_does_not_invent_a_largest_factor_or_garbled_limitations():
    state = json.loads(TRACE.read_text(encoding="utf-8"))["state_snapshot"]
    result = diagnostic.diagnostic_finalize(copy.deepcopy(state))
    analysis = result["analysis_result"]
    assert "largest measured contribution" not in analysis["summary"]
    assert "all returned group changes are zero" in analysis["summary"]
    assert not any("???" in str(item) for item in analysis.get("limitations", []))
