"""Negative controls for shared offline diagnostic fixes."""
from datetime import date, timedelta

import pytest

from app.analysis.language_guardrails import unsupported_causal_assertions
from app.graph.nodes.critic import _review_diagnostic_trajectory


def test_non_causal_termination_does_not_hide_a_following_causal_assertion():
    disclaimer = "Current evidence does not establish the business root cause."
    assert not unsupported_causal_assertions(disclaimer)
    assert unsupported_causal_assertions(disclaimer + " Shipping delays caused lower ratings.")


def test_pending_steps_permit_explicit_abstention_but_not_completion():
    state = {"diagnostic_plan": {"steps": []}, "diagnostic_step_state": {"s1": "pending"},
             "finish_decision": {"action_type": "ABSTAIN"}}
    assert _review_diagnostic_trajectory(state)["status"] == "approved"
    state["finish_decision"]["action_type"] = "FINISH"
    assert _review_diagnostic_trajectory(state)["status"] == "rejected"


@pytest.mark.parametrize("mutation", [None, "metric", "date", "value", "missing_series", "unconfirmed"])
def test_point_decomposition_requires_matching_confirmed_target_observations(monkeypatch, mutation):
    from app.graph.nodes import diagnostic as node

    monkeypatch.setattr(node, "_dates", lambda _: ("2024-04-28", "2024-05-04"))
    state = {"user_message": "客单价上升", "diagnostic_target_metric": "avg_order_value",
             "tenant_id": "platform", "user_id": "guard-test", "premise_check": {
                 "status": "confirmed", "metric_name": "avg_order_value",
                 "baseline_date": "2024-05-03", "current_date": "2024-05-04",
                 "baseline_value": 10.0, "current_value": 20.0,
             }}
    values = {date(2024, 4, 28) + timedelta(days=i): 10.0 for i in range(7)}
    values[date(2024, 5, 4)] = 20.0
    monkeypatch.setattr(node, "_trusted_daily_series", lambda _, metric: (
        None if mutation == "missing_series" and metric == "avg_order_value" else dict(values)))
    if mutation == "metric":
        state["premise_check"]["metric_name"] = "freight_rate"
    elif mutation == "date":
        state["premise_check"]["baseline_date"] = "2024-05-02"
    elif mutation == "value":
        state["premise_check"]["current_value"] = 999
    elif mutation == "unconfirmed":
        state["premise_check"]["status"] = "unavailable"
    available = node._available_chat_decompositions(state, "avg_order_value", "platform")
    assert ("symmetric_ratio_v1" in available) is (mutation is None)
