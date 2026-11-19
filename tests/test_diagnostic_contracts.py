import pytest
from pydantic import TypeAdapter, ValidationError

from app.agents.diagnostic_contracts import (
    CallCandidateAction,
    CallToolAction,
    DiagnosticAction,
    DiagnosticPlan,
    FinishAction,
    ReplanAction,
)


def _plan() -> dict:
    return {
        "goal": "解释 GMV 异常",
        "hypotheses": ["订单量或客单价变化"],
        "steps": [{
            "step_id": "trend", "objective": "确认趋势",
            "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"],
            "completion_condition": "获得趋势证据",
        }],
        "stop_conditions": ["证据足够"],
        "budget": {"max_rounds": 3, "max_tool_calls": 4, "max_tokens": 1000, "max_cost": 0.1},
    }


def test_diagnostic_plan_and_call_tool_are_strongly_typed() -> None:
    plan = DiagnosticPlan.model_validate(_plan())
    action = TypeAdapter(DiagnosticAction).validate_python({
        "action_type": "CALL_TOOL", "tool_name": "query_metric_trend", "metric_name": "gmv",
        "period": {"start_date": "2025-03-25", "end_date": "2025-03-31", "granularity": "day"},
    })
    assert plan.steps[0].step_id == "trend"
    assert isinstance(action, CallToolAction)


def test_action_discriminator_rejects_mixed_fields() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(DiagnosticAction).validate_python({
            "action_type": "FINISH", "conclusion": "done", "evidence_ids": ["ev_1"],
            "tool_name": "query_gmv",
        })


def test_call_tool_requires_exactly_one_time_selector() -> None:
    with pytest.raises(ValidationError):
        CallToolAction(action_type="CALL_TOOL", tool_name="query_gmv", metric_name="gmv")


def test_candidate_selection_requires_current_candidate_set_handle() -> None:
    with pytest.raises(ValidationError):
        CallCandidateAction.model_validate({
            "action_type": "CALL_CANDIDATE", "candidate_id": "candidate-1",
        })
    action = CallCandidateAction.model_validate({
        "action_type": "CALL_CANDIDATE", "candidate_set_id": "set-current",
        "candidate_id": "candidate-1",
    })
    assert action.candidate_set_id == "set-current"


def test_parallel_fact_query_requires_exactly_one_time_selector() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(DiagnosticAction).validate_python({
            "action_type": "PARALLEL_FACTS",
            "facts": [{"tool_name": "query_metric_trend", "metric_name": "gmv"}],
        })


def test_replan_cannot_be_constructed_with_finish_fields() -> None:
    with pytest.raises(ValidationError):
        ReplanAction.model_validate({
            "action_type": "REPLAN", "steps": [], "stop_conditions": ["stop"],
            "conclusion": "not allowed",
        })


def test_finish_requires_existing_reference_shape() -> None:
    action = FinishAction(action_type="FINISH", conclusion="done", evidence_ids=["ev_1"])
    assert action.evidence_ids == ["ev_1"]
