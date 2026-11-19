"""Required Critic review policy and durable terminal-status contract."""

from app.agents.critic_agent import review_single_data_query
from app.graph.nodes.critic import _is_bound_single_no_data_result, _required_llm_review_names
from app.graph.terminal_status import result_terminal_status


def test_task_review_profiles_require_groundedness_after_deterministic_checks():
    assert _required_llm_review_names("metric_query", False) == {"groundedness"}
    assert _required_llm_review_names("anomaly_analysis", False) == {"groundedness"}
    assert _required_llm_review_names("report_generation", False) == {"groundedness"}
    assert _required_llm_review_names("metric_query", True) == {"groundedness"}


def test_required_review_incomplete_is_not_persisted_as_completed():
    incomplete_with_deliverable = {
        "errors": [{"code": "critic_review_incomplete"}],
        "analysis_presentation": {"status": "partial"},
    }
    incomplete_without_deliverable = {
        "errors": [{"code": "critic_review_incomplete"}],
        "analysis_presentation": {"status": "failed"},
    }
    rejected_with_deliverable = {
        "errors": [{"code": "critic_rejected"}],
        "analysis_presentation": {"status": "completed"},
    }

    assert result_terminal_status(incomplete_with_deliverable) == "abstained"
    assert result_terminal_status(incomplete_without_deliverable) == "failed"
    assert result_terminal_status(rejected_with_deliverable) == "failed"


def test_unsupported_metric_plan_is_persisted_as_abstained():
    assert result_terminal_status({
        "metric_planning_decision": {
            "status": "unsupported",
            "reason_code": "unsupported_dimension",
        },
        "errors": [],
    }) == "abstained"


def _bound_no_data_case():
    request_id = "trace-no-data:1:1"
    evidence_id = "ev-no-data"
    inner_result = {
        "status": "not_found",
        "parsed": {
            "metric_name": "gmv",
            "query_type": "metric_value",
            "authorized_metric": {"metric_id": "gmv"},
        },
    }
    plan = {"metric_name": "gmv", "query_type": "metric_value"}
    wrapper = {
        "request_id": request_id,
        "evidence_id": evidence_id,
        "server": "metric-registry",
        "tool_name": "registry_query",
        "result": inner_result,
    }
    call = {
        "request_id": request_id,
        "server": "metric-registry",
        "tool_name": "registry_query",
        "planned_metric_id": "gmv",
        "planned_operation": "metric_value",
        "status": "not_found",
    }
    evidence = {
        "evidence_id": evidence_id,
        "source_ref": "registry_query",
        "content": {
            "request_id": request_id,
            "metric_id": "gmv",
            "query_type": "metric_value",
            "result": inner_result,
        },
    }
    review = review_single_data_query(wrapper, plan, [evidence], [call])
    state = {
        "task_type": "metric_query",
        "terminal_code": "DATA_INSUFFICIENT",
        "errors": [],
        "answer": "2018-10-30 的 GMV 暂无数据。",
        "query_results": [wrapper],
        "metric_query_batch_plan": {"plans": [plan]},
        "tool_calls": [call],
        "metric_planning_decision": {"status": "confirmed"},
        "analysis_presentation": {
            "status": "partial",
            "facts": [],
            "changes": [],
            "contributions": [],
            "metric_decomposition": None,
        },
    }
    return state, review


def test_only_fully_bound_single_no_data_query_can_complete_as_no_data():
    state, data_review = _bound_no_data_case()

    assert data_review["status"] == "partial"
    assert data_review["items"][0]["status"] == "unavailable"
    assert _is_bound_single_no_data_result(
        state, data_review, {"status": "passed"},
    ) is True

    invalid_states = []
    auth_failure = {**state, "errors": [{"code": "authorization_blocked"}]}
    invalid_states.append(auth_failure)
    numeric_presentation = {
        **state,
        "analysis_presentation": {**state["analysis_presentation"], "facts": [{"value": 123}]},
    }
    invalid_states.append(numeric_presentation)
    wrong_answer = {**state, "answer": "GMV 为 123 BRL。"}
    invalid_states.append(wrong_answer)
    for invalid in invalid_states:
        assert _is_bound_single_no_data_result(
            invalid, data_review, {"status": "passed"},
        ) is False

    assert _is_bound_single_no_data_result(
        state, data_review, {"status": "rejected"},
    ) is False


def test_unbound_or_multi_item_no_data_batch_cannot_use_completion_path():
    state, data_review = _bound_no_data_case()
    broken_review = {
        **data_review,
        "items": [{
            **data_review["items"][0],
            "binding_checks": {**data_review["items"][0]["binding_checks"], "exact_tool_invocation_matches": False},
        }],
    }
    assert _is_bound_single_no_data_result(
        state, broken_review, {"status": "passed"},
    ) is False

    multi_item_state = {**state, "query_results": [*state["query_results"], state["query_results"][0]]}
    assert _is_bound_single_no_data_result(
        multi_item_state, data_review, {"status": "passed"},
    ) is False
