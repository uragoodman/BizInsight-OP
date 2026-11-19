from __future__ import annotations

import copy

from evaluation.provider_candidate_protocol import (
    score_provider_candidate_run,
    score_provider_candidate_run_v5,
)
from tests.test_provider_candidate_protocol import _aov_candidate, _full_trace


def _current_full_trace() -> dict:
    trace = copy.deepcopy(_full_trace())
    state = trace["workflow_state"]
    review = state["critic_result"]["llm_review"]
    review["status"] = "passed"
    for index, name in enumerate(("groundedness", "completeness", "consistency"), start=1):
        review[name].update({
            "completion_status": "completed",
            "call_ids": [f"critic-call-{index}"],
            "sdk_attempt_ids": [f"critic-attempt-{index}"],
        })
    return trace


def test_v5_accepts_current_completed_full_critic_status_and_preserves_v4_result() -> None:
    trace = _current_full_trace()
    old = score_provider_candidate_run(trace, candidate=_aov_candidate(), arm="full", replicate=1)
    new = score_provider_candidate_run_v5(trace, candidate=_aov_candidate(), arm="full", replicate=1)

    assert old["scorer_version"] == "provider-candidate-run-v4"
    assert old["task_acceptance_pass"] is False
    assert new["scorer_version"] == "provider-candidate-run-v5"
    assert new["critic"]["observed_llm_review_status"] == "passed"
    assert new["critic"]["review_execution_complete"] is True
    assert new["critic"]["review_contract_pass"] is True
    assert new["task_acceptance_pass"] is True


def test_v5_fails_closed_when_review_call_binding_is_missing_or_mismatched() -> None:
    for replacement in ([], ["not-a-real-call"]):
        trace = _current_full_trace()
        trace["workflow_state"]["critic_result"]["llm_review"]["completeness"]["call_ids"] = replacement
        result = score_provider_candidate_run_v5(trace, candidate=_aov_candidate(), arm="full", replicate=1)
        assert result["critic"]["review_execution_complete"] is False
        assert result["critic"]["review_contract_pass"] is False
        assert result["task_acceptance_pass"] is False


def test_v5_off_requires_explicit_ablation_and_zero_optional_review_calls() -> None:
    trace = copy.deepcopy(_full_trace())
    state = trace["workflow_state"]
    review = state["critic_result"]["llm_review"]
    review.update({"mode": "critic_llm_review_off", "status": "evaluation_ablation"})
    for name in ("groundedness", "completeness", "consistency"):
        review[name] = {"verdict": "evaluation_ablation", "reason": "llm_review_disabled_for_evaluation"}
    state["critic_result"]["status"] = "approved"
    state["llm_calls"] = []
    result = score_provider_candidate_run_v5(trace, candidate=_aov_candidate(), arm="critic_llm_review_off", replicate=1)
    assert result["critic"]["mode_matches"] is True
    assert result["critic"]["critic_review_sdk_logical_calls"] == 0
    assert result["task_acceptance_pass"] is True
