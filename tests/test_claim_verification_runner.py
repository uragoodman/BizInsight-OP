import pytest

from evaluation.claim_verification.runner import evaluate_oracle_case, route_claim


def test_unknown_and_causal_claims_are_escalated():
    assert route_claim({"claim_id": "c", "claim_type": "causal_claim"}).route == "needs_semantic_review"
    assert route_claim({"claim_id": "u", "claim_type": "unknown"}).route == "needs_semantic_review"


def test_deterministic_claim_requires_evidence_binding():
    assert route_claim({"claim_id": "n", "claim_type": "numeric_fact"}).route == "needs_semantic_review"
    assert route_claim({"claim_id": "n", "claim_type": "numeric_fact", "evidence_ids": ["ev_1"]}).route == "deterministic"


def test_oracle_runner_does_not_fabricate_llm_metrics():
    result = evaluate_oracle_case({
        "case_id": "P1", "oracle_claim_labels": True, "task_requirements": ["answer_change"],
        "requirement_coverage": {"answer_change": True},
        "claims": [{"claim_id": "n", "claim_type": "numeric_fact", "evidence_ids": ["ev_1"], "is_correct": False}],
    })
    assert result["deterministic_known_error_claim_ids"] == ["n"]
    assert result["llm_verdict"] == "pending_not_executed"


def test_draft_cannot_be_promoted_to_oracle_by_runner():
    with pytest.raises(ValueError, match="explicitly verified"):
        evaluate_oracle_case({"case_id": "draft", "oracle_claim_labels": False})


def test_pending_coverage_dictionary_is_not_treated_as_verified():
    result = evaluate_oracle_case({
        "oracle_claim_labels": True, "task_requirements": ["date"],
        "requirement_coverage": {"date": {"status": "pending_independent_review"}},
    })
    assert result["missing_requirements"] == ["date"]
    assert result["needs_semantic_review"] is True
