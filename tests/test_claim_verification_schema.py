import pytest

from evaluation.claim_verification import annotation_schema
from evaluation.claim_verification.annotation_schema import CLAIM_ROUTES, route_for, validate_annotation_row
from evaluation.claim_verification.claim_routing import plan
from evaluation.claim_verification.runner import route_claim


def _row(**overrides):
    answer = "总变化 -18。物流延迟导致下降。"
    row = {
        "case_id": "C1",
        "answer": answer,
        "claims": [
            {"claim_id": "C1", "start": 0, "end": 8, "text_span": "总变化 -18。", "claim_type": "numeric_fact",
             "evidence_ids": ["ev_1"], "correctness": "supported", "requirement_ids": []},
            {"claim_id": "C2", "start": 8, "end": 17, "text_span": "物流延迟导致下降。", "claim_type": "causal_claim",
             "evidence_ids": [], "correctness": "unsupported", "requirement_ids": []},
        ],
        "task_requirements": ["explain"],
        "requirement_coverage": {"explain": "covered"},
        "provenance": "real_failure",
        "expected_decision": "needs_semantic_review",
        "annotation_status": "human_verified",
        "human_verified": True,
        "oracle_claim_labels": True,
    }
    row.update(overrides)
    return row


def test_fully_labelled_row_is_structurally_valid():
    result = validate_annotation_row(_row())
    assert result["issues"] == []
    assert result["oracle_claim_labels"] is True


def test_oracle_requires_human_verification():
    result = validate_annotation_row(_row(human_verified=False, annotation_status="pending_human_review"))
    assert "oracle_without_human_verification" in result["issues"]
    assert result["oracle_claim_labels"] is False


def test_numeric_claim_without_evidence_is_rejected_but_causal_is_not():
    claims = _row()["claims"]
    claims[0]["evidence_ids"] = []
    assert "claim_0_missing_evidence_binding" in validate_annotation_row(_row(claims=claims))["issues"]


def test_span_must_match_offsets_and_be_unique():
    claims = _row()["claims"]
    claims[0]["text_span"] = "missing"
    assert "claim_0_invalid_span" in validate_annotation_row(_row(claims=claims))["issues"]


def test_unverified_requirement_coverage_blocks_readiness():
    result = validate_annotation_row(_row(requirement_coverage={"explain": "uncertain"}))
    assert result["issues"] == []
    assert result["structurally_valid"] is True
    assert "requirement_coverage_uncertain:explain" in result["unresolved"]
    assert result["ready_for_benchmark"] is False


def test_ready_row_requires_no_unresolved_labels():
    assert validate_annotation_row(_row())["ready_for_benchmark"] is True


def test_human_verification_pending_is_unresolved_not_invalid():
    result = validate_annotation_row(_row(human_verified=False, annotation_status="pending_human_review", oracle_claim_labels=False))
    assert result["structurally_valid"] is True
    assert "human_verification_pending" in result["unresolved"]
    assert result["ready_for_benchmark"] is False


def test_pass_decision_requires_every_claim_supported():
    claims = _row()["claims"]
    claims[1]["correctness"] = "unsupported"
    result = validate_annotation_row(_row(claims=claims, expected_decision="pass"))
    assert "pass_decision_with_unsupported_claim" in result["issues"]


def test_unknown_type_is_not_a_valid_annotation_type():
    claims = _row()["claims"]
    claims[0]["claim_type"] = "numeric"
    assert "claim_0_invalid_type" in validate_annotation_row(_row(claims=claims))["issues"]


@pytest.mark.parametrize("claim_type", ["unknown", "made_up_type"])
def test_both_routers_escalate_unrecognised_or_unknown(claim_type):
    claim = {"claim_id": "c", "claim_type": claim_type, "evidence_ids": ["ev_1"]}
    assert route_claim(claim).route == "needs_semantic_review"
    assert route_for(claim_type) == annotation_schema.ROUTE_SEMANTIC


def test_limitation_or_refusal_is_semantic_in_every_router():
    answer = "无法确认。"
    assert CLAIM_ROUTES["limitation_or_refusal"] == annotation_schema.ROUTE_SEMANTIC
    claim = {"claim_id": "c", "claim_type": "limitation_or_refusal", "evidence_ids": []}
    assert route_claim(claim).route == "needs_semantic_review"
    annotations = {"oracle_claim_labels": True, "coverage_verified": True, "requirement_coverage": {}, "claims": [
        {"claim_id": "c", "start": 0, "end": len(answer), "text_span": answer, "claim_type": "limitation_or_refusal", "evidence_ids": []},
    ]}
    assert plan({"answer": answer, "task_requirements": []}, annotations, [])["claim_decisions"][0]["route"] == "semantic"


def test_definition_or_scope_has_no_check_producer_and_stays_semantic_in_plan():
    answer = "范围是 2017-01-05。"
    annotations = {"oracle_claim_labels": True, "coverage_verified": True, "requirement_coverage": {}, "claims": [
        {"claim_id": "c", "start": 0, "end": len(answer), "text_span": answer, "claim_type": "definition_or_scope", "evidence_ids": ["ev_1"]},
    ]}
    assert plan({"answer": answer, "task_requirements": []}, annotations, [])["claim_decisions"][0]["route"] == "semantic"
