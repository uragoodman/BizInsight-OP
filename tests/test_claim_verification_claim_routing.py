from evaluation.claim_verification.claim_routing import plan


def _input():
    case = {"answer": "总变化 -18。物流延迟导致下降。", "task_requirements": ["change"]}
    annotations = {"oracle_claim_labels": True, "coverage_verified": True,
                   "requirement_coverage": {"change": "covered"}, "claims": [
                       {"claim_id": "c1", "start": 0, "end": 7, "text_span": "总变化 -18", "claim_type": "numeric_fact", "evidence_ids": ["e"]},
                       {"claim_id": "c2", "start": 8, "end": 16, "text_span": "物流延迟导致下降", "claim_type": "causal_claim", "evidence_ids": []}]}
    return case, annotations


def test_label_alone_never_counts_as_deterministic_verification():
    case, annotations = _input()
    result = plan(case, annotations, [])
    assert [c["route"] for c in result["claim_decisions"]] == ["semantic", "semantic"]
    assert result["semantic_scope"] == "claim_batch_and_global_checks"


def test_matching_actual_check_routes_only_covered_claim():
    case, annotations = _input()
    check = {"start": 0, "end": 7, "evidence_id": "e", "matched": True}
    result = plan(case, annotations, [check])
    assert [c["route"] for c in result["claim_decisions"]] == ["deterministic", "semantic"]
    assert result["whole_answer_verified"] is False
    check["matched"] = False
    assert plan(case, annotations, [check])["claim_decisions"][0]["check_status"] == "mismatch"


def test_bad_coverage_and_invalid_spans_escalate_whole_answer():
    case, annotations = _input()
    annotations["coverage_verified"] = False
    annotations["claims"][0]["text_span"] = "wrong"
    result = plan(case, annotations, [])
    assert result["semantic_scope"] == "whole_answer"
    assert set(result["escalation_reasons"]) == {"claim_coverage_unverified", "invalid_claim_span"}


def test_partial_numeric_check_does_not_cover_compound_claim():
    case, annotations = _input()
    check = {"start": 4, "end": 7, "evidence_id": "e", "matched": True}
    assert plan(case, annotations, [check])["claim_decisions"][0]["route"] == "semantic"
