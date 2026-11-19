from evaluation.claim_verification.audit_ai_annotation import audit


def test_uncertain_and_invalid_are_never_promoted_to_oracle():
    result = audit({"case_id": "c", "answer": "x", "evidence_bundles": []}, {"validation": {"status": "invalid", "coverage_uncertain": True, "issues": ["bad"]}})
    assert {"bad", "coverage_uncertain", "invalid_structure"} <= set(result["issues"])
    assert result["review_status"] == "needs_manual_or_ai_review"
    assert not result["human_verified"] and not result["oracle_claim_labels"]
