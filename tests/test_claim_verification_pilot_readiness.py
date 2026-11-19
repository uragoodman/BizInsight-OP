from evaluation.claim_verification.check_pilot_readiness import audit


def test_mutation_label_leak_and_incomplete_coverage_fail_readiness():
    report = audit([{
        "pilot_id": "P1", "provenance": "targeted_mutation", "answer": "value is 2",
        "atomic_claims": [{"text_span": "value is 2", "correctness": "injected_known_error"}],
        "requirement_coverage": {"value": {"status": "pending_independent_review"}},
        "human_verified": False, "oracle_claim_labels": False,
    }])
    assert report["ready"] == 0
    assert "ground_truth_in_reviewer_input" in report["cases"][0]["reasons"]
    assert "mutation_claims_do_not_cover_source_answer" in report["cases"][0]["reasons"]


def test_non_verbatim_span_fails_readiness():
    report = audit([{
        "pilot_id": "P2", "provenance": "real_trace_draft", "answer": "actual answer",
        "atomic_claims": [{"text_span": "invented fragment"}],
        "requirement_coverage": {}, "human_verified": True, "oracle_claim_labels": True,
    }])
    assert "claim_span_not_verbatim" in report["cases"][0]["reasons"]
