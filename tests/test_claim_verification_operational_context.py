from evaluation.claim_verification.operational_context import for_review


def test_original_answer_and_derived_summary_never_exposed():
    source = {"terminal_code": "CAPABILITY_GAP", "analysis_result": {"summary": "ORIGINAL ANSWER"},
              "decision_records": [{"metric_id": "m", "limitation_code": "CAPABILITY_GAP", "answer": "ORIGINAL ANSWER"}],
              "investigation_state": {"stop_reason": {"code": "X", "detail": "ORIGINAL ANSWER"},
                                      "branches": [{"status": "blocked", "label": "ORIGINAL ANSWER"}]}}
    result = for_review(source)
    assert "ORIGINAL ANSWER" not in repr(result)
    assert result["decision_records"][0]["limitation_code"] == "CAPABILITY_GAP"
    assert result["branches"][0]["status"] == "blocked"


def test_missing_snapshot_is_not_fabricated():
    assert for_review(None) is None
