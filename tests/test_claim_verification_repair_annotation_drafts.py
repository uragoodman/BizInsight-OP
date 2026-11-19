import pytest
from evaluation.claim_verification.repair_annotation_drafts import repair


def test_percentage_repairs_only_exact_source_and_keeps_uncertainty():
    case = {"case_id": "a33c7d15b5834e04c97b", "answer": "整体下降 0.59（-20.1%）"}
    raw = {"claims": [{}, {}, {}, {"span": "整体下降 0.59"}, {"span": "(-20.1%)"}]}
    result, actions = repair(case, raw)
    assert result["claims"][3]["span"] == case["answer"]
    assert result["claims"][3]["correctness"] == "uncertain"
    assert len(result["claims"]) == 4 and actions
    assert len(raw["claims"]) == 5
    with pytest.raises(ValueError, match="not unique"):
        repair({**case, "answer": "different"}, raw)


def test_capability_repair_refuses_false_sql_state():
    case = {"case_id": "9db34cd00433eeb8ce25", "operational_context": {"terminal_code": "CAPABILITY_GAP", "decision_records": [{"sql_executed": True, "limitation_code": "CAPABILITY_GAP"}]}}
    with pytest.raises(ValueError, match="capability state"):
        repair(case, {})
