from evaluation.claim_verification.freeze_repaired_detection import REQUIREMENT, repair


def test_omission_explicitly_requires_units_and_removes_value_units():
    case = {"case_id": "H4-FBG047", "user_query": "比较客户数。", "answer": "当前值 4.00 人，变化率 -31.59 %。", "task_requirements": []}
    result = repair(case)
    assert "请明确说明" in result["user_query"]
    assert REQUIREMENT in result["task_requirements"]
    assert result["answer"] == "当前值 4.00，变化率 -31.59。"
    assert case["answer"] == "当前值 4.00 人，变化率 -31.59 %。"


def test_repair_preserves_existing_number_and_date():
    case = {"case_id": "H4-FBG052", "user_query": "比较。", "answer": "基线日 2017-08-20：3.0；当前日 2017-08-21：4.0。"}
    result = repair(case)
    assert result["answer"] == case["answer"]
    assert result["case_id"] == "H4-FBG052-unit-required"
