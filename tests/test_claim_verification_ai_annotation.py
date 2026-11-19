from evaluation.claim_verification.annotate_development import validate


def case():
    return {"answer": "订单下降 10%。原因不明。", "task_requirements": ["说明变化"], "evidence_bundles": [{"evidence_id": "e", "trusted": True}]}


def response():
    return {"claims": [{"span": "订单下降 10%", "type": "comparison", "evidence_ids": ["e"], "correctness": "supported"}], "requirements": [{"requirement": "说明变化", "status": "covered"}], "coverage_uncertain": True}


def test_valid_structure_is_not_human_gold():
    result = validate(case(), response())
    assert result["status"] == "structurally_valid"
    assert not result["oracle_claim_labels"] and not result["human_verified"]
    assert result["coverage_uncertain"] is True


def test_unknown_evidence_and_nonexistent_span_fail():
    data = response()
    data["claims"][0]["evidence_ids"] = ["missing"]
    assert validate(case(), data)["status"] == "invalid"
    data["claims"][0]["span"] = "invented"
    assert validate(case(), data)["status"] == "invalid"


def test_duplicate_span_is_ambiguous_and_rejected():
    data = case()
    data["answer"] += "订单下降 10%。"
    assert validate(data, response())["status"] == "invalid"


def test_operational_reference_separate_from_business_evidence():
    data = case()
    data.update({"operational_context": {"terminal_code": "CAPABILITY_GAP"}, "operational_context_provenance": "pre_critic_snapshot"})
    value = response()
    value["claims"][0].update({"type": "limitation_or_refusal", "evidence_ids": ["operational_context"]})
    valid = validate(data, value)
    assert valid["status"] == "structurally_valid"
    assert valid["claims"][0]["evidence_ids"] == []
    assert valid["claims"][0]["operational_reference"] == "pre_critic_snapshot"
    value["claims"][0]["type"] = "numeric_fact"
    assert validate(data, value)["status"] == "invalid"
