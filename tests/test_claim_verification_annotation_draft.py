from evaluation.claim_verification.build_ai_assisted_annotation_draft import annotate_case


def test_safe_stop_becomes_limitation_without_invented_evidence() -> None:
    result = annotate_case({
        "original_answer_seen_by_critic": "未执行数据查询。当前范围不支持按支付方式拆分；未改用其他维度代替。",
        "evidence_bundles": [], "runtime_claims_unverified": [], "task_requirements": ["no unsupported substitute"],
        "request_contract": {},
    })
    assert result["atomic_claims"][0]["claim_type"] == "limitation_or_refusal"
    assert result["atomic_claims"][0]["evidence_ids"] == []
    assert result["oracle_claim_labels"] is False


def test_contribution_is_not_labelled_as_causal() -> None:
    result = annotate_case({
        "original_answer_seen_by_critic": "这些是数值比较与统计贡献，不证明业务因果。",
        "evidence_bundles": [{"evidence_id": "ev-1"}],
        "runtime_claims_unverified": [{"claim_type": "analysis", "value": "订单量变化 -18"}],
        "task_requirements": [],
        "request_contract": {"operation_intent": "dimensional_statistical_contribution"},
    })
    assert result["atomic_claims"][0]["claim_type"] == "comparison"
    assert result["atomic_claims"][1]["claim_type"] == "limitation_or_refusal"
