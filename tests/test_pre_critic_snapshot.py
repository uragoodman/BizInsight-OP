from app.graph.pre_critic_snapshot import capture_pre_critic_answer_snapshot


def test_capture_preserves_json_copy_and_hash_across_later_mutations():
    state = {
        "trace_id": "trace-1",
        "task_type": "anomaly_analysis",
        "user_message": "GMV 为什么下降？",
        "answer": "订单量下降。",
        "final_answer": "订单量下降。",
        "claims": [{"text": "订单量下降", "source_evidence_ids": ["ev-1"]}],
        "evidence_bundles": [{"evidence_id": "ev-1", "value": {"orders": -12}}],
        "context_compilation": {"request_contract": {"requirements": ["解释变化"]}},
        "llm_calls": [{"call_id": "call-1"}],
        "llm_token_summary": {"total_tokens": 123},
        "_revision_counts": {"generate_answer": 0},
        "critic_llm_review_mode": "full",
        "diagnostic_step_state": {"target": "data_insufficient"},
    }

    snapshot = capture_pre_critic_answer_snapshot(state)
    state["answer"] = "被后续节点改写"
    state["claims"][0]["text"] = "被后续节点改写"
    state["evidence_bundles"][0]["value"]["orders"] = 0
    state["diagnostic_step_state"]["target"] = "completed"

    assert snapshot["snapshot_index"] == 1
    assert snapshot["answer"] == "订单量下降。"
    assert snapshot["claims"][0]["text"] == "订单量下降"
    assert snapshot["evidence_bundles"][0]["value"]["orders"] == -12
    assert snapshot["evidence_bundle_ids"] == ["ev-1"]
    assert snapshot["llm_call_offset"] == 1
    assert snapshot["schema_version"] == "pre_critic_answer_snapshot_v2"
    assert snapshot["operational_context"]["diagnostic_step_state"]["target"] == "data_insufficient"
    assert len(snapshot["snapshot_sha256"]) == 64


def test_capture_keeps_each_critic_entry_instead_of_overwriting_drafts():
    state = {"answer": "首稿", "claims": [], "evidence_bundles": [], "llm_calls": []}

    first = capture_pre_critic_answer_snapshot(state)
    state["answer"] = "返工稿"
    second = capture_pre_critic_answer_snapshot(state)

    assert [item["answer"] for item in state["pre_critic_answer_snapshots"]] == ["首稿", "返工稿"]
    assert first["snapshot_index"] == 1
    assert second["snapshot_index"] == 2
