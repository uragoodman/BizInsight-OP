from __future__ import annotations

import hashlib
import json

from app.graph.critic_review_policy import decide_optional_llm_review


def _passing_reviews():
    return {
        "request_contract_review": {"status": "passed"},
        "data_review": {"status": "passed"},
        "claim_review": {"status": "approved"},
        "numeric_review": {"status": "not_applicable"},
        "completeness_review": {"status": "skipped"},
        "causal_review": {"status": "approved"},
    }


def _deterministic_answer_state(answer: str = "已验证的确定性分解答案。"):
    ledger = {
        "schema_version": "registered_decomposition_answer_contract_v1",
        "metric_id": "avg_order_value",
        "method_id": "symmetric_ratio_v1",
        "evidence_ids": ["ev-1"],
        "business_cause_confirmed": False,
    }
    ledger_hash = hashlib.sha256(
        json.dumps(ledger, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return {
        "answer": answer,
        "adaptive_answer_contract": {
            "schema_version": ledger["schema_version"], "ledger": ledger,
            "ledger_sha256": ledger_hash, "rendered_answer": answer,
            "rendered_answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        },
    }


def test_adaptive_policy_skips_only_fully_validated_registered_decomposition():
    decision = decide_optional_llm_review(
        task_type="metric_query", registered_decomposition_review=True,
        terminal_execution_failure=False, state=_deterministic_answer_state(),
        deterministic_reviews={
            **_passing_reviews(),
            "data_review": {"status": "passed", "target_metric_id": "avg_order_value", "method_id": "symmetric_ratio_v1", "evidence_ids": ["ev-1"]},
        },
    )

    assert decision["action"] == "skip"
    assert decision["reason_code"] == "registered_decomposition_all_deterministic_gates_passed"
    assert decision["passed_gates"]


def test_adaptive_policy_fails_closed_when_final_answer_differs_from_rendered_contract():
    state = _deterministic_answer_state()
    state["answer"] += "\n订单量是客单价上涨的主要推动因素。"
    decision = decide_optional_llm_review(
        task_type="metric_query", registered_decomposition_review=True,
        terminal_execution_failure=False, state=state,
        deterministic_reviews={
            **_passing_reviews(),
            "data_review": {"status": "passed", "target_metric_id": "avg_order_value", "method_id": "symmetric_ratio_v1", "evidence_ids": ["ev-1"]},
        },
    )
    assert decision["action"] == "repair"
    assert decision["reason_code"] == "answer_not_deterministic_render"


def test_adaptive_policy_rejects_a_tampered_answer_contract():
    state = _deterministic_answer_state()
    state["adaptive_answer_contract"]["ledger_sha256"] = "not-a-real-hash"
    decision = decide_optional_llm_review(
        task_type="metric_query", registered_decomposition_review=True,
        terminal_execution_failure=False, state=state,
        deterministic_reviews={
            **_passing_reviews(),
            "data_review": {"status": "passed", "target_metric_id": "avg_order_value", "method_id": "symmetric_ratio_v1", "evidence_ids": ["ev-1"]},
        },
    )
    assert decision["action"] == "reject"
    assert decision["reason_code"] == "answer_contract_ledger_hash_mismatch"


def test_adaptive_policy_fails_closed_for_diagnostics_errors_and_failed_gates():
    diagnostic = decide_optional_llm_review(
        task_type="anomaly_analysis", registered_decomposition_review=False,
        terminal_execution_failure=False, state={"answer": "诊断结论"},
        deterministic_reviews=_passing_reviews(),
    )
    assert diagnostic["action"] == "review"
    assert diagnostic["reason_code"] == "high_or_unclassified_risk_path"

    failed_reviews = _passing_reviews()
    failed_reviews["claim_review"] = {"status": "rejected"}
    failed_reviews["data_review"] = {
        "status": "passed", "target_metric_id": "avg_order_value",
        "method_id": "symmetric_ratio_v1", "evidence_ids": ["ev-1"],
    }
    failed = decide_optional_llm_review(
        task_type="metric_query", registered_decomposition_review=True,
        terminal_execution_failure=False, state=_deterministic_answer_state(),
        deterministic_reviews=failed_reviews,
    )
    assert failed["action"] == "review"
    assert failed["reason_code"] == "deterministic_gate_not_passed"
    assert failed["failed_gates"] == ["claim_review"]

    errored = decide_optional_llm_review(
        task_type="metric_query", registered_decomposition_review=True,
        terminal_execution_failure=False,
        state={**_deterministic_answer_state(), "errors": [{"code": "tool_timeout"}]},
        deterministic_reviews=_passing_reviews(),
    )
    assert errored["action"] == "review"
    assert errored["reason_code"] == "workflow_errors_present"


def test_adaptive_policy_preserves_terminal_safe_stop_behavior():
    decision = decide_optional_llm_review(
        task_type="anomaly_analysis", registered_decomposition_review=False,
        terminal_execution_failure=True, state={"answer": "无法完成，已安全停止。"},
        deterministic_reviews=_passing_reviews(),
    )
    assert decision["action"] == "not_run"
    assert decision["reason_code"] == "upstream_terminal_execution_failure"
