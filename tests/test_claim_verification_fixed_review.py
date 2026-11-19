import pytest

from evaluation.claim_verification.fixed_review import review


def _case():
    return {"case_id": "blind", "answer": "可能的原因未确认。", "user_query": "为什么？", "task_requirements": [], "evidence_bundles": [], "request_contract": {}}


def test_off_is_inconclusive_and_routed_calls_semantic():
    assert review(_case(), "off")["verdict"] == "inconclusive"
    seen = []
    result = review(_case(), "routed", semantic_fn=lambda system, payload: seen.append(payload) or {"verdict": "passed", "issues": []})
    assert len(seen) == 1
    assert result["verdict"] == "model_no_issue_found"
    assert result["whole_answer_verified"] is False


def test_unavailable_semantic_review_is_not_passed():
    assert review(_case(), "routed", semantic_fn=lambda *_: None)["verdict"] == "review_incomplete"


def test_hidden_labels_rejected():
    with pytest.raises(ValueError, match="scoring labels"):
        review({**_case(), "mutation_type": "error"}, "off")


def test_routed_uses_safe_annotations_without_sending_correctness_labels():
    import json
    case = _case()
    case["route_annotations"] = {"oracle_claim_labels": True, "coverage_verified": True, "claims": [{"claim_id": "c", "start": 0, "end": 7, "text_span": "可能的原因未确", "claim_type": "causal_claim", "evidence_ids": [], "correctness": "secret"}], "requirement_coverage": {}}
    seen = []
    result = review(case, "routed", semantic_fn=lambda _system, payload: seen.append(json.loads(payload)) or {"verdict": "passed", "issues": []})
    assert result["routing"]["semantic_scope"] == "claim_batch_and_global_checks"
    assert "correctness" not in str(seen[0]) and "secret" not in str(seen[0])
    assert seen[0]["required_global_checks"] == ["cross_claim_consistency", "task_completeness"]


def test_routed_receives_snapshot_operational_context_and_provenance():
    import json
    case = {**_case(), "operational_context": {"terminal_code": "CAPABILITY_GAP"}, "operational_context_provenance": "pre_critic_snapshot"}
    seen = []
    review(case, "routed", semantic_fn=lambda _system, payload: seen.append(json.loads(payload)) or {"verdict": "passed", "issues": []})
    assert seen[0]["operational_context"]["terminal_code"] == "CAPABILITY_GAP"
    assert seen[0]["operational_context_provenance"] == "pre_critic_snapshot"


def test_full_each_review_receives_same_trusted_evidence_without_duplicate(monkeypatch):
    from evaluation.claim_verification import fixed_review
    seen = []
    def fake(*args):
        seen.append(args)
        return {"verdict": "passed", "issues": []}
    monkeypatch.setattr(fixed_review, "llm_review_groundedness", fake)
    monkeypatch.setattr(fixed_review, "llm_review_evidence_completeness", fake)
    monkeypatch.setattr(fixed_review, "llm_review_logic_consistency", fake)
    case = {**_case(), "evidence_bundles": [{"evidence_id": "ok", "trusted": True}, {"evidence_id": "bad", "trusted": False}]}
    case["task_type"] = "metric_query"
    assert review(case, "full")["verdict"] == "model_no_issue_found"
    assert len(seen) == 3 and seen[0][1] == ""
    assert seen[1][1] == seen[2][1] == "metric_query"
    assert all(args[2]["verified_numeric_evidence"] == [{"evidence_id": "ok", "trusted": True}] for args in seen)


def test_full_review_journals_each_sdk_call_and_off_sends_none(monkeypatch):
    from types import SimpleNamespace
    from app.core.execution_context import active_state
    from app.llm import client as llm
    from app.llm.call_config import LLMCallConfig
    from app.core.config import get_settings
    from evaluation.claim_verification import fixed_review

    class FakeClient:
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            state = active_state()
            assert state is not None and state["evaluation_run_id"] == "run-full"
            assert state["llm_call_config"]["mode"] == "provider_ab_experiment"
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"verdict":"passed","issues":[]}'), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

    client = FakeClient()
    provider = llm._Provider("primary", client, lambda operation, explicit, quality: ("qwen3.8-flash", "standard"))
    monkeypatch.setattr(llm, "_providers", lambda: [provider])
    monkeypatch.setattr(llm, "_quota_available", lambda: True)
    monkeypatch.setattr(llm, "execute_with_resilience", lambda operation, *a, **k: operation())
    monkeypatch.setattr(llm, "record_usage", lambda **kwargs: None)
    # Each reviewer still traverses the real public llm_extract and SDK recorder.
    monkeypatch.setattr(fixed_review, "llm_review_groundedness", lambda *_: llm.llm_extract("sys", "ground"))
    monkeypatch.setattr(fixed_review, "llm_review_evidence_completeness", lambda *_: llm.llm_extract("sys", "complete"))
    monkeypatch.setattr(fixed_review, "llm_review_logic_consistency", lambda *_: llm.llm_extract("sys", "consistent"))

    journal = []
    config = LLMCallConfig.provider_ab_experiment(get_settings(), max_completion_tokens=4096, timeout_seconds=300)
    result = review(_case(), "full", evaluation_run_id="run-full", event_sink=journal.append, llm_call_config=config)
    assert result["verdict"] == "model_no_issue_found"
    assert len(client.calls) == 3
    assert all(call.get("max_completion_tokens") == 4096 and "max_tokens" not in call for call in client.calls)
    starts = [item for item in journal if item["event"] == "sdk_invocation_started"]
    finishes = [item for item in journal if item["event"] == "sdk_invocation_finished"]
    stages = [item for item in journal if item["event"] == "review_stage_finished"]
    assert len(starts) == len(finishes) == 3
    assert [item["stage"] for item in stages] == ["groundedness", "completeness", "consistency"]
    assert all(item["run_id"] == "run-full" for item in starts + finishes + stages)
    assert len({item["call_id"] for item in starts}) == 3
    assert all(start["attempt_id"] == finish["attempt_id"] for start, finish in zip(starts, finishes))
    assert all(item["usage_status"] == "complete" for item in journal if item["event"] == "usage_recorded")

    off_journal = []
    off = review(_case(), "off", evaluation_run_id="run-off", event_sink=off_journal.append, llm_call_config=config)
    assert off["verdict"] == "inconclusive"
    assert not [item for item in off_journal if item["event"] == "sdk_invocation_started"]
    assert not off["llm_calls"]


def test_capability_stop_requires_exact_authorized_snapshot_and_no_sql():
    from evaluation.claim_verification.fixed_review import _validated_capability_stop
    case = {"request_contract": {"operation_intent": "dimensional_statistical_contribution", "metric_id": "m", "dimension": "d"},
            "operational_context_provenance": "pre_critic_snapshot", "operational_context": {"terminal_code": "CAPABILITY_GAP", "decision_records": [
                {"limitation_code": "CAPABILITY_GAP", "metric_id": "m", "dimension": "d", "operation": "metric_contribution", "authorization_checked": True, "registry_capability_checked": True, "sql_executed": False}]}}
    assert _validated_capability_stop(case)["role"] == "operational_limit_not_numeric_evidence"
    case["operational_context"]["decision_records"][0]["sql_executed"] = True
    assert _validated_capability_stop(case) is None
    case["operational_context"]["decision_records"][0]["sql_executed"] = False
    case["request_contract"]["dimension"] = "different"
    assert _validated_capability_stop(case) is None
