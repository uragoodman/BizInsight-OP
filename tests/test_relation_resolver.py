"""AGENT_PLAN Phase 7: conversation relation resolver tests."""

from __future__ import annotations

from unittest.mock import patch

from app.agents.relation_resolver import build_decision_payload, prior_evidence_ids, resolve_relation_llm


_CURRENT = {"task_id": "ctk_1", "status": "running", "objective": "分析 GMV 下降", "latest_trace_id": "trace-1"}


def test_continue_decision_reuses_whitelisted_evidence_only():
    raw = {"action": "CONTINUE", "task_id": "ctk_1",
           "reusable_evidence_ids": ["ev_ok_1", "ev_unknown"], "reason": "短追问，复用上轮证据"}
    with patch("app.llm.client.llm_extract", lambda *a, **k: raw):
        decision = resolve_relation_llm(_CURRENT, "那这是为什么？", None, ["ev_ok_1", "ev_ok_2"])
    assert decision["action"] == "CONTINUE"
    assert decision["task_id"] == "ctk_1"
    assert decision["reusable_evidence_ids"] == ["ev_ok_1"]  # ev_unknown is not in the prior trace


def test_invalid_action_and_missing_continue_target_degrade_to_none():
    with patch("app.llm.client.llm_extract", lambda *a, **k: {"action": "REVISIT"}):
        assert resolve_relation_llm(_CURRENT, "那这是为什么？", None, []) is None
    with patch("app.llm.client.llm_extract", lambda *a, **k: {"action": "CONTINUE"}):
        assert resolve_relation_llm(None, "那这是为什么？", None, []) is None
    with patch("app.llm.client.llm_extract", lambda *a, **k: "not-json"):
        assert resolve_relation_llm(_CURRENT, "那这是为什么？", None, []) is None


def test_new_and_clarify_decisions_are_mapped():
    with patch("app.llm.client.llm_extract", lambda *a, **k: {"action": "PARALLEL_NEW_TASK", "reason": "并行推进"}):
        assert resolve_relation_llm(_CURRENT, "再看看 7 月的情况", None, [])["action"] == "PARALLEL_NEW_TASK"
    with patch("app.llm.client.llm_extract", lambda *a, **k: {"action": "CLARIFY", "reason": "意图不明"}):
        assert resolve_relation_llm(_CURRENT, "帮我看看", None, [])["action"] == "CLARIFY"
    payload = build_decision_payload(_CURRENT, "那这是为什么？", None, ["ev_ok_1"])
    assert "ctk_1" in payload and "ev_ok_1" in payload


def test_relation_resolution_uses_nonstream_fast_operation():
    captured = {}

    def fake_extract(*_args, **kwargs):
        captured.update(kwargs)
        return {"action": "CLARIFY", "reason": "意图不明"}

    with patch("app.llm.client.llm_extract", fake_extract):
        decision = resolve_relation_llm(_CURRENT, "帮我看看", None, [])

    assert decision["action"] == "CLARIFY"
    assert captured == {
        "operation": "extract.relation_resolution",
        "interaction_policy": "nonstream_fast",
    }


def test_decision_payload_keeps_original_and_compiled_context_separate():
    import json

    payload = json.loads(build_decision_payload(
        _CURRENT, "按品类拆开看看", None, ["ev_ok_1"],
        {
            "standalone_question": "GMV 按品类拆开看看，日期 2018-08-01",
            "relation": "continue",
            "resolved_references": [{"value": "gmv", "source": "previous_turn_context"}],
            "known_slots": {"metric": {"value": "gmv", "source_turn_id": "turn-1"}},
            "context_sources": ["previous_turn_context"],
            "source_turn_id": "turn-1",
        },
    ))
    assert payload["new_message"] == "按品类拆开看看"
    assert payload["resolved_message"] == "GMV 按品类拆开看看，日期 2018-08-01"
    assert payload["context_resolution"]["source_turn_id"] == "turn-1"


def test_service_compiles_message_from_authorized_same_scope_turn_context():
    from app.tasks import service

    prior = {
        "source_turn_id": "turn-1",
        "scope_fingerprint": "scope-fp",
        "context_slots": {
            "metric": {"value": "gmv", "source": "current_turn"},
            "date": {"value": "2018-08-01", "source": "current_turn"},
        },
    }
    with patch.object(service.ChatHistoryRepository, "latest_completed_context", return_value=prior) as load:
        result = service._resolve_message_context(object(), "session-1", "按品类拆开看看", "scope-fp")
    load.assert_called_once_with("session-1", scope_fingerprint="scope-fp")
    assert result["relation"] == "continue"
    assert result["standalone_question"] != "按品类拆开看看"
    assert result["source_turn_id"] == "turn-1"


def test_service_routes_with_compiled_message_and_keeps_provenance():
    from app.core.config import get_settings
    from app.tasks import service

    compiled = {
        "standalone_question": "GMV 按品类拆开看看，日期 2018-08-01",
        "relation": "continue",
        "resolved_references": [{"value": "gmv", "source": "previous_turn_context"}],
        "known_slots": {"metric": {"value": "gmv"}},
        "context_sources": ["previous_turn_context"],
        "source_turn_id": "turn-1",
        "needs_clarification": False,
    }
    seen = {}
    def resolve(task, message, scope, evidence_ids, context_resolution, available_tasks):
        seen.update(message=message, context_resolution=context_resolution, available_tasks=available_tasks)
        return {"action": "CONTINUE", "task_id": "ctk_1", "reusable_evidence_ids": [], "reason": "same task"}

    with patch.object(service, "task_context", return_value={"tasks": [_CURRENT], "current_task": _CURRENT, "memory": {}}), \
         patch.object(service, "_resolve_message_context", return_value=compiled), \
         patch.object(service, "prior_evidence_ids", return_value=[]), \
         patch.object(service, "resolve_relation_llm", side_effect=resolve), \
         patch.object(get_settings(), "relation_resolver_llm_enabled", True):
        result = service.relation(object(), "session-1", "按品类拆开看看", None, scope_fingerprint="scope-fp")

    assert seen["message"] == "按品类拆开看看"
    assert seen["context_resolution"]["source_turn_id"] == "turn-1"
    assert result["action"] == "continue"
    assert result["resolved_message"] == compiled["standalone_question"]


def test_prior_evidence_ids_reads_persisted_trace_whitelist():
    with patch("app.harness.storage.read_trace", return_value={
        "state_snapshot": {"evidence_bundles": [{"evidence_id": "ev_a"}, {"evidence_id": "ev_b"}]},
    }):
        assert prior_evidence_ids("trace-1") == []
    assert prior_evidence_ids(None) == []
    with patch("app.harness.storage.read_trace", side_effect=RuntimeError("storage down")):
        assert prior_evidence_ids("trace-1") == []


def _patched_service(current_task, requested=None):
    from app.tasks import service

    data = {"tasks": [current_task] if current_task else [], "current_task": current_task, "memory": {}}
    return (
        patch.object(service, "task_context", lambda context, session_id: data),
        patch.object(service, "get_task", lambda context, task_id: current_task if task_id == "ctk_1" else None),
    )


def _relation(current_task, message, scope=None, requested=None, llm_enabled=True, expected_data_version="manifest-a"):
    from app.core.config import get_settings
    from app.tasks import service

    patches = _patched_service(current_task, requested)
    with patch.object(get_settings(), "relation_resolver_llm_enabled", llm_enabled), patches[0], patches[1]:
        return service.relation(
            None, "session-1", message, scope, requested,
            scope_fingerprint="scope-a",
            expected_data_version=expected_data_version,
        )


def test_service_relation_maps_llm_decisions_without_breaking_api_contract():
    decision = {"action": "CONTINUE", "task_id": "ctk_1", "reusable_evidence_ids": ["ev_ok_1"], "reason": "短追问"}
    with patch("app.tasks.service.resolve_relation_llm", lambda *a, **k: decision), \
         patch("app.tasks.service.prior_evidence_ids", lambda *a, **k: ["ev_ok_1"]):
        result = _relation(_CURRENT, "那这是为什么？")
    assert result["action"] == "continue"
    assert result["task_id"] == "ctk_1"
    assert result["reusable_evidence_ids"] == ["ev_ok_1"]

    decision = {"action": "PARALLEL_NEW_TASK", "task_id": None, "reusable_evidence_ids": [], "reason": "并行"}
    with patch("app.tasks.service.resolve_relation_llm", lambda *a, **k: decision):
        assert _relation(_CURRENT, "再看看 7 月")["action"] == "new"

    decision = {"action": "CLARIFY", "task_id": None, "reusable_evidence_ids": [], "reason": "意图不明"}
    with patch("app.tasks.service.resolve_relation_llm", lambda *a, **k: decision):
        assert _relation(_CURRENT, "帮我看看")["action"] == "clarify"


def test_service_relation_falls_back_to_keywords_when_llm_unavailable():
    with patch("app.tasks.service.resolve_relation_llm", lambda *a, **k: None):
        scope = {"seller_id": "s1", "start_date": "2018-01-01", "end_date": "2018-01-31"}
        current = {**_CURRENT, "scope_data": scope}
        result = _relation(current, "那这是为什么？继续看看", scope)
    assert {key: result[key] for key in ("action", "task_id", "reason")} == {
        "action": "continue", "task_id": "ctk_1", "reason": "same_scope_follow_up",
    }
    assert result["agent_run"]["domain"] == "relation"
    assert result["agent_run"]["decision_driver"] == "workflow_fallback"
    with patch("app.tasks.service.resolve_relation_llm", lambda *a, **k: None):
        assert _relation(current, "帮我看看别的东西", scope)["action"] == "clarify"


def test_service_relation_llm_disabled_keeps_keyword_path():
    from unittest.mock import MagicMock
    scope = {"seller_id": "s1", "start_date": "2018-01-01", "end_date": "2018-01-31"}
    current = {**_CURRENT, "scope_data": scope}
    should_not_be_called = MagicMock(return_value={"action": "CLARIFY", "task_id": None, "reusable_evidence_ids": [], "reason": ""})
    with patch("app.tasks.service.resolve_relation_llm", should_not_be_called):
        result = _relation(current, "那这是为什么？继续看看", scope, llm_enabled=False)
    should_not_be_called.assert_not_called()
    assert result["action"] == "continue"


def test_continue_can_select_only_a_task_from_session_candidates():
    tasks = [
        {"task_id": "ctk_1", "status": "completed", "title": "GMV 分析"},
        {"task_id": "ctk_2", "status": "running", "title": "退款分析"},
    ]
    raw = {"action": "CONTINUE", "task_id": "ctk_2", "reason": "用户指向退款任务"}
    with patch("app.llm.client.llm_extract", lambda *a, **k: raw):
        selected = resolve_relation_llm(_CURRENT, "继续退款分析", None, [], available_tasks=tasks)
    assert selected["task_id"] == "ctk_2"

    raw["task_id"] = "ctk_other_tenant"
    with patch("app.llm.client.llm_extract", lambda *a, **k: raw):
        assert resolve_relation_llm(_CURRENT, "继续", None, [], available_tasks=tasks) is None


def test_prior_evidence_requires_matching_scope_and_data_version():
    trace = {"state_snapshot": {
        "current_query_scope_fingerprint": "scope-a",
        "data_version_manifest_hash": "manifest-a",
        "evidence_bundles": [
            {"evidence_id": "ev_ok", "trusted": True},
            {"evidence_id": "ev_untrusted", "trusted": False},
        ],
    }}
    with patch("app.harness.storage.read_trace", return_value=trace):
        assert prior_evidence_ids(
            "trace-1", expected_scope_fingerprint="scope-a", expected_data_version="manifest-a",
        ) == ["ev_ok"]
        assert prior_evidence_ids(
            "trace-1", expected_scope_fingerprint="scope-b", expected_data_version="manifest-a",
        ) == []
        assert prior_evidence_ids(
            "trace-1", expected_scope_fingerprint="scope-a", expected_data_version="manifest-b",
        ) == []
        assert prior_evidence_ids("trace-1", expected_scope_fingerprint="scope-a") == []


def test_task_state_is_bounded_and_resets_claims_when_scope_or_data_changes():
    from app.tasks.service import _bounded_task_state

    row = {
        "objective": "分析 GMV",
        "task_state_version": 4,
        "task_state": {
            "schema_version": "task-state-v1", "version": 1,
            "scope_fingerprint": "scope-a", "data_version_manifest_hash": "manifest-a",
            "approved_claims": [{"value": "旧结论", "source_evidence_ids": ["ev-old"]}],
            "hypothesis_chain_summary": {"gmv": [{"hypothesis": "old"}]},
            "evidence_refs": [{"evidence_id": "ev-old"}],
            "recent_turns": [{"turn_id": f"turn-{i}"} for i in range(25)],
        },
    }
    next_state = _bounded_task_state(row, {
        "scope_fingerprint": "scope-b", "data_version_manifest_hash": "manifest-b",
        "approved_claims": [{"value": "新结论", "source_evidence_ids": ["ev-new"]}],
        "hypothesis_chain_summary": {"gmv": [{"hypothesis": "new"}]},
        "evidence_refs": [{"evidence_id": "ev-new", "trusted": True}],
        "completed_steps": ["triage", "data_query"],
    }, status="completed", trace_id="trace-new", turn_id="turn-new")

    assert next_state["state_version"] == 5
    assert next_state["approved_claims"] == [{"value": "新结论", "source_evidence_ids": ["ev-new"]}]
    assert [item["evidence_id"] for item in next_state["evidence_refs"]] == ["ev-new"]
    assert len(next_state["recent_turns"]) == 20
    assert next_state["continuity_reset_reason"] == "scope_or_data_version_changed"

def test_explicit_empty_task_candidate_list_rejects_llm_continue():
    raw = {"action": "CONTINUE", "task_id": "ctk_1"}
    with patch("app.llm.client.llm_extract", lambda *a, **k: raw):
        assert resolve_relation_llm(_CURRENT, "continue", None, [], available_tasks=[]) is None
