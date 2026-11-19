"""Offline contract checks for structured diagnostic context in Critic reviews."""

from __future__ import annotations

import copy
import json
import socket
from pathlib import Path

import httpx
import pytest


TRACE_DIR = Path("docs/testing/provider_aov_token_stop_rerun_20261012/traces")
OFF_TRACE = TRACE_DIR / "provider-trial-aov_ratio-critic_llm_review_off-c5179b4ee02a.json"
FULL_TRACE = TRACE_DIR / "provider-trial-aov_ratio-full-e02c3b4fec26.json"
PAIR_OFF_TRACE = Path(
    "docs/testing/provider_aov_causal_guard_pair_20261012/traces/"
    "provider-trial-aov_ratio-critic_llm_review_off-3b8c7e260f7c.json"
)


def _saved_state(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return copy.deepcopy(payload["state_snapshot"])


def _install_network_block(monkeypatch):
    attempts: list[str] = []

    def blocked(name):
        def fail(*_args, **_kwargs):
            attempts.append(name)
            raise RuntimeError(f"external network blocked by test: {name}")
        return fail

    monkeypatch.setattr(httpx.Client, "send", blocked("httpx.Client.send"))
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked("httpx.AsyncClient.send"))
    monkeypatch.setattr(socket, "create_connection", blocked("socket.create_connection"))

    # Prove the guard is active, then reset the counter before exercising Critic.
    with pytest.raises(RuntimeError, match="external network blocked by test"):
        socket.create_connection(("127.0.0.1", 1), timeout=0.01)
    with pytest.raises(RuntimeError, match="external network blocked by test"):
        httpx.Client().send(httpx.Request("POST", "http://127.0.0.1:1/blocked"))
    assert attempts == ["socket.create_connection", "httpx.Client.send"]
    attempts.clear()
    return attempts


def _run_and_capture(monkeypatch, state: dict, mode: str):
    import importlib

    from app.agents import critic_agent
    from app.core.config import get_settings
    from app.graph.nodes.critic import critic

    captured: list[dict[str, str]] = []

    def offline_extract(prompt, content, *_args, **_kwargs):
        # Captures the actual packed content and intentionally returns no verdict.
        captured.append({"prompt": str(prompt), "content": str(content)})
        return None

    monkeypatch.setattr(critic_agent, "llm_extract", offline_extract)
    # These tests inspect all three prompt projections. The production default
    # stops optional reviews after a failure, so turn that policy off locally.
    critic_module = importlib.import_module("app.graph.nodes.critic")
    settings = get_settings().model_copy(update={"critic_review_stop_on_failure_enabled": False})
    monkeypatch.setattr(critic_module, "get_settings", lambda: settings)
    result = critic(copy.deepcopy(state), llm_review_mode=mode)
    return result, captured


def _context_from_review(review: dict[str, str]) -> dict:
    # WP1 routes the server-owned investigation ledger through the Critic
    # Bundle as tool evidence; it no longer uses a standalone prompt source.
    marker = "refs=investigation_context\n"
    assert marker in review["content"]
    start = review["content"].index(marker) + len(marker)
    end = review["content"].find("\n\n[", start)
    if end < 0:
        end = len(review["content"])
    return json.loads(review["content"][start:end].strip())


def test_full_review_receives_role_separated_context_and_off_adds_no_calls(monkeypatch):
    attempts = _install_network_block(monkeypatch)
    state = _saved_state(OFF_TRACE)
    from app.graph.nodes.critic import _build_diagnostic_investigation_context

    context = _build_diagnostic_investigation_context(state)
    assert context is not None
    assert context["terminal_reason"]["code"] == "DATA_INSUFFICIENT"
    assert context["global_stop"]["may_be_ignored"] is False
    assert context["business_causality"]["causal_effect_estimated"] is False
    assert context["business_causality"]["decomposition_alone_establishes_business_causality"] is False
    assert context["validated_numeric_decompositions"]

    optional = [item for item in context["investigation_limitations"]
                if item["classification"] == "optional_deep_check_incomplete"]
    assert len(optional) == 1
    gap = optional[0]
    assert gap["action_type"] == "CHECK_ANOMALY"
    assert gap["target_metric"] == "order_count"
    assert gap["required"] is False
    assert gap["may_be_ignored"] is False
    assert gap["invalidates_linked_decomposition"] is False
    assert gap["global_stop_may_be_ignored"] is False
    assert gap["observed_valid_bucket_count"] == 7
    assert gap["public_validation_rule"]["minimum"] == 8
    assert gap["bucket_count_source_evidence_id"] in context["verified_evidence_ids"]
    assert gap["action_failure_is_numerical_evidence"] is False
    assert gap["limitation_relation_evidence_id"] == context["validated_numeric_decompositions"][0]["evidence_id"]
    failed_action = next(item for item in context["action_results"] if item["action_id"] == gap["action_id"])
    assert failed_action["verified_evidence_ids"] == []
    assert gap["action_id"] not in {item.get("source_action_id") for item in context["verified_numeric_evidence"]}

    full_result, full_reviews = _run_and_capture(monkeypatch, state, "full")
    # Each review dimension is attempted independently, even if an earlier
    # reviewer returned no verdict; the aggregate remains incomplete.
    assert len(full_reviews) == 3
    captured_context = _context_from_review(full_reviews[0])
    assert captured_context["validated_numeric_decompositions"]
    assert captured_context["investigation_limitations"] == context["investigation_limitations"]
    assert captured_context["business_causality"]["causal_effect_estimated"] is False
    llm_review = full_result["critic_result"]["llm_review"]
    assert llm_review["status"] == "incomplete"
    assert llm_review["groundedness"]["failure_kind"] == "missing_or_invalid_verdict"
    assert llm_review["completeness"]["failure_kind"] == "missing_or_invalid_verdict"
    assert llm_review["consistency"]["failure_kind"] == "missing_or_invalid_verdict"
    assert full_result["critic_verdict"] != "approved"

    off_result, off_reviews = _run_and_capture(monkeypatch, state, "critic_llm_review_off")
    assert off_reviews == []
    assert off_result["critic_result"]["llm_review"]["status"] == "evaluation_ablation"
    for review_name in (
        "data_review", "rag_review", "claim_review", "report_review",
        "diagnostic_review", "completeness_review", "causal_review",
    ):
        assert full_result["critic_result"][review_name] == off_result["critic_result"][review_name]
    assert attempts == []


def test_terminal_database_failure_skips_optional_critic_calls_without_passing(monkeypatch):
    attempts = _install_network_block(monkeypatch)
    from app.agents import critic_agent
    from app.graph.nodes.critic import critic

    state = _saved_state(FULL_TRACE)
    state["errors"] = [{
        "code": "database_error",
        "message": "数据库查询执行失败，未取得该项结果。",
        "failure_context": {
            "category": "database_error",
            "service": "postgresql",
            "code": "database_execution_failed",
            "cause_type": "OperationalError",
        },
    }]
    calls = []

    def forbidden_optional_review(*_args, **_kwargs):
        calls.append("unexpected")
        raise AssertionError("terminal execution failure must not trigger another model request")

    monkeypatch.setattr(critic_agent, "llm_extract", forbidden_optional_review)
    result = critic(state, llm_review_mode="full")

    review = result["critic_result"]["llm_review"]
    assert calls == []
    assert review["status"] == "not_run_due_terminal_execution_failure"
    assert review["groundedness"]["failure_kind"] == "upstream_terminal_execution_failure"
    assert review["completeness"]["failure_kind"] == "upstream_terminal_execution_failure"
    assert review["consistency"]["failure_kind"] == "upstream_terminal_execution_failure"
    assert result["critic_verdict"] != "approved"
    assert attempts == []


def test_skipped_critic_paths_do_not_build_context_bundle(monkeypatch):
    import importlib

    from app.context.orchestrator import ContextOrchestrator
    from app.graph.nodes.critic import critic

    calls: list[str] = []

    def forbidden_bundle(self, *_args, **_kwargs):
        calls.append("bundle")
        raise AssertionError("a skipped Critic review must not pack context")

    monkeypatch.setattr(ContextOrchestrator, "build_specialist_bundle", forbidden_bundle)
    critic_module = importlib.import_module("app.graph.nodes.critic")

    state = _saved_state(OFF_TRACE)
    off = critic(copy.deepcopy(state), llm_review_mode="critic_llm_review_off")
    assert off["critic_result"]["llm_review"]["status"] == "evaluation_ablation"

    monkeypatch.setattr(
        critic_module,
        "decide_optional_llm_review",
        lambda **_kwargs: {"action": "skip", "reason_code": "unit_test_skip"},
    )
    adaptive = critic(copy.deepcopy(state), llm_review_mode="adaptive")
    assert adaptive["critic_result"]["llm_review"]["status"] == "skipped_by_adaptive_policy"

    monkeypatch.setattr(
        critic_module,
        "_review_request_contract",
        lambda _state: {"status": "rejected", "issues": ["contract_invalid"]},
    )
    rejected = critic(copy.deepcopy(state), llm_review_mode="full")
    assert rejected["critic_result"]["request_contract_review"]["status"] == "rejected"
    assert calls == []


@pytest.mark.parametrize(
    ("task_type", "report_id", "report_version", "expected_verdict"),
    [
        ("anomaly_analysis", None, None, "rejected"),
        ("report_generation", "report-1", 1, "human_review"),
    ],
)
def test_eight_required_critic_summaries_overflow_as_incomplete_not_approved(
    monkeypatch, task_type, report_id, report_version, expected_verdict,
):
    import importlib

    from app.context import packer
    from app.core.config import get_settings
    from app.graph.nodes.critic import critic

    critic_module = importlib.import_module("app.graph.nodes.critic")
    state = _saved_state(OFF_TRACE)
    state["errors"] = []
    state["task_type"] = task_type
    if report_id:
        state["report_id"] = report_id
        state["report_version"] = report_version
    state["user_message"] = "PRIVATE_QUESTION_SHOULD_NOT_BE_LOGGED"
    evidence_ids = [f"ev-required-{index}" for index in range(8)]
    state["claims"] = [{
        "claim_type": "analysis",
        "value": "GMV 变化由已验证证据支持。",
        "source_evidence_ids": evidence_ids,
    }]
    state["evidence_refresh"] = {"available": True, "evidence_ids": evidence_ids}
    state["evidence_bundles"] = [
        {
            "evidence_id": evidence_id,
            "source_type": "mcp",
            "source_ref": f"metric:{index}",
            "trusted": True,
            "content": {
                "metric_id": "gmv",
                "query_type": "metric_value",
                "result": {
                    "status": "ok",
                    "parsed": {"query_type": "metric_value", "metric_name": "gmv"},
                    "metric_result": {
                        "metric_name": "gmv", "metric_date": f"2024-01-{index + 1:02d}",
                        "value": 100.0 + index, "unit": "BRL", "status": "ok",
                    },
                },
            },
        }
        for index, evidence_id in enumerate(evidence_ids)
    ]
    state["analysis_presentation"] = {
        **(state.get("analysis_presentation") or {}),
        "status": "completed",
        "evidence": [{"evidence_id": evidence_id} for evidence_id in evidence_ids],
    }
    monkeypatch.setattr(
        critic_module,
        "_build_diagnostic_investigation_context",
        lambda _state: {
            "target": {"metric_id": "gmv"},
            "verified_evidence_ids": evidence_ids,
            "investigation_limitations": [],
            "business_causality": {"causal_effect_estimated": False},
        },
    )
    monkeypatch.setattr(
        critic_module,
        "_review_request_contract",
        lambda _state: {"status": "passed", "issues": []},
    )
    if report_id:
        monkeypatch.setattr(
            critic_module,
            "review_single_data_query",
            lambda *_args, **_kwargs: {"status": "approved", "items": []},
        )
    monkeypatch.setattr(
        critic_module,
        "_review_claim_bindings",
        lambda _state: {"status": "approved", "issues": []},
    )
    monkeypatch.setattr(critic_module, "_review_diagnostic_trajectory", lambda _state: {"status": "approved", "issues": []})
    monkeypatch.setattr(critic_module, "_review_causal_language", lambda _state: {"status": "approved", "issues": []})
    small_budget_settings = get_settings().model_copy(update={"context_critic_input_budget": 64})
    monkeypatch.setattr(packer, "get_settings", lambda: small_budget_settings)

    result = critic(copy.deepcopy(state), llm_review_mode="full")
    review = result["critic_result"]["llm_review"]
    failure = next(
        (error for error in result.get("errors", []) if error.get("code") == "critic_review_incomplete"),
        None,
    )

    assert review["status"] == "incomplete"
    assert review["context"]["required_evidence_ids"] == evidence_ids
    assert review["context"]["estimated_tokens"] > review["context"]["available_tokens"]
    assert review["context"]["required_item_count"] >= 8
    assert review["context"]["request_sent"] is False
    assert review["context"]["included_evidence_ids"] == []
    assert review["context"]["candidate_packed_evidence_ids"] == evidence_ids
    assert review["context"]["retained_evidence_ids"] == []
    assert review["context"]["omitted_evidence_ids"] == evidence_ids
    assert all(
        review[name]["completion_status"] == "not_completed"
        and review[name]["failure_kind"] == "context_budget_exceeded"
        for name in ("groundedness", "completeness", "consistency")
    )
    assert result["critic_verdict"] == expected_verdict
    assert result["analysis_presentation"]["status"] == "partial"
    if expected_verdict == "human_review":
        assert failure is None
        assert review["context_limit_action"] == "human_review"
        from app.graph.workflow import _route_from_critic

        assert _route_from_critic(result) == "human_review"
    else:
        assert failure is not None
        assert failure["failure_kind"] == "context_budget_exceeded"
        assert "PRIVATE_QUESTION_SHOULD_NOT_BE_LOGGED" not in json.dumps(failure["context"])


def test_context_budget_human_review_requires_durable_report_entry() -> None:
    from app.graph.nodes.critic import _critic_context_limit_supports_human_review

    assert _critic_context_limit_supports_human_review("anomaly_analysis", {"report_id": "r", "report_version": 1}) is False
    assert _critic_context_limit_supports_human_review("report_generation", {"report_id": "r"}) is False
    assert _critic_context_limit_supports_human_review("report_generation", {"report_id": "r", "report_version": 2}) is True


@pytest.mark.parametrize("variant", ["required_gap", "evidence_conflict"])
def test_required_gap_and_conflict_are_not_described_as_ignorable(monkeypatch, variant):
    attempts = _install_network_block(monkeypatch)
    state = _saved_state(OFF_TRACE)
    results = state["investigation_state"]["action_results"]
    if variant == "required_gap":
        state["diagnostic_step_state"]["target_trend"] = "data_insufficient"
        required_action = next(item for item in results if item["action_id"] == "action_1")
        required_action.update({
            "status": "failed",
            "failure_code": "DATA_INSUFFICIENT",
            "failure_detail": "Required target trend is incomplete in this saved-state validation variant.",
            "evidence_ids": [],
            "validation_result": {
                "status": "incomplete_result",
                "checks": [{"name": "required_trend", "status": "failed", "detail": "required data missing"}],
                "failed_fields_or_groups": [],
            },
        })
    else:
        failed = next(item for item in results if item.get("status") == "failed")
        known = state["investigation_state"]["evidence"][0]["evidence_id"]
        failed.update({
            "failure_code": "INVALID_DATA",
            "failure_detail": "The saved-state validation variant contains a conflicting aggregate.",
            "validation_result": {
                "status": "invalid_data",
                "checks": [{
                    "name": "cross_evidence_reconciliation", "status": "failed",
                    "detail": "aggregate mismatch", "conflicting_evidence_ids": [known],
                }],
                "failed_fields_or_groups": [],
            },
        })

    full_result, reviews = _run_and_capture(monkeypatch, state, "full")
    assert len(reviews) == 3
    context = _context_from_review(reviews[0])
    assert context["investigation_limitations"]
    assert all(item["may_be_ignored"] is False for item in context["investigation_limitations"])
    if variant == "required_gap":
        issue = next(item for item in context["investigation_limitations"]
                     if item["classification"] == "required_or_unresolved_data_gap")
        assert issue["required"] is True
        assert issue["severity"] == "blocking_or_unresolved"
    else:
        issue = next(item for item in context["investigation_limitations"]
                     if item["classification"] == "evidence_conflict_or_invalid_data")
        assert issue["severity"] == "blocking"
        assert context["conflicting_evidence_ids"]
    # Missing verdicts remain incomplete across independently attempted checks.
    llm_review = full_result["critic_result"]["llm_review"]
    assert llm_review["status"] == "incomplete"
    assert llm_review["groundedness"]["failure_kind"] == "missing_or_invalid_verdict"
    assert llm_review["completeness"]["failure_kind"] == "missing_or_invalid_verdict"
    assert full_result["critic_verdict"] != "approved"
    assert attempts == []


def _replay_current_budget_stop_to_answer():
    """Replay current stop handling and terminal projection from the saved off state."""
    from app.graph.nodes.answer import generate_answer
    from app.graph.nodes.diagnostic import _record_candidate_data_gap, diagnostic_finalize
    from app.investigation.contracts import InvestigationState
    from app.tenancy import TenantContext, activate_tenant_context

    state = _saved_state(OFF_TRACE)
    state["errors"] = []
    investigation = InvestigationState.model_validate(state["investigation_state"])
    failed = investigation.action_results[-1]
    candidate = next(item for item in investigation.candidate_actions
                     if item.candidate_id == failed.candidate_id)
    state["diagnostic_candidate_data_gaps"] = []
    _record_candidate_data_gap(
        state, investigation, failed,
        fallback_candidate=candidate,
        failure_detail=failed.failure_detail,
    )
    state["investigation_state"] = investigation.model_dump(mode="json")
    state["errors"] = [{"code": "token_budget_exceeded", "message": "offline saved-run replay"}]

    context = TenantContext(
        tenant_id="test-tenant", user_id="test-user", role="admin", scope_type="platform",
    )
    with activate_tenant_context(context):
        finalized = diagnostic_finalize(state)
        answered = generate_answer(finalized)
    return answered


def test_budget_stop_bridge_reaches_actual_critic_input_after_terminal_and_answer(monkeypatch):
    attempts = _install_network_block(monkeypatch)
    monkeypatch.setattr("app.core.execution_context.enqueue_sse_event", lambda _event: None)
    state = _replay_current_budget_stop_to_answer()
    before_tool_calls = copy.deepcopy(state.get("tool_calls", []))
    before_actions = copy.deepcopy(state["investigation_state"]["action_results"])

    result, reviews = _run_and_capture(monkeypatch, state, "full")

    # Budget exhaustion is an authoritative terminal stop. Critic retains the
    # deterministic stop context but deliberately does not spend another
    # optional LLM-review call after the budget has been exhausted.
    assert reviews == []
    assert all(
        action.get("verified_evidence_ids", []) == []
        for action in state["investigation_state"]["action_results"]
        if action.get("failure_code") == "DATA_INSUFFICIENT"
    )
    assert state["analysis_presentation"]["status"] == "partial"
    assert "10.00" in state["answer"] and "20.00" in state["answer"]
    assert "业务原因仍未确认" in state["answer"]
    assert state.get("tool_calls", []) == before_tool_calls
    assert state["investigation_state"]["action_results"] == before_actions
    llm_review = result["critic_result"]["llm_review"]
    assert llm_review["status"] == "not_run_due_terminal_execution_failure"
    assert llm_review["groundedness"]["failure_kind"] == "upstream_terminal_execution_failure"
    assert result["critic_verdict"] != "approved"
    assert attempts == []


def test_text_only_critic_revision_rebuilds_answer_without_replanning(monkeypatch):
    attempts = _install_network_block(monkeypatch)
    from app.agents import critic_agent
    from app.graph.nodes.answer import generate_answer
    from app.graph.nodes.critic import critic
    from app.graph.workflow import _route_from_critic
    from app.core.config import get_settings

    state = _saved_state(PAIR_OFF_TRACE)
    state["critic_llm_review_mode"] = "full"
    state.setdefault("_revision_counts", {})["generate_answer"] = get_settings().max_revisions
    initial_plan = copy.deepcopy(state["diagnostic_plan"])
    initial_steps = copy.deepcopy(state["diagnostic_step_state"])
    initial_actions = copy.deepcopy(state["investigation_state"]["action_results"])
    initial_decomposition = copy.deepcopy(state["analysis_result"]["metric_decomposition"])
    initial_limitations = copy.deepcopy(state["analysis_result"]["limitations"])
    initial_answer = state["answer"]
    calls: list[dict[str, str]] = []

    def offline_review(prompt, content, *_args, **_kwargs):
        calls.append({"prompt": str(prompt), "content": str(content)})
        if len(calls) in {2, 5}:
            return {"verdict": "failed", "issues": ["Please state the verified trend change explicitly."]}
        return {"verdict": "passed", "issues": []}

    monkeypatch.setattr(critic_agent, "llm_extract", offline_review)
    from app.graph.critic_review_mode import CriticLLMReviewMode

    reviewed = critic(copy.deepcopy(state), llm_review_mode=CriticLLMReviewMode.FULL)
    assert "10.00" in calls[1]["content"] and "20.00" in calls[1]["content"]
    assert "+2.67" in calls[1]["content"] and "+7.33" in calls[1]["content"]
    assert reviewed["critic_verdict"] == "needs_revision"
    assert reviewed["critic_revision_target"] == "generate_answer"
    assert _route_from_critic(reviewed) == "generate_answer"

    rewritten = generate_answer(reviewed)
    assert rewritten["diagnostic_plan"] == initial_plan
    assert rewritten["diagnostic_step_state"] == initial_steps
    assert rewritten["investigation_state"]["action_results"] == initial_actions
    assert rewritten["analysis_result"]["metric_decomposition"] == initial_decomposition
    assert rewritten["analysis_result"]["stop_reason"] == "budget_exhausted"
    assert rewritten["analysis_result"]["limitations"] == initial_limitations
    assert rewritten["analysis_presentation"]["status"] == "partial"
    assert rewritten["answer"] != initial_answer
    assert "按已验证的基线期（2024-05-03 至 2024-05-03）与当前期（2024-05-04 至 2024-05-04）比较" in rewritten["answer"]
    assert "从 10.00 雷亚尔/单升至 20.00 雷亚尔/单，增加 10.00 雷亚尔/单" in rewritten["answer"]
    assert "不证明期间内持续变化，也不构成异常判断" in rewritten["answer"]
    assert "持续上涨" not in rewritten["answer"]
    assert "10.00" in rewritten["answer"] and "20.00" in rewritten["answer"]
    assert "+2.67" in rewritten["answer"] and "+7.33" in rewritten["answer"]
    assert "业务原因仍未确认" in rewritten["answer"]
    assert rewritten["answer_revision"]["status"] == "revised"
    assert rewritten["answer_revision"]["feedback_categories"] == ["completeness"]
    assert rewritten["answer_revision"]["applied_changes"] == ["verified_period_comparison"]
    assert rewritten["answer_revision"]["evidence_ids"] == initial_decomposition["evidence_ids"]
    assert rewritten["answer_revision"]["feedback_used_as_evidence"] is False
    revision_claim = next(
        claim for claim in rewritten["claims"]
        if "按已验证的基线期" in claim.get("value", "")
    )
    assert revision_claim["source_evidence_ids"] == initial_decomposition["evidence_ids"]
    assert rewritten["analysis_presentation"]["metric_decomposition"] == initial_decomposition
    presented_evidence_ids = {
        item.get("evidence_id") for item in rewritten["analysis_presentation"]["evidence"]
    }
    assert set(initial_decomposition["evidence_ids"]) <= presented_evidence_ids
    assert rewritten["tool_calls"] == state["tool_calls"]
    assert attempts == []
    first_verdict = rewritten["critic_verdict"]
    assert first_verdict == "needs_revision"

    # Re-enter the actual Critic node; a synthetic failed verdict exercises the
    # revision cap and is not treated as a real answer-quality judgment.
    reviewed_again = critic(rewritten, llm_review_mode=CriticLLMReviewMode.FULL)
    assert "按已验证的基线期" in calls[4]["content"]
    assert reviewed_again["critic_verdict"] == "rejected"
    assert reviewed_again["critic_revision_target"] is None
    revision_limit_error = next(
        error for error in reviewed_again.get("errors", [])
        if error.get("code") == "critic_revision_limit_exceeded"
    )
    assert revision_limit_error["revision_target"] == "generate_answer"
    assert revision_limit_error["revision_count"] == get_settings().max_revisions + 1
    assert "答案生成" in revision_limit_error["message"]
    assert "不表示数据查询重复执行" in revision_limit_error["message"]
    assert _route_from_critic(reviewed_again) != "generate_answer"
    assert reviewed_again["diagnostic_plan"] == initial_plan
    assert reviewed_again["investigation_state"]["action_results"] == initial_actions
    assert reviewed_again["tool_calls"] == state["tool_calls"]

    assert first_verdict == "needs_revision"
    assert len(calls) == 6
    assert attempts == []


def test_answer_revision_does_not_invent_unsupported_trend_or_cause(monkeypatch):
    attempts = _install_network_block(monkeypatch)
    from app.graph.nodes.answer import generate_answer

    state = _saved_state(PAIR_OFF_TRACE)
    state["analysis_result"]["metric_decomposition"] = None
    state["analysis_result"]["summary"] = (
        "A later optional check is incomplete; the business cause remains unconfirmed."
    )
    state["critic_verdict"] = "needs_revision"
    state["critic_revision_target"] = "generate_answer"
    state["critic_result"] = {
        "llm_review": {
            "groundedness": {"verdict": "passed", "issues": []},
            "completeness": {
                "verdict": "failed",
                "issues": ["Describe the trend and identify the business root cause."],
            },
            "consistency": {"verdict": "passed", "issues": []},
        },
    }

    revised = generate_answer(state)
    assert "10.00 雷亚尔/单" not in revised["answer"]
    assert "20.00 雷亚尔/单" not in revised["answer"]
    assert "证据不足以支持补充趋势变化描述" in revised["answer"]
    assert "不能确认业务原因或因果关系" in revised["answer"]
    assert revised["answer_revision"]["status"] == "bounded_or_limited"
    assert revised["answer_revision"]["evidence_ids"] == []
    assert revised["answer_revision"]["feedback_used_as_evidence"] is False
    assert attempts == []


def test_replan_reuses_only_semantically_compatible_valid_evidence(monkeypatch):
    attempts = _install_network_block(monkeypatch)
    from app.graph.nodes.diagnostic import diagnostic_plan
    from app.llm import client as llm_client
    from app.tenancy import TenantContext, activate_tenant_context

    source = _saved_state(PAIR_OFF_TRACE)
    original_plan = copy.deepcopy(source["diagnostic_plan"])
    original_steps = copy.deepcopy(source["diagnostic_step_state"])
    context = TenantContext(
        tenant_id=source["tenant_id"], user_id=source["user_id"],
        role="admin", scope_type="platform",
    )

    def run_plan(state, payload):
        monkeypatch.setattr(llm_client, "llm_agent_action", lambda *_args, **_kwargs: copy.deepcopy(payload))
        with activate_tenant_context(context):
            return diagnostic_plan(copy.deepcopy(state))

    renamed = copy.deepcopy(original_plan)
    for step in renamed["steps"]:
        step["step_id"] = f"replanned_{step['step_id']}"
    same_scope = run_plan(source, renamed)
    for old_step in original_plan["steps"]:
        new_id = f"replanned_{old_step['step_id']}"
        expected = original_steps[old_step["step_id"]]
        assert same_scope["diagnostic_step_state"][new_id] == expected

    changed_requirement = copy.deepcopy(renamed)
    trend_step = next(step for step in changed_requirement["steps"] if step["kind"] == "trend")
    trend_step["completion_condition"] += " and verify a second comparison window"
    changed = run_plan(source, changed_requirement)
    assert changed["diagnostic_step_state"][trend_step["step_id"]] == "pending"

    changed_filter_goal = copy.deepcopy(renamed)
    changed_filter_goal["goal"] += " with the requested product_category filter"
    filtered = run_plan(source, changed_filter_goal)
    assert all(value == "pending" for value in filtered["diagnostic_step_state"].values())

    invalid_evidence = copy.deepcopy(source)
    target_trend_action = next(
        item for item in invalid_evidence["investigation_state"]["action_results"]
        if item.get("action_type") == "ANALYZE_TREND"
        and item.get("metric_id") == invalid_evidence["diagnostic_target_metric"]
    )
    target_trend_action["validation_result"] = {
        "status": "incomplete_result", "checks": [{
            "name": "evidence_contract", "status": "failed", "detail": "invalid saved evidence variant",
        }], "failed_fields_or_groups": [],
    }
    invalid_replan = run_plan(invalid_evidence, renamed)
    assert invalid_replan["diagnostic_step_state"]["replanned_trend"] == "pending"
    assert invalid_replan["diagnostic_step_state"]["replanned_related_metric"] == "completed"

    added_requirement = copy.deepcopy(renamed)
    added_requirement["steps"].append({
        "step_id": "new_required_check", "objective": "Check a new required evidence category",
        "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"],
        "completion_condition": "Obtain the additional required evidence",
        "kind": "dimension_contribution", "required": True,
    })
    added = run_plan(source, added_requirement)
    assert added["diagnostic_step_state"]["new_required_check"] == "pending"

    changed_target = copy.deepcopy(source)
    changed_target["user_message"] = changed_target["user_message"].replace("2024-05-04", "2024-05-05")
    changed_period = run_plan(changed_target, renamed)
    assert changed_period.get("diagnostic_investigation_invalidated") == "target_changed"
    assert all(value == "pending" for value in changed_period["diagnostic_step_state"].values())

    changed_metric = copy.deepcopy(source)
    changed_metric["diagnostic_target_metric"] = "gmv"
    target_changed = run_plan(changed_metric, renamed)
    assert target_changed.get("diagnostic_investigation_invalidated") == "target_changed"
    assert all(value == "pending" for value in target_changed["diagnostic_step_state"].values())

    changed_scope = copy.deepcopy(source)
    changed_scope["user_id"] = f"{source['user_id']}-different-scope"
    scope_changed = run_plan(changed_scope, renamed)
    assert scope_changed.get("diagnostic_investigation_invalidated") == "target_changed"
    assert all(value == "pending" for value in scope_changed["diagnostic_step_state"].values())

    conflict = copy.deepcopy(source)
    conflict_result = next(item for item in conflict["investigation_state"]["action_results"]
                           if item["status"] == "completed")
    conflict_id = conflict_result["evidence_ids"][0]
    failed = next(item for item in conflict["investigation_state"]["action_results"]
                  if item["status"] == "failed")
    failed.update({
        "failure_code": "INVALID_DATA",
        "validation_result": {"status": "invalid_data", "checks": [{
            "name": "cross_evidence_reconciliation", "status": "failed",
            "conflicting_evidence_ids": [conflict_id],
        }], "failed_fields_or_groups": []},
    })
    conflicted = run_plan(conflict, renamed)
    assert all(value == "pending" for value in conflicted["diagnostic_step_state"].values())
    assert attempts == []


@pytest.mark.parametrize(
    "variant",
    ["forged_link", "invalid_decomposition_scope", "required_step_incomplete", "conflict"],
)
def test_budget_stop_does_not_trust_optional_marker_without_valid_linkage(monkeypatch, variant):
    from app.graph.nodes.critic import _build_diagnostic_investigation_context

    state = _replay_current_budget_stop_to_answer()
    gap = state["diagnostic_candidate_data_gaps"][0]
    if variant == "forged_link":
        gap["related_decomposition_evidence_id"] = "forged-evidence-id"
    elif variant == "invalid_decomposition_scope":
        linked_id = gap["related_decomposition_evidence_id"]
        decomposition = next(item for item in state["investigation_state"]["evidence"]
                             if item["evidence_id"] == linked_id)
        decomposition["scope_fingerprint"] = "different-authorized-scope"
    elif variant == "required_step_incomplete":
        state["diagnostic_step_state"]["target_trend"] = "data_insufficient"
    else:
        action = state["investigation_state"]["action_results"][-1]
        action["failure_code"] = "INVALID_DATA"
        action["validation_result"] = {
            "status": "invalid_data",
            "checks": [{
                "name": "cross_evidence_reconciliation", "status": "failed",
                "detail": "conflicting inputs",
                "conflicting_evidence_ids": ["ev_3d1455e29f398e78"],
            }],
            "failed_fields_or_groups": [],
        }

    context = _build_diagnostic_investigation_context(state)
    assert context is not None
    assert context["global_stop"]["code"] == "BUDGET_EXHAUSTED"
    assert all(item["classification"] != "optional_deep_check_incomplete"
               for item in context["investigation_limitations"])
    unresolved = next(item for item in context["investigation_limitations"]
                      if item["action_id"] == gap["action_id"])
    expected_classification = (
        "evidence_conflict_or_invalid_data" if variant == "conflict"
        else "required_or_unresolved_data_gap"
    )
    assert unresolved["classification"] == expected_classification
    assert unresolved["may_be_ignored"] is False
    assert unresolved["limitation_relation_evidence_id"] is None
    assert unresolved["observed_valid_bucket_count"] is None


def test_answer_omission_still_has_verified_numbers_in_review_context(monkeypatch):
    attempts = _install_network_block(monkeypatch)
    state = _saved_state(OFF_TRACE)
    state["answer"] = "Statistical contribution was observed; business cause remains unconfirmed."
    _result, reviews = _run_and_capture(monkeypatch, state, "full")
    assert len(reviews) == 3
    context = _context_from_review(reviews[0])
    decomposition = context["validated_numeric_decompositions"]
    assert decomposition
    assert all(item["evidence_id"] in context["verified_evidence_ids"]
               for item in decomposition)
    values = json.dumps(decomposition, ensure_ascii=False)
    assert any(character.isdigit() for character in values)
    assert "omitted" not in values.lower()
    assert "Independently check" in reviews[0]["prompt"]
    assert _result["critic_result"]["llm_review"]["status"] == "incomplete"
    assert _result["critic_verdict"] != "approved"
    assert attempts == []


def test_saved_full_trace_is_supported_for_review_context_construction():
    """The other preserved real arm remains parseable without repairing history."""
    from app.graph.nodes.critic import _build_diagnostic_investigation_context

    state = _saved_state(FULL_TRACE)
    context = _build_diagnostic_investigation_context(state)
    assert context is not None
    assert context["target"]["metric_id"] == "avg_order_value"
    assert context["terminal_reason"]["code"] == "DATA_INSUFFICIENT"
    assert context["validated_numeric_decompositions"]
