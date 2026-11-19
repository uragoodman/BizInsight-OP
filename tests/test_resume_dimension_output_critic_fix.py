"""Focused output/critic checks against the saved governed order-dimension run."""
from __future__ import annotations

import copy
import json
import os
import socket
from pathlib import Path
from pathlib import Path

import pytest

from test_platform_metric_trend_acceptance import isolated_trend_db  # noqa: F401


SAVED_RUN = Path(
    "docs/testing/resume_dev_pair_order_dimension_20261015_b937cbf9/traces/"
    "40d40a54-107c-4a56-a687-76265ad56792.run_record.json"
)


def _saved_state():
    record = json.loads(SAVED_RUN.read_text(encoding="utf-8-sig"))
    return copy.deepcopy(record["trace"]["state_snapshot"])


def _block_external_network(monkeypatch):
    def blocked(self, address):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError(f"unexpected external network attempt: {host}")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", lambda address, *a, **k: blocked(None, address))


def test_saved_dimension_contribution_answer_and_presentation_schema_round_trip(monkeypatch):
    _block_external_network(monkeypatch)
    from app.graph.nodes import answer
    from app.llm import client as llm
    from app.schemas.analysis_presentation import AnalysisPresentation

    monkeypatch.setattr(llm, "llm_chat", lambda *args, **kwargs: None)
    state = _saved_state()
    output = answer.generate_answer(state)

    assert "2024-05-03" in output["final_answer"]
    assert "2024-05-04" in output["final_answer"]
    assert "10.00 单" in output["final_answer"]
    assert "6.00 单" in output["final_answer"]
    assert "-4.00 单" in output["final_answer"]
    assert "业务原因仍未确认" in output["final_answer"]

    presentation = AnalysisPresentation.model_validate(output["analysis_presentation"]).model_dump()
    assert presentation["status"] == "partial"
    assert len(presentation["contributions"]) == 1
    card = presentation["contributions"][0]
    assert (card["baseline_value"], card["current_value"], card["unit"]) == (10.0, 6.0, "orders")
    assert (card["baseline_date"], card["current_date"]) == ("2024-05-03", "2024-05-04")
    assert card["total_delta"] == -4.0
    assert [(x["dimension_value"], x["baseline_value"], x["current_value"], x["delta_value"])
            for x in card["items"]] == [("A", 8.0, 4.0, -4.0), ("B", 2.0, 2.0, 0.0)]
    assert card["evidence_ids"] == ["ev_4ef8632d172c431c"]
    claim = next(c for c in output["claims"] if c["claim_type"] == "analysis")
    assert claim["source_evidence_ids"] == card["evidence_ids"]


def test_saved_critic_feedback_revises_answer_without_changing_investigation(monkeypatch):
    _block_external_network(monkeypatch)
    from app.graph.nodes import answer
    from app.llm import client as llm

    monkeypatch.setattr(llm, "llm_chat", lambda *args, **kwargs: None)
    state = _saved_state()
    state["critic_verdict"] = "needs_revision"
    state["critic_revision_target"] = "generate_answer"
    state["critic_result"] = {
        "llm_review": {
            "groundedness": {"verdict": "passed", "issues": []},
                "completeness": {"verdict": "failed", "issues": ["Please state the trend comparison values and change."]},
            "consistency": {"verdict": "passed", "issues": []},
        }
    }
    original_plan = copy.deepcopy(state["diagnostic_plan"])
    original_actions = copy.deepcopy(state["investigation_state"]["action_results"])
    original_tool_calls = copy.deepcopy(state["tool_calls"])
    original_ids = sorted(x["evidence_id"] for x in state["investigation_state"]["evidence"])

    output = answer.generate_answer(state)

    assert output["answer_revision"]["status"] == "revised"
    assert "已验证的维度比较" in output["final_answer"]
    assert "10.00 单" in output["final_answer"] and "6.00 单" in output["final_answer"]
    assert "-4.00 单" in output["final_answer"]
    assert "不证明期间内持续趋势或异常" in output["final_answer"]
    assert "业务原因仍未确认" in output["final_answer"]
    assert output["answer_revision"]["evidence_ids"] == ["ev_4ef8632d172c431c"]
    assert output["diagnostic_plan"] == original_plan
    assert output["investigation_state"]["action_results"] == original_actions
    assert output["tool_calls"] == original_tool_calls
    assert sorted(x["evidence_id"] for x in output["investigation_state"]["evidence"]) == original_ids


def test_non_additive_registered_ratio_card_keeps_its_method_and_output(monkeypatch):
    _block_external_network(monkeypatch)
    from app.graph.nodes import answer
    from app.llm import client as llm
    from app.schemas.analysis_presentation import AnalysisPresentation

    monkeypatch.setattr(llm, "llm_chat", lambda *args, **kwargs: None)
    state = _saved_state()
    state["investigation_state"] = {}
    state["analysis_result"] = {
        "summary": "已注册的运费率比率分解：10%变为12%，增加2个百分点。",
        "task_type": "anomaly_analysis",
        "contribution": [{
            "dimension": "month_position", "metric_name": "freight_rate",
            "method": "symmetric_mix_within_v1", "unit": "%",
            "baseline_value": 10.0, "current_value": 12.0, "total_delta": 2.0,
            "baseline_date": "2024-04", "current_date": "2024-05",
            "items": [{"dimension_value": "day_1_10", "baseline_value": 4.0,
                       "current_value": 5.0, "delta_value": 1.0}],
            "evidence_ids": ["ev_4ef8632d172c431c"],
        }],
        "limitations": ["业务因果未确认。"],
    }
    output = answer.generate_answer(state)
    card = AnalysisPresentation.model_validate(output["analysis_presentation"]).model_dump()["contributions"][0]
    assert card["method"] == "symmetric_mix_within_v1"
    assert (card["baseline_value"], card["current_value"], card["unit"]) == (10.0, 12.0, "%")
    assert "10%变为12%" in output["final_answer"]


@pytest.mark.parametrize("mutation", ["scope", "period", "unit", "action", "conflict"])
def test_invalid_dimension_evidence_is_not_projected(monkeypatch, mutation):
    _block_external_network(monkeypatch)
    from app.graph.nodes import answer
    from app.llm import client as llm
    from app.graph.nodes.diagnostic import validated_dimension_contribution_evidence

    monkeypatch.setattr(llm, "llm_chat", lambda *args, **kwargs: None)
    state = _saved_state()
    investigation = state["investigation_state"]
    evidence = next(x for x in investigation["evidence"] if x["evidence_id"] == "ev_4ef8632d172c431c")
    if mutation == "scope":
        evidence["scope_fingerprint"] = "wrong-scope"
    elif mutation == "period":
        evidence["comparison"]["comparison"]["current_date"] = "2024-05-05"
    elif mutation == "unit":
        evidence["unit"] = "kg"
    elif mutation == "action":
        action = next(x for x in investigation["action_results"] if x.get("candidate_id") == evidence["source_action_id"])
        action["status"] = "failed"
    else:
        action = next(x for x in investigation["action_results"] if x.get("candidate_id") == evidence["source_action_id"])
        action["validation_result"]["checks"].append({
            "name": "evidence_conflict", "status": "failed",
            "conflicting_evidence_ids": [evidence["evidence_id"]],
        })

    assert validated_dimension_contribution_evidence(state) == []
    output = answer.generate_answer(state)
    assert output["analysis_presentation"]["contributions"] == []
    assert "统计贡献：" not in output["final_answer"]
    assert "100.0%" not in output["final_answer"]
    assert "ev_4ef8632d172c431c" not in output["final_answer"]
    assert "业务因果尚未确认" in output["final_answer"]
    assert output["analysis_presentation"]["limitations"]


def test_critic_investigation_context_keeps_local_gap_and_global_budget_separate(monkeypatch):
    _block_external_network(monkeypatch)
    from app.graph.nodes.critic import _build_diagnostic_investigation_context

    context = _build_diagnostic_investigation_context(_saved_state())
    assert context is not None
    assert context["terminal_reason"]["code"] == "BUDGET_EXHAUSTED"
    evidence = context["verified_numeric_evidence"]
    contribution = next(item for item in evidence if item.get("evidence_type") == "dimension_contribution")
    assert contribution["values"]["baseline_value"] == 10.0
    assert contribution["values"]["current_value"] == 6.0
    assert contribution["values"]["total_delta"] == -4.0
    gap = next(item for item in context["investigation_limitations"] if item.get("action_type") == "CHECK_ANOMALY")
    assert gap["classification"] == "optional_deep_check_incomplete"
    assert gap["observed_valid_bucket_count"] == 2
    assert gap["public_validation_rule"]["minimum"] == 8
    assert gap["invalidates_linked_decomposition"] is False


def test_full_and_off_critic_review_completion_states_are_explicit(monkeypatch):
    _block_external_network(monkeypatch)
    import app.graph.nodes.critic as critic_node

    state = _saved_state()
    captured_contexts = []
    calls = []

    def passed(*args, **kwargs):
        calls.append((args, kwargs))
        captured_contexts.append(args[-1])
        return {"verdict": "passed", "issues": []}

    monkeypatch.setattr(critic_node, "llm_review_groundedness", passed)
    monkeypatch.setattr(critic_node, "llm_review_evidence_completeness", passed)
    monkeypatch.setattr(critic_node, "llm_review_logic_consistency", passed)
    full = critic_node.critic(copy.deepcopy(state), llm_review_mode="full")
    assert len(calls) == 3
    assert full["critic_result"]["llm_review"]["status"] == "passed"
    assert [full["critic_result"]["llm_review"][name]["completion_status"]
            for name in ("groundedness", "completeness", "consistency")] == ["completed"] * 3
    assert all(ctx and ctx["business_causality"]["status"] == "unconfirmed" for ctx in captured_contexts)
    assert any(item.get("evidence_type") == "dimension_contribution" for item in captured_contexts[0]["verified_numeric_evidence"])
    assert any(item.get("action_type") == "CHECK_ANOMALY" for item in captured_contexts[0]["investigation_limitations"])
    assert captured_contexts[0]["global_stop"]["code"] == "BUDGET_EXHAUSTED"

    calls.clear()
    off = critic_node.critic(copy.deepcopy(state), llm_review_mode="critic_llm_review_off")
    assert calls == []
    assert off["critic_result"]["llm_review"]["status"] == "evaluation_ablation"
    assert all(off["critic_result"]["llm_review"][name]["verdict"] == "evaluation_ablation"
               for name in ("groundedness", "completeness", "consistency"))


def test_full_critic_invalid_or_missing_review_is_not_approved(monkeypatch):
    _block_external_network(monkeypatch)
    import app.graph.nodes.critic as critic_node

    state = _saved_state()
    monkeypatch.setattr(critic_node, "llm_review_groundedness", lambda *a, **k: {"verdict": "passed", "issues": []})
    monkeypatch.setattr(critic_node, "llm_review_evidence_completeness", lambda *a, **k: None)
    monkeypatch.setattr(critic_node, "llm_review_logic_consistency", lambda *a, **k: {"verdict": "passed", "issues": []})
    result = critic_node.critic(state, llm_review_mode="full")
    review = result["critic_result"]["llm_review"]
    assert review["status"] == "incomplete"
    assert review["completeness"]["failure_kind"] == "missing_or_invalid_verdict"
    assert review["consistency"]["failure_kind"] == "not_attempted_after_prior_review_incomplete"
    assert result["critic_verdict"] == "rejected"


@pytest.mark.parametrize(
    "scenario,expected_status,expected_failure",
    [
        ("failed_empty", "failed", None),
        ("invalid_verdict", "incomplete", "missing_or_invalid_verdict"),
        ("not_sent", "incomplete", "not_sent"),
        ("timeout", "incomplete", "timeout"),
    ],
)
def test_critic_review_completion_classifies_failure_without_passing(monkeypatch, scenario, expected_status, expected_failure):
    _block_external_network(monkeypatch)
    import app.graph.nodes.critic as critic_node

    state = _saved_state()

    def first_review(*_args, **_kwargs):
        if scenario == "failed_empty":
            return {"verdict": "failed", "issues": []}
        if scenario == "invalid_verdict":
            return {"verdict": "unknown", "issues": []}
        if scenario in {"not_sent", "timeout"}:
            state["llm_calls"].append({
                "call_id": f"{scenario}-call", "outcome": "not_sent" if scenario == "not_sent" else "provider_error",
                "not_sent_reason": "token_budget_exhausted" if scenario == "not_sent" else None,
                "attempts": [] if scenario == "not_sent" else [{"attempt_id": "timeout-attempt", "exception_type": "APITimeoutError"}],
            })
            return None
        raise AssertionError(scenario)

    monkeypatch.setattr(critic_node, "llm_review_groundedness", first_review)
    monkeypatch.setattr(critic_node, "llm_review_evidence_completeness", lambda *a, **k: {"verdict": "passed", "issues": []})
    monkeypatch.setattr(critic_node, "llm_review_logic_consistency", lambda *a, **k: {"verdict": "passed", "issues": []})
    result = critic_node.critic(state, llm_review_mode="full")
    review = result["critic_result"]["llm_review"]
    assert review["status"] == expected_status
    if expected_failure:
        assert review["groundedness"]["failure_kind"] == expected_failure
        assert result["critic_verdict"] != "approved"
    else:
        assert review["groundedness"]["completion_status"] == "completed"
        assert result["critic_verdict"] != "approved"


def test_non_applicable_metric_query_does_not_claim_optional_review(monkeypatch):
    _block_external_network(monkeypatch)
    import app.graph.nodes.critic as critic_node

    state = _saved_state()
    state["task_type"] = "metric_query"
    state["registered_decomposition_request"] = None
    state["metric_planning_decision"] = {"operation_intent": "ordinary_metric_query"}
    result = critic_node.critic(state, llm_review_mode="full")
    review = result["critic_result"]["llm_review"]
    assert review["status"] == "not_applicable"
    # The existing contract marks inapplicable reviews as skipped; it does
    # not claim they were executed or completed.
    assert all(review[name]["verdict"] == "skipped"
               for name in ("groundedness", "completeness", "consistency"))


@pytest.mark.parametrize("mutation,expected", [("correct", True), ("wrong_evidence", False), ("omit_values", False)])
def test_resume_pair_scorer_requires_claim_source_binding_and_visible_values(mutation, expected):
    from scripts.run_resume_dev_pair import _score

    state = {
        "trend_result": {
            "metric_name": "order_count", "start_date": "2024-05-03", "end_date": "2024-05-04",
            "points": [{"bucket": "2024-05-03", "value": 10}, {"bucket": "2024-05-04", "value": 6}],
        },
        "contribution_result": {
            "metric_name": "order_count", "dimension": "product_category",
            "baseline_date": "2024-05-03", "current_date": "2024-05-04", "total_delta": -4,
            "baseline_items": [{"dimension_value": "A", "value": 8}, {"dimension_value": "B", "value": 2}],
            "current_items": [{"dimension_value": "A", "value": 4}, {"dimension_value": "B", "value": 2}],
        },
        "evidence_bundles": [{"evidence_id": "ev_valid"}],
        "claims": [{"source_evidence_ids": ["ev_valid"]}],
    }
    response = {"answer": "2024-05-03基线10，2024-05-04当前6，订单量下降4；统计贡献来自A。"}
    if mutation == "wrong_evidence":
        state["claims"][0]["source_evidence_ids"] = ["ev_wrong"]
    elif mutation == "omit_values":
        response["answer"] = "2024-05-03与2024-05-04订单量下降；统计贡献来自A。"

    score = _score(state, response, "governed_workflow")
    assert score["content_valid"] is expected
    # This content-only fixture supplies neither a completed terminal nor
    # completed full reviews; valid numbers alone cannot prove task success.
    assert score["task_pass"] is False
    assert score["checks"]["answer_claim_evidence_binding"] is (mutation != "wrong_evidence")
    assert score["checks"]["answer_baseline_current_values_visible"] is (mutation != "omit_values")


def test_normal_workflow_full_off_dimension_contribution_uses_isolated_sql_and_critic_sdk_boundary(monkeypatch, isolated_trend_db):
    """Run an order-dimension task through workflow, SQL, answer and actual SDK wrapper."""
    from app.agents import critic_agent
    from app.core import config as core_config
    from app.graph import workflow as workflow_module
    from app.llm import client as llm
    from scripts.run_resume_dev_pair import QUESTION as ORDER_DIMENSION_QUESTION
    from test_registered_numeric_decomposition_intent import _install_offline_workflow

    real_critic_extract = critic_agent.llm_extract
    real_critic = workflow_module.critic
    critic_passes = []

    def capture_critic_pass(state, **kwargs):
        result = real_critic(state, **kwargs)
        critic = result.get("critic_result") or {}
        critic_passes.append({
            "verdict": result.get("critic_verdict"),
            "revision_target": result.get("critic_revision_target"),
            "data_review": critic.get("data_review"),
            "diagnostic_review": critic.get("diagnostic_review"),
            "claim_review": critic.get("claim_review"),
            "llm_review": critic.get("llm_review"),
            "errors": copy.deepcopy(result.get("errors") or []),
        })
        return result

    monkeypatch.setattr(workflow_module, "critic", capture_critic_pass)
    invoke, captured, statements, _fallback_calls, external_attempts = _install_offline_workflow(
        monkeypatch, isolated_trend_db, "order_dimension",
    )
    settings = core_config.get_settings().model_copy(update={
        "deepseek_api_key": "offline-sdk-placeholder",
        "deepseek_base_url": "https://127.0.0.1/v1",
        "llm_providers_json": "[]",
    })
    monkeypatch.setattr(core_config, "get_settings", lambda: settings)
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    # _record_usage's normal production sink writes to the application usage
    # store. The workflow's trace/token accounting remains real; only that
    # unrelated external persistence sink is isolated at the test boundary.
    monkeypatch.setattr(llm, "record_usage", lambda **_kwargs: None)

    class SDK:
        def __init__(self):
            self.requests = []

        def create(self, **kwargs):
            self.requests.append(kwargs)
            body = json.dumps({"verdict": "passed", "issues": []})
            return type("Response", (), {
                "choices": [type("Choice", (), {
                    "message": type("Message", (), {"content": body})(),
                    "finish_reason": "stop",
                })()],
                "usage": type("Usage", (), {"prompt_tokens": 90, "completion_tokens": 30})(),
            })()

    sdk = SDK()
    client_constructions = []

    class OfflineClient:
        def __init__(self, **kwargs):
            client_constructions.append({k: kwargs.get(k) for k in ("base_url", "timeout", "max_retries")})
            self.chat = type("Chat", (), {"completions": sdk})()

        def close(self):
            pass

    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: OfflineClient(**kwargs))
    monkeypatch.setattr(critic_agent, "llm_extract", real_critic_extract)
    question = ORDER_DIMENSION_QUESTION
    runs = {}
    for arm, mode in (("full", "full"), ("off", "critic_llm_review_off")):
        captured.clear()
        critic_passes.clear()
        before_sdk = len(sdk.requests)
        before_sql = len(statements)
        response = invoke(question, mode=mode)
        runs[arm] = {
            "state": copy.deepcopy(captured), "response": response,
            "sdk": sdk.requests[before_sdk:], "sql": statements[before_sql:],
            "critic_passes": copy.deepcopy(critic_passes),
        }
    assert external_attempts == []

    from _artifact_safety import isolated_output_path
    diagnostic_dir = isolated_output_path("RESUME_DIMENSION_OUTPUT_DIAGNOSTICS")
    if diagnostic_dir:
        def diagnostic_snapshot(run):
            state = run["state"]
            critic = state.get("critic_result") or {}
            reviews = critic.get("llm_review") or {}
            selected_review_fields = {
                name: {
                    key: reviews.get(name, {}).get(key)
                    for key in ("verdict", "completion_status", "failure_kind", "issues", "call_ids", "sdk_attempt_ids")
                    if key in reviews.get(name, {})
                }
                for name in ("groundedness", "completeness", "consistency")
            }
            return {
                "task_type": state.get("task_type"),
                "critic_decisions": state.get("critic_decisions"),
                "final_critic": {
                    "verdict": state.get("critic_verdict"),
                    "revision_target": state.get("critic_revision_target"),
                    "llm_review_status": reviews.get("status"),
                    "reviews": selected_review_fields,
                    "data_review": critic.get("data_review"),
                    "claim_review_status": (critic.get("claim_review") or {}).get("status"),
                    "diagnostic_review": critic.get("diagnostic_review"),
                    "causal_review": critic.get("causal_review"),
                },
                "errors": state.get("errors"),
                "revision_counts": state.get("_revision_counts"),
                "critic_history": [item for item in state.get("node_history", []) if item.get("node") == "critic"],
                "tool_calls": [
                    {key: call.get(key) for key in ("tool_name", "arguments", "status", "request_id")}
                    for call in state.get("tool_calls", [])
                ],
                "runtime_budget": state.get("runtime_budget"),
                "workflow_nodes": [
                    item.get("node") for item in state.get("node_history", [])
                    if item.get("node") in {"data_query", "premise_check", "diagnostic_plan", "policy_execute", "observe", "diagnostic_decide", "generate_answer", "critic"}
                ],
                "sdk_sends": len(run["sdk"]),
                "sdk_attempt_count": (state.get("llm_token_summary") or {}).get("sdk_attempt_count"),
                "critic_extract_calls": [
                    {key: call.get(key) for key in ("call_id", "node", "operation", "outcome", "attempts")}
                    for call in state.get("llm_calls", [])
                    if call.get("node") == "critic" and call.get("operation") == "extract"
                ],
                "presentation_status": (state.get("analysis_presentation") or {}).get("status"),
                "answer": run["response"].get("answer"),
                "critic_passes": run["critic_passes"],
            }

        diagnostic_path = Path(diagnostic_dir)
        diagnostic_path.mkdir(parents=True, exist_ok=True)
        (diagnostic_path / "order_dimension_full_off_critic_diagnostics.json").write_text(
            json.dumps({arm: diagnostic_snapshot(run) for arm, run in runs.items()}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    full = runs["full"]
    off = runs["off"]
    full_review = full["state"]["critic_result"]["llm_review"]
    full_critic_calls = [
        call for call in full["state"]["llm_calls"]
        if call.get("node") == "critic" and call.get("operation") == "extract"
    ]
    assert len(full["sdk"]) == len(full_critic_calls) >= 3
    assert all(full_review[name]["completion_status"] == "completed"
               for name in ("groundedness", "completeness", "consistency"))
    assert all(full_review[name]["call_ids"] and full_review[name]["sdk_attempt_ids"]
               for name in ("groundedness", "completeness", "consistency"))
    assert all(call.get("node") == "critic" for call in full_critic_calls)
    assert full_review["status"] == "passed"
    assert all(full["state"]["critic_result"]["llm_review"][name]["completion_status"] == "completed"
               for name in ("groundedness", "completeness", "consistency"))
    assert len(off["sdk"]) == 0
    assert off["state"]["critic_result"]["llm_review"]["status"] == "evaluation_ablation"
    off_cards = off["state"]["analysis_presentation"]["contributions"]
    assert len(off_cards) == 1
    assert off_cards[0]["dimension"] == "product_category"
    assert (off_cards[0]["baseline_value"], off_cards[0]["current_value"], off_cards[0]["total_delta"]) == (10.0, 6.0, -4.0)
    assert off_cards[0]["evidence_ids"]
    assert "统计贡献" in off["response"]["answer"]
    assert full["sql"] and off["sql"]
    assert any("fact_orders" in str(sql).lower() or "daily_metrics" in str(sql).lower() for sql in full["sql"])
    assert full["state"]["analysis_presentation"]["status"] == "partial"
    contribution_cards = full["state"]["analysis_presentation"]["contributions"]
    assert len(contribution_cards) == 1
    card = contribution_cards[0]
    assert card["dimension"] == "product_category"
    assert card["baseline_value"] == 10.0 and card["current_value"] == 6.0
    assert card["total_delta"] == -4.0 and card["evidence_ids"]
    assert "统计贡献" in full["response"]["answer"]
    assert "业务原因仍未确认" in full["response"]["answer"]
    assert any(item.get("evidence_type") == "dimension_contribution"
               for item in full["state"]["investigation_state"]["evidence"])
    assert full["state"]["llm_token_summary"]["sdk_attempt_count"] == len(full["sdk"])
    assert off["state"]["llm_token_summary"]["sdk_attempt_count"] == 0
    assert len(client_constructions) == len(full["sdk"])
    assert all(item["base_url"] == "https://127.0.0.1/v1" for item in client_constructions)
