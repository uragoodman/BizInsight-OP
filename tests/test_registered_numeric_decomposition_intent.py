"""Normal-workflow acceptance for Registry-governed numeric decomposition intent."""

from __future__ import annotations

import copy
import json
import os
import re
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from test_non_gmv_diagnostic_workflow import seed_facts
from test_platform_metric_trend_acceptance import isolated_trend_db  # noqa: F401


def _save_trace(trace: dict, label: str) -> None:
    # Test artifacts live outside the repository so a rerun can never replace
    # historical evidence under docs/testing.
    trace_id = str(trace.get("trace_id") or uuid4().hex)
    path = Path(tempfile.gettempdir()) / "bizinsight_registered_decomposition_tests" / trace_id
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{label}.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )


def _block_external_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    attempts: list[str] = []

    def blocked(*args, **kwargs):
        attempts.append("network")
        raise AssertionError("external network is blocked in this offline workflow test")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)
    return attempts


def _install_offline_workflow(
    monkeypatch, isolated_trend_db, case_id: str, *,
    critic_sdk_fault: str | None = None, start_date=None,
    evaluation_tool_surface: dict | None = None,
):
    from app.analysis import attribution_methods
    from app.core import time_expression
    from app.data.repositories import daily_metrics
    from app.graph import workflow
    from app.graph.nodes import critic as critic_node
    from app.graph.nodes import triage as triage_node
    from app.llm import client as llm
    from app.agents import critic_agent
    from app.mcp import client as mcp, tools
    from app.tenancy import TenantContext

    engine, statements, _sql_events, _case_results = isolated_trend_db
    seed_facts(engine, case_id, **({"start_date": start_date} if start_date is not None else {}))
    monkeypatch.setattr(daily_metrics, "engine", engine)
    monkeypatch.setattr(time_expression, "engine", engine)
    external_attempts = _block_external_network(monkeypatch)
    model_calls: list[str] = []

    def unavailable(name):
        def call(*args, **kwargs):
            model_calls.append(name)
            return None
        return call

    monkeypatch.setattr(triage_node, "llm_classify", unavailable("classify"))
    for name in ("llm_extract", "llm_chat", "llm_agent_action", "llm_diagnostic_plan"):
        if hasattr(llm, name):
            monkeypatch.setattr(llm, name, unavailable(name))
    if critic_sdk_fault is None:
        if hasattr(critic_agent, "llm_extract"):
            monkeypatch.setattr(critic_agent, "llm_extract", unavailable("critic_extract"))
        if hasattr(critic_node, "llm_extract"):
            monkeypatch.setattr(critic_node, "llm_extract", unavailable("critic_node_extract"))
    else:
        # Keep the public llm_extract and SDK-attempt recorder real for Critic.
        # Only the SDK transport response is replaced with a deterministic
        # transport exception; no network client is used.
        from openai import APIConnectionError, APITimeoutError

        exception_type = {
            "timeout": APITimeoutError,
            "connection": APIConnectionError,
        }[critic_sdk_fault]

        class OfflineFailingClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            def create(self, **kwargs):
                model_calls.append("critic_sdk_fault")
                raise exception_type(request=httpx.Request("POST", "https://offline.invalid/chat/completions"))

        provider = llm._Provider(
            "offline-primary", OfflineFailingClient(),
            lambda operation, explicit, quality: (explicit or "offline-test-model", "standard"),
        )
        monkeypatch.setattr(llm, "_providers", lambda: [provider])
        from app.resilience.policy import _BREAKERS
        _BREAKERS.clear()

    async def local_transport(tool_name, arguments):
        # Replace only the local MCP subprocess boundary. Registry planning,
        # gateway authorization, tool validation and repository SQL remain real.
        assert hasattr(tools, tool_name), f"unexpected tool dispatch: {tool_name}"
        return getattr(tools, tool_name)(**arguments)

    monkeypatch.setattr(mcp, "_call_local_mcp_tool", local_transport)
    captured: dict = {}
    monkeypatch.setattr(workflow, "write_trace", lambda _trace_id, state: captured.update(state))
    monkeypatch.setattr(workflow, "_persist_chat_turn", lambda *args, **kwargs: None)

    def invoke(question: str, mode: str | None = None):
        kwargs = {}
        if mode is not None:
            kwargs["_critic_llm_review_mode"] = mode
        response = workflow.run_chat_workflow(
            question,
            tenant_context=TenantContext(
                tenant_id="platform", user_id="numeric-decomposition-offline",
                role="admin", scope_type="platform",
            ),
            memory_enabled=False,
            trace_id=f"registered-decomposition-{uuid4().hex}",
            **({"_evaluation_tool_surface": evaluation_tool_surface} if evaluation_tool_surface is not None else {}),
            **kwargs,
        )
        assert external_attempts == []
        return response

    return invoke, captured, statements, model_calls, external_attempts


@pytest.mark.parametrize(
    "metric,case_id,method,expected_baseline,expected_current,expected_delta,factors",
    [
        ("GMV", "aov_ratio", "product_shapley_v1", 100.0, 120.0, 20.0,
         {"order_count", "avg_order_value"}),
        ("客单价", "aov_ratio", "symmetric_ratio_v1", 10.0, 20.0, 10.0,
         {"gmv", "order_count"}),
        ("运费率", "freight_ratio", "symmetric_ratio_v1", 10.0, 20.0, 10.0,
         {"freight_value", "gmv"}),
    ],
)
def test_registered_numeric_decomposition_runs_through_normal_workflow(
    monkeypatch, isolated_trend_db, metric, case_id, method,
    expected_baseline, expected_current, expected_delta, factors,
):
    invoke, trace, statements, model_calls, _external_attempts = _install_offline_workflow(
        monkeypatch, isolated_trend_db, case_id,
    )
    expected_target = {
        "GMV": "gmv", "客单价": "avg_order_value", "运费率": "freight_rate",
    }[metric]
    question = (
        f"请对{metric}按注册分解方法提供数值变化贡献，比较2024-05-03与2024-05-04，"
        "只说明数值解释，不推断业务因果。"
    )
    invoke(question)
    _save_trace(trace, f"decomposition_{expected_target}")

    compilation = trace["context_compilation"]
    decision = trace["metric_planning_decision"]
    request = decision["registered_decomposition"]
    investigation = trace["investigation_state"]
    decomposition_evidence = [
        item for item in investigation["evidence"]
        if item.get("evidence_type") == "metric_decomposition"
        and item.get("method_id") == method
    ]
    analysis = trace["analysis_result"]
    presentation = trace.get("analysis_presentation") or {}
    card = presentation.get("metric_decomposition")

    assert trace["task_type"] == "metric_query"
    assert compilation["operation_intent"]["kind"] == "registered_numeric_decomposition"
    assert compilation["operation_intent"]["target_metric_id"] == expected_target
    assert compilation["known_slots"]["baseline_date"]["value"] == "2024-05-03"
    assert compilation["known_slots"]["current_date"]["value"] == "2024-05-04"
    assert decision["operation_intent"] == "registered_numeric_decomposition"
    assert decision["confirmed_metrics"] == [expected_target]
    assert request["status"] == "ready"
    assert request["target_metric_id"] == expected_target
    assert request["method_id"] == method
    assert set(request["factor_metric_ids"]) == factors
    assert request["baseline_date"] == "2024-05-03"
    assert request["current_date"] == "2024-05-04"
    plan = trace["metric_query_batch_plan"]["plans"][0]
    assert plan["baseline_date"] == request["baseline_date"]
    assert plan["current_date"] == request["current_date"]
    assert len(decomposition_evidence) == 1
    values = decomposition_evidence[0]["values"]
    assert values["baseline_value"] == pytest.approx(expected_baseline)
    assert values["current_value"] == pytest.approx(expected_current)
    assert values["total_delta"] == pytest.approx(expected_delta)
    expected_metric_unit = {"gmv": "BRL", "avg_order_value": "BRL/order", "freight_rate": "%"}[expected_target]
    expected_component_unit = "percentage_points" if expected_metric_unit == "%" else expected_metric_unit
    assert decomposition_evidence[0]["unit"] == expected_metric_unit
    assert all(
        item["unit"] == expected_component_unit
        for item in values["display_components"]
    )
    assert decomposition_evidence[0]["authorized_metric_provenance"] == "tool_gateway_policy_validated"
    assert analysis["task_type"] == "metric_query"
    assert analysis["abstain"] is False
    assert analysis["metric_decomposition"]["evidence_ids"] == [decomposition_evidence[0]["evidence_id"]]
    assert card and card["evidence_ids"] == [decomposition_evidence[0]["evidence_id"]]
    assert card["metric_id"] == expected_target
    assert all(call["tool_name"] == "registry_trend" for call in trace["tool_calls"])
    assert len(trace["tool_calls"]) == len(request["input_metric_ids"])
    assert statements
    assert "metric_compare" not in [call["tool_name"] for call in trace["tool_calls"]]
    assert "dimension_contribution" not in [item.get("evidence_type") for item in investigation["evidence"]]
    # The deterministic decomposition remains valid, but this test's full-mode
    # Critic stub deliberately returns no review verdict. The final presentation
    # must therefore disclose the incomplete review instead of retaining the
    # answer node's earlier completed status.
    assert presentation["status"] == "partial"
    assert "可选 LLM 审核未完整完成；当前结果未获完整审核。" in presentation["limitations"]
    assert presentation["scope"] == {
        "seller_id": None, "start_date": "2024-05-03", "end_date": "2024-05-04", "granularity": "day",
    }
    assert card["unit"] == expected_metric_unit
    assert card["change_unit"] == expected_component_unit


    assert all(item["unit"] == expected_component_unit for item in card["factors"])
    assert "\u4e1a\u52a1\u53d8\u5316\u539f\u56e0\u5c1a\u672a\u786e\u8ba4" in "".join(analysis["limitations"])
    answer = trace.get("answer", "")
    expected_unit = expected_metric_unit
    expected_change_unit = (
        "\u4e2a\u767e\u5206\u70b9" if expected_component_unit == "percentage_points" else expected_component_unit
    )
    expected_fragments = (
        f"{expected_baseline:.2f} {expected_unit}", f"{expected_current:.2f} {expected_unit}",
        f"+{expected_delta:.2f} {expected_change_unit}",
    )
    assert all(value in answer for value in expected_fragments), {
        "expected_fragments": expected_fragments,
        "actual_answer": answer,
    }
    assert trace["critic_result"]["data_review"]["status"] == "passed"
    assert trace["critic_result"]["causal_review"]["status"] == "approved"
    assert trace["critic_verdict"] == "rejected"
    # Query selection, registered decomposition, and evidence construction do
    # not depend on an LLM response. The first unavailable optional review
    # stops full review; it cannot approve or supply any facts.
    critic_calls = [call for call in model_calls if call in {"critic_extract", "critic_node_extract"}]
    task_model_calls = [call for call in model_calls if call not in {"critic_extract", "critic_node_extract"}]
    assert task_model_calls == []
    assert len(critic_calls) == 1
    assert trace["critic_llm_review_mode"] == "full"
@pytest.mark.parametrize(
    "target_metric,case_id,method_id,input_metric_ids",
    [
        ("gmv", "aov_ratio", "product_shapley_v1", {"gmv", "order_count", "avg_order_value"}),
        ("avg_order_value", "aov_ratio", "symmetric_ratio_v1", {"avg_order_value", "gmv", "order_count"}),
        ("freight_rate", "freight_ratio", "symmetric_ratio_v1", {"freight_rate", "freight_value", "gmv"}),
    ],
)
def test_registered_decomposition_with_frozen_bg_tool_surface_executes_registered_inputs(
    monkeypatch, isolated_trend_db, target_metric, case_id, method_id, input_metric_ids,
):
    from evaluation.resume_bg_tool_surface import build_evaluation_tool_surface
    from app.tenancy import TenantContext

    identity = TenantContext(
        tenant_id="platform", user_id="numeric-decomposition-surface",
        role="admin", scope_type="platform",
    )
    surface = build_evaluation_tool_surface(
        {"target_metric": target_metric, "scope": "platform"}, identity,
    )
    invoke, trace, statements, _model_calls, external_attempts = _install_offline_workflow(
        monkeypatch, isolated_trend_db, case_id,
        evaluation_tool_surface=surface,
    )
    metric_phrase = {
        "gmv": "GMV", "avg_order_value": "\u5ba2\u5355\u4ef7", "freight_rate": "\u8fd0\u8d39\u7387",
    }[target_metric]
    question = (
        f"\u8bf7\u5bf9{metric_phrase}\u6309\u6ce8\u518c\u6570\u503c\u5206\u89e3\u65b9\u6cd5\u63d0\u4f9b\u6570\u503c\u53d8\u5316\u8d21\u732e\uff0c"
        f"\u6bd4\u8f832024-05-03\u4e0e2024-05-04\uff0c\u53ea\u8bf4\u660e\u6570\u503c\u89e3\u91ca\uff0c\u4e0d\u63a8\u65ad\u4e1a\u52a1\u56e0\u679c\u3002"
    )
    invoke(question, mode="critic_llm_review_off")

    request = trace["registered_decomposition_request"]
    assert request["status"] == "ready"
    assert request["target_metric_id"] == target_metric
    assert request["method_id"] == method_id
    assert set(request["input_metric_ids"]) == input_metric_ids
    calls = trace["tool_calls"]
    assert len(calls) == len(input_metric_ids)
    assert {call["arguments"]["metric_name"] for call in calls} == set(request["input_metric_ids"])
    assert all(call["tool_name"] == "registry_trend" and call["status"] == "ok" for call in calls)
    assert not any(
        error.get("code") == "evaluation_tool_surface_violation"
        for error in trace.get("errors", [])
    )
    decomposition = trace["analysis_presentation"]["metric_decomposition"]
    assert decomposition["metric_id"] == target_metric
    assert decomposition["evidence_ids"]
    assert trace["analysis_presentation"]["status"] == "completed"
    assert statements and external_attempts == []
    assert trace["critic_result"]["llm_review"]["status"] == "evaluation_ablation"
    assert not any(call in {"critic_extract", "critic_node_extract"} for call in _model_calls)


def test_registered_decomposition_off_still_runs_deterministic_causal_review(
    monkeypatch, isolated_trend_db,
):
    from app.graph.nodes.critic import _review_causal_language

    invoke, trace, _statements, model_calls, _external_attempts = _install_offline_workflow(
        monkeypatch, isolated_trend_db, "aov_ratio",
    )
    invoke(
        "请对客单价按注册分解方法提供数值变化贡献，比较2024-05-03与2024-05-04；"
        "只说明数值解释，不推断业务因果。",
        mode="critic_llm_review_off",
    )
    assert trace["critic_result"]["causal_review"]["status"] == "approved"
    assert trace["critic_result"]["llm_review"]["status"] == "evaluation_ablation"
    assert not any(call in {"critic_extract", "critic_node_extract"} for call in model_calls)

    unsupported = dict(trace)
    unsupported["answer"] = "Order decline caused the GMV drop."
    assert _review_causal_language(unsupported)["status"] == "rejected"

    ordinary = {"task_type": "metric_query", "answer": "Order decline caused the GMV drop."}
    assert _review_causal_language(ordinary)["status"] == "skipped"


@pytest.mark.parametrize(
    "fault,expected_failure_kind,expected_exception",
    [
        ("timeout", "timeout", "APITimeoutError"),
        ("connection", "provider_error", "APIConnectionError"),
    ],
)
def test_registered_decomposition_critic_transport_fault_preserves_result_and_downgrades_status(
    monkeypatch, isolated_trend_db, fault, expected_failure_kind, expected_exception,
):
    """A real public LLM SDK failure must leave the valid decomposition visible but not completed."""
    from app.llm import client as llm

    settings = llm.get_settings().model_copy(update={
        "llm_resilience_attempts": 1,
        "llm_timeout_seconds": 1.0,
    })
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    invoke, trace, statements, model_calls, external_attempts = _install_offline_workflow(
        monkeypatch, isolated_trend_db, "aov_ratio", critic_sdk_fault=fault,
    )
    response = invoke(
        "请对客单价按注册分解方法提供数值变化贡献，比较2024-05-03与2024-05-04，"
        "只说明数值解释，不推断业务因果。"
    )
    _save_trace(trace, f"critic_sdk_{fault}")

    calls = trace["llm_calls"]
    assert len(calls) == 1
    call = calls[0]
    assert call["node"] == "critic"
    assert call["operation"] == "extract"
    assert call["outcome"] == "provider_error"
    assert len(call["attempts"]) == 1
    attempt = call["attempts"][0]
    assert attempt["sdk_request_sent"] is True
    assert attempt["exception_type"] == expected_exception
    assert attempt["input_tokens"] is None and attempt["output_tokens"] is None
    assert trace["critic_result"]["llm_review"]["groundedness"]["failure_kind"] == expected_failure_kind
    assert trace["critic_result"]["llm_review"]["completeness"]["failure_kind"] == "not_attempted_after_prior_review_incomplete"
    assert trace["critic_result"]["llm_review"]["consistency"]["failure_kind"] == "not_attempted_after_prior_review_incomplete"
    assert trace["critic_verdict"] == "rejected"
    assert trace["critic_result"]["data_review"]["status"] == "passed"
    assert trace["registered_decomposition_request"]["status"] == "ready"
    valid_decomposition = [
        item for item in trace["investigation_state"]["evidence"]
        if item.get("evidence_type") == "metric_decomposition"
        and item.get("method_id") == "symmetric_ratio_v1"
    ]
    assert len(valid_decomposition) == 1
    values = valid_decomposition[0]["values"]
    assert values["baseline_value"] == pytest.approx(10.0)
    assert values["current_value"] == pytest.approx(20.0)
    assert values["total_delta"] == pytest.approx(10.0)
    assert trace["analysis_result"]["metric_decomposition"]["evidence_ids"] == [valid_decomposition[0]["evidence_id"]]
    assert trace["analysis_presentation"]["status"] == "partial"
    assert "可选 LLM 审核未完整完成；当前结果未获完整审核。" in trace["analysis_presentation"]["limitations"]
    assert response["presentation"]["status"] == "partial"
    assert "10.00 BRL/order" in response["answer"]
    assert "20.00 BRL/order" in response["answer"]
    assert "+10.00 BRL/order" in response["answer"]
    assert statements and model_calls == ["critic_sdk_fault"]
    assert external_attempts == []


def test_ordinary_compare_and_single_day_distribution_keep_their_operations(
    monkeypatch, isolated_trend_db,
):
    invoke, trace, statements, _model_calls, _external_attempts = _install_offline_workflow(
        monkeypatch, isolated_trend_db, "aov_ratio",
    )
    invoke("比较2024-05-03与2024-05-04的GMV。")
    _save_trace(trace, "ordinary_metric_compare")
    comparison = trace["metric_query_batch_plan"]["decision"]
    assert trace["context_compilation"]["operation_intent"]["kind"] == "ordinary_metric_query"
    assert comparison["query_type"] == "metric_compare"
    assert trace["tool_result"]["result"]["parsed"]["query_type"] == "metric_compare"
    assert any("daily_metrics" in sql.lower() for sql in statements)

    invoke, trace, statements, _model_calls, _external_attempts = _install_offline_workflow(
        monkeypatch, isolated_trend_db, "aov_ratio",
    )
    invoke("查询2024-05-04的GMV按商品类别分布。")
    _save_trace(trace, "single_day_dimension_breakdown")
    breakdown = trace["metric_query_batch_plan"]["decision"]
    assert breakdown["query_type"] == "metric_breakdown"
    assert trace["tool_result"]["result"]["parsed"]["query_type"] == "metric_breakdown"
    assert trace["tool_result"]["result"]["parsed"]["dimension"] == "product_category"
    assert statements


@pytest.mark.parametrize(
    "question,expected_baseline,expected_current,expected_delta,expected_ratio",
    [
        ("Compare 2024-05-04 and 2024-05-03 GMV", "2024-05-03", "2024-05-04", 20.0, 20.0),
        (
            "Compare baseline date 2024-05-04 with current date 2024-05-03 GMV",
            "2024-05-04", "2024-05-03", -20.0, -16.67,
        ),
        (
            "Compare current date 2024-05-03 with 2024-05-04 GMV",
            "2024-05-04", "2024-05-03", -20.0, -16.67,
        ),
        (
            "\u6bd4\u8f83\u57fa\u7ebf\u671f2024-05-04\u548c\u5f53\u524d\u671f2024-05-03\u7684GMV",
            "2024-05-04", "2024-05-03", -20.0, -16.67,
        ),
    ],
)
def test_normal_workflow_keeps_comparison_date_roles_from_question_to_answer(
    monkeypatch, isolated_trend_db, question, expected_baseline, expected_current,
    expected_delta, expected_ratio,
):
    invoke, trace, statements, _model_calls, _external_attempts = _install_offline_workflow(
        monkeypatch, isolated_trend_db, "aov_ratio",
    )
    invoke(question)
    _save_trace(trace, f"normal_compare_roles_{expected_baseline}_{expected_current}")

    slots = trace["context_compilation"]["known_slots"]
    plan = trace["metric_query_batch_plan"]["plans"][0]
    result = trace["tool_result"]["result"]["compare_result"]
    change = trace["analysis_presentation"]["changes"][0]
    assert slots["baseline_date"]["value"] == expected_baseline
    assert slots["current_date"]["value"] == expected_current
    assert plan["baseline_date"] == expected_baseline
    assert plan["current_date"] == expected_current
    # The old repository contract is left=baseline/right=current. The
    # public result is normalized back to canonical and current-oriented fields.
    assert trace["tool_result"]["arguments"]["left_date"] == expected_baseline
    assert trace["tool_result"]["arguments"]["right_date"] == expected_current
    assert result["baseline_date"] == expected_baseline
    assert result["current_date"] == expected_current
    assert result["baseline_value"] == pytest.approx(120.0 if expected_baseline.endswith("04") else 100.0)
    assert result["current_value"] == pytest.approx(120.0 if expected_current.endswith("04") else 100.0)
    assert result["delta_value"] == pytest.approx(expected_delta)
    assert result["delta_ratio"] == pytest.approx(expected_ratio)
    assert change["baseline_value"] == result["baseline_value"]
    assert change["current_value"] == result["current_value"]
    assert change["delta_value"] == result["delta_value"]
    assert change["delta_ratio"] == result["delta_ratio"]
    assert change["baseline_period"] == expected_baseline
    assert change["current_period"] == expected_current
    assert f"\u5f53\u524d\u671f {expected_current}" in trace["answer"]
    assert f"\u57fa\u7ebf\u671f {expected_baseline}" in trace["answer"]
    assert f"{expected_delta:.2f}" in trace["answer"]
    assert f"{expected_ratio:.2f}" in trace["answer"]
    assert statements


@pytest.mark.parametrize("missing_baseline", [False, True], ids=["zero-baseline", "missing-baseline"])
def test_normal_workflow_compare_keeps_zero_and_missing_semantics(
    monkeypatch, isolated_trend_db, missing_baseline,
):
    from sqlalchemy import text

    invoke, trace, _statements, _model_calls, _external_attempts = _install_offline_workflow(
        monkeypatch, isolated_trend_db, "aov_ratio",
    )
    engine, _captured_statements, _sql_events, _case_results = isolated_trend_db
    if missing_baseline:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM analytics.daily_metrics WHERE metric_date = '2024-05-03'"))
    else:
        with engine.begin() as conn:
            conn.execute(text("UPDATE analytics.daily_metrics SET gmv = 0 WHERE metric_date = '2024-05-03'"))

    invoke("Compare baseline date 2024-05-03 with current date 2024-05-04 GMV")
    _save_trace(trace, "normal_compare_missing_baseline" if missing_baseline else "normal_compare_zero_baseline")
    result = trace["tool_result"]["result"]["compare_result"]
    assert trace["tool_result"]["arguments"]["left_date"] == "2024-05-03"
    assert trace["tool_result"]["arguments"]["right_date"] == "2024-05-04"
    assert result["baseline_date"] == "2024-05-03"
    assert result["current_date"] == "2024-05-04"
    assert result["current_value"] == pytest.approx(120.0)

    if missing_baseline:
        assert result["baseline_value"] is None
        assert result["status"] == "not_found"
        assert result["delta_value"] is None
        assert result["delta_ratio"] is None
        assert trace["analysis_presentation"]["changes"] == []
        assert "120.00" in trace["answer"]
        assert "暂无数据" in trace["answer"]
    else:
        assert result["baseline_value"] == 0.0
        assert result["status"] == "ok"
        assert result["delta_value"] == pytest.approx(120.0)
        assert result["delta_ratio"] is None
        card = trace["analysis_presentation"]["changes"][0]
        assert (card["baseline_value"], card["current_value"], card["delta_value"]) == (0.0, 120.0, 120.0)
        assert card["delta_ratio"] is None
        assert "120.00" in trace["answer"]
        assert re.search(r"变化率\s*=\s*暂无数据\s*%", trace["answer"])


def test_metric_definition_still_uses_definition_route(monkeypatch, isolated_trend_db):
    from app.graph.nodes import metric_definition as definition_node
    from app.rag.schemas import RagSearchResult

    invoke, trace, _statements, _model_calls, _external_attempts = _install_offline_workflow(
        monkeypatch, isolated_trend_db, "aov_ratio",
    )

    def empty_local_retrieval(query, **kwargs):
        return RagSearchResult(query=query, backend="local", top_k=3, hits=[], context="")

    monkeypatch.setattr(definition_node, "retrieve_knowledge", empty_local_retrieval)
    monkeypatch.setattr(definition_node, "retrieve_routed", empty_local_retrieval)
    invoke("GMV的定义是什么？")
    _save_trace(trace, "definition_route")
    assert trace["task_type"] == "metric_definition"
    assert trace["context_compilation"]["operation_intent"]["kind"] == "ordinary_metric_query"
    assert trace["metric_authoritative_definition"]["metric_id"] == "gmv"
    assert not trace.get("diagnostic_plan")


@pytest.mark.parametrize(
    "question,case_id,expected_status",
    [
        ("请对review_score_avg按注册数值分解方法解释2024-05-03与2024-05-04的变化。", "aov_ratio", "unsupported"),
        ("请对avg_order_value按注册数值分解方法比较2024-05-03与2024-05-04。", "ratio_missing", "partial"),
    ],
)
def test_unsupported_method_and_missing_input_do_not_fall_back_to_compare(
    monkeypatch, isolated_trend_db, question, case_id, expected_status,
):
    invoke, trace, statements, _model_calls, _external_attempts = _install_offline_workflow(
        monkeypatch, isolated_trend_db, case_id,
    )
    invoke(question)
    _save_trace(trace, "unsupported_registered_method" if expected_status == "unsupported" else "missing_registered_input")
    decision = trace.get("metric_planning_decision") or {}
    assert decision.get("operation_intent") == "registered_numeric_decomposition"
    assert trace.get("tool_result", {}).get("result", {}).get("status") in {
        "unsupported", "planned", "ok", "not_found",
    }
    assert not any(call.get("tool_name") == "compare_metric" for call in trace.get("tool_calls", []))
    if expected_status == "unsupported":
        assert decision.get("registered_decomposition", {}).get("status") == "unsupported"
        assert trace["analysis_presentation"]["status"] == "unsupported"
        assert not any(call.get("tool_name") not in {"metric_clarification"} for call in trace.get("tool_calls", []))
        assert trace["critic_result"]["data_review"]["status"] == "incomplete"
        assert not any(error.get("code") == "critic_revision_limit_exceeded" for error in trace.get("errors", []))
    else:
        stop = (trace.get("investigation_state") or {}).get("stop_reason") or {}
        assert stop.get("code") == "DATA_INSUFFICIENT"
        assert not any(
            item.get("evidence_type") == "metric_decomposition"
            for item in (trace.get("investigation_state") or {}).get("evidence", [])
        )
        assert trace["analysis_presentation"]["status"] == "partial"
        assert trace["critic_result"]["data_review"]["status"] == "incomplete"
        gaps = trace.get("diagnostic_candidate_data_gaps", [])
        assert any(gap.get("failure_code") == "DATA_INSUFFICIENT" and gap.get("missing_input_metric_ids") for gap in gaps)
        assert len(trace["tool_calls"]) == len(decision["registered_decomposition"]["input_metric_ids"])
        assert any(sql.lstrip().lower().startswith(("select", "with")) for sql in statements)


def test_factor_authorization_is_checked_before_any_query(monkeypatch):
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    question = (
        "GMV registered numeric decomposition for 2024-05-03 and 2024-05-04; "
        "use order_count and avg_order_value factors."
    )
    plan = plan_confirmed_metric_queries(
        question,
        scope="platform",
        allowed_metrics={"gmv"},
        scope_data={"tenant_id": "platform", "user_id": "test", "scope_type": "platform",
                    "allowed_metrics": ["gmv"]},
        operation_intent={"kind": "registered_numeric_decomposition", "target_metric_id": "gmv"},
    )
    assert plan.decision.operation_intent == "registered_numeric_decomposition"
    assert plan.decision.status == "unsupported"
    assert plan.decision.registered_decomposition.status == "authorization_blocked"
    assert not plan.plans
