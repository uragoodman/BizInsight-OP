"""Mechanical offline check for the development-only bare tool loop."""

from __future__ import annotations

import ipaddress
import asyncio
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from test_non_gmv_diagnostic_workflow import seed_facts  # noqa: F401
from test_platform_metric_trend_acceptance import isolated_trend_db  # noqa: F401


QUESTION = (
    "请解释平台已交付订单量从 2024-05-03 到 2024-05-04 的变化，"
    "先核对趋势，再按 product_category 展示统计贡献；统计贡献不等于业务因果。"
)


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def is_loopback(address):
        host = address[0] if isinstance(address, tuple) else address
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except (TypeError, ValueError):
            return False

    def guarded_connect(sock, address):
        if is_loopback(address):
            return original_connect(sock, address)
        raise AssertionError(f"external network blocked during offline B-loop check: {address!r}")

    def guarded_connect_ex(sock, address):
        if is_loopback(address):
            return original_connect_ex(sock, address)
        raise AssertionError(f"external network blocked during offline B-loop check: {address!r}")

    def guarded_create_connection(address, *args, **kwargs):
        if is_loopback(address):
            return original_create_connection(address, *args, **kwargs)
        raise AssertionError(f"external network blocked during offline B-loop check: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected SDK request in offline B-loop check")
        content = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(content, ensure_ascii=False)),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=60),
        )


def test_bare_loop_preserves_provider_timeout_instead_of_calling_it_invalid_action(
    monkeypatch, tmp_path: Path,
):
    """An SDK timeout after a successful tool result is an execution failure, not bad JSON/action."""
    import httpx
    from openai import APITimeoutError

    from app.llm.call_config import LLMCallConfig
    from app.tenancy import TenantContext
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop
    from test_provider_candidate_context_boundary import _install_offline_client

    question = (
        "请比较 2017-02-03 与 2017-02-04 平台已交付订单的客单价，"
        "按已注册分解方法拆分数值变化贡献；只作数值解释，不推断业务因果。"
    )
    action = {
        "action": "TOOL_CALL", "tool": "query_trend",
        "arguments": {
            "metric_id": "avg_order_value", "start_date": "2017-02-03", "end_date": "2017-02-04",
        },
    }
    sdk_calls = 0

    def sdk_response_or_timeout(_kwargs):
        nonlocal sdk_calls
        sdk_calls += 1
        if sdk_calls == 1:
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(action, ensure_ascii=False)),
                    finish_reason="stop",
                )],
                usage=SimpleNamespace(prompt_tokens=808, completion_tokens=198),
            )
        raise APITimeoutError(request=httpx.Request("POST", "https://offline.invalid"))

    llm, call_config, ledger, sends = _install_offline_client(
        monkeypatch, tmp_path, timeout=0.1, raw_create=sdk_response_or_timeout,
    )
    assert isinstance(call_config, LLMCallConfig)
    context = TenantContext(
        tenant_id="platform", user_id="resume-b-timeout-regression", role="admin", scope_type="platform",
    )

    output = run_bare_tool_loop(
        question, tenant_context=context, run_id="bare-provider-timeout", llm_call_config=call_config,
        tool_fn=lambda **_kwargs: {
            "tool_name": "query_metric_trend",
            "result": {"status": "ok", "trend_result": {"status": "ok", "points": []}},
        },
    )

    assert sdk_calls == 2
    assert sends == {"bare-provider-timeout": 2}
    assert [call["outcome"] for call in output["llm_calls"]] == ["success", "provider_error"]
    assert output["tool_calls"][0]["call_status"] == "succeeded"
    assert output["terminal_code"] == "EXECUTION_FAILED"
    assert output["presentation"]["status"] == "failed"
    assert output["presentation"]["failure_kind"] == "timeout"
    assert output["errors"] == [{
        "code": "llm_provider_error", "failure_kind": "timeout",
        "exception_type": "APITimeoutError", "operation": "extract",
        "call_id": output["llm_calls"][1]["call_id"],
        "attempt_id": output["llm_calls"][1]["attempts"][0]["attempt_id"],
    }]
    assert all(row["event"] != "sdk_send_started" or row["run_id"] == "bare-provider-timeout"
               for row in ledger.events)
    assert output["final"] == {
        "action": "FINAL",
        "answer": "模型请求超时，分析未完成；此前成功执行的查询结果已保留，但未形成最终结论。",
        "claims": [],
        "presentation": output["presentation"],
    }
    assert output["final_display"] == output["final"]
    assert output["final_output_adaptation"]["source"] == "deterministic_terminal"
    assert output["final"]["presentation"]["status"] == "failed"
    assert output["final"].get("claims") == []  # Failure notice does not invent an evidence-backed conclusion.


def test_bare_loop_keeps_malformed_model_action_distinct_from_provider_timeout(monkeypatch):
    from app.tenancy import TenantContext
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop

    question = "请比较 2017-02-03 与 2017-02-04 平台已交付订单的客单价。"
    context = TenantContext(
        tenant_id="platform", user_id="resume-b-invalid-action-regression", role="admin", scope_type="platform",
    )
    output = run_bare_tool_loop(
        question, tenant_context=context, run_id="bare-invalid-action",
        model_fn=lambda *_args: None,
        tool_fn=lambda **_kwargs: pytest.fail("invalid action must stop before tool execution"),
    )

    assert output["errors"] == [{"code": "invalid_baseline_action"}]
    assert output.get("terminal_code") is None


def test_bare_loop_records_empty_provider_response_as_terminal_failure_after_tools(monkeypatch, tmp_path: Path):
    """An empty SDK response after valid evidence is visible, bounded, and never scored as completion."""
    from app.tenancy import TenantContext
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop
    from test_provider_candidate_context_boundary import _install_offline_client

    question = (
        "请解释平台已交付订单量从2017-04-07到2017-04-08的变化，先核对趋势，"
        "再按product_category展示统计贡献。"
    )
    action_rows = [
        {"action": "TOOL_CALL", "tool": "query_trend", "arguments": {
            "metric_id": "order_count", "start_date": "2017-04-07", "end_date": "2017-04-08",
        }},
        {"action": "TOOL_CALL", "tool": "metric_contribution", "arguments": {
            "metric_id": "order_count", "baseline_date": "2017-04-07",
            "current_date": "2017-04-08", "dimension": "product_category",
        }},
    ]
    responses = iter(action_rows)
    sdk_sends = 0

    def raw_create(_kwargs):
        nonlocal sdk_sends
        sdk_sends += 1
        action = next(responses, None)
        content = json.dumps(action, ensure_ascii=False) if action is not None else ""
        completion_tokens = 80 if action is not None else 4096
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop" if action is not None else "length",
            )],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=completion_tokens),
        )

    _llm, call_config, ledger, send_counts = _install_offline_client(
        monkeypatch, tmp_path, raw_create=raw_create,
    )
    context = TenantContext(
        tenant_id="platform", user_id="resume-b-empty-response", role="admin", scope_type="platform",
    )

    def tool_fn(*, tool, arguments, **_kwargs):
        if tool == "query_trend":
            return {"tool_name": "query_metric_trend", "result": {
                "status": "ok", "trend_result": {
                    "status": "ok", "metric_name": "order_count", "unit": "orders",
                    "points": [
                        {"bucket": "2017-04-07", "value": 82},
                        {"bucket": "2017-04-08", "value": 64},
                    ],
                },
            }}
        assert tool == "metric_contribution"
        return {"tool_name": "compare_breakdown_by_dimension", "result": {
            "status": "ok", "contribution_result": {
                "status": "ok", "metric_name": arguments["metric_id"],
                "dimension": arguments["dimension"],
            },
        }}

    output = run_bare_tool_loop(
        question, tenant_context=context, run_id="bare-empty-provider-response",
        llm_call_config=call_config, tool_fn=tool_fn,
    )

    assert sdk_sends == 3
    assert send_counts == {"bare-empty-provider-response": 3}
    assert [call["call_status"] for call in output["tool_calls"]] == ["succeeded", "succeeded"]
    assert len(output["evidence_bundles"]) == 2
    assert [call["outcome"] for call in output["llm_calls"]] == [
        "success", "success", "empty_response",
    ]
    failed_call = output["llm_calls"][-1]
    failed_attempt = failed_call["attempts"][-1]
    assert failed_attempt["sdk_request_sent"] is True
    assert failed_attempt["output_tokens"] == 4096
    assert output["terminal_code"] == "EXECUTION_FAILED"
    assert output["presentation"] == {
        "status": "failed", "limitation_code": "EXECUTION_FAILED",
        "failure_kind": "empty_model_response",
    }
    assert output["errors"] == [{
        "code": "llm_empty_response", "failure_kind": "empty_model_response",
        "operation": "extract", "call_id": failed_call["call_id"],
        "attempt_id": failed_attempt["attempt_id"], "sdk_request_sent": True,
        "usage_status": "complete",
    }]
    assert output["final"]["answer"] == "数据查询已执行，但模型未返回可用的最终回答；本次分析未完成。"
    assert output["final"]["presentation"]["status"] == "failed"
    assert output["model_authored_final"] is None
    assert output["final_output_adaptation"]["source"] == "deterministic_terminal"
    assert len([row for row in ledger.events if row["event"] == "sdk_send_started"]) == 3


@pytest.mark.parametrize("inside_running_loop", [False, True])
def test_bare_loop_uses_real_authorized_gateway_and_isolated_sql(
    monkeypatch, tmp_path: Path, isolated_trend_db, inside_running_loop: bool,
):
    from app.llm import client as llm
    import app.core.config as core_config
    from app.tenancy import TenantContext
    from app.llm.call_config import LLMCallConfig
    from evaluation.provider_candidate_batch import (
        CostPolicy, EstimatedCostLedger, ExperimentConfig, RunLedger,
        _reconcile_cost_accounting, bind_isolated_experiment_runtime,
        token_usage_snapshot,
    )
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop
    from test_non_gmv_diagnostic_workflow import seed_facts

    engine, sql_statements, _sql_events, _cases = isolated_trend_db
    seed_facts(engine, "order_dimension")

    run_id = f"resume-b-offline-{uuid4()}"
    actions = [
        {"action": "TOOL_CALL", "tool": "query_trend", "arguments": {
            "metric_id": "order_count", "start_date": "2024-05-03", "end_date": "2024-05-04",
        }},
        {"action": "TOOL_CALL", "tool": "metric_contribution", "arguments": {
            "metric_id": "order_count", "baseline_date": "2024-05-03", "current_date": "2024-05-04",
            "dimension": "product_category",
        }},
        {"action": "FINAL", "answer": "订单量从 10 降至 6，减少 4；贡献是统计解释，不是业务因果。",
         "claims": [{"text": "订单量变化为 -4", "evidence_ids": [f"{run_id}:evidence:1", f"{run_id}:evidence:2"]}]},
    ]
    fake = _FakeCompletions(actions)

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=fake)

        def close(self):
            return None

    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: FakeClient(**kwargs))
    safe_settings = core_config.get_settings().model_copy(update={
        "deepseek_api_key": "offline-placeholder",
        "llm_providers_json": "[]",
    })
    monkeypatch.setattr(core_config, "get_settings", lambda: safe_settings)
    data_dir = tmp_path / "traces"
    ledger = RunLedger(tmp_path / "run_events.jsonl")
    config = ExperimentConfig(max_latency_ms=240_000)
    context = TenantContext(
        tenant_id="platform", user_id="resume-dev-pair-order-dimension", role="admin", scope_type="platform",
    )

    with bind_isolated_experiment_runtime(engine=engine, trace_dir=data_dir, ledger=ledger,
                                          config=config):
        call_config = LLMCallConfig.provider_ab_experiment(
            safe_settings.model_copy(update={"llm_timeout_seconds": config.model_timeout_seconds}),
            provider_id=config.provider_id,
            max_completion_tokens=config.max_completion_tokens,
        )
        def invoke():
            return run_bare_tool_loop(
                QUESTION, tenant_context=context, run_id=run_id, llm_call_config=call_config,
            )
        if inside_running_loop:
            async def on_running_loop():
                return invoke()
            output = asyncio.run(on_running_loop())
        else:
            output = invoke()

    assert output["errors"] == []
    assert len(fake.requests) == 3
    assert len(output["tool_calls"]) == 2
    assert [call["tool"] for call in output["tool_calls"]] == ["query_trend", "metric_contribution"]
    assert all(call["request_id"] and call["status"] == "ok" for call in output["tool_calls"])
    assert all(call["response"]["result"]["parsed"]["metric_name"] == "order_count"
               for call in output["tool_calls"])
    assert output["tool_calls"][1]["arguments"]["dimension"] == "product_category"
    assert output["final"]["action"] == "FINAL"
    assert len(sql_statements) >= 2
    sends = [row for row in ledger.events if row["run_id"] == run_id and row["event"] == "sdk_send_started"]
    assert len(sends) == 3
    assert all(row.get("call_id") and row.get("attempt_id") for row in sends)
    assert output["llm_call_summary"]["sdk_attempts"] == 3
    usage = token_usage_snapshot(
        {"state_snapshot": output}, ledger.events, run_id=run_id,
    )
    assert usage["status"] == "complete"
    assert usage["sdk_sends"] == 3
    assert usage["total_input_tokens"] == 360
    assert usage["total_output_tokens"] == 180
    assert usage["total_tokens"] == 540
    cost_policy = CostPolicy(cumulative_estimate_limit_cny=5.0)
    cost_ledger = EstimatedCostLedger(cost_policy, tmp_path / "cost_reservations.jsonl")
    reservation = cost_ledger.reserve(run_id)
    accounting = cost_ledger.record_usage(run_id, usage)
    _reconcile_cost_accounting(
        usage, accounting, cost_policy, expected_reservation_cny=reservation,
    )
    assert usage["metering_errors"] == []
    assert accounting["reconciliation_matches"] is True

    trend_envelope = output["tool_calls"][0]["response"]
    trend = trend_envelope["result"]["trend_result"]
    assert trend_envelope["result"]["status"] == "ok"
    assert trend["start_date"] == "2024-05-03" and trend["end_date"] == "2024-05-04"
    assert [float(point["value"]) for point in trend["points"]] == [10.0, 6.0]
    contribution_envelope = output["tool_calls"][1]["response"]
    contribution = contribution_envelope["result"]["contribution_result"]
    assert contribution_envelope["result"]["status"] == "ok"
    current_values = {item["dimension_value"]: float(item["value"]) for item in contribution["current_items"]}
    baseline_values = {item["dimension_value"]: float(item["value"]) for item in contribution["baseline_items"]}
    items = {key: current_values[key] - baseline_values[key] for key in current_values}
    assert items == {"A": -4.0, "B": 0.0}
    assert float(contribution["total_delta"]) == -4.0
    # The actual attribution result must reconcile to the independently hand-
    # checked order_dimension oracle: 10 -> 6, with category A -4 and B 0.
    assert contribution["baseline_date"] == "2024-05-03"
    assert contribution["current_date"] == "2024-05-04"
    assert output["final"]["claims"][0]["evidence_ids"] == [
        f"{run_id}:evidence:1", f"{run_id}:evidence:2",
    ]
    assert len(sql_statements) > 0


def test_contribution_tool_failure_is_bound_and_stops_before_next_model_call(
    monkeypatch, tmp_path: Path, isolated_trend_db,
):
    from app.llm import client as llm
    import app.core.config as core_config
    from app.tenancy import TenantContext
    from app.llm.call_config import LLMCallConfig
    from app.resilience import ExternalServiceError
    from evaluation.provider_candidate_batch import ExperimentConfig, RunLedger, bind_isolated_experiment_runtime
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop
    from scripts.run_resume_dev_pair import _stop_reason_after_run
    import app.mcp.tools as tools

    engine, _sql_statements, _sql_events, _cases = isolated_trend_db
    from test_non_gmv_diagnostic_workflow import seed_facts
    seed_facts(engine, "order_dimension")
    run_id = f"resume-b-tool-failure-{uuid4()}"
    fake = _FakeCompletions([
        {"action": "TOOL_CALL", "tool": "query_trend", "arguments": {
            "metric_id": "order_count", "start_date": "2024-05-03", "end_date": "2024-05-04",
        }},
        {"action": "TOOL_CALL", "tool": "metric_contribution", "arguments": {
            "metric_id": "order_count", "baseline_date": "2024-05-03", "current_date": "2024-05-04",
            "dimension": "product_category",
        }},
        {"action": "FINAL", "answer": "must not be generated after tool failure"},
    ])

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=fake)
        def close(self):
            return None

    def fail_repository(*_args, **_kwargs):
        raise ExternalServiceError("postgresql", "external_call_failed", "synthetic safe test detail")

    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: FakeClient(**kwargs))
    config = ExperimentConfig(max_latency_ms=240_000)
    safe_settings = core_config.get_settings().model_copy(update={
        "deepseek_api_key": "offline-placeholder", "llm_providers_json": "[]",
        "llm_timeout_seconds": config.model_timeout_seconds,
    })
    monkeypatch.setattr(core_config, "get_settings", lambda: safe_settings)
    monkeypatch.setattr(tools, "repository_compare_breakdown_by_dimension", fail_repository)
    ledger = RunLedger(tmp_path / "failure_events.jsonl")
    context = TenantContext(
        tenant_id="platform", user_id="resume-dev-pair-order-dimension", role="admin", scope_type="platform",
    )
    with bind_isolated_experiment_runtime(
        engine=engine, trace_dir=tmp_path / "failure_traces", ledger=ledger,
        config=config,
    ):
        call_config = LLMCallConfig.provider_ab_experiment(
            safe_settings, provider_id="primary", max_completion_tokens=4096,
        )
        output = run_bare_tool_loop(
            QUESTION, tenant_context=context, run_id=run_id, llm_call_config=call_config,
        )

    assert len(fake.requests) == 2
    assert [row["tool"] for row in output["tool_calls"]] == ["query_trend", "metric_contribution"]
    failed_call = output["tool_calls"][1]
    assert failed_call["call_status"] == "failed"
    assert failed_call["request_id"].startswith(f"{run_id}:tool:2:")
    assert failed_call["arguments"] == {
        "metric_id": "order_count", "baseline_date": "2024-05-03",
        "current_date": "2024-05-04", "dimension": "product_category",
    }
    assert failed_call["service"] == "postgresql"
    # The public query gateway exposes its normalized, sanitized failure
    # contract; the bare runner must not reconstruct an inner SDK/DB exception.
    assert failed_call["failure_category"] == "database_error"
    assert failed_call["service_error_code"] == "database_execution_failed"
    assert failed_call["resolved_tool_name"] == "compare_breakdown_by_dimension"
    assert failed_call["resolved_plan"]["metric_name"] == "order_count"
    assert failed_call["resolved_plan"]["baseline_date"] == "2024-05-03"
    assert failed_call["resolved_plan"]["current_date"] == "2024-05-04"
    assert "synthetic safe test detail" not in json.dumps(output["errors"])
    assert len(output["evidence_bundles"]) == 1
    assert _stop_reason_after_run(
        output, {"status": "complete", "cost_record_matches": True}, sdk_failed_sends=0,
    ) == "tool_infrastructure_error"


def test_unadapted_model_final_is_preserved_separately_without_adapter():
    from app.tenancy import TenantContext
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop

    authored = {
        "action": "FINAL",
        "answer": "平台订单量为 6 单。",
        "claims": [{"text": "订单量为 6 单", "evidence_ids": []}],
        "presentation": {"metric_id": "order_count", "value": 6, "unit": "orders"},
    }
    context = TenantContext(
        tenant_id="platform", user_id="resume-b-snapshot-test", role="admin", scope_type="platform",
    )
    output = run_bare_tool_loop(
        "查询平台订单量。",
        tenant_context=context,
        run_id=f"resume-b-snapshot-{uuid4()}",
        model_fn=lambda _system, _user: authored,
    )

    assert output["model_authored_final"] == authored
    assert output["final_display"] == authored
    assert output["model_authored_final"] is not output["final"]
    assert output["final_output_adaptation"] == {
        "adapter_id": "registered_display_adapter_v1",
        "source": "model_final",
        "applied": False,
        "operations": [],
    }
    assert authored["presentation"]["value"] == 6
    assert output["model_evidence_reference_capture"] == {
        "schema_version": "bare-model-evidence-reference-capture-v1",
        "source": "model_authored_final",
        "model_final_present": True,
        "claims_field_present": True,
        "claims": [{
            "claim_index": 0,
            "claim_is_mapping": True,
            "evidence_ids_field_present": True,
            "evidence_ids": [],
            "claim_text_sha256": "9ceb3b91b1c30798e15e4bc9aed08cc7697741606882c1acb89b97f2a0e826aa",
        }],
        "presentation_field_present": True,
        "presentation_evidence_ids": [],
    }


def test_b_period_month_route_uses_registered_ratio_facts_tool_and_exact_ranges():
    from app.tenancy import TenantContext
    from evaluation.resume_bare_tool_loop import _authorized_plan

    question = "请比较平台运费率从2017年3月到2017年4月的变化，按 month_position 展示统计贡献。"
    args = {
        "metric_id": "freight_rate", "dimension": "month_position",
        "baseline_start": "2017-03-01", "baseline_end": "2017-03-31",
        "current_start": "2017-04-01", "current_end": "2017-04-30",
        "granularity": "month",
    }
    context = TenantContext(tenant_id="platform", user_id="bg-preflight", role="admin", scope_type="platform")
    plan = _authorized_plan(question=question, tool="period_metric_contribution", arguments=args,
                            context=context, run_id="offline-dc07", sequence=1)
    assert plan.query_type == "metric_contribution"
    assert plan.metric_name == "freight_rate"
    assert plan.dimension == "month_position"
    assert plan.baseline_start == "2017-03-01" and plan.baseline_end == "2017-03-31"
    assert plan.current_start == "2017-04-01" and plan.current_end == "2017-04-30"
    assert plan.tool_name == "compare_freight_rate_ratio_facts_by_dimension_period"


def test_b_period_month_route_rejects_model_supplied_ranges_not_in_question():
    from app.tenancy import TenantContext
    from evaluation.resume_bare_tool_loop import _authorized_plan

    question = "请比较平台运费率从2017年3月到2017年4月的变化，按 month_position 展示统计贡献。"
    args = {
        "metric_id": "freight_rate", "dimension": "month_position",
        "baseline_start": "2017-03-01", "baseline_end": "2017-03-31",
        "current_start": "2017-04-02", "current_end": "2017-04-30",
        "granularity": "month",
    }
    context = TenantContext(tenant_id="platform", user_id="bg-preflight", role="admin", scope_type="platform")
    with pytest.raises(ValueError, match="complete months named in the question"):
        _authorized_plan(question=question, tool="period_metric_contribution", arguments=args,
                         context=context, run_id="offline-dc07-invalid", sequence=1)


def test_b_unsupported_registered_dimension_is_not_mislabelled_as_auth_denial():
    from app.tenancy import TenantContext
    from evaluation.resume_bare_tool_loop import RegisteredCapabilityRefusal, _authorized_plan

    question = "平台订单量按 payment_type 解释变化，期间：{'baseline': {'start_date': '2017-05-01', 'end_date': '2017-05-31'}, 'current': {'start_date': '2017-06-01', 'end_date': '2017-06-30'}}"
    args = {"metric_id": "order_count", "dimension": "payment_type",
            "baseline_start": "2017-05-01", "baseline_end": "2017-05-31",
            "current_start": "2017-06-01", "current_end": "2017-06-30", "granularity": "month"}
    context = TenantContext(tenant_id="platform", user_id="bg-preflight", role="admin", scope_type="platform")
    with pytest.raises(RegisteredCapabilityRefusal) as exc_info:
        _authorized_plan(question=question, tool="period_metric_contribution", arguments=args,
                         context=context, run_id="offline-li03", sequence=1)
    assert exc_info.value.limitation_code == "CAPABILITY_GAP"


@pytest.mark.parametrize(
    "metric_id,question,expected_method,expected_baseline,expected_current",
    [
        (
            "gmv",
            "请对平台GMV在2024-05-03到2024-05-04的变化做注册数值分解，只解释数值，不推断业务原因。",
            "product_shapley_v1", 100.0, 60.0,
        ),
        (
            "avg_order_value",
            "请对平台客单价在2024-05-03到2024-05-04的变化做注册数值分解，只解释数值，不推断业务原因。",
            "symmetric_ratio_v1", 10.0, 10.0,
        ),
    ],
)
def test_b_registered_numeric_decomposition_uses_authorized_registry_trends_and_shared_formula(
    monkeypatch, tmp_path: Path, isolated_trend_db,
    metric_id, question, expected_method, expected_baseline, expected_current,
):
    from app.llm import client as llm
    import app.core.config as core_config
    from app.tenancy import TenantContext
    from app.llm.call_config import LLMCallConfig
    from evaluation.provider_candidate_batch import ExperimentConfig, RunLedger, bind_isolated_experiment_runtime
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop
    from test_non_gmv_diagnostic_workflow import seed_facts

    engine, sql_statements, _sql_events, _cases = isolated_trend_db
    seed_facts(engine, "order_dimension")
    run_id = f"resume-b-decomposition-{uuid4()}"
    actions = [
        {"action": "TOOL_CALL", "tool": "registered_numeric_decomposition", "arguments": {
            "metric_id": metric_id, "baseline_date": "2024-05-03", "current_date": "2024-05-04",
        }},
        {"action": "FINAL", "answer": "这是按注册公式得到的数值解释，不代表业务因果。",
         "claims": [{"text": "注册数值分解结果", "evidence_ids": [f"{run_id}:evidence:1"]}],
         "presentation": {"metric_id": metric_id, "baseline_date": "2024-05-03",
                          "current_date": "2024-05-04"}},
    ]
    fake = _FakeCompletions(actions)

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=fake)
        def close(self):
            return None

    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: FakeClient(**kwargs))
    config = ExperimentConfig(max_latency_ms=240_000)
    safe_settings = core_config.get_settings().model_copy(update={
        "deepseek_api_key": "offline-placeholder", "llm_providers_json": "[]",
        "llm_timeout_seconds": config.model_timeout_seconds,
    })
    monkeypatch.setattr(core_config, "get_settings", lambda: safe_settings)
    ledger = RunLedger(tmp_path / f"{metric_id}_events.jsonl")
    context = TenantContext(
        tenant_id="platform", user_id="resume-dev-decomposition", role="admin", scope_type="platform",
    )
    with bind_isolated_experiment_runtime(
        engine=engine, trace_dir=tmp_path / f"{metric_id}_traces", ledger=ledger,
        config=config,
    ):
        call_config = LLMCallConfig.provider_ab_experiment(
            safe_settings, provider_id="primary", max_completion_tokens=4096,
        )
        output = run_bare_tool_loop(
            question, tenant_context=context, run_id=run_id, llm_call_config=call_config,
        )

    assert output["errors"] == []
    assert len(fake.requests) == 2
    assert len(output["tool_calls"]) == 1
    call = output["tool_calls"][0]
    assert call["tool"] == "registered_numeric_decomposition"
    assert call["call_status"] == "succeeded"
    assert call["request_id"] == call["response"]["request_id"]
    result = call["response"]["result"]
    assert result["status"] == "ok"
    assert result["operation_intent"] == "registered_numeric_decomposition"
    assert result["target_metric_id"] == metric_id
    assert result["baseline_date"] == "2024-05-03"
    assert result["current_date"] == "2024-05-04"
    decomposition = result["decomposition_result"]
    assert decomposition["method_id"] == expected_method
    assert decomposition["baseline_value"] == pytest.approx(expected_baseline)
    assert decomposition["current_value"] == pytest.approx(expected_current)
    assert decomposition["causal_claim"] is False
    assert decomposition["evidence_ids"] == result["source_evidence_ids"]
    source_queries = call["response"]["source_queries"]
    assert len(source_queries) >= 2
    assert all(item["request_id"] and item["evidence_id"] for item in source_queries)
    assert all(item["envelope"]["request_id"] == item["request_id"] for item in source_queries)
    provenance = {
        item["metric_id"]: item["envelope"]["result"]["parsed"]["authorized_metric"]["provenance_type"]
        for item in source_queries
    }
    assert provenance[metric_id] == "exact"
    assert all(value == "registered_relation" for key, value in provenance.items() if key != metric_id)
    assert len(sql_statements) >= len(source_queries)
    assert output["final"]["action"] == "FINAL"


def test_b_single_date_incomplete_trend_stops_once_with_traceable_data_limit(
    monkeypatch, tmp_path: Path, isolated_trend_db,
):
    from app.llm import client as llm
    import app.core.config as core_config
    from app.tenancy import TenantContext
    from app.llm.call_config import LLMCallConfig
    from evaluation.provider_candidate_batch import ExperimentConfig, RunLedger, bind_isolated_experiment_runtime
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop
    from test_non_gmv_diagnostic_workflow import seed_facts

    engine, _sql_statements, _sql_events, _cases = isolated_trend_db
    seed_facts(engine, "order_dimension")
    run_id = f"resume-b-single-date-gap-{uuid4()}"
    question = "2018-04-25 的 GMV 情况怎么样，有异常吗？"
    fake = _FakeCompletions([{
        "action": "TOOL_CALL", "tool": "query_trend", "arguments": {
            "metric_id": "gmv", "start_date": "2018-04-25", "end_date": "2018-04-25",
        },
    }])

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=fake)
        def close(self):
            return None

    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: FakeClient(**kwargs))
    config = ExperimentConfig(max_latency_ms=240_000)
    safe_settings = core_config.get_settings().model_copy(update={
        "deepseek_api_key": "offline-placeholder", "llm_providers_json": "[]",
        "llm_timeout_seconds": config.model_timeout_seconds,
    })
    monkeypatch.setattr(core_config, "get_settings", lambda: safe_settings)
    context = TenantContext(tenant_id="platform", user_id="bg-s01", role="admin", scope_type="platform")
    with bind_isolated_experiment_runtime(
        engine=engine, trace_dir=tmp_path / "s01_traces",
        ledger=RunLedger(tmp_path / "s01_events.jsonl"), config=config,
    ):
        output = run_bare_tool_loop(
            question, tenant_context=context, run_id=run_id,
            llm_call_config=LLMCallConfig.provider_ab_experiment(
                safe_settings, provider_id="primary", max_completion_tokens=4096,
            ),
        )
    assert len(fake.requests) == 1
    assert output["errors"] == []
    assert output["terminal_code"] == "DATA_INSUFFICIENT"
    assert output["presentation"]["status"] == "partial"
    assert output["presentation"]["current_date"] == "2018-04-25"
    assert len(output["evidence_bundles"]) == 1
    assert output["final"]["claims"][0]["evidence_ids"] == [output["evidence_bundles"][0]["evidence_id"]]
    parsed = output["evidence_bundles"][0]["content"]["result"]["parsed"]
    assert parsed["start_date"] == parsed["end_date"] == "2018-04-25"


def test_b_li03_registered_capability_refusal_precedes_sql_and_never_substitutes_dimension(
    monkeypatch, tmp_path: Path, isolated_trend_db,
):
    from app.llm import client as llm
    import app.core.config as core_config
    from app.tenancy import TenantContext
    from app.llm.call_config import LLMCallConfig
    from evaluation.provider_candidate_batch import ExperimentConfig, RunLedger, bind_isolated_experiment_runtime
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop

    engine, sql_statements, _sql_events, _cases = isolated_trend_db
    run_id = f"resume-b-li03-{uuid4()}"
    sql_statements.clear()
    question = "平台订单量拆分不支持 payment_type；请说明限制。 期间：{'baseline': {'start_date': '2017-05-01', 'end_date': '2017-05-31'}, 'current': {'start_date': '2017-06-01', 'end_date': '2017-06-30'}}。"
    fake = _FakeCompletions([{
        "action": "TOOL_CALL", "tool": "period_metric_contribution", "arguments": {
            "metric_id": "order_count", "dimension": "payment_type",
            "baseline_start": "2017-05-01", "baseline_end": "2017-05-31",
            "current_start": "2017-06-01", "current_end": "2017-06-30", "granularity": "month",
        },
    }])

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=fake)
        def close(self):
            return None

    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: FakeClient(**kwargs))
    config = ExperimentConfig(max_latency_ms=240_000)
    safe_settings = core_config.get_settings().model_copy(update={
        "deepseek_api_key": "offline-placeholder", "llm_providers_json": "[]",
        "llm_timeout_seconds": config.model_timeout_seconds,
    })
    monkeypatch.setattr(core_config, "get_settings", lambda: safe_settings)
    context = TenantContext(tenant_id="platform", user_id="bg-li03", role="admin", scope_type="platform")
    with bind_isolated_experiment_runtime(
        engine=engine, trace_dir=tmp_path / "li03_traces",
        ledger=RunLedger(tmp_path / "li03_events.jsonl"), config=config,
    ):
        output = run_bare_tool_loop(
            question, tenant_context=context, run_id=run_id,
            llm_call_config=LLMCallConfig.provider_ab_experiment(
                safe_settings, provider_id="primary", max_completion_tokens=4096,
            ),
        )
    assert len(fake.requests) == 1
    assert output["terminal_code"] == "CAPABILITY_GAP"
    assert output["presentation"]["status"] == "unsupported"
    assert output["tool_calls"][0]["call_status"] == "refused"
    assert output["tool_calls"][0]["limitation_code"] == "CAPABILITY_GAP"
    assert output["decision_records"][0]["dimension"] == "payment_type"
    assert output["decision_records"][0]["sql_executed"] is False
    assert sql_statements == []


def test_b_dc07_natural_month_contribution_uses_actual_registered_public_route(
    monkeypatch, tmp_path: Path, isolated_trend_db,
):
    from app.llm import client as llm
    import app.core.config as core_config
    from app.tenancy import TenantContext
    from app.llm.call_config import LLMCallConfig
    from evaluation.provider_candidate_batch import ExperimentConfig, RunLedger, bind_isolated_experiment_runtime
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop
    from sqlalchemy import text

    engine, sql_statements, _sql_events, _cases = isolated_trend_db
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM analytics.fact_orders"))
        rows = [
            ("m3-start", "2017-03-02", 100, 10), ("m3-mid", "2017-03-15", 100, 20),
            ("m3-end", "2017-03-25", 100, 30), ("m4-start", "2017-04-02", 100, 20),
            ("m4-mid", "2017-04-15", 100, 20), ("m4-end", "2017-04-25", 100, 20),
        ]
        conn.execute(text("""INSERT INTO analytics.fact_orders
            (order_id,customer_unique_id,item_count,item_gross_value,freight_value,purchase_date,order_status)
            VALUES (:id,:customer,1,:gross,:freight,:day,'delivered')"""), [
                {"id": row[0], "customer": row[0], "day": row[1], "gross": row[2], "freight": row[3]}
                for row in rows
            ])
    sql_statements.clear()
    run_id = f"resume-b-dc07-{uuid4()}"
    question = "请比较平台运费率在2017年3月与2017年4月的变化，按 month_position 展示统计贡献；只陈述数值贡献，不推断业务因果。"
    fake = _FakeCompletions([
        {"action": "TOOL_CALL", "tool": "period_metric_contribution", "arguments": {
            "metric_id": "freight_rate", "dimension": "month_position",
            "baseline_start": "2017-03-01", "baseline_end": "2017-03-31",
            "current_start": "2017-04-01", "current_end": "2017-04-30", "granularity": "month",
        }},
        {"action": "FINAL", "answer": "两个月的注册分组结果提供统计贡献，不表示业务因果。",
         "claims": [{"text": "月度统计贡献", "evidence_ids": [f"{run_id}:evidence:1"]}]},
    ])

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=fake)
        def close(self):
            return None

    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: FakeClient(**kwargs))
    config = ExperimentConfig(max_latency_ms=240_000)
    safe_settings = core_config.get_settings().model_copy(update={
        "deepseek_api_key": "offline-placeholder", "llm_providers_json": "[]",
        "llm_timeout_seconds": config.model_timeout_seconds,
    })
    monkeypatch.setattr(core_config, "get_settings", lambda: safe_settings)
    context = TenantContext(tenant_id="platform", user_id="bg-dc07", role="admin", scope_type="platform")
    with bind_isolated_experiment_runtime(
        engine=engine, trace_dir=tmp_path / "dc07_traces",
        ledger=RunLedger(tmp_path / "dc07_events.jsonl"), config=config,
    ):
        output = run_bare_tool_loop(
            question, tenant_context=context, run_id=run_id,
            llm_call_config=LLMCallConfig.provider_ab_experiment(
                safe_settings, provider_id="primary", max_completion_tokens=4096,
            ),
        )

    assert output["errors"] == []
    assert len(fake.requests) == 2
    assert len(sql_statements) > 0
    call = output["tool_calls"][0]
    assert call["call_status"] == "succeeded"
    assert call["tool_name"] == "compare_freight_rate_ratio_facts_by_dimension_period"
    parsed = call["response"]["result"]["parsed"]
    assert parsed["baseline_start"] == "2017-03-01" and parsed["baseline_end"] == "2017-03-31"
    assert parsed["current_start"] == "2017-04-01" and parsed["current_end"] == "2017-04-30"
    assert parsed["dimension"] == "month_position"
    contribution = call["response"]["result"]["contribution_result"]
    assert contribution["status"] == "ok"
    assert contribution["baseline_period"] == {"start_date": "2017-03-01", "end_date": "2017-03-31"}
    assert contribution["current_period"] == {"start_date": "2017-04-01", "end_date": "2017-04-30"}
    assert contribution["baseline_numerator_total"] == pytest.approx(60.0)
    assert contribution["baseline_denominator_total"] == pytest.approx(360.0)
    assert contribution["current_numerator_total"] == pytest.approx(60.0)
    assert contribution["current_denominator_total"] == pytest.approx(360.0)
    baseline_group_stats = sorted(contribution["baseline_groups"].values(), key=lambda item: item["numerator"])
    assert baseline_group_stats == [
        {"numerator": 10.0, "denominator": 110.0},
        {"numerator": 20.0, "denominator": 120.0},
        {"numerator": 30.0, "denominator": 130.0},
    ]

