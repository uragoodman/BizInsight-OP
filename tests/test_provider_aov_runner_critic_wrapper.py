"""Offline end-to-end guard for the AOV provider-runner Critic wrappers.

The normal workflow, query gateway, Critic node, review functions, and runner
wrappers remain real. Only the OpenAI-compatible SDK response is replaced.
"""

import importlib.util
import inspect
import json
import os
import socket
import sys
import threading
import contextvars
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from test_non_gmv_diagnostic_workflow import seed_facts
from test_platform_metric_trend_acceptance import isolated_trend_db  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(os.environ.get(
    "AOV_RUNNER_SCRIPT_PATH",
    str(ROOT / "docs/testing/provider_aov_ratio_full_off_nonthinking_20261012_01/run_aov_full_off.py"),
))


def _load_runner():
    spec = importlib.util.spec_from_file_location("aov_full_off_runner_preflight", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_wrappers_preserve_request_options_and_execute_full_critic_offline(
    monkeypatch, isolated_trend_db, tmp_path,
):
    runner = _load_runner()
    from app.agents import critic_agent
    from app.core.config import get_settings
    from app.core import time_expression
    from app.data.repositories import daily_metrics
    from app.graph import workflow
    import app.graph.nodes.critic as critic_node
    from app.graph.critic_review_mode import CriticLLMReviewMode
    from app.graph.critic_llm_thinking_mode import CriticLLMThinkingMode
    from app.llm import client as llm
    from app.llm.call_config import LLMCallConfig
    from app.mcp import client as mcp, tools
    from app.tenancy import TenantContext
    from evaluation.provider_candidate_batch import (
        CostPolicy, EstimatedCostLedger, ExperimentConfig, RunLedger,
        run_workflow_unit, bind_isolated_experiment_runtime,
    )
    from evaluation.provider_candidate_protocol import PROVIDER_CANDIDATE_SCORER_VERSION

    # Explicitly block non-loopback sockets. The SDK is replaced below, while
    # the isolated PostgreSQL fixture remains available on localhost.
    real_connect = socket.socket.connect

    def block_external_connect(sock, address):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError(f"offline preflight blocked external socket to {host}")
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", block_external_connect)
    try:
        socket.create_connection(("203.0.113.1", 443), timeout=0.01)
    except AssertionError as exc:
        assert "blocked external socket" in str(exc)
    else:  # pragma: no cover - the guard must stop this before a network call
        raise AssertionError("external-network guard did not block the probe")

    engine, sql_statements, _, _ = isolated_trend_db
    settings = get_settings().model_copy(update={
        "deepseek_api_key": "offline-sdk-stub-key",
        "deepseek_base_url": "https://provider.invalid/compatible-mode/v1",
        "llm_providers_json": "[]",
        "llm_timeout_seconds": 60.0,
        "llm_resilience_attempts": 1,
        "max_tokens": 60_000,
        "max_graph_steps": 48,
        "max_tool_calls": 8,
        "max_latency_ms": 240_000,
        "max_revisions": 1,
    })
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "_persist_chat_turn", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm, "record_usage", lambda **kwargs: None)
    monkeypatch.setattr(daily_metrics, "engine", engine)
    monkeypatch.setattr(time_expression, "engine", engine)

    trial = runner.TRIAL
    call_config = LLMCallConfig.provider_ab_experiment(
        settings, provider_id="primary", max_completion_tokens=trial.max_completion_tokens,
    )
    active = {"arm": None, "critic_review_name": None, "workflow_kind": "aov", "overlap_barrier": None}
    sdk_requests = []
    review_sdk_requests = []
    constructions = []
    sdk_attempt_observations = []

    real_begin_sdk_attempt = llm._begin_sdk_attempt

    def observe_sdk_attempt(call, provider, model, state=None):
        observation = {
            "arm": active["arm"],
            "review_name": active.get("critic_review_name"),
            "call_id": call.get("call_id"),
            "trace_id": call.get("trace_id"),
            "node": call.get("node"),
            "operation": call.get("operation"),
            "request_options": dict(call.get("request_options") or {}),
        }
        attempt = real_begin_sdk_attempt(call, provider, model, state)
        observation["attempt_id"] = attempt.get("attempt_id")
        sdk_attempt_observations.append(observation)
        return attempt

    monkeypatch.setattr(llm, "_begin_sdk_attempt", observe_sdk_attempt)
    # Preserve request-local ContextVars through the resilience worker. The
    # production resilience policy (including attempts and timeout) stays live.
    real_execute_with_resilience = llm.execute_with_resilience

    def execute_with_request_context(fn, service, **kwargs):
        context = contextvars.copy_context()
        return real_execute_with_resilience(lambda: context.run(fn), service, **kwargs)

    monkeypatch.setattr(llm, "execute_with_resilience", execute_with_request_context)

    class FakeCompletions:
        def create(self, **kwargs):
            review_name = active.get("critic_review_name")
            current_trace_id = (llm.active_state() or {}).get("trace_id")
            attempt = next((item for item in reversed(sdk_attempt_observations)
                            if item["arm"] == active["arm"]
                            and item["review_name"] == review_name
                            and item["trace_id"] == current_trace_id), {})
            barrier = active.get("overlap_barrier")
            if barrier is not None and attempt.get("trace_id") not in active.setdefault("overlapped_trace_ids", set()):
                active["overlapped_trace_ids"].add(attempt.get("trace_id"))
                barrier.wait(timeout=15)
            operation = attempt.get("operation")
            system_text = str((kwargs.get("messages") or [{}])[0].get("content") or "")
            request = {
                "arm": active["arm"],
                "operation": operation,
                "review_name": review_name,
                "call_id": attempt.get("call_id"),
                "attempt_id": attempt.get("attempt_id"),
                "node": attempt.get("node"),
                "trace_id": attempt.get("trace_id"),
                "model": kwargs.get("model"),
                "max_completion_tokens": kwargs.get("max_completion_tokens"),
                "max_tokens_present": "max_tokens" in kwargs,
                "extra_body": kwargs.get("extra_body"),
                "request_options": dict(attempt.get("request_options") or {}),
            }
            assert request["model"] == trial.expected_model
            assert request["max_completion_tokens"] == trial.max_completion_tokens
            assert request["max_tokens_present"] is False
            if attempt.get("node") == "critic":
                assert active["arm"] == "full"
                assert request["extra_body"] == {"enable_thinking": False}
                review_sdk_requests.append(request)
                content = json.dumps({"verdict": "passed", "issues": []})
            elif operation == "classify" or all(token in system_text for token in (
                "metric_definition", "metric_query", "anomaly_analysis", "report_generation",
            )):
                content = "metric_query" if active.get("workflow_kind") == "metric_query" else "anomaly_analysis"
            else:
                # Keep non-Critic planning/answer stages on the existing
                # deterministic fallback path; only the external response is
                # stubbed, not the workflow node or graph routing.
                content = ""

            sdk_requests.append(request)
            return SimpleNamespace(
                model=trial.expected_model,
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3, total_tokens=14,
                                      completion_tokens_details=SimpleNamespace(reasoning_tokens=0, text_tokens=3)),
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason="stop",
                )],
            )

    fake_completions = FakeCompletions()

    class FakeChat:
        def __init__(self):
            self.completions = fake_completions

    class FakeSDKClient:
        def __init__(self):
            self.chat = FakeChat()

    per_run_send_count = {}

    def reserve_call(kwargs):
        arm = active["arm"]
        run_trace_id = (llm.active_state() or {}).get("trace_id")
        if per_run_send_count.get(run_trace_id, 0) >= trial.max_sdk_calls_per_workflow:
            raise runner.PreflightBudgetStop("SDK call cap reached before local fake SDK execution.")
        if kwargs.get("model") != trial.expected_model:
            raise AssertionError("wrong model reached offline SDK boundary")
        if kwargs.get("max_completion_tokens") != trial.max_completion_tokens or "max_tokens" in kwargs:
            raise AssertionError("wrong output token parameter reached offline SDK boundary")
        review_name = active.get("critic_review_name")
        if review_name is not None and (kwargs.get("extra_body") or {}).get("enable_thinking") is not False:
            raise AssertionError("Critic request lost explicit non-thinking setting")
        per_run_send_count[run_trace_id] = per_run_send_count.get(run_trace_id, 0) + 1
        return {"request_index": per_run_send_count[run_trace_id], "arm": arm, "review_name": review_name}

    def record_response(exchange, response=None):
        exchange["result"] = "response_received" if response is not None else "provider_exception"

    def fake_openai(**kwargs):
        assert kwargs.get("max_retries") == 0
        constructions.append({"max_retries": kwargs["max_retries"], "base_url": kwargs["base_url"]})
        return runner._OpenAIClientProxy(FakeSDKClient(), reserve_call, record_response)

    monkeypatch.setattr(llm, "OpenAI", fake_openai)
    restore_call_limit = runner.install_logical_call_limit(
        llm, trial.max_sdk_calls_per_workflow, stop_exception=runner.PreflightBudgetStop,
    )
    originals = runner.install_critic_review_labels(critic_node, active)
    monkeypatch.setattr(critic_agent, "llm_extract", llm.llm_extract)

    async def local_transport(tool_name, arguments):
        tool = getattr(tools, tool_name, None)
        assert tool is not None and not tool_name.startswith("_")
        return tool(**arguments)

    monkeypatch.setattr(mcp, "_call_local_mcp_tool", local_transport)
    traces = {}
    candidate_runner_trace_dir = Path(os.environ.get(
        "AOV_RUNNER_PREFLIGHT_ARTIFACT_DIR", str(tmp_path),
    )) / "candidate_executor_smoke"

    def capture_trace(trace_id, state):
        traces[trace_id] = dict(state)
        candidate_runner_trace_dir.mkdir(parents=True, exist_ok=True)
        (candidate_runner_trace_dir / f"{trace_id}.json").write_text(json.dumps(runner._json_safe({
            "trace_id": trace_id,
            "state_snapshot": state,
            "workflow_result": {
                "answer": state.get("answer", ""),
                "presentation": state.get("analysis_presentation"),
            },
        }), ensure_ascii=False, indent=2), encoding="utf-8")

    monkeypatch.setattr(workflow, "write_trace", capture_trace)
    tenant = TenantContext(
        tenant_id="platform", user_id="aov-runner-offline-preflight", role="admin", scope_type="platform",
    )
    question = "为什么 2024-05-04 的客单价上升了？请检查证据，区分统计分解与业务原因。"
    results = {}
    try:
        for arm, mode in (
            ("full", CriticLLMReviewMode.FULL),
            ("critic_llm_review_off", CriticLLMReviewMode.CRITIC_LLM_REVIEW_OFF),
        ):
            seed_facts(engine, "aov_ratio")
            active["arm"] = arm
            sql_start = len(sql_statements)
            response = workflow.run_chat_workflow(
                question,
                tenant_context=tenant,
                memory_enabled=False,
                trace_id=f"offline-aov-critic-wrapper-{arm}-{uuid4().hex[:8]}",
                _critic_llm_review_mode=mode,
                _critic_llm_thinking_mode=CriticLLMThinkingMode.NON_THINKING_EXPERIMENT,
                _llm_call_config=call_config,
            )
            state = traces[next(key for key in traces if key.startswith(f"offline-aov-critic-wrapper-{arm}-"))]
            results[arm] = {
                "response": response,
                "state": state,
                "sql_count": len(sql_statements) - sql_start,
                "sdk_requests": [item for item in sdk_requests if item["arm"] == arm],
                "review_requests": [item for item in review_sdk_requests if item["arm"] == arm],
            }

    finally:
        restore_call_limit()
        critic_node.llm_review_groundedness = originals["groundedness"]
        critic_node.llm_review_evidence_completeness = originals["completeness"]
        critic_node.llm_review_logic_consistency = originals["consistency"]

    full = results["full"]
    full_state = full["state"]
    full_reviews = (full_state.get("critic_result") or {}).get("llm_review") or {}
    assert full["response"].get("error") is False
    assert any(item.get("node") == "critic" for item in full_state.get("node_history", []))
    assert full_state.get("critic_verdict") == "approved"
    assert full_reviews.get("status") == "invoked"
    assert [full_reviews[name].get("verdict") for name in ("groundedness", "completeness", "consistency")] == [
        "passed", "passed", "passed",
    ], f"reviews={[(name, (full_reviews.get(name) or {}).get('verdict')) for name in ('groundedness', 'completeness', 'consistency')]}; " \
        f"critic_sdk_requests={len(full['review_requests'])}; " \
        f"attempts={[(item.get('operation'), item.get('node')) for item in sdk_attempt_observations if item.get('arm') == 'full']}; " \
        f"calls={len(full_state.get('llm_calls', []))}"
    assert len(full["review_requests"]) == 3
    assert all(item["node"] == "critic" and item["operation"] == "extract"
               for item in full["review_requests"])
    review_attempts = [item for item in sdk_attempt_observations
                       if item["arm"] == "full" and item["node"] == "critic"]
    assert len(review_attempts) == 3
    assert {item["node"] for item in review_attempts} == {"critic"}
    assert {item["call_id"] for item in review_attempts} == {
        item["call_id"] for item in full["state"].get("llm_calls", []) if item.get("node") == "critic"
    }
    assert len([item for item in full["state"].get("llm_calls", []) if item.get("node") == "critic"]) == 3
    assert all(item["call_id"] and item["attempt_id"] and item["node"] == "critic"
               for item in full["review_requests"])
    assert all(item["extra_body"] == {"enable_thinking": False} for item in full["review_requests"])
    assert full["sql_count"] > 0

    off = results["critic_llm_review_off"]
    off_state = off["state"]
    off_reviews = (off_state.get("critic_result") or {}).get("llm_review") or {}
    assert off["response"].get("error") is False
    assert any(item.get("node") == "critic" for item in off_state.get("node_history", []))
    assert off_reviews.get("status") == "evaluation_ablation"
    assert off["review_requests"] == []
    assert not any(item["arm"] == "critic_llm_review_off" and item["node"] == "critic"
                   for item in sdk_attempt_observations)
    assert not any(item.get("node") == "critic" for item in off_state.get("llm_calls", []))
    assert off["sql_count"] > 0

    # Calls created in the same normal compiled workflow must retain the true
    # execution node. Verify planning/data work and an LLM-backed answer path.
    for arm_result in (full, off):
        state_calls = arm_result["state"].get("llm_calls", [])
        sdk_count = len(arm_result["sdk_requests"])
        assert len(state_calls) == sdk_count
        assert sum(len(call.get("attempts") or []) for call in state_calls) == sdk_count
        assert all(call.get("trace_id") for call in state_calls)
        assert all(attempt.get("call_id") == call.get("call_id") for call in state_calls
                   for attempt in call.get("attempts", []))
        assert all(attempt.get("usage_status") == "complete" for call in state_calls
                   for attempt in call.get("attempts", []))
        trace_id = state_calls[0].get("trace_id")
        observations_by_id = {item["call_id"]: item for item in sdk_attempt_observations
                              if item["trace_id"] == trace_id}
        assert set(observations_by_id) == {call["call_id"] for call in state_calls}
        for call in state_calls:
            assert observations_by_id[call["call_id"]]["node"] == call["node"]
            assert observations_by_id[call["call_id"]]["attempt_id"] == call["attempts"][0]["attempt_id"]
    assert any(call.get("operation") == "classify" and call.get("node") == "triage"
               for call in full_state.get("llm_calls", []))
    assert any(call.get("operation") == "extract" and call.get("node") == "diagnostic_plan"
               for call in full_state.get("llm_calls", []))
    assert any(call.get("operation") == "extract" and call.get("node") == "diagnostic_decide"
               for call in full_state.get("llm_calls", []))

    # Two concurrent requests pass through run_chat_workflow and the compiled
    # graph. A barrier forces overlapping SDK calls; each trace must retain its
    # own call records and node labels.
    active["workflow_kind"] = "metric_query"
    active["arm"] = "full"
    active["overlap_barrier"] = threading.Barrier(2)
    active["overlapped_trace_ids"] = set()
    concurrent_ids = [f"offline-node-isolation-{uuid4().hex[:8]}" for _ in range(2)]

    def run_metric_query(trace_id):
        return workflow.run_chat_workflow(
            "查询 2024-05-04 的客单价",
            tenant_context=tenant,
            memory_enabled=False,
            trace_id=trace_id,
            _critic_llm_review_mode=CriticLLMReviewMode.CRITIC_LLM_REVIEW_OFF,
            _critic_llm_thinking_mode=CriticLLMThinkingMode.NON_THINKING_EXPERIMENT,
            _llm_call_config=call_config,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent_responses = list(pool.map(run_metric_query, concurrent_ids))
    active["overlap_barrier"] = None
    assert len(active["overlapped_trace_ids"]) == 2
    for trace_id, response in zip(concurrent_ids, concurrent_responses):
        assert response.get("error") is False
        state_calls = traces[trace_id].get("llm_calls", [])
        assert state_calls
        assert all(call.get("trace_id") == trace_id for call in state_calls)
        assert all(call.get("node") in {"resolve_explorer_context", "triage", "data_query", "generate_answer"}
                   for call in state_calls)
        assert any(call.get("operation") == "chat" and call.get("node") == "generate_answer"
                   for call in state_calls)
        assert not any(call.get("node") == "critic" for call in state_calls)
        observed_for_trace = [item for item in sdk_attempt_observations if item["trace_id"] == trace_id]
        assert len(observed_for_trace) == sum(len(call.get("attempts") or []) for call in state_calls)
        for call in state_calls:
            observed = next(item for item in observed_for_trace if item["call_id"] == call["call_id"])
            assert observed["node"] == call["node"]
            assert observed["attempt_id"] == call["attempts"][0]["attempt_id"]

    assert all(count <= trial.max_sdk_calls_per_workflow for count in per_run_send_count.values())
    assert constructions and all(item["max_retries"] == 0 for item in constructions)
    assert inspect.signature(llm._begin_logical_call).parameters["request_options"].kind is inspect.Parameter.KEYWORD_ONLY

    # Exercise the new batch executor's real run_chat_workflow adapter on an
    # existing development case. The candidate expectation is used only by its
    # post-run scorer; the Agent receives only the original question. This is
    # still an SDK-stubbed smoke run, not a candidate experiment observation.
    candidate_rows = json.loads((ROOT / "docs/testing/provider_candidate_protocol_17_20261013/candidate_protocol_17.json").read_text(encoding="utf-8"))["cases"]
    scorer_candidate = dict(candidate_rows[0])
    scorer_candidate["user_question"] = question
    smoke_entry = {"case_id": "dev-aov-ratio", "arm": "full", "replicate": 1,
                   "run_id": f"candidate-executor-dev-{uuid4().hex}"}
    active["arm"] = "full"
    smoke_ledger = RunLedger(candidate_runner_trace_dir / "events.jsonl")
    smoke_cost = EstimatedCostLedger(CostPolicy(), candidate_runner_trace_dir / "cost_reservations.jsonl")
    with bind_isolated_experiment_runtime(
        engine=engine, trace_dir=candidate_runner_trace_dir,
        ledger=smoke_ledger, config=ExperimentConfig(),
    ):
        executor_smoke = run_workflow_unit(
            smoke_entry, scorer_candidate, tenant_context=tenant,
            trace_dir=candidate_runner_trace_dir, ledger=smoke_ledger,
            cost_ledger=smoke_cost, experiment_config=ExperimentConfig(),
        )
    assert executor_smoke["score"]["scorer_version"] == PROVIDER_CANDIDATE_SCORER_VERSION
    assert executor_smoke["measurements"]["sdk_sends"] > 0
    assert executor_smoke["measurements"]["sql_attempts"] > 0
    assert smoke_ledger.latest()[smoke_entry["run_id"]]["event"] == "completed"

    artifact_root = os.environ.get("AOV_RUNNER_PREFLIGHT_ARTIFACT_DIR")
    if artifact_root:
        def summarize(arm_result):
            state = arm_result["state"]
            critic_result = state.get("critic_result") or {}
            llm_review = critic_result.get("llm_review") or {}
            return {
                "critic_entered": any(item.get("node") == "critic" for item in state.get("node_history", [])),
                "critic_verdict": state.get("critic_verdict"),
                "critic_call_count_by_recorded_node": sum(
                    1 for call in state.get("llm_calls", []) if call.get("node") == "critic"
                ),
                "logical_call_count": len(state.get("llm_calls", [])),
                "sdk_attempt_count": sum(len(call.get("attempts") or []) for call in state.get("llm_calls", [])),
                "review_status": llm_review.get("status"),
                "review_verdicts": {
                    name: (llm_review.get(name) or {}).get("verdict")
                    for name in ("groundedness", "completeness", "consistency")
                },
                "optional_review_sdk_requests": arm_result["review_requests"],
                "sdk_request_count": len(arm_result["sdk_requests"]),
                "sql_attempt_count": arm_result["sql_count"],
                "error": arm_result["response"].get("error"),
            }

        artifact = {
            "network_blocked": True,
            "provider_calls": 0,
            "sdk_stub_used": True,
            "normal_entrypoint": "run_chat_workflow",
            "arms": {
                "full": summarize(full),
                "critic_llm_review_off": summarize(off),
            },
            "request_options_seen_at_sdk_attempt": [
                item for item in sdk_attempt_observations
                if item["node"] == "critic"
            ],
            "call_attribution_observations": sdk_attempt_observations,
            "concurrent_metric_query_trace_ids": concurrent_ids,
            "candidate_batch_executor_dev_smoke": {
                "run_id": smoke_entry["run_id"],
                "case_kind": "existing aov_ratio development fixture; no candidate Agent run",
                "entrypoint": "run_chat_workflow",
                "scorer_version": executor_smoke["score"]["scorer_version"],
                "task_acceptance_pass": executor_smoke["score"]["task_acceptance_pass"],
                "sdk_sends": executor_smoke["measurements"]["sdk_sends"],
                "logical_calls": executor_smoke["measurements"]["llm_calls"],
                "sql_attempts": executor_smoke["measurements"]["sql_attempts"],
                "tool_calls": executor_smoke["measurements"]["tool_calls"],
                "elapsed_ms": executor_smoke["measurements"]["elapsed_ms"],
                "trace_file": f"candidate_executor_smoke/{smoke_entry['run_id']}.json",
                "run_record_file": f"candidate_executor_smoke/{smoke_entry['run_id']}.run_record.json",
            },
        }
        output_path = Path(artifact_root) / "integration_trace.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")


def test_runner_logical_call_wrapper_keeps_request_options_and_logical_send_cap(monkeypatch):
    runner = _load_runner()
    from app.llm import client as llm

    original = llm._begin_logical_call
    seen = {}

    def replacement(operation, requested_model, *, request_options=None):
        seen.update(operation=operation, requested_model=requested_model, request_options=request_options)
        return {"call_id": "local-call", "attempts": []}, None

    monkeypatch.setattr(llm, "_begin_logical_call", replacement)
    restore = runner.install_logical_call_limit(llm, 10, stop_exception=runner.PreflightBudgetStop)
    try:
        llm._begin_logical_call("extract", "qwen3.8-flash", request_options={"enable_thinking": False})
        monkeypatch.setattr(llm, "active_state", lambda: {"llm_calls": [{} for _ in range(10)]})
        with pytest.raises(runner.PreflightBudgetStop, match="logical model-call limit"):
            llm._begin_logical_call("extract", "qwen3.8-flash", request_options={"enable_thinking": False})
    finally:
        restore()
    assert seen == {
        "operation": "extract",
        "requested_model": "qwen3.8-flash",
        "request_options": {"enable_thinking": False},
    }
    assert len(seen) == 3
