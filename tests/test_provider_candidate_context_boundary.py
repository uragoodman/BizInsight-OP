from __future__ import annotations

import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.execution_context import activate_state, active_state, reset_state
from app.llm.call_config import LLMCallConfig
from app.resilience import execute_with_resilience
from evaluation.provider_candidate_batch import (
    DEFAULT_EXPERIMENT_CONFIG,
    RunLedger,
    _BudgetedClient,
    _BudgetedCompletions,
)


def _block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*_args, **_kwargs):
        raise AssertionError("network access is blocked in this offline test")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


def _install_offline_client(monkeypatch, tmp_path: Path, *, timeout: float = 1.0, max_sends: int = 10,
                            raw_create=None):
    import app.llm.client as llm

    _block_external_network(monkeypatch)
    settings = SimpleNamespace(llm_timeout_seconds=timeout, llm_chat_max_tokens=4096)
    call_config = LLMCallConfig.provider_ab_experiment(
        settings, provider_id="primary", max_completion_tokens=4096,
    )
    ledger = RunLedger(tmp_path / "events.jsonl")
    send_counts: dict[str, int] = {}

    class RawCompletions:
        def create(self, **kwargs):
            if raw_create is not None:
                return raw_create(kwargs)
            current = active_state()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="offline-ok"), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
            )

    raw_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=RawCompletions().create))
    )
    experiment_config = DEFAULT_EXPERIMENT_CONFIG.__class__(
        **{**DEFAULT_EXPERIMENT_CONFIG.__dict__, "max_sdk_sends_per_workflow": max_sends}
    )
    wrapped_client = _BudgetedClient(
        raw_client, ledger=ledger, config=experiment_config, send_counts=send_counts,
    )
    proxy = wrapped_client.chat.completions
    provider = SimpleNamespace(
        id="primary",
        client=wrapped_client,
        resolve=lambda _operation, explicit, _quality: (explicit or "qwen3.8-flash", "standard"),
    )
    monkeypatch.setattr(llm, "_providers", lambda: [provider])
    monkeypatch.setattr(llm, "_quota_available", lambda: True)

    def record_offline_usage(response, _model, _operation, _profile, *, call_record=None, **_kwargs):
        if call_record:
            attempt = call_record["attempts"][-1]
            attempt["input_tokens"] = response.usage.prompt_tokens
            attempt["output_tokens"] = response.usage.completion_tokens
            attempt["usage_status"] = "complete"
            llm._refresh_llm_call_summary(active_state())

    monkeypatch.setattr(llm, "_record_usage", record_offline_usage)
    return llm, call_config, ledger, send_counts


def _state(run_id: str, config: LLMCallConfig) -> dict:
    return {
        "trace_id": run_id,
        "current_agent": "triage",
        "runtime_budget": {"max_tokens": 60_000, "max_cost": 5.0},
        "llm_usage": {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0},
        "llm_call_config": config.to_state(),
        "critic_llm_review_mode": "critic_llm_review_off",
    }


def test_resilience_worker_propagates_state_to_experiment_send_gate(monkeypatch, tmp_path):
    llm, config, ledger, send_counts = _install_offline_client(monkeypatch, tmp_path)
    state = _state("dev-context-run-1", config)
    token = activate_state(state)
    try:
        result = llm.llm_chat("system", "offline request")
    finally:
        reset_state(token)

    assert result == "offline-ok"
    call = state["llm_calls"][0]
    attempt = call["attempts"][0]
    assert call["trace_id"] == state["trace_id"]
    assert call["node"] == "triage"
    assert attempt["call_id"] == call["call_id"]
    assert attempt["trace_id"] == call["trace_id"]
    assert attempt["sdk_request_sent"] is True
    assert attempt["result"] == "response_received"
    assert attempt["input_tokens"] == 7 and attempt["output_tokens"] == 3
    events = ledger.events
    started = [row for row in events if row["event"] == "sdk_send_started"]
    received = [row for row in events if row["event"] == "sdk_response_received"]
    assert len(started) == len(received) == 1
    assert started[0]["run_id"] == received[0]["run_id"] == state["trace_id"]
    assert started[0]["call_id"] == received[0]["call_id"] == call["call_id"]
    assert started[0]["attempt_id"] == received[0]["attempt_id"] == attempt["attempt_id"]
    assert send_counts == {state["trace_id"]: 1}


@pytest.mark.parametrize(
    ("operation", "expected"),
    [("chat", ""), ("classify", ""), ("extract", None)],
)
def test_missing_context_is_recorded_as_local_refusal_before_send(monkeypatch, tmp_path, operation, expected):
    llm, config, ledger, send_counts = _install_offline_client(monkeypatch, tmp_path)
    state = _state("", config)
    state.pop("trace_id")
    token = activate_state(state)
    try:
        if operation == "chat":
            result = llm.llm_chat("system", "must not be sent")
        elif operation == "classify":
            result = llm.llm_classify("system", "must not be sent", allowed=["yes"])
        else:
            result = llm.llm_extract("system", "must not be sent")
    finally:
        reset_state(token)

    call = state["llm_calls"][0]
    assert result == expected
    assert call["outcome"] == "not_sent"
    assert call["not_sent_reason"] == "missing_run_id"
    assert call["attempts"][0]["result"] == "locally_refused"
    assert call["attempts"][0]["sdk_request_sent"] is False
    assert state["llm_token_summary"]["provider_exception_count"] == 0
    assert state["llm_token_summary"]["local_send_refusal_count"] == 1
    assert state["llm_token_summary"]["sdk_attempt_count"] == 0
    assert not [row for row in ledger.events if row["event"] == "sdk_send_started"]
    refusal = [row for row in ledger.events if row["event"] == "sdk_send_refused"]
    assert len(refusal) == 1
    assert refusal[0]["reason_code"] == "missing_run_id"
    assert refusal[0]["send_started"] is False
    assert refusal[0]["call_id"] == call["call_id"]
    assert send_counts == {}


def test_absent_active_context_is_refused_inside_resilience_worker(monkeypatch, tmp_path):
    _llm, _config, ledger, send_counts = _install_offline_client(monkeypatch, tmp_path)
    kwargs = {
        "model": "qwen3.8-flash",
        "max_completion_tokens": 4096,
        "messages": [{"role": "user", "content": "offline-only"}],
    }
    with pytest.raises(_llm.LocalSDKSendRefused):
        execute_with_resilience(
            lambda: _llm._providers()[0].client.chat.completions.create(**kwargs),
            "llm", attempts=1, timeout_seconds=1.0,
        )
    assert send_counts == {}
    assert not [row for row in ledger.events if row["event"] == "sdk_send_started"]
    refused = [row for row in ledger.events if row["event"] == "sdk_send_refused"]
    assert len(refused) == 1
    assert refused[0]["run_id"] is None
    assert refused[0]["reason_code"] == "missing_run_id"
    assert refused[0]["send_started"] is False


def test_concurrent_resilience_workers_keep_request_context_isolated(monkeypatch, tmp_path):
    barrier = threading.Barrier(2)
    seen_worker_contexts = []

    def raw_create(_kwargs):
        current = active_state()
        assert current is not None
        seen_worker_contexts.append((current["trace_id"], current["current_agent"], current["runtime_budget"]))
        barrier.wait(timeout=2)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="offline-ok"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        )

    llm, config, ledger, send_counts = _install_offline_client(
        monkeypatch, tmp_path, raw_create=raw_create,
    )
    states = {
        "run-context-a": _state("run-context-a", config),
        "run-context-b": _state("run-context-b", config),
    }
    states["run-context-b"]["current_agent"] = "diagnostic_plan"
    states["run-context-b"]["runtime_budget"] = {"max_tokens": 31_000, "max_cost": 1.5}

    def run(name):
        state = states[name]
        token = activate_state(state)
        try:
            assert llm.llm_chat("system", name) == "offline-ok"
        finally:
            reset_state(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, states))

    assert sorted((trace, node) for trace, node, _ in seen_worker_contexts) == [
        ("run-context-a", "triage"), ("run-context-b", "diagnostic_plan"),
    ]
    assert {trace: budget for trace, _node, budget in seen_worker_contexts} == {
        "run-context-a": {"max_tokens": 60_000, "max_cost": 5.0},
        "run-context-b": {"max_tokens": 31_000, "max_cost": 1.5},
    }
    assert send_counts == {"run-context-a": 1, "run-context-b": 1}
    for run_id, state in states.items():
        call = state["llm_calls"][0]
        attempt = call["attempts"][0]
        assert call["trace_id"] == attempt["trace_id"] == run_id
        assert attempt["call_id"] == call["call_id"]
        matching = [row for row in ledger.events if row.get("run_id") == run_id]
        assert [row["event"] for row in matching] == ["sdk_send_started", "sdk_response_received"]
        assert all(row["call_id"] == call["call_id"] for row in matching)
        assert all(row["attempt_id"] == attempt["attempt_id"] for row in matching)


def test_timeout_keeps_one_send_and_followup_limit_refusal_is_not_a_provider_error(monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_raw_create(_kwargs):
        import httpx
        from openai import APITimeoutError
        started.set()
        release.wait(timeout=0.06)
        finished.set()
        raise APITimeoutError(request=httpx.Request("POST", "https://offline.invalid"))

    llm, config, ledger, send_counts = _install_offline_client(
        monkeypatch, tmp_path, timeout=0.05, max_sends=1, raw_create=slow_raw_create,
    )
    state = _state("timeout-run", config)
    token = activate_state(state)
    try:
        first_result = llm.llm_chat("system", "slow request")
        assert started.wait(timeout=1)
        assert first_result == ""
        assert len([row for row in ledger.events if row["event"] == "sdk_send_started"]) == 1

        # The native SDK timeout must terminate and be recorded before the
        # caller returns; the one-send cap still blocks a second request.
        release.set()
        assert finished.wait(timeout=1)
        second_result = llm.llm_chat("system", "must be blocked by send cap")
    finally:
        release.set()
        reset_state(token)

    assert second_result == ""
    assert send_counts == {"timeout-run": 1}
    assert len([row for row in ledger.events if row["event"] == "sdk_send_started"]) == 1
    refused = [row for row in ledger.events if row["event"] == "sdk_send_refused"]
    assert len(refused) == 1 and refused[0]["reason_code"] == "send_limit_reached"
    calls = state["llm_calls"]
    assert calls[0]["outcome"] == "provider_error"  # the caller observed timeout
    assert calls[0]["attempts"][0]["sdk_request_sent"] is True
    assert calls[1]["outcome"] == "not_sent"
    assert calls[1]["not_sent_reason"] == "send_limit_reached"
    assert calls[1]["attempts"][0]["result"] == "locally_refused"
    assert calls[1]["attempts"][0]["sdk_request_sent"] is False
    assert state["llm_token_summary"]["provider_exception_count"] == 1
    assert state["llm_token_summary"]["local_send_refusal_count"] == 1
