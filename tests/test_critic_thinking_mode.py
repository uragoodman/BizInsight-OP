"""Offline checks for the request-scoped optional Critic thinking mode."""

from __future__ import annotations

import copy
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import httpx
import pytest


TRACE = Path(
    "docs/testing/provider_aov_token_stop_rerun_20261012/traces/"
    "provider-trial-aov_ratio-critic_llm_review_off-c5179b4ee02a.json"
)


def _network_block(monkeypatch):
    blocked: list[str] = []

    def fail(name):
        def _blocked(*_args, **_kwargs):
            blocked.append(name)
            raise RuntimeError(f"external network blocked by test: {name}")
        return _blocked

    monkeypatch.setattr(socket, "create_connection", fail("socket.create_connection"))
    monkeypatch.setattr(socket.socket, "connect", fail("socket.socket.connect"))
    monkeypatch.setattr(httpx.Client, "send", fail("httpx.Client.send"))
    with pytest.raises(RuntimeError, match="external network blocked"):
        socket.create_connection(("127.0.0.1", 1), timeout=0.01)
    with pytest.raises(RuntimeError, match="external network blocked"):
        httpx.Client().send(httpx.Request("POST", "http://127.0.0.1:1/blocked"))
    assert blocked == ["socket.create_connection", "httpx.Client.send"]
    blocked.clear()
    return blocked


def _state(trace_id: str) -> dict:
    state = copy.deepcopy(json.loads(TRACE.read_text(encoding="utf-8"))["state_snapshot"])
    state["trace_id"] = trace_id
    state["llm_calls"] = []
    state["llm_usage"] = {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0}
    state["llm_token_summary"] = {}
    state["runtime_budget"] = {
        **(state.get("runtime_budget") or {}),
        "max_tokens": 0,
        "max_graph_steps": 48,
        "max_latency_ms": 600_000,
    }
    state["runtime_steps"] = 0
    state["runtime_started_at"] = perf_counter()
    state["node_timings"] = []
    state["errors"] = [
        error for error in (state.get("errors") or [])
        if error.get("code") != "token_budget_exceeded"
    ]
    return state


class _FakeCompletions:
    def __init__(self, responses=None, barrier=None):
        self.calls: list[dict] = []
        self.responses = list(responses or [])
        self.barrier = barrier
        self.lock = threading.Lock()

    def create(self, **kwargs):
        with self.lock:
            self.calls.append(copy.deepcopy(kwargs))
            call_index = len(self.calls)
            content = self.responses.pop(0) if self.responses else '{"verdict":"passed","issues":[]}'
        # Overlap the two request-local configurations once, then allow each
        # request to finish its remaining sequential Critic checks.
        if self.barrier and call_index <= 2:
            self.barrier.wait(timeout=10)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )],
            # Missing usage intentionally remains unknown in call records.
            usage=None,
        )


def _install_fake_provider(monkeypatch, *, responses=None, barrier=None, model="qwen3.8-flash"):
    from app.llm import client as llm

    completions = _FakeCompletions(responses=responses, barrier=barrier)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = llm._Provider(
        id="primary",
        client=fake_client,
        resolve=lambda _operation, explicit, _quality: (explicit or model, "fast"),
    )
    monkeypatch.setattr(llm, "_providers", lambda: [provider])
    return completions


def _invoke_critic(mode: str, thinking_mode: str, state: dict) -> dict:
    from app.graph.workflow import build_chat_graph

    graph = build_chat_graph(
        critic_llm_review_mode=mode,
        critic_llm_thinking_mode=thinking_mode,
    )
    return graph.nodes["critic"].invoke(state)


def test_actual_sdk_params_respect_critic_and_general_defaults(monkeypatch):
    blocked = _network_block(monkeypatch)
    from app.core.config import get_settings

    settings = copy.copy(get_settings())
    settings.critic_default_non_thinking = True
    monkeypatch.setattr("app.graph.nodes.critic.get_settings", lambda: settings)
    completions = _install_fake_provider(monkeypatch)

    experiment = _invoke_critic(
        "full", "non_thinking_experiment", _state("critic-thinking-experiment")
    )
    assert len(completions.calls) == 3
    assert all(call["extra_body"] == {"enable_thinking": False} for call in completions.calls)
    assert all(call.get("max_completion_tokens") == 4096 and "max_tokens" not in call
               for call in completions.calls)
    assert experiment["critic_llm_thinking_config"] == {
        "mode": "non_thinking_experiment",
        "scope": "optional_llm_critic_reviews",
        "enable_thinking": False,
        "sdk_parameter": "extra_body.enable_thinking",
        "default_non_thinking": True,
        "effective_policy": "disabled_if_supported",
    }
    policy = experiment["llm_call_config"]
    assert policy["sdk_max_retries"] == 0
    assert policy["application_attempts"] == 1
    assert policy["fallback_enabled"] is False
    experiment_calls = experiment["llm_calls"]
    assert [call["request_options"]["enable_thinking"] for call in experiment_calls] == [False] * 3
    assert all(call["attempts"][0]["request_options"]["enable_thinking"] is False for call in experiment_calls)
    assert all(call["attempts"][0]["input_tokens"] is None for call in experiment_calls)
    assert all(call["attempts"][0]["output_tokens"] is None for call in experiment_calls)

    completions.calls.clear()
    default = _invoke_critic("full", "provider_default", _state("critic-thinking-default"))
    assert len(completions.calls) == 3
    assert all(call["extra_body"] == {"enable_thinking": False} for call in completions.calls)
    assert default["critic_llm_thinking_config"]["enable_thinking"] is None
    assert default["critic_llm_thinking_config"]["sdk_parameter"] == "omitted_provider_default"
    assert default["critic_llm_thinking_config"]["effective_policy"] == "disabled_if_supported"

    # Unrelated public model operations keep their default request shape.
    from app.llm import client as llm

    completions.calls.clear()
    assert llm.llm_extract("return JSON", "{}") == {"verdict": "passed", "issues": []}
    llm.llm_classify("choose", "x", ["x"])
    llm.llm_chat("system", "ordinary request")
    assert len(completions.calls) == 3
    assert all("extra_body" not in call for call in completions.calls)
    assert blocked == []


def test_off_mode_makes_no_optional_calls_and_missing_review_output_is_not_passed(monkeypatch):
    _network_block(monkeypatch)
    from app.core.config import get_settings

    settings = copy.copy(get_settings())
    settings.critic_review_stop_on_failure_enabled = True
    monkeypatch.setattr("app.graph.nodes.critic.get_settings", lambda: settings)
    completions = _install_fake_provider(monkeypatch)
    off = _invoke_critic("critic_llm_review_off", "non_thinking_experiment", _state("critic-off"))
    assert completions.calls == []
    assert off["critic_result"]["llm_review"]["status"] == "evaluation_ablation"
    assert off["critic_llm_thinking_config"]["enable_thinking"] is False
    assert [off["critic_result"]["llm_review"][key]["verdict"] for key in (
        "groundedness", "completeness", "consistency",
    )] == ["evaluation_ablation"] * 3

    completions = _install_fake_provider(
        monkeypatch,
        responses=[
            '{"verdict":"failed","issues":["unsupported claim"]}',
            "",
            "not-json",
        ],
    )
    missing = _invoke_critic("full", "non_thinking_experiment", _state("critic-missing-review"))
    reviews = missing["critic_result"]["llm_review"]
    assert [reviews[key]["verdict"] for key in ("groundedness", "completeness", "consistency")] == [
        "failed", "not_completed", "not_run",
    ]
    assert len(completions.calls) == 2
    assert all(reviews[key]["verdict"] != "passed" for key in (
        "groundedness", "completeness", "consistency",
    ))
    assert all(call["extra_body"] == {"enable_thinking": False} for call in completions.calls)


def test_combined_experiment_one_call_and_incomplete_fail_closed(monkeypatch):
    _network_block(monkeypatch)
    completions = _install_fake_provider(monkeypatch)
    combined = _invoke_critic("combined_review_experiment", "non_thinking_experiment", _state("combined-pass"))
    review = combined["critic_result"]["llm_review"]
    assert len(completions.calls) == 1
    assert review["status"] == "passed"
    assert review["combined_experiment"]["verdict"] == "passed"
    assert review["groundedness"]["verdict"] == "not_run_combined_experiment"
    assert review["approval_basis"] == "deterministic_and_combined_experimental_review_passed"

    completions = _install_fake_provider(monkeypatch, responses=["not-json"])
    failed = _invoke_critic("combined_review_experiment", "non_thinking_experiment", _state("combined-invalid"))
    assert len(completions.calls) == 1
    assert failed["critic_result"]["llm_review"]["status"] == "incomplete"
    assert failed["critic_verdict"] == "rejected"

    completions = _install_fake_provider(monkeypatch, responses=['{"verdict":"failed","issues":["unsupported claim"]}'])
    rejected = _invoke_critic("combined_review_experiment", "non_thinking_experiment", _state("combined-failed"))
    assert len(completions.calls) == 1
    assert rejected["critic_result"]["llm_review"]["status"] == "failed"
    assert rejected["critic_verdict"] != "approved"


def test_request_scoped_thinking_config_does_not_leak_between_interleaved_graphs(monkeypatch):
    _network_block(monkeypatch)
    from app.core.config import get_settings

    settings = copy.copy(get_settings())
    settings.critic_default_non_thinking = False
    monkeypatch.setattr("app.graph.nodes.critic.get_settings", lambda: settings)
    barrier = threading.Barrier(2)
    completions = _install_fake_provider(monkeypatch, barrier=barrier)

    def run(thinking_mode: str, trace_id: str):
        return _invoke_critic("full", thinking_mode, _state(trace_id))

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(run, "non_thinking_experiment", "critic-concurrent-a")
        b = pool.submit(run, "provider_default", "critic-concurrent-b")
        result_a, result_b = a.result(timeout=20), b.result(timeout=20)

    a_calls = [call for call in result_a["llm_calls"] if call["operation"] == "extract"]
    b_calls = [call for call in result_b["llm_calls"] if call["operation"] == "extract"]
    assert a_calls
    assert b_calls
    assert len(completions.calls) == len(a_calls) + len(b_calls)
    assert all(call["request_options"]["enable_thinking"] is False for call in a_calls)
    assert all(call["request_options"]["enable_thinking"] is None for call in b_calls)
    assert sum("extra_body" in call for call in completions.calls) == len(a_calls)


def test_unknown_thinking_mode_is_rejected_before_graph_use():
    from app.graph.workflow import build_chat_graph

    with pytest.raises(ValueError, match="Invalid internal Critic thinking mode"):
        build_chat_graph(critic_llm_thinking_mode="guess")


def test_non_thinking_mode_refuses_unsupported_model_without_sending(monkeypatch):
    _network_block(monkeypatch)
    completions = _install_fake_provider(monkeypatch, model="some-other-model")
    from app.core.execution_context import activate_state, reset_state
    from app.llm import client as llm

    state = _state("critic-thinking-unsupported-model")
    token = activate_state(state)
    try:
        with pytest.raises(ValueError, match="requires the configured primary"):
            llm.llm_extract("review", "content", _enable_thinking=False)
    finally:
        reset_state(token)
    assert completions.calls == []
    assert state["llm_calls"][0]["outcome"] == "configuration_error"
    assert state["llm_calls"][0]["not_sent_reason"] == "unsupported_critic_thinking_target"
    assert state["llm_calls"][0]["attempts"] == []
