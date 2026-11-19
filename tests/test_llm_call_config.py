from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from app.core.execution_context import activate_state, reset_state
from app.llm import client as llm
from app.llm.call_config import LLMCallConfig
from app.resilience import ExternalServiceError
from evaluation.provider_candidate_batch import ExperimentConfig


def _settings():
    return SimpleNamespace(
        deepseek_api_key="offline-primary-key",
        deepseek_base_url="https://offline-primary.invalid/v1",
        llm_timeout_seconds=1.25,
        llm_resilience_attempts=3,
        llm_providers_json=json.dumps([{
            "id": "backup",
            "api_key_env": "OFFLINE_BACKUP_KEY",
            "base_url": "https://offline-backup.invalid/v1",
            "default_model": "backup-model",
        }]),
        llm_model="normal-model",
        llm_chat_max_tokens=512,
        llm_extract_max_tokens=512,
        user_id="offline-test-user",
    )


def _state(trace_id: str, config: LLMCallConfig) -> dict:
    return {
        "trace_id": trace_id,
        "current_agent": "critic",
        "llm_call_config": config.to_state(),
        "llm_call_config_summary": config.trace_summary(),
        "llm_calls": [],
        "runtime_budget": {},
        "llm_usage": {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0},
    }


def _response(text="offline-ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
    )


def test_only_retryable_upstream_failures_enter_provider_fallback(monkeypatch):
    monkeypatch.setattr(llm, "_RECOVERABLE_ERRORS", (RuntimeError,))

    retry_exhausted = ExternalServiceError(
        "llm", "external_call_failed", "retries exhausted",
    )
    retry_exhausted.__cause__ = RuntimeError("offline transport failure")
    assert llm._is_recoverable_provider_failure(retry_exhausted)
    assert llm._is_recoverable_provider_failure(
        ExternalServiceError("llm", "timeout", "offline timeout"),
    )
    assert not llm._is_recoverable_provider_failure(
        ExternalServiceError("llm", "circuit_open", "circuit open"),
    )
    application_error = ExternalServiceError(
        "llm", "external_call_failed", "response processing failed",
    )
    application_error.__cause__ = ValueError("invalid response")
    assert not llm._is_recoverable_provider_failure(application_error)


def test_experiment_config_reaches_sdk_and_one_real_resilience_attempt(monkeypatch, tmp_path):
    settings = _settings()
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setenv("OFFLINE_BACKUP_KEY", "offline-backup-key")
    monkeypatch.setattr(llm, "model_for", lambda operation, explicit, quality=False: (f"model-{operation}", "standard"))
    monkeypatch.setattr(llm, "_quota_available", lambda: True)
    monkeypatch.setattr(llm, "_record_usage", lambda *args, **kwargs: None)

    constructed = []
    sdk_calls = []

    class FakeClient:
        def __init__(self, provider):
            self.provider = provider
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            sdk_calls.append((self.provider, kwargs["model"]))
            raise RuntimeError("offline retryable failure")

    def fake_openai(**kwargs):
        provider = "backup" if "backup.invalid" in kwargs["base_url"] else "primary"
        constructed.append((provider, kwargs))
        return FakeClient(provider)

    monkeypatch.setattr(llm, "OpenAI", fake_openai)
    policy = LLMCallConfig.provider_ab_experiment(settings, provider_id="primary")
    state = _state("experiment-failure", policy)
    token = activate_state(state)
    try:
        assert llm.llm_chat("system", "offline input") == ""
    finally:
        reset_state(token)

    assert constructed == [("primary", {
        "api_key": "offline-primary-key",
        "base_url": "https://offline-primary.invalid/v1",
        "timeout": 1.25,
        "max_retries": 0,
    })]
    assert sdk_calls == [("primary", "model-chat")]
    attempts = state["llm_calls"][0]["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["provider"] == "primary"
    assert attempts[0]["attempt_order"] == 1
    assert attempts[0]["result"] == "provider_exception"
    summary = state["llm_call_config_summary"]
    assert summary == {
        "mode": "provider_ab_experiment",
        "provider_id": "primary",
        "model": ["model-chat"],
        "sdk_max_retries": 0,
        "application_attempts": 1,
        "fallback_enabled": False,
        "timeout_seconds": 1.25,
        "timeout_owner": "sdk_transport",
        "max_completion_tokens": 4096,
        "connect_retry_attempts": 0,
        "connect_timeout_seconds": None,
        "retry_scope": "none",
        "effective_max_sdk_invocations_per_logical_call": 1,
    }
    assert state["llm_token_summary"]["sdk_attempt_count"] == 1
    artifact_dir = tmp_path / "provider_retry_isolation"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "sample_trace.json").write_text(json.dumps({
        "trace_id": state["trace_id"],
        "llm_call_config_summary": state["llm_call_config_summary"],
        "llm_token_summary": state["llm_token_summary"],
        "llm_calls": state["llm_calls"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def test_frozen_timeout_overrides_runtime_setting_at_actual_sdk_boundary(monkeypatch):
    """The timeout checked from the freeze must be the timeout the SDK receives."""
    settings = _settings()
    settings.llm_timeout_seconds = 300.0
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setattr(llm, "model_for", lambda operation, explicit, quality=False: ("qwen3.8-flash", "standard"))
    monkeypatch.setattr(llm, "_quota_available", lambda: True)
    monkeypatch.setattr(llm, "_record_usage", lambda *args, **kwargs: None)

    frozen = asdict(ExperimentConfig())
    arms = {
        "B": {**frozen, "arm": "baseline_tool_loop", "max_rounds": frozen["max_graph_steps"]},
        "G": {**frozen, "arm": "governed_workflow"},
    }
    arms["B"]["critic_review_mode"] = "not_applicable_baseline_has_no_governed_critic"
    arms["B"]["optional_critic_reviews"] = []
    arms["G"]["critic_review_mode"] = "full"
    arms["G"]["optional_critic_reviews"] = ["groundedness", "completeness", "consistency"]
    config = ExperimentConfig.from_frozen_arm_configs(arms)
    assert config.model_timeout_seconds == 60.0
    call_config = config.build_llm_call_config(settings)
    assert call_config.timeout_seconds == 60.0

    constructed = []

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: _response("ok")))

    def fake_openai(**kwargs):
        constructed.append(kwargs)
        return FakeClient()

    monkeypatch.setattr(llm, "OpenAI", fake_openai)
    state = _state("frozen-timeout-effective", call_config)
    token = activate_state(state)
    try:
        assert llm.llm_chat("system", "offline request") == "ok"
    finally:
        reset_state(token)

    assert len(constructed) == 1
    assert constructed[0]["timeout"] == 60.0
    assert constructed[0]["max_retries"] == 0
    assert state["llm_call_config_summary"]["timeout_seconds"] == 60.0
    assert state["llm_token_summary"]["sdk_attempt_count"] == 1


def test_frozen_b_g_call_timeout_mismatch_is_rejected():
    frozen = asdict(ExperimentConfig())
    arms = {"B": dict(frozen), "G": dict(frozen)}
    arms["G"]["model_timeout_seconds"] = 300.0
    try:
        ExperimentConfig.from_frozen_arm_configs(arms)
    except ValueError as exc:
        assert "model_timeout_seconds" in str(exc)
    else:
        raise AssertionError("Mismatched B/G frozen timeout must fail closed")


def test_experiment_uses_only_max_completion_tokens_for_every_chat_operation(monkeypatch):
    settings = _settings()
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setattr(llm, "model_for", lambda operation, explicit, quality=False: (f"model-{operation}", "standard"))
    monkeypatch.setattr(llm, "_quota_available", lambda: True)
    monkeypatch.setattr(llm, "_record_usage", lambda *args, **kwargs: None)
    sent = []

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            sent.append(dict(kwargs))
            content = {
                "model-classify": "classification",
                "model-extract": '{"ok": true}',
                "model-chat": "offline-ok",
            }[kwargs["model"]]
            return _response(content)

    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: FakeClient())
    config = LLMCallConfig.provider_ab_experiment(settings, provider_id="primary", max_completion_tokens=4096)
    operations = [
        lambda: llm.llm_classify("system", "input", ["classification"], model="classify"),
        lambda: llm.llm_extract("system", "input", model="extract", max_tokens=1024),
        lambda: llm.llm_chat("system", "input", model="chat"),
    ]
    for index, operation in enumerate(operations):
        state = _state(f"completion-cap-{index}", config)
        token = activate_state(state)
        try:
            operation()
        finally:
            reset_state(token)

    normal_state = _state("completion-cap-normal-control", LLMCallConfig.normal(settings))
    normal_token = activate_state(normal_state)
    try:
        llm.llm_chat("system", "normal control", max_tokens=37)
        llm.llm_extract("system", "normal extract control", max_tokens=1024)
    finally:
        reset_state(normal_token)

    assert len(sent) == 5
    assert [item["model"] for item in sent[:3]] == ["model-classify", "model-extract", "model-chat"]
    assert all(item["max_completion_tokens"] == 4096 for item in sent[:3])
    assert all("max_tokens" not in item for item in sent[:3])
    assert sent[3]["max_tokens"] == 37
    assert "max_completion_tokens" not in sent[3]
    assert sent[4]["max_tokens"] == 1024
    assert "max_completion_tokens" not in sent[4]


def test_normal_policy_keeps_configured_retry_limit_and_provider_fallback(monkeypatch):
    settings = _settings()
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setenv("OFFLINE_BACKUP_KEY", "offline-backup-key")
    monkeypatch.setattr(llm, "model_for", lambda operation, explicit, quality=False: (f"model-{operation}", "standard"))
    monkeypatch.setattr(llm, "_quota_available", lambda: True)
    monkeypatch.setattr(llm, "_record_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm, "_RECOVERABLE_ERRORS", (RuntimeError,))

    constructed = []
    sdk_calls = []

    class FakeClient:
        def __init__(self, provider):
            self.provider = provider
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            sdk_calls.append(self.provider)
            if self.provider == "primary":
                raise RuntimeError("offline primary failure")
            return _response("backup-ok")

    def fake_openai(**kwargs):
        provider = "backup" if "backup.invalid" in kwargs["base_url"] else "primary"
        constructed.append((provider, kwargs))
        return FakeClient(provider)

    monkeypatch.setattr(llm, "OpenAI", fake_openai)
    resilience_calls = []

    real_resilience = llm.execute_with_resilience

    def offline_resilience(operation, service, *, attempts, timeout_seconds,
                           operation_manages_timeout=False, backoff_seconds=0.05,
                           retry_if=None):
        resilience_calls.append((service, attempts, timeout_seconds))
        # Exercise the real retry loop without sleeping between offline calls.
        return real_resilience(
            operation, service, attempts=attempts,
            timeout_seconds=timeout_seconds, backoff_seconds=0, retry_if=retry_if,
        )

    monkeypatch.setattr(llm, "execute_with_resilience", offline_resilience)
    state = _state("normal-fallback", LLMCallConfig.normal(settings))
    token = activate_state(state)
    try:
        assert llm.llm_chat("system", "offline input") == "backup-ok"
    finally:
        reset_state(token)

    assert [provider for provider, _ in constructed] == ["primary", "backup"]
    assert all(options["max_retries"] == 2 for _, options in constructed)
    assert all(options["timeout"] == 1.25 for _, options in constructed)
    # Three application-level attempts exhaust the primary before one successful
    # backup attempt. These are SDK invocation attempts, not HTTP request counts.
    assert sdk_calls == ["primary", "primary", "primary", "backup"]
    assert resilience_calls == [("llm", 3, 1.25), ("llm", 3, 1.25)]
    call = state["llm_calls"][0]
    assert [item["provider"] for item in call["attempts"]] == [
        "primary", "primary", "primary", "backup",
    ]
    assert [item["attempt_order"] for item in call["attempts"]] == [1, 2, 3, 4]
    assert [item["result"] for item in call["attempts"]] == [
        "provider_exception", "provider_exception", "provider_exception",
        "response_received",
    ]
    assert state["llm_call_config_summary"]["fallback_enabled"] is True


def test_critic_experiment_configs_match_and_do_not_leak_across_concurrent_requests(monkeypatch):
    settings = _settings()
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setenv("OFFLINE_BACKUP_KEY", "offline-backup-key")
    monkeypatch.setattr(llm, "model_for", lambda operation, explicit, quality=False: (f"model-{operation}", "standard"))
    monkeypatch.setattr(llm, "_quota_available", lambda: True)
    monkeypatch.setattr(llm, "_record_usage", lambda *args, **kwargs: None)

    constructed = []
    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: _response()))

    def fake_openai(**kwargs):
        active = llm.active_state()
        trace = active.get("trace_id") if active else None
        constructed.append((trace, kwargs["base_url"], kwargs["max_retries"], kwargs["timeout"]))
        return FakeClient()

    monkeypatch.setattr(llm, "OpenAI", fake_openai)
    app_limits = []

    def offline_resilience(operation, service, *, attempts, timeout_seconds,
                           operation_manages_timeout=False, backoff_seconds=0.05,
                           retry_if=None):
        active = llm.active_state()
        assert operation_manages_timeout is (active["llm_call_config"]["mode"] == "provider_ab_experiment")
        app_limits.append((active.get("trace_id") if active else None, attempts, timeout_seconds))
        return operation()

    monkeypatch.setattr(llm, "execute_with_resilience", offline_resilience)

    shared_ab_config = LLMCallConfig.provider_ab_experiment(settings, provider_id="primary")
    # The two Critic arms receive the same immutable request policy.
    arm_configs = {"critic_full": shared_ab_config, "critic_llm_review_off": shared_ab_config}
    assert arm_configs["critic_full"] == arm_configs["critic_llm_review_off"]
    states = {arm: _state(arm, config) for arm, config in arm_configs.items()}
    states["normal"] = _state("normal", LLMCallConfig.normal(settings))

    def run(name):
        token = activate_state(states[name])
        try:
            assert llm.llm_chat("system", name) == "offline-ok"
        finally:
            reset_state(token)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run, states))

    for arm in ("critic_full", "critic_llm_review_off"):
        summary = states[arm]["llm_call_config_summary"]
        assert summary["mode"] == "provider_ab_experiment"
        assert summary["provider_id"] == "primary"
        assert summary["sdk_max_retries"] == 0
        assert summary["application_attempts"] == 1
        assert summary["fallback_enabled"] is False
        assert summary["max_completion_tokens"] == 4096
        assert summary["model"] == ["model-chat"]
        assert len(states[arm]["llm_calls"][0]["attempts"]) == 1
    assert states["normal"]["llm_call_config_summary"]["sdk_max_retries"] == 2
    assert states["normal"]["llm_call_config_summary"]["application_attempts"] == 3
    assert states["normal"]["llm_call_config_summary"]["fallback_enabled"] is True
    assert sorted((name, attempts) for name, attempts, _ in app_limits) == [
        ("critic_full", 1), ("critic_llm_review_off", 1), ("normal", 3),
    ]
    assert {trace for trace, *_ in constructed} == set(states)
    assert all(timeout == 1.25 for *_, timeout in constructed)
    assert all(
        retries == (0 if trace.startswith("critic_") else 2)
        for trace, _, retries, _ in constructed
    )


def test_experiment_retries_one_connection_handshake_only(monkeypatch):
    from openai import APIConnectionError
    import httpx

    settings = _settings()
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setattr(llm, "model_for", lambda operation, explicit, quality=False: ("qwen3.8-flash", "standard"))
    monkeypatch.setattr(llm, "_quota_available", lambda: True)
    monkeypatch.setattr(llm, "_RECOVERABLE_ERRORS", (RuntimeError,))

    constructor_args = []
    sdk_attempts = []

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            sdk_attempts.append(dict(kwargs))
            if len(sdk_attempts) == 1:
                error = APIConnectionError(
                    message="Connection error.",
                    request=httpx.Request("POST", "https://offline-primary.invalid/v1/chat/completions"),
                )
                error.__cause__ = httpx.ConnectTimeout("TLS handshake operation timed out")
                raise error
            return _response("connected-after-one-retry")

    def fake_openai(**kwargs):
        constructor_args.append(kwargs)
        return FakeClient()

    monkeypatch.setattr(llm, "OpenAI", fake_openai)
    config = LLMCallConfig.provider_ab_experiment(
        settings, provider_id="primary", max_completion_tokens=4096,
        timeout_seconds=60, connect_retry_attempts=1, connect_timeout_seconds=20,
    )
    state = _state("tls-connect-retry", config)
    token = activate_state(state)
    try:
        assert llm.llm_chat("system", "offline handshake test") == "connected-after-one-retry"
    finally:
        reset_state(token)

    assert len(constructor_args) == 1
    assert constructor_args[0]["max_retries"] == 0
    timeout = constructor_args[0]["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 20
    assert timeout.read == 60
    assert [call["model"] for call in sdk_attempts] == ["qwen3.8-flash"] * 2
    attempts = state["llm_calls"][0]["attempts"]
    assert len(attempts) == 2
    assert [attempt["attempt_order"] for attempt in attempts] == [1, 2]
    assert [attempt["result"] for attempt in attempts] == ["provider_exception", "response_received"]
    assert attempts[0]["sdk_request_sent"] is True
    assert attempts[1]["sdk_request_sent"] is True
    assert attempts[0]["call_id"] == attempts[1]["call_id"]
    assert attempts[0]["attempt_id"] != attempts[1]["attempt_id"]
    assert state["llm_token_summary"]["sdk_attempt_count"] == 2
    assert state["llm_token_summary"]["unknown_usage_attempt_count"] == 1
    assert state["llm_token_summary"]["usage_complete"] is False
    assert state["llm_token_summary"]["total_tokens"] is None
    assert state["llm_call_config_summary"]["retry_scope"] == "connect_establishment_only"
    assert state["llm_call_config_summary"]["effective_max_sdk_invocations_per_logical_call"] == 2


def test_experiment_does_not_retry_read_timeout_or_provider_response_failure(monkeypatch):
    import httpx

    settings = _settings()
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setattr(llm, "model_for", lambda operation, explicit, quality=False: ("qwen3.8-flash", "standard"))
    monkeypatch.setattr(llm, "_quota_available", lambda: True)
    monkeypatch.setattr(llm, "_record_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm, "_RECOVERABLE_ERRORS", (RuntimeError,))

    sent = []

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            sent.append(kwargs)
            raise httpx.ReadTimeout("response body read timed out")

    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: FakeClient())
    config = LLMCallConfig.provider_ab_experiment(
        settings, provider_id="primary", timeout_seconds=60,
        connect_retry_attempts=1, connect_timeout_seconds=20,
    )
    state = _state("tls-read-timeout-not-retried", config)
    token = activate_state(state)
    try:
        assert llm.llm_chat("system", "offline read-timeout test") == ""
    finally:
        reset_state(token)

    assert len(sent) == 1
    attempts = state["llm_calls"][0]["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["result"] == "provider_exception"
    assert state["llm_token_summary"]["sdk_attempt_count"] == 1
    assert not llm._is_transient_connection_establishment_error(
        httpx.ReadTimeout("TLS handshake operation timed out")
    )


def test_normal_workflow_entry_passes_same_internal_config_to_both_critic_arms(monkeypatch):
    from app.graph import workflow
    from app.graph.critic_review_mode import CriticLLMReviewMode
    from app.schemas.chat import ChatRequest

    settings = _settings()
    policy = LLMCallConfig.provider_ab_experiment(settings, provider_id="primary")
    seen = []
    built_modes = []

    class CapturingGraph:
        def invoke(self, graph_input, **kwargs):
            seen.append(dict(graph_input))
            result = dict(graph_input)
            result.update({"task_type": "anomaly_analysis", "answer": "offline", "errors": []})
            return result

    def build_graph(**kwargs):
        built_modes.append(kwargs["critic_llm_review_mode"])
        return CapturingGraph()

    monkeypatch.setattr(workflow, "build_chat_graph", build_graph)
    for mode in (CriticLLMReviewMode.FULL, CriticLLMReviewMode.CRITIC_LLM_REVIEW_OFF):
        workflow.run_chat_workflow(
            "离线测试问题",
            trace_id=f"trace-{mode.value}",
            _critic_llm_review_mode=mode,
            _llm_call_config=policy,
        )

    assert len(seen) == 2
    assert built_modes == [CriticLLMReviewMode.FULL, CriticLLMReviewMode.CRITIC_LLM_REVIEW_OFF]
    assert all(state["llm_call_config"] == policy.to_state() for state in seen)
    assert all(state["llm_call_config_summary"] == policy.trace_summary() for state in seen)
    assert "llm_call_config" not in ChatRequest.model_fields
    assert "_llm_call_config" not in ChatRequest.model_fields
