from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import threading

import pytest

from app.core.config import Settings
from app.core.execution_context import activate_state, reset_state
from app.llm import client as llm
from app.resilience import policy as resilience_policy
from app.llm.interaction_policy import (
    OPERATION_POLICY_PROPOSAL,
    OPERATION_PROFILE_PROPOSAL,
    InteractionPolicy,
    POLICIES,
    policy_for,
)


DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _settings():
    return SimpleNamespace(
        llm_timeout_seconds=1.0,
        llm_resilience_attempts=2,
        llm_extract_max_tokens=256,
        llm_chat_max_tokens=256,
        deepseek_api_key="offline-key",
        deepseek_base_url=DASHSCOPE_URL,
        llm_model="qwen3.8-flash",
        user_id="offline-user",
    )


@pytest.fixture
def offline_llm(monkeypatch, request):
    resilience_policy._BREAKERS.clear()
    request.addfinalizer(resilience_policy._BREAKERS.clear)
    settings = _settings()
    recorded_usage = []
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setattr(llm, "price_for", lambda profile: (0.01, 0.02))
    monkeypatch.setattr(llm, "record_usage", lambda **kwargs: recorded_usage.append(kwargs))
    monkeypatch.setattr(llm, "consume_llm_usage", lambda *_args: None)
    monkeypatch.setattr(llm, "record_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm, "_quota_available", lambda: True)
    return settings, recorded_usage


@contextmanager
def _active_state(trace_id="interaction-modes"):
    state = {
        "trace_id": trace_id,
        "llm_calls": [],
        "runtime_budget": {},
        "llm_usage": {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0},
    }
    token = activate_state(state)
    try:
        yield state
    finally:
        reset_state(token)


def _chunk(*, content=None, reasoning=None, finish_reason=None, usage=None):
    choices = []
    if content is not None or reasoning is not None or finish_reason is not None:
        choices.append(SimpleNamespace(
            delta=SimpleNamespace(content=content, reasoning_content=reasoning),
            finish_reason=finish_reason,
        ))
    return SimpleNamespace(choices=choices, usage=usage)


def _usage(input_tokens=8, output_tokens=3):
    return SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens)


def _completion(text="offline answer", usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text), finish_reason="stop",
        )],
        usage=usage,
    )


class FakeStream:
    def __init__(self, chunks=(), *, failure=None, block_after=False, interval=0.0):
        self._chunks = iter(chunks)
        self.failure = failure
        self.block_after = block_after
        self.interval = interval
        self.closed = threading.Event()
        self._failure_raised = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed.is_set():
            raise RuntimeError("fake stream closed")
        if self.interval:
            self.closed.wait(self.interval)
            if self.closed.is_set():
                raise RuntimeError("fake stream closed")
        try:
            return next(self._chunks)
        except StopIteration:
            if self.failure is not None and not self._failure_raised:
                self._failure_raised = True
                raise self.failure
            if self.block_after:
                self.closed.wait(0.8)
                raise RuntimeError("fake stream closed while waiting")
            raise

    def close(self):
        self.closed.set()


class InfiniteStream:
    def __init__(self, chunk, interval=0.003):
        self.chunk = chunk
        self.interval = interval
        self.closed = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed.is_set():
            raise RuntimeError("fake stream closed")
        self.closed.wait(self.interval)
        if self.closed.is_set():
            raise RuntimeError("fake stream closed")
        return self.chunk

    def close(self):
        self.closed.set()


class FakeClient:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self.streams = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.handler(kwargs)
        if hasattr(result, "close") and hasattr(result, "__iter__"):
            self.streams.append(result)
        return result

    def close(self):
        self.closed = True
        for stream in self.streams:
            stream.close()


class ProviderHTTPError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.body = {"error": {"message": message}}
        super().__init__(message)


def _provider(handler, *, provider_id="primary", model="qwen3.8-flash", base_url=DASHSCOPE_URL):
    client = FakeClient(handler)
    provider = llm._Provider(
        provider_id,
        client,
        lambda _operation, _explicit, _quality: (model, "fast"),
        base_url,
    )
    return provider, client


def _install_providers(monkeypatch, *providers):
    monkeypatch.setattr(llm, "_providers", lambda **_kwargs: list(providers))


def _short_stream_policy(*, idle=0.05, total=0.4, thinking="provider_default", retries=0):
    return InteractionPolicy(
        name="test_stream",
        transport="stream",
        thinking=thinking,
        connect_timeout_seconds=0.05,
        idle_timeout_seconds=idle,
        total_timeout_seconds=total,
        retry_before_first_token=retries,
    )


def test_policy_preset_tables_are_point_named_proposals():
    assert {
        name: (
            policy.transport, policy.thinking, policy.connect_timeout_seconds,
            policy.idle_timeout_seconds, policy.total_timeout_seconds,
        )
        for name, policy in POLICIES.items()
    } == {
        "stream_thinking": ("stream", "on", 10.0, 30.0, 180.0),
        "stream_fast": ("stream", "off", 10.0, 20.0, 90.0),
        "nonstream_fast": ("nonstream", "off", 10.0, None, 25.0),
        "nonstream_thinking": ("nonstream", "on", 10.0, None, 180.0),
    }
    assert len(OPERATION_POLICY_PROPOSAL) == 20
    assert OPERATION_POLICY_PROPOSAL["extract.semantic_interpreter"] == "stream_fast"
    assert OPERATION_PROFILE_PROPOSAL["extract.semantic_interpreter"] == "standard"
    assert {
        OPERATION_PROFILE_PROPOSAL[name]
        for name in (
            "agent_action.diagnostic",
            "agent_action.insight",
            "agent_action.open_diagnosis",
        )
    } == {"standard"}
    assert policy_for("classify.task_type").name == "nonstream_fast"
    assert policy_for("chat.answer").transport == "nonstream"
    assert policy_for("chat") is None


def test_default_none_keeps_nonstream_request_path(offline_llm, monkeypatch):
    provider, client = _provider(lambda _kwargs: _completion("default path", _usage()))
    _install_providers(monkeypatch, provider)

    with _active_state() as state:
        assert llm.llm_chat("system", "question", interaction_policy=None) == "default path"

    assert "stream" not in client.calls[0]
    assert "stream_options" not in client.calls[0]
    call = state["llm_calls"][0]
    assert call["request_options"] == {}
    assert "first_token_ms" not in call


def test_stream_fast_accumulates_content_and_final_usage(offline_llm, monkeypatch):
    stream = FakeStream([
        _chunk(content='{"ok":'),
        _chunk(content="true}", finish_reason="stop"),
        _chunk(usage=_usage(11, 4)),
    ])
    provider, client = _provider(lambda _kwargs: stream)
    _install_providers(monkeypatch, provider)

    with _active_state() as state:
        result = llm.llm_extract("JSON", "request", interaction_policy="stream_fast")

    assert result == {"ok": True}
    request = client.calls[0]
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    assert request["extra_body"] == {"enable_thinking": False}
    assert "response_format" not in request
    assert request["timeout"].connect == 10.0
    assert request["timeout"].read == 20.0
    call = state["llm_calls"][0]
    assert call["request_options"]["total_timeout_seconds"] == 90.0
    assert call["usage_status"] == "complete"
    assert call["input_tokens_subtotal"] == 11
    assert call["output_tokens_subtotal"] == 4
    assert call["partial_output_chars"] == 0


def test_stream_fast_forwards_optional_json_schema_response_format(offline_llm, monkeypatch):
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "tiny_candidate",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        },
    }
    provider, client = _provider(lambda _kwargs: FakeStream([
        _chunk(content='{"ok":true}', finish_reason="stop"),
        _chunk(usage=_usage()),
    ]))
    _install_providers(monkeypatch, provider)

    with _active_state() as state:
        assert llm.llm_extract(
            "JSON", "request", interaction_policy="stream_fast",
            response_format=response_format,
        ) == {"ok": True}

    assert client.calls[0]["response_format"] == response_format
    assert state["llm_calls"][0]["request_options"]["response_format_type"] == "json_schema"
    assert state["llm_calls"][0]["request_options"]["response_schema_name"] == "tiny_candidate"


def test_legacy_extract_forwards_optional_json_schema_response_format(offline_llm, monkeypatch):
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "tiny_candidate",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        },
    }
    provider, client = _provider(lambda _kwargs: _completion('{"ok":true}', _usage()))
    _install_providers(monkeypatch, provider)

    with _active_state() as state:
        assert llm.llm_extract("JSON", "legacy request") == {"ok": True}
        assert llm.llm_extract("JSON", "request", response_format=response_format) == {"ok": True}

    assert "response_format" not in client.calls[0]
    assert client.calls[1]["response_format"] == response_format
    assert "response_format_type" not in state["llm_calls"][0]["request_options"]
    assert state["llm_calls"][1]["request_options"]["response_format_type"] == "json_schema"


def test_eval_parse_failure_diagnostic_is_captured_only_for_marked_semantic_case(
    offline_llm, monkeypatch,
):
    stream = FakeStream([
        _chunk(content="not JSON", finish_reason="stop"),
        _chunk(usage=_usage()),
    ])
    provider, _client = _provider(lambda _kwargs: stream)
    _install_providers(monkeypatch, provider)

    with _active_state(trace_id="v341-required-011-11") as state:
        state["_semantic_interpreter_eval_raw_output_case_id"] = "v341-required-011-11"
        assert llm.llm_extract(
            "JSON", "request", operation="extract.semantic_interpreter",
            interaction_policy="stream_fast",
        ) is None

    call = state["llm_calls"][0]
    assert call["outcome"] == "parse_failure"
    diagnostic = call["semantic_interpreter_eval_raw_output_diagnostic"]
    assert diagnostic["raw_output_chars"] == len("not JSON")
    assert diagnostic["truncated_text"] == "not JSON"
    assert diagnostic["truncated"] is False

    plain_provider, _plain_client = _provider(
        lambda _kwargs: FakeStream([_chunk(content="not JSON", finish_reason="stop")]),
    )
    _install_providers(monkeypatch, plain_provider)
    with _active_state(trace_id="ordinary-request") as plain_state:
        assert llm.llm_extract(
            "JSON", "request", operation="extract.semantic_interpreter",
            interaction_policy="stream_fast",
        ) is None
    assert "semantic_interpreter_eval_raw_output_diagnostic" not in plain_state["llm_calls"][0]


def test_stream_thinking_counts_reasoning_but_returns_only_content(offline_llm, monkeypatch):
    chunks = [_chunk(reasoning="reasoning-") for _ in range(8)]
    chunks += [
        _chunk(content='{"answer":'),
        _chunk(content='"ok"}', finish_reason="stop"),
        _chunk(usage=_usage()),
    ]
    stream = FakeStream(chunks, interval=0.002)
    provider, client = _provider(lambda _kwargs: stream)
    _install_providers(monkeypatch, provider)

    with _active_state() as state:
        result = llm.llm_extract("JSON", "request", interaction_policy="stream_thinking")

    assert result == {"answer": "ok"}
    assert client.calls[0]["extra_body"] == {"enable_thinking": True}
    assert state["llm_calls"][0]["reasoning_chars"] == 80
    assert state["llm_calls"][0]["first_token_ms"] is not None
    assert "reasoning-" not in repr(state["llm_calls"][0])


def test_retry_before_first_token_retries_once(offline_llm, monkeypatch):
    count = 0

    def handler(_kwargs):
        nonlocal count
        count += 1
        if count == 1:
            raise RuntimeError("offline pre-token failure")
        return FakeStream([_chunk(content="answer", finish_reason="stop"), _chunk(usage=_usage())])

    provider, client = _provider(handler)
    _install_providers(monkeypatch, provider)
    monkeypatch.setattr(llm, "_RECOVERABLE_ERRORS", (RuntimeError,))

    with _active_state() as state:
        assert llm.llm_chat("system", "question", interaction_policy="stream_fast") == "answer"

    assert len(client.calls) == 2
    assert len(state["llm_calls"][0]["attempts"]) == 2


def test_error_after_first_token_does_not_retry_or_fallback(offline_llm, monkeypatch):
    private_content = "PRIVATE_STREAM_FRAGMENT"
    primary, primary_client = _provider(
        lambda _kwargs: FakeStream(
            [_chunk(content=private_content)], failure=RuntimeError("after first token"),
        ),
    )
    backup, backup_client = _provider(
        lambda _kwargs: FakeStream([_chunk(content="backup")]),
        provider_id="backup", model="qwen3.8-flash",
    )
    _install_providers(monkeypatch, primary, backup)
    monkeypatch.setattr(llm, "_RECOVERABLE_ERRORS", (RuntimeError,))

    with _active_state() as state:
        assert llm.llm_chat("system", "question", interaction_policy="stream_fast") == ""

    assert len(primary_client.calls) == 1
    assert backup_client.calls == []
    call = state["llm_calls"][0]
    assert call["partial_output_chars"] == len(private_content)
    assert private_content not in repr(call)


def test_idle_timeout_records_kind_and_partial_length(offline_llm, monkeypatch):
    stream = FakeStream([_chunk(content="part")], block_after=True)
    provider, _client = _provider(lambda _kwargs: stream)
    _install_providers(monkeypatch, provider)
    policy = _short_stream_policy(idle=0.04, total=0.4)

    with _active_state() as state:
        assert llm.llm_chat("system", "question", interaction_policy=policy) == ""

    call = state["llm_calls"][0]
    assert call["timeout_kind"] == "idle"
    assert call["partial_output_chars"] == 4
    assert call["outcome"] == "timeout"


def test_total_timeout_closes_stream_despite_continuing_chunks(offline_llm, monkeypatch):
    stream = InfiniteStream(_chunk(content="x"), interval=0.003)
    provider, client = _provider(lambda _kwargs: stream)
    _install_providers(monkeypatch, provider)
    policy = _short_stream_policy(idle=0.05, total=0.09)

    with _active_state() as state:
        assert llm.llm_chat("system", "question", interaction_policy=policy) == ""

    call = state["llm_calls"][0]
    assert call["timeout_kind"] == "total"
    assert call["partial_output_chars"] > 0
    assert client.closed is True
    assert not any(
        thread.name == "llm-stream-watchdog" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_missing_stream_usage_is_unknown_and_does_not_invent_cost(offline_llm, monkeypatch):
    provider, _client = _provider(
        lambda _kwargs: FakeStream([_chunk(content="answer", finish_reason="stop")]),
    )
    _install_providers(monkeypatch, provider)

    with _active_state() as state:
        assert llm.llm_chat("system", "question", interaction_policy="stream_fast") == "answer"

    call = state["llm_calls"][0]
    assert call["usage_status"] == "unknown"
    assert call["estimated_cost"] is None
    assert call["total_tokens"] is None
    assert call["attempts"][0]["input_tokens"] is None


@pytest.mark.parametrize(
    ("policy_name", "thinking"),
    [("nonstream_fast", False), ("nonstream_thinking", True)],
)
def test_nonstream_modes_set_transport_and_thinking(
    offline_llm, monkeypatch, policy_name, thinking,
):
    provider, client = _provider(lambda _kwargs: _completion("ok", _usage()))
    _install_providers(monkeypatch, provider)

    with _active_state():
        assert llm.llm_chat("system", "question", interaction_policy=policy_name) == "ok"

    request = client.calls[0]
    assert request["stream"] is False
    assert request["extra_body"] == {"enable_thinking": thinking}
    assert request["timeout"].connect == 10.0


@pytest.mark.parametrize("status", [400, 422])
def test_stream_options_error_retries_without_options_and_records_unknown_usage(
    offline_llm, monkeypatch, status,
):
    stream = FakeStream([_chunk(content="answer", finish_reason="stop")])

    def handler(kwargs):
        if "stream_options" in kwargs:
            raise ProviderHTTPError(status, "unsupported stream_options include_usage")
        return stream

    provider, client = _provider(handler)
    _install_providers(monkeypatch, provider)

    with _active_state() as state:
        assert llm.llm_chat("system", "question", interaction_policy="stream_fast") == "answer"

    call = state["llm_calls"][0]
    assert len(client.calls) == 2
    assert "stream_options" in client.calls[0]
    assert "stream_options" not in client.calls[1]
    assert call["stream_options_fallback"] is True
    assert call["request_options"]["stream_options_fallback"] is True
    assert call["usage_status"] == "unknown"


def test_stream_options_fallback_restarts_for_each_provider(offline_llm, monkeypatch):
    def primary_handler(kwargs):
        if "stream_options" in kwargs:
            raise ProviderHTTPError(400, "unsupported stream_options include_usage")
        raise RuntimeError("primary failed before first token")

    def backup_handler(kwargs):
        if "stream_options" in kwargs:
            raise ProviderHTTPError(422, "unsupported stream_options include_usage")
        return FakeStream([
            _chunk(content="backup", finish_reason="stop"),
            _chunk(usage=_usage()),
        ])

    primary, _primary_client = _provider(primary_handler)
    backup, backup_client = _provider(backup_handler, provider_id="backup")
    _install_providers(monkeypatch, primary, backup)
    monkeypatch.setattr(llm, "_RECOVERABLE_ERRORS", (RuntimeError,))
    resilience_policy._BREAKERS["llm"] = resilience_policy.CircuitBreaker(failure_threshold=10)

    with _active_state() as state:
        assert llm.llm_chat("system", "question", interaction_policy="stream_fast") == "backup"

    assert len(backup_client.calls) == 2
    assert "stream_options" in backup_client.calls[0]
    assert "stream_options" not in backup_client.calls[1]
    assert state["llm_calls"][0]["stream_options_fallback"] is True


def test_provider_without_streaming_falls_back_to_nonstream(offline_llm, monkeypatch):
    def handler(kwargs):
        if kwargs.get("stream") is True:
            raise ProviderHTTPError(400, "streaming is not supported")
        return _completion("nonstream fallback", _usage())

    provider, client = _provider(handler)
    _install_providers(monkeypatch, provider)

    with _active_state() as state:
        assert llm.llm_chat("system", "question", interaction_policy="stream_fast") == "nonstream fallback"

    assert len(client.calls) == 2
    assert client.calls[1]["stream"] is False
    assert state["llm_calls"][0]["transport_fallback"] is True
    assert state["llm_calls"][0]["usage_status"] == "partial"


def test_provider_default_thinking_does_not_require_qwen_or_dashscope(offline_llm, monkeypatch):
    provider, client = _provider(
        lambda _kwargs: FakeStream([_chunk(content="ok", finish_reason="stop")]),
        model="other-model",
        base_url="https://provider.invalid/v1",
    )
    _install_providers(monkeypatch, provider)

    with _active_state():
        assert llm.llm_chat(
            "system", "question", interaction_policy=_short_stream_policy(),
        ) == "ok"

    assert "extra_body" not in client.calls[0]


def test_unsupported_backup_is_skipped_and_recorded(offline_llm, monkeypatch):
    primary, primary_client = _provider(
        lambda _kwargs: (_ for _ in ()).throw(RuntimeError("primary offline failure")),
    )
    backup, backup_client = _provider(
        lambda _kwargs: _completion("must not be sent"),
        provider_id="backup", model="other-model", base_url="https://backup.invalid/v1",
    )
    _install_providers(monkeypatch, primary, backup)
    monkeypatch.setattr(llm, "_RECOVERABLE_ERRORS", (RuntimeError,))
    policy = _short_stream_policy(thinking="on", retries=0)

    with _active_state() as state:
        assert llm.llm_chat("system", "question", interaction_policy=policy) == ""

    assert len(primary_client.calls) == 1
    assert backup_client.calls == []
    call = state["llm_calls"][0]
    assert call["policy_unsupported_provider"] == "backup"
    assert call["outcome"] == "provider_error"


def test_all_incompatible_providers_fail_closed_as_configuration_error(
    offline_llm, monkeypatch,
):
    provider, client = _provider(
        lambda _kwargs: _completion("must not be sent"),
        model="other-model", base_url="https://provider.invalid/v1",
    )
    _install_providers(monkeypatch, provider)
    policy = _short_stream_policy(thinking="on", retries=0)

    with _active_state() as state:
        assert llm.llm_chat("system", "question", interaction_policy=policy) == ""

    call = state["llm_calls"][0]
    assert client.calls == []
    assert call["attempts"] == []
    assert call["outcome"] == "configuration_error"
    assert call["not_sent_reason"] == "policy_unsupported_provider"
    assert call["policy_unsupported_provider"] == "primary"


def test_llm_chat_policy_uses_operation_and_quality_for_profile(offline_llm, monkeypatch):
    resolver_calls = []
    provider, client = _provider(
        lambda _kwargs: FakeStream([_chunk(content="report", finish_reason="stop"), _chunk(usage=_usage())]),
    )

    def resolve(operation, explicit, quality):
        resolver_calls.append((operation, explicit, quality))
        return "qwen3.8-flash", "quality" if quality else "standard"

    provider.resolve = resolve
    _install_providers(monkeypatch, provider)

    with _active_state() as state:
        result = llm.llm_chat(
            "system", "report request", operation="report", quality=True,
            interaction_policy="stream_thinking",
        )

    assert result == "report"
    assert resolver_calls == [("report", None, True)]
    assert state["llm_calls"][0]["operation"] == "report"
    assert state["llm_calls"][0]["profile"] == "quality"
    assert client.calls[0]["model"] == "qwen3.8-flash"


def test_llm_chat_default_path_uses_operation_and_quality(offline_llm, monkeypatch):
    _settings_value, recorded_usage = offline_llm
    resolver_calls = []
    provider, _client = _provider(lambda _kwargs: _completion("report", _usage()))

    def resolve(operation, explicit, quality):
        resolver_calls.append((operation, explicit, quality))
        return "qwen3.8-flash", "quality" if quality else "standard"

    provider.resolve = resolve
    _install_providers(monkeypatch, provider)

    with _active_state() as state:
        result = llm.llm_chat("system", "report request", operation="report", quality=True)

    assert result == "report"
    assert resolver_calls == [("report", None, True)]
    assert state["llm_calls"][0]["operation"] == "report"
    assert recorded_usage[0]["operation"] == "report"
    assert recorded_usage[0]["route_profile"] == "quality"


def test_llm_classify_accepts_interaction_policy_and_keeps_allowed_validation(
    offline_llm, monkeypatch,
):
    provider, client = _provider(lambda _kwargs: _completion("route-a", _usage()))
    _install_providers(monkeypatch, provider)

    with _active_state() as state:
        result = llm.llm_classify(
            "Choose the route", "question", ["route-a", "route-b"],
            interaction_policy="nonstream_fast",
        )

    assert result == "route-a"
    assert client.calls[0]["stream"] is False
    assert "- route-a" in client.calls[0]["messages"][0]["content"]
    assert "- route-b" in client.calls[0]["messages"][0]["content"]
    assert state["llm_calls"][0]["operation"] == "classify"
    assert state["llm_calls"][0]["outcome"] == "success"


def test_llm_classify_custom_operation_is_used_by_policy_path(offline_llm, monkeypatch):
    provider, client = _provider(lambda _kwargs: _completion("route-a", _usage()))
    _install_providers(monkeypatch, provider)

    with _active_state() as state:
        result = llm.llm_classify(
            "Choose the route", "question", ["route-a", "route-b"],
            operation="classify.task_type",
            interaction_policy="nonstream_fast",
        )

    assert result == "route-a"
    assert client.calls[0]["stream"] is False
    assert state["llm_calls"][0]["operation"] == "classify.task_type"


def test_llm_classify_custom_operation_is_used_by_legacy_path(offline_llm, monkeypatch, capsys):
    _settings_value, recorded_usage = offline_llm
    resolver_calls = []
    provider, _client = _provider(lambda _kwargs: _completion("route-a", _usage()))

    def resolve(operation, explicit, quality):
        resolver_calls.append((operation, explicit, quality))
        return "qwen3.8-flash", "fast"

    provider.resolve = resolve
    _install_providers(monkeypatch, provider)

    with _active_state() as state:
        result = llm.llm_classify(
            "Choose the route", "question", ["route-a", "route-b"],
            operation="classify.task_type",
        )

    assert result == "route-a"
    assert resolver_calls == [("classify.task_type", None, False)]
    assert state["llm_calls"][0]["operation"] == "classify.task_type"
    assert recorded_usage[0]["operation"] == "classify.task_type"
    assert "classify.task_type → 'route-a'" in capsys.readouterr().out


def test_llm_classify_policy_rejects_values_outside_allowed(offline_llm, monkeypatch):
    provider, _client = _provider(lambda _kwargs: _completion("invented-route", _usage()))
    _install_providers(monkeypatch, provider)

    with _active_state() as state:
        result = llm.llm_classify(
            "Choose the route", "question", ["route-a", "route-b"],
            interaction_policy="nonstream_fast",
        )

    assert result == ""
    assert state["llm_calls"][0]["outcome"] == "parse_failure"
    assert state["llm_calls"][0]["attempts"][-1]["processing_result"] == "parse_failure"


def test_llm_agent_action_accepts_interaction_policy(offline_llm, monkeypatch):
    provider, client = _provider(
        lambda _kwargs: _completion('{"action":"stop"}', _usage()),
    )
    _install_providers(monkeypatch, provider)

    with _active_state() as state:
        result = llm.llm_agent_action(
            "Return an action object", "decide", interaction_policy="nonstream_fast",
        )

    assert result == {"action": "stop"}
    assert client.calls[0]["stream"] is False
    assert state["llm_calls"][0]["outcome"] == "success"
    assert state["llm_calls"][0]["request_options"]["interaction_policy"] == "nonstream_fast"


def test_llm_agent_action_default_keeps_legacy_extract_kwargs(monkeypatch):
    calls = []

    def fake_extract(system_prompt, user_message, **kwargs):
        calls.append((system_prompt, user_message, kwargs))
        return {"action": "stop"}

    monkeypatch.setattr(llm, "llm_extract", fake_extract)

    assert llm.llm_agent_action("system", "question") == {"action": "stop"}
    assert calls == [(
        "system", "question",
        {"model": None, "quality": False, "max_tokens": 4096},
    )]


def test_critic_extract_calls_use_supported_max_tokens_only(monkeypatch):
    from app.agents import critic_agent

    settings = SimpleNamespace(
        llm_extract_max_tokens=256,
        critic_max_output_tokens=1024,
    )
    monkeypatch.setattr(critic_agent, "get_settings", lambda: settings)
    monkeypatch.setattr(critic_agent, "model_for", lambda *_args, **_kwargs: ("offline-model", "fast"))
    monkeypatch.setattr(
        critic_agent, "pack_for_llm",
        lambda *_args, **_kwargs: SimpleNamespace(text="packed critic input"),
    )
    extract_kwargs = []

    def fake_extract(_system, _user, **kwargs):
        extract_kwargs.append(kwargs)
        return {"verdict": "passed", "issues": []}

    monkeypatch.setattr(critic_agent, "llm_extract", fake_extract)
    critic_agent.llm_review_groundedness("answer", "legacy evidence", budget_optional=True)
    critic_agent.llm_review_evidence_completeness("answer", "task", budget_optional=True)
    critic_agent.llm_review_logic_consistency("answer", "task", budget_optional=True)
    critic_agent.llm_review_combined_experiment(
        "answer", "legacy evidence", "task", "question", budget_optional=True,
    )

    assert [kwargs["max_tokens"] for kwargs in extract_kwargs] == [1024, 1024, 1024, 1024]
    assert all("_max_output_tokens" not in kwargs for kwargs in extract_kwargs)
    assert all("_soft_limit_optional" not in kwargs for kwargs in extract_kwargs)


def test_critic_output_budget_is_configurable_and_independent(monkeypatch):
    monkeypatch.delenv("CRITIC_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("LLM_EXTRACT_MAX_TOKENS", raising=False)

    settings = Settings(_env_file=None)

    assert settings.critic_max_output_tokens == 1024
    assert settings.llm_extract_max_tokens == 8192
