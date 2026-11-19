"""Request-scoped LLM SDK attempt and token-availability accounting."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from app.core.execution_context import activate_state, reset_state
from app.llm import client as llm


def _response(content, usage_marker="default"):
    if usage_marker == "missing":
        usage = None
    elif usage_marker == "zero":
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0)
    elif isinstance(usage_marker, tuple):
        usage = SimpleNamespace(prompt_tokens=usage_marker[0], completion_tokens=usage_marker[1])
    else:
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=2)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=usage,
    )


class _ProviderClient:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responder(**kwargs)


def _provider(provider_id, client, model):
    return llm._Provider(provider_id, client, lambda operation, explicit, quality: (model, "standard"))


def _state(trace_id="trace-test", node="triage", max_tokens=1000):
    return {
        "tenant_id": "tenant-test", "user_id": "user-test", "trace_id": trace_id,
        "current_agent": node, "runtime_budget": {"max_tokens": max_tokens, "max_cost": 0},
        "errors": [],
    }


def _invoke_in_state(state, callback):
    token = activate_state(state)
    try:
        return callback()
    finally:
        reset_state(token)


def _install(monkeypatch, providers):
    monkeypatch.setattr(llm, "_providers", lambda: providers)
    monkeypatch.setattr(llm, "_quota_available", lambda: True)
    # Keep the public llm_* methods and SDK-boundary recorders real; fake only
    # the provider response and the resilience wrapper's execution behavior.
    monkeypatch.setattr(llm, "execute_with_resilience", lambda operation, *a, **k: operation())
    monkeypatch.setattr(llm, "record_usage", lambda **kwargs: None)


def test_public_chat_classify_extract_record_known_missing_and_zero_usage(monkeypatch):
    responses = iter([
        _response("hello", (7, 3)),
        _response("route-a", "zero"),
        _response('{"metric":"gmv"}', "missing"),
    ])
    fake = _ProviderClient(lambda **kwargs: next(responses))
    _install(monkeypatch, [_provider("offline", fake, "offline-model")])
    state = _state()

    def calls():
        assert llm.llm_chat("system", "chat-input") == "hello"
        assert llm.llm_classify("system", "classify-input", ["route-a", "route-b"]) == "route-a"
        assert llm.llm_extract("system", "extract-input") == {"metric": "gmv"}

    _invoke_in_state(state, calls)
    logical = state["llm_calls"]
    assert len(logical) == 3
    assert len({item["call_id"] for item in logical}) == 3
    assert [item["operation"] for item in logical] == ["chat", "classify", "extract"]
    assert all(item["trace_id"] == "trace-test" and item["node"] == "triage" for item in logical)
    assert all(item["attempts"][0]["call_id"] == item["call_id"] for item in logical)
    assert all(item["attempts"][0]["provider"] == "offline" for item in logical)
    assert all(item["attempts"][0]["model"] == "offline-model" for item in logical)
    assert all(item["attempts"][0]["attempt_order"] == 1 for item in logical)

    assert logical[0]["attempts"][0]["input_tokens"] == 7
    assert logical[0]["attempts"][0]["output_tokens"] == 3
    assert logical[0]["usage_complete"] is True and logical[0]["total_tokens"] == 10
    assert logical[1]["attempts"][0]["input_tokens"] == 0
    assert logical[1]["attempts"][0]["output_tokens"] == 0
    assert logical[1]["usage_complete"] is True and logical[1]["total_tokens"] == 0
    assert logical[2]["attempts"][0]["input_tokens"] is None
    assert logical[2]["attempts"][0]["output_tokens"] is None
    assert logical[2]["usage_complete"] is False and logical[2]["total_tokens"] is None

    summary = state["llm_token_summary"]
    assert summary["request_status"] == "requested"
    assert summary["logical_call_count"] == 3 and summary["sdk_attempt_count"] == 3
    assert summary["known_input_tokens_subtotal"] == 7
    assert summary["known_output_tokens_subtotal"] == 3
    assert summary["known_tokens_subtotal"] == 10
    assert summary["input_tokens"] is None and summary["output_tokens"] is None
    assert summary["total_tokens"] is None and summary["usage_complete"] is False
    assert summary["unknown_usage_attempt_count"] == 1
    serialized = json.dumps({"calls": logical, "summary": summary}, ensure_ascii=False)
    assert "chat-input" not in serialized and "classify-input" not in serialized and "extract-input" not in serialized


def test_extract_operation_is_overridable_and_default_remains_extract(monkeypatch):
    from app.core.config import get_settings

    fake = _ProviderClient(lambda **kwargs: _response('{"ok":true}', (5, 2)))
    resolved = []
    profiles = []

    def resolve(operation, explicit, quality):
        resolved.append(operation)
        profiles.append(llm.profile_for(operation, quality=quality))
        return "offline-model", profiles[-1]

    provider = llm._Provider("offline", fake, resolve)
    _install(monkeypatch, [provider])
    state = _state("trace-extract-operation")

    def calls():
        assert llm.llm_extract("system", "default") == {"ok": True}
        assert llm.llm_extract(
            "system", "semantic", operation="semantic_interpreter", max_tokens=768,
        ) == {"ok": True}

    _invoke_in_state(state, calls)

    assert resolved == ["extract", "semantic_interpreter"]
    assert profiles == ["fast", "standard"]
    assert [item["operation"] for item in state["llm_calls"]] == resolved
    assert fake.calls[0]["max_tokens"] == get_settings().llm_extract_max_tokens
    assert fake.calls[1]["max_tokens"] == 768


def test_evaluation_journal_is_flushed_before_sdk_and_correlates_usage(monkeypatch):
    journal = []
    response = _response('{"ok":true}', (9, 4))
    fake = _ProviderClient(lambda **kwargs: response)
    _install(monkeypatch, [_provider("primary", fake, "offline-model")])
    state = _state("trace-journal", "critic")
    state["evaluation_run_id"] = "run-journal"
    state["_evaluation_event_sink"] = journal.append
    fake.responder = lambda **kwargs: (
        (_ for _ in ()).throw(AssertionError("SDK entered before durable start event"))
        if not any(event["event"] == "sdk_invocation_started" for event in journal)
        else response
    )

    assert _invoke_in_state(state, lambda: llm.llm_extract("secret system", "secret payload")) == {"ok": True}
    events = [event["event"] for event in journal]
    assert events == ["sdk_invocation_started", "sdk_invocation_finished", "usage_recorded", "logical_call_finished"]
    start, finish, usage, logical = journal
    assert start["run_id"] == "run-journal" and start["trace_id"] == "trace-journal"
    assert start["call_id"] == finish["call_id"] == usage["call_id"] == logical["call_id"]
    assert start["attempt_id"] == finish["attempt_id"] == usage["attempt_id"]
    assert usage["input_tokens"] == 9 and usage["output_tokens"] == 4 and usage["usage_status"] == "complete"
    serialized = json.dumps(journal, ensure_ascii=False)
    assert "secret system" not in serialized and "secret payload" not in serialized


def test_evaluation_journal_failure_refuses_before_sdk(monkeypatch):
    fake = _ProviderClient(lambda **kwargs: _response('{"ok":true}'))
    _install(monkeypatch, [_provider("primary", fake, "offline-model")])
    state = _state("trace-journal-fail")
    state["_evaluation_event_sink"] = lambda _event: (_ for _ in ()).throw(OSError("disk unavailable"))

    assert _invoke_in_state(state, lambda: llm.llm_extract("system", "payload")) is None
    assert fake.calls == []
    attempt = state["llm_calls"][0]["attempts"][0]
    assert attempt["result"] == "locally_refused"
    assert attempt["sdk_request_sent"] is False
    assert attempt["local_refusal_reason"] == "evaluation_journal_unavailable"
    assert state["llm_token_summary"]["sdk_attempt_count"] == 0
    assert state["llm_token_summary"]["local_send_refusal_count"] == 1


def test_evaluation_journal_distinguishes_provider_error_from_local_refusal(monkeypatch):
    fake = _ProviderClient(lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline provider failure")))
    _install(monkeypatch, [_provider("primary", fake, "offline-model")])
    journal = []
    state = _state("trace-provider-error")
    state["evaluation_run_id"] = "run-provider-error"
    state["_evaluation_event_sink"] = journal.append

    assert _invoke_in_state(state, lambda: llm.llm_extract("system", "payload")) is None
    started = next(event for event in journal if event["event"] == "sdk_invocation_started")
    finished = next(event for event in journal if event["event"] == "sdk_invocation_finished")
    assert started["attempt_id"] == finished["attempt_id"]
    assert finished["result"] == "provider_exception" and finished["sdk_request_sent"] is True
    assert finished.get("local_refusal_reason") is None
    assert state["llm_token_summary"]["sdk_attempt_count"] == 1
    assert state["llm_token_summary"]["total_tokens"] is None


def test_provider_exception_and_backup_attempts_are_correlated(monkeypatch, tmp_path):
    def fail(**kwargs):
        raise RuntimeError("offline primary failure")

    primary = _ProviderClient(fail)
    backup = _ProviderClient(lambda **kwargs: _response("backup reply", (4, 2)))
    _install(monkeypatch, [
        _provider("primary", primary, "primary-model"),
        _provider("backup", backup, "backup-model"),
    ])
    monkeypatch.setattr(llm, "_RECOVERABLE_ERRORS", (RuntimeError,))
    state = _state("trace-fallback", "generate_answer")

    assert _invoke_in_state(state, lambda: llm.llm_chat("private system prompt", "private user prompt")) == "backup reply"
    call = state["llm_calls"][0]
    assert call["outcome"] == "success" and call["operation"] == "chat"
    assert [attempt["attempt_order"] for attempt in call["attempts"]] == [1, 2]
    assert [attempt["provider"] for attempt in call["attempts"]] == ["primary", "backup"]
    assert [attempt["model"] for attempt in call["attempts"]] == ["primary-model", "backup-model"]
    assert call["attempts"][0]["result"] == "provider_exception"
    assert call["attempts"][0]["exception_type"] == "RuntimeError"
    assert call["attempts"][1]["result"] == "response_received"
    assert call["attempts"][1]["processing_result"] == "success"
    assert all(attempt["call_id"] == call["call_id"] for attempt in call["attempts"])
    assert all(attempt["trace_id"] == "trace-fallback" and attempt["node"] == "generate_answer" for attempt in call["attempts"])
    summary = state["llm_token_summary"]
    assert summary["sdk_attempt_count"] == 2
    assert summary["provider_exception_count"] == 1 and summary["successful_response_count"] == 1
    assert summary["known_tokens_subtotal"] == 6
    assert summary["total_tokens"] is None and summary["usage_complete"] is False
    assert summary["unknown_usage_attempt_count"] == 1

    sample_dir = tmp_path / "llm_call_recording"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "sample_trace.json").write_text(
        json.dumps({"trace_id": state["trace_id"], "llm_calls": state["llm_calls"],
                    "llm_token_summary": state["llm_token_summary"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_extract_parse_failure_is_distinct_from_provider_failure(monkeypatch):
    fake = _ProviderClient(lambda **kwargs: _response("not valid JSON", (11, 4)))
    _install(monkeypatch, [_provider("offline", fake, "extract-model")])
    state = _state("trace-parse", "data_query")

    assert _invoke_in_state(state, lambda: llm.llm_extract("system", "payload")) is None
    call = state["llm_calls"][0]
    attempt = call["attempts"][0]
    assert call["outcome"] == "parse_failure"
    assert attempt["result"] == "response_received"
    assert attempt["processing_result"] == "parse_failure"
    assert attempt["exception_type"] is None
    assert call["usage_complete"] is True and call["total_tokens"] == 15
    assert state["llm_token_summary"]["parse_failure_count"] == 1
    assert state["llm_token_summary"]["provider_exception_count"] == 0


def test_no_provider_request_is_distinct_from_unknown_usage(monkeypatch):
    from app.graph.workflow import init_state

    untouched = init_state({"runtime_budget": {}})
    assert untouched["llm_token_summary"]["request_status"] == "no_model_request"
    assert untouched["llm_token_summary"]["sdk_attempt_count"] == 0
    assert untouched["llm_token_summary"]["total_tokens"] == 0
    assert untouched["semantic_interpreter_observation"] == {"mode": "off", "status": "not_run"}

    _install(monkeypatch, [])
    state = _state("trace-no-request")
    assert _invoke_in_state(state, lambda: llm.llm_chat("system", "question")) == ""
    call = state["llm_calls"][0]
    summary = state["llm_token_summary"]
    assert call["outcome"] == "not_sent" and call["not_sent_reason"] == "no_provider_configured"
    assert call["attempts"] == []
    assert summary["request_status"] == "not_sent" and summary["sdk_attempt_count"] == 0
    assert summary["total_tokens"] == 0 and summary["usage_complete"] is True

    fake = _ProviderClient(lambda **kwargs: _response("response without usage", "missing"))
    _install(monkeypatch, [_provider("offline", fake, "model")])
    requested = _state("trace-unknown-usage")
    _invoke_in_state(requested, lambda: llm.llm_chat("system", "question"))
    unknown = requested["llm_token_summary"]
    assert unknown["request_status"] == "requested" and unknown["sdk_attempt_count"] == 1
    assert unknown["total_tokens"] is None and unknown["usage_complete"] is False


def test_request_context_isolation_under_concurrent_calls(monkeypatch):
    def respond(**kwargs):
        user_text = kwargs["messages"][1]["content"]
        usage = (1, 2) if user_text == "request-A" else (10, 20)
        return _response(f"reply-{user_text}", usage)

    fake = _ProviderClient(respond)
    _install(monkeypatch, [_provider("offline", fake, "concurrent-model")])
    states = {"A": _state("trace-A", "triage"), "B": _state("trace-B", "critic")}

    def task(name):
        return _invoke_in_state(states[name], lambda: llm.llm_chat("system", f"request-{name}"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(task, ("A", "B")))
    assert answers == ["reply-request-A", "reply-request-B"]
    a_call, b_call = states["A"]["llm_calls"][0], states["B"]["llm_calls"][0]
    assert a_call["call_id"] != b_call["call_id"]
    assert a_call["trace_id"] == "trace-A" and a_call["node"] == "triage"
    assert b_call["trace_id"] == "trace-B" and b_call["node"] == "critic"
    assert states["A"]["llm_token_summary"]["total_tokens"] == 3
    assert states["B"]["llm_token_summary"]["total_tokens"] == 30
    assert states["A"]["llm_token_summary"]["logical_call_count"] == 1
    assert states["B"]["llm_token_summary"]["logical_call_count"] == 1


def test_existing_token_budget_counter_and_failure_behavior_remain(monkeypatch):
    fake = _ProviderClient(lambda **kwargs: _response("too many", (4, 3)))
    _install(monkeypatch, [_provider("offline", fake, "budget-model")])
    state = _state("trace-budget", "triage", max_tokens=5)

    assert _invoke_in_state(state, lambda: llm.llm_chat("system", "question")) == ""
    assert state["llm_usage"]["input_tokens"] == 4
    assert state["llm_usage"]["output_tokens"] == 3
    assert any(item["code"] == "token_budget_exceeded" for item in state["errors"])
    call = state["llm_calls"][0]
    assert call["outcome"] == "processing_error"
    assert call["attempts"][0]["result"] == "response_received"
    assert call["attempts"][0]["processing_result"] == "processing_error"
    assert state["llm_token_summary"]["total_tokens"] == 7
    assert state["llm_token_summary"]["usage_complete"] is True


def test_over_budget_response_is_kept_and_all_later_public_calls_are_not_sent(monkeypatch):
    real_resilience = llm.execute_with_resilience
    recorded_usage = []
    primary = _ProviderClient(lambda **kwargs: _response("first response", (4, 3)))
    backup = _ProviderClient(lambda **kwargs: _response("backup", (1, 1)))
    _install(monkeypatch, [
        _provider("primary", primary, "primary-model"),
        _provider("backup", backup, "backup-model"),
    ])
    monkeypatch.setattr(llm, "record_usage", lambda **kwargs: recorded_usage.append(kwargs))
    # Exercise the real application retry wrapper. The over-budget exception
    # is raised after a successful SDK response, so it must not re-enter it.
    monkeypatch.setattr(llm, "execute_with_resilience", real_resilience)
    monkeypatch.setattr(llm, "_RECOVERABLE_ERRORS", (RuntimeError,))
    state = _state("trace-budget-stop", "diagnostic_plan", max_tokens=5)

    def calls():
        assert llm.llm_chat("system", "first") == ""
        assert llm.llm_classify("system", "second", ["yes", "no"]) == ""
        assert llm.llm_extract("system", "third") is None
        assert llm.llm_chat("system", "fourth") == ""

    _invoke_in_state(state, calls)

    assert len(primary.calls) == 1
    assert backup.calls == []
    assert len(state["llm_calls"]) == 4
    first, *blocked = state["llm_calls"]
    assert first["operation"] == "chat"
    assert first["attempts"][0]["result"] == "response_received"
    assert first["attempts"][0]["input_tokens"] == 4
    assert first["attempts"][0]["output_tokens"] == 3
    assert first["outcome"] == "processing_error"
    assert [call["operation"] for call in blocked] == ["classify", "extract", "chat"]
    assert all(call["outcome"] == "not_sent" for call in blocked)
    assert all(call["not_sent_reason"] == "budget_exhausted" for call in blocked)
    assert all(call["attempts"] == [] and call["total_tokens"] == 0 for call in blocked)
    assert state["llm_usage"] == {
        "input_tokens": 4,
        "output_tokens": 3,
        "estimated_cost": state["llm_usage"]["estimated_cost"],
    }
    assert [error["code"] for error in state["errors"]].count("token_budget_exceeded") == 1
    summary = state["llm_token_summary"]
    assert summary["sdk_attempt_count"] == 1
    assert summary["provider_exception_count"] == 0
    assert summary["unknown_usage_attempt_count"] == 0
    assert summary["known_tokens_subtotal"] == 7
    assert summary["total_tokens"] == 7 and summary["usage_complete"] is True
    assert len(recorded_usage) == 1
    assert recorded_usage[0]["input_tokens"] == 4
    assert recorded_usage[0]["output_tokens"] == 3


def test_exact_token_limit_allows_current_response_but_blocks_next_request(monkeypatch):
    fake = _ProviderClient(lambda **kwargs: _response("at limit", (3, 2)))
    _install(monkeypatch, [_provider("offline", fake, "limit-model")])
    state = _state("trace-exact-token-limit", max_tokens=5)

    def calls():
        assert llm.llm_chat("system", "reaches limit") == "at limit"
        assert llm.llm_extract("system", "after limit") is None

    _invoke_in_state(state, calls)

    assert len(fake.calls) == 1
    assert state["llm_calls"][0]["outcome"] == "success"
    assert state["llm_calls"][0]["total_tokens"] == 5
    assert state["llm_calls"][1]["outcome"] == "not_sent"
    assert state["llm_calls"][1]["not_sent_reason"] == "budget_exhausted"
    assert state["llm_calls"][1]["attempts"] == []
    assert any(error["code"] == "token_budget_exceeded" for error in state["errors"])
    assert state["llm_token_summary"]["total_tokens"] == 5


def test_already_exhausted_budget_blocks_each_public_operation_without_unknown_usage(monkeypatch):
    fake = _ProviderClient(lambda **kwargs: _response("must not arrive", (1, 1)))
    _install(monkeypatch, [_provider("offline", fake, "blocked-model")])
    state = _state("trace-pre-exhausted", max_tokens=5)
    state["llm_usage"] = {"input_tokens": 2, "output_tokens": 3, "estimated_cost": 0.0}

    def calls():
        assert llm.llm_classify("system", "classify", ["yes"]) == ""
        assert llm.llm_extract("system", "extract") is None
        assert llm.llm_chat("system", "chat") == ""

    _invoke_in_state(state, calls)

    assert fake.calls == []
    assert [call["operation"] for call in state["llm_calls"]] == ["classify", "extract", "chat"]
    assert all(call["outcome"] == "not_sent" for call in state["llm_calls"])
    assert all(call["not_sent_reason"] == "budget_exhausted" for call in state["llm_calls"])
    assert all(call["attempts"] == [] for call in state["llm_calls"])
    assert state["llm_token_summary"]["sdk_attempt_count"] == 0
    assert state["llm_token_summary"]["unknown_usage_attempt_count"] == 0
    assert state["llm_token_summary"]["total_tokens"] == 0
    assert state["llm_token_summary"]["usage_complete"] is True
    assert state["llm_token_summary"]["request_status"] == "not_sent"


def test_unlimited_and_remaining_budget_requests_keep_original_behavior(monkeypatch):
    fake = _ProviderClient(lambda **kwargs: _response("ok", (2, 1)))
    _install(monkeypatch, [_provider("offline", fake, "control-model")])
    limited = _state("trace-budget-remaining", max_tokens=100)
    unlimited = _state("trace-budget-unlimited", max_tokens=0)

    _invoke_in_state(limited, lambda: (llm.llm_chat("s", "a"), llm.llm_chat("s", "b")))
    _invoke_in_state(unlimited, lambda: (llm.llm_chat("s", "a"), llm.llm_chat("s", "b")))

    assert len(fake.calls) == 4
    assert all(call["outcome"] == "success" for call in limited["llm_calls"] + unlimited["llm_calls"])
    assert not limited["errors"] and not unlimited["errors"]
    assert limited["llm_token_summary"]["total_tokens"] == 6
    assert unlimited["llm_token_summary"]["total_tokens"] == 6


def test_exhausted_budget_state_is_isolated_between_requests(monkeypatch):
    fake = _ProviderClient(lambda **kwargs: _response("healthy request", (2, 1)))
    _install(monkeypatch, [_provider("offline", fake, "isolated-model")])
    exhausted = _state("trace-exhausted-request", max_tokens=5)
    exhausted["llm_usage"] = {"input_tokens": 4, "output_tokens": 1, "estimated_cost": 0.0}
    healthy = _state("trace-healthy-request", max_tokens=20)

    def invoke(name):
        state = exhausted if name == "exhausted" else healthy
        return _invoke_in_state(state, lambda: llm.llm_chat("system", name))

    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(invoke, ("exhausted", "healthy")))

    assert answers == ["", "healthy request"]
    assert len(fake.calls) == 1
    assert exhausted["llm_calls"][0]["outcome"] == "not_sent"
    assert exhausted["llm_calls"][0]["attempts"] == []
    assert exhausted["errors"][0]["code"] == "token_budget_exceeded"
    assert healthy["llm_calls"][0]["outcome"] == "success"
    assert healthy["llm_calls"][0]["attempts"][0]["trace_id"] == "trace-healthy-request"
    assert not healthy["errors"]


def test_call_records_survive_runtime_node_and_final_trace(monkeypatch):
    from app.graph.runtime import instrument_node
    from app.graph import workflow

    fake = _ProviderClient(lambda **kwargs: _response("trace-visible", (3, 1)))
    _install(monkeypatch, [_provider("offline", fake, "trace-model")])
    state = workflow.init_state({
        "trace_id": "trace-persisted-call", "tenant_id": "tenant-test", "user_id": "user-test",
        "user_message": "offline trace assertion", "runtime_budget": {"max_tokens": 20, "max_cost": 0},
        "errors": [],
    })

    def runtime_probe(current):
        current["current_agent"] = "trace_probe"
        current["probe_reply"] = llm.llm_chat("offline system", "offline message")
        return current

    completed = instrument_node("trace_probe", runtime_probe)(state)
    persisted = {}
    monkeypatch.setattr(workflow, "write_trace", lambda trace_id, payload: persisted.update(payload))
    workflow.persist_trace(completed)

    assert completed["probe_reply"] == "trace-visible"
    assert persisted["llm_calls"][0]["trace_id"] == "trace-persisted-call"
    assert persisted["llm_calls"][0]["node"] == "trace_probe"
    assert persisted["llm_token_summary"]["input_tokens"] == 3
    assert persisted["llm_token_summary"]["output_tokens"] == 1
    assert persisted["llm_token_summary"]["total_tokens"] == 4
    assert "offline message" not in json.dumps(persisted, ensure_ascii=False)
