from __future__ import annotations

import pytest

from app.core.execution_context import activate_state, reset_state, token_budget_is_exhausted
from app.llm import client as llm


def test_preflight_reserves_estimated_prompt_margin_and_output_cap(monkeypatch):
    monkeypatch.setattr("app.context.packer.estimate_tokens", lambda _text, _model: 50)
    state = {
        "runtime_budget": {"max_tokens": 180},
        "llm_usage": {"input_tokens": 10, "output_tokens": 0},
        "errors": [],
    }
    call = {"call_id": "call-preflight-denied", "operation": "extract.test"}
    kwargs = {
        "messages": [{"role": "user", "content": "private synthetic prompt"}],
        "max_tokens": 100,
    }

    with pytest.raises(llm.LocalSDKSendRefused) as refused:
        llm._preflight_request_token_budget(call, state, "test-model", kwargs)

    assert refused.value.reason_code == "token_budget_preflight"
    assert call["token_budget_preflight"] == {
        "allowed": False,
        "estimated_input_tokens": 50,
        "input_margin_tokens": 32,
        "reserved_output_tokens": 100,
        "required_tokens": 182,
        "remaining_tokens": 170,
        "budget_limit_tokens": 180,
        "prior_failed_attempt_reserve_tokens": 0,
    }
    assert "private synthetic prompt" not in repr(call)
    assert state["errors"][-1]["code"] == "token_budget_exceeded"


def test_preflight_allows_request_when_full_reservation_fits(monkeypatch):
    monkeypatch.setattr("app.context.packer.estimate_tokens", lambda _text, _model: 50)
    state = {
        "runtime_budget": {"max_tokens": 300},
        "llm_usage": {"input_tokens": 10, "output_tokens": 0},
        "errors": [],
    }
    call = {"call_id": "call-preflight-allowed", "operation": "extract.test"}
    kwargs = {
        "messages": [{"role": "user", "content": "synthetic prompt"}],
        "max_completion_tokens": 100,
    }

    llm._preflight_request_token_budget(call, state, "test-model", kwargs)

    assert call["token_budget_preflight"]["allowed"] is True
    assert call["token_budget_preflight"]["required_tokens"] == 182
    assert call["token_budget_preflight"]["reserved_output_tokens"] == 100
    assert state["errors"] == []


def test_preflight_is_unbounded_when_turn_budget_is_zero(monkeypatch):
    def unexpected_estimate(*_args, **_kwargs):
        pytest.fail("unlimited requests should not need token estimation")

    monkeypatch.setattr("app.context.packer.estimate_tokens", unexpected_estimate)
    state = {"runtime_budget": {"max_tokens": 0}, "llm_usage": {}, "errors": []}
    call = {"call_id": "call-preflight-unlimited"}

    llm._preflight_request_token_budget(
        call, state, "test-model", {"messages": [], "max_tokens": 10},
    )

    assert "token_budget_preflight" not in call


def test_preflight_accounts_for_unknown_usage_reserved_by_failed_provider_attempt(monkeypatch):
    monkeypatch.setattr("app.context.packer.estimate_tokens", lambda _text, _model: 20)
    state = {
        "runtime_budget": {"max_tokens": 200},
        "llm_usage": {
            "input_tokens": 10, "output_tokens": 0,
            "preflight_reserved_tokens": 50,
        },
        "errors": [],
    }
    call = {"call_id": "call-after-provider-failure", "operation": "extract.test"}

    with pytest.raises(llm.LocalSDKSendRefused) as refused:
        llm._preflight_request_token_budget(
            call,
            state,
            "test-model",
            {"messages": [], "max_tokens": 120},
        )

    assert refused.value.reason_code == "token_budget_preflight"
    assert call["token_budget_preflight"]["remaining_tokens"] == 140
    assert call["token_budget_preflight"]["prior_failed_attempt_reserve_tokens"] == 0


def test_failed_attempt_reservation_is_committed_without_counting_as_observed_usage():
    state = {
        "trace_id": "trace-unknown-usage-reserve",
        "runtime_budget": {"max_tokens": 100},
        "llm_usage": {"input_tokens": 5, "output_tokens": 5, "estimated_cost": 0.0},
        "errors": [],
        "llm_calls": [],
    }
    token = activate_state(state)
    try:
        call, active = llm._begin_logical_call("extract.test", None)
        call["unknown_usage_reserved_tokens"] = 90
        llm._finish_logical_call(call, active, "provider_error")
    finally:
        reset_state(token)

    assert state["llm_usage"]["input_tokens"] == 5
    assert state["llm_usage"]["output_tokens"] == 5
    assert state["llm_usage"]["preflight_reserved_tokens"] == 90
    assert token_budget_is_exhausted(state) is True
    assert "_preflight_reserved_accounted" not in state["llm_calls"][0]
