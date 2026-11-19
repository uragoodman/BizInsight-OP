from __future__ import annotations

from copy import deepcopy

import pytest

from app.context import semantic_verifier as verifier


QUESTION = "按地区范围的 GMV"
MENTIONS = [{"id": "m1", "raw_text": "地区", "start": 1, "end": 3, "canonical": "customer_state"}]
PROPOSALS = [{"id": "m1", "role": "reference", "start": 1, "end": 3}]
VALID_OUTPUT = {
    "roles": [{
        "id": "m1", "start": 1, "end": 3,
        "role": "reference", "reason_code": "scope_reference",
    }],
    "abstain": False,
}


def _run(invoke=None, **overrides):
    clear_cache = overrides.pop("clear_cache", True)
    if clear_cache:
        verifier._CACHE.clear()
    args = {
        "question": QUESTION,
        "mentions": MENTIONS,
        "proposed_roles": PROPOSALS,
        "eligible_ids": ["m1"],
        "risk_tier": "open",
        "mode": "shadow",
        "model": "test-model",
        "allow_raw_question": True,
        "invoke_model": invoke or (lambda *_: deepcopy(VALID_OUTPUT)),
    }
    args.update(overrides)
    return verifier.verify_roles(**args)


def test_off_mode_does_not_call_verifier():
    calls = []
    record = _run(lambda *args: calls.append(args), mode="off")
    assert calls == []
    assert record["status"] == "skipped"
    assert record["reason_code"] == "mode_off"
    assert record["applied"] is False


def test_shadow_mode_is_observation_only():
    record = _run()
    assert record["status"] == "success"
    assert record["verifier_roles"] == {"m1": "reference"}
    assert record["applied"] is False


def test_success_payload_is_minimal_and_reason_codes_are_retained():
    requests = []

    def invoke(_system, request, _model):
        requests.append(request)
        return deepcopy(VALID_OUTPUT)

    record = _run(
        invoke,
        proposed_roles=[{
            "id": "m1", "role": "reference", "start": 1, "end": 3,
            "evidence": ("reference_suffix",),
        }],
    )

    assert record["status"] == "success"
    assert record["verifier_reason_codes"] == {"m1": "scope_reference"}
    assert len(requests) == 1
    request = requests[0]
    assert set(request) == {
        "mentions", "proposed_roles", "eligible_mention_ids", "allowed_roles", "reason_codes",
    }
    assert "question" not in request
    assert not {
        "gold", "gold_roles", "scope", "seller", "seller_id", "seller_ids",
        "date", "metric", "metric_id", "sql",
    } & set(request)
    assert request["proposed_roles"][0]["evidence"] == ("reference_suffix",)


def test_proposals_must_be_unique_subset_and_span_bound_to_mentions():
    invalid_proposals = [
        # Unknown proposal IDs cannot create candidate mentions.
        PROPOSALS + [{"id": "m999", "role": "grouping", "start": 1, "end": 3}],
        # Duplicate proposal IDs are ambiguous even if their labels match.
        PROPOSALS + [deepcopy(PROPOSALS[0])],
        # Existing IDs cannot be rebound to a different span.
        [{"id": "m1", "role": "reference", "start": 0, "end": 2}],
        # Evidence accepts closed codes only, never arbitrary question fragments.
        [{**PROPOSALS[0], "evidence": ["reference_suffix:secret phrase"]}],
        # Every eligible ID must have a proposal.
        [],
    ]
    for proposals in invalid_proposals:
        calls = []
        record = _run(lambda *args: calls.append(args), proposed_roles=proposals)
        assert record["status"] == "invalid_output"
        assert calls == []

    duplicate_eligible = _run(eligible_ids=["m1", "m1"])
    assert duplicate_eligible["status"] == "invalid_output"


def test_privacy_default_blocks_external_question_send():
    calls = []
    record = _run(lambda *args: calls.append(args), allow_raw_question=False)
    assert calls == []
    assert record["status"] == "blocked_by_privacy"
    assert record["reason_code"] == "raw_question_not_permitted"


def test_high_risk_does_not_call_verifier():
    calls = []
    record = _run(lambda *args: calls.append(args), risk_tier="high")
    assert calls == []
    assert record["reason_code"] == "risk_not_eligible"


def test_open_only_calls_only_open_routes():
    calls = []
    invoke = lambda *args: (calls.append(args) or deepcopy(VALID_OUTPUT))
    open_record = _run(invoke, mode="open_only", risk_tier="open")
    conflict_record = _run(invoke, mode="open_only", risk_tier="conflict")
    elevated_record = _run(invoke, mode="open_only", risk_tier="elevated")
    assert open_record["status"] == "success"
    assert conflict_record["reason_code"] == "open_only_route_mismatch"
    assert elevated_record["reason_code"] == "open_only_route_mismatch"
    assert len(calls) == 1


def test_verifier_exception_is_recorded_and_not_raised():
    def timeout(*_):
        raise TimeoutError("provider detail must not be persisted")

    record = _run(timeout)
    assert record["status"] == "provider_error"
    assert record["reason_code"] == "verifier_call_failed"
    assert record["applied"] is False
    assert "provider detail" not in repr(record)


def test_verifier_timeout_is_recorded_without_applying_output():
    def timeout(*_):
        raise verifier.VerifierUnavailable("timeout")

    record = _run(timeout)
    assert record["status"] == "timeout"
    assert record["reason_code"] == "timeout"
    assert record["applied"] is False


def test_llm_timeout_override_is_bounded_and_does_not_change_general_timeout(monkeypatch):
    from app.core.config import Settings
    from app.llm.call_config import LLMCallConfig
    from app.llm.client import _with_timeout_override

    settings = Settings(llm_timeout_seconds=300.0, llm_resilience_attempts=3)
    assert Settings().semantic_verifier_timeout_seconds == 25.0
    normal = LLMCallConfig.normal(settings)
    verifier_call = _with_timeout_override(normal, Settings().semantic_verifier_timeout_seconds)

    assert normal.timeout_seconds == 300.0
    assert verifier_call.timeout_seconds == 25.0
    assert normal.application_attempts == 3
    assert verifier_call.application_attempts == 1
    assert normal.sdk_max_retries == 2
    assert verifier_call.sdk_max_retries == 0
    assert verifier_call.connect_retry_attempts == 0
    assert verifier_call.connect_timeout_seconds is None
    assert verifier_call.fallback_enabled is normal.fallback_enabled


@pytest.mark.parametrize(
    ("call_record", "expected_reason"),
    [
        ({"outcome": "timeout"}, "timeout"),
        ({"outcome": "provider_timeout"}, "timeout"),
        ({"outcome": "parse_failure"}, "parse_failure"),
        ({"outcome": "empty_response"}, "parse_failure"),
        ({"outcome": "not_sent", "not_sent_reason": "quota_blocked"}, "quota_blocked"),
        ({"outcome": "provider_error"}, "provider_error"),
    ],
)
def test_default_invoke_passes_dynamic_policy_and_maps_llm_outcomes(
    monkeypatch, call_record, expected_reason: str,
):
    from app.core.config import Settings
    from app.llm.interaction_policy import InteractionPolicy

    state = {"llm_calls": []}
    monkeypatch.setattr("app.core.execution_context.active_state", lambda: state)
    monkeypatch.setattr("app.core.config.get_settings", lambda: Settings(
        semantic_verifier_timeout_seconds=21.0,
    ))
    observed = {}

    def fake_extract(_system, _message, **kwargs):
        observed.update(kwargs)
        state["llm_calls"].append(dict(call_record))
        return None

    monkeypatch.setattr("app.llm.client.llm_extract", fake_extract)
    with pytest.raises(verifier.VerifierUnavailable) as error:
        verifier._default_invoke("system", {"mentions": []}, "test-model")

    assert error.value.reason_code == expected_reason
    assert observed["operation"] == "extract.semantic_verifier"
    assert "_timeout_seconds" not in observed
    policy = observed["interaction_policy"]
    assert isinstance(policy, InteractionPolicy)
    assert policy.name == "semantic_verifier"
    assert policy.transport == "nonstream"
    assert policy.thinking == "provider_default"
    assert policy.connect_timeout_seconds == 10.0
    assert policy.idle_timeout_seconds is None
    assert policy.total_timeout_seconds == 21.0
    assert policy.provider_fallback_enabled is False


def test_interaction_policy_uses_one_verifier_deadline_without_provider_fallback(monkeypatch):
    from app.core.config import Settings
    from app.llm import client as llm
    from app.llm.interaction_policy import InteractionPolicy
    from app.resilience import ExternalServiceError

    settings = Settings(llm_timeout_seconds=300.0, llm_resilience_attempts=3)
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    resolved = []
    providers = [
        llm._Provider("primary", object(), lambda *_: (resolved.append("primary") or ("model", "standard"))),
        llm._Provider("backup", object(), lambda *_: (resolved.append("backup") or ("model", "standard"))),
    ]
    call_configs = []
    outcomes = []
    state = {
        "trace_id": "semantic-verifier-policy-test",
        "llm_calls": [],
        "runtime_budget": {},
        "llm_usage": {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0},
    }
    monkeypatch.setattr(llm, "active_state", lambda: state)
    monkeypatch.setattr(llm, "_providers", lambda **kwargs: providers)

    def timeout_call(
        _call, _state, _provider, _model, _operation, *, call_config=None, **_kwargs,
    ):
        call_configs.append(call_config)
        raise ExternalServiceError("llm", "timeout", "bounded verifier timeout")

    monkeypatch.setattr(llm, "_invoke_provider_sdk", timeout_call)
    monkeypatch.setattr(llm, "_finish_logical_call", lambda _call, _state, outcome, *_: outcomes.append(outcome))

    policy = InteractionPolicy(
        name="semantic_verifier",
        transport="nonstream",
        thinking="provider_default",
        connect_timeout_seconds=10.0,
        idle_timeout_seconds=None,
        total_timeout_seconds=19.0,
        provider_fallback_enabled=False,
    )
    result = llm.llm_extract(
        "system", "private payload", operation="extract.semantic_verifier",
        interaction_policy=policy,
    )

    assert result is None
    assert resolved == ["primary"]
    assert len(call_configs) == 1
    assert call_configs[0].timeout_seconds == 19.0
    assert call_configs[0].sdk_max_retries == 0
    assert call_configs[0].application_attempts == 1
    assert outcomes == ["timeout"]
    assert state["llm_calls"][0]["operation"] == "extract.semantic_verifier"
    assert state["llm_calls"][0]["request_options"]["provider_fallback_enabled"] is False


def test_unknown_mention_id_is_rejected():
    output = deepcopy(VALID_OUTPUT)
    output["roles"][0]["id"] = "m999"
    record = _run(lambda *_: output)
    assert record["status"] == "invalid_output"
    assert record["verifier_roles"] == {}


def test_changed_span_is_rejected():
    output = deepcopy(VALID_OUTPUT)
    output["roles"][0]["end"] = 4
    record = _run(lambda *_: output)
    assert record["status"] == "invalid_output"


def test_non_closed_role_is_rejected():
    output = deepcopy(VALID_OUTPUT)
    output["roles"][0]["role"] = "sql"
    record = _run(lambda *_: output)
    assert record["status"] == "invalid_output"


def test_extra_metric_or_scope_fields_are_rejected():
    output = deepcopy(VALID_OUTPUT)
    output["metric"] = "order_count"
    record = _run(lambda *_: output)
    assert record["status"] == "invalid_output"


def test_verifier_cannot_return_metric_scope_seller_date_or_sql():
    for field, value in (
        ("metric_id", "order_count"),
        ("scope", {"scope_type": "all"}),
        ("seller_id", "seller-99"),
        ("date", "2025-01-01"),
        ("sql", "SELECT 1"),
        ("query", "group the seller data"),
    ):
        output = deepcopy(VALID_OUTPUT)
        output[field] = value
        assert _run(lambda *_: output)["status"] == "invalid_output"


def test_verifier_cannot_add_or_omit_eligible_mentions():
    output = deepcopy(VALID_OUTPUT)
    output["roles"].append({
        "id": "m2", "start": 3, "end": 4,
        "role": "grouping", "reason_code": "explicit_grouping",
    })
    record = _run(lambda *_: output)
    assert record["status"] == "invalid_output"
    output = {"roles": [], "abstain": False}
    assert _run(lambda *_: output)["status"] == "invalid_output"


def test_abstain_is_not_used_as_a_role_decision():
    record = _run(lambda *_: {"roles": [], "abstain": True})
    assert record["status"] == "abstained"
    assert record["verifier_roles"] == {}
    assert record["applied"] is False


def test_cache_key_includes_versions_model_and_mentions():
    verifier._CACHE.clear()
    calls = []

    def invoke(*_):
        calls.append(1)
        return deepcopy(VALID_OUTPUT), {"input_tokens": 10, "output_tokens": 2}

    first = _run(invoke)
    second = _run(invoke, clear_cache=False)
    changed_model = _run(invoke, clear_cache=False, model="other-model")
    changed_rules = _run(invoke, clear_cache=False, rules_version="rules-next")
    changed_prompt = _run(invoke, clear_cache=False, prompt_version="prompt-next")
    changed_question = _run(invoke, question="按照地区范围的 GMV", mentions=[{
        "id": "m1", "raw_text": "地区", "start": 2, "end": 4, "canonical": "customer_state",
    }], proposed_roles=[{"id": "m1", "role": "reference", "start": 2, "end": 4}], clear_cache=False)
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["input_tokens"] == 0
    assert changed_model["cache_hit"] is False
    assert changed_rules["cache_hit"] is False
    assert changed_prompt["cache_hit"] is False
    assert changed_question["cache_hit"] is False
    assert len(calls) == 5


def test_logs_exclude_raw_question_and_arbitrary_text(monkeypatch):
    events = []
    monkeypatch.setattr("app.observability.record_event", lambda *args: events.append(args))
    secret_question = "按地区范围的 GMV customer-secret-987"
    mentions = [{"id": "m1", "raw_text": "地区", "start": 1, "end": 3, "canonical": "customer_state"}]
    record = _run(
        lambda *_: deepcopy(VALID_OUTPUT),
        question=secret_question,
        mentions=mentions,
    )
    assert events
    serialized = repr(events)
    assert secret_question not in serialized
    assert "customer-secret-987" not in serialized
    assert "地区" not in serialized
    assert "latency_ms" in serialized
    assert "span_ids" in serialized
    assert "verifier_reason_codes" in serialized
    assert "scope_reference" in serialized


def test_source_span_must_match_exact_mention_text():
    record = _run(mentions=[{
        "id": "m1", "raw_text": "店铺", "start": 1, "end": 3, "canonical": "seller",
    }])
    assert record["status"] == "invalid_output"
