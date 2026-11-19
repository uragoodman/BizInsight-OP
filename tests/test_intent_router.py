from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.graph import workflow
from app.graph.nodes import semantic_parse as semantic_parse_node
from app.graph.nodes import intent_router as intent_router_node


@pytest.mark.parametrize("route_mode", ["off", "shadow"])
def test_phase0_off_and_shadow_never_select_a_complex_route(monkeypatch, route_mode: str) -> None:
    settings = SimpleNamespace(
        complex_route_mode=route_mode,
        semantic_interpreter_mode="off",
        complex_selector_mode="active",
    )
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    state = {
        "user_message": "RFM 客户分群",
        "semantic_contract": {"semantic_status": "CONFIRMED"},
        "node_history": [],
    }

    result = intent_router_node.intent_router(state)

    assert result["complex_route_decision"] == {}
    assert result["route_decision_status"] == route_mode
    assert "complex_selector_observation" not in result
    assert workflow._route_after_intent_router(result) == "triage"


def test_phase0_active_fails_closed_without_consulting_old_router(monkeypatch) -> None:
    settings = SimpleNamespace(
        complex_route_mode="active",
        semantic_interpreter_mode="active",
        complex_selector_mode="active",
    )
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    state = {
        "user_message": "RFM 客户分群",
        # A legacy/v1-looking mapping cannot be promoted to a v2 route decision.
        "semantic_contract": {"semantic_status": "CONFIRMED", "primary_intent": "RFM"},
        "complex_template_confirmation": "rfm_segmentation",
        "node_history": [],
    }

    result = intent_router_node.intent_router(state)

    assert result["complex_route_decision"] == {
        "kind": "clarify",
        "route_decision_status": "NEEDS_CONTEXT",
        "evidence": {"clarification_reason": "semantic_router_not_ready"},
    }
    assert result["semantic_interpreter_observation"]["status"] == "reserved_phase0_noop"
    assert result["complex_clarification"]["reason_code"] == "semantic_router_not_ready"
    assert workflow._route_after_intent_router(result) == "complex_clarification"


def test_active_route_ignores_forged_dict_in_graph_state(monkeypatch) -> None:
    monkeypatch.setattr(
        workflow,
        "get_settings",
        lambda: SimpleNamespace(complex_route_mode="active"),
    )
    forged = {
        "kind": "template",
        "route_decision_status": "SELECTED",
        "templates": [{"template_id": "rfm_segmentation"}],
    }
    assert workflow._route_after_intent_router({"complex_route_decision": forged}) == "complex_clarification"


def test_active_router_preserves_compiler_clarification_reason(monkeypatch) -> None:
    outcome = SimpleNamespace(
        status="NEEDS_CONTEXT",
        route_id=None,
        capability_ids=(),
        candidate_capability_ids=(),
        decision_basis=SimpleNamespace(rejection_reasons=()),
    )
    monkeypatch.setattr(
        intent_router_node.SemanticContractV21,
        "model_validate_json",
        classmethod(lambda _cls, _value: object()),
    )
    monkeypatch.setattr(intent_router_node, "route_semantic_contract", lambda _contract: outcome)
    monkeypatch.setattr(intent_router_node, "_shadow_route_observation", lambda *_args: {"status": "routed"})
    monkeypatch.setattr(intent_router_node, "_record_active_canary_event", lambda *_args, **_kwargs: None)
    state = {
        "complex_route_canary_observation": {"eligible": True},
        "semantic_interpreter_observation": {"status": "ok"},
        "semantic_contract": {"candidate": "already validated by semantic_parse"},
        "context_compilation": {
            "needs_clarification": True,
            "clarification_reason": "comparison_dimension_not_supported",
        },
    }

    intent_router_node._active_canary_route(state, SimpleNamespace())

    assert state["complex_clarification"]["reason_code"] == "comparison_dimension_not_supported"


def _dimension_required_state() -> dict:
    from app.context.request_contract import seal_request_contract

    contract = seal_request_contract({
        "metric_id": "average_order_value",
        "operation_intent": "dimensional_statistical_contribution",
        "dimension": None,
        "comparison": {
            "kind": "day_pair",
            "baseline_date": "2018-08-03",
            "current_date": "2018-08-04",
        },
        "causal_boundary": "statistical_contribution_not_business_causation",
        "required_capability": "compare",
        "required_tool": "compare_breakdown_by_dimension",
        "required_evidence_sequence": ["trend", "dimension_contribution"],
        "source": "current_user_message",
        "parse_status": "clarification_required",
        "uncertainty_reason": "dimension_required",
        "candidate_metrics": [],
        "candidate_dimensions": [],
    })
    contract_hash = contract["contract_hash"]
    compilation = {
        "needs_clarification": True,
        "clarification_reason": "dimension_required",
        "operation_intent": {"kind": "dimensional_statistical_contribution"},
        "request_contract": contract,
        "request_contract_hash": contract_hash,
    }
    return {
        "trace_id": "trace-current",
        "context_compilation": compilation,
        "request_contract_hash": contract_hash,
        "turn_resolution": {
            "turn_id": "trace-current",
            "relation": "clarify",
            "needs_clarification": True,
            "clarification_reason": "dimension_required",
            "request_contract_hash": contract_hash,
        },
        "complex_route_canary_observation": {"eligible": True},
        "semantic_interpreter_observation": {"status": "ok"},
        "semantic_contract": {"candidate": "validated in semantic_parse"},
        "node_history": [],
    }


@pytest.mark.parametrize(
    ("route_status", "rejection_reasons", "expected_reason"),
    [
        ("NEEDS_CONTEXT", ("semantic_status_missing",), "dimension_required"),
        ("NEEDS_CONTEXT", ("scope_required",), "scope_required"),
        ("AMBIGUOUS", ("semantic_status_missing",), "semantic_status_missing"),
    ],
)
def test_active_router_uses_compiler_reason_only_for_generic_needs_context(
    monkeypatch, route_status: str, rejection_reasons: tuple[str, ...], expected_reason: str,
) -> None:
    outcome = SimpleNamespace(
        status=route_status,
        route_id=None,
        capability_ids=(),
        candidate_capability_ids=(),
        decision_basis=SimpleNamespace(rejection_reasons=rejection_reasons),
    )
    monkeypatch.setattr(
        intent_router_node.SemanticContractV21,
        "model_validate_json",
        classmethod(lambda _cls, _value: object()),
    )
    monkeypatch.setattr(intent_router_node, "route_semantic_contract", lambda _contract: outcome)
    monkeypatch.setattr(intent_router_node, "_shadow_route_observation", lambda *_args: {"status": "routed"})
    monkeypatch.setattr(intent_router_node, "_record_active_canary_event", lambda *_args, **_kwargs: None)
    state = _dimension_required_state()

    intent_router_node._active_canary_route(state, SimpleNamespace())

    assert state["route_decision_status"] == route_status
    assert state["complex_route_decision"]["evidence"]["clarification_reason"] == expected_reason
    assert state["complex_clarification"]["reason_code"] == expected_reason
    assert state["semantic_contract"] == {}
    assert state["complex_route_canary_observation"]["router_reason_code"] == rejection_reasons[0]
    assert state["complex_route_canary_observation"]["clarification_reason_code"] == expected_reason


def test_active_router_fails_closed_with_compiler_clarification_when_interpreter_candidate_is_invalid(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        intent_router_node.SemanticContractV21,
        "model_validate_json",
        classmethod(lambda *_args, **_kwargs: pytest.fail("schema-invalid candidates must not be validated")),
    )
    monkeypatch.setattr(
        intent_router_node,
        "route_semantic_contract",
        lambda _contract: pytest.fail("invalid candidates must not reach Router"),
    )
    monkeypatch.setattr(intent_router_node, "_record_active_canary_event", lambda *_args, **_kwargs: None)
    state = _dimension_required_state()
    state["semantic_interpreter_observation"] = {
        "status": "schema_invalid",
        "reason_code": "candidate_schema_invalid",
    }
    # Keep a non-empty candidate so this test proves the schema_invalid status
    # itself prevents parsing/routing, rather than passing because the payload
    # is absent. Preserve sentinels to detect accidental execution-side writes.
    state["semantic_contract"] = {"candidate": {"non_empty": True}}
    query_results = [{"sentinel": "existing_query"}]
    tool_calls = [{"sentinel": "existing_tool_call"}]
    state["query_results"] = query_results
    state["tool_calls"] = tool_calls

    intent_router_node._active_canary_route(state, SimpleNamespace())

    assert state["route_decision_status"] == "NEEDS_CONTEXT"
    assert state["complex_route_decision"]["evidence"]["clarification_reason"] == "dimension_required"
    assert state["complex_route_canary_observation"]["route_reason_code"] == "semantic_interpreter_failed"
    assert state["complex_route_canary_observation"]["clarification_reason_source"] == "context_compiler"
    assert state["semantic_contract"] == {}
    assert state["query_results"] is query_results
    assert state["tool_calls"] is tool_calls

    from app.graph.nodes.complex_clarification import complex_clarification

    result = complex_clarification(state)
    assert result["complex_clarification"]["reason_code"] == "dimension_required"
    assert "请明确要按哪个维度展示统计贡献" in result["answer"]
    assert result["query_results"] == query_results
    assert result["tool_calls"] == tool_calls


@pytest.mark.parametrize(
    ("resolution_turn_id", "expected_reason"),
    [
        ("persisted-current", "dimension_required"),
        ("client-current", None),
        ("trace-current", None),
        ("persisted-previous", None),
    ],
)
def test_dimension_required_override_uses_durable_turn_id_precedence(
    resolution_turn_id: str, expected_reason: str | None,
) -> None:
    state = _dimension_required_state()
    state["persistence_turn_id"] = "persisted-current"
    state["chat_turn_id"] = "client-current"
    state["trace_id"] = "trace-current"
    state["turn_resolution"]["turn_id"] = resolution_turn_id

    assert intent_router_node._trusted_dimension_required_reason(state) == expected_reason


@pytest.mark.parametrize(
    "tamper",
    ["state_hash", "compilation_hash", "resolution_hash", "current_turn_id", "contract_reason"],
)
def test_dimension_required_override_rejects_unanchored_or_mismatched_compilation(tamper: str) -> None:
    from app.context.request_contract import seal_request_contract

    state = _dimension_required_state()
    compilation = state["context_compilation"]
    contract = compilation["request_contract"]
    if tamper == "state_hash":
        state["request_contract_hash"] = "stale-state-hash"
    elif tamper == "compilation_hash":
        compilation["request_contract_hash"] = "stale-compilation-hash"
    elif tamper == "resolution_hash":
        state["turn_resolution"]["request_contract_hash"] = "stale-resolution-hash"
    elif tamper == "current_turn_id":
        state["turn_resolution"]["turn_id"] = "stale-turn"
    else:
        altered_contract = dict(contract)
        altered_contract["uncertainty_reason"] = "ambiguous_dimension"
        altered_contract = seal_request_contract(altered_contract)
        compilation["request_contract"] = altered_contract
        compilation["request_contract_hash"] = altered_contract["contract_hash"]
        state["request_contract_hash"] = altered_contract["contract_hash"]
        state["turn_resolution"]["request_contract_hash"] = altered_contract["contract_hash"]

    assert intent_router_node._trusted_dimension_required_reason(state) is None


def test_phase0_status_node_does_not_import_or_call_legacy_router() -> None:
    import inspect

    source = inspect.getsource(intent_router_node)
    assert "app.complex_analysis.router" not in source
    assert "decide_route" not in source
    assert "question" not in source
    assert 'state.get("user_message")' not in source
    assert 'state["user_message"]' not in source


@pytest.mark.parametrize("interpreter_mode", ["off", "active"])
def test_off_and_active_semantic_parse_keep_reserved_noop_without_reading_question(
    monkeypatch, interpreter_mode: str,
) -> None:
    import inspect

    monkeypatch.setattr(
        semantic_parse_node,
        "get_settings",
        lambda: SimpleNamespace(semantic_interpreter_mode=interpreter_mode),
    )

    def should_not_interpret(*_args, **_kwargs):
        raise AssertionError("off/active mode must not invoke the interpreter")

    monkeypatch.setattr(semantic_parse_node, "interpret_semantics", should_not_interpret)

    class NoQuestionRead(dict):
        def get(self, key, default=None):
            if key == "user_message":
                raise AssertionError("off/active mode must not read user_message")
            return super().get(key, default)

    state = {
        "previous_turn_context": {"operations": [{"operation": "segment"}]},
        "semantic_contract": {"forged": "legacy-state"},
        "node_history": [],
    }
    result = semantic_parse_node.semantic_parse(NoQuestionRead(state))
    assert result["semantic_contract"] == {}
    assert result["dialogue_context_resolution"] == {}
    assert result["routing_context"] == {}
    assert result["semantic_parse_status"] == "reserved_phase0_noop"
    assert result["semantic_parse_observation"] == {
        "mode": interpreter_mode,
        "status": "reserved_phase0_noop",
    }
    source = inspect.getsource(semantic_parse_node)
    assert "resolve_dialogue_context" not in source
    assert "parse_semantics" not in source
    assert "user_message" in source


def test_shadow_without_raw_question_is_skipped_and_preserved_by_intent_router(monkeypatch):
    settings = SimpleNamespace(
        semantic_interpreter_mode="shadow",
        semantic_interpreter_inline_shadow_enabled=True,
        semantic_interpreter_send_raw_question=False,
        complex_route_mode="shadow",
        complex_selector_mode="off",
    )
    monkeypatch.setattr(semantic_parse_node, "get_settings", lambda: settings)
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    called = []

    def interpret(question, **kwargs):
        called.append((question, kwargs))
        return {
            "mode": "shadow", "status": "skipped", "reason_code": "raw_question_not_permitted",
            "contract": None,
        }

    monkeypatch.setattr(semantic_parse_node, "interpret_semantics", interpret)
    state = semantic_parse_node.semantic_parse({"trace_id": "turn-shadow", "node_history": []})
    assert called == [("", {
        "turn_id": "turn-shadow", "dialogue_context": None, "mode": "shadow",
    })]
    assert state["semantic_contract"] == {}
    assert state["semantic_parse_status"] == "shadow_skipped"
    expected = dict(state["semantic_interpreter_observation"])

    routed = intent_router_node.intent_router(state)
    assert routed["semantic_interpreter_observation"] == expected
    assert routed["complex_route_decision"] == {}
    assert workflow._route_after_intent_router(routed) == "triage"


def test_shadow_inline_disabled_skips_before_reading_question(monkeypatch):
    settings = SimpleNamespace(
        semantic_interpreter_mode="shadow",
        semantic_interpreter_inline_shadow_enabled=False,
        semantic_interpreter_send_raw_question=True,
    )
    monkeypatch.setattr(semantic_parse_node, "get_settings", lambda: settings)

    def should_not_interpret(*_args, **_kwargs):
        raise AssertionError("inline-disabled shadow must not call the interpreter")

    monkeypatch.setattr(semantic_parse_node, "interpret_semantics", should_not_interpret)

    class NoQuestionRead(dict):
        def get(self, key, default=None):
            if key == "user_message":
                raise AssertionError("inline-disabled shadow must not read user_message")
            return super().get(key, default)

    state = semantic_parse_node.semantic_parse(NoQuestionRead({"trace_id": "offline-only"}))
    assert state["semantic_contract"] == {}
    assert state["dialogue_context_resolution"] == {}
    assert state["routing_context"] == {}
    assert state["semantic_interpreter_observation"] == {
        "status": "skipped",
        "reason_code": "shadow_requires_offline_runner",
    }
    assert state["semantic_parse_status"] == "shadow_skipped_offline_only"


def test_shadow_missing_question_skips_interpreter_even_with_raw_opt_in(monkeypatch):
    from app.context import semantic_interpreter

    settings = SimpleNamespace(
        semantic_interpreter_mode="shadow",
        semantic_interpreter_inline_shadow_enabled=True,
        semantic_interpreter_send_raw_question=True,
        semantic_interpreter_shadow_sample_rate=1.0,
        semantic_interpreter_model="fake-model",
        semantic_interpreter_timeout_seconds=10.0,
        semantic_interpreter_max_tokens=2048,
        llm_standard_model="standard-model",
        llm_model="fallback-model",
    )
    monkeypatch.setattr(semantic_parse_node, "get_settings", lambda: settings)
    monkeypatch.setattr(semantic_interpreter, "get_settings", lambda: settings)
    monkeypatch.setattr(
        semantic_interpreter,
        "_default_invoke",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("empty questions must not reach the provider")
        ),
    )

    result = semantic_parse_node.semantic_parse({"trace_id": "empty-question-turn"})
    assert result["semantic_interpreter_observation"]["status"] == "skipped"
    assert result["semantic_interpreter_observation"]["reason_code"] == "empty_question"
    assert result["semantic_contract"] == {}


def test_shadow_observation_does_not_change_route_decision(monkeypatch):
    settings = SimpleNamespace(
        semantic_interpreter_mode="shadow",
        semantic_interpreter_inline_shadow_enabled=True,
        semantic_interpreter_send_raw_question=True,
        complex_route_mode="shadow",
        complex_selector_mode="off",
    )
    monkeypatch.setattr(semantic_parse_node, "get_settings", lambda: settings)
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    contract = {"semantic_status": "CONFIRMED", "primary_intent": "CUSTOMER_LIFECYCLE"}
    observation = {"mode": "shadow", "status": "ok", "contract": contract}
    monkeypatch.setattr(semantic_parse_node, "interpret_semantics", lambda *_args, **_kwargs: observation)

    state = semantic_parse_node.semantic_parse({"user_message": "客户生命周期分析", "trace_id": "turn-shadow"})
    assert state["semantic_contract"] == contract
    routed = intent_router_node.intent_router(state)
    assert routed["semantic_interpreter_observation"] is observation
    assert routed["complex_route_decision"] == {}
    assert workflow._route_after_intent_router(routed) == "triage"


@pytest.mark.parametrize("repair_enabled", [True, False])
def test_active_semantic_parse_calls_interpreter_only_after_canary_enrollment(monkeypatch, repair_enabled):
    settings = SimpleNamespace(
        complex_route_mode="active",
        complex_route_canary_enabled=True,
        complex_route_canary_tenants=["tenant-1"],
        complex_route_canary_templates=["rfm_segmentation"],
        complex_route_canary_sample_rate=1.0,
        complex_route_canary_salt="phase3-canary-test-salt-0123456789",
        semantic_interpreter_mode="active",
        semantic_interpreter_validation_repair_enabled=repair_enabled,
        semantic_interpreter_send_raw_question=True,
        semantic_interpreter_shadow_sample_rate=0.0,
        semantic_shadow_salt="",
    )
    monkeypatch.setattr(semantic_parse_node, "get_settings", lambda: settings)
    monkeypatch.setattr(
        "app.tenancy.context.current_tenant_context",
        lambda: SimpleNamespace(tenant_id="tenant-1", user_id="user-1"),
    )
    calls = []
    observation = {
        "status": "ok",
        "mode": "active",
        "model": "fake-model",
        "prompt_version": "test-v1",
        "catalog_snapshot_hash": "a" * 64,
        "elapsed_ms": 15,
        "reason_code": None,
        "candidate": {"should_not_be_retained": True},
        "raw_output": "RAW_OUTPUT_SENTINEL",
        "repair": {
            "attempted": True,
            "attempt_count": 2,
            "retry_count": 1,
            "initial_status": "schema_invalid",
            "final_status": "ok",
            "attempts": [{"input_tokens": 1}, {"input_tokens": 2}],
        },
        "contract": {"semantic_status": "CONFIRMED"},
        "issue_codes": [],
    }
    monkeypatch.setattr(
        semantic_parse_node,
        "interpret_semantics",
        lambda question, **kwargs: calls.append((question, kwargs)) or observation,
    )

    result = semantic_parse_node.semantic_parse({
        "trace_id": "phase3-turn",
        "user_message": "合成测试问题",
        "node_history": [],
    })

    assert result["complex_route_canary_selected"] is True
    assert result["semantic_contract"] == {"semantic_status": "CONFIRMED"}
    assert result["semantic_interpreter_observation"]["status"] == "ok"
    assert "candidate" not in result["semantic_interpreter_observation"]
    assert result["semantic_interpreter_observation"]["repair"]["attempt_count"] == 2
    assert "raw_output" not in result["semantic_interpreter_observation"]
    assert calls == [(
        "合成测试问题",
        {
            "turn_id": "phase3-turn",
            "dialogue_context": None,
            "mode": "active",
            "active_canary_authorized": True,
            "validation_repair": repair_enabled,
        },
    )]


def test_active_semantic_parse_without_privacy_opt_in_stays_on_legacy_path(monkeypatch):
    settings = SimpleNamespace(
        complex_route_mode="active",
        complex_route_canary_enabled=True,
        complex_route_canary_tenants=["tenant-1"],
        complex_route_canary_templates=["rfm_segmentation"],
        complex_route_canary_sample_rate=1.0,
        complex_route_canary_salt="phase3-canary-test-salt-0123456789",
        semantic_interpreter_mode="active",
        semantic_interpreter_send_raw_question=False,
    )
    monkeypatch.setattr(semantic_parse_node, "get_settings", lambda: settings)
    monkeypatch.setattr(
        "app.tenancy.context.current_tenant_context",
        lambda: SimpleNamespace(tenant_id="tenant-1", user_id="user-1"),
    )
    monkeypatch.setattr(
        semantic_parse_node,
        "interpret_semantics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("privacy-disabled request must not call the interpreter")
        ),
    )

    result = semantic_parse_node.semantic_parse({"user_message": "sensitive", "node_history": []})

    assert result["complex_route_canary_selected"] is False
    assert result["complex_route_canary_observation"]["reason_code"] == "raw_question_not_permitted"
    assert result["semantic_contract"] == {}
