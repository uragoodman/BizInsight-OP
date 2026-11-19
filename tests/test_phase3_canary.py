from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.complex_analysis.canary import (
    evaluate_canary_enrollment,
    issue_sealed_route_decision,
    template_is_canary_allowlisted,
)
from app.core.config import Settings
from app.complex_analysis.capability_router import route_semantic_contract
from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG
from app.context.semantic_contract_v2 import (
    ArgumentValue,
    Clause,
    Mention,
    OperationCandidate,
    RequestFrame,
    Role,
    SemanticContractCandidateV21,
    Span,
)
from app.context.semantic_contract_validator import validate_semantic_contract
from app.graph import workflow
from app.graph.nodes import complex_clarification
from app.graph.nodes import intent_router as intent_router_node


def _settings(**updates):
    values = {
        "complex_route_mode": "active",
        "complex_route_canary_enabled": True,
        "complex_route_canary_tenants": ["tenant-1"],
        "complex_route_canary_templates": ["rfm_segmentation"],
        "complex_route_canary_sample_rate": 1.0,
        "complex_route_canary_salt": "phase3-canary-test-salt-0123456789",
        "semantic_interpreter_mode": "active",
        "semantic_interpreter_send_raw_question": True,
        "complex_selector_mode": "off",
        "semantic_interpreter_shadow_sample_rate": 0.0,
        "semantic_shadow_salt": "",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _tenant(tenant_id: str = "tenant-1", user_id: str = "user-1"):
    return SimpleNamespace(tenant_id=tenant_id, user_id=user_id)


def _validated_rfm_contract(*, two_basis: bool = True):
    text = "按购买频次和金额给客户分层" if two_basis else "按购买频次给客户分层"
    span = Span(start=0, end=len(text))
    surfaces = [("m_frequency", "购买频次", "PURCHASE_FREQUENCY")]
    if two_basis:
        surfaces.append(("m_monetary", "金额", "MONETARY"))
    surfaces.append(("m_customer", "客户", "CUSTOMER"))
    mentions = tuple(
        Mention(
            id=mention_id,
            clause_id="c1",
            span=Span(start=text.index(surface), end=text.index(surface) + len(surface)),
            surface=surface,
            concept_id=concept_id,
            grounding_source="exact_alias",
            source_turn_id="phase3-turn",
        )
        for mention_id, surface, concept_id in surfaces
    )
    basis_values = [
        ArgumentValue(
            mention_id="m_frequency",
            concept_id="PURCHASE_FREQUENCY",
            stance="ADOPTED",
            evidence_span=span,
        )
    ]
    if two_basis:
        basis_values.append(ArgumentValue(
            mention_id="m_monetary",
            concept_id="MONETARY",
            stance="ADOPTED",
            evidence_span=span,
        ))
    roles = tuple(
        Role(
            id=f"role-{index}",
            clause_id="c1",
            mention_id=value.mention_id,
            concept_id=value.concept_id,
            name="segment_basis",
        )
        for index, value in enumerate(basis_values, start=1)
    )
    candidate = SemanticContractCandidateV21(
        contract_version="semantic-contract-v2.1",
        turn_id="phase3-turn",
        parse_status="PARSED",
        domain="BUSINESS_ANALYSIS",
        clauses=(Clause(id="c1", span=span),),
        mentions=mentions,
        operations=(OperationCandidate(
            id="op-segment",
            clause_id="c1",
            span=span,
            kind="SEGMENT",
            target_mention_ids=("m_customer",),
            target_concept_ids=("CUSTOMER",),
            mentioned_arguments={"basis": tuple(basis_values)},
            mode="EXECUTE",
            polarity="POSITIVE",
            evidence_spans=(span,),
        ),),
        request_frame=RequestFrame(
            request_mode="EXECUTE",
            modality="FACTUAL",
            dialogue_act_provisional="NEW",
            dialogue_act_validated="NEW",
        ),
        roles=roles,
        catalog_snapshot_hash=DEFAULT_SEMANTIC_CATALOG.snapshot_hash(),
    )
    validated = validate_semantic_contract(
        candidate,
        source_text=text,
        catalog=DEFAULT_SEMANTIC_CATALOG,
    )
    assert validated.contract.semantic_status == "CONFIRMED"
    return validated.contract


def test_default_settings_enable_active_routing_for_all_tenants(monkeypatch) -> None:
    for name in (
        "COMPLEX_ROUTE_MODE",
        "COMPLEX_ROUTE_CANARY_ENABLED",
        "COMPLEX_ROUTE_CANARY_TENANTS",
        "COMPLEX_ROUTE_CANARY_TEMPLATES",
        "COMPLEX_ROUTE_CANARY_SAMPLE_RATE",
        "SEMANTIC_INTERPRETER_MODE",
        "SEMANTIC_INTERPRETER_VALIDATION_REPAIR_ENABLED",
        "SEMANTIC_INTERPRETER_SEND_RAW_QUESTION",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert settings.complex_route_mode == "active"
    assert settings.complex_route_canary_enabled is True
    assert settings.complex_route_canary_tenants == ["*"]
    assert set(settings.complex_route_canary_templates) == {
        "rfm_segmentation", "cohort_retention", "customer_lifecycle", "growth_diagnostics",
    }
    assert settings.complex_route_canary_sample_rate == 1.0
    assert settings.semantic_interpreter_mode == "active"
    assert settings.semantic_interpreter_validation_repair_enabled is True
    assert settings.semantic_interpreter_send_raw_question is True
    # Full sampling covers every tenant and does not need a sampling salt.
    assert evaluate_canary_enrollment(settings, _tenant()).reason_code == "eligible"
    assert evaluate_canary_enrollment(settings, _tenant("another-tenant")).reason_code == "eligible"

    monkeypatch.setenv("COMPLEX_ROUTE_MODE", "off")
    rollback_settings = Settings(_env_file=None)
    assert rollback_settings.complex_route_mode == "off"
    assert evaluate_canary_enrollment(rollback_settings, _tenant()).reason_code == "route_mode_not_active"


def test_canary_still_fails_closed_when_required_runtime_gates_are_missing() -> None:
    tenant = _tenant()
    assert evaluate_canary_enrollment(SimpleNamespace(), tenant).reason_code == "route_mode_not_active"
    assert evaluate_canary_enrollment(_settings(complex_route_canary_enabled=False), tenant).reason_code == "canary_disabled"
    assert evaluate_canary_enrollment(_settings(semantic_interpreter_send_raw_question=False), tenant).reason_code == "raw_question_not_permitted"
    assert evaluate_canary_enrollment(_settings(), _tenant("other-tenant")).reason_code == "tenant_not_allowlisted"
    assert evaluate_canary_enrollment(_settings(complex_route_canary_templates=[]), tenant).reason_code == "template_allowlist_empty"
    assert evaluate_canary_enrollment(_settings(complex_route_canary_templates=["not-a-template"]), tenant).reason_code == "template_allowlist_invalid"
    assert evaluate_canary_enrollment(_settings(complex_route_canary_sample_rate=0.0), tenant).reason_code == "sampling_disabled"
    assert evaluate_canary_enrollment(
        _settings(complex_route_canary_sample_rate=0.5, complex_route_canary_salt="short"),
        tenant,
    ).reason_code == "sampling_salt_unconfigured"


def test_canary_sampling_is_stable_for_authenticated_tenant_user() -> None:
    settings = _settings(complex_route_canary_sample_rate=0.5)
    tenant = _tenant()
    first = evaluate_canary_enrollment(settings, tenant)
    second = evaluate_canary_enrollment(settings, tenant)
    assert first == second
    assert first.reason_code in {"eligible", "sample_miss"}


def test_only_catalog_known_allowlisted_templates_can_be_selected() -> None:
    settings = _settings()
    assert template_is_canary_allowlisted("rfm_segmentation", settings)
    assert not template_is_canary_allowlisted("customer_lifecycle", settings)
    assert not template_is_canary_allowlisted("not-a-template", settings)


def test_selected_confirmed_route_is_sealed_against_frozen_catalog() -> None:
    settings = _settings()
    contract = _validated_rfm_contract()
    outcome = route_semantic_contract(contract)
    assert outcome.status == "SELECTED"

    decision = issue_sealed_route_decision(contract, outcome, settings)

    assert decision.verify_seal()
    assert decision.route_id == outcome.route_id
    assert decision.capability_id == "customer_rfm_segmentation"
    assert decision.template_id == "rfm_segmentation"
    assert decision.effective_arguments_digest == outcome.decision_basis.effective_arguments_digest


def test_incomplete_or_unallowlisted_route_cannot_be_sealed() -> None:
    contract = _validated_rfm_contract(two_basis=False)
    outcome = route_semantic_contract(contract)
    assert outcome.status == "NEEDS_CONTEXT"
    with pytest.raises(ValueError, match="only a selected route"):
        issue_sealed_route_decision(contract, outcome, _settings())

    complete_contract = _validated_rfm_contract()
    complete_outcome = route_semantic_contract(complete_contract)
    with pytest.raises(ValueError, match="not enrolled in the canary"):
        issue_sealed_route_decision(
            complete_contract,
            complete_outcome,
            _settings(complex_route_canary_templates=[]),
        )


def test_active_canary_router_issues_typed_decision_only_for_allowlisted_template(monkeypatch) -> None:
    settings = _settings()
    events = []
    measurements = []
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    monkeypatch.setattr(intent_router_node, "record_event", lambda name, attrs: events.append((name, attrs)))
    monkeypatch.setattr(
        intent_router_node,
        "record_measurement",
        lambda name, value=1, *, labels=None: measurements.append((name, value, labels)),
    )
    monkeypatch.setattr(
        intent_router_node,
        "_shadow_route_observation",
        lambda *_args, **_kwargs: {"status": "routed", "audit_status": "skipped"},
    )
    state = {
        "semantic_contract": _validated_rfm_contract().model_dump(mode="json"),
        "semantic_interpreter_observation": {
            "status": "ok", "elapsed_ms": 42, "input_tokens": 100,
            "output_tokens": 40, "estimated_cost": 0.0, "first_token_ms": 8,
        },
        "complex_route_canary_selected": True,
        "complex_route_canary_observation": {"eligible": True, "sample_rate": 1.0},
        "task_type": "anomaly_analysis",
        "next_agent": "data_query",
        "triage_unresolved": False,
        "routing_context": {},
        "node_history": [],
    }

    result = intent_router_node.intent_router(state)

    assert result["route_decision_status"] == "SELECTED"
    assert result["complex_route_canary_observation"]["action"] == "complex_plan"
    assert result["complex_route_decision"].verify_seal()
    assert workflow._route_after_intent_router(result) == "complex_plan"
    assert events[0][0] == "semantic_router.canary"
    assert events[0][1]["route_diff_status"] == "unmapped"
    assert events[0][1]["duration_ms"] == 42
    assert events[0][1]["first_token_ms"] == 8
    assert events[0][1]["template_id"] == "rfm_segmentation"
    assert events[0][1]["llm.estimated_cost"] == 0.0
    assert any(item[0] == "semantic_route_canary_actions" for item in measurements)
    assert any(item[0] == "semantic_route_canary_latency_ms" for item in measurements)
    assert "按购买频次" not in str(events)
    assert "tenant-1" not in str(events)


def test_active_canary_preserves_resolved_multi_dimension_metric_query(monkeypatch) -> None:
    settings = _settings()
    events = []
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    monkeypatch.setattr(intent_router_node, "record_event", lambda name, attrs: events.append((name, attrs)))
    monkeypatch.setattr(
        intent_router_node,
        "route_semantic_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a resolved ordinary metric query must bypass the template router")
        ),
    )
    from app.context.compiler import compile_turn_context

    compilation = compile_turn_context({
        "user_message": "查询 2018-08-20 的 GMV，按商品品类和客户州交叉汇总。",
    })["context_compilation"]
    contract_hash = compilation["request_contract"]["contract_hash"]
    state = {
        "task_type": "metric_query",
        "next_agent": "data_query",
        "context_compilation": compilation,
        "request_contract_hash": contract_hash,
        # A model-level ambiguous semantic candidate must not override the
        # exact, compiler-resolved metric query contract.
        "semantic_contract": {"semantic_status": "AMBIGUOUS"},
        "semantic_interpreter_observation": {"status": "ok"},
        "complex_route_canary_selected": True,
        "complex_route_canary_observation": {"eligible": True, "sample_rate": 1.0},
        "routing_context": {},
        "node_history": [],
    }

    result = intent_router_node.intent_router(state)

    assert result["complex_route_decision"] == {}
    assert result["semantic_contract"] == {}
    assert result["complex_route_canary_observation"]["action"] == "legacy_triage"
    assert result["complex_route_canary_observation"]["reason_code"] == "resolved_multi_dimension_metric_query"
    assert workflow._route_after_intent_router(result) == "retrieve_analysis_memories"
    assert events[0][1]["route_status"] == "BYPASSED"


def test_active_canary_needs_context_clarifies_without_plan(monkeypatch) -> None:
    settings = _settings()
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    monkeypatch.setattr(intent_router_node, "record_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(intent_router_node, "_shadow_route_observation", lambda *_args, **_kwargs: {"status": "routed"})
    state = {
        "semantic_contract": _validated_rfm_contract(two_basis=False).model_dump(mode="json"),
        "semantic_interpreter_observation": {"status": "ok"},
        "complex_route_canary_selected": True,
        "complex_route_canary_observation": {"eligible": True},
        "task_type": "metric_query",
        "next_agent": "data_query",
        "triage_unresolved": False,
        "routing_context": {},
        "node_history": [],
    }

    result = intent_router_node.intent_router(state)

    assert result["route_decision_status"] == "NEEDS_CONTEXT"
    assert result["complex_route_canary_observation"]["action"] == "clarify"
    assert result["semantic_contract"] == {}
    assert workflow._route_after_intent_router(result) == "complex_clarification"
    complex_clarification.complex_clarification(result)
    assert "至少两个不同" in result["answer"]


def test_active_canary_routes_valid_semantic_candidate_when_compiler_mislabels_complex_request(monkeypatch) -> None:
    settings = _settings()
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    monkeypatch.setattr(intent_router_node, "record_event", lambda *_args, **_kwargs: None)
    from app.context.compiler import compile_turn_context

    compilation = compile_turn_context({
        "user_message": "RFM 客户分群，请使用最近购买和购买频次两个不同依据，并重点识别挽留风险。",
    })["context_compilation"]
    contract = _validated_rfm_contract().model_dump(mode="json")
    state = {
        "task_type": "metric_query",
        "next_agent": "data_query",
        "triage_unresolved": False,
        "context_compilation": compilation,
        "request_contract_hash": compilation["request_contract"]["contract_hash"],
        "semantic_contract": contract,
        "semantic_interpreter_observation": {"status": "ok"},
        "complex_route_canary_selected": True,
        "complex_route_canary_observation": {"eligible": True},
        "routing_context": {},
        "node_history": [],
    }

    result = intent_router_node.intent_router(state)

    assert result["route_decision_status"] == "SELECTED"
    assert result["complex_route_canary_observation"]["action"] == "complex_plan"
    assert result["complex_route_decision"].template_id == "rfm_segmentation"


def test_active_canary_invalid_contract_fails_closed_and_records_guardrail(monkeypatch) -> None:
    settings = _settings()
    measurements = []
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    monkeypatch.setattr(intent_router_node, "record_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        intent_router_node,
        "record_measurement",
        lambda name, value=1, *, labels=None: measurements.append((name, value, labels)),
    )
    payload = _validated_rfm_contract().model_dump(mode="json")
    payload["contract_version"] = "semantic-contract-v9"
    state = {
        "semantic_contract": payload,
        "semantic_interpreter_observation": {"status": "ok"},
        "complex_route_canary_selected": True,
        "complex_route_canary_observation": {"eligible": True},
        "routing_context": {},
        "node_history": [],
    }

    result = intent_router_node.intent_router(state)

    assert result["complex_route_canary_observation"]["reason_code"] == "router_validation_failed"
    assert result["complex_route_decision"]["route_decision_status"] == "NEEDS_CONTEXT"
    assert result["semantic_contract"] == {}
    assert workflow._route_after_intent_router(result) == "complex_clarification"
    assert any(
        name == "semantic_route_canary_guardrails"
        and labels == {"reason": "router_validation_failure"}
        for name, _value, labels in measurements
    )


def test_active_canary_template_miss_preserves_legacy_triage(monkeypatch) -> None:
    settings = _settings(complex_route_canary_templates=[])
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    monkeypatch.setattr(intent_router_node, "record_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(intent_router_node, "_shadow_route_observation", lambda *_args, **_kwargs: {"status": "routed"})
    state = {
        "semantic_contract": _validated_rfm_contract().model_dump(mode="json"),
        "semantic_interpreter_observation": {"status": "ok"},
        "complex_route_canary_selected": True,
        "complex_route_canary_observation": {"eligible": True},
        "task_type": "anomaly_analysis",
        "next_agent": "data_query",
        "triage_unresolved": False,
        "routing_context": {},
        "node_history": [],
    }

    result = intent_router_node.intent_router(state)

    assert result["complex_route_canary_observation"]["reason_code"] == "template_not_allowlisted"
    assert result["semantic_contract"] == {}
    assert workflow._route_after_intent_router(result) == "retrieve_analysis_memories"
