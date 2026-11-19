from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.context.route_shadow_audit import (
    build_route_shadow_record,
    validate_route_shadow_record,
)
from app.context.semantic_catalog_v2 import (
    DEFAULT_SEMANTIC_CATALOG,
    CapabilityReference,
    RouteOutcome,
)
from app.context.semantic_contract_v2 import (
    CapabilityDecisionBasis,
    Clause,
    Mention,
    SemanticContractV21,
    Span,
)
from app.graph.nodes import intent_router as intent_router_node


def _contract() -> SemanticContractV21:
    return SemanticContractV21.model_validate({
        "contract_version": "semantic-contract-v2.1",
        "turn_id": "turn-private-phrase",
        "parse_status": "PARSED",
        "semantic_status": "CONFIRMED",
        "domain": "BUSINESS_ANALYSIS",
        "clauses": (Clause(id="clause-private", span=Span(start=0, end=22)),),
        "mentions": (Mention(
            id="mention-private",
            clause_id="clause-private",
            span=Span(start=0, end=22),
            surface="PRIVATE MENTION SURFACE",
            concept_id="CUSTOMER",
            grounding_source="exact_alias",
            source_turn_id="turn-private-phrase",
        ),),
        "operations": (),
        "request_frame": {
            "request_mode": "EXECUTE",
            "modality": "FACTUAL",
            "dialogue_act_provisional": "NEW",
            "dialogue_act_validated": "NEW",
        },
        "catalog_snapshot_hash": DEFAULT_SEMANTIC_CATALOG.snapshot_hash(),
    })


def _settings(**overrides):
    values = {
        "semantic_interpreter_shadow_sample_rate": 1.0,
        "semantic_shadow_salt": "route-audit-test-salt",
        "complex_route_mode": "shadow",
        "semantic_interpreter_mode": "shadow",
        "complex_selector_mode": "off",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _selected_outcome(contract: SemanticContractV21) -> RouteOutcome:
    route = next(item for item in DEFAULT_SEMANTIC_CATALOG.routes if item.route_id == "customer_analysis")
    ref = CapabilityReference(capability_id="customer_rfm_segmentation")
    return RouteOutcome(
        status="SELECTED",
        domain="BUSINESS_ANALYSIS",
        route_id="customer_analysis",
        capability_ids=(ref,),
        route_capability_ids=route.capability_ids,
        decision_basis=CapabilityDecisionBasis(
            contract_version=contract.contract_version,
            catalog_snapshot_hash=contract.catalog_snapshot_hash,
            operation_ids=("operation-private-identifier",),
            clause_ids=("clause-private-identifier",),
            capability_ids=(ref,),
            effective_arguments_digest="a" * 64,
        ),
    )


def test_route_shadow_record_is_allowlisted_and_contains_only_opaque_input_ids() -> None:
    contract = _contract()
    outcome = _selected_outcome(contract)
    record = build_route_shadow_record(
        {
            "trace_id": "trace-private-text",
            "turn_id": "turn-private-text",
            "user_message": "PRIVATE QUESTION",
            "task_type": "metric_query",
            "next_agent": "data_query",
            "triage_unresolved": False,
        },
        contract,
        outcome,
        settings=_settings(),
        sampled_at=datetime(2026, 11, 14, tzinfo=timezone.utc),
    )

    assert record is not None
    encoded = json.dumps(record, ensure_ascii=False)
    for private_value in (
        "trace-private-text", "turn-private-text", "PRIVATE QUESTION",
        "PRIVATE MENTION SURFACE",
        "operation-private-identifier", "clause-private-identifier",
    ):
        assert private_value not in encoded
    assert record["legacy_task_type"] == "metric_query"
    assert record["legacy_next_agent"] == "data_query"
    assert record["legacy_unresolved"] is False
    assert validate_route_shadow_record(record)


def test_route_shadow_records_actual_triage_baseline_when_router_does_not_select() -> None:
    contract = _contract()
    outcome = RouteOutcome(
        status="NO_MATCH",
        domain="BUSINESS_ANALYSIS",
        decision_basis=CapabilityDecisionBasis(
            contract_version=contract.contract_version,
            catalog_snapshot_hash=contract.catalog_snapshot_hash,
            capability_ids=(),
            rejection_reasons=("no_executable_operation",),
        ),
    )
    record = build_route_shadow_record(
        {
            "trace_id": "trace-no-match",
            "turn_id": contract.turn_id,
            "task_type": "clarification",
            "next_agent": "context_clarification",
            "triage_unresolved": True,
        },
        contract,
        outcome,
        settings=_settings(),
    )

    assert record is not None
    assert record["legacy_task_type"] == "clarification"
    assert record["legacy_unresolved"] is True
    assert validate_route_shadow_record(record)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record.update(raw_text="secret"),
        lambda record: record.update(route_status="PICK_TEMPLATE"),
        lambda record: record.update(capability_ids=["outside_catalog"]),
    ],
)
def test_route_shadow_record_rejects_unknown_keys_or_values(mutation) -> None:
    contract = _contract()
    outcome = RouteOutcome(
        status="NO_MATCH",
        domain="BUSINESS_ANALYSIS",
        decision_basis=CapabilityDecisionBasis(
            contract_version=contract.contract_version,
            catalog_snapshot_hash=contract.catalog_snapshot_hash,
            capability_ids=(),
            rejection_reasons=("no_executable_operation",),
        ),
    )
    record = build_route_shadow_record(
        {
            "trace_id": "trace-invalid",
            "turn_id": contract.turn_id,
            "task_type": "metric_query",
            "next_agent": "data_query",
            "triage_unresolved": False,
        },
        contract,
        outcome,
        settings=_settings(),
    )
    assert record is not None
    mutation(record)
    assert not validate_route_shadow_record(record)


def test_shadow_router_attaches_only_safe_audit_and_keeps_legacy_route(monkeypatch) -> None:
    settings = _settings(complex_selector_mode="active")
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    contract = _contract()
    state = {
        "trace_id": "trace-safe-test",
        "turn_id": "turn-safe-test",
        "user_message": "PRIVATE USER TEXT",
        "task_type": "metric_query",
        "next_agent": "data_query",
        "triage_unresolved": False,
        "semantic_contract": contract.model_dump(mode="json"),
        "node_history": [],
    }

    result = intent_router_node.intent_router(state)

    assert result["complex_route_decision"] == {}
    observation = result["routing_context"]["capability_router"]
    assert observation["outcome"]["status"] == "NO_MATCH"
    assert observation["legacy_decision"] == {
        "task_type": "metric_query",
        "next_agent": "data_query",
        "unresolved": False,
    }
    assert observation["audit"]["legacy_task_type"] == "metric_query"
    assert "PRIVATE USER TEXT" not in json.dumps(result["routing_context"], ensure_ascii=False)
    assert "PRIVATE MENTION SURFACE" not in json.dumps(result["routing_context"], ensure_ascii=False)
    assert "complex_selector_observation" not in result


def test_complex_selector_mode_does_not_affect_capability_route(monkeypatch) -> None:
    contract = _contract()
    outcomes = []
    for selector_mode in ("off", "shadow", "active"):
        settings = _settings(
            complex_selector_mode=selector_mode,
            semantic_interpreter_shadow_sample_rate=0.0,
        )
        monkeypatch.setattr(intent_router_node, "get_settings", lambda settings=settings: settings)
        state = {
            "trace_id": "same-route-input",
            "semantic_contract": contract.model_dump(mode="json"),
            "node_history": [],
        }
        result = intent_router_node.intent_router(state)
        outcomes.append(result["routing_context"]["capability_router"]["outcome"])
        assert result["complex_route_decision"] == {}
        assert "complex_selector_observation" not in result

    assert outcomes[0] == outcomes[1] == outcomes[2]
