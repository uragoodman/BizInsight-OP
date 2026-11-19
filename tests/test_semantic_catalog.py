from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.context.semantic_catalog_v2 import (
    BasisRequirement,
    DEFAULT_SEMANTIC_CATALOG,
    OperationArgumentSpec,
    ROUTE_DECISION_STATUSES,
    RouteOutcome,
    RouteSpec,
    SemanticCatalog,
)
from app.context.semantic_contract_v2 import CapabilityDecisionBasis, CapabilityReference


def _decision_basis(*capability_ids: str, execution_basis: bool = False) -> CapabilityDecisionBasis:
    return CapabilityDecisionBasis(
        contract_version="semantic-contract-v2.1",
        catalog_snapshot_hash=DEFAULT_SEMANTIC_CATALOG.snapshot_hash(),
        operation_ids=("op-1",) if execution_basis else (),
        clause_ids=("clause-1",) if execution_basis else (),
        capability_ids=tuple(CapabilityReference(capability_id=value) for value in capability_ids),
        effective_arguments_digest="a" * 64 if execution_basis else None,
    )


def test_rfm_basis_minimum_is_frozen_and_gates_capability_selection() -> None:
    capability = next(
        item
        for item in DEFAULT_SEMANTIC_CATALOG.capabilities
        if item.capability_id == "customer_rfm_segmentation"
    )
    assert capability.basis_requirement is not None
    assert capability.basis_requirement.decision_status == "FROZEN"
    assert capability.basis_requirement.min_basis_matches == 2
    assert capability.method_concept_ids == ("RFM",)
    assert "RFM" not in capability.concept_ids
    assert DEFAULT_SEMANTIC_CATALOG.concept("RFM").concept_type == "ANALYSIS_METHOD"
    assert not capability.basis_requirement.is_satisfied_by(())
    assert not capability.basis_requirement.is_satisfied_by(("PURCHASE_FREQUENCY",))
    assert capability.basis_requirement.is_satisfied_by(("PURCHASE_FREQUENCY", "MONETARY"))
    assert not capability.basis_requirement.is_satisfied_by(
        ("PURCHASE_FREQUENCY", "PURCHASE_FREQUENCY")
    )
    with pytest.raises(ValidationError):
        BasisRequirement(
            concept_ids=("PURCHASE_FREQUENCY", "MONETARY"),
            min_basis_matches=2,
            decision_status="OPEN",
        )


def test_capability_route_and_template_ids_are_explicit_separate_namespaces() -> None:
    capability = next(
        item
        for item in DEFAULT_SEMANTIC_CATALOG.capabilities
        if item.capability_id == "customer_rfm_segmentation"
    )
    route = next(item for item in DEFAULT_SEMANTIC_CATALOG.routes if item.route_id == capability.route_id)
    assert capability.capability_id != capability.route_id
    assert capability.route_id != capability.template_id
    assert capability.capability_id in {item.capability_id for item in route.capability_ids}
    assert not hasattr(route, "utterances")
    with pytest.raises(ValidationError):
        RouteSpec(
            route_id="business_route",
            domain="BUSINESS_ANALYSIS",
            capability_ids=("customer_rfm_segmentation",),
        )


def test_rfm_veto_requirements_use_exact_kind_target_and_clause_scope() -> None:
    capability = next(
        item
        for item in DEFAULT_SEMANTIC_CATALOG.capabilities
        if item.capability_id == "customer_rfm_segmentation"
    )
    assert {
        (item.kind, item.target_concept_id, item.clause_scope)
        for item in capability.veto_requirements
    } == {
        ("SEGMENT", "RFM", "SAME_CLAUSE"),
        ("SEGMENT", "RFM", "REQUEST"),
    }


def test_operation_paths_are_closed_per_kind_without_case_normalization() -> None:
    assert DEFAULT_SEMANTIC_CATALOG.operation_paths("SEGMENT") == ("basis", "method")
    assert DEFAULT_SEMANTIC_CATALOG.operation_paths("EXPLAIN_CAUSE") == (
        "outcome",
        "metric",
        "dimension",
    )
    assert DEFAULT_SEMANTIC_CATALOG.operation_paths("BREAKDOWN") == ("metric", "dimension")
    assert DEFAULT_SEMANTIC_CATALOG.operation_paths("ENUMERATE") == ("entity", "filter")
    assert DEFAULT_SEMANTIC_CATALOG.operation_paths("segment") == ()  # type: ignore[arg-type]
    assert all(
        "method" not in DEFAULT_SEMANTIC_CATALOG.operation_paths(kind)
        for kind in ("ANALYZE", "FILTER", "SORT", "AGGREGATE", "QUERY")
    )


def test_sort_direction_concepts_are_registered_in_the_catalog_path_domain() -> None:
    ascending = DEFAULT_SEMANTIC_CATALOG.concept("SORT_ASCENDING")
    descending = DEFAULT_SEMANTIC_CATALOG.concept("SORT_DESCENDING")

    assert ascending is not None
    assert ascending.concept_type == "ANALYSIS_PARAMETER"
    assert set(ascending.exact_aliases) == {"升序", "从低到高", "由低到高", "正序", "从小到大"}
    assert descending is not None
    assert descending.concept_type == "ANALYSIS_PARAMETER"
    assert set(descending.exact_aliases) == {"降序", "从高到低", "由高到低", "倒序", "从大到小"}
    assert DEFAULT_SEMANTIC_CATALOG.operation_path_concept_domain("SORT", "direction") == (
        "SORT_ASCENDING",
        "SORT_DESCENDING",
    )
    assert DEFAULT_SEMANTIC_CATALOG.version == "semantic-catalog-v2.2-phase0.3"


@pytest.mark.parametrize(
    ("path", "concept_ids", "error"),
    [
        ("direction", ("UNKNOWN_DIRECTION",), "unknown concept_id"),
        ("direction", ("MONETARY",), "ANALYSIS_PARAMETER"),
        ("direction", ("SORT_ASCENDING", "SORT_ASCENDING"), "must be unique"),
        ("method", ("SORT_ASCENDING",), "allowed argument paths"),
    ],
)
def test_operation_argument_path_domains_are_validated(path, concept_ids, error) -> None:
    payload = DEFAULT_SEMANTIC_CATALOG.model_dump(mode="python")
    sort_spec = next(item for item in payload["operation_arguments"] if item["kind"] == "SORT")
    sort_spec["path_concept_domains"] = {path: concept_ids}

    with pytest.raises(ValidationError, match=error):
        SemanticCatalog.model_validate(payload)


def test_operation_argument_spec_rejects_domains_for_undeclared_paths() -> None:
    with pytest.raises(ValidationError, match="allowed argument paths"):
        OperationArgumentSpec(
            kind="SORT",
            allowed_paths=("key", "direction"),
            path_concept_domains={"method": ("SORT_ASCENDING",)},
        )


@pytest.mark.parametrize(
    ("method_ids", "error"),
    [
        (("RFM", "RFM"), "method concept ids must be unique"),
        (("UNKNOWN_METHOD",), "unknown method concept_id"),
        (("CUSTOMER",), "ANALYSIS_METHOD concepts"),
    ],
)
def test_capability_method_concepts_must_be_unique_known_methods(method_ids, error):
    payload = DEFAULT_SEMANTIC_CATALOG.model_dump(mode="python")
    capability = next(
        item for item in payload["capabilities"]
        if item["capability_id"] == "customer_rfm_segmentation"
    )
    capability["method_concept_ids"] = method_ids

    with pytest.raises(ValidationError, match=error):
        SemanticCatalog.model_validate(payload)


def test_general_conversation_route_has_no_capability_tool_plan_or_database_access() -> None:
    route = next(
        item
        for item in DEFAULT_SEMANTIC_CATALOG.routes
        if item.domain == "GENERAL_CONVERSATION"
    )
    assert route.capability_ids == ()
    assert route.tool_ids == ()
    assert route.creates_plan is False
    assert route.database_access is False
    with pytest.raises(ValidationError):
        RouteSpec(
            route_id="general_conversation",
            domain="GENERAL_CONVERSATION",
            tool_ids=("data_query",),
        )


def test_route_outcome_statuses_keep_general_no_match_unsupported_and_context_separate() -> None:
    assert RouteOutcome(
        status="GENERAL_CONVERSATION",
        domain="GENERAL_CONVERSATION",
        route_id="general_conversation",
        decision_basis=_decision_basis(),
    ).status == "GENERAL_CONVERSATION"
    assert RouteOutcome(
        status="NO_MATCH", domain="BUSINESS_ANALYSIS", decision_basis=_decision_basis()
    ).status == "NO_MATCH"
    assert RouteOutcome(
        status="UNSUPPORTED", domain="BUSINESS_ANALYSIS", decision_basis=_decision_basis()
    ).status == "UNSUPPORTED"
    assert RouteOutcome(
        status="NEEDS_CONTEXT",
        domain="BUSINESS_ANALYSIS",
        route_id="customer_analysis",
        clarification_required=True,
        decision_basis=_decision_basis(),
    ).status == "NEEDS_CONTEXT"
    with pytest.raises(ValidationError):
        RouteOutcome(
            status="SELECTED",
            domain="GENERAL_CONVERSATION",
            route_id="general_conversation",
            decision_basis=_decision_basis(),
        )
    selected = RouteOutcome(
        status="SELECTED",
        domain="BUSINESS_ANALYSIS",
        route_id="customer_analysis",
        capability_ids=(CapabilityReference(capability_id="customer_rfm_segmentation"),),
        route_capability_ids=next(
            route.capability_ids for route in DEFAULT_SEMANTIC_CATALOG.routes
            if route.route_id == "customer_analysis"
        ),
        decision_basis=_decision_basis("customer_rfm_segmentation", execution_basis=True),
    )
    assert selected.decision_basis.capability_ids[0].capability_id == "customer_rfm_segmentation"
    with pytest.raises(ValidationError, match="must cite the selected capability references"):
        RouteOutcome(
            status="SELECTED",
            domain="BUSINESS_ANALYSIS",
            route_id="customer_analysis",
            capability_ids=(CapabilityReference(capability_id="customer_rfm_segmentation"),),
            route_capability_ids=next(
                route.capability_ids for route in DEFAULT_SEMANTIC_CATALOG.routes
                if route.route_id == "customer_analysis"
            ),
            decision_basis=_decision_basis("customer_cohort_retention", execution_basis=True),
        )


def test_selected_route_outcome_requires_nonempty_decision_basis() -> None:
    with pytest.raises(ValidationError, match="non-empty operation_ids"):
        RouteOutcome(
            status="SELECTED",
            domain="BUSINESS_ANALYSIS",
            route_id="customer_analysis",
            capability_ids=(CapabilityReference(capability_id="customer_rfm_segmentation"),),
            route_capability_ids=next(
                route.capability_ids for route in DEFAULT_SEMANTIC_CATALOG.routes
                if route.route_id == "customer_analysis"
            ),
            decision_basis=_decision_basis("customer_rfm_segmentation"),
        )


def test_selected_outcome_rejects_capability_outside_route_membership() -> None:
    route = next(
        item for item in DEFAULT_SEMANTIC_CATALOG.routes
        if item.route_id == "customer_analysis"
    )
    outsider = CapabilityReference(capability_id="unregistered_capability")
    with pytest.raises(ValidationError, match="must belong to the selected route"):
        RouteOutcome(
            status="SELECTED",
            domain="BUSINESS_ANALYSIS",
            route_id=route.route_id,
            capability_ids=(outsider,),
            route_capability_ids=route.capability_ids,
            decision_basis=_decision_basis("unregistered_capability", execution_basis=True),
        )


def test_grounding_sources_and_route_outcomes_are_explicit() -> None:
    customer = DEFAULT_SEMANTIC_CATALOG.concept("CUSTOMER")
    assert customer is not None and "客户" in customer.exact_aliases
    assert {"NO_MATCH", "UNSUPPORTED", "NEEDS_CONTEXT", "GENERAL_CONVERSATION"} <= set(
        ROUTE_DECISION_STATUSES
    )
    with pytest.raises(ValidationError):
        RouteOutcome(
            status="NEEDS_CONTEXT",
            domain="BUSINESS_ANALYSIS",
            decision_basis=_decision_basis(),
        )
    with pytest.raises(ValidationError):
        RouteOutcome(
            status="NEEDS_CONTEXT",
            domain="GENERAL_CONVERSATION",
            clarification_required=True,
            decision_basis=_decision_basis(),
        )
    for status in ("AMBIGUOUS", "CONFLICT"):
        with pytest.raises(ValidationError, match="GENERAL_CONVERSATION domain must use"):
            RouteOutcome(
                status=status,
                domain="GENERAL_CONVERSATION",
                clarification_required=True,
                decision_basis=_decision_basis(),
            )
    with pytest.raises(ValidationError):
        RouteOutcome(
            status="NO_MATCH",
            domain="BUSINESS_ANALYSIS",
            decision_basis=_decision_basis("customer_rfm_segmentation"),
        )
    assert len(DEFAULT_SEMANTIC_CATALOG.snapshot_hash()) == 64
