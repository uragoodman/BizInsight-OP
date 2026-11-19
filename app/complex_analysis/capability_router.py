"""Deterministic Capability Router for validated Semantic Contract v2.1.

Phase 2 shadow router. It consumes only a validated contract and the frozen
catalog. It never reads user text, template names, aliases, embeddings, or
provider output, so the same contract always produces the same outcome.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence

from app.context.semantic_catalog_v2 import (
    CapabilitySpec,
    DEFAULT_SEMANTIC_CATALOG,
    RouteOutcome,
    SemanticCatalog,
)
from app.context.semantic_contract_v2 import (
    CapabilityDecisionBasis,
    CapabilityReference,
    Operation,
    SemanticContractV21,
)

_CONTEXT_REQUIRED_REASONS = frozenset({
    "explicit_method_missing_required_arguments",
    "missing_required_arguments",
    "insufficient_basis",
})


def _require_anchored_contract(contract: SemanticContractV21, catalog: SemanticCatalog) -> None:
    if not isinstance(contract, SemanticContractV21):
        raise TypeError("route_semantic_contract requires a validated SemanticContractV21")
    if contract.contract_version != "semantic-contract-v2.1":
        raise ValueError("contract version is not semantic-contract-v2.1")
    if contract.catalog_snapshot_hash != catalog.snapshot_hash():
        raise ValueError("contract catalog snapshot does not match the router catalog")
    known_concepts = {item.concept_id for item in catalog.concepts}
    referenced_concepts = {mention.concept_id for mention in contract.mentions}
    referenced_concepts.update(role.concept_id for role in contract.roles)
    for operation in contract.operations:
        referenced_concepts.update(operation.target_concept_ids)
        for values in operation.mentioned_arguments.values():
            referenced_concepts.update(value.concept_id for value in values)
        for values in operation.effective_arguments.values():
            referenced_concepts.update(value.concept_id for value in values)
    referenced_concepts.update(
        endpoint.value
        for relation in contract.relations
        for endpoint in (relation.subject, relation.object)
        if endpoint.endpoint_type == "concept_id"
    )
    unknown_concepts = referenced_concepts - known_concepts
    if unknown_concepts:
        raise ValueError("contract references concept ids outside the router catalog")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _effective_arguments_digest(operations: Iterable[Operation]) -> str:
    payload = {
        operation.id: {
            path: [value.model_dump(mode="json") for value in values]
            for path, values in operation.effective_arguments.items()
        }
        for operation in operations
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _capability_references(capability_ids: Iterable[str]) -> tuple[CapabilityReference, ...]:
    return tuple(
        CapabilityReference(capability_id=capability_id)
        for capability_id in sorted(set(capability_ids))
    )


def _decision_basis(
    contract: SemanticContractV21,
    *,
    capability_ids: Iterable[str] = (),
    operation_ids: Iterable[str] = (),
    clause_ids: Iterable[str] = (),
    effective_arguments_digest: str | None = None,
    rejected_candidates: Iterable[str] = (),
    rejection_reasons: Iterable[str] = (),
) -> CapabilityDecisionBasis:
    return CapabilityDecisionBasis(
        contract_version=contract.contract_version,
        catalog_snapshot_hash=contract.catalog_snapshot_hash,
        operation_ids=tuple(sorted(set(operation_ids))),
        clause_ids=tuple(sorted(set(clause_ids))),
        capability_ids=_capability_references(capability_ids),
        effective_arguments_digest=effective_arguments_digest,
        rejected_candidates=tuple(sorted(set(rejected_candidates))),
        rejection_reasons=tuple(sorted(set(rejection_reasons))),
    )


def _operation_references(operations: Iterable[Operation]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    operation_ids = [operation.id for operation in operations]
    clause_ids = [operation.clause_id for operation in operations]
    return tuple(operation_ids), tuple(clause_ids)


def _positive_operations(contract: SemanticContractV21) -> list[Operation]:
    return [
        operation
        for operation in contract.operations
        if operation.polarity == "POSITIVE" and operation.mode == "EXECUTE"
    ]


def _non_executable_outcome(
    contract: SemanticContractV21,
    domain: str,
    reason: str,
) -> RouteOutcome:
    operation_ids, clause_ids = _operation_references(contract.operations)
    return RouteOutcome(
        status="NO_MATCH",
        domain=domain,
        decision_basis=_decision_basis(
            contract,
            operation_ids=operation_ids,
            clause_ids=clause_ids,
            rejection_reasons=(reason,),
        ),
    )


def _requirement_candidates(
    capability: CapabilitySpec,
    requirement,
    positives: Sequence[Operation],
) -> list[Operation]:
    return [
        operation
        for operation in positives
        if operation.kind == requirement.kind
        and set(requirement.target_concept_ids) <= set(operation.target_concept_ids)
    ]


def _explicit_method_present(capability: CapabilitySpec, operations: Iterable[Operation]) -> bool:
    method_concepts = set(capability.method_concept_ids)
    if not method_concepts:
        return False
    return any(
        value.concept_id in method_concepts
        for operation in operations
        for value in operation.effective_arguments.get("method", ())
    )


def _match_capability(
    capability: CapabilitySpec,
    positives: Sequence[Operation],
    contract: SemanticContractV21,
) -> tuple[list[Operation] | None, str | None]:
    """Return the positive operations that satisfy the capability, or a reason.

    A missing required argument is reported separately from a structural
    mismatch: a grounded operation for a supported capability with no usable
    required argument returns NEEDS_CONTEXT rather than NO_MATCH.
    """

    satisfying: list[Operation] = []
    missing_required_arguments = False
    explicit_method_missing = False

    for requirement in capability.operation_requirements:
        candidates = _requirement_candidates(capability, requirement, positives)
        if not candidates:
            return None, "no_matching_operation"
        with_required_paths = [
            operation
            for operation in candidates
            if all(operation.effective_arguments.get(path) for path in requirement.required_argument_paths)
        ]
        if not with_required_paths:
            missing_required_arguments = True
            if _explicit_method_present(capability, candidates):
                explicit_method_missing = True
            continue
        satisfying.extend(with_required_paths)

    if missing_required_arguments:
        if explicit_method_missing:
            return None, "explicit_method_missing_required_arguments"
        return None, "missing_required_arguments"

    if capability.basis_requirement is not None:
        basis_concept_ids = {
            value.concept_id
            for operation in satisfying
            for value in operation.effective_arguments.get("basis", ())
        }
        if not capability.basis_requirement.is_satisfied_by(basis_concept_ids):
            return None, "insufficient_basis"

    if capability.required_roles:
        role_mention_ids = set()
        for operation in satisfying:
            role_mention_ids.update(operation.target_mention_ids)
            for values in operation.effective_arguments.values():
                role_mention_ids.update(value.mention_id for value in values)
        for role_name in capability.required_roles:
            if not any(
                role.name == role_name and role.mention_id in role_mention_ids
                for role in contract.roles
            ):
                return None, f"missing_required_role:{role_name}"

    if capability.required_relations:
        role_by_id = {role.id: role for role in contract.roles}
        roles_by_mention_id: dict[str, set[str]] = {}
        roles_by_concept_id: dict[str, set[str]] = {}
        for role in contract.roles:
            roles_by_mention_id.setdefault(role.mention_id, set()).add(role.name)
            roles_by_concept_id.setdefault(role.concept_id, set()).add(role.name)

        def endpoint_roles(endpoint) -> set[str]:
            if endpoint.endpoint_type == "role_id":
                role = role_by_id.get(endpoint.value)
                return {role.name} if role is not None else set()
            if endpoint.endpoint_type == "mention_id":
                return roles_by_mention_id.get(endpoint.value, set())
            if endpoint.endpoint_type == "concept_id":
                return roles_by_concept_id.get(endpoint.value, set())
            return set()

        for requirement in capability.required_relations:
            matched = False
            for relation in contract.relations:
                if relation.predicate != requirement.predicate:
                    continue
                if (
                    requirement.subject_role in endpoint_roles(relation.subject)
                    and requirement.object_role in endpoint_roles(relation.object)
                ):
                    matched = True
                    break
            if not matched:
                return None, f"missing_required_relation:{requirement.predicate}"

    return satisfying, None


def _vetoing_operations(
    capability: CapabilitySpec,
    contract: SemanticContractV21,
    satisfying: Sequence[Operation],
) -> list[Operation]:
    vetoes: list[Operation] = []
    for requirement in capability.veto_requirements:
        for operation in contract.operations:
            if operation.polarity != "NEGATIVE":
                continue
            if operation.kind != requirement.kind:
                continue
            if set(operation.target_concept_ids) != {requirement.target_concept_id}:
                continue
            if operation.clause_scope != requirement.clause_scope:
                continue
            if requirement.clause_scope == "SAME_CLAUSE" and not any(
                positive.clause_id == operation.clause_id for positive in satisfying
            ):
                continue
            vetoes.append(operation)
    return vetoes


def route_semantic_contract(
    contract: SemanticContractV21,
    catalog: SemanticCatalog = DEFAULT_SEMANTIC_CATALOG,
) -> RouteOutcome:
    """Map a validated contract to a capability outcome without text heuristics."""

    _require_anchored_contract(contract, catalog)
    domain = contract.domain
    all_operation_ids, all_clause_ids = _operation_references(contract.operations)

    if domain == "GENERAL_CONVERSATION":
        return RouteOutcome(
            status="GENERAL_CONVERSATION",
            domain=domain,
            decision_basis=_decision_basis(
                contract,
                operation_ids=all_operation_ids,
                clause_ids=all_clause_ids,
            ),
        )

    if contract.semantic_status == "UNSUPPORTED":
        return RouteOutcome(
            status="UNSUPPORTED",
            domain=domain,
            decision_basis=_decision_basis(
                contract,
                operation_ids=all_operation_ids,
                clause_ids=all_clause_ids,
            ),
        )

    if contract.semantic_status == "CONFLICT":
        return RouteOutcome(
            status="CONFLICT",
            domain=domain,
            clarification_required=True,
            decision_basis=_decision_basis(
                contract,
                operation_ids=all_operation_ids,
                clause_ids=all_clause_ids,
            ),
        )

    if contract.semantic_status == "AMBIGUOUS":
        return RouteOutcome(
            status="AMBIGUOUS",
            domain=domain,
            clarification_required=True,
            decision_basis=_decision_basis(
                contract,
                operation_ids=all_operation_ids,
                clause_ids=all_clause_ids,
            ),
        )

    if contract.semantic_status == "MISSING":
        return RouteOutcome(
            status="NEEDS_CONTEXT",
            domain=domain,
            clarification_required=True,
            decision_basis=_decision_basis(
                contract,
                operation_ids=all_operation_ids,
                clause_ids=all_clause_ids,
                rejection_reasons=("semantic_status_missing",),
            ),
        )

    if contract.request_frame.modality != "FACTUAL":
        return _non_executable_outcome(contract, domain, "hypothetical_not_executable")
    if contract.request_frame.request_mode == "ASK":
        return _non_executable_outcome(contract, domain, "request_mode_ask")

    positives = _positive_operations(contract)
    eligible: list[tuple[CapabilitySpec, list[Operation]]] = []
    needing_context: list[str] = []
    rejected: list[str] = []
    reasons: list[str] = []
    considered: list[str] = []
    relevant_operations: list[Operation] = list(contract.operations)

    for capability in catalog.capabilities:
        satisfying, reason = _match_capability(capability, positives, contract)
        if satisfying is None:
            if reason in _CONTEXT_REQUIRED_REASONS:
                considered.append(capability.capability_id)
                needing_context.append(capability.capability_id)
                reasons.append(f"{capability.capability_id}:{reason}")
            elif reason and reason != "no_matching_operation":
                considered.append(capability.capability_id)
                rejected.append(capability.capability_id)
                reasons.append(f"{capability.capability_id}:{reason}")
            continue
        vetoes = _vetoing_operations(capability, contract, satisfying)
        if vetoes:
            considered.append(capability.capability_id)
            rejected.append(capability.capability_id)
            reasons.append(f"{capability.capability_id}:vetoed_by_negative_operation")
            relevant_operations.extend(vetoes)
            continue
        considered.append(capability.capability_id)
        eligible.append((capability, satisfying))

    operation_ids = [operation.id for operation in relevant_operations]
    clause_ids = [operation.clause_id for operation in relevant_operations]

    if len(eligible) == 1 and not needing_context:
        capability, satisfying = eligible[0]
        route_spec = next(
            (route for route in catalog.routes if route.route_id == capability.route_id),
            None,
        )
        if route_spec is None:
            raise ValueError("selected capability references a route outside the catalog")
        selected_operation_ids = [operation.id for operation in satisfying]
        selected_clause_ids = [operation.clause_id for operation in satisfying]
        capability_reference = (CapabilityReference(capability_id=capability.capability_id),)
        return RouteOutcome(
            status="SELECTED",
            domain=domain,
            route_id=capability.route_id,
            capability_ids=capability_reference,
            route_capability_ids=route_spec.capability_ids,
            decision_basis=_decision_basis(
                contract,
                capability_ids=(capability.capability_id,),
                operation_ids=selected_operation_ids,
                clause_ids=selected_clause_ids,
                effective_arguments_digest=_effective_arguments_digest(satisfying),
                rejected_candidates=rejected,
                rejection_reasons=reasons,
            ),
        )

    if eligible or needing_context:
        status = "AMBIGUOUS" if eligible else "NEEDS_CONTEXT"
        if len(eligible) > 1:
            reasons.append("multiple_capabilities")
        # Only expose choice candidates when every candidate is independently
        # executable. A capability that still needs context must not be
        # presented as a selectable template before its required evidence is
        # supplied.
        candidate_capability_ids = (
            _capability_references(item.capability_id for item, _ in eligible)
            if len(eligible) > 1 and not needing_context
            else ()
        )
        return RouteOutcome(
            status=status,
            domain=domain,
            candidate_capability_ids=candidate_capability_ids,
            clarification_required=True,
            decision_basis=_decision_basis(
                contract,
                capability_ids=considered,
                operation_ids=operation_ids,
                clause_ids=clause_ids,
                rejected_candidates=rejected,
                rejection_reasons=reasons,
            ),
        )

    if not positives:
        reasons.append("no_executable_operation")
    return RouteOutcome(
        status="NO_MATCH",
        domain=domain,
        decision_basis=_decision_basis(
            contract,
            operation_ids=operation_ids,
            clause_ids=clause_ids,
            rejected_candidates=rejected,
            rejection_reasons=reasons,
        ),
    )
