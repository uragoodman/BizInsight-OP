"""Tenant enrollment and sampling controls for semantic complex-analysis routing."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any

from app.context.semantic_catalog_v2 import (
    DEFAULT_SEMANTIC_CATALOG,
    RouteOutcome,
    SemanticCatalog,
)
from app.context.semantic_contract_v2 import (
    CapabilityReference,
    SealedRouteDecision,
    SemanticContractV21,
)


@dataclass(frozen=True, slots=True)
class CanaryEnrollment:
    eligible: bool
    reason_code: str
    sample_rate: float


def evaluate_canary_enrollment(settings: Any, tenant_context: Any) -> CanaryEnrollment:
    """Check operator enrollment and stable user sampling without reading text."""

    if getattr(settings, "complex_route_mode", "off") != "active":
        return CanaryEnrollment(False, "route_mode_not_active", 0.0)
    if not bool(getattr(settings, "complex_route_canary_enabled", False)):
        return CanaryEnrollment(False, "canary_disabled", 0.0)
    if getattr(settings, "semantic_interpreter_mode", "off") != "active":
        return CanaryEnrollment(False, "interpreter_not_active", 0.0)
    if not bool(getattr(settings, "semantic_interpreter_send_raw_question", False)):
        return CanaryEnrollment(False, "raw_question_not_permitted", 0.0)
    if tenant_context is None:
        return CanaryEnrollment(False, "tenant_context_missing", 0.0)

    tenant_id = str(getattr(tenant_context, "tenant_id", "") or "")
    user_id = str(getattr(tenant_context, "user_id", "") or "")
    allowed_tenants = {
        str(value).strip()
        for value in (getattr(settings, "complex_route_canary_tenants", ()) or ())
        if str(value).strip()
    }
    allowed_templates = {
        str(value).strip()
        for value in (getattr(settings, "complex_route_canary_templates", ()) or ())
        if str(value).strip()
    }
    if not tenant_id or (tenant_id not in allowed_tenants and "*" not in allowed_tenants):
        return CanaryEnrollment(False, "tenant_not_allowlisted", 0.0)
    if not user_id:
        return CanaryEnrollment(False, "canary_subject_missing", 0.0)
    known_plan_templates = {
        capability.template_id
        for capability in DEFAULT_SEMANTIC_CATALOG.capabilities
        if capability.template_id
    }
    if not allowed_templates:
        return CanaryEnrollment(False, "template_allowlist_empty", 0.0)
    if not allowed_templates <= known_plan_templates:
        return CanaryEnrollment(False, "template_allowlist_invalid", 0.0)

    try:
        sample_rate = float(getattr(settings, "complex_route_canary_sample_rate", 0.0))
    except (TypeError, ValueError):
        sample_rate = 0.0
    if not 0.0 < sample_rate <= 1.0:
        return CanaryEnrollment(False, "sampling_disabled", 0.0)
    if sample_rate < 1.0:
        salt = str(getattr(settings, "complex_route_canary_salt", "") or "")
        if len(salt.encode("utf-8")) < 32:
            return CanaryEnrollment(False, "sampling_salt_unconfigured", sample_rate)

        # Assign the same authenticated user to the same cohort across turns
        # and workers. IDs and the HMAC bucket are never emitted to telemetry.
        material = f"semantic-route-canary-v1:{tenant_id}:{user_id}".encode("utf-8")
        bucket = int.from_bytes(
            hmac.new(salt.encode("utf-8"), material, hashlib.sha256).digest()[:8],
            "big",
        )
        if bucket >= sample_rate * (1 << 64):
            return CanaryEnrollment(False, "sample_miss", sample_rate)
    return CanaryEnrollment(True, "eligible", sample_rate)


def template_is_canary_allowlisted(
    template_id: str | None,
    settings: Any,
    *,
    catalog: SemanticCatalog = DEFAULT_SEMANTIC_CATALOG,
) -> bool:
    """Allow only a known, plan-creating catalog template explicitly enrolled."""

    if not template_id:
        return False
    allowed = {
        str(value).strip()
        for value in (getattr(settings, "complex_route_canary_templates", ()) or ())
        if str(value).strip()
    }
    template_ids = {
        capability.template_id
        for capability in catalog.capabilities
        if capability.template_id
    }
    if template_id not in allowed or template_id not in template_ids:
        return False
    capability = next(
        (item for item in catalog.capabilities if item.template_id == template_id),
        None,
    )
    if capability is None:
        return False
    route = next((item for item in catalog.routes if item.route_id == capability.route_id), None)
    return bool(route and route.creates_plan and route.database_access)


def issue_sealed_route_decision(
    contract: SemanticContractV21,
    outcome: RouteOutcome,
    settings: Any,
    *,
    catalog: SemanticCatalog = DEFAULT_SEMANTIC_CATALOG,
) -> SealedRouteDecision:
    """Materialize only a uniquely selected, allowlisted catalog decision."""

    if not isinstance(contract, SemanticContractV21):
        raise TypeError("a validated semantic contract is required")
    if not isinstance(outcome, RouteOutcome) or outcome.status != "SELECTED":
        raise ValueError("only a selected route can be sealed")
    if (
        contract.semantic_status != "CONFIRMED"
        or contract.request_frame.request_mode != "EXECUTE"
        or contract.request_frame.modality != "FACTUAL"
    ):
        raise ValueError("non-confirmed or non-executable contract cannot be sealed")
    if contract.catalog_snapshot_hash != catalog.snapshot_hash():
        raise ValueError("contract catalog hash is stale")
    if outcome.decision_basis.catalog_snapshot_hash != contract.catalog_snapshot_hash:
        raise ValueError("route decision catalog hash is stale")
    if outcome.decision_basis.contract_version != contract.contract_version:
        raise ValueError("route decision contract version is stale")
    if len(outcome.capability_ids) != 1 or len(outcome.decision_basis.capability_ids) != 1:
        raise ValueError("only one selected capability can be sealed")

    capability_id = outcome.capability_ids[0].capability_id
    if outcome.decision_basis.capability_ids != (CapabilityReference(capability_id=capability_id),):
        raise ValueError("route decision basis does not match selected capability")
    capability = next(
        (item for item in catalog.capabilities if item.capability_id == capability_id),
        None,
    )
    if capability is None or capability.route_id != outcome.route_id:
        raise ValueError("selected capability is outside the route catalog")
    if not template_is_canary_allowlisted(capability.template_id, settings, catalog=catalog):
        raise ValueError("selected template is not enrolled in the canary")

    operations_by_id = {item.id: item for item in contract.operations}
    clauses = {item.id for item in contract.clauses}
    selected_operations = [
        operations_by_id.get(operation_id)
        for operation_id in outcome.decision_basis.operation_ids
    ]
    if (
        not selected_operations
        or any(
            operation is None
            or operation.polarity != "POSITIVE"
            or operation.mode != "EXECUTE"
            for operation in selected_operations
        )
        or any(clause_id not in clauses for clause_id in outcome.decision_basis.clause_ids)
        or not outcome.decision_basis.effective_arguments_digest
    ):
        raise ValueError("sealed decision basis is incomplete or non-executable")

    return SealedRouteDecision.issue(
        contract_version=contract.contract_version,
        catalog_snapshot_hash=contract.catalog_snapshot_hash,
        route_decision_status="SELECTED",
        route_id=str(outcome.route_id),
        capability_id=capability_id,
        template_id=str(capability.template_id),
        operation_ids=outcome.decision_basis.operation_ids,
        clause_ids=outcome.decision_basis.clause_ids,
        effective_arguments_digest=outcome.decision_basis.effective_arguments_digest,
        decision_basis=outcome.decision_basis,
    )
