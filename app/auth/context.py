"""Runtime authorization context derived from IAM and policy registries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from typing import Iterable

from app.tenancy import SellerGrant, TenantContext


class AuthorizationDenied(PermissionError):
    """A fail-closed authorization decision."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class AuthorizationContext:
    """Server-derived authorization decision for one tenant/user snapshot."""

    tenant_id: str
    user_id: str
    role: str
    scope_type: str
    permission_version: str
    policy_set_version: str
    authz_ref: str
    authz_digest: str
    allowed_seller_ids: tuple[str, ...]
    seller_grants: tuple[SellerGrant, ...]
    allowed_metric_ids: tuple[str, ...]
    allowed_dimensions: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    allowed_classifications: tuple[str, ...] = ("public", "internal")
    allowed_multi_dimension_dimensions: tuple[str, ...] = ()

    def seller_grant(self, seller_id: str) -> SellerGrant | None:
        return next(
            (grant for grant in self.seller_grants if grant.seller_id == seller_id),
            None,
        )

    def require_action(self, action: str) -> None:
        if action not in self.allowed_actions:
            raise AuthorizationDenied("AUTH_DENIED", f"Action {action!r} is not authorized.")

    def require_sellers(self, seller_ids: Iterable[str]) -> None:
        requested = {str(value) for value in seller_ids}
        if self.scope_type == "platform" and self.role == "admin":
            return
        denied = sorted(requested - set(self.allowed_seller_ids))
        if denied:
            raise AuthorizationDenied(
                "AUTH_DENIED",
                f"Seller scope is not authorized: {', '.join(denied)}",
            )

    def require_metrics(self, metric_ids: Iterable[str]) -> None:
        denied = sorted({str(value) for value in metric_ids} - set(self.allowed_metric_ids))
        if denied:
            raise AuthorizationDenied(
                "AUTH_DENIED",
                f"Metric scope is not authorized: {', '.join(denied)}",
            )

    def require_dimensions(self, dimensions: Iterable[str]) -> None:
        denied = sorted({str(value) for value in dimensions} - set(self.allowed_dimensions))
        if denied:
            raise AuthorizationDenied(
                "AUTH_DENIED",
                f"Dimension scope is not authorized: {', '.join(denied)}",
            )

    def require_multi_dimension_dimensions(self, dimensions: Iterable[str]) -> None:
        denied = sorted(
            {str(value) for value in dimensions}
            - set(self.allowed_multi_dimension_dimensions)
        )
        if denied:
            raise AuthorizationDenied(
                "AUTH_DENIED",
                f"Multi-dimension scope is not authorized: {', '.join(denied)}",
            )

    def require_tool(self, tool_name: str) -> None:
        if tool_name not in self.allowed_tools:
            raise AuthorizationDenied("TOOL_DENIED", f"Tool {tool_name!r} is not authorized.")

    def require_date_range(self, seller_id: str, start: date, end: date) -> None:
        if end < start:
            raise AuthorizationDenied("SCOPE_INVALID", "The end date precedes the start date.")
        grant = self.seller_grant(seller_id)
        if self.scope_type == "platform" and self.role == "admin":
            return
        if grant is None and self.allowed_seller_ids:
            raise AuthorizationDenied("AUTH_DENIED", "Seller grant is missing.")
        grant_start = date.fromisoformat(grant.start_date) if grant and grant.start_date else None
        grant_end = date.fromisoformat(grant.end_date) if grant and grant.end_date else None
        if grant_start and start < grant_start:
            raise AuthorizationDenied("AUTH_DENIED", "Start date is before seller entitlement.")
        if grant_end and end > grant_end:
            raise AuthorizationDenied("AUTH_DENIED", "End date is after seller entitlement.")

    def as_audit_dict(self) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "role": self.role,
            "scope_type": self.scope_type,
            "permission_version": self.permission_version,
            "policy_set_version": self.policy_set_version,
            "authz_ref": self.authz_ref,
            "authz_digest": self.authz_digest,
            "seller_scope_type": "explicit",
            "seller_grant_count": len(self.seller_grants),
            "allowed_classifications": list(self.allowed_classifications),
        }


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _policy_tools(role: str, tenant_id: str, scope_type: str) -> tuple[str, ...]:
    from app.mcp.tools import get_supported_tools
    from app.security.governance import GOVERNED_HTTP_CAPABILITIES, ToolPermissionPolicy

    policy = ToolPermissionPolicy()
    return tuple(sorted(
        tool
        for tool in (*get_supported_tools(), *GOVERNED_HTTP_CAPABILITIES)
        if policy.allows(tenant_id, role, "data_query", tool, scope_type=scope_type)
    ))


def _registry_metrics(scope_type: str) -> tuple[str, ...]:
    from app.metrics.registry import get_registry

    registry = get_registry()
    result: list[str] = []
    for definition in registry.list():
        try:
            if registry.supports(definition.metric_id, "value", scope=scope_type):
                result.append(definition.metric_id)
        except Exception:
            continue
    return tuple(sorted(set(result)))


def build_authorization_context(
    tenant_context: TenantContext,
    *,
    allowed_metric_ids: Iterable[str] | None = None,
    allowed_dimensions: Iterable[str] | None = None,
    allowed_multi_dimension_dimensions: Iterable[str] | None = None,
    allowed_tools: Iterable[str] | None = None,
    metric_scope_type: str | None = None,
) -> AuthorizationContext:
    """Derive a runtime snapshot; optional collections can only narrow it."""

    from app.iam.service import permissions

    actions = tuple(sorted(permissions(tenant_context)))
    metrics = set(_registry_metrics(metric_scope_type or tenant_context.scope_type))
    if allowed_metric_ids is not None:
        metrics.intersection_update(str(item) for item in allowed_metric_ids)

    dimensions = {
        "product_category",
        "payment_type",
        "month_position",
        "day_of_week",
        "hour_of_day",
        "quarter_of_year",
        "half_of_year",
    }
    from app.core.config import get_settings
    if (
        tenant_context.scope_type in {"platform", "seller"}
        and get_settings().seller_dimension_enabled
    ):
        dimensions.add("seller")
    if allowed_dimensions is not None:
        dimensions.intersection_update(str(item) for item in allowed_dimensions)

    # Multi-axis queries are a separate aggregate capability. They inherit the
    # ordinary safe dimensions plus customer_state for platform administrators
    # only. Customer city and seller state remain excluded. Intersect with the
    # repository's SQL expression map so policy cannot advertise missing code.
    from app.data.repositories.multidimensional import MULTI_DIMENSION_QUERY_DIMENSIONS

    multi_dimensions = set(dimensions) & set(MULTI_DIMENSION_QUERY_DIMENSIONS)
    if tenant_context.is_platform_admin:
        multi_dimensions.add("customer_state")
    multi_dimensions.intersection_update(MULTI_DIMENSION_QUERY_DIMENSIONS)
    if allowed_dimensions is not None:
        multi_dimensions.intersection_update(str(item) for item in allowed_dimensions)
    if allowed_multi_dimension_dimensions is not None:
        multi_dimensions.intersection_update(
            str(item) for item in allowed_multi_dimension_dimensions
        )

    tools = set(_policy_tools(
        tenant_context.role,
        tenant_context.tenant_id,
        tenant_context.scope_type,
    ))
    if allowed_tools is not None:
        tools.intersection_update(str(item) for item in allowed_tools)

    classifications = {"public", "internal"}
    if tenant_context.is_platform_admin or tenant_context.role in {"owner", "admin"}:
        classifications.add("confidential")

    grants = tuple(sorted(tenant_context.seller_grants, key=lambda item: item.seller_id))
    permission_payload = {
        "tenant_id": tenant_context.tenant_id,
        "user_id": tenant_context.user_id,
        "role": tenant_context.role,
        "seller_grants": [
            {
                "seller_id": item.seller_id,
                "start_date": item.start_date,
                "end_date": item.end_date,
            }
            for item in grants
        ],
        "actions": actions,
    }
    permission_version = _digest(permission_payload)
    policy_payload = {
        "metrics": sorted(metrics),
        "dimensions": sorted(dimensions),
        "multi_dimension_dimensions": sorted(multi_dimensions),
        "tools": sorted(tools),
    }
    policy_set_version = _digest(policy_payload)
    authz_digest = _digest({
        **permission_payload,
        **policy_payload,
        "permission_version": permission_version,
        "policy_set_version": policy_set_version,
    })
    authz_ref = f"authz:{permission_version[:16]}:{authz_digest[:16]}"

    result = AuthorizationContext(
        tenant_id=tenant_context.tenant_id,
        user_id=tenant_context.user_id,
        role=tenant_context.role,
        scope_type=tenant_context.scope_type,
        permission_version=permission_version,
        policy_set_version=policy_set_version,
        authz_ref=authz_ref,
        authz_digest=authz_digest,
        allowed_seller_ids=tuple(sorted(tenant_context.allowed_seller_ids)),
        seller_grants=grants,
        allowed_metric_ids=tuple(sorted(metrics)),
        allowed_dimensions=tuple(sorted(dimensions)),
        allowed_multi_dimension_dimensions=tuple(sorted(multi_dimensions)),
        allowed_actions=actions,
        allowed_tools=tuple(sorted(tools)),
        allowed_classifications=tuple(sorted(classifications)),
    )
    # Runtime event only records references/digests.  It intentionally does
    # not persist the complete allow-list or expose it to the client.
    from app.observability import record_event
    record_event("authz.context.created", result.as_audit_dict())
    return result
