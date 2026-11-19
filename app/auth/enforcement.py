"""Security-boundary checks shared by planner, HITL, RAG and MCP paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.auth.context import AuthorizationContext, AuthorizationDenied
from app.chat_lifecycle.errors import ExecutionErrorCode


@dataclass(frozen=True, slots=True)
class AuthorizationBoundary:
    """The authorization values captured when an execution was authorized."""

    permission_version: str
    policy_set_version: str
    authorization_fingerprint: str
    query_scope_fingerprint: str


class StaleAuthorization(AuthorizationDenied):
    """Raised when an execution crosses a boundary with stale authorization."""

    def __init__(self, message: str, *, code: ExecutionErrorCode = ExecutionErrorCode.AUTH_STALE) -> None:
        super().__init__(code.value, message)


def coerce_authorization_boundary(value: AuthorizationBoundary | dict[str, Any]) -> AuthorizationBoundary:
    """Drop non-boundary lifecycle metadata before boundary validation."""
    if isinstance(value, AuthorizationBoundary):
        return value
    fields = {
        "permission_version",
        "policy_set_version",
        "authorization_fingerprint",
        "query_scope_fingerprint",
    }
    return AuthorizationBoundary(**{
        key: str(value.get(key) or "")
        for key in fields
    })


def assert_authorization_boundary(
    authz: AuthorizationContext,
    boundary: AuthorizationBoundary,
    *,
    current_scope_fingerprint: str,
) -> None:
    """Fail closed when auth or query scope changed since authorization.

    This deliberately compares immutable server values.  A client/LLM cannot
    repair a stale boundary by sending a new fingerprint.
    """

    from app.observability import record_event

    def stale(reason: str, code: ExecutionErrorCode = ExecutionErrorCode.AUTH_STALE) -> None:
        record_event("authz.boundary.stale", {
            "tenant_id": authz.tenant_id,
            "user_id": authz.user_id,
            "authz_ref": authz.authz_ref,
            "reason": reason,
            "code": code.value,
        })
        raise StaleAuthorization(reason, code=code)

    if authz.permission_version != boundary.permission_version:
        stale("permission version changed")
    if authz.policy_set_version != boundary.policy_set_version:
        stale("policy set version changed")
    if authz.authz_digest != boundary.authorization_fingerprint:
        stale("authorization fingerprint changed")
    if current_scope_fingerprint != boundary.query_scope_fingerprint:
        stale("query scope fingerprint changed", ExecutionErrorCode.SCOPE_STALE)
