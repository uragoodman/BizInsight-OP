"""Trusted tenant context propagated through API, graph, MCP and repositories.

Tenant identifiers are deliberately never read from request payloads, prompts, or
tool arguments.  The API authentication layer creates this immutable value and
activates it for the lifetime of a request/workflow.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Iterator, Literal


PLATFORM_SCOPE = "platform"
SELLER_SCOPE = "seller"
TenantScope = Literal["platform", "seller"]
TenantRole = Literal["owner", "analyst", "reviewer", "viewer", "admin"]


@dataclass(frozen=True, slots=True)
class SellerGrant:
    """One server-issued seller entitlement, optionally bounded in time."""

    seller_id: str
    start_date: str | None = None
    end_date: str | None = None


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Authorization facts derived from a credential, never caller supplied."""

    tenant_id: str
    user_id: str
    role: TenantRole
    scope_type: TenantScope
    seller_id: str | None = None
    seller_grants: tuple[SellerGrant, ...] = ()
    # Opaque server-side session reference used to refresh authentication at
    # the MCP boundary. Never serialize it into graph state or audit payloads.
    auth_session_id: str | None = field(default=None, repr=False, compare=False)
    # API credentials are similarly revalidated at the data-access boundary.
    auth_credential_id: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.user_id:
            raise ValueError("tenant_id and user_id are required")
        if self.scope_type == SELLER_SCOPE and not (self.seller_id or self.seller_grants):
            raise ValueError("seller_id or seller grant is required for seller scope")
        if self.scope_type == PLATFORM_SCOPE and self.seller_id is not None:
            raise ValueError("platform scope must not carry seller_id")
        if self.scope_type == PLATFORM_SCOPE and self.role != "admin":
            raise ValueError("platform scope requires admin role")

    @property
    def is_platform_admin(self) -> bool:
        return self.scope_type == PLATFORM_SCOPE and self.role == "admin"

    @property
    def allowed_seller_ids(self) -> tuple[str, ...]:
        ids = [grant.seller_id for grant in self.seller_grants]
        if self.seller_id and self.seller_id not in ids:
            ids.append(self.seller_id)
        return tuple(ids)

    def seller_grant(self, seller_id: str) -> SellerGrant | None:
        for grant in self.seller_grants:
            if grant.seller_id == seller_id:
                return grant
        if self.seller_id == seller_id:
            return SellerGrant(seller_id=seller_id)
        return None

    def as_state(self) -> dict[str, str | None]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "user_role": self.role,
            "scope_type": self.scope_type,
            "seller_id": self.seller_id,
        }


_CURRENT: ContextVar[TenantContext | None] = ContextVar("tenant_context", default=None)


def current_tenant_context() -> TenantContext | None:
    return _CURRENT.get()


def require_tenant_context() -> TenantContext:
    context = current_tenant_context()
    if context is None:
        raise PermissionError("A trusted TenantContext is required")
    return context


@contextmanager
def activate_tenant_context(context: TenantContext) -> Iterator[TenantContext]:
    token: Token[TenantContext | None] = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)
