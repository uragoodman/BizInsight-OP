"""Read-only preference Registry boundary for WP3.

WP4 owns the Registry schema, mutable states, Outbox and Overlay.  WP3 must not
pretend that Mem0 metadata is the Registry.  Without an installed Registry the
default implementation fails closed: Mem0 hits are not allowed into a Bundle,
and the provider reports a safe degradation reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from sqlalchemy import text

from app.data.database import engine
from app.data.rls import set_local_context
from app.tenancy import TenantContext


class RegistryUnavailable(RuntimeError):
    """The authoritative preference Registry cannot be consulted."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class RegistryDecision:
    active: bool
    reason: str | None = None


class PreferenceRegistry(Protocol):
    def validate(
        self,
        *,
        context: TenantContext,
        entity_scope: str,
        memory_id: str,
        operation_id: str | None,
        memory_type: str,
        memory_key: str,
        expect_active: bool = True,
    ) -> RegistryDecision:
        ...


class PostgresPreferenceRegistry:
    """Minimal read-only Registry boundary.

    The schema is installed by an Alembic migration. If it is unavailable,
    this adapter raises ``RegistryUnavailable`` and callers degrade without
    preferences.
    """

    def __init__(self, context: TenantContext, *, engine_override: Any | None = None) -> None:
        self.context = context
        self._engine = engine_override or engine

    def ensure_available(self) -> None:
        """Fail closed before touching Mem0 when WP4's Registry is absent."""
        try:
            with self._engine.begin() as connection:
                set_local_context(connection, self.context, scope_type=self.context.scope_type)
                connection.execute(text("SELECT 1 FROM governance.memory_registry LIMIT 0"))
        except Exception as exc:  # noqa: BLE001
            raise RegistryUnavailable(_registry_error_reason(exc)) from exc

    def list_pending_overlays(
        self, *, context: TenantContext, chat_session_id: str,
    ) -> list[dict[str, Any]]:
        """Read unexpired same-session writes for read-your-writes behavior."""
        if (context.tenant_id, context.user_id) != (
            self.context.tenant_id, self.context.user_id
        ):
            raise RegistryUnavailable("registry_identity_mismatch")
        if not chat_session_id:
            return []
        try:
            with self._engine.begin() as connection:
                set_local_context(connection, context, scope_type=context.scope_type)
                rows = connection.execute(text("""
                    SELECT overlay_id, memory_key, memory_type, value,
                           source_turn_id, operation_id, status
                      FROM governance.memory_preference_overlays
                     WHERE tenant_id=:tenant_id AND user_id=:user_id
                       AND seller_id IS NOT DISTINCT FROM :seller_id
                       AND chat_session_id=:chat_session_id
                       AND status='pending' AND expires_at > clock_timestamp()
                     ORDER BY version, created_at
                """), {
                    "tenant_id": context.tenant_id, "user_id": context.user_id,
                    "seller_id": context.seller_id,
                    "chat_session_id": chat_session_id,
                }).mappings().all()
        except Exception as exc:  # noqa: BLE001
            raise RegistryUnavailable(_registry_error_reason(exc)) from exc
        return [dict(row) for row in rows]

    def validate(
        self,
        *,
        context: TenantContext,
        entity_scope: str,
        memory_id: str,
        operation_id: str | None,
        memory_type: str,
        memory_key: str,
        expect_active: bool = True,
    ) -> RegistryDecision:
        if (context.tenant_id, context.user_id) != (
            self.context.tenant_id, self.context.user_id
        ):
            raise RegistryUnavailable("registry_identity_mismatch")
        try:
            with self._engine.begin() as connection:
                set_local_context(connection, context, scope_type=context.scope_type)
                row = connection.execute(text("""
                    SELECT status, memory_type, memory_key, mem0_memory_id,
                           operation_id, tenant_id, user_id, seller_id, expires_at
                      FROM governance.memory_registry
                    WHERE tenant_id=:tenant_id AND user_id=:user_id
                       AND seller_id IS NOT DISTINCT FROM :seller_id
                       AND mem0_memory_id=:memory_id
                       AND operation_id=:operation_id
                     LIMIT 1
                """), {
                    "tenant_id": context.tenant_id,
                    "user_id": context.user_id,
                    "seller_id": context.seller_id,
                    "memory_id": memory_id,
                    "operation_id": operation_id,
                }).mappings().first()
        except Exception as exc:  # noqa: BLE001 - missing WP4 schema is a hard capability boundary
            raise RegistryUnavailable(_registry_error_reason(exc)) from exc
        if row is None:
            return RegistryDecision(False, "registry_record_missing")
        if row["memory_type"] != memory_type or row["memory_key"] != memory_key:
            return RegistryDecision(False, "registry_type_or_key_mismatch")
        if expect_active and row["status"] != "active":
            return RegistryDecision(False, "registry_not_active")
        expires_at = row.get("expires_at")
        if expect_active and expires_at is not None and expires_at < datetime.now(timezone.utc):
            return RegistryDecision(False, "registry_expired")
        return RegistryDecision(True, None)


def _registry_error_reason(exc: Exception) -> str:
    reason = f"{type(exc).__name__} {exc}".lower()
    if "undefinedtable" in reason or "does not exist" in reason or "no such table" in reason:
        return "registry_table_missing"
    return "registry_query_failed"


class NullPreferenceRegistry:
    """Explicit no-Registry mode: every Mem0 hit is rejected."""

    def validate(self, **_: Any) -> RegistryDecision:
        raise RegistryUnavailable("registry_not_configured")
