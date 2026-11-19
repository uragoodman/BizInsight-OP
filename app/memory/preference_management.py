"""Tenant-scoped read, retry and setting operations for user preferences."""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from app.core.config import get_settings
from app.data.database import engine
from app.data.rls import set_local_context
from app.memory.mem0_adapter import Mem0PreferenceAdapter, Mem0Error
from app.memory.preference_payload import decrypt_preference_payload, payload_key_material
from app.memory.preference_registry import PostgresPreferenceRegistry
from app.tenancy import PLATFORM_SCOPE, SELLER_SCOPE, SellerGrant, TenantContext


class PreferenceManagementError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _management_context_for_seller(
    context: TenantContext, seller_id: str | None,
) -> TenantContext:
    """Bind Mem0 management reads to this user's stored seller partition.

    This context is used only to read/manage the same authenticated user's own
    preference record. Runtime retrieval continues to use the original,
    currently authorized request context.
    """
    if context.scope_type == PLATFORM_SCOPE:
        if seller_id is None:
            return context
        return TenantContext(
            tenant_id=context.tenant_id, user_id=context.user_id, role="admin",
            scope_type=SELLER_SCOPE, seller_id=seller_id,
            seller_grants=(SellerGrant(seller_id),),
        )
    if seller_id is not None:
        return replace(context, seller_id=seller_id)
    if context.seller_grants:
        return replace(context, seller_id=None)
    # Keep a valid seller context while selecting the unscoped preference ID.
    return replace(
        context, seller_id=None, seller_grants=(SellerGrant("__unscoped_management__"),),
    )


def list_preferences(
    context: TenantContext, *, chat_session_id: str | None = None,
    engine_override: Any | None = None,
) -> list[dict[str, Any]]:
    db = engine_override or engine
    with db.begin() as connection:
        set_local_context(connection, context, scope_type=context.scope_type)
        rows = connection.execute(text("""
            SELECT registry_id, seller_id, memory_type, memory_key, mem0_memory_id,
                   status, version, operation_id, source_turn_id, updated_at,
                   expires_at
              FROM governance.memory_registry r
             WHERE tenant_id=:tenant_id AND user_id=:user_id
               AND (
                   status IN ('active','pending_create','pending_update','pending_delete','failed')
                   OR (
                       memory_type='scope_default' AND status='superseded'
                       AND expires_at <= clock_timestamp()
                       AND NOT EXISTS (
                           SELECT 1 FROM governance.memory_registry newer
                            WHERE newer.tenant_id=r.tenant_id
                              AND newer.user_id=r.user_id
                              AND newer.seller_id IS NOT DISTINCT FROM r.seller_id
                              AND newer.memory_key=r.memory_key
                              AND newer.version > r.version
                       )
                   )
               )
             ORDER BY updated_at DESC, version DESC
        """), {
            "tenant_id": context.tenant_id, "user_id": context.user_id,
        }).mappings().all()
        overlays = []
        if chat_session_id:
            overlays = connection.execute(text("""
                SELECT overlay_id, registry_id, memory_key, memory_type, value,
                       source_turn_id, operation_id, status, error_code,
                       expires_at, created_at
                  FROM governance.memory_preference_overlays
                 WHERE tenant_id=:tenant_id AND user_id=:user_id
                   AND chat_session_id=:session_id
                   AND status IN ('pending','failed')
                   AND expires_at > clock_timestamp()
                 ORDER BY created_at DESC
            """), {
                "tenant_id": context.tenant_id, "user_id": context.user_id,
                "session_id": chat_session_id,
            }).mappings().all()
    overlay_by_registry = {str(row["registry_id"]): dict(row) for row in overlays}
    adapters: dict[str | None, Mem0PreferenceAdapter] = {}
    result: list[dict[str, Any]] = []
    for item in rows:
        row = dict(item)
        expires_at = row.get("expires_at")
        requires_confirmation = bool(
            row["memory_type"] == "scope_default"
            and expires_at is not None
            and expires_at <= datetime.now(timezone.utc)
        )
        overlay = overlay_by_registry.get(str(row["registry_id"]))
        value = overlay.get("value") if overlay else None
        if (row["status"] == "active" or requires_confirmation) and row["mem0_memory_id"]:
            try:
                seller_id = str(row["seller_id"]) if row["seller_id"] else None
                if seller_id not in adapters:
                    scoped_context = _management_context_for_seller(context, seller_id)
                    registry = PostgresPreferenceRegistry(scoped_context, engine_override=db)
                    adapters[seller_id] = Mem0PreferenceAdapter(
                        scoped_context, registry=registry,
                    )
                hit = adapters[seller_id].get_validated(
                    str(row["mem0_memory_id"]), operation_id=str(row["operation_id"]),
                    memory_type=str(row["memory_type"]), memory_key=str(row["memory_key"]),
                    include_expired=requires_confirmation,
                )
                if hit is not None:
                    value = hit.value
            except Mem0Error:
                # Listing durable status remains useful while optional Mem0 is down.
                pass
        result.append({
            "registry_id": row["registry_id"], "memory_key": row["memory_key"],
            "memory_type": row["memory_type"], "value": value,
            "scope": {"seller_id": row["seller_id"]} if row["seller_id"] else {"type": "user"},
            "status": row["status"], "version": row["version"],
            "source_turn_id": row["source_turn_id"],
            "requires_confirmation": requires_confirmation,
            "updated_at": row["updated_at"], "expires_at": row["expires_at"],
            "overlay": ({
                "status": overlay["status"], "error_code": overlay["error_code"],
                "expires_at": overlay["expires_at"],
            } if overlay else None),
        })
    return result


def preference_history(
    context: TenantContext, registry_id: str, *, engine_override: Any | None = None,
) -> list[dict[str, Any]]:
    db = engine_override or engine
    with db.begin() as connection:
        set_local_context(connection, context, scope_type=context.scope_type)
        rows = connection.execute(text("""
            SELECT audit_id, operation_id, event_type, status, reason_code, created_at
              FROM governance.memory_preference_audit
             WHERE tenant_id=:tenant_id AND user_id=:user_id
               AND registry_id=:registry_id
             ORDER BY created_at DESC LIMIT 100
        """), {
            "tenant_id": context.tenant_id, "user_id": context.user_id,
            "registry_id": registry_id,
        }).mappings().all()
    return [dict(row) for row in rows]


def get_personalization_enabled(
    context: TenantContext, *, engine_override: Any | None = None,
) -> bool:
    db = engine_override or engine
    with db.begin() as connection:
        set_local_context(connection, context, scope_type=context.scope_type)
        enabled = connection.execute(text("""
            SELECT personalization_enabled
              FROM governance.memory_preference_settings
             WHERE tenant_id=:tenant_id AND user_id=:user_id
        """), {"tenant_id": context.tenant_id, "user_id": context.user_id}).scalar_one_or_none()
    return True if enabled is None else bool(enabled)


def set_personalization_enabled(
    context: TenantContext, enabled: bool, *, engine_override: Any | None = None,
) -> bool:
    db = engine_override or engine
    with db.begin() as connection:
        set_local_context(connection, context, scope_type=context.scope_type)
        connection.execute(text("""
            INSERT INTO governance.memory_preference_settings
                (tenant_id, user_id, personalization_enabled, updated_at)
            VALUES (:tenant_id, :user_id, :enabled, now())
            ON CONFLICT (tenant_id, user_id) DO UPDATE
                SET personalization_enabled=EXCLUDED.personalization_enabled,
                    updated_at=now()
        """), {
            "tenant_id": context.tenant_id, "user_id": context.user_id,
            "enabled": bool(enabled),
        })
        connection.execute(text("""
            INSERT INTO governance.memory_preference_audit
                (audit_id, tenant_id, user_id, seller_id, registry_id,
                 operation_id, event_type, actor_user_id, status, reason_code)
            VALUES (:audit_id, :tenant_id, :user_id, :seller_id, NULL,
                    :operation_id, :event_type, :actor, :status, 'user_setting_changed')
        """), {
            "audit_id": f"maud_{uuid.uuid4().hex}",
            "tenant_id": context.tenant_id, "user_id": context.user_id,
            "seller_id": context.seller_id, "operation_id": f"op_{uuid.uuid4().hex}",
            "event_type": "personalization_enabled" if enabled else "personalization_disabled",
            "actor": context.user_id, "status": "enabled" if enabled else "disabled",
        })
    return bool(enabled)


def retry_failed_preference(
    context: TenantContext, registry_id: str, *, engine_override: Any | None = None,
) -> str:
    """Requeue one failed/dead-letter operation with its stable operation ID."""
    db = engine_override or engine
    settings = get_settings()
    with db.begin() as connection:
        set_local_context(connection, context, scope_type=context.scope_type)
        row = connection.execute(text("""
            SELECT r.registry_id, r.status AS registry_status, r.operation_id,
                   r.version, r.memory_type, r.memory_key,
                   r.seller_id,
                   o.event_id, o.event_type, o.status AS outbox_status,
                   o.operation_id AS outbox_operation_id, o.payload_ciphertext,
                   o.payload_nonce, o.payload_key_version, o.overlay_id
              FROM governance.memory_registry r
              JOIN governance.memory_outbox o ON o.registry_id=r.registry_id
             WHERE r.registry_id=:registry_id AND r.tenant_id=:tenant_id
               AND r.user_id=:user_id
               AND r.status='failed' AND o.status='dead_letter'
             ORDER BY o.created_at DESC LIMIT 1 FOR UPDATE OF r,o
        """), {
            "registry_id": registry_id, "tenant_id": context.tenant_id,
            "user_id": context.user_id,
        }).mappings().first()
        if row is None:
            raise PreferenceManagementError("failed_preference_operation_not_found")
        row = dict(row)
        overlay_value: dict[str, Any] | None = None
        if row["event_type"] in {"add", "update"}:
            if not row["payload_ciphertext"] or not row["overlay_id"]:
                raise PreferenceManagementError("retry_payload_unavailable")
            try:
                key_material, _ = payload_key_material(settings)
                overlay_value = decrypt_preference_payload(
                    row["payload_ciphertext"], row["payload_nonce"],
                    key=key_material,
                    key_version=str(row["payload_key_version"]),
                    tenant_id=context.tenant_id, user_id=context.user_id,
                    operation_id=str(row["outbox_operation_id"]),
                )
            except Exception as exc:  # noqa: BLE001
                raise PreferenceManagementError("retry_payload_decryption_failed") from exc
        next_status = {
            "add": "pending_create", "update": "pending_update",
            "delete": "pending_delete", "reconcile": "pending_update",
        }.get(str(row["event_type"]))
        if next_status is None:
            raise PreferenceManagementError("unsupported_retry_event")
        enabled = connection.execute(text("""
            SELECT personalization_enabled FROM governance.memory_preference_settings
             WHERE tenant_id=:tenant_id AND user_id=:user_id
        """), {"tenant_id": context.tenant_id, "user_id": context.user_id}).scalar_one_or_none()
        if row["event_type"] in {"add", "update", "reconcile"} and enabled is False:
            raise PreferenceManagementError("personalization_disabled")
        outbox_changed = connection.execute(text("""
            UPDATE governance.memory_outbox
               SET status='pending', attempts=0, max_attempts=5,
                   next_attempt_at=now(), lease_token=NULL, lease_expires_at=NULL,
                   completed_at=NULL, last_error_code=NULL,
                   expires_at=now()+interval '30 days', updated_at=now()
             WHERE event_id=:event_id AND status='dead_letter'
        """), {"event_id": row["event_id"]}).rowcount
        if outbox_changed != 1:
            raise PreferenceManagementError("retry_operation_state_conflict")
        registry_changed = connection.execute(text("""
            UPDATE governance.memory_registry
               SET status=:status, last_error=NULL, updated_at=now()
             WHERE registry_id=:registry_id AND status='failed'
        """), {"status": next_status, "registry_id": registry_id}).rowcount
        if registry_changed != 1:
            raise PreferenceManagementError("retry_registry_state_conflict")
        if overlay_value is not None:
            overlay_changed = connection.execute(text("""
                UPDATE governance.memory_preference_overlays
                   SET status='pending', value=CAST(:value AS jsonb), error_code=NULL,
                       expires_at=now()+interval '24 hours', cleared_at=NULL
                 WHERE overlay_id=:overlay_id AND registry_id=:registry_id
            """), {
                "value": json.dumps(overlay_value, ensure_ascii=False),
                "overlay_id": row["overlay_id"], "registry_id": registry_id,
            }).rowcount
            if overlay_changed != 1:
                raise PreferenceManagementError("retry_overlay_state_conflict")
        connection.execute(text("""
            INSERT INTO governance.memory_preference_audit
                (audit_id, tenant_id, user_id, seller_id, registry_id,
                 operation_id, event_type, actor_user_id, status, reason_code)
            VALUES (:audit_id, :tenant_id, :user_id, :seller_id, :registry_id,
                    :operation_id, 'retry_requested', :actor, :status, 'user_retry')
        """), {
            "audit_id": f"maud_{uuid.uuid4().hex}", "tenant_id": context.tenant_id,
            "user_id": context.user_id, "seller_id": row["seller_id"],
            "registry_id": registry_id, "operation_id": row["outbox_operation_id"],
            "actor": context.user_id, "status": next_status,
        })
    return next_status
