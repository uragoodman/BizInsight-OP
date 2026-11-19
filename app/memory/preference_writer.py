"""Transactional WP4 preference write queue.

This module accepts an already policy-approved, explicitly user-confirmed,
typed preference. It atomically records the Registry version, same-session
overlay, Outbox reference, and metadata-only audit event. It never calls Mem0;
external delivery belongs to the lease worker.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from app.core.config import get_settings
from app.data.database import engine
from app.data.rls import set_local_context
from app.memory.preference_payload import encrypt_preference_payload, payload_key_material
from app.memory.policy import MemoryPolicyDecision
from app.tenancy import TenantContext


_CONTEXT_SELLER = object()


class PreferenceWriteError(RuntimeError):
    """Base class for safe, user-presentable preference write errors."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class PreferenceWriteConflict(PreferenceWriteError):
    """The Registry version changed or the target is no longer active."""


class PreferenceWriteRejected(PreferenceWriteError):
    """The request lacks a required policy, identity, or session fact."""


@dataclass(frozen=True)
class QueuedPreferenceOperation:
    registry_id: str
    operation_id: str
    outbox_event_id: str
    overlay_id: str | None
    memory_key: str
    memory_type: str
    version: int
    status: str
    changed: bool = True


class PreferenceWriter:
    """PostgreSQL transaction boundary for accepted preference mutations."""

    def __init__(
        self, context: TenantContext, *, engine_override: Any | None = None,
        payload_key: str | bytes | None = None,
    ) -> None:
        if not isinstance(context, TenantContext):
            raise PreferenceWriteRejected("trusted_tenant_context_required")
        self.context = context
        self._engine = engine_override or engine
        self._payload_key = payload_key

    @staticmethod
    def _failed_delete_targets(
        row: dict[str, Any], failed_operation: dict[str, Any],
        *, previous_memory_id: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Resolve only fixed IDs/operation IDs from Registry and dead-letter state."""
        event_type = failed_operation.get("event_type")
        if event_type == "delete":
            return failed_operation.get("target_memory_id"), failed_operation.get("target_operation_id")
        memory_id = row.get("mem0_memory_id")
        if event_type == "update":
            memory_id = previous_memory_id or memory_id
        return memory_id, row.get("operation_id")

    def queue_upsert(
        self,
        *,
        memory_key: str,
        memory_type: str,
        value: dict[str, Any],
        chat_session_id: str,
        source_turn_id: str | None,
        source_trace_id: str | None = None,
        expected_version: int,
        idempotency_key: str,
        policy_decision: MemoryPolicyDecision,
        seller_id: str | None | object = _CONTEXT_SELLER,
    ) -> QueuedPreferenceOperation:
        """Queue an explicitly confirmed create/update after policy validation.

        The caller passes only a typed value. The Outbox contains references,
        never preference text; the worker renders the Mem0 text from this value.
        """
        self._validate_mutation(
            memory_key=memory_key, memory_type=memory_type, value=value,
            chat_session_id=chat_session_id, context=self.context,
            policy_decision=policy_decision,
        )
        if expected_version < 0:
            raise PreferenceWriteRejected("expected_version_invalid")
        self._validate_idempotency_key(idempotency_key)
        seller_scope = self.context.seller_id if seller_id is _CONTEXT_SELLER else seller_id
        if seller_scope is not None and not isinstance(seller_scope, str):
            raise PreferenceWriteRejected("seller_scope_invalid")
        operation_id = uuid.uuid4().hex
        registry_id = f"mreg_{uuid.uuid4().hex}"
        overlay_id = f"ovl_{uuid.uuid4().hex}"
        event_id = f"mout_{uuid.uuid4().hex}"
        audit_id = f"maud_{uuid.uuid4().hex}"
        canonical_value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        value_hash = hashlib.sha256(canonical_value.encode("utf-8")).hexdigest()
        settings = get_settings()
        key_material, key_version = payload_key_material(settings, override=self._payload_key)
        encrypted = encrypt_preference_payload(
            value, key=key_material, tenant_id=self.context.tenant_id,
            user_id=self.context.user_id, operation_id=operation_id,
            key_version=key_version,
        )

        with self._engine.begin() as connection:
            set_local_context(connection, self.context, scope_type=self.context.scope_type)
            personalization_enabled = connection.execute(text("""
                SELECT personalization_enabled
                  FROM governance.memory_preference_settings
                 WHERE tenant_id=:tenant_id AND user_id=:user_id
            """), {
                "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id,
            }).scalar_one_or_none()
            if personalization_enabled is False:
                raise PreferenceWriteRejected("personalization_disabled")
            if not connection.execute(text("""
                SELECT 1 FROM iam.tenant_memberships
                 WHERE tenant_id=:tenant_id AND user_id=:user_id AND revoked_at IS NULL
                 FOR KEY SHARE
            """), {
                "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id,
            }).first():
                raise PreferenceWriteRejected("membership_inactive")
            if not connection.execute(text("""
                SELECT 1 FROM governance.chat_sessions
                 WHERE tenant_id=:tenant_id AND user_id=:user_id
                   AND chat_session_id=:session_id
                 FOR KEY SHARE
            """), {
                "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id,
                "session_id": chat_session_id,
            }).first():
                raise PreferenceWriteRejected("chat_session_not_owned")

            connection.execute(text("""
                SELECT pg_advisory_xact_lock(hashtextextended(:lock_key, 0))
            """), {
                "lock_key": "|".join((
                    "idempotency", self.context.tenant_id,
                    self.context.user_id, idempotency_key,
                )),
            })

            prior = connection.execute(text("""
                SELECT o.event_id, o.operation_id, o.registry_id, o.overlay_id,
                       o.registry_version, r.memory_key, r.memory_type, r.version,
                       r.status, r.value_hash, r.chat_session_id, r.source_turn_id
                  FROM governance.memory_outbox o
                  JOIN governance.memory_registry r ON r.registry_id=o.registry_id
                 WHERE o.tenant_id=:tenant_id AND o.user_id=:user_id
                   AND o.seller_id IS NOT DISTINCT FROM :seller_id
                   AND o.idempotency_key=:idempotency_key
            """), {
                "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id,
                "seller_id": seller_scope,
                "idempotency_key": idempotency_key,
            }).mappings().first()
            if prior:
                if (
                    prior["value_hash"] != value_hash
                    or prior["memory_key"] != memory_key
                    or prior["memory_type"] != memory_type
                    or prior["chat_session_id"] != chat_session_id
                    or int(prior["registry_version"]) != expected_version + 1
                    or prior["source_turn_id"] != source_turn_id
                ):
                    raise PreferenceWriteConflict("idempotency_key_reused_with_different_request")
                return QueuedPreferenceOperation(
                    registry_id=str(prior["registry_id"]),
                    operation_id=str(prior["operation_id"]),
                    outbox_event_id=str(prior["event_id"]),
                    overlay_id=str(prior["overlay_id"]) if prior["overlay_id"] else None,
                    memory_key=str(prior["memory_key"]), memory_type=str(prior["memory_type"]),
                    version=int(prior["version"]), status=str(prior["status"]), changed=False,
                )

            connection.execute(text("""
                SELECT pg_advisory_xact_lock(hashtextextended(:lock_key, 0))
            """), {
                "lock_key": "|".join((
                    self.context.tenant_id, self.context.user_id,
                    seller_scope or "", memory_key,
                )),
            })
            latest = connection.execute(text("""
                SELECT registry_id, memory_type, status, version, mem0_memory_id,
                       value_hash, operation_id, expires_at,
                       (expires_at IS NOT NULL AND expires_at <= clock_timestamp()) AS is_expired
                  FROM governance.memory_registry
                 WHERE tenant_id=:tenant_id AND user_id=:user_id
                   AND seller_id IS NOT DISTINCT FROM :seller_id
                   AND memory_key=:memory_key
                 ORDER BY version DESC
                 LIMIT 1 FOR UPDATE
            """), {
                "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id,
                "seller_id": seller_scope,
                "memory_key": memory_key,
            }).mappings().first()
            current_version = int(latest["version"]) if latest else 0
            if current_version != expected_version:
                raise PreferenceWriteConflict("registry_version_conflict")
            if latest and latest["status"] not in {"active", "superseded", "deleted"}:
                raise PreferenceWriteConflict("registry_operation_in_progress")
            if latest and latest["memory_type"] != memory_type:
                raise PreferenceWriteConflict("memory_key_type_conflict")
            if (
                latest and latest["status"] == "superseded"
                and latest["memory_type"] == "scope_default" and latest["is_expired"]
                and (not latest["mem0_memory_id"] or not latest["operation_id"])
            ):
                raise PreferenceWriteConflict("expired_scope_memory_reference_missing")
            if (
                latest and latest["status"] == "active" and not latest["is_expired"]
                and latest.get("value_hash") == value_hash
            ):
                return QueuedPreferenceOperation(
                    registry_id=str(latest["registry_id"]),
                    operation_id=str(latest["operation_id"]), outbox_event_id="", overlay_id=None,
                    memory_key=memory_key, memory_type=memory_type,
                    version=current_version, status="active", changed=False,
                )

            previous_id = str(latest["registry_id"]) if latest and latest["status"] == "active" else None
            expired_scope_id = (
                str(latest["registry_id"])
                if latest and latest["status"] == "superseded"
                and latest["memory_type"] == "scope_default" and latest["is_expired"]
                and latest["mem0_memory_id"] and latest["operation_id"]
                else None
            )
            update_target_id = previous_id or expired_scope_id
            target = latest if update_target_id else None
            next_version = current_version + 1
            status = "pending_update" if update_target_id else "pending_create"
            event_type = "update" if update_target_id else "add"
            if previous_id:
                connection.execute(text("""
                    UPDATE governance.memory_registry
                       SET status='superseded', updated_at=now()
                     WHERE registry_id=:registry_id AND status='active'
                """), {"registry_id": previous_id})
                connection.execute(text("""
                    UPDATE governance.memory_preference_overlays
                       SET status='superseded', value='{}'::jsonb, cleared_at=now()
                     WHERE tenant_id=:tenant_id AND user_id=:user_id
                       AND seller_id IS NOT DISTINCT FROM :seller_id
                       AND memory_key=:memory_key AND status IN ('pending','failed')
                """), {
                    "tenant_id": self.context.tenant_id,
                    "user_id": self.context.user_id,
                    "seller_id": seller_scope,
                    "memory_key": memory_key,
                })

            connection.execute(text("""
                INSERT INTO governance.memory_registry
                    (registry_id, tenant_id, user_id, seller_id, memory_type,
                     memory_key, mem0_memory_id, status, version, operation_id,
                     source_turn_id, source_trace_id, user_confirmed,
                     chat_session_id, value_hash, supersedes_registry_id,
                     attempts, expires_at, created_at, updated_at)
                VALUES
                    (:registry_id, :tenant_id, :user_id, :seller_id, :memory_type,
                     :memory_key, NULL, :status, :version, :operation_id,
                     :source_turn_id, :source_trace_id, true,
                     :chat_session_id, :value_hash, :supersedes_registry_id,
                     0, CASE WHEN :scope_default THEN now()+interval '90 days' ELSE NULL END,
                     now(), now())
            """), {
                "registry_id": registry_id, "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id, "seller_id": seller_scope,
                "memory_type": memory_type, "memory_key": memory_key,
                "status": status, "version": next_version,
                "operation_id": operation_id, "source_turn_id": source_turn_id,
                "source_trace_id": source_trace_id,
                "chat_session_id": chat_session_id, "value_hash": value_hash,
                "supersedes_registry_id": update_target_id,
                "scope_default": memory_type == "scope_default",
            })
            connection.execute(text("""
                INSERT INTO governance.memory_preference_overlays
                    (overlay_id, tenant_id, user_id, seller_id, chat_session_id,
                     registry_id, memory_key, memory_type, value, source_turn_id,
                     operation_id, version, status, created_at, expires_at)
                VALUES
                    (:overlay_id, :tenant_id, :user_id, :seller_id, :session_id,
                     :registry_id, :memory_key, :memory_type, CAST(:value AS jsonb),
                     :source_turn_id, :operation_id, :version, 'pending', now(),
                     now() + interval '24 hours')
            """), {
                "overlay_id": overlay_id, "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id, "seller_id": seller_scope,
                "session_id": chat_session_id, "registry_id": registry_id,
                "memory_key": memory_key, "memory_type": memory_type,
                "value": canonical_value, "source_turn_id": source_turn_id,
                "operation_id": operation_id, "version": next_version,
            })
            connection.execute(text("""
                INSERT INTO governance.memory_outbox
                    (event_id, operation_id, tenant_id, user_id, seller_id,
                     event_type, registry_id, overlay_id, target_memory_id,
                     target_operation_id, payload_ciphertext, payload_nonce,
                     payload_key_version, idempotency_key, registry_version,
                     status, attempts, max_attempts,
                     next_attempt_at, created_at, updated_at)
                VALUES
                    (:event_id, :operation_id, :tenant_id, :user_id, :seller_id,
                     :event_type, :registry_id, :overlay_id, :target_memory_id,
                     :target_operation_id, :payload_ciphertext, :payload_nonce,
                     :payload_key_version, :idempotency_key, :version,
                     'pending', 0, 5, now(), now(), now())
            """), {
                "event_id": event_id, "operation_id": operation_id,
                "tenant_id": self.context.tenant_id, "user_id": self.context.user_id,
                "seller_id": seller_scope, "event_type": event_type,
                "registry_id": registry_id, "overlay_id": overlay_id,
                "target_memory_id": target["mem0_memory_id"] if target else None,
                "target_operation_id": target["operation_id"] if target else None,
                "payload_ciphertext": encrypted.ciphertext,
                "payload_nonce": encrypted.nonce,
                "payload_key_version": encrypted.key_version,
                "idempotency_key": idempotency_key, "version": next_version,
            })
            self._audit(
                connection, audit_id=audit_id, registry_id=registry_id,
                operation_id=operation_id, event_type=event_type,
                status=status, reason_code="user_confirmed_policy_accepted",
                seller_id=seller_scope,
            )
        return QueuedPreferenceOperation(
            registry_id=registry_id, operation_id=operation_id,
            outbox_event_id=event_id, overlay_id=overlay_id,
            memory_key=memory_key, memory_type=memory_type,
            version=next_version, status=status,
        )

    def queue_delete(
        self,
        *,
        registry_id: str,
        expected_version: int,
        idempotency_key: str,
        policy_decision: MemoryPolicyDecision,
    ) -> QueuedPreferenceOperation:
        if (
            not isinstance(policy_decision, MemoryPolicyDecision)
            or policy_decision.action != "delete"
            or policy_decision.subject != registry_id
            or policy_decision.tenant_id != self.context.tenant_id
            or policy_decision.user_id != self.context.user_id
            or not policy_decision.accepted or not policy_decision.user_confirmed
        ):
            raise PreferenceWriteRejected(
                getattr(policy_decision, "reason", "explicit_user_confirmation_required")
            )
        self._validate_idempotency_key(idempotency_key)
        operation_id = uuid.uuid4().hex
        event_id = f"mout_{uuid.uuid4().hex}"
        audit_id = f"maud_{uuid.uuid4().hex}"
        with self._engine.begin() as connection:
            set_local_context(connection, self.context, scope_type=self.context.scope_type)
            owned = connection.execute(text("""
                SELECT seller_id FROM governance.memory_registry
                 WHERE registry_id=:registry_id AND tenant_id=:tenant_id AND user_id=:user_id
            """), {
                "registry_id": registry_id, "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id,
            }).mappings().first()
            if owned is None:
                raise PreferenceWriteRejected("preference_not_found")
            seller_scope = owned["seller_id"]
            connection.execute(text("""
                SELECT pg_advisory_xact_lock(hashtextextended(:lock_key, 0))
            """), {
                "lock_key": "|".join((
                    "idempotency", self.context.tenant_id,
                    self.context.user_id, idempotency_key,
                )),
            })
            prior = connection.execute(text("""
                SELECT o.event_id, o.operation_id, o.registry_id, o.target_memory_id,
                       o.registry_version,
                       r.memory_key, r.memory_type, r.version, r.status
                  FROM governance.memory_outbox o
                  JOIN governance.memory_registry r ON r.registry_id=o.registry_id
                 WHERE o.tenant_id=:tenant_id AND o.user_id=:user_id
                   AND o.seller_id IS NOT DISTINCT FROM :seller_id
                   AND o.idempotency_key=:idempotency_key
            """), {
                "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id,
                "seller_id": seller_scope,
                "idempotency_key": idempotency_key,
            }).mappings().first()
            if prior:
                if (
                    prior["registry_id"] != registry_id
                    or int(prior["registry_version"]) != expected_version
                ):
                    raise PreferenceWriteConflict("idempotency_key_reused_with_different_request")
                return QueuedPreferenceOperation(
                    registry_id=str(prior["registry_id"]),
                    operation_id=str(prior["operation_id"]),
                    outbox_event_id=str(prior["event_id"]), overlay_id=None,
                    memory_key=str(prior["memory_key"]), memory_type=str(prior["memory_type"]),
                    version=int(prior["version"]), status=str(prior["status"]), changed=False,
                )
            row = connection.execute(text("""
                SELECT registry_id, seller_id, memory_type, memory_key,
                       mem0_memory_id, operation_id, version, status, expires_at,
                       supersedes_registry_id,
                       (expires_at IS NOT NULL AND expires_at <= clock_timestamp()) AS is_expired
                  FROM governance.memory_registry
                 WHERE registry_id=:registry_id AND tenant_id=:tenant_id
                   AND user_id=:user_id AND seller_id IS NOT DISTINCT FROM :seller_id
                 FOR UPDATE
            """), {
                "registry_id": registry_id,
                "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id,
                "seller_id": seller_scope,
            }).mappings().first()
            if row is None:
                raise PreferenceWriteRejected("preference_not_found")
            expired_scope_default = (
                row["status"] == "superseded" and row["memory_type"] == "scope_default"
                and row["is_expired"] and row["operation_id"]
            )
            failed_operation = None
            if row["status"] == "failed":
                failed_operation = connection.execute(text("""
                    SELECT event_id, event_type, target_memory_id, target_operation_id
                      FROM governance.memory_outbox
                     WHERE registry_id=:registry_id AND status='dead_letter'
                     ORDER BY created_at DESC LIMIT 1 FOR UPDATE
                """), {"registry_id": registry_id}).mappings().first()
            if (
                row["status"] not in {"active", "pending_create"}
                and not expired_scope_default and failed_operation is None
            ):
                raise PreferenceWriteConflict("preference_not_active")
            if int(row["version"]) != expected_version:
                raise PreferenceWriteConflict("registry_version_conflict")
            if not row["operation_id"]:
                raise PreferenceWriteRejected("preference_mem0_reference_missing")
            target_memory_id = row["mem0_memory_id"]
            target_operation_id = row["operation_id"]
            if failed_operation is not None:
                failed_operation = dict(failed_operation)
                previous_memory_id = None
                if failed_operation["event_type"] == "update" and row["supersedes_registry_id"]:
                    previous = connection.execute(text("""
                        SELECT mem0_memory_id FROM governance.memory_registry
                         WHERE registry_id=:registry_id AND tenant_id=:tenant_id
                           AND user_id=:user_id
                    """), {
                        "registry_id": row["supersedes_registry_id"],
                        "tenant_id": self.context.tenant_id,
                        "user_id": self.context.user_id,
                    }).mappings().first()
                    if previous is not None:
                        previous_memory_id = previous["mem0_memory_id"]
                target_memory_id, target_operation_id = self._failed_delete_targets(
                    dict(row), failed_operation, previous_memory_id=previous_memory_id,
                )
                connection.execute(text("""
                    UPDATE governance.memory_outbox
                       SET payload_ciphertext=NULL, payload_nonce=NULL,
                           payload_key_version=NULL, updated_at=now()
                     WHERE event_id=:event_id AND status='dead_letter'
                """), {"event_id": failed_operation["event_id"]})
            if row["status"] == "active" and not target_memory_id:
                raise PreferenceWriteRejected("preference_mem0_reference_missing")
            if row["status"] == "pending_create":
                pending = connection.execute(text("""
                    SELECT event_id, attempts FROM governance.memory_outbox
                     WHERE registry_id=:registry_id AND operation_id=:operation_id
                       AND event_type='add' AND status='pending'
                     FOR UPDATE
                """), {
                    "registry_id": registry_id, "operation_id": row["operation_id"],
                }).mappings().first()
                if pending is None or int(pending["attempts"]) != 0:
                    raise PreferenceWriteConflict("preference_operation_in_progress")
                connection.execute(text("""
                    UPDATE governance.memory_outbox
                       SET status='succeeded', payload_ciphertext=NULL, payload_nonce=NULL,
                           payload_key_version=NULL, lease_token=NULL, lease_expires_at=NULL,
                           completed_at=now(), updated_at=now(),
                           last_error_code='cancelled_before_external_delivery',
                           expires_at=now()+interval '30 days'
                     WHERE event_id=:event_id AND status='pending'
                """), {"event_id": pending["event_id"]})
                connection.execute(text("""
                    UPDATE governance.memory_preference_overlays
                       SET status='superseded', value='{}'::jsonb, cleared_at=now()
                     WHERE registry_id=:registry_id AND operation_id=:operation_id
                       AND status='pending'
                """), {
                    "registry_id": registry_id, "operation_id": row["operation_id"],
                })
                self._audit(
                    connection, audit_id=f"maud_{uuid.uuid4().hex}",
                    registry_id=registry_id, operation_id=str(row["operation_id"]),
                    event_type="cancelled_before_delivery", status="cancelled",
                    reason_code="user_deleted_pending_create",
                    seller_id=seller_scope,
                )
            connection.execute(text("""
                UPDATE governance.memory_registry
                   SET status='pending_delete', operation_id=:operation_id,
                       user_confirmed=true, updated_at=now()
                 WHERE registry_id=:registry_id AND status=:previous_status
                   AND version=:version
            """), {
                "operation_id": operation_id, "registry_id": registry_id,
                "version": expected_version, "previous_status": row["status"],
            })
            connection.execute(text("""
                UPDATE governance.memory_preference_overlays
                   SET status='superseded', value='{}'::jsonb, error_code=NULL,
                       cleared_at=now()
                 WHERE registry_id=:registry_id AND status IN ('pending','failed')
            """), {"registry_id": registry_id})
            connection.execute(text("""
                INSERT INTO governance.memory_outbox
                    (event_id, operation_id, tenant_id, user_id, seller_id,
                    event_type, registry_id, target_memory_id, target_operation_id,
                    idempotency_key, registry_version,
                     status, attempts, max_attempts, next_attempt_at, created_at, updated_at)
                VALUES (:event_id, :operation_id, :tenant_id, :user_id, :seller_id,
                        'delete', :registry_id, :memory_id, :target_operation_id,
                        :idempotency_key, :version,
                        'pending', 0, 5, now(), now(), now())
            """), {
                "event_id": event_id, "operation_id": operation_id,
                "tenant_id": self.context.tenant_id, "user_id": self.context.user_id,
                "seller_id": row["seller_id"], "registry_id": registry_id,
                "memory_id": target_memory_id, "idempotency_key": idempotency_key,
                "target_operation_id": target_operation_id,
                "version": expected_version,
            })
            self._audit(
                connection, audit_id=audit_id, registry_id=registry_id,
                operation_id=operation_id, event_type="delete",
                status="pending_delete", reason_code="user_confirmed_delete",
                seller_id=seller_scope,
            )
        return QueuedPreferenceOperation(
            registry_id=registry_id, operation_id=operation_id,
            outbox_event_id=event_id, overlay_id=None,
            memory_key=str(row["memory_key"]), memory_type=str(row["memory_type"]),
            version=int(row["version"]), status="pending_delete",
        )

    def queue_reconciliation(
        self, *, registry_id: str, expected_version: int, idempotency_key: str,
    ) -> QueuedPreferenceOperation:
        """Recheck a durable active preference against Mem0 using its stable operation ID."""
        self._validate_idempotency_key(idempotency_key)
        with self._engine.begin() as connection:
            set_local_context(connection, self.context, scope_type=self.context.scope_type)
            owned = connection.execute(text("""
                SELECT seller_id FROM governance.memory_registry
                 WHERE registry_id=:registry_id AND tenant_id=:tenant_id AND user_id=:user_id
            """), {
                "registry_id": registry_id, "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id,
            }).mappings().first()
            if owned is None:
                raise PreferenceWriteRejected("preference_not_found")
            seller_scope = owned["seller_id"]
            connection.execute(text("""
                SELECT pg_advisory_xact_lock(hashtextextended(:lock_key, 0))
            """), {"lock_key": "|".join((
                "idempotency", self.context.tenant_id, self.context.user_id, idempotency_key,
            ))})
            prior = connection.execute(text("""
                SELECT o.event_id, o.operation_id, o.registry_id, o.registry_version,
                       o.event_type, o.status, r.memory_key, r.memory_type
                  FROM governance.memory_outbox o
                  JOIN governance.memory_registry r ON r.registry_id=o.registry_id
                 WHERE o.tenant_id=:tenant_id AND o.user_id=:user_id
                   AND o.seller_id IS NOT DISTINCT FROM :seller_id
                   AND o.idempotency_key=:idempotency_key
            """), {
                "tenant_id": self.context.tenant_id, "user_id": self.context.user_id,
                "seller_id": seller_scope, "idempotency_key": idempotency_key,
            }).mappings().first()
            if prior:
                if prior["registry_id"] != registry_id or prior["event_type"] != "reconcile":
                    raise PreferenceWriteConflict("idempotency_key_reused_with_different_request")
                return QueuedPreferenceOperation(
                    registry_id=str(prior["registry_id"]),
                    operation_id=str(prior["operation_id"]),
                    outbox_event_id=str(prior["event_id"]), overlay_id=None,
                    memory_key=str(prior["memory_key"]), memory_type=str(prior["memory_type"]),
                    version=int(prior["registry_version"]), status=str(prior["status"]),
                    changed=False,
                )
            row = connection.execute(text("""
                SELECT registry_id, memory_key, memory_type, mem0_memory_id,
                       operation_id, version, status
                  FROM governance.memory_registry
                 WHERE registry_id=:registry_id AND tenant_id=:tenant_id
                   AND user_id=:user_id AND seller_id IS NOT DISTINCT FROM :seller_id
                 FOR UPDATE
            """), {
                "registry_id": registry_id, "tenant_id": self.context.tenant_id,
                "user_id": self.context.user_id, "seller_id": seller_scope,
            }).mappings().first()
            if row is None:
                raise PreferenceWriteRejected("preference_not_found")
            if row["status"] != "active" or not row["mem0_memory_id"] or not row["operation_id"]:
                raise PreferenceWriteConflict("preference_not_reconcilable")
            if int(row["version"]) != expected_version:
                raise PreferenceWriteConflict("registry_version_conflict")
            event_id = f"mout_{uuid.uuid4().hex}"
            changed = connection.execute(text("""
                UPDATE governance.memory_registry
                   SET status='pending_update', last_error=NULL, updated_at=now()
                 WHERE registry_id=:registry_id AND status='active' AND version=:version
            """), {"registry_id": registry_id, "version": expected_version}).rowcount
            if changed != 1:
                raise PreferenceWriteConflict("registry_version_conflict")
            prior_event = connection.execute(text("""
                SELECT event_id FROM governance.memory_outbox
                 WHERE tenant_id=:tenant_id AND user_id=:user_id
                   AND operation_id=:operation_id AND status='succeeded'
                 ORDER BY completed_at DESC LIMIT 1 FOR UPDATE
            """), {
                "tenant_id": self.context.tenant_id, "user_id": self.context.user_id,
                "operation_id": row["operation_id"],
            }).scalar_one_or_none()
            if prior_event:
                event_id = str(prior_event)
                connection.execute(text("""
                    UPDATE governance.memory_outbox
                       SET event_type='reconcile', overlay_id=NULL,
                           idempotency_key=:idempotency_key,
                           target_memory_id=:memory_id, target_operation_id=:operation_id,
                           payload_ciphertext=NULL, payload_nonce=NULL, payload_key_version=NULL,
                           status='pending', attempts=0, next_attempt_at=now(),
                           lease_token=NULL, lease_expires_at=NULL, completed_at=NULL,
                           last_error_code=NULL, expires_at=now()+interval '30 days', updated_at=now()
                     WHERE event_id=:event_id
                """), {
                    "idempotency_key": idempotency_key, "memory_id": row["mem0_memory_id"],
                    "operation_id": row["operation_id"], "event_id": event_id,
                })
            else:
                connection.execute(text("""
                    INSERT INTO governance.memory_outbox
                        (event_id, operation_id, tenant_id, user_id, seller_id,
                         event_type, registry_id, target_memory_id, target_operation_id,
                         idempotency_key, registry_version, status, attempts,
                         max_attempts, next_attempt_at, created_at, updated_at)
                    VALUES (:event_id, :operation_id, :tenant_id, :user_id, :seller_id,
                            'reconcile', :registry_id, :memory_id, :operation_id,
                            :idempotency_key, :version, 'pending', 0, 5, now(), now(), now())
                """), {
                    "event_id": event_id, "operation_id": row["operation_id"],
                    "tenant_id": self.context.tenant_id, "user_id": self.context.user_id,
                    "seller_id": seller_scope, "registry_id": registry_id,
                    "memory_id": row["mem0_memory_id"], "idempotency_key": idempotency_key,
                    "version": expected_version,
                })
            self._audit(
                connection, audit_id=f"maud_{uuid.uuid4().hex}", registry_id=registry_id,
                operation_id=str(row["operation_id"]), event_type="reconcile",
                status="pending_update", reason_code="user_requested_reconciliation",
                seller_id=seller_scope,
            )
        return QueuedPreferenceOperation(
            registry_id=registry_id, operation_id=str(row["operation_id"]),
            outbox_event_id=event_id, overlay_id=None,
            memory_key=str(row["memory_key"]), memory_type=str(row["memory_type"]),
            version=int(row["version"]), status="pending_update",
        )

    def _audit(
        self, connection: Any, *, audit_id: str, registry_id: str,
        operation_id: str, event_type: str, status: str, reason_code: str,
        seller_id: str | None | object = _CONTEXT_SELLER,
    ) -> None:
        connection.execute(text("""
            INSERT INTO governance.memory_preference_audit
                (audit_id, tenant_id, user_id, seller_id, registry_id,
                 operation_id, event_type, actor_user_id, status, reason_code)
            VALUES (:audit_id, :tenant_id, :user_id, :seller_id, :registry_id,
                    :operation_id, :event_type, :actor, :status, :reason_code)
        """), {
            "audit_id": audit_id, "tenant_id": self.context.tenant_id,
            "user_id": self.context.user_id,
            "seller_id": self.context.seller_id if seller_id is _CONTEXT_SELLER else seller_id,
            "registry_id": registry_id, "operation_id": operation_id,
            "event_type": event_type, "actor": self.context.user_id,
            "status": status, "reason_code": reason_code,
        })

    @staticmethod
    def _validate_mutation(
        *, memory_key: str, memory_type: str, value: dict[str, Any],
        chat_session_id: str, context: TenantContext,
        policy_decision: MemoryPolicyDecision,
    ) -> None:
        if not chat_session_id:
            raise PreferenceWriteRejected("chat_session_required_for_overlay")
        try:
            canonical = json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise PreferenceWriteRejected("preference_value_must_be_json") from exc
        expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if (
            not isinstance(policy_decision, MemoryPolicyDecision)
            or policy_decision.action != "upsert"
            or policy_decision.tenant_id != context.tenant_id
            or policy_decision.user_id != context.user_id
            or policy_decision.subject != memory_key
            or policy_decision.memory_type != memory_type
            or policy_decision.value_hash != expected_hash
            or not policy_decision.accepted or not policy_decision.user_confirmed
        ):
            raise PreferenceWriteRejected(
                getattr(policy_decision, "reason", "memory_policy_did_not_accept")
            )

    @staticmethod
    def _validate_idempotency_key(value: str) -> None:
        if not value or len(value) > 256:
            raise PreferenceWriteRejected("idempotency_key_required")
