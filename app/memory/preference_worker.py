"""At-least-once WP4 worker for the durable preference Outbox."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from sqlalchemy import create_engine, text

from app.core.config import get_settings
from app.data.rls import set_local_context
from app.memory.mem0_adapter import Mem0PreferenceAdapter, Mem0Error
from app.memory.policy import MemoryPolicyDecision
from app.memory.preference_payload import (
    PreferencePayloadError,
    decrypt_preference_payload,
    payload_key_material,
)
from app.memory.preference_registry import PostgresPreferenceRegistry
from app.memory.preference_writer import PreferenceWriter
from app.observability import record_event
from app.observability.tracing import prometheus_payload
from app.tenancy import PLATFORM_SCOPE, TenantContext


logger = logging.getLogger(__name__)
LEASE_SECONDS = 60


class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/metrics":
            self.send_error(404)
            return
        payload = prometheus_payload()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: Any) -> None:
        # Metrics scrapes are frequent and contain no useful application log data.
        return


def start_metrics_server(port: int, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    """Expose this worker process's counters for Prometheus scraping."""
    server = ThreadingHTTPServer((host, port), _MetricsHandler)
    thread = threading.Thread(
        target=server.serve_forever, name="preference-worker-metrics", daemon=True,
    )
    thread.start()
    return server


def _worker_engine() -> Any:
    settings = get_settings()
    url = str(getattr(settings, "memory_preference_database_url", "") or "").strip()
    if not url:
        raise RuntimeError("memory_preference_database_url_required")
    return create_engine(
        url, future=True, pool_pre_ping=True,
        connect_args={"connect_timeout": int(settings.database_timeout_seconds)},
    )


def _context(row: dict[str, Any]) -> TenantContext:
    seller_id = str(row["seller_id"]) if row.get("seller_id") else None
    return TenantContext(
        tenant_id=str(row["tenant_id"]), user_id=str(row["user_id"]),
        role="analyst" if seller_id else "admin",
        scope_type="seller" if seller_id else PLATFORM_SCOPE,
        seller_id=seller_id,
    )


def _record_audit(
    connection: Any, row: dict[str, Any], *, event_type: str,
    status: str, reason_code: str | None = None,
) -> None:
    connection.execute(text("""
        INSERT INTO governance.memory_preference_audit
            (audit_id, tenant_id, user_id, seller_id, registry_id,
             operation_id, event_type, actor_user_id, status, reason_code)
        VALUES (:audit_id, :tenant_id, :user_id, :seller_id, :registry_id,
                :operation_id, :event_type, :actor, :status, :reason)
    """), {
        "audit_id": f"maud_{uuid.uuid4().hex}",
        "tenant_id": row["tenant_id"], "user_id": row["user_id"],
        "seller_id": row["seller_id"], "registry_id": row["registry_id"],
        "operation_id": row["operation_id"], "event_type": event_type,
        "actor": row["user_id"], "status": status, "reason": reason_code,
    })


def _recover_expired_leases(connection: Any) -> None:
    rows = connection.execute(text("""
        SELECT event_id, operation_id, tenant_id, user_id, seller_id,
               registry_id, overlay_id, attempts, max_attempts
          FROM governance.memory_outbox
         WHERE status='leased' AND lease_expires_at <= clock_timestamp()
         ORDER BY lease_expires_at
         FOR UPDATE SKIP LOCKED
    """)).mappings().all()
    for item in rows:
        row = dict(item)
        exhausted = int(row["attempts"]) >= int(row["max_attempts"])
        connection.execute(text("""
            UPDATE governance.memory_outbox
               SET status=:status, lease_token=NULL, lease_expires_at=NULL,
                   next_attempt_at=now(), updated_at=now(),
                   completed_at=CASE WHEN :exhausted THEN now() ELSE NULL END,
                   expires_at=CASE WHEN :exhausted THEN now()+interval '90 days' ELSE expires_at END,
                   last_error_code=CASE WHEN :exhausted THEN 'lease_expired_retry_exhausted'
                                        ELSE 'lease_expired_recovered' END
             WHERE event_id=:event_id AND status='leased'
        """), {
            "status": "dead_letter" if exhausted else "pending",
            "exhausted": exhausted, "event_id": row["event_id"],
        })
        if exhausted:
            connection.execute(text("""
                UPDATE governance.memory_registry
                   SET status='failed', last_error='lease_expired_retry_exhausted',
                       updated_at=now()
                 WHERE registry_id=:registry_id AND tenant_id=:tenant_id
                   AND user_id=:user_id AND operation_id=:operation_id
                   AND status IN ('pending_create','pending_update','pending_delete')
            """), row)
            if row["overlay_id"]:
                overlay_failed = connection.execute(text("""
                    UPDATE governance.memory_preference_overlays
                       SET status='failed', error_code='lease_expired_retry_exhausted'
                     WHERE overlay_id=:overlay_id AND tenant_id=:tenant_id
                       AND user_id=:user_id AND operation_id=:operation_id
                       AND status='pending'
                """), row).rowcount
                if overlay_failed:
                    record_event("memory.preference_overlay_failed", {})
            _record_audit(
                connection, row, event_type="worker_dead_letter",
                status="failed", reason_code="lease_expired_retry_exhausted",
            )


def cleanup_expired_state(engine: Any) -> None:
    """Clear expired plaintext overlays and expire due scope defaults."""
    with engine.begin() as connection:
        expired_overlays = connection.execute(text("""
            UPDATE governance.memory_preference_overlays
               SET status='expired', value='{}'::jsonb, cleared_at=now(),
                   error_code=COALESCE(error_code, 'overlay_ttl_expired')
             WHERE status IN ('pending','failed') AND expires_at <= clock_timestamp()
            RETURNING overlay_id, tenant_id, user_id, seller_id, registry_id,
                      operation_id,
                      EXTRACT(EPOCH FROM (clock_timestamp()-created_at))*1000 AS age_ms
        """)).mappings().all()
        if expired_overlays:
            for overlay in expired_overlays:
                _record_audit(
                    connection, dict(overlay), event_type="overlay_expired",
                    status="expired", reason_code="overlay_ttl_expired",
                )
                record_event("memory.preference_overlay_expired_item", {
                    "duration_ms": float(overlay["age_ms"] or 0),
                })
            record_event("memory.preference_overlay_expired", {
                "memory.count": len(expired_overlays),
                "memory.max_age_ms": max(float(row["age_ms"] or 0) for row in expired_overlays),
                "duration_ms": max(float(row["age_ms"] or 0) for row in expired_overlays),
            })
        expired_scope_defaults = connection.execute(text("""
            UPDATE governance.memory_registry
               SET status='superseded', updated_at=now()
             WHERE memory_type='scope_default' AND status='active'
               AND expires_at IS NOT NULL AND expires_at <= clock_timestamp()
             RETURNING registry_id, tenant_id, user_id, seller_id, operation_id
        """)).mappings().all()
        for expired in expired_scope_defaults:
            _record_audit(
                connection, dict(expired), event_type="scope_default_expired",
                status="superseded", reason_code="scope_default_waiting_for_confirmation",
            )
        if expired_scope_defaults:
            for _ in expired_scope_defaults:
                record_event("memory.preference_scope_default_expired_item", {})
            record_event("memory.preference_scope_default_expired", {
                "memory.count": len(expired_scope_defaults),
            })
        expired_outbox = connection.execute(text("""
            DELETE FROM governance.memory_outbox
             WHERE status IN ('succeeded','dead_letter')
               AND expires_at <= clock_timestamp()
        """)).rowcount
        expired_overlays_retained = connection.execute(text("""
            DELETE FROM governance.memory_preference_overlays o
             USING governance.memory_registry r
             WHERE o.registry_id=r.registry_id
               AND r.status IN ('deleted','failed')
               AND r.updated_at <= now()-interval '180 days'
               AND o.status IN ('cleared','expired','superseded')
        """)).rowcount
        expired_registry = connection.execute(text("""
            DELETE FROM governance.memory_registry r
             WHERE r.status IN ('deleted','failed')
               AND r.updated_at <= now()-interval '180 days'
               AND NOT EXISTS (
                   SELECT 1 FROM governance.memory_outbox o WHERE o.registry_id=r.registry_id
               )
               AND NOT EXISTS (
                   SELECT 1 FROM governance.memory_preference_overlays p
                    WHERE p.registry_id=r.registry_id
               )
        """)).rowcount
    if expired_outbox:
        record_event("memory.preference_outbox_expired", {"memory.count": expired_outbox})
        for _ in range(expired_outbox):
            record_event("memory.preference_outbox_expired_item", {})
    if expired_overlays_retained:
        record_event("memory.preference_overlay_retention_purged", {
            "memory.count": expired_overlays_retained,
        })
    if expired_registry:
        record_event("memory.preference_registry_retention_purged", {
            "memory.count": expired_registry,
        })


def _claim(engine: Any, *, lease_seconds: int = LEASE_SECONDS) -> dict[str, Any] | None:
    cleanup_expired_state(engine)
    with engine.begin() as connection:
        _recover_expired_leases(connection)
        item = connection.execute(text("""
            SELECT event_id, operation_id, tenant_id, user_id, seller_id,
                   event_type, registry_id, overlay_id, target_memory_id,
                   target_operation_id, registry_version, attempts, max_attempts
              FROM governance.memory_outbox
             WHERE status='pending' AND next_attempt_at <= clock_timestamp()
               AND (event_type='delete' OR COALESCE((
                    SELECT s.personalization_enabled
                      FROM governance.memory_preference_settings s
                     WHERE s.tenant_id=memory_outbox.tenant_id
                       AND s.user_id=memory_outbox.user_id
               ), true))
             ORDER BY next_attempt_at, created_at
             FOR UPDATE SKIP LOCKED
             LIMIT 1
        """)).mappings().first()
        if item is None:
            return None
        row = dict(item)
        token = uuid.uuid4().hex
        claimed = connection.execute(text("""
            UPDATE governance.memory_outbox
               SET status='leased', attempts=attempts+1, lease_token=:token,
                   lease_expires_at=now()+(:lease_seconds || ' seconds')::interval,
                   updated_at=now()
             WHERE event_id=:event_id AND status='pending'
             RETURNING attempts
        """), {
            "token": token, "lease_seconds": int(lease_seconds),
            "event_id": row["event_id"],
        }).scalar_one_or_none()
        if claimed is None:
            return None
        row["attempts"] = int(claimed)
        row["lease_token"] = token
        return row


def _render_preference(
    memory_key: str, memory_type: str, value: dict[str, Any],
) -> str:
    """Create a human-readable Mem0 text from the typed preference value.

    The structured value remains in metadata (`value_json`); this text exists
    for semantic recall, so it must read like a preference statement rather
    than a ``key: JSON`` payload.  JSON-only text scores below unrelated
    natural-language preferences for colloquial queries.
    """
    style_labels = {"concise": "简洁", "balanced": "平衡", "detailed": "详细"}
    format_labels = {"table": "使用表格", "bullets": "使用项目符号", "narrative": "使用叙述文字"}
    language_labels = {"zh-CN": "中文", "zh-TW": "繁体中文", "en-US": "英文"}
    operation_labels = {
        "compare": "先做同比对比", "trend": "先看趋势",
        "breakdown": "按维度拆解", "statistics": "先看统计特征",
    }
    metric_labels = {
        "gmv": "成交总额", "order_count": "订单量", "customer_count": "成交客户数",
        "aov": "客单价", "repeat_purchase_rate": "复购率",
    }
    dimension_labels = {
        "product_category": "按商品品类", "customer_state": "按客户州",
        "seller_state": "按卖家州", "payment_type": "按支付方式",
    }
    if memory_type == "presentation":
        if memory_key == "response_style" and value.get("style"):
            return f"回答风格偏好：{style_labels.get(str(value['style']), str(value['style']))}。"
        if memory_key == "report_style" and value.get("format"):
            return f"报告排版偏好：{format_labels.get(str(value['format']), str(value['format']))}。"
        if memory_key == "response_language" and value.get("language"):
            return f"输出语言偏好：{language_labels.get(str(value['language']), str(value['language']))}。"
        details = "；".join(f"{key}={item}" for key, item in sorted(value.items()))
        return f"回答呈现偏好：{details}。"
    if memory_type == "analysis_method":
        parts = []
        if value.get("operation"):
            parts.append(operation_labels.get(str(value["operation"]), f"先做{value['operation']}"))
        if value.get("metric"):
            metric = value["metric"]
            metrics = metric if isinstance(metric, list) else [metric]
            labels = [metric_labels.get(str(item), str(item)) for item in metrics]
            parts.append("同时关注" + "、".join(labels))
        if value.get("dimension"):
            parts.append(dimension_labels.get(str(value["dimension"]), f"按{value['dimension']}"))
        if value.get("date"):
            parts.append(f"默认时间范围 {value['date']}")
        if parts:
            return "分析习惯偏好：" + "；".join(parts) + "。"
        details = "；".join(f"{key}={item}" for key, item in sorted(value.items()))
        return f"分析习惯偏好：{details}。"
    if memory_type == "scope_default":
        if value.get("start_date") or value.get("end_date"):
            return (
                "默认统计周期："
                f"{value.get('start_date') or '未指定'} 至 {value.get('end_date') or '未指定'}。"
            )
        if value.get("date"):
            return f"默认分析日期：{value['date']}。"
        details = "；".join(f"{key}={item}" for key, item in sorted(value.items()))
        return f"默认分析范围偏好：{details}。"
    details = "；".join(f"{key}={item}" for key, item in sorted(value.items()))
    return f"{memory_key} 偏好：{details}。"


def _reconcile_callable(adapter: Any) -> Any:
    """Prefer the Registry-free pending reconcile without touching ``reconcile`` eagerly."""
    pending = getattr(adapter, "reconcile_pending", None)
    return pending or adapter.reconcile


def _unique_reconciled_hit(adapter: Any, operation_id: str) -> Any | None:
    reconcile = _reconcile_callable(adapter)
    hits = sorted(reconcile(operation_id), key=lambda item: item.memory_id)
    record_event("memory.preference_reconcile", {
        "memory.outcome": "hit" if hits else "miss", "memory.match_count": len(hits),
    })
    if not hits:
        return None
    duplicate_count = max(0, len(hits) - 1)
    for hit in hits[1:]:
        adapter.delete(hit.memory_id)
        record_event("memory.preference_reconcile_duplicate_removed", {})
    remaining = reconcile(operation_id)
    if len(remaining) != 1:
        raise RuntimeError("operation_reconcile_duplicate_cleanup_incomplete")
    if duplicate_count:
        record_event("memory.preference_reconcile_duplicates", {
            "memory.duplicate_count": duplicate_count,
        })
    return remaining[0]


def _deliver(row: dict[str, Any], engine: Any, adapter_factory: Callable[[TenantContext], Any]) -> None:
    context = _context(row)
    with engine.begin() as connection:
        set_local_context(connection, context, scope_type=context.scope_type)
        current = connection.execute(text("""
            SELECT o.event_id, o.operation_id, o.tenant_id, o.user_id, o.seller_id,
                   o.event_type, o.registry_id, o.overlay_id, o.target_memory_id,
                   o.target_operation_id, o.payload_ciphertext, o.payload_nonce,
                   o.payload_key_version, o.registry_version, o.status, o.lease_token,
                   r.memory_type, r.memory_key, r.status AS registry_status,
                   r.version, r.source_turn_id, r.source_trace_id
              FROM governance.memory_outbox o
              JOIN governance.memory_registry r ON r.registry_id=o.registry_id
             WHERE o.event_id=:event_id AND o.tenant_id=:tenant_id
               AND o.user_id=:user_id AND o.status='leased'
               AND o.lease_token=:lease_token
               AND r.tenant_id=o.tenant_id AND r.user_id=o.user_id
               AND r.operation_id=o.operation_id
        """), {
            "event_id": row["event_id"], "tenant_id": row["tenant_id"],
            "user_id": row["user_id"], "lease_token": row["lease_token"],
        }).mappings().first()
        if current is None:
            return
        operation = dict(current)
    adapter = adapter_factory(context)

    if operation["event_type"] == "delete":
        delete_started = time.perf_counter()
        target_id = operation["target_memory_id"]
        if target_id:
            # False means the fixed, scope-checked memory ID is already absent;
            # a retry after a successful external delete is therefore idempotent.
            adapter.delete(str(target_id))
        old_operation = operation["target_operation_id"]
        if old_operation:
            reconcile = _reconcile_callable(adapter)
            for duplicate in reconcile(str(old_operation)):
                adapter.delete(duplicate.memory_id)
            if reconcile(str(old_operation)):
                raise RuntimeError("preference_delete_reconcile_incomplete")
        result_memory_id = None
        record_event("memory.preference_delete", {
            "memory.status": "verified", "duration_ms": (time.perf_counter() - delete_started) * 1000,
        })
    elif operation["event_type"] == "reconcile":
        result = _unique_reconciled_hit(adapter, str(operation["operation_id"]))
        if result is None:
            record_event("memory.preference_reconcile", {
                "memory.outcome": "miss", "memory.operation": "reconcile",
            })
            raise RuntimeError("mem0_reconcile_result_missing")
        result_memory_id = str(result.memory_id)
        record_event("memory.preference_reconcile", {
            "memory.outcome": "hit", "memory.operation": "reconcile",
        })
    elif operation["event_type"] in {"add", "update"}:
        settings = get_settings()
        key_material, _ = payload_key_material(settings)
        overlay_value = decrypt_preference_payload(
            operation["payload_ciphertext"], operation["payload_nonce"],
            key=key_material,
            key_version=str(operation["payload_key_version"]),
            tenant_id=str(operation["tenant_id"]), user_id=str(operation["user_id"]),
            operation_id=str(operation["operation_id"]),
        )
        text_value = _render_preference(
            str(operation["memory_key"]), str(operation["memory_type"]), overlay_value,
        )
        result = _unique_reconciled_hit(adapter, str(operation["operation_id"]))
        if result is None:
            if operation["event_type"] == "add":
                adapter.add(
                    text_value, memory_key=str(operation["memory_key"]),
                    memory_type=str(operation["memory_type"]),
                    operation_id=str(operation["operation_id"]), value=overlay_value,
                    source_turn_id=operation["source_turn_id"],
                    source_trace_id=operation["source_trace_id"], infer=False,
                )
            elif operation["event_type"] == "update":
                target_id = operation["target_memory_id"]
                if not target_id or not adapter.update(
                    str(target_id), operation_id=str(operation["operation_id"]),
                    text=text_value, memory_status="active", value=overlay_value,
                    source_turn_id=operation["source_turn_id"],
                ):
                    raise RuntimeError("preference_update_target_missing")
            else:
                raise RuntimeError("unsupported_preference_event")
            result = _unique_reconciled_hit(adapter, str(operation["operation_id"]))
        if result is None:
            raise RuntimeError("mem0_operation_result_not_reconciled")
        old_operation = operation["target_operation_id"]
        if old_operation and old_operation != operation["operation_id"]:
            reconcile = _reconcile_callable(adapter)
            for duplicate in reconcile(str(old_operation)):
                if duplicate.memory_id != result.memory_id:
                    adapter.delete(duplicate.memory_id)
        result_memory_id = str(result.memory_id)
    else:
        raise RuntimeError("unsupported_preference_event")

    _complete(operation, row, engine, context, result_memory_id)


def _complete(
    operation: dict[str, Any], lease: dict[str, Any], engine: Any,
    context: TenantContext, memory_id: str | None,
) -> None:
    with engine.begin() as connection:
        set_local_context(connection, context, scope_type=context.scope_type)
        changed = connection.execute(text("""
            UPDATE governance.memory_outbox
               SET status='succeeded', lease_token=NULL, lease_expires_at=NULL,
                   payload_ciphertext=NULL, payload_nonce=NULL, payload_key_version=NULL,
                   updated_at=now(), completed_at=now(), last_error_code=NULL,
                   expires_at=now()+interval '30 days'
             WHERE event_id=:event_id AND tenant_id=:tenant_id AND user_id=:user_id
               AND status='leased' AND lease_token=:lease_token
        """), {
            "event_id": lease["event_id"], "tenant_id": lease["tenant_id"],
            "user_id": lease["user_id"], "lease_token": lease["lease_token"],
        }).rowcount
        if changed != 1:
            return
        if operation["event_type"] == "delete":
            registry_changed = connection.execute(text("""
                UPDATE governance.memory_registry
                   SET status='deleted', last_error=NULL, updated_at=now()
                 WHERE registry_id=:registry_id AND tenant_id=:tenant_id
                   AND user_id=:user_id AND operation_id=:operation_id
                   AND version=:version AND status='pending_delete'
            """), operation).rowcount
        else:
            registry_changed = connection.execute(text("""
                UPDATE governance.memory_registry
                   SET status='active', mem0_memory_id=:memory_id,
                       last_error=NULL, updated_at=now()
                 WHERE registry_id=:registry_id AND tenant_id=:tenant_id
                   AND user_id=:user_id AND operation_id=:operation_id
                   AND version=:version AND status IN ('pending_create','pending_update')
            """), {**operation, "memory_id": memory_id}).rowcount
            if operation["event_type"] in {"add", "update"} and operation.get("overlay_id"):
                overlay_changed = connection.execute(text("""
                    UPDATE governance.memory_preference_overlays
                       SET status='cleared', value='{}'::jsonb, cleared_at=now(),
                           error_code=NULL
                     WHERE overlay_id=:overlay_id AND tenant_id=:tenant_id
                       AND user_id=:user_id AND operation_id=:operation_id
                       AND status='pending'
                """), operation).rowcount
                if overlay_changed != 1:
                    raise RuntimeError("preference_overlay_completion_conflict")
        if registry_changed != 1:
            raise RuntimeError("preference_registry_completion_conflict")
        _record_audit(
            connection, operation, event_type="worker_succeeded",
            status="succeeded",
        )
    record_event("memory.preference_outbox", {
        "memory.operation": operation["event_type"],
        "memory.status": "succeeded",
        "memory.registry_id": operation["registry_id"],
    })


def _fail(lease: dict[str, Any], engine: Any, error_code: str) -> None:
    context = _context(lease)
    attempts = int(lease["attempts"])
    maximum = int(lease["max_attempts"])
    exhausted = attempts >= maximum
    delay = min(32, 2 ** max(0, attempts - 1)) + random.uniform(0, 1)
    with engine.begin() as connection:
        set_local_context(connection, context, scope_type=context.scope_type)
        changed = connection.execute(text("""
            UPDATE governance.memory_outbox
               SET status=:status, next_attempt_at=now()+(:delay || ' seconds')::interval,
                   lease_token=NULL, lease_expires_at=NULL, last_error_code=:error,
                   updated_at=now(), completed_at=CASE WHEN :exhausted THEN now() ELSE NULL END,
                   expires_at=CASE WHEN :exhausted THEN now()+interval '90 days' ELSE expires_at END
             WHERE event_id=:event_id AND tenant_id=:tenant_id AND user_id=:user_id
               AND status='leased' AND lease_token=:lease_token
             RETURNING registry_id, overlay_id, operation_id
        """), {
            "status": "dead_letter" if exhausted else "pending",
            "delay": delay, "error": error_code, "exhausted": exhausted,
            "event_id": lease["event_id"], "tenant_id": lease["tenant_id"],
            "user_id": lease["user_id"], "lease_token": lease["lease_token"],
        }).mappings().first()
        if changed is None:
            return
        if exhausted:
            connection.execute(text("""
                UPDATE governance.memory_registry
                   SET status='failed', last_error=:error, updated_at=now()
                 WHERE registry_id=:registry_id AND tenant_id=:tenant_id
                   AND user_id=:user_id AND operation_id=:operation_id
                   AND status IN ('pending_create','pending_update','pending_delete')
            """), {**dict(changed), **lease, "error": error_code})
            if changed["overlay_id"]:
                overlay_failed = connection.execute(text("""
                    UPDATE governance.memory_preference_overlays
                       SET status='failed', error_code=:error
                     WHERE overlay_id=:overlay_id AND tenant_id=:tenant_id
                       AND user_id=:user_id AND operation_id=:operation_id
                       AND status='pending'
                """), {**dict(changed), **lease, "error": error_code}).rowcount
                if overlay_failed:
                    record_event("memory.preference_overlay_failed", {})
        _record_audit(
            connection, {**lease, **dict(changed)},
            event_type="worker_dead_letter" if exhausted else "worker_retry_scheduled",
            status="failed" if exhausted else "pending",
            reason_code=error_code,
        )
    record_event("memory.preference_outbox", {
        "memory.operation": lease["event_type"],
        "memory.status": "dead_letter" if exhausted else "retry_scheduled",
        "memory.registry_id": lease["registry_id"],
        "memory.error_code": error_code,
    })
    if exhausted:
        # A dedicated counter lets the worker's Prometheus target alert on
        # dead-letter transitions; generic event attributes are not labels.
        record_event("memory.preference_dead_letter", {
            "memory.operation": lease["event_type"],
        })


def _claim_membership_erasure(engine: Any, *, lease_seconds: int) -> dict[str, Any] | None:
    """Lease one tenant-membership memory-erasure job using SKIP LOCKED."""
    with engine.begin() as connection:
        connection.execute(text("""
            UPDATE governance.memory_membership_erasure_jobs
               SET status=CASE WHEN attempts >= max_attempts THEN 'dead_letter' ELSE 'pending' END,
                   lease_token=NULL, lease_expires_at=NULL, updated_at=now(),
                   next_attempt_at=CASE WHEN attempts >= max_attempts THEN next_attempt_at ELSE now() END,
                   last_error_code=COALESCE(last_error_code, 'lease_expired_recovered')
             WHERE status='leased' AND lease_expires_at <= clock_timestamp()
        """))
        row = connection.execute(text("""
            SELECT job_id, tenant_id, subject_user_id, attempts, max_attempts
              FROM governance.memory_membership_erasure_jobs
             WHERE status='pending' AND next_attempt_at <= clock_timestamp()
             ORDER BY next_attempt_at, created_at
             FOR UPDATE SKIP LOCKED LIMIT 1
        """)).mappings().first()
        if row is None:
            return None
        token = uuid.uuid4().hex
        claimed = connection.execute(text("""
            UPDATE governance.memory_membership_erasure_jobs
               SET status='leased', attempts=attempts+1, lease_token=:token,
                   lease_expires_at=now()+(:lease_seconds || ' seconds')::interval,
                   updated_at=now()
             WHERE job_id=:job_id AND status='pending'
             RETURNING attempts
        """), {
            "token": token, "lease_seconds": max(1, int(lease_seconds)),
            "job_id": row["job_id"],
        }).scalar_one_or_none()
        if claimed is None:
            return None
        result = dict(row)
        result["attempts"] = int(claimed)
        result["lease_token"] = token
        return result


def _release_membership_erasure(
    engine: Any, job: dict[str, Any], *, error_code: str | None = None,
    completed: bool = False,
) -> None:
    attempts = int(job["attempts"])
    # Work-in-progress states are expected while the regular Outbox loop
    # delivers the queued delete. Poll them quickly; use exponential backoff
    # only for actual failures.
    progress_codes = {
        "preference_delete_pending", "preference_write_pending",
        "preference_delete_queued",
    }
    progress = not completed and error_code in progress_codes
    exhausted = not completed and not progress and attempts >= int(job["max_attempts"])
    delay = (
        1.0 if progress
        else min(300, 2 ** max(0, attempts - 1)) + random.uniform(0, 1)
    )
    with engine.begin() as connection:
        if completed:
            changed = connection.execute(text("""
                UPDATE governance.memory_membership_erasure_jobs
                   SET status='completed', lease_token=NULL, lease_expires_at=NULL,
                       updated_at=now(), completed_at=now(), last_error_code=NULL
                 WHERE job_id=:job_id AND status='leased' AND lease_token=:lease_token
            """), job).rowcount
        else:
            changed = connection.execute(text("""
                UPDATE governance.memory_membership_erasure_jobs
                   SET status=:status, lease_token=NULL, lease_expires_at=NULL,
                       attempts=CASE WHEN :progress THEN GREATEST(attempts-1,0) ELSE attempts END,
                       next_attempt_at=now()+(:delay || ' seconds')::interval,
                       updated_at=now(), last_error_code=:error
                 WHERE job_id=:job_id AND status='leased' AND lease_token=:lease_token
            """), {
                **job, "status": "dead_letter" if exhausted else "pending",
                "delay": delay, "error": error_code or "erasure_more_work_pending",
                "progress": progress,
            }).rowcount
    if changed:
        record_event("memory.membership_erasure_job", {
            "memory.status": "completed" if completed else (
                "dead_letter" if exhausted else "retry_scheduled"
            ),
            "memory.error_code": error_code or "",
        })
        if exhausted:
            record_event("memory.membership_erasure_dead_letter", {})


def _process_membership_erasure(
    engine: Any, job: dict[str, Any],
) -> None:
    """Queue at most one preference deletion or purge local memory after remote deletes.

    Membership and auth sessions are revoked synchronously by the API. This
    worker drains Mem0-backed preferences first; only then does it remove the
    tenant/user's PostgreSQL session and analysis memories and preference
    settings. Durable audit/Registry tombstones remain for accountability.
    """
    with engine.begin() as connection:
        candidate = connection.execute(text("""
            SELECT registry_id, memory_key, memory_type, status, version,
                   seller_id, mem0_memory_id, operation_id
              FROM governance.memory_registry
             WHERE tenant_id=:tenant_id AND user_id=:user_id
               AND (
                   status IN ('active','pending_create','pending_update','pending_delete','failed')
                   OR (
                       status='superseded' AND memory_type='scope_default'
                       AND expires_at IS NOT NULL AND expires_at <= clock_timestamp()
                       AND NOT EXISTS (
                           SELECT 1 FROM governance.memory_registry newer
                            WHERE newer.tenant_id=memory_registry.tenant_id
                              AND newer.user_id=memory_registry.user_id
                              AND newer.seller_id IS NOT DISTINCT FROM memory_registry.seller_id
                              AND newer.memory_key=memory_registry.memory_key
                              AND newer.version > memory_registry.version
                       )
                   )
               )
             ORDER BY CASE status WHEN 'pending_delete' THEN 0
                                  WHEN 'pending_create' THEN 1
                                  WHEN 'pending_update' THEN 2 ELSE 3 END,
                      updated_at, version
             LIMIT 1
        """), {
            "tenant_id": job["tenant_id"], "user_id": job["subject_user_id"],
        }).mappings().first()
    if candidate is not None:
        row = dict(candidate)
        if row["status"] == "pending_delete":
            _release_membership_erasure(engine, job, error_code="preference_delete_pending")
            return
        if row["status"] in {"pending_create", "pending_update"}:
            _release_membership_erasure(engine, job, error_code="preference_write_pending")
            return
        seller_id = str(row["seller_id"]) if row.get("seller_id") else None
        context = TenantContext(
            tenant_id=str(job["tenant_id"]), user_id=str(job["subject_user_id"]),
            role="analyst" if seller_id else "admin",
            scope_type="seller" if seller_id else PLATFORM_SCOPE, seller_id=seller_id,
        )
        writer = PreferenceWriter(context, engine_override=engine)
        decision = MemoryPolicyDecision(
            action="delete", tenant_id=context.tenant_id, user_id=context.user_id,
            subject=str(row["registry_id"]), memory_type=None, value_hash=None,
            accepted=True, user_confirmed=True, reason="membership_revocation_erasure",
        )
        writer.queue_delete(
            registry_id=str(row["registry_id"]), expected_version=int(row["version"]),
            idempotency_key=f"membership-erasure:{job['job_id']}:{row['registry_id']}",
            policy_decision=decision,
        )
        _release_membership_erasure(engine, job, error_code="preference_delete_queued")
        record_event("memory.membership_erasure_preference_queued", {})
        return

    # No live Registry entries remain. Clear residual overlays and the other
    # PostgreSQL memory stores before acknowledging the erasure job.
    with engine.begin() as connection:
        connection.execute(text("""
            UPDATE governance.memory_preference_overlays
               SET status='expired', value='{}'::jsonb, cleared_at=COALESCE(cleared_at, now()),
                   error_code=COALESCE(error_code, 'membership_revoked')
             WHERE tenant_id=:tenant_id AND user_id=:user_id
        """), {"tenant_id": job["tenant_id"], "user_id": job["subject_user_id"]})
        connection.execute(text("""
            DELETE FROM governance.memory_sessions
             WHERE tenant_id=:tenant_id AND user_id=:user_id
        """), {"tenant_id": job["tenant_id"], "user_id": job["subject_user_id"]})
        connection.execute(text("""
            DELETE FROM governance.analysis_memories
             WHERE tenant_id=:tenant_id AND user_id=:user_id
        """), {"tenant_id": job["tenant_id"], "user_id": job["subject_user_id"]})
        connection.execute(text("""
            DELETE FROM governance.memory_preference_settings
             WHERE tenant_id=:tenant_id AND user_id=:user_id
        """), {"tenant_id": job["tenant_id"], "user_id": job["subject_user_id"]})
        connection.execute(text("""
            DELETE FROM governance.memory_consolidation_jobs
             WHERE tenant_id=:tenant_id AND user_id=:user_id
        """), {"tenant_id": job["tenant_id"], "user_id": job["subject_user_id"]})
        connection.execute(text("""
            DELETE FROM governance.memory_source_snapshots
             WHERE tenant_id=:tenant_id AND user_id=:user_id
        """), {"tenant_id": job["tenant_id"], "user_id": job["subject_user_id"]})
        connection.execute(text("""
            DELETE FROM governance.memory_decisions
             WHERE tenant_id=:tenant_id AND user_id=:user_id
        """), {"tenant_id": job["tenant_id"], "user_id": job["subject_user_id"]})
    _release_membership_erasure(engine, job, completed=True)


def run_membership_erasure_once(
    engine: Any, *, lease_seconds: int = LEASE_SECONDS,
) -> bool:
    """Process at most one erasure coordinator job; primarily useful in tests."""
    job = _claim_membership_erasure(engine, lease_seconds=lease_seconds)
    if job is None:
        with engine.begin() as connection:
            oldest_age_ms = connection.execute(text("""
                SELECT COALESCE(MAX(EXTRACT(EPOCH FROM
                    (clock_timestamp()-created_at))*1000),0)
                  FROM governance.memory_membership_erasure_jobs
                 WHERE status IN ('pending','leased')
            """)).scalar_one()
        if float(oldest_age_ms or 0) >= 60_000:
            record_event("memory.membership_erasure_stalled", {
                "memory.max_age_ms": float(oldest_age_ms),
            })
        return False
    try:
        _process_membership_erasure(engine, job)
    except Exception as exc:
        if isinstance(exc, (Mem0Error, PreferencePayloadError)):
            code = exc.reason if isinstance(exc, Mem0Error) else exc.code
        else:
            code = _exception_code(exc)
        _release_membership_erasure(engine, job, error_code=str(code)[:128])
        logger.warning("membership memory erasure deferred: %s", code)
    return True


def _claim_account_erasure(engine: Any, *, lease_seconds: int) -> dict[str, Any] | None:
    with engine.begin() as connection:
        connection.execute(text("""
            UPDATE governance.account_erasure_jobs
               SET status=CASE WHEN attempts >= max_attempts THEN 'dead_letter' ELSE 'pending' END,
                   lease_token=NULL, lease_expires_at=NULL, updated_at=now(),
                   next_attempt_at=CASE WHEN attempts >= max_attempts THEN next_attempt_at ELSE now() END,
                   last_error_code=COALESCE(last_error_code,'lease_expired_recovered')
             WHERE status='leased' AND lease_expires_at <= clock_timestamp()
        """))
        row = connection.execute(text("""
            SELECT job_id,subject_user_id,anonymous_user_id,attempts,max_attempts
              FROM governance.account_erasure_jobs
             WHERE status='pending' AND next_attempt_at <= clock_timestamp()
             ORDER BY next_attempt_at,created_at
             FOR UPDATE SKIP LOCKED LIMIT 1
        """)).mappings().first()
        if row is None:
            return None
        token = uuid.uuid4().hex
        attempts = connection.execute(text("""
            UPDATE governance.account_erasure_jobs
               SET status='leased',attempts=attempts+1,lease_token=:token,
                   lease_expires_at=now()+(:lease_seconds || ' seconds')::interval,
                   updated_at=now()
             WHERE job_id=:job_id AND status='pending'
             RETURNING attempts
        """), {"token": token, "lease_seconds": max(1, int(lease_seconds)),
               "job_id": row["job_id"]}).scalar_one_or_none()
        if attempts is None:
            return None
        result = dict(row)
        result.update(attempts=int(attempts), lease_token=token)
        return result


def _release_account_erasure(
    engine: Any, job: dict[str, Any], *, error_code: str | None = None,
    completed: bool = False,
) -> None:
    attempts = int(job["attempts"])
    progress = not completed and error_code == "preference_erasure_pending"
    exhausted = not completed and not progress and attempts >= int(job["max_attempts"])
    delay = 1.0 if progress else min(300, 2 ** max(0, attempts - 1)) + random.uniform(0, 1)
    with engine.begin() as connection:
        if completed:
            changed = connection.execute(text("""
                UPDATE governance.account_erasure_jobs
                   SET status='completed',lease_token=NULL,lease_expires_at=NULL,
                       updated_at=now(),completed_at=now(),last_error_code=NULL
                 WHERE job_id=:job_id AND status='leased' AND lease_token=:lease_token
            """), job).rowcount
        else:
            changed = connection.execute(text("""
                UPDATE governance.account_erasure_jobs
                   SET status=:status,lease_token=NULL,lease_expires_at=NULL,
                       attempts=CASE WHEN :progress THEN GREATEST(attempts-1,0) ELSE attempts END,
                       next_attempt_at=now()+(:delay || ' seconds')::interval,
                       updated_at=now(),last_error_code=:error
                 WHERE job_id=:job_id AND status='leased' AND lease_token=:lease_token
            """), {**job,"status":"dead_letter" if exhausted else "pending",
            "delay":delay,"error":error_code or "account_erasure_pending",
            "progress":progress}).rowcount
    if changed:
        record_event("iam.account_erasure", {
            "status": "completed" if completed else ("dead_letter" if exhausted else "retry_scheduled"),
            "error_code": error_code or "",
        })


def _finish_account_erasure(engine: Any, job: dict[str, Any]) -> bool:
    user_id = str(job["subject_user_id"])
    anonymous_id = str(job["anonymous_user_id"])
    with engine.begin() as connection:
        children = connection.execute(text("""
            SELECT count(*) FILTER (WHERE status <> 'completed') AS pending,
                   count(*) FILTER (WHERE status = 'dead_letter') AS dead
              FROM governance.memory_membership_erasure_jobs
             WHERE subject_user_id=:user_id OR requested_by_user_id=:user_id
        """), {"user_id":user_id}).mappings().one()
        if int(children["dead"] or 0):
            return False
        if int(children["pending"] or 0):
            return False
        running_code_jobs = connection.execute(text("""
            SELECT 1 FROM governance.code_analysis_jobs
             WHERE user_id=:user_id
               AND status IN ('awaiting_confirmation','queued','generating','validating','running')
             LIMIT 1
        """), {"user_id":user_id}).first()
        if running_code_jobs:
            return False
        object_keys = connection.execute(text("""
            SELECT object_key FROM governance.code_analysis_files WHERE user_id=:user_id
            UNION
            SELECT object_key FROM governance.code_analysis_artifacts WHERE user_id=:user_id
        """), {"user_id":user_id}).scalars().all()

    if object_keys:
        from app.code_analysis.storage import get_artifact_store

        artifact_store = get_artifact_store()
        # Object deletion is idempotent. Keep database references if any object
        # store operation fails so the account erasure job retries safely.
        for object_key in object_keys:
            artifact_store.delete(str(object_key))

    with engine.begin() as connection:
        # Recheck child work after the external object-store boundary.
        unfinished = connection.execute(text("""
            SELECT 1 FROM governance.memory_membership_erasure_jobs
             WHERE (subject_user_id=:user_id OR requested_by_user_id=:user_id)
               AND status <> 'completed' LIMIT 1
        """), {"user_id":user_id}).first()
        if unfinished:
            return False

        params = {"user_id":user_id,"anonymous_id":anonymous_id}
        # Delete the user's conversational content and task state. Execution and
        # review rows cascade from turns; tasks do not, so they are explicit.
        connection.execute(text("DELETE FROM governance.conversation_tasks WHERE user_id=:user_id"), params)
        connection.execute(text("DELETE FROM governance.chat_sessions WHERE user_id=:user_id"), params)
        connection.execute(text("DELETE FROM governance.tenant_traces WHERE user_id=:user_id"), params)

        # Keep minimal audit facts with an irreversible random subject token.
        connection.execute(text("""
            UPDATE governance.audit_events
               SET user_id=:anonymous_id,detail='',request_path=NULL,correlation_id=NULL
             WHERE user_id=:user_id
        """), params)
        connection.execute(text("""
            UPDATE governance.lifecycle_audit_events
               SET user_id=:anonymous_id,authz_ref=NULL,turn_id=NULL,execution_id=NULL,
                   trace_id=NULL,payload='{}'::jsonb
             WHERE user_id=:user_id
        """), params)
        connection.execute(text("""
            UPDATE governance.authorization_snapshots SET user_id=:anonymous_id
             WHERE user_id=:user_id
        """), params)
        connection.execute(text("""
            UPDATE governance.memory_preference_audit
               SET user_id=:anonymous_id,actor_user_id=:anonymous_id
             WHERE user_id=:user_id OR actor_user_id=:user_id
        """), params)
        connection.execute(text("""
            UPDATE governance.memory_registry
               SET user_id=:anonymous_id,value_hash=NULL,mem0_memory_id=NULL,
                   source_turn_id=NULL,source_trace_id=NULL
             WHERE user_id=:user_id
        """), params)
        connection.execute(text("""
            UPDATE governance.memory_membership_erasure_jobs
               SET subject_user_id=:anonymous_id,requested_by_user_id=:anonymous_id
             WHERE subject_user_id=:user_id OR requested_by_user_id=:user_id
        """), params)
        connection.execute(text("""
            UPDATE governance.analysis_execution_audits
               SET user_id=:anonymous_id,trace_id=NULL,metadata='{}'::jsonb
             WHERE user_id=:user_id
        """), params)
        connection.execute(text("""
            UPDATE governance.data_replay_audits
               SET user_id=:anonymous_id,result='{}'::jsonb
             WHERE user_id=:user_id
        """), params)
        connection.execute(text("""
            UPDATE governance.permission_event_outbox
               SET actor_user_id=:anonymous_id,subject_user_id=CASE
                     WHEN subject_user_id=:user_id THEN :anonymous_id ELSE subject_user_id END,
                   payload_json='{}'::jsonb,last_error=NULL
             WHERE actor_user_id=:user_id OR subject_user_id=:user_id
        """), params)
        connection.execute(text("""
            UPDATE governance.llm_usage_events
               SET user_id=:anonymous_id,trace_id=NULL,chat_session_id=NULL
             WHERE user_id=:user_id
        """), params)
        connection.execute(text("""
            UPDATE governance.tenant_report_insights
               SET reviewed_by=CASE WHEN reviewed_by=:user_id THEN :anonymous_id ELSE reviewed_by END,
                   review_reason=CASE WHEN reviewed_by=:user_id THEN '' ELSE review_reason END
             WHERE reviewed_by=:user_id
        """), params)
        connection.execute(text("""
            UPDATE governance.tenant_reports
               SET actor=:anonymous_id,reason=''
             WHERE actor=:user_id
        """), params)
        connection.execute(text("""
            UPDATE governance.abac_policy_rules
               SET created_by=CASE WHEN created_by=:user_id THEN :anonymous_id ELSE created_by END,
                   updated_by=CASE WHEN updated_by=:user_id THEN :anonymous_id ELSE updated_by END
             WHERE created_by=:user_id OR updated_by=:user_id
        """), params)
        connection.execute(text("""
            UPDATE analytics.tenant_metric_targets SET created_by=:anonymous_id
             WHERE created_by=:user_id
        """), params)
        connection.execute(text("""
            UPDATE governance.memory_preference_overlays
               SET value='{}'::jsonb WHERE user_id=:user_id
        """), params)
        connection.execute(text("DELETE FROM governance.memory_outbox WHERE user_id=:user_id"), params)
        connection.execute(text("DELETE FROM governance.memory_preference_overlays WHERE user_id=:user_id"), params)

        connection.execute(text("DELETE FROM iam.membership_seller_grants WHERE user_id=:user_id OR granted_by=:user_id"), params)
        connection.execute(text("DELETE FROM iam.auth_sessions WHERE user_id=:user_id"), params)
        connection.execute(text("DELETE FROM iam.api_credentials WHERE user_id=:user_id"), params)
        connection.execute(text("DELETE FROM iam.tenant_memberships WHERE user_id=:user_id"), params)
        connection.execute(text("DELETE FROM governance.code_analysis_jobs WHERE user_id=:user_id"), params)
        connection.execute(text("DELETE FROM governance.code_analysis_files WHERE user_id=:user_id"), params)
        connection.execute(text("""
            UPDATE iam.users SET user_id=:anonymous_id,email=:email,display_name='Deleted user',
                password_hash=NULL,status='erased',updated_at=now()
             WHERE user_id=:user_id
        """), {**params,"email":f"{anonymous_id}@deleted.invalid"})
        connection.execute(text("""
            UPDATE governance.account_erasure_jobs
               SET subject_user_id=:anonymous_id,idempotency_key='completed:' || job_id,
                   status='completed',lease_token=NULL,
                   lease_expires_at=NULL,updated_at=now(),completed_at=now(),last_error_code=NULL
             WHERE job_id=:job_id AND status='leased' AND lease_token=:lease_token
        """), {**job,"anonymous_id":anonymous_id})
    return True


def run_account_erasure_once(engine: Any, *, lease_seconds: int = LEASE_SECONDS) -> bool:
    job = _claim_account_erasure(engine, lease_seconds=lease_seconds)
    if job is None:
        return False
    try:
        if _finish_account_erasure(engine, job):
            record_event("iam.account_erasure_completed", {})
        else:
            _release_account_erasure(engine, job, error_code="preference_erasure_pending")
    except Exception as exc:
        code = _exception_code(exc)
        _release_account_erasure(engine, job, error_code=code)
        logger.warning("global account erasure deferred: %s", code)
    return True


def run_one(
    *, engine_override: Any | None = None,
    adapter_factory: Callable[[TenantContext], Any] | None = None,
) -> bool:
    """Lease and process at most one Outbox event; return whether one was found."""
    settings = get_settings()
    db = engine_override or _worker_engine()
    started = time.perf_counter()
    lease_seconds = int(getattr(settings, "memory_preference_worker_lease_seconds", LEASE_SECONDS))
    erasure_job_processed = run_membership_erasure_once(
        db, lease_seconds=max(1, lease_seconds),
    )
    mem0_enabled = adapter_factory is not None or bool(getattr(settings, "mem0_enabled", False))
    if not mem0_enabled:
        account_job_processed = run_account_erasure_once(
            db, lease_seconds=max(1, lease_seconds),
        )
        if erasure_job_processed or account_job_processed:
            return True
        record_event("memory.preference_worker", {"memory.status": "mem0_disabled"})
        return False
    lease = _claim(db, lease_seconds=max(1, lease_seconds))
    if lease is None:
        with db.begin() as connection:
            pending_age_ms = connection.execute(text("""
                SELECT COALESCE(MAX(EXTRACT(EPOCH FROM (clock_timestamp()-o.created_at))*1000),0)
                  FROM governance.memory_outbox o
                  JOIN governance.memory_registry r ON r.registry_id=o.registry_id
                 WHERE o.status IN ('pending','leased')
                   AND r.status IN ('pending_create','pending_update','pending_delete')
            """)).scalar_one()
        if float(pending_age_ms or 0) >= 60_000:
            record_event("memory.preference_registry_outbox_inconsistency", {
                "memory.max_age_ms": float(pending_age_ms),
            })
        account_job_processed = run_account_erasure_once(
            db, lease_seconds=max(1, lease_seconds),
        )
        return erasure_job_processed or account_job_processed
    factory = adapter_factory or (
        lambda context: Mem0PreferenceAdapter(
            context, registry=PostgresPreferenceRegistry(context, engine_override=db),
        )
    )
    try:
        _deliver(lease, db, factory)
        record_event("memory.preference_worker_operation", {
            "memory.operation": lease["event_type"], "memory.status": "processed",
            "duration_ms": (time.perf_counter() - started) * 1000,
        })
    except Exception as exc:  # persisted error is a bounded code, never raw content
        if isinstance(exc, (Mem0Error, PreferencePayloadError)):
            error_code = exc.reason if isinstance(exc, Mem0Error) else exc.code
        else:
            error_code = _exception_code(exc)
        logger.warning("preference Outbox operation failed: %s", error_code)
        _fail(lease, db, str(error_code)[:128])
    run_account_erasure_once(db, lease_seconds=max(1, lease_seconds))
    return True


def main() -> None:  # pragma: no cover - deployment loop
    cleanup_only = "--cleanup-only" in sys.argv[1:]
    interval_seconds = 60
    if "--interval-seconds" in sys.argv[1:]:
        try:
            interval_seconds = max(1, int(sys.argv[sys.argv.index("--interval-seconds") + 1]))
        except (ValueError, IndexError):
            raise SystemExit("--interval-seconds requires a positive integer")
    db = _worker_engine()
    try:
        metrics_port = int(os.getenv("MEMORY_PREFERENCE_METRICS_PORT", "0"))
    except ValueError as exc:
        db.dispose()
        raise SystemExit("MEMORY_PREFERENCE_METRICS_PORT must be an integer") from exc
    metrics_server = start_metrics_server(metrics_port) if metrics_port > 0 else None
    try:
        while True:
            try:
                if cleanup_only:
                    cleanup_expired_state(db)
                    time.sleep(interval_seconds)
                elif not run_one(engine_override=db):
                    time.sleep(2)
            except Exception as exc:
                logger.error("preference Outbox worker iteration failed: %s", type(exc).__name__)
                time.sleep(5)
    finally:
        if metrics_server is not None:
            metrics_server.shutdown()
            metrics_server.server_close()
        db.dispose()


def _exception_code(exc: Exception) -> str:
    """Keep an explicit safe code while avoiding arbitrary exception text."""
    candidate = getattr(exc, "code", None) or getattr(exc, "reason", None)
    if candidate is None and len(getattr(exc, "args", ())) == 1:
        candidate = exc.args[0]
    if isinstance(candidate, str) and candidate and len(candidate) <= 96:
        if "_" in candidate and all(char.islower() or char.isdigit() or char in "._-" for char in candidate):
            return candidate
    return type(exc).__name__


if __name__ == "__main__":  # pragma: no cover
    main()
