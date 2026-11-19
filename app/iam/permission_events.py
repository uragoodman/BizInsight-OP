"""Durable permission-change outbox and optional cross-system delivery."""

from __future__ import annotations

import hashlib
import hmac
import json
import urllib.request
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy import create_engine, text

from app.core.config import get_settings
from app.data.database import engine


class PermissionEventSink(Protocol):
    def publish(self, event: dict[str, Any]) -> None: ...


class WebhookPermissionEventSink:
    def publish(self, event: dict[str, Any]) -> None:
        settings = get_settings()
        if not settings.permission_event_webhook_url:
            raise RuntimeError("permission event webhook is not configured")
        body = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            settings.permission_event_webhook_url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "X-BizInsight-Event": "permission_changed"},
        )
        if settings.permission_event_webhook_secret:
            signature = hmac.new(settings.permission_event_webhook_secret.encode(), body, hashlib.sha256).hexdigest()
            request.add_header("X-BizInsight-Signature", signature)
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status >= 300:
                raise RuntimeError(f"permission event webhook returned {response.status}")


def enqueue_permission_event(*, tenant_id: str, actor_user_id: str, subject_user_id: str | None, event_type: str, payload: dict[str, Any]) -> str:
    event_id = f"pev_{uuid4().hex}"
    idempotency = hashlib.sha256(f"{tenant_id}:{event_type}:{subject_user_id}:{json.dumps(payload, sort_keys=True, ensure_ascii=False)}".encode()).hexdigest()
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": tenant_id})
        conn.execute(text("SELECT set_config('app.user_id', :user, true)"), {"user": actor_user_id})
        row = conn.execute(text("""
            INSERT INTO governance.permission_event_outbox
              (event_id,tenant_id,actor_user_id,subject_user_id,event_type,idempotency_key,payload_json)
            VALUES (:id,:tenant,:actor,:subject,:type,:key,CAST(:payload AS jsonb))
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING event_id
        """), {"id": event_id, "tenant": tenant_id, "actor": actor_user_id,
                "subject": subject_user_id, "type": event_type, "key": idempotency,
                "payload": json.dumps(payload, ensure_ascii=False)}).scalar_one_or_none()
    return str(row or event_id)


def process_permission_events(*, sink: PermissionEventSink | None = None, limit: int = 50) -> dict[str, int]:
    """Claim and deliver a bounded batch; safe for cron or a worker loop."""

    if sink is None and not get_settings().permission_event_webhook_url:
        # An outbox is still durable and auditable without an external bus.
        # Never consume events or turn them into repeated failures merely
        # because delivery has not been configured.
        return {"claimed": 0, "delivered": 0, "failed": 0, "skipped": 1}
    sink = sink or WebhookPermissionEventSink()
    delivered = failed = 0
    settings = get_settings()
    worker_engine = engine
    disposable = None
    if settings.permission_event_database_url:
        disposable = create_engine(settings.permission_event_database_url, future=True, pool_pre_ping=True)
        worker_engine = disposable
    try:
      with worker_engine.begin() as conn:
        # The worker runs under the platform operational context.  RLS still
        # applies to tenant-local callers; the worker role is explicitly
        # provisioned separately when cross-tenant delivery is enabled.
        conn.execute(text("SELECT set_config('app.tenant_id', 'platform', true)"))
        rows = conn.execute(text("""
          SELECT * FROM governance.permission_event_outbox
          WHERE status IN ('pending','failed') AND attempts < 5
          ORDER BY created_at LIMIT :limit FOR UPDATE SKIP LOCKED
        """), {"limit": max(1, min(limit, 500))}).mappings().all()
        for row in rows:
            event = {"event_id": row["event_id"], "tenant_id": row["tenant_id"], "actor_user_id": row["actor_user_id"],
                     "subject_user_id": row["subject_user_id"], "event_type": row["event_type"],
                     "payload": row["payload_json"], "created_at": row["created_at"].isoformat() if isinstance(row["created_at"], datetime) else str(row["created_at"])}
            try:
                sink.publish(event)
            except Exception as exc:
                failed += 1
                conn.execute(text("""UPDATE governance.permission_event_outbox
                    SET status='failed', attempts=attempts+1, last_error=:error, claimed_at=now()
                    WHERE event_id=:id"""), {"id": row["event_id"], "error": type(exc).__name__})
            else:
                delivered += 1
                conn.execute(text("""UPDATE governance.permission_event_outbox
                    SET status='delivered', attempts=attempts+1, delivered_at=now(), claimed_at=NULL, last_error=NULL
                    WHERE event_id=:id"""), {"id": row["event_id"]})
      return {"claimed": len(rows), "delivered": delivered, "failed": failed}
    finally:
      if disposable is not None:
          disposable.dispose()
