"""Tenant notification contract with idempotent delivery audit.

Adapters intentionally receive a secret *reference*, never a tenant OAuth token
or database credential. Real Slack/email/Teams delivery is enabled only by a
deployment-owned adapter that resolves that reference.
"""
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import text

from app.data.database import engine
from app.tenancy.context import TenantContext


class TenantNotificationAdapter(Protocol):
    def deliver(self, *, channel: str, secret_reference: str, payload: dict, idempotency_key: str) -> None: ...


@dataclass
class RecordingNotificationAdapter:
    """Test/mock adapter; records contract calls and performs no external I/O."""
    deliveries: list[dict]
    def deliver(self, *, channel: str, secret_reference: str, payload: dict, idempotency_key: str) -> None:
        self.deliveries.append({"channel": channel, "secret_reference": secret_reference, "payload": payload, "idempotency_key": idempotency_key})


def deliver_tenant_event(context: TenantContext, event_id: str, payload: dict, adapter: TenantNotificationAdapter) -> int:
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": context.tenant_id})
        conn.execute(text("SELECT set_config('app.scope_type', :scope, true)"), {"scope": context.scope_type})
        channels = conn.execute(text("SELECT channel,secret_reference FROM iam.tenant_notification_channels WHERE enabled")).mappings().all()
        delivered = 0
        for row in channels:
            key = f"{event_id}:{row['channel']}"
            exists = conn.execute(text("SELECT 1 FROM governance.tenant_delivery_audit WHERE event_id=:event AND channel=:channel"), {"event": event_id, "channel": row["channel"]}).first()
            if exists: continue
            try:
                adapter.deliver(channel=row["channel"], secret_reference=row["secret_reference"], payload=payload, idempotency_key=key)
                status, detail = "delivered", ""
                delivered += 1
            except Exception as exc:
                status, detail = "failed", type(exc).__name__
            conn.execute(text("""INSERT INTO governance.tenant_delivery_audit(tenant_id,event_id,channel,delivery_key,status,detail)
                VALUES(:tenant,:event,:channel,:key,:status,:detail) ON CONFLICT DO NOTHING"""), {"tenant": context.tenant_id,"event":event_id,"channel":row["channel"],"key":key,"status":status,"detail":detail})
        return delivered


def process_tenant_outbox(adapter: TenantNotificationAdapter, limit: int = 100, max_attempts: int = 3) -> int:
    """Deliver pending tenant events with a database lease and retry state.

    The worker reads with platform scope only to locate the tenant, then creates
    a seller-scoped context for each delivery.  Channel selection and delivery
    audit therefore remain protected by the same RLS policies as API requests.
    """
    processed = 0
    for _ in range(limit):
        with engine.begin() as conn:
            conn.execute(text("SELECT set_config('app.tenant_id', 'platform', true)"))
            conn.execute(text("SELECT set_config('app.scope_type', 'platform', true)"))
            row = conn.execute(text("""
                SELECT o.tenant_id,o.event_id,o.payload_json,t.scope_type,tsa.seller_id
                FROM governance.tenant_outbox o
                JOIN iam.tenants t ON t.tenant_id=o.tenant_id AND t.status='active'
                LEFT JOIN iam.tenant_seller_accounts tsa ON tsa.tenant_id=o.tenant_id
                WHERE o.status='pending' AND o.attempts < :max_attempts
                ORDER BY o.occurred_at FOR UPDATE OF o SKIP LOCKED LIMIT 1
            """), {"max_attempts": max_attempts}).mappings().first()
            if row is None:
                break
            conn.execute(text("""UPDATE governance.tenant_outbox
                SET status='processing',attempts=attempts+1 WHERE tenant_id=:tenant AND event_id=:event"""),
                {"tenant": row["tenant_id"], "event": row["event_id"]})
        try:
            # Build the scoped context inside the failure boundary. A malformed
            # seller tenant (for example, a missing seller grant) must not leave
            # the claimed outbox row stuck in `processing` indefinitely.
            context = TenantContext(
                tenant_id=row["tenant_id"], user_id="tenant-outbox-worker", role="admin",
                scope_type=row["scope_type"], seller_id=row["seller_id"],
            )
            deliver_tenant_event(context, row["event_id"], dict(row["payload_json"]), adapter)
        except Exception as exc:
            with engine.begin() as conn:
                conn.execute(text("SELECT set_config('app.tenant_id', 'platform', true)"))
                conn.execute(text("SELECT set_config('app.scope_type', 'platform', true)"))
                conn.execute(text("""UPDATE governance.tenant_outbox
                    SET status=CASE WHEN attempts >= :max_attempts THEN 'dead_letter' ELSE 'pending' END,
                    last_error=:error WHERE tenant_id=:tenant AND event_id=:event"""),
                    {"max_attempts": max_attempts, "error": type(exc).__name__, "tenant": row["tenant_id"], "event": row["event_id"]})
        else:
            with engine.begin() as conn:
                conn.execute(text("SELECT set_config('app.tenant_id', 'platform', true)"))
                conn.execute(text("SELECT set_config('app.scope_type', 'platform', true)"))
                conn.execute(text("""UPDATE governance.tenant_outbox SET status='processed',processed_at=now(),last_error=''
                    WHERE tenant_id=:tenant AND event_id=:event"""), {"tenant": row["tenant_id"], "event": row["event_id"]})
            processed += 1
    return processed
