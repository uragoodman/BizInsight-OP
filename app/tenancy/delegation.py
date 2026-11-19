"""Tenant-bound A2A delegation tokens and idempotent task envelope."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Protocol
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.tenancy.context import TenantContext
from sqlalchemy import text
from app.data.database import engine


class DelegationAdapter(Protocol):
    """Transport boundary for a remote A2A specialist."""
    def dispatch(self, *, task: dict, delegation_token: str) -> dict: ...


@dataclass(frozen=True)
class DelegationClaims:
    task_id: str
    tenant_id: str
    user_id: str
    role: str
    scope_type: str
    seller_id: str | None
    correlation_id: str
    exp: int


def issue_delegation_token(context: TenantContext, *, correlation_id: str, ttl_seconds: int = 300, task_id: str | None = None, secret: str) -> str:
    """Issue a compact HMAC token for a remote specialist; no DB credential leaks."""
    if not secret:
        raise ValueError("delegation signing secret is required")
    claims = asdict(DelegationClaims(task_id or str(uuid4()), context.tenant_id, context.user_id, context.role, context.scope_type, context.seller_id, correlation_id, int((datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).timestamp())))
    payload = base64.urlsafe_b64encode(json.dumps(claims, separators=(",", ":")).encode()).rstrip(b"=")
    signature = hmac.new(secret.encode(), payload, hashlib.sha256).digest()
    return payload.decode() + "." + base64.urlsafe_b64encode(signature).rstrip(b"=").decode()


def verify_delegation_token(token: str, *, secret: str) -> DelegationClaims:
    try:
        payload, signature = token.split(".")
        raw = payload.encode()
        expected = hmac.new(secret.encode(), raw, hashlib.sha256).digest()
        supplied = base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4))
        if not hmac.compare_digest(expected, supplied): raise ValueError("invalid signature")
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        result = DelegationClaims(**claims)
    except Exception as exc:
        raise ValueError("invalid delegation token") from exc
    if result.exp <= int(datetime.now(timezone.utc).timestamp()):
        raise ValueError("expired delegation token")
    return result


def create_delegated_task(context: TenantContext, claims: DelegationClaims, payload: dict, idempotency_key: str) -> dict:
    """Persist an idempotent, cancellable remote-task envelope under RLS."""
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": context.tenant_id})
        conn.execute(text("SELECT set_config('app.scope_type', :scope, true)"), {"scope": context.scope_type})
        conn.execute(text("""INSERT INTO governance.tenant_delegated_tasks(tenant_id,task_id,idempotency_key,correlation_id,status,payload_json,expires_at)
            VALUES(:tenant,:task,:key,:correlation,'pending',CAST(:payload AS jsonb),to_timestamp(:exp)) ON CONFLICT(tenant_id,idempotency_key) DO NOTHING"""), {"tenant":context.tenant_id,"task":claims.task_id,"key":idempotency_key,"correlation":claims.correlation_id,"payload":json.dumps(payload),"exp":claims.exp})
        row=conn.execute(text("SELECT * FROM governance.tenant_delegated_tasks WHERE idempotency_key=:key"), {"key":idempotency_key}).mappings().one()
    return dict(row)


def cancel_delegated_task(context: TenantContext, task_id: str) -> bool:
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": context.tenant_id})
        conn.execute(text("SELECT set_config('app.scope_type', :scope, true)"), {"scope": context.scope_type})
        result=conn.execute(text("UPDATE governance.tenant_delegated_tasks SET status='cancelled',updated_at=now() WHERE task_id=:task AND status IN ('pending','running')"), {"task":task_id})
    return result.rowcount == 1


def dispatch_delegated_task(
    context: TenantContext, task_id: str, *, adapter: DelegationAdapter, secret: str,
) -> dict:
    """Send one persisted task through an injected A2A transport.

    The remote specialist receives a short-lived signed token containing the
    tenant boundary.  Its response cannot choose a different tenant because the
    local task row is read and updated under the caller's RLS context.
    """
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": context.tenant_id})
        conn.execute(text("SELECT set_config('app.scope_type', :scope, true)"), {"scope": context.scope_type})
        row = conn.execute(text("""SELECT * FROM governance.tenant_delegated_tasks
            WHERE task_id=:task AND status='pending' FOR UPDATE"""), {"task": task_id}).mappings().first()
        if not row:
            raise KeyError("Pending delegated task not found")
        conn.execute(text("UPDATE governance.tenant_delegated_tasks SET status='running',updated_at=now() WHERE task_id=:task"), {"task": task_id})
        task = dict(row)
    token = issue_delegation_token(context, task_id=task_id, correlation_id=task["correlation_id"], ttl_seconds=max(1, int(task["expires_at"].timestamp() - datetime.now(timezone.utc).timestamp())), secret=secret)
    try:
        response = adapter.dispatch(task=task, delegation_token=token)
    except Exception as exc:
        with engine.begin() as conn:
            conn.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": context.tenant_id})
            conn.execute(text("SELECT set_config('app.scope_type', :scope, true)"), {"scope": context.scope_type})
            conn.execute(text("UPDATE governance.tenant_delegated_tasks SET status='failed',updated_at=now() WHERE task_id=:task"), {"task": task_id})
        raise RuntimeError("Remote delegation failed") from exc
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": context.tenant_id})
        conn.execute(text("SELECT set_config('app.scope_type', :scope, true)"), {"scope": context.scope_type})
        conn.execute(text("UPDATE governance.tenant_delegated_tasks SET status='completed',updated_at=now() WHERE task_id=:task"), {"task": task_id})
    return response
