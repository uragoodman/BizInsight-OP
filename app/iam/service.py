"""Password identity, session and seller-grant services.

Passwords and database membership are the source of truth.  JWTs only carry a
short-lived session reference; every authenticated request reloads membership
and active seller grants, so revocation takes effect immediately.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import text

from app.core.config import get_settings
from app.data.database import engine
from app.data.rls import set_local_context
from app.tenancy import PLATFORM_SCOPE, SELLER_SCOPE, SellerGrant, TenantContext
from app.observability import record_event
from app.iam.permission_events import enqueue_permission_event


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def hash_password(password: str) -> str:
    minimum = 8 if get_settings().environment.lower() not in {"production", "prod"} else 12
    if not minimum <= len(password) <= 256:
        raise ValueError(f"Password must be {minimum} to 256 characters")
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt$16384$8$1${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str | None) -> bool:
    try:
        kind, n, r, p, salt, expected = str(encoded).split("$")
        if kind != "scrypt":
            return False
        actual = hashlib.scrypt(password.encode("utf-8"), salt=base64.urlsafe_b64decode(salt + "=" * (-len(salt) % 4)), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(actual, base64.urlsafe_b64decode(expected + "=" * (-len(expected) % 4)))
    except (ValueError, TypeError):
        return False


def encode_session_token(*, session_id: str, tenant_id: str, user_id: str, secret: str, expires_at: datetime) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64(json.dumps({"sid": session_id, "tenant_id": tenant_id, "sub": user_id, "exp": int(expires_at.timestamp())}, separators=(",", ":")).encode())
    signature = _b64(hmac.new(secret.encode("utf-8"), f"{header}.{payload}".encode("ascii"), hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}"


def _active_context(conn, *, tenant_id: str, user_id: str, session_id: str | None = None) -> TenantContext | None:
    if session_id:
        valid = conn.execute(text("""SELECT 1 FROM iam.auth_sessions
          WHERE session_id=:sid AND tenant_id=:tenant AND user_id=:user
            AND revoked_at IS NULL AND expires_at > now()"""), {"sid": session_id, "tenant": tenant_id, "user": user_id}).scalar()
        if not valid:
            return None
    row = conn.execute(text("""SELECT t.scope_type,m.role
      FROM iam.tenants t JOIN iam.tenant_memberships m ON m.tenant_id=t.tenant_id
      JOIN iam.users u ON u.user_id=m.user_id
      WHERE t.tenant_id=:tenant AND m.user_id=:user AND t.status='active'
        AND m.revoked_at IS NULL AND u.status='active'"""), {"tenant": tenant_id, "user": user_id}).mappings().one_or_none()
    if not row:
        return None
    grants = conn.execute(text("""SELECT seller_id,access_start_date,access_end_date
      FROM iam.membership_seller_grants
      WHERE tenant_id=:tenant AND user_id=:user AND revoked_at IS NULL
      ORDER BY seller_id"""), {"tenant": tenant_id, "user": user_id}).mappings().all()
    scope = str(row["scope_type"])
    if scope == PLATFORM_SCOPE:
        return TenantContext(tenant_id=tenant_id, user_id=user_id, role="admin", scope_type=PLATFORM_SCOPE)
    seller_grants = tuple(SellerGrant(str(item["seller_id"]), str(item["access_start_date"]) if item["access_start_date"] else None, str(item["access_end_date"]) if item["access_end_date"] else None) for item in grants)
    if not seller_grants:
        return None
    return TenantContext(tenant_id=tenant_id, user_id=user_id, role=str(row["role"]), scope_type=SELLER_SCOPE, seller_id=seller_grants[0].seller_id if len(seller_grants) == 1 else None, seller_grants=seller_grants)


def authenticate_session(*, tenant_id: str, user_id: str, session_id: str) -> TenantContext | None:
    with engine.begin() as conn:
        context = _active_context(conn, tenant_id=tenant_id, user_id=user_id, session_id=session_id)
        if context:
            conn.execute(text("UPDATE iam.auth_sessions SET last_seen_at=now() WHERE session_id=:sid"), {"sid": session_id})
            context = replace(context, auth_session_id=session_id)
        return context


def authenticate_api_credential(
    *, tenant_id: str, user_id: str, credential_id: str,
) -> TenantContext | None:
    """Reload API-key validity and current membership/grants from IAM."""
    with engine.begin() as conn:
        valid = conn.execute(text("""
            SELECT 1 FROM iam.api_credentials
            WHERE credential_id=:credential AND tenant_id=:tenant AND user_id=:user
              AND revoked_at IS NULL
              AND (expires_at IS NULL OR expires_at > now())
        """), {
            "credential": credential_id, "tenant": tenant_id, "user": user_id,
        }).scalar()
        if not valid:
            return None
        context = _active_context(conn, tenant_id=tenant_id, user_id=user_id)
        if context is not None:
            context = replace(context, auth_credential_id=credential_id)
        return context


def load_active_tenant_context(*, tenant_id: str, user_id: str) -> TenantContext | None:
    """Reload one user's current tenant membership and grants from IAM.

    Operational workers use this instead of constructing caller identity or
    role data themselves.  It deliberately omits session-token validation,
    but still requires an active tenant, user, membership, and seller grants.
    """

    with engine.begin() as conn:
        return _active_context(conn, tenant_id=tenant_id, user_id=user_id)


def login(email: str, password: str, *, secret: str, session_hours: int) -> tuple[str, TenantContext, datetime]:
    normalized = email.strip().lower()
    with engine.begin() as conn:
        rows = conn.execute(text("""SELECT u.user_id,u.password_hash,m.tenant_id
          FROM iam.users u JOIN iam.tenant_memberships m ON m.user_id=u.user_id
          JOIN iam.tenants t ON t.tenant_id=m.tenant_id
          WHERE lower(u.email)=:email AND u.status='active' AND m.revoked_at IS NULL AND t.status='active'
          ORDER BY CASE WHEN t.scope_type='seller' THEN 0 ELSE 1 END, m.created_at LIMIT 2"""), {"email": normalized}).mappings().all()
        if len(rows) != 1 or not verify_password(password, rows[0]["password_hash"]):
            raise PermissionError("Invalid email or password")
        user_id, tenant_id = str(rows[0]["user_id"]), str(rows[0]["tenant_id"])
        context = _active_context(conn, tenant_id=tenant_id, user_id=user_id)
        if not context:
            raise PermissionError("No active organization access")
        expires_at = datetime.now(timezone.utc) + timedelta(hours=max(1, min(session_hours, 24)))
        session_id = uuid4().hex
        conn.execute(text("INSERT INTO iam.auth_sessions(session_id,tenant_id,user_id,expires_at) VALUES(:sid,:tenant,:user,:expires)"), {"sid": session_id, "tenant": tenant_id, "user": user_id, "expires": expires_at})
    return encode_session_token(session_id=session_id, tenant_id=tenant_id, user_id=user_id, secret=secret, expires_at=expires_at), context, expires_at


def logout(session_id: str) -> None:
    with engine.begin() as conn:
        conn.execute(text("UPDATE iam.auth_sessions SET revoked_at=now() WHERE session_id=:sid AND revoked_at IS NULL"), {"sid": session_id})


def permissions(context: TenantContext) -> set[str]:
    if context.is_platform_admin or context.role in {"owner", "admin"}:
        return {
            "read", "analyze", "report_export", "manage_members", "manage_sellers",
            "manage_knowledge", "code_analysis", "code_analysis_read", "code_analysis_share",
            "code_analysis_audit", "code_analysis_admin", "manage_usage",
        }
    if context.role == "analyst":
        return {"read", "analyze", "report_export", "code_analysis", "code_analysis_read"}
    if context.role == "reviewer":
        return {"read", "analyze", "report_export", "code_analysis_read"}
    return {"read"}


def require_permission(context: TenantContext, permission: str) -> None:
    if permission not in permissions(context):
        raise PermissionError(f"Permission denied: {permission}")


def _record_permission_change(context: TenantContext, event: str, *, subject_user_id: str, role: str | None = None, grants: list[dict] | None = None) -> None:
    """Audit IAM changes without making audit storage a permissions source."""
    payload = {"subject_user_id": subject_user_id, "role": role, "grants": grants or []}
    record_event("iam.permission_changed", {
        "event": event, "tenant_id": context.tenant_id,
        "user_id": context.user_id, **payload,
    })
    try:
        enqueue_permission_event(
            tenant_id=context.tenant_id,
            actor_user_id=context.user_id,
            subject_user_id=subject_user_id,
            event_type="iam.permission_changed",
            payload={"event": event, **payload},
        )
        with engine.begin() as conn:
            conn.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": context.tenant_id})
            conn.execute(text("SELECT set_config('app.user_id', :user, true)"), {"user": context.user_id})
            conn.execute(text("SELECT set_config('app.scope_type', :scope, true)"), {"scope": context.scope_type})
            conn.execute(text("""
                INSERT INTO governance.lifecycle_audit_events
                  (event_name,tenant_id,user_id,payload)
                VALUES ('iam.permission_changed',:tenant,:user,CAST(:payload AS jsonb))
            """), {"tenant": context.tenant_id, "user": context.user_id, "payload": json.dumps({"event": event, **payload}, ensure_ascii=False)})
    except Exception:
        # Keep IAM state changes compatible during a rolling migration; the
        # runtime event remains available for monitoring.
        return


def profile(context: TenantContext) -> dict:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT email,display_name FROM iam.users WHERE user_id=:id"), {"id": context.user_id}).mappings().one()
        tenant = conn.execute(text("SELECT tenant_name FROM iam.tenants WHERE tenant_id=:id"), {"id": context.tenant_id}).scalar_one()
    return {"user_id": context.user_id, "email": row["email"], "display_name": row["display_name"], "tenant_id": context.tenant_id, "tenant_name": tenant, "role": context.role, "scope_type": context.scope_type, "seller_grants": [{"seller_id": grant.seller_id, "start_date": grant.start_date, "end_date": grant.end_date} for grant in context.seller_grants], "permissions": sorted(permissions(context))}


def bootstrap(*, email: str, password: str, display_name: str) -> None:
    """Create the one internal platform administrator, only on an empty IAM DB."""
    with engine.begin() as conn:
        if conn.execute(text("SELECT 1 FROM iam.users LIMIT 1")).scalar():
            raise ValueError("Bootstrap is already complete")
        user_id = f"usr_{uuid4().hex}"
        conn.execute(text("INSERT INTO iam.tenants(tenant_id,tenant_name,scope_type) VALUES('platform','平台运营团队','platform') ON CONFLICT DO NOTHING"))
        conn.execute(text("INSERT INTO iam.users(user_id,display_name,email,password_hash) VALUES(:id,:name,:email,:hash)"), {"id": user_id, "name": display_name.strip() or "平台管理员", "email": email.strip().lower(), "hash": hash_password(password)})
        conn.execute(text("INSERT INTO iam.tenant_memberships(tenant_id,user_id,role,granted_by) VALUES('platform',:id,'admin',:id)"), {"id": user_id})


def list_members(context: TenantContext) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(text("""SELECT u.user_id,u.email,u.display_name,u.status,m.role,m.created_at,
          COALESCE(jsonb_agg(jsonb_build_object('seller_id',g.seller_id,'start_date',g.access_start_date,'end_date',g.access_end_date)
            ORDER BY g.seller_id) FILTER(WHERE g.seller_id IS NOT NULL),'[]'::jsonb) AS grants
          FROM iam.tenant_memberships m JOIN iam.users u ON u.user_id=m.user_id
          LEFT JOIN iam.membership_seller_grants g ON g.tenant_id=m.tenant_id AND g.user_id=m.user_id AND g.revoked_at IS NULL
          WHERE m.tenant_id=:tenant AND m.revoked_at IS NULL
          GROUP BY u.user_id,u.email,u.display_name,u.status,m.role,m.created_at ORDER BY m.created_at"""), {"tenant": context.tenant_id}).mappings().all()
    return [dict(row) for row in rows]


def create_member(context: TenantContext, *, email: str, display_name: str, password: str, role: str, grants: list[dict]) -> dict:
    if role not in {"admin", "analyst", "reviewer", "viewer"}:
        raise ValueError("Unsupported member role")
    if context.scope_type == PLATFORM_SCOPE and role != "admin":
        raise ValueError("Platform members must be administrators")
    normalized = email.strip().lower()
    with engine.begin() as conn:
        set_local_context(conn, context, scope_type=context.scope_type)
        existing = conn.execute(text("SELECT user_id FROM iam.users WHERE lower(email)=:email"), {"email": normalized}).scalar()
        user_id = str(existing) if existing else f"usr_{uuid4().hex}"
        if existing:
            # Serialize reactivation with revocation so a create_member request
            # cannot pass the active-job check, wait behind revocation, and
            # then re-enable the membership while erasure is still draining.
            conn.execute(text("""
                SELECT pg_advisory_xact_lock(hashtextextended(:lock_key, 0))
            """), {"lock_key": "|".join((
                "membership-erasure-subject", context.tenant_id, user_id,
            ))})
        if existing and conn.execute(text("""
            SELECT 1 FROM governance.memory_membership_erasure_jobs
             WHERE tenant_id=:tenant AND subject_user_id=:user
               AND status IN ('pending','leased')
             LIMIT 1
        """), {"tenant": context.tenant_id, "user": user_id}).first():
            raise ValueError("Member memory erasure is still in progress")
        if not existing:
            conn.execute(text("INSERT INTO iam.users(user_id,display_name,email,password_hash) VALUES(:id,:name,:email,:hash)"), {"id": user_id, "name": display_name.strip() or normalized, "email": normalized, "hash": hash_password(password)})
        conn.execute(text("""INSERT INTO iam.tenant_memberships(tenant_id,user_id,role,granted_by,revoked_at)
          VALUES(:tenant,:user,:role,:actor,NULL)
          ON CONFLICT(tenant_id,user_id) DO UPDATE SET role=EXCLUDED.role,granted_by=EXCLUDED.granted_by,revoked_at=NULL"""), {"tenant": context.tenant_id, "user": user_id, "role": role, "actor": context.user_id})
        _replace_grants(conn, context, user_id, grants)
    _record_permission_change(context, "member_created_or_reactivated", subject_user_id=user_id, role=role, grants=grants)
    return {"user_id": user_id, "email": normalized, "role": role}


def update_member(context: TenantContext, user_id: str, *, role: str, grants: list[dict]) -> None:
    if user_id == context.user_id and role not in {"owner", "admin"}:
        raise ValueError("An administrator cannot remove their own administrative access")
    if role not in {"admin", "analyst", "reviewer", "viewer"}:
        raise ValueError("Unsupported member role")
    with engine.begin() as conn:
        exists = conn.execute(text("SELECT 1 FROM iam.tenant_memberships WHERE tenant_id=:tenant AND user_id=:user AND revoked_at IS NULL"), {"tenant": context.tenant_id, "user": user_id}).scalar()
        if not exists:
            raise ValueError("Member not found")
        conn.execute(text("UPDATE iam.tenant_memberships SET role=:role,granted_by=:actor WHERE tenant_id=:tenant AND user_id=:user"), {"role": role, "actor": context.user_id, "tenant": context.tenant_id, "user": user_id})
        _replace_grants(conn, context, user_id, grants)
    _record_permission_change(context, "member_updated", subject_user_id=user_id, role=role, grants=grants)


class MembershipErasureError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class AccountErasureError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def request_account_erasure(
    context: TenantContext, *, idempotency_key: str,
) -> dict:
    """Immediately revoke a global account and enqueue cross-tenant erasure.

    The database function is the authorization boundary: it reads the
    authenticated user from transaction-local context and creates the memory
    erasure fan-out before revoking all sessions, credentials and memberships.
    """
    if not idempotency_key or len(idempotency_key) > 128:
        raise AccountErasureError("account_erasure_idempotency_key_invalid")
    with engine.begin() as connection:
        set_local_context(connection, context, scope_type=context.scope_type)
        row = connection.execute(text("""
            SELECT job_id,status FROM governance.request_account_erasure(:idempotency_key)
        """), {
            "idempotency_key": idempotency_key,
        }).mappings().one()
    return dict(row)


def request_membership_erasure(
    context: TenantContext, subject_user_id: str, *, idempotency_key: str,
) -> dict:
    """Revoke one tenant membership immediately and durably erase its memories.

    This is tenant-membership scoped. It does not disable a global IAM user that
    may still belong to another tenant.
    """
    if subject_user_id == context.user_id:
        raise MembershipErasureError("self_membership_revocation_forbidden")
    if not idempotency_key or len(idempotency_key) > 128:
        raise MembershipErasureError("erasure_idempotency_key_invalid")
    job_id = f"merase_{uuid4().hex}"
    with engine.begin() as connection:
        set_local_context(connection, context, scope_type=context.scope_type)
        for lock_key in (
            "|".join(("membership-erasure-idempotency", context.tenant_id,
                      context.user_id, idempotency_key)),
            "|".join(("membership-erasure-subject", context.tenant_id, subject_user_id)),
        ):
            connection.execute(text("""
                SELECT pg_advisory_xact_lock(hashtextextended(:lock_key, 0))
            """), {"lock_key": lock_key})
        prior = connection.execute(text("""
            SELECT job_id, status, subject_user_id, created_at, completed_at
              FROM governance.memory_membership_erasure_jobs
             WHERE tenant_id=:tenant AND requested_by_user_id=:actor
               AND idempotency_key=:idempotency_key
        """), {
            "tenant": context.tenant_id, "actor": context.user_id,
            "idempotency_key": idempotency_key,
        }).mappings().first()
        if prior is not None:
            if prior["subject_user_id"] != subject_user_id:
                raise MembershipErasureError("erasure_idempotency_key_reused")
            return dict(prior)
        active = connection.execute(text("""
            SELECT job_id, status, subject_user_id, created_at, completed_at
              FROM governance.memory_membership_erasure_jobs
             WHERE tenant_id=:tenant AND subject_user_id=:subject
               AND status IN ('pending','leased')
             ORDER BY created_at DESC LIMIT 1
        """), {"tenant": context.tenant_id, "subject": subject_user_id}).mappings().first()
        if active is not None:
            return dict(active)
        membership = connection.execute(text("""
            SELECT role, revoked_at FROM iam.tenant_memberships
             WHERE tenant_id=:tenant AND user_id=:subject FOR UPDATE
        """), {"tenant": context.tenant_id, "subject": subject_user_id}).mappings().first()
        if membership is None or membership["revoked_at"] is not None:
            raise MembershipErasureError("active_member_not_found")
        if membership["role"] == "owner":
            other_owner = connection.execute(text("""
                SELECT 1 FROM iam.tenant_memberships
                 WHERE tenant_id=:tenant AND user_id<>:subject
                   AND role='owner' AND revoked_at IS NULL
                 LIMIT 1
            """), {"tenant": context.tenant_id, "subject": subject_user_id}).first()
            if other_owner is None:
                raise MembershipErasureError("cannot_revoke_last_owner")
        created = connection.execute(text("""
            INSERT INTO governance.memory_membership_erasure_jobs
                (job_id, tenant_id, subject_user_id, requested_by_user_id, idempotency_key)
            VALUES (:job_id, :tenant, :subject, :actor, :idempotency_key)
            RETURNING job_id, status, subject_user_id, created_at, completed_at
        """), {
            "job_id": job_id, "tenant": context.tenant_id,
            "subject": subject_user_id, "actor": context.user_id,
            "idempotency_key": idempotency_key,
        }).mappings().one()
        revoked = connection.execute(text("""
            UPDATE iam.tenant_memberships SET revoked_at=now()
             WHERE tenant_id=:tenant AND user_id=:subject AND revoked_at IS NULL
        """), {"tenant": context.tenant_id, "subject": subject_user_id}).rowcount
        if revoked != 1:
            raise MembershipErasureError("membership_state_conflict")
        connection.execute(text("""
            UPDATE iam.auth_sessions SET revoked_at=now()
             WHERE tenant_id=:tenant AND user_id=:subject AND revoked_at IS NULL
        """), {"tenant": context.tenant_id, "subject": subject_user_id})
    _record_permission_change(
        context, "member_revoked_memory_erasure_requested",
        subject_user_id=subject_user_id,
    )
    return dict(created)


def get_membership_erasure_job(
    context: TenantContext, job_id: str,
) -> dict | None:
    with engine.begin() as connection:
        set_local_context(connection, context, scope_type=context.scope_type)
        row = connection.execute(text("""
            SELECT job_id, subject_user_id, status, attempts, last_error_code,
                   created_at, updated_at, completed_at
              FROM governance.memory_membership_erasure_jobs
             WHERE tenant_id=:tenant AND job_id=:job_id
        """), {
            "tenant": context.tenant_id, "job_id": job_id,
        }).mappings().first()
    return dict(row) if row else None


def retry_membership_erasure(
    context: TenantContext, job_id: str,
) -> dict:
    """Explicitly retry a dead-letter membership memory-erasure job."""
    with engine.begin() as connection:
        set_local_context(connection, context, scope_type=context.scope_type)
        row = connection.execute(text("""
            SELECT job_id, subject_user_id, status
              FROM governance.memory_membership_erasure_jobs
             WHERE tenant_id=:tenant AND job_id=:job_id
             FOR UPDATE
        """), {
            "tenant": context.tenant_id, "job_id": job_id,
        }).mappings().first()
        if row is None:
            raise MembershipErasureError("erasure_job_not_found")
        if row["status"] != "dead_letter":
            raise MembershipErasureError("erasure_job_not_retryable")
        retried = connection.execute(text("""
            UPDATE governance.memory_membership_erasure_jobs
               SET status='pending', attempts=0, next_attempt_at=now(),
                   lease_token=NULL, lease_expires_at=NULL,
                   last_error_code=NULL, completed_at=NULL, updated_at=now()
             WHERE job_id=:job_id
             RETURNING job_id, subject_user_id, status, attempts,
                       last_error_code, created_at, updated_at, completed_at
        """), {"job_id": job_id}).mappings().one()
    return dict(retried)


def _replace_grants(conn, context: TenantContext, user_id: str, grants: list[dict]) -> None:
    if context.scope_type == PLATFORM_SCOPE:
        if grants:
            raise ValueError("Platform administrators do not use seller grants")
        return
    normalized: dict[str, dict] = {}
    for grant in grants:
        seller_id = str(grant.get("seller_id", "")).strip()
        if not seller_id:
            raise ValueError("seller_id is required for a seller grant")
        if context.seller_grant(seller_id) is None:
            raise ValueError("Cannot grant a seller you cannot access")
        start, end = grant.get("start_date") or None, grant.get("end_date") or None
        if start and end and str(start) > str(end):
            raise ValueError("Seller grant end date cannot be earlier than start date")
        normalized[seller_id] = {"start": start, "end": end}
    if not normalized:
        raise ValueError("A seller-scope member needs at least one seller grant")
    conn.execute(text("UPDATE iam.membership_seller_grants SET revoked_at=now() WHERE tenant_id=:tenant AND user_id=:user AND revoked_at IS NULL"), {"tenant": context.tenant_id, "user": user_id})
    for seller_id, grant in normalized.items():
        conn.execute(text("""INSERT INTO iam.membership_seller_grants(tenant_id,user_id,seller_id,access_start_date,access_end_date,granted_by,revoked_at)
          VALUES(:tenant,:user,:seller,:start,:end,:actor,NULL)
          ON CONFLICT(tenant_id,user_id,seller_id) DO UPDATE SET access_start_date=EXCLUDED.access_start_date,access_end_date=EXCLUDED.access_end_date,granted_by=EXCLUDED.granted_by,revoked_at=NULL"""), {"tenant": context.tenant_id, "user": user_id, "seller": seller_id, "start": grant["start"], "end": grant["end"], "actor": context.user_id})


def create_organization(context: TenantContext, *, tenant_name: str, email: str, display_name: str, password: str, grants: list[dict]) -> dict:
    if not context.is_platform_admin:
        raise PermissionError("Only a platform administrator can create organizations")
    if not grants:
        raise ValueError("An organization needs at least one seller grant")
    normalized = email.strip().lower()
    with engine.begin() as conn:
        if conn.execute(text("SELECT 1 FROM iam.users WHERE lower(email)=:email"), {"email": normalized}).scalar():
            raise ValueError("Email already belongs to an account; create a membership instead")
        seller_ids = {str(item.get("seller_id", "")).strip() for item in grants}
        if "" in seller_ids:
            raise ValueError("seller_id is required for every seller grant")
        found = conn.execute(text("SELECT seller_id FROM raw.sellers WHERE seller_id = ANY(CAST(:ids AS varchar[]))"), {"ids": list(seller_ids)}).scalars().all()
        if set(found) != seller_ids:
            raise ValueError("One or more seller IDs do not exist")
        tenant_id, user_id = f"tenant_{uuid4().hex}", f"usr_{uuid4().hex}"
        conn.execute(text("INSERT INTO iam.tenants(tenant_id,tenant_name,scope_type) VALUES(:tenant,:name,'seller')"), {"tenant": tenant_id, "name": tenant_name.strip()})
        conn.execute(text("INSERT INTO iam.users(user_id,display_name,email,password_hash) VALUES(:user,:name,:email,:hash)"), {"user": user_id, "name": display_name.strip() or normalized, "email": normalized, "hash": hash_password(password)})
        conn.execute(text("INSERT INTO iam.tenant_memberships(tenant_id,user_id,role,granted_by) VALUES(:tenant,:user,'owner',:actor)"), {"tenant": tenant_id, "user": user_id, "actor": context.user_id})
        for grant in grants:
            start, end = grant.get("start_date") or None, grant.get("end_date") or None
            if start and end and str(start) > str(end):
                raise ValueError("Seller grant end date cannot be earlier than start date")
            conn.execute(text("INSERT INTO iam.membership_seller_grants(tenant_id,user_id,seller_id,access_start_date,access_end_date,granted_by) VALUES(:tenant,:user,:seller,:start,:end,:actor)"), {"tenant": tenant_id, "user": user_id, "seller": str(grant["seller_id"]), "start": start, "end": end, "actor": context.user_id})
    return {"tenant_id": tenant_id, "owner_user_id": user_id, "email": normalized}
