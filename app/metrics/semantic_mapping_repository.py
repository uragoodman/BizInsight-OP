"""Database-backed semantic mapping state machine.

The repository deliberately stores candidate handles and the server-resolved
metric ID together.  API callers can submit only mapping IDs; ownership,
scope, catalog and state are checked again before a mapping is consumed.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import text

from app.data.database import engine
from app.metrics.authorization import scope_fingerprint
from app.tenancy import TenantContext


def _scope(conn, context: TenantContext) -> None:
    conn.execute(text("SELECT set_config('app.tenant_id', :value, true)"), {"value": context.tenant_id})
    conn.execute(text("SELECT set_config('app.user_id', :value, true)"), {"value": context.user_id})


def create_pending_mappings(
    context: TenantContext,
    *,
    resolution_id: str,
    mention_id: str,
    candidates: list[dict],
    scope: dict,
    catalog_digest: str,
    trace_id: str,
    session_id: str | None,
    turn_id: str | None,
    ttl_minutes: int = 30,
) -> list[dict]:
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=max(1, ttl_minutes))
    scope_fp = scope_fingerprint(scope)
    created: list[dict] = []
    with engine.begin() as conn:
        _scope(conn, context)
        for candidate in candidates[:5]:
            if candidate.get("relation") == "no_match" or candidate.get("candidate_no") is None:
                continue
            mapping_id = str(uuid4())
            payload = {
                "mapping_id": mapping_id,
                "resolution_id": resolution_id,
                "mention_id": mention_id,
                "candidate_no": int(candidate["candidate_no"]),
                "metric_id": candidate.get("metric_id"),
                "tenant_id": context.tenant_id,
                "user_id": context.user_id,
                "session_id": session_id,
                "turn_id": turn_id,
                "scope_fingerprint": scope_fp,
                "catalog_digest": catalog_digest,
                "relation": candidate.get("relation", "related_but_different"),
                "reason": candidate.get("reason", ""),
                "evidence": candidate.get("evidence", ""),
                "trace_id": trace_id,
                "expires_at": expires_at,
            }
            conn.execute(text("""
                INSERT INTO governance.semantic_metric_mappings
                  (mapping_id,resolution_id,mention_id,candidate_no,metric_id,
                   tenant_id,user_id,session_id,turn_id,scope_fingerprint,
                   catalog_digest,relation,reason,evidence,trace_id,expires_at)
                VALUES (:mapping_id,:resolution_id,:mention_id,:candidate_no,:metric_id,
                        :tenant_id,:user_id,:session_id,:turn_id,:scope_fingerprint,
                        :catalog_digest,:relation,:reason,:evidence,:trace_id,:expires_at)
            """), payload)
            created.append(payload)
    return created


def load_confirmed_mappings(
    context: TenantContext,
    mapping_ids: list[str],
    *,
    scope: dict,
    catalog_digest: str,
    session_id: str | None,
) -> list[dict]:
    if not mapping_ids:
        return []
    with engine.begin() as conn:
        _scope(conn, context)
        rows = conn.execute(text("""
            SELECT mapping_id,resolution_id,mention_id,metric_id,scope_fingerprint,
                   catalog_digest,relation,status,session_id
            FROM governance.semantic_metric_mappings
            WHERE mapping_id = ANY(:mapping_ids)
              AND tenant_id=:tenant_id AND user_id=:user_id
              AND status='confirmed' AND expires_at > now()
              AND scope_fingerprint=:scope_fingerprint
              AND catalog_digest=:catalog_digest
              AND (session_id IS NOT DISTINCT FROM :session_id)
            ORDER BY created_at
        """), {
            "mapping_ids": mapping_ids,
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "scope_fingerprint": scope_fingerprint(scope),
            "catalog_digest": catalog_digest,
            "session_id": session_id,
        }).mappings().all()
    return [dict(row) for row in rows]


def confirm_mapping(
    context: TenantContext,
    mapping_id: str,
    *,
    scope: dict,
    catalog_digest: str,
    session_id: str | None,
    execution_request_id: str | None = None,
) -> dict | None:
    """Confirm one mapping with a conditional transactional state change."""
    request_id = execution_request_id or str(uuid4())
    with engine.begin() as conn:
        _scope(conn, context)
        params = {
            "mapping_id": mapping_id,
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "scope_fingerprint": scope_fingerprint(scope),
            "catalog_digest": catalog_digest,
            "session_id": session_id,
            "execution_request_id": request_id,
        }
        updated = conn.execute(text("""
            UPDATE governance.semantic_metric_mappings
               SET status='confirmed', confirmed_at=now(), execution_request_id=:execution_request_id
             WHERE mapping_id=:mapping_id AND tenant_id=:tenant_id AND user_id=:user_id
               AND status='pending' AND expires_at > now()
               AND scope_fingerprint=:scope_fingerprint AND catalog_digest=:catalog_digest
               AND (session_id IS NOT DISTINCT FROM :session_id)
        """), params).rowcount
        row = conn.execute(text("""
            SELECT mapping_id,resolution_id,mention_id,metric_id,scope_fingerprint,
                   catalog_digest,relation,status,execution_request_id
            FROM governance.semantic_metric_mappings
            WHERE mapping_id=:mapping_id AND tenant_id=:tenant_id AND user_id=:user_id
              AND scope_fingerprint=:scope_fingerprint
              AND catalog_digest=:catalog_digest
              AND (session_id IS NOT DISTINCT FROM :session_id)
        """), params).mappings().first()
    # A repeated request returns the same terminal row.  A stale/cross-scope
    # request returns None and can never authorize a query.
    return dict(row) if row and (updated or row["status"] == "confirmed") else None


def skip_mapping(
    context: TenantContext,
    mapping_id: str,
    *,
    scope: dict,
    catalog_digest: str,
    session_id: str | None,
) -> dict | None:
    """Record an explicit user decision to skip a pending candidate.

    The database status remains ``cancelled`` for terminal-state compatibility;
    ``user_decision`` distinguishes an intentional skip from expiry or a
    generic cancellation.  The conditional update makes retries idempotent.
    """
    with engine.begin() as conn:
        _scope(conn, context)
        params = {
            "mapping_id": mapping_id,
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "scope_fingerprint": scope_fingerprint(scope),
            "catalog_digest": catalog_digest,
            "session_id": session_id,
        }
        updated = conn.execute(text("""
            UPDATE governance.semantic_metric_mappings
               SET status='cancelled', user_decision='skipped_by_user', confirmed_at=now()
             WHERE mapping_id=:mapping_id AND tenant_id=:tenant_id AND user_id=:user_id
               AND status='pending' AND expires_at > now()
               AND scope_fingerprint=:scope_fingerprint AND catalog_digest=:catalog_digest
               AND (session_id IS NOT DISTINCT FROM :session_id)
        """), params).rowcount
        row = conn.execute(text("""
            SELECT mapping_id,resolution_id,mention_id,metric_id,scope_fingerprint,
                   catalog_digest,relation,status,user_decision,session_id
            FROM governance.semantic_metric_mappings
            WHERE mapping_id=:mapping_id AND tenant_id=:tenant_id AND user_id=:user_id
              AND scope_fingerprint=:scope_fingerprint
              AND catalog_digest=:catalog_digest
              AND (session_id IS NOT DISTINCT FROM :session_id)
        """), params).mappings().first()
    return dict(row) if row and (updated or row.get("user_decision") == "skipped_by_user") else None


def load_skipped_mappings(
    context: TenantContext,
    mapping_ids: list[str],
    *,
    scope: dict,
    catalog_digest: str,
    session_id: str | None,
) -> list[dict]:
    """Load only server-recorded user skips for the current scope/session."""
    if not mapping_ids:
        return []
    with engine.begin() as conn:
        _scope(conn, context)
        rows = conn.execute(text("""
            SELECT mapping_id,resolution_id,mention_id,metric_id,scope_fingerprint,
                   catalog_digest,relation,status,user_decision,session_id
            FROM governance.semantic_metric_mappings
            WHERE mapping_id = ANY(:mapping_ids)
              AND tenant_id=:tenant_id AND user_id=:user_id
              AND status='cancelled' AND user_decision='skipped_by_user'
              AND expires_at > now()
              AND scope_fingerprint=:scope_fingerprint
              AND catalog_digest=:catalog_digest
              AND (session_id IS NOT DISTINCT FROM :session_id)
            ORDER BY created_at
        """), {
            "mapping_ids": mapping_ids,
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "scope_fingerprint": scope_fingerprint(scope),
            "catalog_digest": catalog_digest,
            "session_id": session_id,
        }).mappings().all()
    return [dict(row) for row in rows]
