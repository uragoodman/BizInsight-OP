"""Tenant-scoped ABAC policy registry.

The JSON file remains a bootstrap policy for local development.  Once a
tenant has database rules, those rules are the authoritative tenant policy;
all writes are administrator-only and versioned.  The module deliberately
accepts only a small, auditable rule shape rather than arbitrary executable
policy code.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from sqlalchemy import text

from app.core.config import get_settings
from app.data.database import engine
from app.observability import record_event
from app.tenancy import TenantContext


def validate_policy_rule(rule: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(rule, dict):
        raise ValueError("ABAC rule must be an object")
    role = str(rule.get("role", "")).strip()
    agent = str(rule.get("agent", "")).strip()
    if not role or not agent or len(role) > 64 or len(agent) > 128:
        raise ValueError("ABAC rule requires a bounded role and agent")
    allow = rule.get("allow", [])
    deny = rule.get("deny", [])
    conditions = rule.get("conditions", {})
    if not isinstance(allow, list) or not isinstance(deny, list) or not isinstance(conditions, dict):
        raise ValueError("ABAC allow, deny and conditions must be JSON arrays/objects")
    allowed_condition_keys = {"scope_types", "tenant_ids", "actions"}
    if set(conditions) - allowed_condition_keys:
        raise ValueError("Unsupported ABAC condition")
    normalized = {
        "role": role,
        "agent": agent,
        "allow": sorted({str(item) for item in allow if str(item).strip()}),
        "deny": sorted({str(item) for item in deny if str(item).strip()}),
        "conditions": {
            key: sorted({str(item) for item in value})
            for key, value in conditions.items()
            if isinstance(value, list)
        },
    }
    return normalized


def _admin(context: TenantContext) -> None:
    if not (context.is_platform_admin or context.role in {"owner", "admin"}):
        raise PermissionError("ABAC policy management requires administrator role")


def _require_database_policy_management() -> None:
    if not get_settings().abac_database_enabled:
        raise PermissionError("Tenant-managed ABAC policies are currently disabled")


def list_policy_rules(context: TenantContext) -> list[dict[str, Any]]:
    _admin(context)
    _require_database_policy_management()
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": context.tenant_id})
        conn.execute(text("SELECT set_config('app.user_id', :user, true)"), {"user": context.user_id})
        rows = conn.execute(text("""
            SELECT policy_id, policy_version, status, rule_json, created_by, updated_by,
                   created_at, updated_at
            FROM governance.abac_policy_rules
            WHERE tenant_id=:tenant ORDER BY policy_id
        """), {"tenant": context.tenant_id}).mappings().all()
    return [dict(row) for row in rows]


def save_policy_rule(context: TenantContext, policy_id: str, rule: dict[str, Any], *, status: str = "active") -> dict[str, Any]:
    _admin(context)
    _require_database_policy_management()
    if not policy_id or len(policy_id) > 128:
        raise ValueError("policy_id is required")
    if status not in {"active", "disabled"}:
        raise ValueError("unsupported ABAC policy status")
    normalized = validate_policy_rule(rule)
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": context.tenant_id})
        conn.execute(text("SELECT set_config('app.user_id', :user, true)"), {"user": context.user_id})
        row = conn.execute(text("""
            INSERT INTO governance.abac_policy_rules
              (tenant_id,policy_id,policy_version,status,rule_json,created_by,updated_by)
            VALUES (:tenant,:id,1,:status,CAST(:rule AS jsonb),:actor,:actor)
            ON CONFLICT (tenant_id,policy_id) DO UPDATE SET
              policy_version=governance.abac_policy_rules.policy_version+1,
              status=EXCLUDED.status, rule_json=EXCLUDED.rule_json,
              updated_by=EXCLUDED.updated_by, updated_at=now()
            RETURNING policy_id,policy_version,status,rule_json,created_by,updated_by,created_at,updated_at
        """), {"tenant": context.tenant_id, "id": policy_id, "status": status,
                "rule": json.dumps(normalized, ensure_ascii=False), "actor": context.user_id}).mappings().one()
    record_event("abac.policy_changed", {"tenant_id": context.tenant_id, "user_id": context.user_id, "policy_id": policy_id, "status": status})
    return dict(row)


def delete_policy_rule(context: TenantContext, policy_id: str) -> None:
    _admin(context)
    _require_database_policy_management()
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": context.tenant_id})
        conn.execute(text("SELECT set_config('app.user_id', :user, true)"), {"user": context.user_id})
        conn.execute(text("DELETE FROM governance.abac_policy_rules WHERE tenant_id=:tenant AND policy_id=:id"), {"tenant": context.tenant_id, "id": policy_id})
    record_event("abac.policy_deleted", {"tenant_id": context.tenant_id, "user_id": context.user_id, "policy_id": policy_id})


def database_rules(tenant_id: str) -> dict[str, dict[str, dict[str, Any]]] | None:
    """Return active tenant rules, or ``None`` when DB policy is not present.

    A database error is not converted into an allow.  Production callers get
    an empty policy (default deny); development can still use the explicitly
    configured bootstrap file when the table has no rows.
    """

    if not get_settings().abac_database_enabled:
        return None
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": tenant_id})
            rows = conn.execute(text("""
                SELECT rule_json FROM governance.abac_policy_rules
                WHERE tenant_id=:tenant AND status='active'
            """), {"tenant": tenant_id}).mappings().all()
    except Exception:
        if get_settings().environment.lower() in {"production", "prod"}:
            return {}
        return None
    if not rows:
        return None
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        rule = validate_policy_rule(dict(row["rule_json"]))
        result.setdefault(tenant_id, {}).setdefault(rule["role"], {}).setdefault(rule["agent"], {"allow": [], "deny": [], "conditions": {}})
        target = result[tenant_id][rule["role"]][rule["agent"]]
        target["allow"] = sorted(set(target["allow"]) | set(rule["allow"]))
        target["deny"] = sorted(set(target["deny"]) | set(rule["deny"]))
        target["conditions"] = rule["conditions"]
    return result


def new_policy_id() -> str:
    return f"policy_{uuid4().hex}"
