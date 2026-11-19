"""Minimal tool governance and secret-safe logging boundary."""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any

from app.mcp.tools import get_supported_tools
from app.core.config import get_settings

# Governed HTTP capabilities participate in the same ABAC allow/deny policy
# as MCP tools, but are intentionally not advertised to the LLM as MCP tools.
GOVERNED_HTTP_CAPABILITIES = ("estimate_scoped_randomized_experiment",)

_SUSPICIOUS_PROMPT_PATTERNS = ("ignore previous", "system prompt", "developer message", "忽略之前", "系统提示")
_SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|authorization|password|postgres(?:ql)?://)\s*[:=]\s*[^\s,]+")


def ensure_safe_prompt(value: str) -> None:
    lowered = value.lower()
    if any(pattern in lowered for pattern in _SUSPICIOUS_PROMPT_PATTERNS):
        raise ValueError("Potential prompt injection was blocked.")


def ensure_safe_untrusted_content(value: Any, source: str) -> None:
    """Apply the same instruction-injection policy to retrieved/tool content."""
    serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    try:
        ensure_safe_prompt(serialized)
    except ValueError as exc:
        raise ValueError(f"Potential prompt injection in {source} was blocked.") from exc


class ToolPermissionPolicy:
    """Default-deny tool authorization with an optional tenant/role policy file."""

    def __init__(self, policy_path: str | None = None) -> None:
        self.policy_path = Path(policy_path or get_settings().tool_permission_policy_path)

    def _rules(self, tenant_id: str | None = None) -> dict[str, Any]:
        if tenant_id:
            # Database policy rules are authoritative when a tenant has
            # published them.  The import is lazy to keep the static policy
            # and unit-test path dependency-light.
            try:
                from app.security.abac import database_rules

                persisted = database_rules(tenant_id)
                if persisted is not None:
                    return persisted
            except Exception:
                # Production database-policy failures return an empty policy
                # from database_rules; never turn an exception into an allow.
                if get_settings().environment.lower() in {"production", "prod"}:
                    return {}
        if self.policy_path.exists():
            return json.loads(self.policy_path.read_text(encoding="utf-8"))
        tools = sorted(get_supported_tools())
        # Local development keeps the historical public analyst path, while the
        # authenticated platform context used by Phase 7 has its own explicit
        # administrator grant. Seller scope never reaches this fallback.
        return {"public": {"analyst": {"data_query": tools}}, "platform": {"admin": {"data_query": tools}}}

    def allows(
        self,
        tenant_id: str,
        role: str,
        agent_name: str,
        tool_name: str,
        *,
        scope_type: str | None = None,
    ) -> bool:
        # A platform-scoped request still carries the caller's membership
        # tenant_id.  Permission policy is keyed by the data boundary, so use
        # the platform policy for that scope instead of treating the tenant as
        # an unrelated public tenant.  Seller-scoped requests retain their
        # tenant-specific policy and never inherit the platform grant.
        policy_tenant = "platform" if scope_type == "platform" else tenant_id
        # Seller-scoped requests use a checked-in RBAC baseline for ordinary
        # data-query tools. Tenant-specific rules remain available only as
        # deployment-managed static entries; the database policy editor is
        # disabled by default. Governed capabilities never inherit the seller
        # baseline and require an explicit tenant rule.
        rules = self._rules(policy_tenant)
        if scope_type == "seller":
            tenant_rules = rules.get(policy_tenant, {})
        else:
            tenant_rules = rules.get(policy_tenant, rules.get("public", {}))
        role_rules = tenant_rules.get(role, {})
        baseline = rules.get("seller_baseline", {}) if scope_type == "seller" else {}
        baseline_roles = {str(item) for item in baseline.get("roles", [])}
        baseline_tools = {
            str(item)
            for item in baseline.get("agents", {}).get(agent_name, [])
        }
        baseline_allowed = (
            role in baseline_roles
            and tool_name in baseline_tools
            and tool_name not in GOVERNED_HTTP_CAPABILITIES
        )
        if baseline_allowed and not role_rules:
            return True

        # A legacy policy file may predate Phase 7 and contain only the public
        # analyst grant. Platform admin is the trusted replacement for that
        # local platform scope, never a seller fallback.
        if policy_tenant == "platform" and role == "admin" and not role_rules:
            role_rules = rules.get("public", {}).get("analyst", {})
        raw_rule = role_rules.get(agent_name, [])
        # A governed capability in seller scope is never inherited from the
        # public/platform fallback: it must appear in that tenant's own rule.
        if (
            scope_type == "seller"
            and tool_name in GOVERNED_HTTP_CAPABILITIES
            and not role_rules
        ):
            return False
        if isinstance(raw_rule, dict):
            allowed = {str(item) for item in raw_rule.get("allow", [])}
            denied = {str(item) for item in raw_rule.get("deny", [])}
            conditions = raw_rule.get("conditions") or {}
            if not isinstance(conditions, dict):
                return False
            condition_matches = True
            if tool_name in allowed:
                allowed_scopes = conditions.get("scope_types")
                if allowed_scopes is not None and (scope_type is None or scope_type not in {str(item) for item in allowed_scopes}):
                    condition_matches = False
                allowed_tenants = conditions.get("tenant_ids")
                if allowed_tenants is not None and tenant_id not in {str(item) for item in allowed_tenants}:
                    condition_matches = False
                allowed_actions = conditions.get("actions")
                if allowed_actions is not None and "execute" not in {str(item) for item in allowed_actions}:
                    condition_matches = False
            # Explicit deny always wins over an explicit/role allow.
            return (tool_name in allowed and condition_matches or baseline_allowed) and tool_name not in denied
        return tool_name in {str(item) for item in raw_rule} or baseline_allowed


def ensure_tool_allowed(
    agent_name: str,
    tool_name: str,
    *,
    tenant_id: str | None = None,
    role: str | None = None,
    scope_type: str | None = None,
) -> None:
    settings = get_settings()
    if not ToolPermissionPolicy().allows(
        tenant_id or settings.tenant_id,
        role or settings.user_role,
        agent_name,
        tool_name,
        scope_type=scope_type,
    ):
        raise PermissionError(f"Tool {tool_name!r} is not authorized for {agent_name!r}.")


def is_trusted_knowledge_source(source_path: str) -> bool:
    """Only approved files under the configured knowledge root may enter production evidence."""
    root = Path(get_settings().rag_knowledge_root).resolve()
    try:
        Path(source_path).resolve().relative_to(root)
        return True
    except ValueError:
        return False


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if key.lower() in {"authorization", "password", "api_key", "deepseek_api_key", "postgres_url"} else redact_sensitive(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return _SECRET_PATTERN.sub("[REDACTED]", value) if isinstance(value, str) else value
