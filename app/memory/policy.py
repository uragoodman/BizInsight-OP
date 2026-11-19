"""Server-side Memory Policy for explicit, typed preference mutations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.tenancy import TenantContext


@dataclass(frozen=True)
class MemoryPolicyDecision:
    action: str
    tenant_id: str
    user_id: str
    subject: str
    memory_type: str | None
    value_hash: str | None
    accepted: bool
    user_confirmed: bool
    policy_version: str = "memory-policy-v1"
    reason: str = "accepted"


class MemoryPolicy:
    """Allow only an explicit user request matching a small typed schema."""

    PRESENTATION = {
        "response_style": ("style", {"concise", "balanced", "detailed"}),
        "report_style": ("format", {"table", "bullets", "narrative"}),
        "response_language": ("language", {"zh-CN", "zh-TW", "en-US"}),
    }
    ANALYSIS_KEYS = {"analysis_method", "analysis_steps"}
    ANALYSIS_FIELDS = {"metric", "dimension", "operation", "date"}
    SCOPE_KEYS = {"default_period", "default_date"}

    @classmethod
    def evaluate_upsert(
        cls, context: TenantContext, *, memory_key: str, memory_type: str,
        value: dict[str, Any], explicit_user_request: bool, user_confirmed: bool,
    ) -> MemoryPolicyDecision:
        reason = cls._validate(memory_key, memory_type, value)
        accepted = reason is None and explicit_user_request and user_confirmed
        if reason is None and not explicit_user_request:
            reason = "explicit_save_request_required"
        if reason is None and not user_confirmed:
            reason = "explicit_user_confirmation_required"
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return MemoryPolicyDecision(
            action="upsert", tenant_id=context.tenant_id, user_id=context.user_id,
            subject=memory_key, memory_type=memory_type,
            value_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            accepted=accepted, user_confirmed=bool(user_confirmed),
            reason=reason or "accepted",
        )

    @staticmethod
    def evaluate_delete(
        context: TenantContext, *, registry_id: str, user_confirmed: bool,
    ) -> MemoryPolicyDecision:
        return MemoryPolicyDecision(
            action="delete", tenant_id=context.tenant_id, user_id=context.user_id,
            subject=registry_id, memory_type=None, value_hash=None,
            accepted=bool(user_confirmed), user_confirmed=bool(user_confirmed),
            reason="accepted" if user_confirmed else "explicit_user_confirmation_required",
        )

    @classmethod
    def _validate(cls, memory_key: str, memory_type: str, value: Any) -> str | None:
        if not isinstance(value, dict) or not value:
            return "typed_preference_value_required"
        if memory_type == "presentation":
            schema = cls.PRESENTATION.get(memory_key)
            if schema is None:
                return "unsupported_preference_key"
            field, allowed = schema
            candidate = value.get(field)
            if (
                set(value) != {field}
                or not isinstance(candidate, str)
                or candidate not in allowed
            ):
                return "preference_value_schema_invalid"
            return None
        if memory_type == "analysis_method":
            if memory_key not in cls.ANALYSIS_KEYS:
                return "unsupported_preference_key"
            if not set(value).issubset(cls.ANALYSIS_FIELDS):
                return "preference_value_schema_invalid"
            if not value or any(not cls._bounded_scalar(item) for item in value.values()):
                return "preference_value_schema_invalid"
            return None
        if memory_type == "scope_default":
            if memory_key not in cls.SCOPE_KEYS:
                return "unsupported_preference_key"
            allowed = {"start_date", "end_date"} if memory_key == "default_period" else {"date"}
            if set(value) != allowed or any(not cls._valid_date(item) for item in value.values()):
                return "preference_value_schema_invalid"
            if memory_key == "default_period" and cls._date_key(value["start_date"]) > cls._date_key(value["end_date"]):
                return "preference_value_schema_invalid"
            return None
        return "unsupported_memory_type"

    @staticmethod
    def _bounded_scalar(value: Any) -> bool:
        if isinstance(value, str):
            return 0 < len(value.strip()) <= 128
        if isinstance(value, list):
            return 0 < len(value) <= 8 and all(
                isinstance(item, str) and 0 < len(item.strip()) <= 128 for item in value
            )
        return False

    @staticmethod
    def _valid_date(value: Any) -> bool:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", value):
            return False
        try:
            if len(value) == 7:
                date.fromisoformat(f"{value}-01")
            else:
                date.fromisoformat(value)
        except ValueError:
            return False
        return True

    @staticmethod
    def _date_key(value: str) -> date:
        return date.fromisoformat(f"{value}-01" if len(value) == 7 else value)
