"""Allowlisted, privacy-safe audit records for CapabilityRouter shadow runs."""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from app.context.semantic_catalog_v2 import (
    DEFAULT_SEMANTIC_CATALOG,
    ROUTE_DECISION_STATUSES,
    RouteOutcome,
    SemanticCatalog,
)
from app.context.semantic_contract_v2 import SemanticContractV21
from app.llm.triage_prompt import ROUTING_TABLE

ROUTE_SHADOW_AUDIT_SCHEMA = "capability-router-shadow-v2"
ROUTE_SHADOW_SAMPLING_VERSION = "capability-router-sample-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORD_KEYS = frozenset({
    "schema_version",
    "sampling_version",
    "trace_id_hash",
    "turn_id_hash",
    "contract_version",
    "catalog_snapshot_hash",
    "semantic_status",
    "route_status",
    "route_id",
    "capability_ids",
    "operation_ids",
    "clause_ids",
    "rejection_reasons",
    "legacy_task_type",
    "legacy_next_agent",
    "legacy_unresolved",
    "sampled_at",
    "sample_rate",
    "sample_bucket",
})
_SEMANTIC_STATUSES = frozenset({"CONFIRMED", "MISSING", "AMBIGUOUS", "CONFLICT", "UNSUPPORTED"})
_STATIC_REASONS = frozenset({
    "hypothetical_not_executable",
    "request_mode_ask",
    "semantic_status_missing",
    "no_executable_operation",
    "multiple_capabilities",
})
_CAPABILITY_REASONS = frozenset({
    "no_matching_operation",
    "missing_required_arguments",
    "explicit_method_missing_required_arguments",
    "insufficient_basis",
    "vetoed_by_negative_operation",
})
_LEGACY_TASK_TYPES = frozenset(ROUTING_TABLE) | {"clarification"}
_LEGACY_NEXT_AGENTS = frozenset(ROUTING_TABLE.values()) | {"context_clarification"}


def _sample_rate(settings: Any) -> float:
    try:
        return min(1.0, max(0.0, float(getattr(settings, "semantic_interpreter_shadow_sample_rate", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _opaque_hash(value: Any, *, salt: str, purpose: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    material = f"{ROUTE_SHADOW_SAMPLING_VERSION}:{purpose}:{text}".encode("utf-8")
    return hmac.new(salt.encode("utf-8"), material, hashlib.sha256).hexdigest()


def _bucket(trace_id: str, *, salt: str) -> int:
    material = f"{ROUTE_SHADOW_SAMPLING_VERSION}:bucket:{trace_id}".encode("utf-8")
    digest = hmac.new(salt.encode("utf-8"), material, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _safe_reference_ids(values: Any, *, salt: str, purpose: str) -> list[str] | None:
    if not isinstance(values, (tuple, list)):
        return None
    encoded = [_opaque_hash(value, salt=salt, purpose=purpose) for value in values]
    if any(value is None for value in encoded) or len(encoded) != len(set(encoded)):
        return None
    return sorted(value for value in encoded if value is not None)


def _allowed_reason(reason: Any, catalog: SemanticCatalog) -> bool:
    if not isinstance(reason, str) or not reason or len(reason) > 160:
        return False
    if reason in _STATIC_REASONS:
        return True
    capability_id, separator, reason_code = reason.partition(":")
    if not separator or capability_id not in {item.capability_id for item in catalog.capabilities}:
        return False
    if reason_code in _CAPABILITY_REASONS:
        return True
    nested, nested_separator, value = reason_code.partition(":")
    if not nested_separator:
        return False
    if nested == "missing_required_role":
        return value in set(catalog.role_names)
    if nested == "missing_required_relation":
        return value in set(catalog.relation_predicates)
    return False


def _validate_route_shadow_record(
    record: Any,
    *,
    catalog: SemanticCatalog = DEFAULT_SEMANTIC_CATALOG,
) -> bool:
    """Reject records with unexpected keys, values, or non-catalog references."""
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        return False
    if record.get("schema_version") != ROUTE_SHADOW_AUDIT_SCHEMA:
        return False
    if record.get("sampling_version") != ROUTE_SHADOW_SAMPLING_VERSION:
        return False
    if not _SHA256.fullmatch(str(record.get("trace_id_hash") or "")):
        return False
    if not _SHA256.fullmatch(str(record.get("turn_id_hash") or "")):
        return False
    if record.get("contract_version") != "semantic-contract-v2.1":
        return False
    if record.get("catalog_snapshot_hash") != catalog.snapshot_hash():
        return False
    if record.get("semantic_status") not in _SEMANTIC_STATUSES:
        return False
    if record.get("route_status") not in ROUTE_DECISION_STATUSES:
        return False
    route_id = record.get("route_id")
    route_by_id = {item.route_id: item for item in catalog.routes}
    if route_id is not None and route_id not in route_by_id:
        return False
    capability_ids = record.get("capability_ids")
    capability_by_id = {item.capability_id: item for item in catalog.capabilities}
    if (
        not isinstance(capability_ids, list)
        or len(capability_ids) != len(set(capability_ids))
        or any(item not in capability_by_id for item in capability_ids)
    ):
        return False
    if record.get("route_status") == "SELECTED":
        if route_id is None:
            return False
        route_capability_ids = {
            item.capability_id for item in route_by_id[route_id].capability_ids
        }
        if not capability_ids or not set(capability_ids) <= route_capability_ids:
            return False
    if not all(
        isinstance(record.get(key), list)
        and len(record[key]) == len(set(record[key]))
        and all(_SHA256.fullmatch(str(value)) for value in record[key])
        for key in ("operation_ids", "clause_ids")
    ):
        return False
    reasons = record.get("rejection_reasons")
    if (
        not isinstance(reasons, list)
        or len(reasons) != len(set(reasons))
        or any(not _allowed_reason(reason, catalog) for reason in reasons)
    ):
        return False
    if record.get("legacy_task_type") not in _LEGACY_TASK_TYPES:
        return False
    if record.get("legacy_next_agent") not in _LEGACY_NEXT_AGENTS:
        return False
    if not isinstance(record.get("legacy_unresolved"), bool):
        return False
    try:
        sampled_at = datetime.fromisoformat(str(record.get("sampled_at") or ""))
    except ValueError:
        return False
    if sampled_at.tzinfo is None:
        return False
    sample_rate = record.get("sample_rate")
    sample_bucket = record.get("sample_bucket")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, float))
        or not 0.0 < float(sample_rate) <= 1.0
        or isinstance(sample_bucket, bool)
        or not isinstance(sample_bucket, int)
        or not 0 <= sample_bucket < (1 << 64)
        or sample_bucket >= float(sample_rate) * (1 << 64)
    ):
        return False
    return True


def validate_route_shadow_record(
    record: Any,
    *,
    catalog: SemanticCatalog = DEFAULT_SEMANTIC_CATALOG,
) -> bool:
    """Return false for malformed records rather than raising on bad shapes."""
    try:
        return _validate_route_shadow_record(record, catalog=catalog)
    except (TypeError, ValueError, AttributeError):
        return False


def build_route_shadow_record(
    state: Mapping[str, Any],
    contract: SemanticContractV21,
    outcome: RouteOutcome,
    *,
    settings: Any,
    catalog: SemanticCatalog = DEFAULT_SEMANTIC_CATALOG,
    sampled_at: datetime | None = None,
) -> dict[str, Any] | None:
    """Build a sampled record containing only catalog values and opaque IDs."""
    trace_id = str(state.get("trace_id") or "")
    salt = str(getattr(settings, "semantic_shadow_salt", "") or "")
    sample_rate = _sample_rate(settings)
    if not trace_id or not salt or sample_rate <= 0.0:
        return None
    sample_bucket = _bucket(trace_id, salt=salt)
    if sample_bucket >= sample_rate * (1 << 64):
        return None
    turn_id = (
        state.get("chat_turn_id")
        or state.get("persistence_turn_id")
        or state.get("turn_id")
        or contract.turn_id
    )
    trace_id_hash = _opaque_hash(trace_id, salt=salt, purpose="trace")
    turn_id_hash = _opaque_hash(turn_id, salt=salt, purpose="turn")
    if trace_id_hash is None or turn_id_hash is None:
        return None
    operation_ids = _safe_reference_ids(
        outcome.decision_basis.operation_ids, salt=salt, purpose="operation",
    )
    clause_ids = _safe_reference_ids(
        outcome.decision_basis.clause_ids, salt=salt, purpose="clause",
    )
    if operation_ids is None or clause_ids is None:
        return None
    legacy_task_type = state.get("task_type")
    legacy_next_agent = state.get("next_agent")
    legacy_unresolved = state.get("triage_unresolved")
    if (
        legacy_task_type not in _LEGACY_TASK_TYPES
        or legacy_next_agent not in _LEGACY_NEXT_AGENTS
        or not isinstance(legacy_unresolved, bool)
    ):
        return None
    capability_ids = sorted({
        item.capability_id
        for item in (*outcome.capability_ids, *outcome.decision_basis.capability_ids)
    })
    sampled = sampled_at or datetime.now(timezone.utc)
    if sampled.tzinfo is None:
        sampled = sampled.replace(tzinfo=timezone.utc)
    record = {
        "schema_version": ROUTE_SHADOW_AUDIT_SCHEMA,
        "sampling_version": ROUTE_SHADOW_SAMPLING_VERSION,
        "trace_id_hash": trace_id_hash,
        "turn_id_hash": turn_id_hash,
        "contract_version": contract.contract_version,
        "catalog_snapshot_hash": contract.catalog_snapshot_hash,
        "semantic_status": contract.semantic_status,
        "route_status": outcome.status,
        "route_id": outcome.route_id,
        "capability_ids": capability_ids,
        "operation_ids": operation_ids,
        "clause_ids": clause_ids,
        "rejection_reasons": list(outcome.decision_basis.rejection_reasons),
        "legacy_task_type": legacy_task_type,
        "legacy_next_agent": legacy_next_agent,
        "legacy_unresolved": legacy_unresolved,
        "sampled_at": sampled.astimezone(timezone.utc).isoformat(),
        "sample_rate": sample_rate,
        "sample_bucket": sample_bucket,
    }
    if not validate_route_shadow_record(record, catalog=catalog):
        raise ValueError("route shadow audit record failed allowlist validation")
    return record
