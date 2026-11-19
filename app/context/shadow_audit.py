"""Privacy-bounded sampling and persistence for P4 semantic shadow audits.

The audit runs after the workflow has completed. It only samples successfully
executed, contract-anchored requests, never calls a model, and stores a strict
allowlist of categorical evidence in a separate local quality database.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


AUDIT_SCHEMA_VERSION = "semantic-shadow-audit-v1"
AUDIT_SAMPLING_VERSION = "hmac-sha256-trace-bucket-v1"
_SAFE_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_SAFE_SPAN_ID = re.compile(r"^m[1-9][0-9]{0,4}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ROLES = {
    "grouping", "filter", "reference", "comparison", "unclear",
    "conflict", "unresolved",
}
_SAFE_RISKS = {"high", "conflict", "elevated", "open", "low", "unknown"}
_SAFE_OBSERVATION_STATUSES = {
    "success", "skipped", "abstained", "timeout", "provider_error",
    "invalid_output", "blocked_by_privacy", "invalid_configuration",
}
_SAFE_OPERATIONS = {
    "ordinary_metric_query", "dimensional_statistical_contribution",
    "registered_numeric_decomposition", "business_causal_investigation",
    "unknown",
}
_ALLOWED_AUDIT_KEYS = {
    "schema_version", "sampling_version", "audit_id", "trace_id_hash",
    "turn_id_hash", "execution_id_hash", "rules_version", "prompt_version",
    "verifier_mode", "verifier_status", "model", "risk_tier", "span_ids",
    "span_dimensions", "span_confidence", "rule_roles", "verifier_roles",
    "disagreement_span_ids", "operation", "terminal", "clarification_reason",
    "contract_hash", "authorization_outcome", "support_outcome", "sampled_at",
    "sample_rate", "sample_bucket", "salt_fingerprint", "actual_execution",
    "human_review_status", "review_target", "reviewed_label", "reviewed_span_labels",
    "contract_review_status", "wrong_contract", "contract_review_source",
    "execution_review_status", "wrong_execution", "execution_review_source",
}


def _default_audit_db_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "quality" / "semantic_shadow_audit.sqlite3"


def shadow_sample_rate(settings: Any | None = None) -> float:
    config = settings or _settings()
    try:
        return max(0.0, min(100.0, float(getattr(config, "semantic_shadow_sample_rate", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def shadow_enabled(settings: Any | None = None) -> bool:
    config = settings or _settings()
    return bool(getattr(config, "semantic_shadow_enabled", False)) and bool(
        str(getattr(config, "semantic_shadow_salt", "") or "")
    ) and shadow_sample_rate(config) > 0.0


def sampling_bucket(trace_id: str, *, settings: Any | None = None) -> int | None:
    """Return a stable 64-bit salted bucket without exposing the trace ID."""
    config = settings or _settings()
    salt = str(getattr(config, "semantic_shadow_salt", "") or "")
    if not trace_id or not salt:
        return None
    digest = hmac.new(
        salt.encode("utf-8"),
        f"{AUDIT_SAMPLING_VERSION}:{trace_id}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def should_sample(
    trace_id: str,
    *,
    settings: Any | None = None,
    eligible: bool = False,
) -> bool:
    """Return a stable sample decision only for an explicitly eligible turn."""
    if eligible is not True:
        return False
    config = settings or _settings()
    if not trace_id or not shadow_enabled(config):
        return False
    bucket = sampling_bucket(trace_id, settings=config)
    if bucket is None:
        return False
    threshold = (shadow_sample_rate(config) / 100.0) * (1 << 64)
    return bucket < threshold


def _hmac_id(raw_value: Any, *, salt: str, purpose: str) -> str | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    return hmac.new(
        salt.encode("utf-8"),
        f"semantic-shadow-id-v1:{purpose}:{value}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _safe_code(value: Any, *, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    return text if _SAFE_ID.fullmatch(text) else default


def _safe_version(value: Any) -> str:
    text = str(value or "").strip()
    return text if _SAFE_VERSION.fullmatch(text) else "unknown"


def _safe_role_map(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for span_id, role in value.items():
        key, label = str(span_id), str(role)
        if _SAFE_SPAN_ID.fullmatch(key) and label in _SAFE_ROLES:
            result[key] = label
    return dict(sorted(result.items()))


def _successful_metric_execution_pairs(
    state: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]] | None:
    """Bind every current query result to its exact successful metric call.

    ``tool_calls`` contains history across a graph turn, while ``query_results``
    contains the current data-query batch. Only IDs present in both are eligible;
    a successful unrelated call cannot make a failed or partial batch auditable.
    """
    query_results = state.get("query_results")
    tool_calls = state.get("tool_calls")
    if not isinstance(query_results, list) or not query_results:
        return None
    if not isinstance(tool_calls, list):
        return None

    results_by_id: dict[str, Mapping[str, Any]] = {}
    for item in query_results:
        if not isinstance(item, Mapping):
            return None
        request_id = str(item.get("request_id") or "")
        result = item.get("result")
        if not request_id or request_id in results_by_id or not isinstance(result, Mapping):
            return None
        if result.get("status") != "ok":
            return None
        results_by_id[request_id] = item

    metric_calls_by_id: dict[str, Mapping[str, Any]] = {}
    for call in tool_calls:
        if not isinstance(call, Mapping) or not call.get("planned_metric_id"):
            continue
        request_id = str(call.get("request_id") or "")
        if not request_id or request_id in metric_calls_by_id:
            return None
        metric_calls_by_id[request_id] = call

    def batch_key(request_id: str) -> str:
        # data_query uses ``<trace>:<call-index>:<batch-index>`` for batch
        # items and ``<trace>:<call-index>`` for single calls. The trace ID
        # itself is generated by the server and does not contain colons.
        parts = request_id.split(":")
        if len(parts) >= 3 and parts[-1].isdigit():
            return ":".join(parts[:-1])
        return request_id

    result_batch_keys = {batch_key(request_id) for request_id in results_by_id}
    current_metric_call_ids = {
        request_id for request_id in metric_calls_by_id
        if batch_key(request_id) in result_batch_keys
    }
    if current_metric_call_ids != set(results_by_id):
        return None

    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for request_id, result in results_by_id.items():
        call = metric_calls_by_id.get(request_id)
        if call is None or call.get("status") != "ok":
            return None
        snapshot = call.get("authorization_snapshot")
        if snapshot is not None:
            if not isinstance(snapshot, Mapping):
                return None
            snapshot_request_id = str(snapshot.get("request_id") or "")
            if snapshot_request_id and snapshot_request_id != request_id:
                return None
            if snapshot.get("decision") != "allowed":
                return None
        pairs.append((call, result))

    batch = state.get("metric_batch_execution")
    if isinstance(batch, Mapping) and batch.get("status") not in {None, "all_success"}:
        return None
    return pairs


def _successful_execution(state: Mapping[str, Any]) -> bool:
    for key in ("_status", "task_status"):
        status = state.get(key)
        if key in state and (not isinstance(status, str) or status != "completed"):
            return False
    errors = state.get("errors")
    if not isinstance(errors, list) or errors:
        return False
    if state.get("__interrupt__") or state.get("awaiting_user_confirmation"):
        return False
    compilation = state.get("context_compilation")
    if not isinstance(compilation, Mapping) or compilation.get("needs_clarification"):
        return False
    compiled_contract = compilation.get("request_contract")
    if not isinstance(compiled_contract, Mapping):
        return False
    try:
        from app.context.request_contract import validate_compiled_request_contract

        valid, _reason = validate_compiled_request_contract(
            compilation,
            state_contract_hash=state.get("request_contract_hash"),
            require_state_anchor=True,
        )
    except Exception:  # noqa: BLE001 - an audit eligibility failure is fail closed
        return False
    if not valid:
        return False
    return bool(_successful_metric_execution_pairs(state))


def _known_dimension_ids() -> set[str]:
    """Return canonical dimension IDs from the existing metric registry."""
    try:
        from app.metrics.registry import get_registry

        registry = get_registry()
        known = {item.dimension_id for item in registry.dimensions()}
        for metric in registry.list():
            for implementation in metric.implementations.values():
                known.update(str(value) for value in implementation.dimensions)
        return {value for value in known if _SAFE_ID.fullmatch(value)}
    except Exception:  # noqa: BLE001 - unknown IDs are redacted to "other"
        return set()


def _span_metadata(
    compilation: Mapping[str, Any], *, known_dimensions: set[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    dimensions: dict[str, str] = {}
    confidence: dict[str, str] = {}
    roles: dict[str, str] = {}
    index = 0
    assignments = compilation.get("dimension_roles") or []
    for item in assignments:
        if not isinstance(item, Mapping):
            continue
        try:
            if int(item.get("end") or 0) <= int(item.get("start") or 0):
                continue
        except (TypeError, ValueError):
            continue
        index += 1
        span_id = f"m{index}"
        raw_dimension = str(item.get("dimension") or "").strip().lower()
        dimensions[span_id] = raw_dimension if raw_dimension in known_dimensions else "other"
        raw_confidence = str(item.get("confidence_class") or "unknown").lower()
        confidence[span_id] = raw_confidence if raw_confidence in {"closed", "open", "unresolved"} else "unknown"
        raw_role = str(item.get("role") or "unresolved")
        roles[span_id] = raw_role if raw_role in _SAFE_ROLES else "unresolved"
    return dimensions, confidence, roles


def build_shadow_audit_record(
    state: Mapping[str, Any],
    *,
    settings: Any | None = None,
    sampled_at: datetime | None = None,
) -> dict[str, Any] | None:
    """Build one allowlisted record for a completed, auditable execution."""
    config = settings or _settings()
    if not shadow_enabled(config):
        return None
    execution_pairs = _successful_metric_execution_pairs(state)
    if not execution_pairs:
        return None
    if not _successful_execution(state):
        return None
    trace_id = str(state.get("trace_id") or "")
    if not should_sample(trace_id, settings=config, eligible=True):
        return None

    compilation = state["context_compilation"]
    contract = compilation["request_contract"]
    contract_hash = str(contract.get("contract_hash") or "")
    if not _SHA256.fullmatch(contract_hash):
        return None
    observation = state.get("semantic_verifier_observation")
    observation = observation if isinstance(observation, Mapping) else {}
    span_dimensions, span_confidence, rule_roles = _span_metadata(
        compilation, known_dimensions=_known_dimension_ids(),
    )
    known_span_ids = set(span_dimensions)
    observed_rules = _safe_role_map(observation.get("rule_roles"))
    if observed_rules:
        rule_roles.update({key: value for key, value in observed_rules.items() if key in known_span_ids})
    verifier_roles = {
        key: value for key, value in _safe_role_map(observation.get("verifier_roles")).items()
        if key in known_span_ids
    }
    span_ids = sorted(known_span_ids)
    disagreement_span_ids = sorted(
        span_id for span_id in set(rule_roles) & set(verifier_roles)
        if rule_roles[span_id] != verifier_roles[span_id]
    )

    raw_risk = str(observation.get("risk_tier") or "unknown")
    risk_tier = raw_risk if raw_risk in _SAFE_RISKS else "unknown"
    raw_mode = str(observation.get("mode") or getattr(config, "semantic_verifier_mode", "off"))
    verifier_mode = raw_mode if raw_mode in {"off", "shadow", "open_only"} else "off"
    contract_operation = contract.get("operation_intent")
    if isinstance(contract_operation, Mapping):
        contract_operation = contract_operation.get("kind")
    compiled_operation = compilation.get("operation_intent")
    if isinstance(compiled_operation, Mapping):
        compiled_operation = compiled_operation.get("kind")
    raw_operation = str(contract_operation or compiled_operation or "unknown")
    operation = raw_operation if raw_operation in _SAFE_OPERATIONS else "unknown"
    sampled = sampled_at or datetime.now(timezone.utc)
    if sampled.tzinfo is None:
        sampled = sampled.replace(tzinfo=timezone.utc)
    bucket = sampling_bucket(trace_id, settings=config)
    salt = str(getattr(config, "semantic_shadow_salt", "") or "")
    if bucket is None or not salt:
        return None

    snapshots = [
        call.get("authorization_snapshot")
        for call, _result in execution_pairs
        if isinstance(call.get("authorization_snapshot"), Mapping)
    ]
    if snapshots and len(snapshots) == len(execution_pairs):
        authorization = {"status": "passed", "reason_code": "authorization_checks_passed"}
    else:
        authorization = {"status": "unknown", "reason_code": "authorization_snapshot_incomplete"}
    # Every result in execution_pairs is a request-ID-bound successful query.
    # This is execution evidence, not a copied verifier preflight claim.
    support = {"status": "passed", "reason_code": "query_results_all_ok"}

    turn_id = state.get("chat_turn_id") or state.get("persistence_turn_id") or state.get("turn_id")
    execution_id = (
        state.get("lifecycle_execution_id")
        or state.get("execution_id")
        or (state.get("authorization_boundary") or {}).get("execution_id")
    )
    model = str(observation.get("model") or getattr(config, "semantic_verifier_model", "") or "")[:120]
    if model and not re.fullmatch(r"[A-Za-z0-9._:/-]{1,120}", model):
        model = "unknown"
    record = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "sampling_version": AUDIT_SAMPLING_VERSION,
        "audit_id": _hmac_id(trace_id, salt=salt, purpose="audit"),
        "trace_id_hash": _hmac_id(trace_id, salt=salt, purpose="trace"),
        "turn_id_hash": _hmac_id(turn_id, salt=salt, purpose="turn"),
        "execution_id_hash": _hmac_id(execution_id, salt=salt, purpose="execution"),
        "rules_version": _safe_version(observation.get("rules_version")),
        "prompt_version": _safe_version(observation.get("prompt_version")),
        "verifier_mode": verifier_mode,
        "verifier_status": (
            str(observation.get("status"))
            if str(observation.get("status")) in _SAFE_OBSERVATION_STATUSES else "not_run"
        ),
        "model": model or None,
        "risk_tier": risk_tier,
        "span_ids": span_ids,
        "span_dimensions": {key: span_dimensions[key] for key in span_ids if key in span_dimensions},
        "span_confidence": {key: span_confidence[key] for key in span_ids if key in span_confidence},
        "rule_roles": dict(sorted((key, value) for key, value in rule_roles.items() if key in span_ids)),
        "verifier_roles": dict(sorted(verifier_roles.items())),
        "disagreement_span_ids": disagreement_span_ids,
        "operation": operation,
        "terminal": "auto_executed",
        "clarification_reason": None,
        "contract_hash": contract_hash,
        "authorization_outcome": {
            "status": _status(authorization.get("status")),
            "reason_code": _safe_code(authorization.get("reason_code")),
        },
        "support_outcome": {
            "status": _status(support.get("status")),
            "reason_code": _safe_code(support.get("reason_code")),
        },
        "sampled_at": sampled.astimezone(timezone.utc).isoformat(),
        "sample_rate": shadow_sample_rate(config),
        "sample_bucket": bucket,
        "salt_fingerprint": hashlib.sha256(salt.encode("utf-8")).hexdigest()[:16],
        "actual_execution": True,
        "human_review_status": "pending",
        "review_target": "rule_role",
        "reviewed_label": None,
        "reviewed_span_labels": {},
        "contract_review_status": "pending",
        "wrong_contract": None,
        "contract_review_source": None,
        "execution_review_status": "pending",
        "wrong_execution": None,
        "execution_review_source": None,
    }
    if not validate_audit_record(record):
        return None
    return record


def _status(value: Any) -> str:
    text = str(value or "unknown")
    return text if text in {"passed", "denied", "unknown", "not_evaluated"} else "unknown"


def _one_of(value: Any, choices: set[str]) -> bool:
    return isinstance(value, str) and value in choices


def validate_audit_record(record: Any) -> bool:
    """Validate exact fields and privacy-safe value shapes before persistence."""
    if not isinstance(record, Mapping) or set(record) != _ALLOWED_AUDIT_KEYS:
        return False
    if record.get("schema_version") != AUDIT_SCHEMA_VERSION:
        return False
    if not _SHA256.fullmatch(str(record.get("audit_id") or "")):
        return False
    if not _SHA256.fullmatch(str(record.get("trace_id_hash") or "")):
        return False
    if not _SHA256.fullmatch(str(record.get("contract_hash") or "")):
        return False
    if record.get("terminal") != "auto_executed" or record.get("actual_execution") is not True:
        return False
    if not _one_of(record.get("human_review_status"), {"pending", "reviewed", "adjudicated"}):
        return False
    if record.get("review_target") != "rule_role":
        return False
    if record.get("reviewed_label") is not None and not _one_of(
        record.get("reviewed_label"), {"fp", "not_fp", "unclear"},
    ):
        return False
    if record.get("wrong_contract") is not None and not isinstance(record.get("wrong_contract"), bool):
        return False
    if record.get("wrong_execution") is not None and not isinstance(record.get("wrong_execution"), bool):
        return False
    if (
        record.get("reviewed_label") is not None
        or bool(record.get("reviewed_span_labels"))
    ) and record.get("human_review_status") not in {"reviewed", "adjudicated"}:
        return False
    if record.get("wrong_contract") is not None and (
        not _one_of(record.get("contract_review_status"), {"reviewed", "adjudicated"})
        or not _one_of(record.get("contract_review_source"), {"human", "calibration"})
    ):
        return False
    if record.get("wrong_execution") is not None and (
        not _one_of(record.get("execution_review_status"), {"reviewed", "adjudicated"})
        or not _one_of(record.get("execution_review_source"), {"human_gold", "executable_validator"})
        or record.get("actual_execution") is not True
    ):
        return False
    for key in ("turn_id_hash", "execution_id_hash"):
        if record.get(key) is not None and not _SHA256.fullmatch(str(record[key])):
            return False
    span_ids = record.get("span_ids")
    if (
        not isinstance(span_ids, list)
        or any(not isinstance(item, str) for item in span_ids)
        or len(span_ids) != len(set(span_ids))
    ):
        return False
    if not all(_SAFE_SPAN_ID.fullmatch(item) for item in span_ids):
        return False
    allowed_span_ids = set(span_ids)
    for key in ("rule_roles", "verifier_roles"):
        mapping = record.get(key)
        if not isinstance(mapping, Mapping) or any(
            not _SAFE_SPAN_ID.fullmatch(str(span_id))
            or str(span_id) not in allowed_span_ids
            or not _one_of(role, _SAFE_ROLES)
            for span_id, role in mapping.items()
        ):
            return False
    if record.get("reviewed_label") is not None and len(record.get("rule_roles") or {}) != 1:
        return False
    for key in ("span_dimensions", "span_confidence"):
        mapping = record.get(key)
        if not isinstance(mapping, Mapping) or any(
            not _SAFE_SPAN_ID.fullmatch(str(span_id))
            or str(span_id) not in allowed_span_ids
            or not isinstance(value, str)
            for span_id, value in mapping.items()
        ):
            return False
    if set(record.get("rule_roles") or {}) != allowed_span_ids:
        return False
    if set(record.get("span_dimensions") or {}) != allowed_span_ids:
        return False
    if set(record.get("span_confidence") or {}) != allowed_span_ids:
        return False
    known_dimensions = _known_dimension_ids()
    if any(
        value != "other" and value not in known_dimensions
        for value in (record.get("span_dimensions") or {}).values()
    ):
        return False
    if any(
        not _one_of(value, {"closed", "open", "unresolved", "unknown"})
        for value in (record.get("span_confidence") or {}).values()
    ):
        return False
    if not isinstance(record.get("reviewed_span_labels"), Mapping) or any(
        span_id not in (record.get("rule_roles") or {})
        or not _one_of(label, {"fp", "not_fp", "unclear"})
        for span_id, label in record["reviewed_span_labels"].items()
    ):
        return False
    disagreements = record.get("disagreement_span_ids")
    if (
        not isinstance(disagreements, list)
        or any(not isinstance(span_id, str) for span_id in disagreements)
        or len(disagreements) != len(set(disagreements))
    ):
        return False
    if any(
        not _SAFE_SPAN_ID.fullmatch(str(span_id))
        or span_id not in allowed_span_ids
        for span_id in disagreements
    ):
        return False
    expected_disagreements = sorted(
        span_id for span_id in set(record.get("rule_roles") or {}) & set(record.get("verifier_roles") or {})
        if record["rule_roles"][span_id] != record["verifier_roles"][span_id]
    )
    if disagreements != expected_disagreements:
        return False
    if not all(
        _one_of(record.get(key), {"pending", "reviewed", "adjudicated"})
        for key in ("contract_review_status", "execution_review_status")
    ):
        return False
    if record.get("contract_review_source") is not None and not _one_of(
        record.get("contract_review_source"), {"human", "calibration"},
    ):
        return False
    if record.get("execution_review_source") is not None and not _one_of(
        record.get("execution_review_source"), {"human_gold", "executable_validator"},
    ):
        return False
    for key in ("authorization_outcome", "support_outcome"):
        outcome = record.get(key)
        if not isinstance(outcome, Mapping) or set(outcome) != {"status", "reason_code"}:
            return False
        if not _one_of(outcome.get("status"), {"passed", "denied", "unknown", "not_evaluated"}):
            return False
        if not _SAFE_ID.fullmatch(str(outcome.get("reason_code") or "unknown")):
            return False
    if not _one_of(record.get("verifier_mode"), {"off", "shadow", "open_only"}):
        return False
    if not _one_of(record.get("risk_tier"), _SAFE_RISKS):
        return False
    if not _one_of(record.get("operation"), _SAFE_OPERATIONS):
        return False
    if not _one_of(record.get("verifier_status"), _SAFE_OBSERVATION_STATUSES | {"not_run"}):
        return False
    if record.get("sampling_version") != AUDIT_SAMPLING_VERSION:
        return False
    if not _SAFE_VERSION.fullmatch(str(record.get("rules_version") or "unknown")):
        return False
    if not _SAFE_VERSION.fullmatch(str(record.get("prompt_version") or "unknown")):
        return False
    model = record.get("model")
    if model is not None and not re.fullmatch(r"[A-Za-z0-9._:/-]{1,120}", str(model)):
        return False
    if not re.fullmatch(r"[0-9a-f]{16}", str(record.get("salt_fingerprint") or "")):
        return False
    try:
        if not 0.0 <= float(record.get("sample_rate", -1)) <= 100.0:
            return False
        if not 0 <= int(record.get("sample_bucket", -1)) < (1 << 64):
            return False
        datetime.fromisoformat(str(record.get("sampled_at") or ""))
    except (TypeError, ValueError):
        return False
    if record.get("clarification_reason") is not None and not _SAFE_ID.fullmatch(
        str(record.get("clarification_reason"))
    ):
        return False
    return True


def persist_shadow_audit_record(record: Mapping[str, Any], *, path: str | Path | None = None) -> bool:
    """Insert once per trace using SQLite's cross-process primary-key guard."""
    if not validate_audit_record(record):
        return False
    target = Path(path) if path is not None else _default_audit_db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    packed = json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # This runs synchronously in the terminal workflow node. Keep lock waits
    # short so a busy audit sidecar cannot stall the user's completed request.
    with sqlite3.connect(target, timeout=0.1) as connection:
        connection.execute("PRAGMA busy_timeout = 100")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS semantic_shadow_audit ("
            "audit_id TEXT PRIMARY KEY, sampled_at TEXT NOT NULL, record_json TEXT NOT NULL)"
        )
        cursor = connection.execute(
            "INSERT OR IGNORE INTO semantic_shadow_audit(audit_id, sampled_at, record_json) "
            "VALUES (?, ?, ?)",
            (str(record["audit_id"]), str(record["sampled_at"]), packed),
        )
        return cursor.rowcount == 1


def read_shadow_audit_records(*, path: str | Path | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path is not None else _default_audit_db_path()
    if not target.exists():
        return []
    with sqlite3.connect(target, timeout=30.0) as connection:
        try:
            rows = connection.execute(
                "SELECT record_json FROM semantic_shadow_audit ORDER BY sampled_at, audit_id"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                raise ValueError("audit database is missing semantic_shadow_audit table") from exc
            raise
    records: list[dict[str, Any]] = []
    for (raw,) in rows:
        try:
            record = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if validate_audit_record(record):
            records.append(record)
    return records


def record_completed_shadow_audit(
    state: Mapping[str, Any], *, path: str | Path | None = None,
) -> bool:
    """Best-effort P4 sidecar called from persist_trace; never raises to flow."""
    try:
        record = build_shadow_audit_record(state)
        return bool(record and persist_shadow_audit_record(record, path=path))
    except Exception:  # noqa: BLE001 - audit storage cannot change workflow result
        return False


def _settings() -> Any:
    from app.core.config import get_settings

    return get_settings()
