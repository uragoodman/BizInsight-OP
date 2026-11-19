"""Fail-closed Semantic Contract v2.1 interpreter for shadow and active routing."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from collections.abc import Mapping
from time import perf_counter
from typing import Any, Callable, get_args

from pydantic import ValidationError

from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG, SemanticCatalog
from app.context.semantic_contract_v2 import (
    ArgumentPath,
    SemanticContractCandidateV21,
    SemanticContractV21,
)
from app.context.semantic_contract_validator import validate_semantic_contract
from app.context.semantic_request_scope_normalizer import (
    normalize_cross_clause_negative_request_scope,
)
from app.context.semantic_span_resolver import (
    normalize_candidate,
    normalize_pre_schema_references,
)
from app.core.config import get_settings
from app.llm.interaction_policy import InteractionPolicy
from app.llm.semantic_interpreter_prompt import (
    PROMPT_VERSION,
    build_request,
    build_repair_system_prompt,
    build_system_prompt,
    REPAIR_PROMPT_VERSION,
)


InvokeModel = Callable[[str, str, str], Any]


def semantic_candidate_response_format(
    catalog: SemanticCatalog = DEFAULT_SEMANTIC_CATALOG,
) -> dict[str, Any]:
    """Build DashScope/OpenAI-compatible strict JSON Schema for a Candidate.

    Pydantic's schema is the source of field and enum shapes. Catalog argument
    paths are compiled into per-operation conditionals so strict generation
    cannot place, for example, ``time_grain`` on a QUERY operation. Semantic
    reference checks remain the responsibility of the local Validator.
    """

    schema = deepcopy(SemanticContractCandidateV21.model_json_schema())
    operation_schema = schema.get("$defs", {}).get("OperationCandidate")
    if not isinstance(operation_schema, dict):
        raise RuntimeError("Candidate schema is missing OperationCandidate.")
    properties = operation_schema.get("properties")
    arguments_schema = properties.get("mentioned_arguments") if isinstance(properties, dict) else None
    if not isinstance(arguments_schema, dict):
        raise RuntimeError("Candidate schema is missing mentioned_arguments.")
    value_array_schema = arguments_schema.get("additionalProperties")
    if not isinstance(value_array_schema, dict):
        raise RuntimeError("Candidate schema has an unsupported mentioned_arguments shape.")

    all_argument_paths = get_args(ArgumentPath)
    argument_properties = {
        path: deepcopy(value_array_schema)
        for path in all_argument_paths
    }
    arguments_schema.clear()
    arguments_schema.update({
        "type": "object",
        "properties": argument_properties,
        "required": [],
        "additionalProperties": False,
    })

    # Restrict the path keys further for each operation kind using JSON Schema
    # conditionals supported by DashScope strict mode.
    operation_conditions = []
    for operation_spec in catalog.operation_arguments:
        allowed_properties = {
            path: deepcopy(argument_properties[path])
            for path in operation_spec.allowed_paths
        }
        operation_conditions.append({
            "if": {
                "properties": {"kind": {"const": operation_spec.kind}},
                "required": ["kind"],
            },
            "then": {
                "properties": {
                    "mentioned_arguments": {
                        "type": "object",
                        "properties": allowed_properties,
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            },
        })
    if operation_conditions:
        operation_schema["allOf"] = [
            *operation_schema.get("allOf", []),
            *operation_conditions,
        ]

    # Pydantic emits OpenAPI's discriminator annotation for RelationEndpoint.
    # The endpoint_type const + oneOf branches already express the constraint;
    # dropping this annotation keeps the provider schema within JSON Schema.
    def remove_non_schema_annotations(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("discriminator", None)
            for child in value.values():
                remove_non_schema_annotations(child)
        elif isinstance(value, list):
            for child in value:
                remove_non_schema_annotations(child)

    remove_non_schema_annotations(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "semantic_contract_candidate_v21",
            "strict": True,
            "schema": schema,
        },
    }


class _InvocationFailure(RuntimeError):
    def __init__(self, status: str, reason_code: str, usage: Mapping[str, Any] | None = None):
        super().__init__(reason_code)
        self.status = status
        self.reason_code = reason_code
        self.usage = dict(usage or {})


def _default_invoke(
    system_prompt: str,
    request_json: str,
    model: str,
    *,
    catalog: SemanticCatalog = DEFAULT_SEMANTIC_CATALOG,
    interaction_policy: InteractionPolicy | str,
    max_tokens: int,
) -> tuple[Any, dict[str, Any]]:
    """Call the normal LLM boundary and read only safe usage metadata."""

    from app.core.execution_context import active_state
    from app.llm.client import llm_extract

    active = active_state()
    calls_before = len(active.get("llm_calls") or []) if isinstance(active, dict) else 0
    result = llm_extract(
        system_prompt,
        request_json,
        model=model or None,
        temperature=0.0,
        operation="extract.semantic_interpreter",
        interaction_policy=interaction_policy,
        max_tokens=max_tokens,
        response_format=semantic_candidate_response_format(catalog),
    )
    usage: dict[str, Any] = {}
    call = None
    if isinstance(active, dict):
        calls = active.get("llm_calls") or []
        if len(calls) > calls_before:
            call = calls[-1]
            request_options = call.get("request_options")
            if isinstance(request_options, Mapping):
                safe_option_keys = {
                    "interaction_policy", "transport", "thinking",
                    "connect_timeout_seconds", "idle_timeout_seconds",
                    "total_timeout_seconds", "stream_options_include_usage",
                    "response_format_type", "response_schema_name",
                }
                safe_options = {
                    key: value for key, value in request_options.items()
                    if key in safe_option_keys
                }
                usage["request_options"] = safe_options
                usage["interaction_policy"] = safe_options.get("interaction_policy")
            for key in ("first_token_ms", "reasoning_chars", "partial_output_chars"):
                usage[key] = call.get(key)
            usage["usage_status"] = call.get("usage_status", "unknown")
            usage["timeout_kind"] = call.get("timeout_kind")
            usage["model"] = call.get("requested_model") or model
            attempts = call.get("attempts") or []
            returned_attempts = [item for item in attempts if item.get("result") == "response_received"]
            if returned_attempts:
                attempt = returned_attempts[-1]
                usage = {
                    **usage,
                    "model": attempt.get("model") or model,
                    "input_tokens": call.get("input_tokens_subtotal") if any(
                        item.get("input_tokens") is not None for item in returned_attempts
                    ) else None,
                    "output_tokens": call.get("output_tokens_subtotal") if any(
                        item.get("output_tokens") is not None for item in returned_attempts
                    ) else None,
                    "estimated_cost": call.get("estimated_cost"),
                }
    if result is None:
        outcome = str((call or {}).get("outcome") or "provider_error")
        not_sent_reason = str((call or {}).get("not_sent_reason") or "")
        if outcome in {"timeout", "provider_timeout"}:
            raise _InvocationFailure("unavailable", "timeout", usage)
        if outcome in {"parse_failure", "empty_response"}:
            raise _InvocationFailure("parse_failure", "model_output_unparseable", usage)
        if not_sent_reason == "quota_blocked":
            raise _InvocationFailure("unavailable", "quota_blocked", usage)
        if not_sent_reason == "budget_exhausted":
            raise _InvocationFailure("unavailable", "budget_blocked", usage)
        if not_sent_reason == "no_provider_configured":
            raise _InvocationFailure("unavailable", "provider_unavailable", usage)
        if not_sent_reason == "evaluation_journal_unavailable":
            raise _InvocationFailure("unavailable", "local_send_refused", usage)
        if any("RateLimit" in str(item.get("exception_type") or "") for item in (call or {}).get("attempts", [])):
            raise _InvocationFailure("unavailable", "rate_limited_429", usage)
        raise _InvocationFailure("unavailable", "provider_error", usage)
    return result, usage


def _sampled(turn_id: str, rate: float) -> bool:
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    digest = hashlib.sha256(turn_id.encode("utf-8")).digest()
    fraction = int.from_bytes(digest, "big") / float(1 << (8 * len(digest)))
    return fraction < rate


def _safe_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 and parsed == value else None


def _raw_output_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return f"<non-json-output:{type(value).__name__}>"


def _record_output_diagnostics(
    record: dict[str, Any], raw_output: str, *, capture_raw_output: bool,
) -> None:
    record["raw_output_chars"] = len(raw_output)
    record["raw_output_sha256"] = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
    if capture_raw_output:
        record["raw_output"] = raw_output[:20_000]


def _safe_schema_errors(exc: ValidationError) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for error in exc.errors()[:20]:
        raw_loc = error.get("loc") or ()
        loc = [
            part if isinstance(part, int) else str(part)[:100]
            for part in raw_loc
            if isinstance(part, (str, int))
        ]
        diagnostics.append({
            "loc": loc,
            "type": str(error.get("type") or "validation_error")[:100],
            "msg": str(error.get("msg") or "Invalid value")[:200],
        })
    return diagnostics


def _apply_usage(record: dict[str, Any], usage: Mapping[str, Any] | None) -> None:
    if not isinstance(usage, Mapping):
        return
    if usage.get("model"):
        record["model"] = str(usage["model"])
    record["input_tokens"] = _safe_nonnegative_int(usage.get("input_tokens"))
    record["output_tokens"] = _safe_nonnegative_int(usage.get("output_tokens"))
    cost = usage.get("estimated_cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
        record["estimated_cost"] = float(cost)
    for key in ("first_token_ms", "reasoning_chars", "partial_output_chars"):
        record[key] = _safe_nonnegative_int(usage.get(key))
    status = usage.get("usage_status")
    record["usage_status"] = (
        status if isinstance(status, str) and status in {"complete", "partial", "unknown"}
        else "unknown"
    )
    timeout_kind = usage.get("timeout_kind")
    record["timeout_kind"] = (
        timeout_kind if isinstance(timeout_kind, str) and timeout_kind in {"idle", "total"}
        else None
    )
    interaction_policy = usage.get("interaction_policy")
    if isinstance(interaction_policy, str) and interaction_policy:
        record["interaction_policy"] = interaction_policy
    request_options = usage.get("request_options")
    if isinstance(request_options, Mapping):
        safe_option_keys = {
            "interaction_policy", "transport", "thinking",
            "connect_timeout_seconds", "idle_timeout_seconds",
            "total_timeout_seconds", "stream_options_include_usage",
            "response_format_type", "response_schema_name",
        }
        record["request_options"] = {
            key: value for key, value in request_options.items()
            if key in safe_option_keys
        }


def _observation(
    *,
    status: str,
    mode: str,
    model: str,
    catalog_hash: str,
    reason_code: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "mode": mode,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "catalog_snapshot_hash": catalog_hash,
        "elapsed_ms": 0,
        "interaction_policy": None,
        "request_options": None,
        "first_token_ms": None,
        "reasoning_chars": 0,
        "partial_output_chars": 0,
        "usage_status": "unknown",
        "timeout_kind": None,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_cost": None,
        "reason_code": reason_code,
        "candidate": None,
        "contract": None,
        "issue_codes": [],
        "coordinate_resolution": None,
        "request_scope_normalization": None,
    }


def _interpret_semantics_once(
    question: str,
    *,
    turn_id: str,
    dialogue_context: dict[str, Any] | None = None,
    catalog: SemanticCatalog = DEFAULT_SEMANTIC_CATALOG,
    mode: str,
    active_canary_authorized: bool = False,
    model: str = "",
    invoke_model: InvokeModel | None = None,
    interaction_policy: InteractionPolicy | str | None = None,
    _repair_feedback: Mapping[str, Any] | None = None,
    _repair_capture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Interpret one opted-in shadow or explicitly enrolled canary request.

    All failures are represented in the returned observation. This function
    never raises provider, parsing, schema, or validator failures to the caller.
    Active mode is unavailable unless the graph's tenant/template enrollment
    gate has already authorized this request.
    """

    settings = get_settings()
    effective_model = (
        model
        or settings.semantic_interpreter_model
        or settings.llm_standard_model
        or settings.llm_model
    )
    catalog_hash = catalog.snapshot_hash()
    if mode not in {"shadow", "active"}:
        return _observation(
            status="skipped", mode=mode, model=effective_model,
            catalog_hash=catalog_hash, reason_code="mode_not_supported",
        )
    if mode == "active" and not active_canary_authorized:
        return _observation(
            status="skipped", mode=mode, model=effective_model,
            catalog_hash=catalog_hash, reason_code="active_canary_not_authorized",
        )
    if not settings.semantic_interpreter_send_raw_question:
        return _observation(
            status="skipped", mode=mode, model=effective_model,
            catalog_hash=catalog_hash, reason_code="raw_question_not_permitted",
        )
    if not str(question).strip():
        return _observation(
            status="skipped", mode=mode, model=effective_model,
            catalog_hash=catalog_hash, reason_code="empty_question",
        )
    sample_rate = (
        float(settings.semantic_interpreter_shadow_sample_rate)
        if mode == "shadow"
        else 1.0
    )
    if mode == "shadow" and not _sampled(str(turn_id), sample_rate):
        return _observation(
            status="skipped", mode=mode, model=effective_model,
            catalog_hash=catalog_hash,
            reason_code="sampling_disabled" if sample_rate <= 0 else "sample_miss",
        )

    record = _observation(
        status="unavailable", mode=mode, model=effective_model,
        catalog_hash=catalog_hash,
    )
    started = perf_counter()
    try:
        system_prompt = (
            build_repair_system_prompt(catalog)
            if _repair_feedback is not None else build_system_prompt(catalog)
        )
        request_json = build_request(
            question, str(turn_id), dialogue_context, catalog,
            repair_feedback=_repair_feedback,
        )
        if invoke_model is None:
            response = _default_invoke(
                system_prompt,
                request_json,
                effective_model,
                catalog=catalog,
                interaction_policy=(
                    interaction_policy
                    if interaction_policy is not None
                    else str(settings.semantic_interpreter_interaction_policy)
                ),
                max_tokens=int(settings.semantic_interpreter_max_tokens),
            )
        else:
            response = invoke_model(system_prompt, request_json, effective_model)
        if isinstance(response, tuple) and len(response) == 2 and isinstance(response[1], Mapping):
            raw_candidate, usage = response
        else:
            raw_candidate, usage = response, {}
        _apply_usage(record, usage)
        raw_output_text = _raw_output_text(raw_candidate)
        if _repair_capture is not None:
            _repair_capture["raw_output_excerpt"] = raw_output_text[:4_000]
        capture_raw_output = bool(settings.semantic_interpreter_capture_raw_output)
        if isinstance(raw_candidate, str):
            try:
                raw_candidate = json.loads(raw_candidate)
            except (json.JSONDecodeError, TypeError):
                _record_output_diagnostics(
                    record, raw_output_text, capture_raw_output=capture_raw_output,
                )
                record.update(status="parse_failure", reason_code="invalid_json")
                return record
        if not isinstance(raw_candidate, dict):
            _record_output_diagnostics(
                record, raw_output_text, capture_raw_output=capture_raw_output,
            )
            record.update(status="parse_failure", reason_code="candidate_not_object")
            return record
        span_resolution_enabled = bool(
            getattr(settings, "semantic_interpreter_span_resolution_enabled", True)
        )
        pre_schema_resolution = None
        candidate_data = raw_candidate
        if span_resolution_enabled:
            try:
                candidate_data, pre_schema_resolution = normalize_pre_schema_references(raw_candidate)
                record["coordinate_resolution"] = {
                    "enabled": True,
                    "pre_schema_reference_normalization": pre_schema_resolution,
                }
                if int(pre_schema_resolution.get("ungrounded_operations_dropped", 0) or 0) > 0:
                    # Removing an operation can change the requested meaning.
                    # Treat this as a repairable candidate defect and fail
                    # closed if the bounded repair does not supply a grounded
                    # representation; never route a silently shortened intent.
                    _record_output_diagnostics(
                        record, raw_output_text, capture_raw_output=capture_raw_output,
                    )
                    record["schema_errors"] = [{
                        "loc": ["operations"],
                        "type": "value_error",
                        "msg": (
                            "Pre-schema normalization would remove an ungrounded operation; "
                            "all requested operations must remain represented."
                        ),
                    }]
                    record.update(
                        status="schema_invalid",
                        reason_code="unsafe_operation_pruning",
                    )
                    return record
            except Exception as exc:  # noqa: BLE001 - deterministic repair must fail closed
                _record_output_diagnostics(
                    record, raw_output_text, capture_raw_output=capture_raw_output,
                )
                record["coordinate_resolution"] = {
                    "enabled": True,
                    "failed": True,
                    "error_type": type(exc).__name__,
                    "pre_schema_reference_normalization": pre_schema_resolution,
                }
                record.update(status="unavailable", reason_code="coordinate_normalization_failed")
                return record
        try:
            candidate_json = json.dumps(candidate_data, ensure_ascii=False, separators=(",", ":"))
            candidate = SemanticContractCandidateV21.model_validate_json(candidate_json)
        except ValidationError as exc:
            _record_output_diagnostics(
                record, raw_output_text, capture_raw_output=capture_raw_output,
            )
            record["schema_errors"] = _safe_schema_errors(exc)
            record.update(status="schema_invalid", reason_code="candidate_schema_invalid")
            return record
        except (TypeError, ValueError) as exc:
            _record_output_diagnostics(
                record, raw_output_text, capture_raw_output=capture_raw_output,
            )
            record["schema_errors"] = [{
                "loc": [],
                "type": type(exc).__name__,
                "msg": "Candidate could not be serialized for schema validation.",
            }]
            record.update(status="schema_invalid", reason_code="candidate_schema_invalid")
            return record

        if span_resolution_enabled:
            try:
                normalized_dict, resolution = normalize_candidate(
                    candidate.model_dump(mode="json"), question, catalog,
                )
            except Exception as exc:  # noqa: BLE001 - deterministic repair must fail closed
                _record_output_diagnostics(
                    record, raw_output_text, capture_raw_output=capture_raw_output,
                )
                record["coordinate_resolution"] = {
                    "enabled": True,
                    "failed": True,
                    "error_type": type(exc).__name__,
                    "pre_schema_reference_normalization": pre_schema_resolution,
                }
                record.update(status="unavailable", reason_code="coordinate_normalization_failed")
                return record
            resolution["pre_schema_reference_normalization"] = pre_schema_resolution
            record["coordinate_resolution"] = resolution
            try:
                normalized_json = json.dumps(
                    normalized_dict, ensure_ascii=False, separators=(",", ":"),
                )
                candidate = SemanticContractCandidateV21.model_validate_json(normalized_json)
            except ValidationError as exc:
                _record_output_diagnostics(
                    record, raw_output_text, capture_raw_output=capture_raw_output,
                )
                record["schema_errors"] = _safe_schema_errors(exc)
                record.update(status="schema_invalid", reason_code="normalized_candidate_schema_invalid")
                return record
            except (TypeError, ValueError) as exc:
                _record_output_diagnostics(
                    record, raw_output_text, capture_raw_output=capture_raw_output,
                )
                record["schema_errors"] = [{
                    "loc": [],
                    "type": type(exc).__name__,
                    "msg": "Normalized Candidate could not be serialized for schema validation.",
                }]
                record.update(status="schema_invalid", reason_code="normalized_candidate_schema_invalid")
                return record

        request_scope_normalization_enabled = bool(getattr(
            settings, "semantic_interpreter_request_scope_normalization_enabled", True,
        ))
        if request_scope_normalization_enabled:
            try:
                candidate_data, request_scope_stats = normalize_cross_clause_negative_request_scope(
                    candidate.model_dump(mode="json"), question,
                )
                record["request_scope_normalization"] = request_scope_stats
            except Exception as exc:  # noqa: BLE001 - normalization must fail closed
                _record_output_diagnostics(
                    record, raw_output_text, capture_raw_output=capture_raw_output,
                )
                record["request_scope_normalization"] = {
                    "enabled": True,
                    "failed": True,
                    "error_type": type(exc).__name__,
                }
                record.update(
                    status="unavailable",
                    reason_code="request_scope_normalization_failed",
                )
                return record
            try:
                normalized_scope_json = json.dumps(
                    candidate_data, ensure_ascii=False, separators=(",", ":"),
                )
                candidate = SemanticContractCandidateV21.model_validate_json(
                    normalized_scope_json,
                )
            except ValidationError as exc:
                _record_output_diagnostics(
                    record, raw_output_text, capture_raw_output=capture_raw_output,
                )
                record["schema_errors"] = _safe_schema_errors(exc)
                record.update(
                    status="schema_invalid",
                    reason_code="request_scope_normalized_candidate_schema_invalid",
                )
                return record
            except (TypeError, ValueError) as exc:
                _record_output_diagnostics(
                    record, raw_output_text, capture_raw_output=capture_raw_output,
                )
                record["schema_errors"] = [{
                    "loc": [],
                    "type": type(exc).__name__,
                    "msg": "Request-scope-normalized Candidate could not be serialized.",
                }]
                record.update(
                    status="schema_invalid",
                    reason_code="request_scope_normalized_candidate_schema_invalid",
                )
                return record
        else:
            record["request_scope_normalization"] = {
                "enabled": False,
                "operations_seen": 0,
                "negative_same_clause_examined": 0,
                "request_scope_rewrites": 0,
            }

        record["candidate"] = candidate.model_dump(mode="json")
        try:
            trusted_prior_contract = (
                dialogue_context.get("trusted_prior_contract")
                if isinstance(dialogue_context, Mapping)
                else None
            )
            if not isinstance(trusted_prior_contract, SemanticContractV21):
                trusted_prior_contract = None
            validated = validate_semantic_contract(
                candidate,
                source_text=question,
                catalog=catalog,
                trusted_prior_contract=trusted_prior_contract,
            )
        except Exception:  # noqa: BLE001 - validator is also an observational boundary
            record.update(status="unavailable", reason_code="validator_failure")
            return record
        record["contract"] = validated.contract.model_dump(mode="json")
        record["validator_diagnostics"] = [
            {
                "code": issue.code,
                "location": issue.location,
                "severity": issue.severity,
            }
            for issue in validated.issues[:50]
        ]
        record["issue_codes"] = sorted({issue.code for issue in validated.issues})
        if record["issue_codes"]:
            record.update(status="validator_issues", reason_code="validator_issues_present")
        else:
            record.update(status="ok", reason_code=None)
        return record
    except _InvocationFailure as exc:
        _apply_usage(record, exc.usage)
        record.update(status=exc.status, reason_code=exc.reason_code)
        return record
    except TimeoutError:
        record.update(status="unavailable", reason_code="timeout")
        return record
    except Exception:  # noqa: BLE001 - provider adapters and shadow parsing must fail closed
        record.update(status="unavailable", reason_code="interpreter_call_failed")
        return record
    finally:
        record["elapsed_ms"] = max(0, round((perf_counter() - started) * 1000))


_REPAIRABLE_STATUSES = frozenset({"parse_failure", "schema_invalid", "validator_issues"})


def _sum_usage(attempts: list[Mapping[str, Any]], key: str) -> int | float | None:
    values = [attempt.get(key) for attempt in attempts]
    if not values or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        return None
    return sum(values)


def _repair_guidance(
    observation: Mapping[str, Any], catalog: SemanticCatalog,
) -> list[dict[str, Any]]:
    """Turn machine diagnostics into bounded, operation-specific repair hints."""

    guidance: list[dict[str, Any]] = []
    candidate = observation.get("candidate")
    operations = candidate.get("operations") if isinstance(candidate, Mapping) else None
    diagnostics = observation.get("validator_diagnostics")
    if isinstance(diagnostics, list) and isinstance(operations, list):
        path_pattern = re.compile(
            r"^operations(?:\[(\d+)\]|\.(\d+))\.mentioned_arguments\.([A-Za-z_]+)$"
        )
        for diagnostic in diagnostics[:20]:
            if not isinstance(diagnostic, Mapping) or diagnostic.get("code") != "invalid_argument_path":
                continue
            location = str(diagnostic.get("location") or "")
            match = path_pattern.match(location)
            if not match:
                continue
            operation_index = int(match.group(1) or match.group(2))
            if operation_index >= len(operations) or not isinstance(operations[operation_index], Mapping):
                continue
            operation = operations[operation_index]
            kind = operation.get("kind")
            if not isinstance(kind, str):
                continue
            allowed_paths = catalog.operation_paths(kind)  # type: ignore[arg-type]
            invalid_path = match.group(3)
            guidance.append({
                "code": "invalid_argument_path",
                "operation_index": operation_index,
                "operation_kind": kind,
                "invalid_path": invalid_path,
                "allowed_paths": list(allowed_paths),
                "instruction": (
                    f"Operation {operation_index} is {kind}; {invalid_path!r} is not an allowed "
                    f"argument path. Allowed paths are: {', '.join(allowed_paths) or '(none)'}. "
                    "Preserve the meaning stated in the original question. Remove the unsupported "
                    "value only if it is not needed to represent that meaning; otherwise return "
                    "an appropriate AMBIGUOUS or INCOMPLETE Candidate instead of guessing."
                ),
            })

    schema_errors = observation.get("schema_errors")
    if isinstance(schema_errors, list):
        for error in schema_errors[:20]:
            if not isinstance(error, Mapping):
                continue
            error_type = str(error.get("type") or "")
            location = error.get("loc") or []
            if error_type == "string_type":
                path = ".".join(str(part) for part in location) or "Candidate field"
                guidance.append({
                    "code": "schema_string_type",
                    "location": path,
                    "instruction": (
                        f"{path} must be a string, not null or an object. Use a grounded string "
                        "value supported by the question/catalog; if the value cannot be grounded, "
                        "represent uncertainty safely instead of inventing an identifier."
                    ),
                })
    return guidance[:20]


def interpret_semantics(
    question: str,
    *,
    turn_id: str,
    dialogue_context: dict[str, Any] | None = None,
    catalog: SemanticCatalog = DEFAULT_SEMANTIC_CATALOG,
    mode: str,
    active_canary_authorized: bool = False,
    model: str = "",
    invoke_model: InvokeModel | None = None,
    interaction_policy: InteractionPolicy | str | None = None,
    validation_repair: bool = False,
) -> dict[str, Any]:
    """Run the Interpreter with an optional single bounded validation repair.

    Repair is limited to parse/schema/Validator failures. Provider timeouts,
    quota errors, and other unavailable states remain fail-closed without retry.
    Shadow callers opt in explicitly; active graph callers use the configured
    repair switch. The original and final attempts are retained as safe metadata.
    """

    started = perf_counter()
    first_capture: dict[str, Any] = {}
    first = _interpret_semantics_once(
        question,
        turn_id=turn_id,
        dialogue_context=dialogue_context,
        catalog=catalog,
        mode=mode,
        active_canary_authorized=active_canary_authorized,
        model=model,
        invoke_model=invoke_model,
        interaction_policy=interaction_policy,
        _repair_capture=first_capture,
    )
    if (
        not validation_repair
        or mode not in {"shadow", "active"}
        or first.get("status") not in _REPAIRABLE_STATUSES
    ):
        return first

    feedback: dict[str, Any] = {
        "previous_status": first.get("status"),
        "reason_code": first.get("reason_code"),
        "schema_errors": first.get("schema_errors") or [],
        "validator_diagnostics": first.get("validator_diagnostics") or [],
        "issue_codes": first.get("issue_codes") or [],
    }
    guidance = _repair_guidance(first, catalog)
    if guidance:
        feedback["repair_guidance"] = guidance
    if isinstance(first.get("candidate"), Mapping):
        feedback["previous_candidate"] = first["candidate"]
    if isinstance(first_capture.get("raw_output_excerpt"), str):
        feedback["previous_output_excerpt"] = first_capture["raw_output_excerpt"]

    second_capture: dict[str, Any] = {}
    repaired = _interpret_semantics_once(
        question,
        turn_id=turn_id,
        dialogue_context=dialogue_context,
        catalog=catalog,
        mode=mode,
        active_canary_authorized=active_canary_authorized,
        model=model,
        invoke_model=invoke_model,
        interaction_policy=interaction_policy,
        _repair_feedback=feedback,
        _repair_capture=second_capture,
    )
    recovered = repaired.get("status") == "ok"
    repaired["repair"] = {
        "attempted": True,
        "prompt_version": REPAIR_PROMPT_VERSION,
        "attempt_count": 2,
        "retry_count": 1,
        "recovered": recovered,
        "initial_status": first.get("status"),
        "initial_reason_code": first.get("reason_code"),
        "initial_schema_errors": first.get("schema_errors") or [],
        "initial_issue_codes": first.get("issue_codes") or [],
        "initial_validator_diagnostics": first.get("validator_diagnostics") or [],
        "final_status": repaired.get("status"),
        "final_reason_code": repaired.get("reason_code"),
        "final_schema_errors": repaired.get("schema_errors") or [],
        "final_issue_codes": repaired.get("issue_codes") or [],
        "final_validator_diagnostics": repaired.get("validator_diagnostics") or [],
        "rejected_contract_present": bool(repaired.get("contract")) and not recovered,
        "attempts": [
            {
                "status": first.get("status"),
                "elapsed_ms": first.get("elapsed_ms"),
                "input_tokens": first.get("input_tokens"),
                "output_tokens": first.get("output_tokens"),
                "estimated_cost": first.get("estimated_cost"),
            },
            {
                "status": repaired.get("status"),
                "elapsed_ms": repaired.get("elapsed_ms"),
                "input_tokens": repaired.get("input_tokens"),
                "output_tokens": repaired.get("output_tokens"),
                "estimated_cost": repaired.get("estimated_cost"),
            },
        ],
    }
    if not recovered:
        # A second candidate with Validator issues is evidence, not a routeable
        # contract. Keep its candidate/diagnostics but enforce the no-route boundary.
        repaired["contract"] = None
    for key in ("input_tokens", "output_tokens", "estimated_cost", "reasoning_chars", "partial_output_chars"):
        repaired[key] = _sum_usage([first, repaired], key)
    usage_statuses = {first.get("usage_status"), repaired.get("usage_status")}
    if usage_statuses == {"complete"}:
        repaired["usage_status"] = "complete"
    elif "partial" in usage_statuses:
        repaired["usage_status"] = "partial"
    else:
        repaired["usage_status"] = "unknown"
    repaired["elapsed_ms"] = max(0, round((perf_counter() - started) * 1000))
    return repaired
