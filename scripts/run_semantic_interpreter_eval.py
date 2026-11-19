"""Run a bounded, manual real-provider evaluation of the shadow interpreter.

This script is intentionally not wired into CI. It accepts only synthetic,
hand-authored evaluation cases and never invokes the graph, router, selector,
or a template. Real provider access requires an explicit CLI acknowledgement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections.abc import Mapping
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_CASES = 100
MAX_TOTAL_TOKENS = 200_000
MAX_TOTAL_TOKENS_HARD_MAX = 1_000_000
INTER_CASE_DELAY_SECONDS = 0.65
PUBLIC_INPUT_PRICE_PER_1K_CNY = 0.001094
PUBLIC_OUTPUT_PRICE_PER_1K_CNY = 0.003427
PARSE_FAILURE_DIAGNOSTIC_CASE_IDS = frozenset({
    "v341-required-011-11",
    "v341-labeled-alias-01",
    "v341-labeled-sort-basis-01",
})
PARSE_FAILURE_DIAGNOSTIC_MAX_CHARS = 500
VALID_STATUSES = {"ok", "validator_issues"}
FAIL_CLOSED_STATUSES = {"parse_failure", "schema_invalid", "validator_issues", "unavailable"}
SPAN_REFERENCE_ISSUES = {
    "span_out_of_range",
    "span_surface_mismatch",
    "invalid_clause_partition",
    "missing_clause_partition",
    "unknown_clause_reference",
    "unknown_concept_reference",
    "unknown_mention_reference",
    "unknown_role_reference",
    "unknown_intent_reference",
    "mention_outside_owned_clause",
    "operation_outside_owned_clause",
    "target_concept_reference_mismatch",
    "argument_concept_reference_mismatch",
    "intent_id_used_as_concept",
}

DEFAULT_DATASETS = (
    "evaluation/datasets/complex_semantic_v341_adversarial.jsonl",
    "evaluation/datasets/complex_semantic_v341_labeled.jsonl",
)
REPORT_PATH = "docs/design/complex-intent-router-v341-p1-eval-report.md"
REVIEW_SHEET_PATH = "docs/design/complex-intent-router-v341-p1-eval-review-sheet.md"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_cases(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    cases: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for path in paths:
        raw = path.read_bytes()
        hashes[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = _sha256_bytes(raw)
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(case, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            case_id = case.get("case_id")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"Missing case_id at {path}:{line_number}")
            if not isinstance(case.get("text"), str) or not case["text"].strip():
                raise ValueError(f"Missing non-empty text at {path}:{line_number}")
            if not isinstance(case.get("expected_contract"), dict):
                raise ValueError(f"Missing expected_contract at {path}:{line_number}")
            if case_id in seen:
                if seen[case_id].get("text") != case.get("text"):
                    raise ValueError(f"case_id {case_id!r} is duplicated with different text")
                continue
            case["dataset"] = path.stem
            case.setdefault("label_status", "draft")
            if case["label_status"] not in {"draft", "reviewed"}:
                raise ValueError(f"Invalid label_status for {case_id!r}")
            case.setdefault("notes", "")
            seen[case_id] = case
            cases.append(case)
    return cases, hashes


def _evaluation_interaction_policy(
    policy_name: str, *, no_provider_retries: bool,
) -> Any:
    from app.llm.interaction_policy import POLICIES

    policy = POLICIES[policy_name]
    if no_provider_retries:
        return replace(
            policy,
            retry_before_first_token=0,
            provider_fallback_enabled=False,
        )
    return policy


def _preflight(
    *, allow_real_provider: bool, model: str, dry_run: bool,
    no_provider_retries: bool,
) -> dict[str, Any]:
    from app.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    problems: list[str] = []
    if not dry_run and not allow_real_provider:
        problems.append("real provider access requires --allow-real-provider")
    if settings.semantic_interpreter_mode != "shadow":
        problems.append("SEMANTIC_INTERPRETER_MODE must be shadow")
    if not settings.semantic_interpreter_inline_shadow_enabled:
        problems.append("SEMANTIC_INTERPRETER_INLINE_SHADOW_ENABLED must be true for the offline evaluator")
    if not settings.semantic_interpreter_send_raw_question:
        problems.append("SEMANTIC_INTERPRETER_SEND_RAW_QUESTION must be true")
    if float(settings.semantic_interpreter_shadow_sample_rate) < 1.0:
        problems.append("SEMANTIC_INTERPRETER_SHADOW_SAMPLE_RATE must be 1.0")
    if not settings.semantic_interpreter_model:
        problems.append("SEMANTIC_INTERPRETER_MODEL must be configured")
    if settings.semantic_interpreter_model != model:
        problems.append("configured interpreter model must match --model")
    if int(settings.semantic_interpreter_max_tokens) > 2048:
        problems.append("SEMANTIC_INTERPRETER_MAX_TOKENS must not exceed 2048")
    if not settings.deepseek_api_key:
        problems.append("DEEPSEEK_API_KEY is missing")
    if not settings.deepseek_base_url:
        problems.append("DEEPSEEK_BASE_URL is missing")
    if problems:
        raise ValueError("Preflight refused: " + "; ".join(problems))
    policy = _evaluation_interaction_policy(
        settings.semantic_interpreter_interaction_policy,
        no_provider_retries=no_provider_retries,
    )
    return {
        "mode": settings.semantic_interpreter_mode,
        "inline_shadow_enabled": bool(settings.semantic_interpreter_inline_shadow_enabled),
        "raw_question_opt_in": bool(settings.semantic_interpreter_send_raw_question),
        "sample_rate": float(settings.semantic_interpreter_shadow_sample_rate),
        "model": model,
        "interaction_policy": policy.name,
        "transport": policy.transport,
        "thinking": policy.thinking,
        "connect_timeout_seconds": policy.connect_timeout_seconds,
        "idle_timeout_seconds": policy.idle_timeout_seconds,
        "total_timeout_seconds": policy.total_timeout_seconds,
        "retry_before_first_token": policy.retry_before_first_token,
        "provider_fallback_enabled": policy.provider_fallback_enabled,
        "legacy_timeout_seconds_ignored_by_policy_path": float(settings.semantic_interpreter_timeout_seconds),
        "max_tokens": int(settings.semantic_interpreter_max_tokens),
        "provider_configured": bool(settings.deepseek_api_key and settings.deepseek_base_url),
        "provider_host": _safe_host(settings.deepseek_base_url),
        "price_version": settings.llm_price_version,
        "input_cost_per_1k": float(settings.llm_input_cost_per_1k),
        "output_cost_per_1k": float(settings.llm_output_cost_per_1k),
        "prompt_version": None,
        "catalog_snapshot_hash": None,
    }


def _safe_host(base_url: str) -> str | None:
    from urllib.parse import urlsplit

    return urlsplit(base_url).hostname or None


def _provider_failure(state: dict[str, Any]) -> dict[str, Any] | None:
    """Project safe failure metadata; never persist provider messages or headers."""
    calls = state.get("llm_calls") or []
    if not calls:
        return None
    call = calls[-1]
    attempts = call.get("attempts") or []
    exception_types = [str(item.get("exception_type") or "") for item in attempts]
    outcome = str(call.get("outcome") or "")
    if outcome in {"timeout", "provider_timeout"} or any(
        "Timeout" in value or "timeout" in value.casefold() for value in exception_types
    ):
        code = "timeout"
    elif any("RateLimit" in value for value in exception_types):
        code = "rate_limited_429"
    elif outcome in {"parse_failure", "empty_response"}:
        code = "provider_response_unparseable"
    elif outcome in {"provider_error", "configuration_error"}:
        code = "provider_error"
    elif outcome == "not_sent":
        code = str(call.get("not_sent_reason") or "request_not_sent")
    else:
        code = None
    if code is None:
        return None
    return {
        "code": code,
        "outcome": outcome,
        "exception_types": sorted(set(value for value in exception_types if value)),
        "attempt_count": len(attempts),
    }


def _known_token_total(observation: dict[str, Any]) -> int:
    return sum(
        int(observation[key])
        for key in ("input_tokens", "output_tokens")
        if isinstance(observation.get(key), int) and not isinstance(observation.get(key), bool)
    )


def _run_case(
    case: dict[str, Any], *, model: str,
    interaction_policy: Any = None,
    validation_repair: bool = False,
) -> dict[str, Any]:
    from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG
    from app.context.semantic_interpreter import interpret_semantics
    from app.core.execution_context import activate_state, reset_state
    from app.core.config import get_settings
    from app.resilience import policy as resilience_policy

    # Cases run sequentially in this standalone evaluator. Clear only the LLM
    # breaker so one transient failure does not contaminate later observations.
    resilience_policy._BREAKERS.pop("llm", None)
    state: dict[str, Any] = {"trace_id": case["case_id"], "llm_calls": []}
    if case["case_id"] in PARSE_FAILURE_DIAGNOSTIC_CASE_IDS:
        state["_semantic_interpreter_eval_raw_output_case_id"] = case["case_id"]
    token = activate_state(state)
    escaped = False
    try:
        observation = interpret_semantics(
            case["text"], turn_id=case["case_id"], mode="shadow", model=model,
            interaction_policy=interaction_policy,
            validation_repair=validation_repair,
        )
    except Exception as exc:  # defensive boundary for the evaluation harness
        escaped = True
        observation = {
            "status": "unavailable",
            "mode": "shadow",
            "model": model,
            "prompt_version": "unknown",
            "catalog_snapshot_hash": DEFAULT_SEMANTIC_CATALOG.snapshot_hash(),
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
            "reason_code": "uncaught_" + type(exc).__name__,
            "candidate": None,
            "contract": None,
            "issue_codes": [],
            "coordinate_resolution": None,
        }
    finally:
        reset_state(token)
    repair = observation.get("repair") if isinstance(observation.get("repair"), dict) else {}
    initial_status = repair.get("initial_status") if repair.get("attempted") is True else observation.get("status")
    if case["case_id"] in PARSE_FAILURE_DIAGNOSTIC_CASE_IDS and initial_status == "parse_failure":
        calls = state.get("llm_calls") or []
        diagnostic = next((
            call.get("semantic_interpreter_eval_raw_output_diagnostic")
            for call in calls
            if isinstance(call, dict)
            and isinstance(call.get("semantic_interpreter_eval_raw_output_diagnostic"), dict)
        ), None)
        observation = dict(observation)
        if isinstance(diagnostic, dict):
            excerpt = diagnostic.get("truncated_text")
            safe_diagnostic = {
                "raw_output_chars": diagnostic.get("raw_output_chars"),
                "raw_output_sha256": diagnostic.get("raw_output_sha256"),
                "truncated_text": (
                    excerpt[:PARSE_FAILURE_DIAGNOSTIC_MAX_CHARS]
                    if isinstance(excerpt, str) else ""
                ),
                "truncated": bool(diagnostic.get("truncated")),
                "capture_status": "captured",
            }
        else:
            safe_diagnostic = {
                "capture_status": "unavailable",
                "reason": "No bounded output diagnostic was recorded by llm_extract.",
            }
        if repair.get("attempted") is True:
            repair = dict(repair)
            repair["initial_raw_output_diagnostic"] = safe_diagnostic
            observation["repair"] = repair
        else:
            observation["raw_output_diagnostic"] = safe_diagnostic
    provider_failure = _provider_failure(state)
    if provider_failure:
        observation = dict(observation)
        if provider_failure["code"] == "rate_limited_429":
            observation["reason_code"] = "rate_limited_429"
        elif provider_failure["code"] == "timeout":
            observation["reason_code"] = "timeout"
    settings = get_settings()
    input_tokens = observation.get("input_tokens")
    output_tokens = observation.get("output_tokens")
    actual_cost = observation.get("estimated_cost")
    configured_input_price = float(settings.llm_input_cost_per_1k)
    configured_output_price = float(settings.llm_output_cost_per_1k)
    fallback_prices = configured_input_price <= 0 or configured_output_price <= 0
    input_price = configured_input_price if configured_input_price > 0 else PUBLIC_INPUT_PRICE_PER_1K_CNY
    output_price = configured_output_price if configured_output_price > 0 else PUBLIC_OUTPUT_PRICE_PER_1K_CNY
    reference_cost = None
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        reference_cost = (input_tokens / 1000 * input_price) + (output_tokens / 1000 * output_price)
    observation = dict(observation)
    observation["reference_cost_estimate_cny"] = reference_cost
    if actual_cost in (None, 0, 0.0) and reference_cost is not None:
        observation["estimated_cost"] = reference_cost
        observation["cost_estimate_source"] = (
            "public_price_fallback_estimate" if fallback_prices else "configured_price_reference"
        )
    elif isinstance(actual_cost, (int, float)) and actual_cost > 0:
        observation["cost_estimate_source"] = "provider_usage_with_configured_price"
    else:
        observation["cost_estimate_source"] = "unknown_usage_or_price"
    result = {
        "case_id": case["case_id"],
        "dataset": case["dataset"],
        "category": case.get("category", "unspecified"),
        "text": case["text"],
        "label_status": case["label_status"],
        "notes": case.get("notes", ""),
        "expected": {
            "status": case.get("expected_status"),
            "contract": case["expected_contract"],
        },
        # Reserved for a future Router evaluation; it is deliberately not
        # included in semantic-status or Candidate/Validator field scoring.
        "expected_capability_outcome": case.get("expected_capability_outcome"),
        "observation": observation,
        "provider_failure": provider_failure,
        "exception_escaped": escaped,
    }
    if isinstance(case.get("review_source"), str):
        result["review_source"] = case["review_source"]
    return result


def _normalize_string_set(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return [str(values)]
    return sorted({str(value) for value in values if value is not None})


def _normalize_argument_map(value: Any, *, with_stance: bool) -> dict[str, list[Any]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, list[Any]] = {}
    for path, raw_values in value.items():
        if not isinstance(raw_values, (list, tuple)):
            raw_values = [raw_values]
        rows: set[tuple[str, str]] = set()
        for item in raw_values:
            if isinstance(item, dict):
                concept_id = item.get("concept_id")
                stance = item.get("stance", "ADOPTED")
            else:
                concept_id = item
                stance = "ADOPTED"
            if concept_id is not None:
                rows.add((str(concept_id), str(stance)))
        if with_stance:
            values_out: list[Any] = [[concept, stance] for concept, stance in sorted(rows)]
        else:
            values_out = sorted({concept for concept, _stance in rows})
        if values_out:
            normalized[str(path)] = values_out
    return normalized


def _actual_excluded_arguments(operation: dict[str, Any]) -> dict[str, list[str]]:
    mentioned = operation.get("mentioned_arguments") or {}
    excluded: dict[str, list[str]] = {}
    if not isinstance(mentioned, dict):
        return excluded
    for path, values in mentioned.items():
        if not isinstance(values, (list, tuple)):
            continue
        concepts = sorted({
            str(value.get("concept_id"))
            for value in values
            if isinstance(value, dict)
            and value.get("stance") == "EXCLUDED"
            and value.get("concept_id") is not None
        })
        if concepts:
            excluded[str(path)] = concepts
    return excluded


def _normalize_negations(value: Any, mentions: dict[str, Any] | None = None) -> list[list[str]]:
    if not isinstance(value, (list, tuple)):
        return []
    mentions = mentions or {}
    projected: set[tuple[str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        concept_id = item.get("concept_id")
        mention_id = item.get("target_value_mention_id")
        if concept_id is None and mention_id is not None:
            mention = mentions.get(str(mention_id))
            if isinstance(mention, dict):
                concept_id = mention.get("concept_id")
        path = item.get("argument_path")
        # Older adversarial fixtures used a non-schema `requires_clarification`
        # field. It is represented by the expected semantic status instead.
        if path is not None and concept_id is not None:
            projected.add((str(path), str(concept_id)))
    return [[path, concept] for path, concept in sorted(projected)]


def _candidate_expected_operation(operation: dict[str, Any]) -> dict[str, Any]:
    candidate = operation.get("candidate")
    if isinstance(candidate, dict):
        return candidate
    known = {
        "polarity", "mode", "clause_scope", "mentioned_arguments", "argument_negations",
    }
    return {key: value for key, value in operation.items() if key in known}


def _validator_expected_operation(operation: dict[str, Any]) -> dict[str, Any]:
    validator = operation.get("validator")
    if isinstance(validator, dict):
        return validator
    known = {"effective_arguments", "excluded_arguments"}
    return {key: value for key, value in operation.items() if key in known}


def _operation_key(operation: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return (
        str(operation.get("kind") or ""),
        tuple(sorted(_normalize_string_set(operation.get("target_concept_ids")))),
    )


def _match_operations(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    remaining = list(actual)
    matched: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for exp in expected:
        expected_key = _operation_key(exp)
        exp_candidate = _candidate_expected_operation(exp)
        candidates = [
            item for item in remaining
            if item.get("kind") == expected_key[0]
            and (
                not expected_key[1]
                or tuple(sorted(_normalize_string_set(item.get("target_concept_ids")))) == expected_key[1]
            )
        ]
        if candidates:
            def score(candidate: dict[str, Any]) -> int:
                return sum(
                    candidate.get(field) == exp_candidate.get(field)
                    for field in ("polarity", "mode", "clause_scope")
                    if field in exp_candidate
                )
            selected = max(candidates, key=score)
            remaining.remove(selected)
            matched.append((exp, selected))
        else:
            matched.append((exp, None))
    return matched


def _compare_fields(result: dict[str, Any]) -> dict[str, Any]:
    expected = result["expected"]["contract"]
    observation = result["observation"]
    candidate_available = isinstance(observation.get("candidate"), dict)
    validator_available = isinstance(observation.get("contract"), dict)
    candidate = observation.get("candidate") if candidate_available else {}
    contract = observation.get("contract") if validator_available else {}
    comparisons: list[dict[str, Any]] = []

    def add(side: str, name: str, expected_value: Any, actual_value: Any, normalizer=None) -> None:
        expected_norm = normalizer(expected_value) if normalizer else expected_value
        available = candidate_available if side == "candidate" else validator_available
        actual_norm = (normalizer(actual_value) if normalizer else actual_value) if available else None
        comparisons.append({
            "side": side,
            "field": name,
            "expected": expected_norm,
            "actual": actual_norm,
            "passed": expected_norm == actual_norm if available else None,
        })

    for field in ("domain", "parse_status", "primary_intent"):
        if field in expected:
            add("candidate", field, expected[field], candidate.get(field))
    if "domain" in expected:
        pass
    if isinstance(expected.get("candidate"), dict):
        for field, exp_value in expected["candidate"].items():
            add("candidate", field, exp_value, candidate.get(field))
    validator_expected = expected.get("validator")
    if isinstance(validator_expected, dict):
        for field, exp_value in validator_expected.items():
            add("validator", field, exp_value, contract.get(field))
    if "semantic_status" in expected:
        add("validator", "semantic_status", expected["semantic_status"], contract.get("semantic_status"))

    frame_expected = expected.get("request_frame")
    if isinstance(frame_expected, dict):
        for field, exp_value in frame_expected.items():
            add("candidate", "request_frame." + field, exp_value, (candidate.get("request_frame") or {}).get(field))
    if "dialogue_act" in expected:
        frame = candidate.get("request_frame") or {}
        actual_dialogue = frame.get("dialogue_act_validated") or frame.get("dialogue_act_provisional")
        add("candidate", "dialogue_act", expected["dialogue_act"], actual_dialogue)

    expected_ops = expected.get("operations")
    if isinstance(expected_ops, list):
        actual_candidate_ops = candidate.get("operations") if isinstance(candidate.get("operations"), list) else []
        actual_contract_ops = contract.get("operations") if isinstance(contract.get("operations"), list) else []
        if len(expected_ops) != len(actual_candidate_ops):
            add("candidate", "operations_count", len(expected_ops), len(actual_candidate_ops))
        candidate_matches = _match_operations(expected_ops, actual_candidate_ops)
        contract_matches = _match_operations(expected_ops, actual_contract_ops)
        candidate_mentions = {
            str(item.get("id")): item
            for item in candidate.get("mentions", [])
            if isinstance(item, dict) and item.get("id")
        }
        if "dialogue_act" not in expected:
            legacy_dialogue = next(
                (item.get("dialogue_act") for item in expected_ops if item.get("dialogue_act")),
                None,
            )
            if legacy_dialogue is not None:
                frame = candidate.get("request_frame") or {}
                actual_dialogue = frame.get("dialogue_act_validated") or frame.get("dialogue_act_provisional")
                add("candidate", "dialogue_act", legacy_dialogue, actual_dialogue)
        for index, (expected_op, actual_op) in enumerate(candidate_matches):
            actual_op = actual_op or {}
            for field in ("kind", "target_concept_ids"):
                if field in expected_op:
                    normalizer = _normalize_string_set if field == "target_concept_ids" else None
                    add("candidate", f"operations[{index}].{field}", expected_op[field], actual_op.get(field), normalizer)
            exp_candidate = _candidate_expected_operation(expected_op)
            for field, exp_value in exp_candidate.items():
                if field == "mentioned_arguments":
                    add("candidate", f"operations[{index}].mentioned_arguments", exp_value, actual_op.get(field), lambda value: _normalize_argument_map(value, with_stance=True))
                elif field == "argument_negations":
                    add(
                        "candidate",
                        f"operations[{index}].argument_negations",
                        exp_value,
                        actual_op.get(field),
                        lambda value: _normalize_negations(value, candidate_mentions),
                    )
                else:
                    add("candidate", f"operations[{index}].{field}", exp_value, actual_op.get(field))

            expected_validator = _validator_expected_operation(expected_op)
            actual_contract_op = contract_matches[index][1] or {}
            for field, exp_value in expected_validator.items():
                if field == "effective_arguments":
                    add("validator", f"operations[{index}].effective_arguments", exp_value, actual_contract_op.get(field), lambda value: _normalize_argument_map(value, with_stance=False))
                elif field == "excluded_arguments":
                    add("validator", f"operations[{index}].excluded_arguments", exp_value, _actual_excluded_arguments(actual_op), lambda value: _normalize_argument_map(value, with_stance=False))
                else:
                    add("validator", f"operations[{index}].{field}", exp_value, actual_contract_op.get(field))

    scored = [item for item in comparisons if item["passed"] is not None]
    passed = sum(1 for item in scored if item["passed"])
    exact_match = (
        passed == len(scored)
        if scored and len(scored) == len(comparisons)
        else None
    )
    return {
        "comparisons": comparisons,
        "candidate_correct": sum(1 for item in scored if item["side"] == "candidate" and item["passed"]),
        "candidate_total": sum(1 for item in scored if item["side"] == "candidate"),
        "validator_correct": sum(1 for item in scored if item["side"] == "validator" and item["passed"]),
        "validator_total": sum(1 for item in scored if item["side"] == "validator"),
        "field_correct": passed,
        "field_total": len(scored),
        "exact_match": exact_match,
    }


def _score_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        result["scores"] = _compare_fields(result)
        contract = result["observation"].get("contract") or {}
        expected_status = result["expected"].get("status")
        actual_status = contract.get("semantic_status") if isinstance(contract, dict) else None
        result["scores"]["status_match"] = (
            expected_status == actual_status
            if expected_status is not None and actual_status is not None
            else None
        )
        result["scores"]["reviewed_status_match"] = (
            result["scores"]["status_match"] if result["label_status"] == "reviewed" else None
        )
        result["scores"]["draft_status_match"] = (
            result["scores"]["status_match"] if result["label_status"] == "draft" else None
        )


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return round(float(ordered[index]), 2)


def _metric_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = len(results)
    schema_valid = sum(item["observation"].get("status") in VALID_STATUSES for item in results)
    initial_statuses = [
        (item["observation"].get("repair") or {}).get("initial_status")
        or item["observation"].get("status")
        for item in results
    ]
    initial_schema_valid = sum(status in VALID_STATUSES for status in initial_statuses)
    repair_attempted = [
        item for item in results
        if (item["observation"].get("repair") or {}).get("attempted") is True
    ]
    repair_recovered = sum(
        (item["observation"].get("repair") or {}).get("recovered") is True
        for item in repair_attempted
    )
    # Keep parse/schema failures out of the span denominator: they never
    # produced a Candidate/Validator result and are covered by separate gates.
    span_reference_results = [
        item for item in results
        if item["observation"].get("status") in VALID_STATUSES
    ]
    span_valid = sum(
        not (set(item["observation"].get("issue_codes") or []) & SPAN_REFERENCE_ISSUES)
        for item in span_reference_results
    )
    reviewed = [
        item for item in results
        if item["label_status"] == "reviewed" and item["scores"]["reviewed_status_match"] is not None
    ]
    draft = [
        item for item in results
        if item["label_status"] == "draft" and item["scores"]["draft_status_match"] is not None
    ]
    fail_closed = [item for item in results if item["observation"].get("status") in FAIL_CLOSED_STATUSES]
    fail_closed_passed = sum(
        not item.get("exception_escaped")
        and (
            item["observation"].get("contract") is None
            or bool(item["observation"].get("issue_codes"))
        )
        for item in fail_closed
    )
    all_comparisons = [
        comparison
        for result in results
        for comparison in result["scores"]["comparisons"]
        if comparison["passed"] is not None
    ]
    candidate_fields = [item for item in all_comparisons if item["side"] == "candidate"]
    validator_fields = [item for item in all_comparisons if item["side"] == "validator"]
    latencies = [
        float(item["observation"].get("elapsed_ms"))
        for item in results
        if isinstance(item["observation"].get("elapsed_ms"), (int, float))
    ]
    input_tokens = [
        int(item["observation"]["input_tokens"])
        for item in results
        if isinstance(item["observation"].get("input_tokens"), int)
        and not isinstance(item["observation"].get("input_tokens"), bool)
    ]
    output_tokens = [
        int(item["observation"]["output_tokens"])
        for item in results
        if isinstance(item["observation"].get("output_tokens"), int)
        and not isinstance(item["observation"].get("output_tokens"), bool)
    ]
    costs = [
        float(item["observation"]["estimated_cost"])
        for item in results
        if isinstance(item["observation"].get("estimated_cost"), (int, float))
        and not isinstance(item["observation"].get("estimated_cost"), bool)
    ]
    cost_sources = Counter(
        str(item["observation"].get("cost_estimate_source") or "unknown")
        for item in results
    )
    provider_failure_sdk_attempts = sum(
        int(item["provider_failure"].get("attempt_count") or 0)
        for item in results if item.get("provider_failure")
    )
    provider_failure_sdk_cases = sum(
        int(item["provider_failure"].get("attempt_count") or 0) > 0
        for item in results if item.get("provider_failure")
    )
    provider_failure_no_sdk_cases = sum(
        int(item["provider_failure"].get("attempt_count") or 0) == 0
        for item in results if item.get("provider_failure")
    )
    exact_scored = [item["scores"]["exact_match"] for item in results if item["scores"]["exact_match"] is not None]
    exact_passed = sum(bool(value) for value in exact_scored)
    return {
        "attempted": attempted,
        "schema_valid_count": schema_valid,
        "schema_valid_rate": _ratio(schema_valid, attempted),
        "initial_schema_valid_count": initial_schema_valid,
        "initial_schema_valid_rate": _ratio(initial_schema_valid, attempted),
        "repair_attempt_count": len(repair_attempted),
        "repair_recovered_count": repair_recovered,
        "repair_recovery_rate": _ratio(repair_recovered, len(repair_attempted)),
        "initial_status_distribution": dict(sorted(Counter(
            f"{status}" for status in initial_statuses
        ).items())),
        "span_reference_valid_count": span_valid,
        "span_reference_result_count": len(span_reference_results),
        "span_reference_valid_rate": _ratio(span_valid, len(span_reference_results)),
        "reviewed_status_count": len(reviewed),
        "reviewed_status_accuracy": _ratio(sum(bool(item["scores"]["reviewed_status_match"]) for item in reviewed), len(reviewed)),
        "draft_status_count": len(draft),
        "draft_status_match_rate": _ratio(sum(bool(item["scores"]["draft_status_match"]) for item in draft), len(draft)),
        "candidate_field_correct": sum(item["passed"] for item in candidate_fields),
        "candidate_field_total": len(candidate_fields),
        "candidate_field_accuracy": _ratio(sum(item["passed"] for item in candidate_fields), len(candidate_fields)),
        "validator_field_correct": sum(item["passed"] for item in validator_fields),
        "validator_field_total": len(validator_fields),
        "validator_field_accuracy": _ratio(sum(item["passed"] for item in validator_fields), len(validator_fields)),
        "field_correct": sum(item["passed"] for item in all_comparisons),
        "field_total": len(all_comparisons),
        "field_accuracy": _ratio(sum(item["passed"] for item in all_comparisons), len(all_comparisons)),
        "exact_match_count": exact_passed,
        "exact_match_total": len(exact_scored),
        "exact_match_rate": _ratio(exact_passed, len(exact_scored)),
        "fail_closed_count": len(fail_closed),
        "fail_closed_passed": fail_closed_passed,
        "fail_closed_rate": _ratio(fail_closed_passed, len(fail_closed)),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "input_tokens_total": sum(input_tokens),
        "input_tokens_mean": round(statistics.mean(input_tokens), 2) if input_tokens else None,
        "input_tokens_known_count": len(input_tokens),
        "output_tokens_total": sum(output_tokens),
        "output_tokens_mean": round(statistics.mean(output_tokens), 2) if output_tokens else None,
        "output_tokens_known_count": len(output_tokens),
        "estimated_cost_total": round(sum(costs), 8) if costs else None,
        "estimated_cost_mean": round(statistics.mean(costs), 8) if costs else None,
        "estimated_cost_known_count": len(costs),
        "cost_estimate_source_distribution": dict(sorted(cost_sources.items())),
        "status_reason_distribution": dict(sorted(Counter(
            f"{item['observation'].get('status')}:{item['observation'].get('reason_code') or 'none'}"
            for item in results
        ).items())),
        "issue_code_distribution": dict(sorted(Counter(
            code for item in results for code in (item["observation"].get("issue_codes") or [])
        ).items())),
        "provider_failure_distribution": dict(sorted(Counter(
            item["provider_failure"]["code"]
            for item in results if item.get("provider_failure")
        ).items())),
        "provider_failure_sdk_attempt_count": provider_failure_sdk_attempts,
        "provider_failure_sdk_attempt_case_count": provider_failure_sdk_cases,
        "provider_failure_no_sdk_attempt_case_count": provider_failure_no_sdk_cases,
    }


def _group_summaries(results: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        groups[str(item.get(field) or "unspecified")].append(item)
    return {key: _metric_summary(value) for key, value in sorted(groups.items())}


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
    return str(value)


def _summary_table(summary: dict[str, Any], *, label: str = "总计") -> str:
    return (
        f"| {label} | {summary['attempted']} | "
        f"{_fmt(summary['schema_valid_rate'])} ({summary['schema_valid_count']}) | "
        f"{_fmt(summary['span_reference_valid_rate'])} "
        f"({summary['span_reference_valid_count']}/{summary['span_reference_result_count']}) | "
        f"{_fmt(summary['reviewed_status_accuracy'])} ({summary['reviewed_status_count']}) | "
        f"{_fmt(summary['draft_status_match_rate'])} ({summary['draft_status_count']}) | "
        f"{_fmt(summary['candidate_field_accuracy'])} ({summary['candidate_field_correct']}/{summary['candidate_field_total']}) | "
        f"{_fmt(summary['validator_field_accuracy'])} ({summary['validator_field_correct']}/{summary['validator_field_total']}) | "
        f"{_fmt(summary['exact_match_rate'])} ({summary['exact_match_count']}/{summary['exact_match_total']}) | "
        f"{_fmt(summary['fail_closed_rate'])} ({summary['fail_closed_passed']}/{summary['fail_closed_count']}) | "
        f"{_fmt(summary['latency_p50_ms'])}/{_fmt(summary['latency_p95_ms'])} | "
        f"{summary['input_tokens_total']}/{summary['output_tokens_total']} | "
        f"{_fmt(summary['estimated_cost_total'])}/{_fmt(summary['estimated_cost_mean'])} |"
    )


def _write_review_sheet(
    results: list[dict[str, Any]], path: Path, prompt_version: str = "v2",
) -> None:
    draft_results = [item for item in results if item.get("label_status") == "draft"]
    lines = [
        f"# Semantic Interpreter {prompt_version} 离线评测人工复核表",
        "",
        f"状态：**AI 初筛，待人工确认**。本表只包含 `{len(draft_results)}` 条 `draft`；已有 reviewed 标签不会再次进入待复核表。请填写下方人工复核字段；不要只把本表里的 `draft` 改成 `reviewed`，正式 reviewed 标签由决策文件合并生成。",
        "",
        "请独立判断，不要直接照抄 draft/期望标签。每条记录需要人工填写 reviewed_status（CONFIRMED、AMBIGUOUS、UNSUPPORTED、MISSING、CONFLICT）、reviewed_operations（数组，每项含 kind、target_concept_ids、polarity；无 operation 时填 []）、reviewed_clarification_required（true/false）、reviewer、带时区的 reviewed_at 和 notes。复核完成后交回本表，我会按 `scripts/merge_semantic_interpreter_reviews.py` 的协议生成决策文件并运行 gate。",
        "",
        "建议错误类别：span 错误、concept 错误、否定/作用域错误、参数否定错误、歧义判定错误、schema 错误、provider/超时。",
        "",
    ]
    for item in draft_results:
        observation = item.get("observation") or {}
        lines.extend([
            f"## {item['case_id']} — {item['category']} ({item['label_status']})",
            "",
            f"- 数据集：`{item['dataset']}`",
            f"- 文本：{item['text']}",
            f"- 期望：`{json.dumps(item['expected'], ensure_ascii=False, sort_keys=True)}`",
            f"- Interpreter status / reason：`{observation.get('status')}` / `{observation.get('reason_code')}`",
            f"- Schema errors：`{json.dumps(observation.get('schema_errors') or [], ensure_ascii=False, sort_keys=True)}`",
            f"- Candidate：`{json.dumps(observation.get('candidate'), ensure_ascii=False, sort_keys=True)}`",
            f"- Contract：`{json.dumps(observation.get('contract'), ensure_ascii=False, sort_keys=True)}`",
            f"- Issue codes：`{json.dumps(observation.get('issue_codes') or [], ensure_ascii=False)}`",
            f"- Provider failure：`{json.dumps(item.get('provider_failure'), ensure_ascii=False, sort_keys=True)}`",
            f"- AI 初筛建议错误类别：{_suggest_error_category(item)}。",
            "- Human reviewed_status / reviewed_operations / reviewed_clarification_required：待填写。",
            "- Reviewer / reviewed_at（含时区）：待填写。",
            f"- Human notes：待填写。原始标注说明：{item.get('notes') or '无'}",
            "",
        ])
        diagnostic = observation.get("raw_output_diagnostic")
        if isinstance(diagnostic, dict):
            diagnostic_json = json.dumps(
                diagnostic, ensure_ascii=False, sort_keys=True,
            ).replace("`", "\\u0060")
            lines[-1:-1] = [
                f"- Raw output diagnostic：`{diagnostic_json}`",
                "",
            ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _suggest_error_category(result: dict[str, Any]) -> str:
    observation = result["observation"]
    status = observation.get("status")
    if result.get("provider_failure") or status == "unavailable":
        return "provider/超时"
    if status in {"parse_failure", "schema_invalid"}:
        return "schema 错误"
    issues = set(observation.get("issue_codes") or [])
    if issues & SPAN_REFERENCE_ISSUES:
        return "span 错误"
    if any("concept" in code or "mention" in code or "reference" in code for code in issues):
        return "concept 错误"
    mismatches = [
        item["field"]
        for item in result.get("scores", {}).get("comparisons", [])
        if item.get("passed") is False
    ]
    if any("argument_negations" in field or "effective_arguments" in field or "excluded_arguments" in field for field in mismatches):
        return "参数否定错误"
    if any("polarity" in field or "clause_scope" in field for field in mismatches):
        return "否定/作用域错误"
    if any("dialogue_act" in field or "parse_status" in field or "semantic_status" in field for field in mismatches):
        return "歧义判定错误"
    if any("target_concept_ids" in field or "domain" in field for field in mismatches):
        return "concept 错误"
    if mismatches:
        return "歧义判定错误"
    return "未发现机器初筛偏差；待人工确认"


def _write_report(
    *, results: list[dict[str, Any]], meta: dict[str, Any], output_path: Path,
    report_path: Path, review_sheet_path: Path, stop_reason: str,
) -> None:
    total = _metric_summary(results)
    datasets = _group_summaries(results, "dataset")
    categories = _group_summaries(results, "category")
    actual_category_counts = Counter(item["category"] for item in results)
    loaded_category_counts = Counter(meta.get("loaded_category_counts") or actual_category_counts)
    unrun_category_counts = {
        category: count - actual_category_counts.get(category, 0)
        for category, count in sorted(loaded_category_counts.items())
        if count > actual_category_counts.get(category, 0)
    }
    reviewed_count = sum(item["label_status"] == "reviewed" for item in results)
    draft_count = sum(item["label_status"] == "draft" for item in results)
    token_count = total["input_tokens_total"] + total["output_tokens_total"]
    lines = [
        f"# Semantic Interpreter {meta.get('environment', {}).get('prompt_version') or 'v2'} Shadow 离线评测报告",
        "",
        f"- 生成时间（UTC）：`{meta['created_at_utc']}`",
        f"- 被评测代码快照：`{meta.get('evaluation_source_snapshot_commit') or meta.get('commit') or 'unknown'}`；运行时 HEAD：`{meta.get('runtime_head_at_run') or meta.get('commit') or 'unknown'}`；评测时工作树 dirty：`{meta.get('runtime_worktree_dirty_at_run', 'unknown')}`；快照是否评测后提交：`{meta.get('source_snapshot_commit_created_after_run', 'unknown')}`",
        f"- 运行说明：{meta.get('evaluation_execution_note', '无额外说明。')}",
        f"- 评测后代码变更：{meta.get('post_evaluation_code_note', '无记录；以被评测代码快照为准。')}",
        f"- 运行模式：`{meta['environment']['mode']}`；raw question opt-in：`{meta['environment']['raw_question_opt_in']}`；sample rate：`{meta['environment']['sample_rate']}`",
        f"- Provider：已配置，host `{meta['environment']['provider_host']}`；模型：`{meta['environment']['model']}`；交互策略：`{meta['environment']['interaction_policy']}`（{meta['environment']['transport']}/{meta['environment']['thinking']}，connect/idle/total `{meta['environment']['connect_timeout_seconds']}/{meta['environment']['idle_timeout_seconds']}/{meta['environment']['total_timeout_seconds']}s`）；输出上限：`{meta['environment']['max_tokens']} tokens`",
        f"- 兼容配置 `semantic_interpreter_timeout_seconds={meta['environment']['legacy_timeout_seconds_ignored_by_policy_path']}s` 保留，但 policy 调用不使用它。",
        f"- 每请求重试与 provider fallback：`retry_before_first_token={(meta.get('environment') or {}).get('retry_before_first_token', 'unspecified')}`，`provider_fallback_enabled={(meta.get('environment') or {}).get('provider_fallback_enabled', 'unspecified')}`。",
        f"- Prompt：`{meta['environment']['prompt_version']}`；system prompt SHA-256：`{meta['environment']['prompt_sha256']}`",
        f"- 校验修复回合：`{meta.get('validation_repair_enabled', False)}`；prompt version：`{meta.get('environment', {}).get('repair_prompt_version', 'disabled')}`；prompt SHA-256：`{meta.get('environment', {}).get('repair_prompt_sha256') or 'disabled'}`；本离线评测 runner 明确使用 shadow 模式，每条最多 1 次。在线 active 是否修复由 `SEMANTIC_INTERPRETER_VALIDATION_REPAIR_ENABLED` 控制。原始首轮与修复后的结果分别统计。",
        f"- Catalog snapshot SHA-256：`{meta['environment']['catalog_snapshot_hash']}`",
        f"- 定价版本：`{meta['environment']['price_version']}`；输入/输出：`¥{meta['environment']['input_cost_per_1k']}/1K` / `¥{meta['environment']['output_cost_per_1k']}/1K`",
        f"- 数据集文件 SHA-256：`{json.dumps(meta['dataset_sha256'], ensure_ascii=False, sort_keys=True)}`",
        f"- 读取去重后 `{meta['loaded_case_count']}` 条，实际评测 `{len(results)}` 条；停止原因：`{stop_reason}`。原始结果：`{output_path.as_posix()}`",
        f"- 标签状态：`{draft_count}` draft，`{reviewed_count}` reviewed。Reviewed 标签保留数据集声明的来源；draft 匹配率仅作初筛，不作为验收准确率。",
        "",
        "## 单例运行观测",
        "",
        "| 用例 | status | policy | transport | thinking | first_token_ms | usage_status | timeout_kind | elapsed_ms | request_options |",
        "|---|---|---|---|---|---:|---|---|---:|---|",
    ]
    for item in results:
        observation = item.get("observation") or {}
        options = observation.get("request_options")
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                item.get("case_id", ""), observation.get("status", ""),
                observation.get("interaction_policy", ""),
                (options or {}).get("transport", ""),
                (options or {}).get("thinking", ""),
                observation.get("first_token_ms"), observation.get("usage_status", "unknown"),
                observation.get("timeout_kind"), observation.get("elapsed_ms"),
                json.dumps(options, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if options else "—",
            )
        )
    repaired_items = [
        item for item in results
        if isinstance((item.get("observation") or {}).get("repair"), Mapping)
        and (item.get("observation") or {})["repair"].get("attempted") is True
    ]
    lines.extend([
        "",
        "## 校验修复调用明细",
        "",
        "只记录状态码和 issue code，不记录原始模型输出或 Candidate 文本。首轮失败保留在明细中，即使修复后最终有效。",
        "",
        "| 用例 | 首轮状态 | 首轮原因 | 最终状态 | 修复成功 | issue codes |",
        "|---|---|---|---|---|---|",
    ])
    for item in repaired_items:
        observation = item.get("observation") or {}
        repair = observation.get("repair") or {}
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                item.get("case_id", ""), repair.get("initial_status", ""),
                repair.get("initial_reason_code", ""), repair.get("final_status", ""),
                repair.get("recovered", False),
                json.dumps(repair.get("initial_issue_codes") or [], ensure_ascii=False),
            )
        )
    if not repaired_items:
        lines.append("| — | — | — | — | — | 本次没有触发修复 |")
    diagnostics_by_case = {
        item.get("case_id"): item
        for item in results
        if item.get("case_id") in PARSE_FAILURE_DIAGNOSTIC_CASE_IDS
    }
    lines.extend([
        "",
        "## 指定 parse_failure 样例原始输出诊断",
        "",
        "仅对 allowlist 中样例的首轮 parse_failure 记录；即使一次修复后恢复，也保留首轮的字符数、SHA-256 和最多 500 字符截断文本。",
        "",
    ])
    for case_id in sorted(PARSE_FAILURE_DIAGNOSTIC_CASE_IDS):
        item = diagnostics_by_case.get(case_id)
        observation = (item or {}).get("observation") or {}
        if not item:
            lines.append(f"- `{case_id}`：本次评测未包含该样例。")
            continue
        repair = observation.get("repair") if isinstance(observation.get("repair"), Mapping) else {}
        initial_status = repair.get("initial_status") if repair.get("attempted") is True else observation.get("status")
        if initial_status != "parse_failure":
            lines.append(
                f"- `{case_id}`：首轮 status=`{initial_status}`、"
                f"reason=`{repair.get('initial_reason_code') or observation.get('reason_code')}`，未触发 parse_failure 文本捕获。"
            )
            continue
        diagnostic = (
            repair.get("initial_raw_output_diagnostic")
            if repair.get("attempted") is True else observation.get("raw_output_diagnostic")
        ) or {}
        lines.append(
            f"- `{case_id}`：capture=`{diagnostic.get('capture_status', 'unavailable')}`；"
            f"raw_output_chars=`{diagnostic.get('raw_output_chars', 'n/a')}`；"
            f"sha256=`{diagnostic.get('raw_output_sha256', 'n/a')}`；"
            f"truncated=`{diagnostic.get('truncated', 'n/a')}`。"
        )
        excerpt = diagnostic.get("truncated_text")
        if isinstance(excerpt, str):
            escaped_excerpt = json.dumps(excerpt, ensure_ascii=False).replace("`", "\\u0060")
            lines.append(f"  - 截断文本（JSON 字符串）：`{escaped_excerpt}`")
    lines.extend([
        "",
        "## 指标定义与总计",
        "",
        "schema 合法率按最终 observation 的 `ok` / `validator_issues` 计；若启用 repair，首轮 schema-valid 另行报告。span/引用合法率只在有 Candidate/Validator 结果且无 span/reference 类 issue 时计为合法。semantic_status 验收准确率只统计 `reviewed` 标签；Candidate 与 Validator 字段比较基于现有标签，draft 只能看作 AI 初筛，不能当成 gold。fail-closed 率统计修复后仍未恢复的 parse/schema/validator/unavailable 结果；首轮失败和恢复次数另行报告。延迟分位数用 nearest-rank。",
        "",
        "| 分组 | 用例数 | Schema 合法率 (n) | Span/引用合法率 (n) | Reviewed status 准确率 (n) | Draft status 初筛匹配 (n) | Candidate 字段准确率 | Validator 字段准确率 | Exact match (n) | Fail-closed (n) | p50/p95 ms | input/output tokens | 成本总计/均值 CNY |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _summary_table(total),
        "",
        "## 按数据集",
        "",
        "| 数据集 | 用例数 | Schema 合法率 (n) | Span/引用合法率 (n) | Reviewed status 准确率 (n) | Draft status 初筛匹配 (n) | Candidate 字段准确率 | Validator 字段准确率 | Exact match (n) | Fail-closed (n) | p50/p95 ms | input/output tokens | 成本总计/均值 CNY |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, summary in datasets.items():
        lines.append(_summary_table(summary, label=name))
    lines.extend([
        "",
        "## 按类别",
        "",
        "| 类别 | 用例数 | Schema 合法率 (n) | Span/引用合法率 (n) | Reviewed status 准确率 (n) | Draft status 初筛匹配 (n) | Candidate 字段准确率 | Validator 字段准确率 | Exact match (n) | Fail-closed (n) | p50/p95 ms | input/output tokens | 成本总计/均值 CNY |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, summary in categories.items():
        lines.append(_summary_table(summary, label=name))
    lines.extend([
        "",
        "## 覆盖、错误分布与门槛",
        "",
        f"- 全量数据集类别：`{json.dumps(dict(sorted(loaded_category_counts.items())), ensure_ascii=False)}`。实际评测类别：`{json.dumps(dict(sorted(actual_category_counts.items())), ensure_ascii=False)}`。",
        f"- 未评测类别/样例数：`{json.dumps(unrun_category_counts, ensure_ascii=False)}`。因 token 预算提前停止的类别不会被记为通过。",
        f"- 状态/原因分布：`{json.dumps(total['status_reason_distribution'], ensure_ascii=False, sort_keys=True)}`。",
        f"- 首轮 schema-valid：`{total['initial_schema_valid_count']}/{total['attempted']}`；修复调用 `{total['repair_attempt_count']}` 次，成功恢复 `{total['repair_recovered_count']}` 次；恢复率 `{_fmt(total['repair_recovery_rate'])}`。状态分布：`{json.dumps(total['initial_status_distribution'], ensure_ascii=False, sort_keys=True)}`。gate 使用最终结果，首轮失败不会从本统计中隐藏。",
        f"- Validator issue 分布：`{json.dumps(total['issue_code_distribution'], ensure_ascii=False, sort_keys=True)}`。",
        f"- Provider failure 分布：`{json.dumps(total['provider_failure_distribution'], ensure_ascii=False, sort_keys=True)}`。是否重试、是否切换 provider 由所选 InteractionPolicy 和统一 provider 配置决定。",
        f"- Provider failure 调用明细：失败记录中共 `{total['provider_failure_sdk_attempt_count']}` 次 SDK attempt，覆盖 `{total['provider_failure_sdk_attempt_case_count']}` 条用例；另有 `{total['provider_failure_no_sdk_attempt_case_count']}` 条失败记录没有 SDK attempt（例如本地拒绝或 circuit-open）。该计数仅覆盖失败记录，不代表成功调用总数。",
        f"- 成本来源分布：`{json.dumps(total['cost_estimate_source_distribution'], ensure_ascii=False, sort_keys=True)}`。若实际 usage/配置价格缺失，逐例会附公开价参考估算并标明来源。",
        f"- 已知 token 总计 `{token_count}` / 上限 `{meta.get('max_total_tokens', MAX_TOTAL_TOKENS)}`；输入均值 `{_fmt(total['input_tokens_mean'])}`，输出均值 `{_fmt(total['output_tokens_mean'])}`。成本统计按调用记录；当前配置价格单位按人民币计。参考公开价格以实际 provider region 为准：[阿里云百炼模型价格](https://help.aliyun.com/zh/model-studio/model-pricing)。",
        f"- Phase 状态：本报告只记录 shadow Interpreter 评测，不单独证明 Router 行为验收或授权 active 路由；Phase 2 离线 Router gate 需单独运行并记录。`reviewed` semantic_status 准确率为 `{_fmt(total['reviewed_status_accuracy'])}`（n=`{total['reviewed_status_count']}`）；draft 指标不是人工 gold。",
        f"- 剩余人工复核表：`{review_sheet_path.as_posix()}`，只含 `{draft_count}` 条 draft。",
        "- 安全边界：本次只调用 shadow Interpreter 并记录 Candidate/Validator observation；不调用在线 Router、selector、计划或模板执行。输入文本均为仓库内合成/对抗样例。",
        "",
        "## 未验证与限制",
        "",
        f"- 本次有 `{reviewed_count}` 条 reviewed gold，其余 `{draft_count}` 条仍是 draft；不把 draft 匹配率宣称为准确率或 Phase 2 路由行为验收结果。",
        "- 若 provider 不返回 usage，相关 token/cost 记录保持未知；成本参考不代替 provider 账单。",
        "- 当前 Validator 输出没有独立的 `excluded_arguments` 字段；字段评测从 Candidate 的 EXCLUDED mention 推导该投影，并单列为 Validator 侧对照。",
        "- RFM basis 标签表达已冻结的“两种不同有效 basis”产品规则；本次 Interpreter shadow 不做 Router 决策，语义状态/字段不能证明模板能力匹配正确。",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", help="JSONL dataset; repeat to load multiple files")
    parser.add_argument("--limit", type=int, help=f"Evaluate at most N cases (hard maximum {MAX_CASES})")
    parser.add_argument(
        "--max-total-tokens", type=int, default=MAX_TOTAL_TOKENS,
        help=f"Cumulative token reserve/stop limit (default {MAX_TOTAL_TOKENS}, hard maximum {MAX_TOTAL_TOKENS_HARD_MAX})",
    )
    parser.add_argument("--output", help="Raw JSON result path (defaults under ignored data/evaluations)")
    parser.add_argument("--report", help="Markdown report path (defaults to the existing Phase 1 report)")
    parser.add_argument("--review-sheet", help="Markdown review sheet path (defaults to the existing Phase 1 review sheet)")
    parser.add_argument("--model", default="qwen3.8-flash", help="Configured interpreter model")
    parser.add_argument("--allow-real-provider", action="store_true", help="Acknowledge that this run will send synthetic cases to the configured provider")
    parser.add_argument("--no-provider-retries", action="store_true", help="Disable pre-first-token retries and provider fallback for every case")
    parser.add_argument(
        "--validation-repair", action="store_true",
        help="Allow one shadow-only repair request after parse/schema/Validator failure; still fail closed if it does not validate",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config/datasets and print the run plan without provider requests")
    parser.add_argument("--report-only", help="Regenerate report/review sheet from an existing raw result JSON; sends no provider requests")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= MAX_CASES:
        parser.error(f"--limit must be between 1 and {MAX_CASES}")
    if not 1 <= args.max_total_tokens <= MAX_TOTAL_TOKENS_HARD_MAX:
        parser.error(f"--max-total-tokens must be between 1 and {MAX_TOTAL_TOKENS_HARD_MAX}")
    if not args.model.strip():
        parser.error("--model must not be empty")
    try:
        if args.report_only:
            result_path = Path(args.report_only).resolve()
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise ValueError("--report-only input must contain a results array")
            results = payload["results"]
            if not payload.get("meta", {}).get("loaded_category_counts"):
                referenced_paths = []
                for raw_path in payload.get("meta", {}).get("dataset_sha256", {}):
                    path = Path(str(raw_path).replace("\\", "/"))
                    if not path.is_absolute():
                        path = ROOT / path
                    if path.is_file():
                        referenced_paths.append(path)
                if referenced_paths:
                    loaded_cases, _ = _load_cases(referenced_paths)
                    payload["meta"]["loaded_category_counts"] = dict(sorted(Counter(
                        str(case.get("category") or "unspecified") for case in loaded_cases
                    ).items()))
            _score_results(results)
            payload["meta"]["postprocessed_at_utc"] = datetime.now(timezone.utc).isoformat()
            result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            report_path = Path(args.report).resolve() if args.report else ROOT / REPORT_PATH
            review_sheet_path = Path(args.review_sheet).resolve() if args.review_sheet else ROOT / REVIEW_SHEET_PATH
            _write_review_sheet(
                results,
                review_sheet_path,
                str(payload.get("meta", {}).get("environment", {}).get("prompt_version") or "v2"),
            )
            _write_report(
                results=results,
                meta=payload["meta"],
                output_path=result_path,
                report_path=report_path,
                review_sheet_path=review_sheet_path,
                stop_reason=str(payload["meta"].get("stop_reason") or "unknown"),
            )
            print(json.dumps({
                "provider_requests": 0,
                "result_count": len(results),
                "raw_result": str(result_path),
                "report": str(report_path),
                "review_sheet": str(review_sheet_path),
            }, ensure_ascii=False, indent=2))
            return 0
        environment = _preflight(
            allow_real_provider=args.allow_real_provider,
            model=args.model.strip(),
            dry_run=args.dry_run,
            no_provider_retries=args.no_provider_retries,
        )
        evaluation_policy = _evaluation_interaction_policy(
            environment["interaction_policy"],
            no_provider_retries=args.no_provider_retries,
        )
        dataset_paths = [
            (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
            for path in (args.dataset or list(DEFAULT_DATASETS))
        ]
        for path in dataset_paths:
            if not path.is_file():
                raise ValueError(f"Dataset not found: {path}")
        cases, dataset_hashes = _load_cases(dataset_paths)
        loaded_case_count = len(cases)
        loaded_category_counts = dict(sorted(Counter(
            str(case.get("category") or "unspecified") for case in cases
        ).items()))
        if args.limit is None and loaded_case_count > MAX_CASES:
            raise ValueError(f"Loaded {loaded_case_count} unique cases, above hard maximum {MAX_CASES}; pass --limit")
        if args.limit is not None:
            cases = cases[: args.limit]
        elif len(cases) > MAX_CASES:
            cases = cases[:MAX_CASES]
        from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG
        from app.llm.semantic_interpreter_prompt import (
            PROMPT_VERSION, REPAIR_PROMPT_VERSION,
            build_repair_system_prompt, build_system_prompt,
        )

        system_prompt = build_system_prompt(DEFAULT_SEMANTIC_CATALOG)
        environment["prompt_version"] = PROMPT_VERSION
        environment["prompt_sha256"] = _sha256_bytes(system_prompt.encode("utf-8"))
        environment["repair_prompt_version"] = (
            REPAIR_PROMPT_VERSION if args.validation_repair else None
        )
        environment["repair_prompt_sha256"] = (
            _sha256_bytes(build_repair_system_prompt(DEFAULT_SEMANTIC_CATALOG).encode("utf-8"))
            if args.validation_repair else None
        )
        environment["catalog_snapshot_hash"] = DEFAULT_SEMANTIC_CATALOG.snapshot_hash()
        if args.dry_run:
            print(json.dumps({
                "dry_run": True,
                "provider_requests": 0,
                "case_count": len(cases),
                "categories": dict(sorted(Counter(case.get("category", "unspecified") for case in cases).items())),
                "environment": environment,
                "dataset_sha256": dataset_hashes,
                "max_total_tokens": args.max_total_tokens,
                "validation_repair_enabled": bool(args.validation_repair),
                "max_cases": MAX_CASES,
                "inter_case_delay_seconds": INTER_CASE_DELAY_SECONDS,
            }, ensure_ascii=False, indent=2))
            return 0

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = Path(args.output).resolve() if args.output else ROOT / "data" / "evaluations" / f"semantic_interpreter_v2_{stamp}.json"
        report_path = Path(args.report).resolve() if args.report else ROOT / REPORT_PATH
        review_sheet_path = Path(args.review_sheet).resolve() if args.review_sheet else ROOT / REVIEW_SHEET_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise ValueError(f"Refusing to overwrite existing output: {output_path}")

        evaluation_source_commit = _git_value("commit")
        evaluation_worktree_dirty = _git_worktree_dirty()

        results: list[dict[str, Any]] = []
        token_total = 0
        token_reserve_total = 0
        stop_reason = "completed"
        # Reserve an approximate input allowance plus the full configured output
        # cap before each send. Actual usage is reconciled after every response.
        for index, case in enumerate(cases):
            from app.llm.semantic_interpreter_prompt import build_request

            request = build_request(case["text"], case["case_id"], catalog=DEFAULT_SEMANTIC_CATALOG)
            estimated_input_reserve = math.ceil((len(system_prompt) + len(request)) / 2)
            max_output_reserve = environment["max_tokens"]
            primary_reserve = estimated_input_reserve + max_output_reserve
            repair_reserve = 0
            repair_allowed = False
            if args.validation_repair:
                repair_request = build_request(
                    case["text"], case["case_id"], catalog=DEFAULT_SEMANTIC_CATALOG,
                    repair_feedback={
                        "previous_status": "schema_invalid",
                        "reason_code": "bounded_repair_budget_reservation",
                        "schema_errors": [{"loc": ["operations", 1], "type": "value_error", "msg": "bounded diagnostic"}],
                        "previous_candidate": {"candidate": "x" * 11_900},
                        "previous_output_excerpt": "x" * 2_000,
                    },
                )
                repair_input_reserve = math.ceil((
                    len(build_repair_system_prompt(DEFAULT_SEMANTIC_CATALOG)) + len(repair_request)
                ) / 2)
                repair_reserve = repair_input_reserve + max_output_reserve
                repair_allowed = (
                    max(token_total, token_reserve_total)
                    + primary_reserve + repair_reserve <= args.max_total_tokens
                )
            if max(token_total, token_reserve_total) + primary_reserve > args.max_total_tokens:
                stop_reason = "token_budget_reserve_before_next_request"
                break
            result = _run_case(
                case, model=args.model.strip(), interaction_policy=evaluation_policy,
                validation_repair=repair_allowed,
            )
            _score_results([result])
            results.append(result)
            token_total += _known_token_total(result["observation"])
            token_reserve_total += primary_reserve
            if (result["observation"].get("repair") or {}).get("attempted") is True:
                token_reserve_total += repair_reserve
            print(json.dumps({
                "completed": len(results),
                "case_id": case["case_id"],
                "status": result["observation"].get("status"),
                "reason_code": result["observation"].get("reason_code"),
                "tokens_so_far": token_total,
            }, ensure_ascii=False), flush=True)
            if token_total >= args.max_total_tokens:
                stop_reason = "token_budget_reached_after_response"
                break
            if index + 1 < len(cases):
                time.sleep(INTER_CASE_DELAY_SECONDS)

        created_at = datetime.now(timezone.utc).isoformat()
        meta = {
            "created_at_utc": created_at,
            "branch": _git_value("branch"),
            "commit": evaluation_source_commit,
            "evaluation_source_snapshot_commit": evaluation_source_commit,
            "runtime_head_at_run": evaluation_source_commit,
            "runtime_worktree_dirty_at_run": evaluation_worktree_dirty,
            "source_snapshot_commit_created_after_run": False,
            "loaded_case_count": len(_load_cases(dataset_paths)[0]),
            "loaded_category_counts": loaded_category_counts,
            "environment": environment,
            "dataset_sha256": dataset_hashes,
            "max_cases": MAX_CASES,
            "max_total_tokens": args.max_total_tokens,
            "validation_repair_enabled": bool(args.validation_repair),
            "inter_case_delay_seconds": INTER_CASE_DELAY_SECONDS,
            "allow_real_provider": bool(args.allow_real_provider),
            "result_count": len(results),
            "stop_reason": stop_reason,
            "token_total_observed": token_total,
            "token_total_reserved": token_reserve_total,
            "output_path": str(output_path.relative_to(ROOT)) if output_path.is_relative_to(ROOT) else str(output_path),
        }
        payload = {"meta": meta, "results": results}
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_review_sheet(results, review_sheet_path, environment["prompt_version"])
        _write_report(
            results=results, meta=meta, output_path=output_path,
            report_path=report_path, review_sheet_path=review_sheet_path,
            stop_reason=stop_reason,
        )
        print(json.dumps({
            "result_count": len(results),
            "token_total": token_total,
            "stop_reason": stop_reason,
            "output": str(output_path),
            "report": str(report_path),
            "review_sheet": str(review_sheet_path),
        }, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def _git_value(name: str) -> str | None:
    import subprocess

    args = ["git", "rev-parse", "--abbrev-ref", "HEAD"] if name == "branch" else ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_worktree_dirty() -> bool | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
