"""Semantic Interpreter boundary for shadow and active routed requests.

Shadow output remains observational. Active interpretation follows the
configured tenant/template enrollment, which now defaults to all tenants.
"""

from __future__ import annotations

import logging
import json
from typing import Any

from app.context.semantic_interpreter import interpret_semantics
from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG
from app.context.semantic_contract_v2 import SemanticContractV21
from app.core.config import get_settings
from app.graph.state import AgentState

LOGGER = logging.getLogger(__name__)

_SEMANTIC_CONTEXT_SLOT_NAMES = frozenset({
    "metric", "dimension", "dimensions", "date", "period",
    "baseline_date", "current_date", "comparison_period", "operation",
    "operation_intent",
})
_SEMANTIC_CONTEXT_SLOT_SOURCES = frozenset({
    "current_turn", "current_turn_correction", "previous_turn_context",
    "current_explorer_scope", "previous_result_context",
    "current_query_evidence", "task_state",
})


def _prior_semantic_context(previous: Any) -> dict[str, Any] | None:
    """Select only a previously validated semantic contract, never prose/scope."""

    if not isinstance(previous, dict):
        return None
    prior_contract = previous.get("semantic_contract")
    if not isinstance(prior_contract, dict):
        turn_context = previous.get("turn_context")
        prior_contract = turn_context.get("semantic_contract") if isinstance(turn_context, dict) else None
    if not isinstance(prior_contract, dict):
        return None
    try:
        validated_prior = SemanticContractV21.model_validate_json(
            json.dumps(prior_contract, ensure_ascii=False, separators=(",", ":"))
        )
    except Exception:  # noqa: BLE001 - untrusted or stale history is omitted
        return None
    if validated_prior.catalog_snapshot_hash != DEFAULT_SEMANTIC_CATALOG.snapshot_hash():
        return None
    selected: dict[str, Any] = {
        "semantic_contract": validated_prior.model_dump(mode="json"),
        "trusted_prior_contract": validated_prior,
    }
    source_turn_id = previous.get("source_turn_id")
    if isinstance(source_turn_id, str):
        selected["source_turn_id"] = source_turn_id
    return selected


def _resolved_turn_context(state: AgentState) -> dict[str, Any] | None:
    """Project compiler-resolved slots for semantic parsing, without raw history.

    The prompt layer applies a second allowlist before serialization. Result
    facts, tool evidence, seller/scope data, permissions, and raw prose are
    deliberately excluded.
    """

    compilation = state.get("context_compilation")
    if not isinstance(compilation, dict):
        return None
    previous = state.get("previous_turn_context")
    previous = previous if isinstance(previous, dict) else {}
    boundary = state.get("authorization_boundary")
    boundary = boundary if isinstance(boundary, dict) else {}
    current_scope = str(
        state.get("current_query_scope_fingerprint")
        or boundary.get("query_scope_fingerprint")
        or ""
    )
    previous_scope = str(previous.get("scope_fingerprint") or "")
    previous_turn_id = str(previous.get("source_turn_id") or "")
    previous_is_trusted = bool(
        (current_scope and previous_turn_id and previous_scope == current_scope)
        or (not current_scope and not previous_scope and not previous_turn_id)
    )

    known_slots: dict[str, dict[str, Any]] = {}
    raw_slots = compilation.get("known_slots")
    if isinstance(raw_slots, dict):
        for name, slot in raw_slots.items():
            if name not in _SEMANTIC_CONTEXT_SLOT_NAMES or not isinstance(slot, dict):
                continue
            source = str(slot.get("source") or "")
            if source not in _SEMANTIC_CONTEXT_SLOT_SOURCES:
                continue
            source_turn_id = str(slot.get("source_turn_id") or "")
            if source in {
                "previous_turn_context", "previous_result_context", "current_query_evidence",
            } and (not previous_is_trusted or source_turn_id != previous_turn_id):
                continue
            value = slot.get("value")
            if not isinstance(value, (str, int, float, bool, list, tuple, dict)):
                continue
            projected: dict[str, Any] = {"value": value, "source": source}
            if source_turn_id:
                projected["source_turn_id"] = source_turn_id
            confidence = slot.get("confidence")
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                projected["confidence"] = max(0.0, min(float(confidence), 1.0))
            known_slots[str(name)] = projected

    raw_missing_slots = compilation.get("missing_slots")
    missing_slots = (
        [item for item in raw_missing_slots[:12] if isinstance(item, str)]
        if isinstance(raw_missing_slots, list) else []
    )
    result: dict[str, Any] = {
        "relation": compilation.get("relation"),
        "intent": compilation.get("intent"),
        "known_slots": known_slots,
        "missing_slots": missing_slots,
        "needs_clarification": bool(compilation.get("needs_clarification")),
        "clarification_reason": compilation.get("clarification_reason"),
        "source_turn_id": compilation.get("source_turn_id"),
    }
    operation_intent = compilation.get("operation_intent")
    if isinstance(operation_intent, dict):
        result["operation_intent"] = dict(operation_intent)
    request_contract = compilation.get("request_contract")
    if isinstance(request_contract, dict):
        contract_fields = (
            "metric_id", "operation_intent", "dimension", "dimensions",
            "comparison", "causal_boundary", "required_evidence_sequence",
            "parse_status", "uncertainty_reason",
        )
        result["request_contract"] = {
            key: request_contract[key]
            for key in contract_fields
            if key in request_contract
        }
    followup = compilation.get("result_followup")
    if isinstance(followup, dict) and isinstance(followup.get("kind"), str):
        result["result_followup"] = {
            key: followup[key]
            for key in ("kind", "direction", "n", "unavailable_reason")
            if key in followup
        }
    return result if any((
        known_slots,
        result.get("operation_intent"),
        result.get("request_contract"),
        result.get("result_followup"),
        result.get("missing_slots"),
    )) else None


def semantic_parse(state: AgentState) -> AgentState:
    settings = get_settings()
    mode = settings.semantic_interpreter_mode
    active_canary_authorized = False
    if mode == "active":
        from app.complex_analysis.canary import evaluate_canary_enrollment
        from app.tenancy.context import current_tenant_context

        enrollment = evaluate_canary_enrollment(settings, current_tenant_context())
        state["complex_route_canary_selected"] = enrollment.eligible
        state["complex_route_canary_observation"] = {
            "eligible": enrollment.eligible,
            "reason_code": enrollment.reason_code,
            "sample_rate": enrollment.sample_rate,
        }
        if not enrollment.eligible:
            state["semantic_contract"] = {}
            state["dialogue_context_resolution"] = {}
            state["routing_context"] = {}
            # Keep the historic active no-op shape when the new Phase 3
            # control is simply disabled; this preserves default behavior.
            if not bool(getattr(settings, "complex_route_canary_enabled", False)):
                state["semantic_parse_status"] = "reserved_phase0_noop"
                state["semantic_parse_observation"] = {
                    "mode": mode,
                    "status": "reserved_phase0_noop",
                }
            else:
                state["semantic_parse_status"] = "active_canary_skipped"
                state["semantic_parse_observation"] = {
                    "mode": mode,
                    "status": "active_canary_skipped",
                    "reason_code": enrollment.reason_code,
                }
            return state
        active_canary_authorized = True
    elif mode != "shadow":
        # Preserve the Phase 0 off/active output exactly. In particular, this
        # path does not read the question, history, or call an interpreter.
        state["semantic_contract"] = {}
        state["dialogue_context_resolution"] = {}
        state["routing_context"] = {}
        state["semantic_parse_status"] = "reserved_phase0_noop"
        state["semantic_parse_observation"] = {
            "mode": mode,
            "status": "reserved_phase0_noop",
        }
        LOGGER.info("semantic interpreter status mode=%s status=reserved_phase0_noop", mode)
        return state

    if mode == "shadow" and not getattr(settings, "semantic_interpreter_inline_shadow_enabled", False):
        # The current graph node invokes the interpreter synchronously. Keep
        # this path disabled in normal chat traffic until shadow work can run
        # without holding the user request open.
        state["semantic_contract"] = {}
        state["dialogue_context_resolution"] = {}
        state["routing_context"] = {}
        state["semantic_interpreter_observation"] = {
            "status": "skipped",
            "reason_code": "shadow_requires_offline_runner",
        }
        state["semantic_parse_status"] = "shadow_skipped_offline_only"
        state["semantic_parse_observation"] = {
            "mode": mode,
            "status": "shadow_skipped_offline_only",
            "reason_code": "shadow_requires_offline_runner",
        }
        LOGGER.info(
            "semantic interpreter status mode=%s status=shadow_skipped_offline_only",
            mode,
        )
        return state

    # Do not touch raw user text unless the explicit privacy opt-in is enabled.
    question = ""
    if settings.semantic_interpreter_send_raw_question:
        raw_question = state.get("user_message")
        question = raw_question if isinstance(raw_question, str) else ""
    turn_id = str(
        state.get("chat_turn_id")
        or state.get("trace_id")
        or state.get("thread_id")
        or "unknown-turn"
    )
    dialogue_context = _prior_semantic_context(state.get("previous_turn_context"))
    resolved_context = _resolved_turn_context(state)
    if resolved_context is not None:
        if dialogue_context is None:
            dialogue_context = {}
        dialogue_context["resolved_turn_context"] = resolved_context
    interpreter_kwargs = {
        "turn_id": turn_id,
        "dialogue_context": dialogue_context,
        "mode": mode,
    }
    if active_canary_authorized:
        interpreter_kwargs["active_canary_authorized"] = True
        interpreter_kwargs["validation_repair"] = bool(getattr(
            settings, "semantic_interpreter_validation_repair_enabled", True,
        ))
    observation = interpret_semantics(question, **interpreter_kwargs)
    if mode == "active":
        # The active path needs the validated contract only until routing and
        # plan preflight complete. Do not retain Candidate spans/raw output in
        # the persisted trace or telemetry projection.
        safe_keys = (
            "status", "mode", "model", "prompt_version", "catalog_snapshot_hash",
            "elapsed_ms", "interaction_policy", "request_options", "first_token_ms",
            "reasoning_chars", "partial_output_chars", "usage_status", "timeout_kind",
            "input_tokens", "output_tokens", "estimated_cost", "reason_code", "issue_codes",
            "repair",
        )
        state["semantic_interpreter_observation"] = {
            key: observation.get(key) for key in safe_keys
        }
    else:
        state["semantic_interpreter_observation"] = observation
    success = (
        observation.get("status") == "ok"
        if mode == "active"
        else observation.get("status") in {"ok", "validator_issues"}
    )
    state["semantic_contract"] = observation.get("contract") if success else {}
    state["dialogue_context_resolution"] = {}
    state["routing_context"] = {}
    parse_prefix = "active_canary" if mode == "active" else "shadow"
    state["semantic_parse_status"] = f"{parse_prefix}_{observation.get('status', 'unavailable')}"
    state["semantic_parse_observation"] = {
        "mode": mode,
        "status": state["semantic_parse_status"],
        "reason_code": observation.get("reason_code"),
    }
    if mode == "active":
        state["complex_route_canary_observation"]["parse_status"] = observation.get("status")
    LOGGER.info(
        "semantic interpreter status mode=%s status=%s reason=%s",
        mode,
        observation.get("status"),
        observation.get("reason_code"),
    )
    return state
