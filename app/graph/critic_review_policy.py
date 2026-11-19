"""Fail-closed routing policy for optional Critic LLM reviews.

This policy deliberately uses only structured workflow facts that are already
available to the Critic.  It is not an LLM classifier and never changes an
authorization, evidence, numeric, or safe-stop decision.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


# Registered decompositions have a dedicated reconciliation/evidence review.
# Its successful state names intentionally differ from the generic numeric and
# completeness reviewers, which are not applicable after that specialised
# validation has run.
_REGISTERED_DECOMPOSITION_ALLOWED_STATUSES = {
    "request_contract_review": {"passed"},
    "data_review": {"passed"},
    "claim_review": {"approved"},
    "numeric_review": {"not_applicable"},
    "completeness_review": {"skipped"},
    "causal_review": {"approved"},
}


def _adaptive_answer_contract_is_valid(
    state: Mapping[str, Any], deterministic_reviews: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, str]:
    """Validate that adaptive mode is reviewing a deterministic final answer.

    A passing data/evidence gate alone is insufficient: the answer may have
    acquired a free-form semantic claim after the card was validated.  The
    contract is created by ``generate_answer`` from the validated card and is
    checked here immediately before optional LLM review would be skipped.
    """
    contract = state.get("adaptive_answer_contract")
    if not isinstance(contract, Mapping):
        return False, "answer_contract_missing"
    ledger = contract.get("ledger")
    rendered = contract.get("rendered_answer")
    if not isinstance(ledger, Mapping) or not isinstance(rendered, str) or not rendered:
        return False, "answer_contract_malformed"
    if ledger.get("schema_version") != "registered_decomposition_answer_contract_v1":
        return False, "answer_contract_schema_invalid"
    ledger_json = json.dumps(dict(ledger), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    if hashlib.sha256(ledger_json.encode("utf-8")).hexdigest() != contract.get("ledger_sha256"):
        return False, "answer_contract_ledger_hash_mismatch"
    if hashlib.sha256(rendered.encode("utf-8")).hexdigest() != contract.get("rendered_answer_sha256"):
        return False, "answer_contract_render_hash_mismatch"
    answer = str(state.get("answer") or state.get("final_answer") or "")
    if answer != rendered:
        return False, "answer_not_deterministic_render"
    data_review = deterministic_reviews.get("data_review") or {}
    if (
        ledger.get("metric_id") != data_review.get("target_metric_id")
        or ledger.get("method_id") != data_review.get("method_id")
        or list(ledger.get("evidence_ids") or []) != list(data_review.get("evidence_ids") or [])
        or ledger.get("business_cause_confirmed") is not False
    ):
        return False, "answer_contract_not_bound_to_validated_data_review"
    return True, "registered_decomposition_deterministic_answer_verified"


def decide_optional_llm_review(
    *,
    task_type: str,
    registered_decomposition_review: bool,
    terminal_execution_failure: bool,
    state: Mapping[str, Any],
    deterministic_reviews: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return an auditable, fail-closed decision for adaptive review mode.

    Only a registered numeric decomposition can skip the optional LLM reviews,
    and only after every listed deterministic check has passed.  Diagnostic,
    report, refusal, unknown and malformed paths always retain LLM review.
    Terminal upstream failures are not reviewed because they are already
    authoritative safe stops; this matches the pre-existing FULL behavior.
    """

    if terminal_execution_failure:
        return {
            "action": "not_run",
            "reason_code": "upstream_terminal_execution_failure",
            "reason": "An authoritative execution failure already requires a terminal safe stop.",
        }
    if task_type != "metric_query" or not registered_decomposition_review:
        return {
            "action": "review",
            "reason_code": "high_or_unclassified_risk_path",
            "reason": "Only fully validated registered numeric decompositions are eligible to skip optional LLM review.",
        }
    if state.get("errors"):
        return {
            "action": "review",
            "reason_code": "workflow_errors_present",
            "reason": "Any recorded workflow error fails closed to optional LLM review.",
        }
    if not str(state.get("answer") or state.get("final_answer") or "").strip():
        return {
            "action": "review",
            "reason_code": "answer_missing",
            "reason": "A missing answer is not eligible for a low-risk skip.",
        }
    contract_valid, contract_reason = _adaptive_answer_contract_is_valid(state, deterministic_reviews)
    if not contract_valid:
        if contract_reason == "answer_not_deterministic_render":
            return {
                "action": "repair",
                "reason_code": contract_reason,
                "reason": "The final answer differs from the trusted deterministic render and must be regenerated without an LLM review.",
            }
        return {
            "action": "reject",
            "reason_code": contract_reason,
            "reason": "The adaptive answer contract or its evidence binding is invalid; the current state cannot be trusted.",
        }
    failed = [
        name for name, allowed in _REGISTERED_DECOMPOSITION_ALLOWED_STATUSES.items()
        if (deterministic_reviews.get(name) or {}).get("status") not in allowed
    ]
    if failed:
        return {
            "action": "review",
            "reason_code": "deterministic_gate_not_passed",
            "failed_gates": failed,
            "reason": "A required deterministic gate did not pass; adaptive routing fails closed to optional LLM review.",
        }
    return {
        "action": "skip",
        "reason_code": "registered_decomposition_all_deterministic_gates_passed",
        "reason": "Registered numeric decomposition has complete authorized evidence and all required deterministic gates passed.",
        "passed_gates": sorted(_REGISTERED_DECOMPOSITION_ALLOWED_STATUSES),
        "answer_contract": contract_reason,
    }


def decide_optional_review_cascade(
    *,
    task_type: str,
    registered_decomposition_review: bool,
    terminal_execution_failure: bool,
    state: Mapping[str, Any],
    deterministic_reviews: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select an auditable reviewer set using deterministic evidence only.

    This cascade is deliberately narrower than the legacy adaptive mode: it
    never skips required groundedness and only shortens review for a validated
    registered numeric decomposition. Free-form diagnostic/report answers,
    workflow errors, incomplete gates, and malformed answer contracts retain
    the complete review set.
    """
    if terminal_execution_failure:
        return {
            "tier": "terminal", "action": "not_run",
            "reason_code": "upstream_terminal_execution_failure",
            "skip_review_names": [],
        }
    if task_type in {"anomaly_analysis", "report_generation"}:
        return {
            "tier": "high", "action": "full_review",
            "reason_code": "high_risk_freeform_task",
            "skip_review_names": [],
        }
    if state.get("errors"):
        return {
            "tier": "high", "action": "full_review",
            "reason_code": "workflow_errors_present",
            "skip_review_names": [],
        }
    if task_type != "metric_query" or not registered_decomposition_review:
        return {
            "tier": "standard", "action": "full_review",
            "reason_code": "not_registered_numeric_decomposition",
            "skip_review_names": [],
        }
    contract_valid, contract_reason = _adaptive_answer_contract_is_valid(
        state, deterministic_reviews,
    )
    if not contract_valid:
        return {
            "tier": "high", "action": "full_review",
            "reason_code": contract_reason,
            "skip_review_names": [],
        }
    failed = [
        name for name, allowed in _REGISTERED_DECOMPOSITION_ALLOWED_STATUSES.items()
        if (deterministic_reviews.get(name) or {}).get("status") not in allowed
    ]
    if failed:
        return {
            "tier": "high", "action": "full_review",
            "reason_code": "deterministic_gate_not_passed",
            "failed_gates": failed,
            "skip_review_names": [],
        }
    return {
        "tier": "low", "action": "groundedness_only",
        "reason_code": "registered_decomposition_all_deterministic_gates_passed",
        "skip_review_names": ["completeness", "consistency"],
        "required_review_names": ["groundedness"],
        "answer_contract": contract_reason,
    }
