from __future__ import annotations

import json

import pytest

from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG
from scripts.check_capability_router_gate import (
    BEHAVIOR_CATEGORIES,
    evaluate_gate,
)
from scripts.evaluate_capability_router import evaluate_payload


def _row(category: str, *, selected: bool = False, request_mode: str = "EXECUTE") -> dict:
    route = next(item for item in DEFAULT_SEMANTIC_CATALOG.routes if item.route_id == "customer_analysis")
    capability_ids = ["customer_rfm_segmentation"] if selected else []
    decision_basis = {
        "contract_version": "semantic-contract-v2.1",
        "catalog_snapshot_hash": DEFAULT_SEMANTIC_CATALOG.snapshot_hash(),
        "operation_ids": ["operation_hash"] if selected else [],
        "clause_ids": ["clause_hash"] if selected else [],
        "capability_ids": capability_ids,
        "effective_arguments_digest": "a" * 64 if selected else None,
        "rejected_candidates": [],
        "rejection_reasons": [],
    }
    outcome = {
        "status": "SELECTED" if selected else "NO_MATCH",
        "domain": "BUSINESS_ANALYSIS",
        "route_id": route.route_id if selected else None,
        "capability_ids": capability_ids,
        "route_capability_ids": [item.capability_id for item in route.capability_ids] if selected else [],
        "clarification_required": False,
    }
    return {
        "case_id": f"fixture-{category}",
        "category": category,
        "contract_present": True,
        "contract_status": "CONFIRMED",
        "domain": "BUSINESS_ANALYSIS",
        "request_mode": request_mode,
        "modality": "FACTUAL",
        "has_positive_execute_operation": selected,
        "operation_summaries": ([{
            "operation_id": "operation_hash",
            "clause_id": "clause_hash",
            "kind": "SEGMENT",
            "target_concept_ids": ["CUSTOMER"],
            "polarity": "POSITIVE",
            "mode": "EXECUTE",
            "arguments": {"basis": [
                {"concept_id": "PURCHASE_FREQUENCY", "stance": "ADOPTED"},
                {"concept_id": "MONETARY", "stance": "ADOPTED"},
            ]},
            "mentioned_arguments": {},
        }] if selected else []),
        "route_status": outcome["status"],
        "route_id": outcome["route_id"],
        "capability_ids": capability_ids,
        "route_capability_ids": outcome["route_capability_ids"],
        "operation_ids": decision_basis["operation_ids"],
        "clause_ids": decision_basis["clause_ids"],
        "rejection_reasons": [],
        "route_outcome": outcome,
        "decision_basis": decision_basis,
        "draft_match": "match",
        "deterministic": True,
        "catalog_bounded": True,
    }


def _payload(*, selected_category: str | None = None, request_mode: str = "EXECUTE") -> dict:
    rows = [
        _row(category, selected=category == selected_category, request_mode=request_mode)
        for category in BEHAVIOR_CATEGORIES
    ]
    return {
        "schema_version": "capability-router-eval-v1",
        "summary": {
            "result_count": len(rows),
            "repeat_count": 2,
            "determinism_passed": True,
            "catalog_bounded": True,
        },
        "results": rows,
    }


def _invalid_input_row(
    case_id: str,
    *,
    contract_status: str = "UNSUPPORTED",
    domain: str = "GENERAL_CONVERSATION",
    route_status: str = "GENERAL_CONVERSATION",
) -> dict:
    row = _row("reminder_negation")
    row.update({
        "case_id": case_id,
        "contract_status": contract_status,
        "domain": domain,
        "route_status": route_status,
        "route_id": None,
        "capability_ids": [],
        "route_capability_ids": [],
        "operation_ids": [],
        "clause_ids": [],
        "operation_summaries": [],
        "has_positive_execute_operation": False,
    })
    row["route_outcome"].update({
        "status": route_status,
        "domain": domain,
        "route_id": None,
        "capability_ids": [],
        "route_capability_ids": [],
        "clarification_required": route_status in {"NEEDS_CONTEXT", "AMBIGUOUS", "CONFLICT"},
    })
    row["decision_basis"].update({
        "operation_ids": [],
        "clause_ids": [],
        "capability_ids": [],
    })
    return row


def _no_contract_row(category: str, case_id: str = "provider-failed-case") -> dict:
    row = _row(category)
    row.update({
        "case_id": case_id,
        "contract_present": False,
        "contract_status": None,
        "domain": None,
        "request_mode": None,
        "modality": None,
        "route_status": "no_contract",
        "route_id": None,
        "capability_ids": [],
        "route_capability_ids": [],
        "route_outcome": None,
        "operation_ids": [],
        "clause_ids": [],
        "operation_summaries": [],
        "decision_basis": {
            "contract_version": None,
            "catalog_snapshot_hash": None,
            "operation_ids": [],
            "clause_ids": [],
            "capability_ids": [],
            "effective_arguments_digest": None,
            "rejected_candidates": [],
            "rejection_reasons": [],
        },
        "has_positive_execute_operation": None,
        "deterministic": None,
        "catalog_bounded": None,
    })
    return row


def test_phase2_gate_passes_bounded_deterministic_fixture() -> None:
    result = evaluate_gate(_payload())

    assert result["passed"] is True
    assert result["metrics"]["behavior_case_count"] == 4
    assert result["metrics"]["behavior_pass_rate"] == 1.0
    assert result["metrics"]["static_boundary_issue_count"] == 0


def test_phase2_gate_counts_invalid_reminders_as_separately_handled() -> None:
    payload = _payload()
    payload["results"].extend([
        _invalid_input_row("reminder-invalid-1"),
        _invalid_input_row("reminder-invalid-2"),
    ])
    payload["summary"]["result_count"] = len(payload["results"])

    result = evaluate_gate(payload)

    assert result["passed"] is True
    assert result["metrics"]["behavior_interpreter_input_invalid_count"] == 2
    assert result["metrics"]["behavior_interpreter_input_invalid_handled_count"] == 2
    assert result["metrics"]["behavior_interpreter_input_invalid_handled_rate"] == 1.0
    assert result["metrics"]["behavior_router_eligible_count"] == 4
    assert result["metrics"]["behavior_router_passed_count"] == 4
    assert result["metrics"]["behavior_router_pass_rate"] == 1.0
    handled = [
        item for item in result["behavior_cases"]
        if item["status"] == "interpreter_input_invalid_handled"
    ]
    assert len(handled) == 2
    assert all(item["route_status"] == "GENERAL_CONVERSATION" for item in handled)


def test_phase2_gate_counts_no_contract_as_separate_fail_closed_observation() -> None:
    payload = _payload()
    payload["results"].append(_no_contract_row("sort_vs_basis"))
    payload["summary"]["result_count"] = len(payload["results"])

    result = evaluate_gate(payload)

    assert result["passed"] is True
    assert result["metrics"]["behavior_interpreter_input_invalid_count"] == 0
    assert result["metrics"]["behavior_no_contract_count"] == 1
    assert result["metrics"]["behavior_no_contract_fail_closed_count"] == 1
    assert result["metrics"]["behavior_no_contract_fail_closed_rate"] == 1.0
    assert result["metrics"]["behavior_router_eligible_count"] == 4
    no_contract = next(
        item for item in result["behavior_cases"] if item["status"] == "no_contract_fail_closed"
    )
    assert no_contract["pass"] is True


def test_phase2_gate_rejects_unrecovered_repair_contract_reaching_router() -> None:
    payload = _payload()
    payload["results"][0]["repair_attempted"] = True
    payload["results"][0]["repair_recovered"] = False

    result = evaluate_gate(payload)

    assert result["passed"] is False
    assert result["metrics"]["repair_attempt_count"] == 1
    assert result["metrics"]["repair_recovered_count"] == 0
    assert any("failed validation repair reached the Router" in item for item in result["failures"])


def test_phase2_gate_rejects_plan_without_interpreter_contract() -> None:
    payload = _payload()
    row = _no_contract_row("sort_vs_basis")
    row["plan_id"] = "unexpected-plan"
    payload["results"].append(row)
    payload["summary"]["result_count"] = len(payload["results"])

    result = evaluate_gate(payload)

    assert result["passed"] is False
    no_contract = next(
        item for item in result["behavior_cases"] if item["status"] == "no_contract_unhandled"
    )
    assert no_contract["pass"] is False
    assert "must not include a plan" in no_contract["reason"]


def test_phase2_gate_does_not_treat_missing_reviewed_route_case_as_coverage() -> None:
    payload = _payload()
    payload["results"].append(
        _no_contract_row("sort_vs_basis", "v341-labeled-sort-basis-01")
    )
    payload["summary"]["result_count"] = len(payload["results"])

    result = evaluate_gate(payload, require_reviewed_behavior_cases=True)

    assert result["metrics"]["behavior_no_contract_count"] == 1
    assert result["metrics"]["behavior_no_contract_fail_closed_count"] == 1
    assert any(
        "v341-labeled-sort-basis-01: reviewed route behavior case did not reach the Router" in failure
        for failure in result["failures"]
    )


@pytest.mark.parametrize(
    ("contract_status", "domain", "route_status", "expected_status"),
    [
        ("UNSUPPORTED", "BUSINESS_ANALYSIS", "UNSUPPORTED", "interpreter_input_invalid_handled"),
        ("UNSUPPORTED", "GENERAL_CONVERSATION", "UNSUPPORTED", "interpreter_input_invalid_unhandled"),
        ("MISSING", "BUSINESS_ANALYSIS", "NO_MATCH", "interpreter_input_invalid_unhandled"),
    ],
)
def test_phase2_gate_checks_safe_route_for_invalid_contract(
    contract_status, domain, route_status, expected_status
) -> None:
    row = _invalid_input_row(
        "invalid-input",
        contract_status=contract_status,
        domain=domain,
        route_status=route_status,
    )

    result = evaluate_gate(_payload_with_behavior_row(row))

    item = next(case for case in result["behavior_cases"] if case["case_id"] == "invalid-input")
    assert item["status"] == expected_status
    assert item["pass"] is (expected_status == "interpreter_input_invalid_handled")


def test_phase2_gate_enforces_adjudicated_missing_basis_clarification() -> None:
    row = _row("postposed_negation")
    row.update({
        "case_id": "v341-labeled-neg-postposed-01",
        "route_status": "NEEDS_CONTEXT",
        "capability_ids": ["customer_rfm_segmentation"],
        "rejection_reasons": ["customer_rfm_segmentation:missing_required_arguments"],
    })
    row["route_outcome"].update({
        "status": "NEEDS_CONTEXT",
        "clarification_required": True,
    })
    row["decision_basis"].update({
        "capability_ids": ["customer_rfm_segmentation"],
        "rejection_reasons": ["customer_rfm_segmentation:missing_required_arguments"],
    })

    result = evaluate_gate(_payload_with_behavior_row(row))

    item = next(case for case in result["behavior_cases"] if case["case_id"] == row["case_id"])
    assert item["pass"] is True
    assert item["route_status"] == "NEEDS_CONTEXT"


def test_phase2_gate_rejects_no_match_for_adjudicated_missing_basis_case() -> None:
    row = _row("postposed_negation")
    row.update({
        "case_id": "v341-labeled-neg-postposed-01",
        "route_status": "NO_MATCH",
    })
    row["route_outcome"].update({
        "status": "NO_MATCH",
        "clarification_required": False,
    })
    row["decision_basis"].update({
        "rejection_reasons": ["customer_rfm_segmentation:missing_required_arguments"],
    })

    result = evaluate_gate(_payload_with_behavior_row(row))

    item = next(case for case in result["behavior_cases"] if case["case_id"] == row["case_id"])
    assert item["pass"] is False
    assert "reviewed case must route to NEEDS_CONTEXT" in item["reason"]


def _payload_with_behavior_row(row: dict) -> dict:
    payload = _payload()
    payload["results"] = [item for item in payload["results"] if item["category"] != row["category"]]
    payload["results"].append(row)
    payload["summary"]["result_count"] = len(payload["results"])
    return payload


@pytest.mark.parametrize(
    ("field", "value", "expected_failure"),
    [
        ("request_mode", "ASK", "selected for request_mode=ASK"),
        ("modality", "HYPOTHETICAL", "selected for HYPOTHETICAL modality"),
        ("contract_status", "UNSUPPORTED", "selected for non-CONFIRMED contract"),
        ("has_positive_execute_operation", False, "selected without a positive EXECUTE operation"),
    ],
)
def test_phase2_gate_blocks_unsafe_selected_fixture(field, value, expected_failure) -> None:
    payload = _payload(selected_category="sort_vs_basis")
    row = next(item for item in payload["results"] if item["category"] == "sort_vs_basis")
    row[field] = value
    result = evaluate_gate(payload)

    assert result["passed"] is False
    assert any(expected_failure in failure for failure in result["failures"])
    behavior = next(item for item in result["behavior_cases"] if item["category"] == "sort_vs_basis")
    if field != "contract_status":
        assert behavior["pass"] is False


def test_offline_evaluation_json_excludes_source_text_surfaces_and_templates() -> None:
    user_text = "PRIVATE USER QUESTION"
    surface = "PRIVATE SURFACE"
    contract = {
        "contract_version": "semantic-contract-v2.1",
        "turn_id": "privacy-turn",
        "parse_status": "PARSED",
        "semantic_status": "CONFIRMED",
        "domain": "GENERAL_CONVERSATION",
        "clauses": [{"id": "c1", "span": {"start": 0, "end": len(surface)}}],
        "mentions": [{
            "id": "m1",
            "clause_id": "c1",
            "span": {"start": 0, "end": len(surface)},
            "surface": surface,
            "concept_id": "CUSTOMER",
            "grounding_source": "exact_alias",
            "source_turn_id": "privacy-turn",
        }],
        "operations": [],
        "request_frame": {
            "request_mode": "ASK",
            "modality": "FACTUAL",
            "dialogue_act_provisional": "NEW",
            "dialogue_act_validated": "NEW",
        },
        "catalog_snapshot_hash": DEFAULT_SEMANTIC_CATALOG.snapshot_hash(),
    }
    source = {"results": [{
        "case_id": "privacy-case",
        "category": "general_conversation",
        "text": user_text,
        "observation": {
            "status": "ok",
            "contract": contract,
            "repair": {
                "attempted": True,
                "recovered": True,
                "initial_status": "schema_invalid",
                "initial_reason_code": "candidate_schema_invalid",
            },
        },
    }]}

    evaluated = evaluate_payload(source, repeat=2)
    serialized = json.dumps(evaluated, ensure_ascii=False)

    assert user_text not in serialized
    assert surface not in serialized
    assert "template_id" not in serialized
    assert "surface" not in serialized
    assert evaluated["results"][0]["repair_attempted"] is True
    assert evaluated["results"][0]["repair_recovered"] is True
    assert evaluated["summary"]["repair_attempt_count"] == 1
