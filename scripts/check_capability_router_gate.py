"""Check deterministic, catalog-bounded completion of the Phase 2 shadow router."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.complex_analysis.capability_router_boundaries import capability_router_static_issues  # noqa: E402
from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG, RouteOutcome  # noqa: E402

BEHAVIOR_CATEGORIES = (
    "cross_clause_replacement",
    "postposed_negation",
    "reminder_negation",
    "sort_vs_basis",
)
_ROUTE_STATUSES = set(RouteOutcome.model_fields["status"].annotation.__args__)
_INTERPRETER_INVALID_STATUSES = {"interpreter_input_invalid"}
_REVIEWED_BEHAVIOR_ROUTE_EXPECTATIONS = {
    # These expectations come from the reviewed Phase 2 case table, not the
    # draft candidate labels.  Pinning them prevents a safe but incorrect
    # fallback such as NO_MATCH from silently counting as a passing route.
    "v341-neg-postposed-01": ("NO_MATCH", None, False),
    "v341-cross-clause-01": ("NEEDS_CONTEXT", None, True),
    "v341-sort-basis-01": ("NO_MATCH", None, False),
    "v341-required-011-02": ("SELECTED", "customer_rfm_segmentation", False),
    "v341-required-011-04": ("NEEDS_CONTEXT", None, True),
    "v341-required-011-10": ("SELECTED", "customer_rfm_segmentation", False),
    "v341-labeled-neg-postposed-01": ("NEEDS_CONTEXT", None, True),
    "v341-labeled-replacement-01": ("NEEDS_CONTEXT", None, True),
    "v341-labeled-sort-basis-01": ("NEEDS_CONTEXT", None, True),
}
_REVIEWED_INVALID_BEHAVIOR_EXPECTATIONS = {
    "v341-neg-reminder-01": ("UNSUPPORTED", "GENERAL_CONVERSATION", "GENERAL_CONVERSATION"),
    "v341-labeled-neg-reminder-01": (
        "UNSUPPORTED",
        "GENERAL_CONVERSATION",
        "GENERAL_CONVERSATION",
    ),
}
_REQUIRED_REVIEWED_BEHAVIOR_CASE_IDS = (
    frozenset(_REVIEWED_BEHAVIOR_ROUTE_EXPECTATIONS)
    | frozenset(_REVIEWED_INVALID_BEHAVIOR_EXPECTATIONS)
)
_ADJUDICATED_REJECTION_REASONS = {
    "v341-labeled-neg-postposed-01": "customer_rfm_segmentation:missing_required_arguments",
}


def _walk_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_mappings(child)


def _catalog_outcome_issues(row: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    route_by_id = {item.route_id: item for item in DEFAULT_SEMANTIC_CATALOG.routes}
    capability_ids = {item.capability_id for item in DEFAULT_SEMANTIC_CATALOG.capabilities}
    route = row.get("route_outcome")
    basis = row.get("decision_basis")
    if not isinstance(route, Mapping) or not isinstance(basis, Mapping):
        return ["missing sanitized route_outcome or decision_basis"]
    route_id = route.get("route_id")
    if route_id is not None and route_id not in route_by_id:
        issues.append("route id outside frozen catalog")
    for field in ("capability_ids", "route_capability_ids"):
        values = route.get(field)
        if not isinstance(values, list) or any(value not in capability_ids for value in values):
            issues.append(f"route outcome {field} outside frozen catalog")
    for field in ("capability_ids", "rejected_candidates"):
        values = basis.get(field)
        if not isinstance(values, list) or any(value not in capability_ids for value in values):
            issues.append(f"decision basis {field} outside frozen catalog")
    if route.get("status") == "SELECTED":
        selected = set(route.get("capability_ids") or [])
        declared = set(route.get("route_capability_ids") or [])
        catalog_declared = {
            item.capability_id for item in route_by_id[route_id].capability_ids
        } if route_id in route_by_id else set()
        if not selected or not selected <= declared or declared != catalog_declared or not selected <= catalog_declared:
            issues.append("selected capability is not declared by selected route")
    if route.get("status") != row.get("route_status"):
        issues.append("route status disagrees with per-case outcome")
    if route.get("domain") != row.get("domain"):
        issues.append("route domain disagrees with the input contract domain")
    if basis.get("contract_version") != "semantic-contract-v2.1":
        issues.append("decision basis contract version is not supported")
    if basis.get("catalog_snapshot_hash") != DEFAULT_SEMANTIC_CATALOG.snapshot_hash():
        issues.append("decision basis catalog snapshot is not frozen catalog")
    return issues


def _fail_closed_issues(row: Mapping[str, Any]) -> list[str]:
    if row.get("route_status") != "SELECTED":
        return []
    issues = []
    if row.get("contract_status") != "CONFIRMED":
        issues.append("selected for non-CONFIRMED contract")
    if row.get("request_mode") == "ASK":
        issues.append("selected for request_mode=ASK")
    if row.get("modality") == "HYPOTHETICAL":
        issues.append("selected for HYPOTHETICAL modality")
    if row.get("has_positive_execute_operation") is not True:
        issues.append("selected without a positive EXECUTE operation")
    return issues


def _is_interpreter_input_invalid(row: Mapping[str, Any]) -> bool:
    return (
        row.get("route_status") in _INTERPRETER_INVALID_STATUSES
        or row.get("contract_status") != "CONFIRMED"
        or row.get("domain") == "GENERAL_CONVERSATION"
    )


def _expected_invalid_input_route(row: Mapping[str, Any]) -> str | None:
    """Return the safe Router status for a non-confirmed interpreter contract."""
    if row.get("domain") == "GENERAL_CONVERSATION":
        return "GENERAL_CONVERSATION"
    return {
        "UNSUPPORTED": "UNSUPPORTED",
        "MISSING": "NEEDS_CONTEXT",
        "AMBIGUOUS": "AMBIGUOUS",
        "CONFLICT": "CONFLICT",
    }.get(row.get("contract_status"))


def _invalid_input_handling_issues(row: Mapping[str, Any]) -> list[str]:
    if row.get("contract_present") is not True:
        return ["no validated contract reached the Router"]
    expected_status = _expected_invalid_input_route(row)
    if expected_status is None:
        return ["no safe Router status is defined for this invalid input"]

    issues: list[str] = []
    if row.get("route_status") != expected_status:
        issues.append(f"invalid input must fail closed to {expected_status}")
    if row.get("route_id") is not None:
        issues.append("invalid input must not select a route")
    if row.get("capability_ids") or row.get("route_capability_ids"):
        issues.append("invalid input must not expose a capability")

    outcome = row.get("route_outcome")
    if not isinstance(outcome, Mapping):
        issues.append("invalid input is missing a route outcome")
    else:
        if outcome.get("status") != expected_status:
            issues.append(f"route outcome must be {expected_status}")
        if outcome.get("route_id") is not None:
            issues.append("route outcome must not select a route")
        if outcome.get("capability_ids") or outcome.get("route_capability_ids"):
            issues.append("route outcome must not expose a capability")
        should_clarify = expected_status in {"NEEDS_CONTEXT", "AMBIGUOUS", "CONFLICT"}
        if outcome.get("clarification_required") is not should_clarify:
            issues.append("route outcome clarification flag does not match its safe status")
    basis = row.get("decision_basis")
    if isinstance(basis, Mapping) and basis.get("capability_ids"):
        issues.append("decision basis must not cite a capability")
    if any(
        forbidden in mapping
        for mapping in _walk_mappings(row)
        for forbidden in ("plan", "plan_id", "sealed_route_decision")
    ):
        issues.append("invalid input must not include a plan or sealed decision")
    return issues


def _no_contract_fail_closed_issues(row: Mapping[str, Any]) -> list[str]:
    """Check the no-route boundary when the Interpreter produced no contract."""
    issues: list[str] = []
    if row.get("contract_present") is not False:
        issues.append("no_contract status must not carry a contract")
    if row.get("route_status") != "no_contract":
        issues.append("missing contract must remain a no_contract result")
    if row.get("route_id") is not None:
        issues.append("missing contract must not select a route")
    if row.get("capability_ids") or row.get("route_capability_ids"):
        issues.append("missing contract must not expose a capability")
    if row.get("route_outcome") is not None:
        issues.append("Router must not produce an outcome without a contract")
    if any(
        forbidden in mapping
        for mapping in _walk_mappings(row)
        for forbidden in ("plan", "plan_id", "sealed_route_decision")
    ):
        issues.append("missing contract must not include a plan or sealed decision")
    return issues


def _behavior_case_result(
    row: Mapping[str, Any],
    *,
    require_reviewed_expectation: bool = False,
) -> dict[str, Any]:
    case_id = str(row.get("case_id") or "unknown")
    category = str(row.get("category") or "unknown")
    if row.get("route_status") == "no_contract":
        issues = _no_contract_fail_closed_issues(row)
        return {
            "case_id": case_id,
            "category": category,
            "status": "no_contract_fail_closed" if not issues else "no_contract_unhandled",
            "pass": not issues,
            "reason": "; ".join(issues) if issues else (
                "Interpreter produced no validated contract; no Router outcome or plan was emitted"
            ),
            "route_status": row.get("route_status"),
            "capability_ids": row.get("capability_ids", []),
            "draft_match": row.get("draft_match", "unavailable"),
        }
    if _is_interpreter_input_invalid(row):
        issues = _invalid_input_handling_issues(row)
        issues.extend(_catalog_outcome_issues(row))
        expected_invalid = _REVIEWED_INVALID_BEHAVIOR_EXPECTATIONS.get(case_id)
        if expected_invalid is not None:
            expected_contract_status, expected_domain, expected_route_status = expected_invalid
            if row.get("contract_status") != expected_contract_status:
                issues.append(f"reviewed invalid case contract must be {expected_contract_status}")
            if row.get("domain") != expected_domain:
                issues.append(f"reviewed invalid case domain must be {expected_domain}")
            if row.get("route_status") != expected_route_status:
                issues.append(f"reviewed invalid case must route to {expected_route_status}")
        if case_id in _REVIEWED_BEHAVIOR_ROUTE_EXPECTATIONS:
            expected_status, _expected_capability_id, _expected_clarification = (
                _REVIEWED_BEHAVIOR_ROUTE_EXPECTATIONS[case_id]
            )
            issues.append("reviewed behavior case must retain a CONFIRMED contract")
            if row.get("route_status") != expected_status:
                issues.append(f"reviewed case must route to {expected_status}")
        return {
            "case_id": case_id,
            "category": category,
            "status": (
                "interpreter_input_invalid_handled"
                if not issues
                else "interpreter_input_invalid_unhandled"
            ),
            "pass": not issues,
            "reason": "; ".join(issues) if issues else (
                "non-confirmed contract received the expected no-route safe outcome"
            ),
            "route_status": row.get("route_status"),
            "capability_ids": row.get("capability_ids", []),
            "draft_match": row.get("draft_match", "unavailable"),
        }

    issues = _fail_closed_issues(row)
    issues.extend(_catalog_outcome_issues(row))
    outcome = row.get("route_outcome") if isinstance(row.get("route_outcome"), Mapping) else {}
    expected_route = _REVIEWED_BEHAVIOR_ROUTE_EXPECTATIONS.get(case_id)
    if expected_route is None and require_reviewed_expectation:
        issues.append("behavior case has no reviewed route expectation")
    if require_reviewed_expectation and case_id in _REVIEWED_INVALID_BEHAVIOR_EXPECTATIONS:
        issues.append("reviewed invalid behavior case must retain its UNSUPPORTED contract")
    if expected_route is not None:
        expected_status, expected_capability_id, expected_clarification = expected_route
        if outcome.get("status") != expected_status:
            issues.append(f"reviewed case must route to {expected_status}")
        if outcome.get("clarification_required") is not expected_clarification:
            issues.append("reviewed case clarification flag does not match its expected route")
        if expected_capability_id is not None:
            if outcome.get("capability_ids") != [expected_capability_id]:
                issues.append(
                    f"reviewed case must select capability {expected_capability_id}"
                )
        elif outcome.get("route_id") is not None or outcome.get("capability_ids"):
            issues.append("reviewed non-selected case must not select a route or capability")
        if expected_status == "SELECTED" and outcome.get("route_id") != "customer_analysis":
            issues.append("reviewed selected case must use the customer_analysis route")
        expected_reason = _ADJUDICATED_REJECTION_REASONS.get(case_id)
        if expected_reason is not None:
            decision_basis = row.get("decision_basis")
            rejection_reasons = (
                decision_basis.get("rejection_reasons", [])
                if isinstance(decision_basis, Mapping)
                else []
            )
            if expected_reason not in rejection_reasons:
                issues.append(f"adjudicated case must record {expected_reason}")
    selected_operation_ids = set((row.get("decision_basis") or {}).get("operation_ids", []))
    operations = row.get("operation_summaries") if isinstance(row.get("operation_summaries"), list) else []
    positive_operation_ids = {
        item.get("operation_id") for item in operations
        if isinstance(item, Mapping)
        and item.get("polarity") == "POSITIVE"
        and item.get("mode") == "EXECUTE"
    }
    if outcome.get("status") == "SELECTED" and not selected_operation_ids <= positive_operation_ids:
        issues.append("selected decision cites a non-positive or non-EXECUTE operation")

    if category == "sort_vs_basis" and "customer_rfm_segmentation" in (outcome.get("capability_ids") or []):
        allowed_basis = {"PURCHASE_FREQUENCY", "MONETARY", "RECENCY"}
        segment_basis = {
            value.get("concept_id")
            for operation in operations
            if isinstance(operation, Mapping)
            and operation.get("kind") == "SEGMENT"
            and operation.get("polarity") == "POSITIVE"
            and operation.get("mode") == "EXECUTE"
            for value in (operation.get("arguments") or {}).get("basis", [])
            if isinstance(value, Mapping) and value.get("stance") == "ADOPTED"
        }
        if len(segment_basis & allowed_basis) < 2:
            issues.append("RFM selection lacks two distinct basis concepts on a positive SEGMENT")

    return {
        "case_id": case_id,
        "category": category,
        "status": "pass" if not issues else "fail",
        "pass": not issues,
        "reason": "; ".join(issues) if issues else "deterministic catalog-bounded fail-closed route",
        "route_status": row.get("route_status"),
        "capability_ids": row.get("capability_ids", []),
        "draft_match": row.get("draft_match", "unavailable"),
    }


def evaluate_gate(
    payload: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
    require_reviewed_behavior_cases: bool = False,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("results"), list):
        raise ValueError("router evaluation must contain a results array")
    rows = payload["results"]
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("every router evaluation result must be an object")

    failures: list[str] = []
    summary = payload.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    repeat_count = summary.get("repeat_count")
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count < 2:
        failures.append("evaluation must repeat every contract at least twice")
    if summary.get("determinism_passed") is not True:
        failures.append("evaluation summary does not confirm determinism")
    if summary.get("catalog_bounded") is not True:
        failures.append("evaluation summary does not confirm catalog bounds")
    if require_reviewed_behavior_cases:
        behavior_ids = [
            str(row.get("case_id") or "")
            for row in rows
            if row.get("category") in BEHAVIOR_CATEGORIES
        ]
        behavior_id_counts = Counter(behavior_ids)
        missing_ids = sorted(_REQUIRED_REVIEWED_BEHAVIOR_CASE_IDS - set(behavior_ids))
        duplicate_ids = sorted(case_id for case_id, count in behavior_id_counts.items() if count > 1)
        if missing_ids:
            failures.append(
                "evaluation is missing required reviewed behavior cases: " + ", ".join(missing_ids)
            )
        if duplicate_ids:
            failures.append(
                "evaluation duplicates reviewed behavior cases: " + ", ".join(duplicate_ids)
            )
        rows_by_id = {str(row.get("case_id") or ""): row for row in rows}
        for case_id in sorted(_REVIEWED_BEHAVIOR_ROUTE_EXPECTATIONS):
            row = rows_by_id.get(case_id)
            if (
                row is None
                or row.get("contract_present") is not True
                or row.get("route_status") not in _ROUTE_STATUSES
            ):
                failures.append(
                    f"{case_id}: reviewed route behavior case did not reach the Router"
                )
    static_issues = capability_router_static_issues(repository_root)
    if static_issues:
        failures.extend(f"static boundary: {issue}" for issue in static_issues)

    routed_rows = [
        row for row in rows
        if row.get("contract_present") is True and row.get("route_status") in _ROUTE_STATUSES
    ]
    deterministic_count = 0
    bounded_count = 0
    fail_closed_check_pass_count = 0
    repair_without_recovery_route_count = 0
    for row in routed_rows:
        if row.get("deterministic") is True:
            deterministic_count += 1
        else:
            failures.append(f"{row.get('case_id')}: deterministic repeat failed")
        if row.get("catalog_bounded") is True:
            bounded_count += 1
        else:
            failures.append(f"{row.get('case_id')}: router outcome is not catalog-bounded")
        outcome_issues = _catalog_outcome_issues(row)
        if outcome_issues:
            failures.extend(f"{row.get('case_id')}: {issue}" for issue in outcome_issues)
        closed_issues = _fail_closed_issues(row)
        if not closed_issues:
            fail_closed_check_pass_count += 1
        else:
            failures.extend(f"{row.get('case_id')}: {issue}" for issue in closed_issues)
        if row.get("repair_attempted") is True and row.get("repair_recovered") is not True:
            repair_without_recovery_route_count += 1
            failures.append(f"{row.get('case_id')}: failed validation repair reached the Router")

        for mapping in _walk_mappings(row):
            if "template_id" in mapping:
                failures.append(f"{row.get('case_id')}: template_id leaked into evaluation output")
                break
            if {"text", "surface", "span", "raw_text", "user_message"}.intersection(mapping):
                failures.append(f"{row.get('case_id')}: raw text or span fields leaked into evaluation output")
                break

    for row in rows:
        if row.get("contract_present") is True and row.get("route_status") == "interpreter_input_invalid":
            continue
        if row.get("contract_present") is True and row.get("route_status") not in _ROUTE_STATUSES:
            failures.append(f"{row.get('case_id')}: router did not produce a valid outcome")

    behavior_results: list[dict[str, Any]] = []
    behavior_failures: list[str] = []
    for category in BEHAVIOR_CATEGORIES:
        category_rows = [row for row in rows if row.get("category") == category]
        if not category_rows:
            behavior_failures.append(f"missing behavior category: {category}")
            continue
        evaluated = [
            _behavior_case_result(
                row,
                require_reviewed_expectation=require_reviewed_behavior_cases,
            )
            for row in category_rows
        ]
        behavior_results.extend(evaluated)
        eligible = [
            item for item in evaluated
            if item["status"] not in {
                "interpreter_input_invalid_handled",
                "interpreter_input_invalid_unhandled",
                "no_contract_fail_closed",
                "no_contract_unhandled",
            }
        ]
        invalid_inputs = [
            item for item in evaluated
            if item["status"] in {
                "interpreter_input_invalid_handled",
                "interpreter_input_invalid_unhandled",
            }
        ]
        if not eligible:
            if not invalid_inputs:
                behavior_failures.append(f"{category}: no interpreter contract to evaluate")
        elif any(item["pass"] is False for item in eligible):
            behavior_failures.append(f"{category}: one or more eligible behavior cases failed")
        if any(item["pass"] is not True for item in invalid_inputs):
            behavior_failures.append(f"{category}: invalid interpreter input was not safely handled")
        no_contracts = [
            item for item in evaluated
            if item["status"] in {"no_contract_fail_closed", "no_contract_unhandled"}
        ]
        if any(item["pass"] is not True for item in no_contracts):
            behavior_failures.append(f"{category}: missing Interpreter contract was not fail-closed")
    failures.extend(behavior_failures)

    invalid_behavior = [
        item for item in behavior_results
        if item["status"] in {
            "interpreter_input_invalid_handled",
            "interpreter_input_invalid_unhandled",
        }
    ]
    invalid_behavior_count = len(invalid_behavior)
    invalid_behavior_handled = sum(item["pass"] is True for item in invalid_behavior)
    no_contract_behavior = [
        item for item in behavior_results
        if item["status"] in {"no_contract_fail_closed", "no_contract_unhandled"}
    ]
    no_contract_fail_closed_count = sum(item["pass"] is True for item in no_contract_behavior)
    eligible_behavior = [
        item for item in behavior_results
        if item["status"] not in {
            "interpreter_input_invalid_handled",
            "interpreter_input_invalid_unhandled",
            "no_contract_fail_closed",
            "no_contract_unhandled",
        }
    ]
    behavior_passed = sum(item["pass"] is True for item in eligible_behavior)
    repair_attempted_rows = [row for row in rows if row.get("repair_attempted") is True]
    repair_recovered_count = sum(row.get("repair_recovered") is True for row in repair_attempted_rows)
    metrics = {
        "result_count": len(rows),
        "routed_contract_count": len(routed_rows),
        "syntactically_invalid_contract_count": sum(
            row.get("route_status") == "interpreter_input_invalid" for row in rows
        ),
        "repeat_count": repeat_count,
        "deterministic_count": deterministic_count,
        "catalog_bounded_count": bounded_count,
        "fail_closed_check_pass_count": fail_closed_check_pass_count,
        "repair_attempt_count": len(repair_attempted_rows),
        "repair_recovered_count": repair_recovered_count,
        "repair_without_recovery_route_count": repair_without_recovery_route_count,
        "static_boundary_issue_count": len(static_issues),
        "behavior_case_count": len(behavior_results),
        "behavior_interpreter_input_invalid_count": invalid_behavior_count,
        "behavior_interpreter_input_invalid_handled_count": invalid_behavior_handled,
        "behavior_interpreter_input_invalid_handled_rate": (
            invalid_behavior_handled / invalid_behavior_count if invalid_behavior_count else None
        ),
        "behavior_no_contract_count": len(no_contract_behavior),
        "behavior_no_contract_fail_closed_count": no_contract_fail_closed_count,
        "behavior_no_contract_fail_closed_rate": (
            no_contract_fail_closed_count / len(no_contract_behavior)
            if no_contract_behavior else None
        ),
        "behavior_router_eligible_count": len(eligible_behavior),
        "behavior_router_passed_count": behavior_passed,
        "behavior_router_pass_rate": (
            behavior_passed / len(eligible_behavior) if eligible_behavior else None
        ),
        "behavior_eligible_count": len(eligible_behavior),
        "behavior_passed_count": behavior_passed,
        "behavior_pass_rate": behavior_passed / len(eligible_behavior) if eligible_behavior else None,
        "draft_match_counts": dict(sorted(Counter(
            row.get("draft_match", "unavailable") for row in rows
        ).items())),
    }
    return {
        "passed": not failures,
        "metrics": metrics,
        "failures": failures,
        "static_boundary_issues": static_issues,
        "behavior_cases": behavior_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=Path("data/evaluations/capability_router_v2_20261114.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.eval.read_text(encoding="utf-8"))
    result = evaluate_gate(
        payload,
        repository_root=ROOT,
        require_reviewed_behavior_cases=True,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print("PHASE2_GATE=" + ("PASS" if result["passed"] else "BLOCKED"))
    print(json.dumps({"metrics": result["metrics"], "failures": result["failures"]}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
