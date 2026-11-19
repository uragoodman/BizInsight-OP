"""Arm-neutral contract and scorer for the 60-case resume B/G study.

This module is separate from the historical six-case development protocol.
It is pure evaluation code: importing it does not start a workflow, database,
network connection, or Provider request.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import random
from typing import Any
import uuid


PROTOCOL_VERSION = "resume-formal-bg-contract-v1"
SCORER_VERSION = "resume-formal-bg-score-v3-field-bound-period-observation"
FORMAL_REPLICATES = (1, 2, 3)
FORMAL_ARMS = ("B", "G")
EXPECTED_STRATA = {
    "normal_query": 24,
    "multi_step_diagnostic": 16,
    "expected_limitation": 8,
    "fault_or_budget": 12,
}
B_ACTION_BY_OPERATION = {
    "metric_value": "query_value",
    "metric_compare": "metric_compare",
    "metric_trend": "query_trend",
    "registered_numeric_decomposition": "registered_numeric_decomposition",
    "dimension_contribution": "metric_contribution",
    "multi_step_diagnostic": ["query_trend", "metric_contribution"],
    "expected_data_insufficiency": "metric_compare",
    "fault_model_connection": "query_value",
    "fault_tool_timeout": "query_value",
    "fault_sql_execution": "query_value",
    "fault_token_budget": "query_value",
}
G_SEMANTIC_ROUTE_BY_OPERATION = {
    "metric_value": "metric_value",
    "metric_compare": "metric_compare",
    "metric_trend": "metric_trend",
    "registered_numeric_decomposition": "registered_numeric_decomposition",
    "dimension_contribution": "metric_contribution",
    "multi_step_diagnostic": ["metric_trend", "metric_contribution"],
    "expected_data_insufficiency": "metric_compare",
    "fault_model_connection": "metric_value",
    "fault_tool_timeout": "metric_value",
    "fault_sql_execution": "metric_value",
    "fault_token_budget": "metric_value",
}


def validate_candidate_set(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate design counts and diversity without claiming data readiness."""
    if len(cases) != sum(EXPECTED_STRATA.values()):
        raise ValueError(f"Expected 60 cases, got {len(cases)}")
    ids = [str(row.get("case_id") or "") for row in cases]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("case_id values must be unique and nonempty")
    strata = Counter(str(row.get("stratum") or "") for row in cases)
    if dict(strata) != EXPECTED_STRATA:
        raise ValueError(f"Strata mismatch: expected {EXPECTED_STRATA}, got {dict(strata)}")
    metric_counts = Counter(str(row.get("target_metric") or "") for row in cases)
    if "" in metric_counts or max(metric_counts.values(), default=0) > 4:
        raise ValueError("Every case needs a target metric and no metric may exceed four cases")
    family_counts = Counter(str(row.get("scenario_family") or "") for row in cases)
    if "" in family_counts or max(family_counts.values(), default=0) > 6:
        raise ValueError("Every case needs a scenario family and no family may exceed six cases")
    domains = {str(row.get("metric_domain") or "") for row in cases}
    aggregations = {str(row.get("aggregation") or "") for row in cases}
    if len(metric_counts) < 20 or len(domains - {""}) < 5 or len(aggregations - {""}) < 3:
        raise ValueError("Candidate set lacks the required metric/domain/aggregation coverage")
    if sum(str(row.get("stratum")) == "normal_query" for row in cases) != 24:
        raise ValueError("Normal query stratum must contain 24 cases")
    return {
        "case_count": len(cases),
        "stratum_counts": dict(strata),
        "unique_metrics": len(metric_counts),
        "metric_counts": dict(sorted(metric_counts.items())),
        "domain_count": len(domains - {""}),
        "domains": sorted(domains - {""}),
        "aggregation_classes": sorted(aggregations - {""}),
        "max_cases_per_metric": max(metric_counts.values()),
        "max_cases_per_scenario_family": max(family_counts.values()),
        "truth_ready_cases": sum(row.get("truth_status") == "verified" for row in cases),
        "qualification_ready_cases": sum(row.get("qualification_status") == "eligible" for row in cases),
    }


def build_formal_manifest(
    cases: Sequence[Mapping[str, Any]], *, seed: int, experiment_id: str,
) -> dict[str, Any]:
    """Create the 360-unit schedule without changing candidate truth or order."""
    validate_candidate_set(cases)
    if not str(experiment_id).strip():
        raise ValueError("experiment_id must be explicit")
    pairs = [(str(case["case_id"]), replicate) for case in cases for replicate in FORMAL_REPLICATES]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    first_arms = ["B"] * (len(pairs) // 2) + ["G"] * (len(pairs) // 2)
    rng.shuffle(first_arms)
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, experiment_id)
    entries: list[dict[str, Any]] = []
    for (case_id, replicate), first in zip(pairs, first_arms):
        pair_id = f"{case_id}-r{replicate}"
        for arm in (first, "G" if first == "B" else "B"):
            unit_key = f"{experiment_id}:{case_id}:{replicate}:{arm}"
            entries.append({
                "schedule_index": len(entries) + 1,
                "case_id": case_id,
                "replicate": replicate,
                "arm": arm,
                "pair_id": pair_id,
                "run_id": str(uuid.uuid5(namespace, unit_key)),
                "status": "not_started",
            })
    return {
        "schema_version": "resume-formal-bg-run-manifest-v1",
        "protocol_version": PROTOCOL_VERSION,
        "scorer_version": SCORER_VERSION,
        "experiment_id": experiment_id,
        "seed": seed,
        "planned_cases": len(cases),
        "replicates_per_case": len(FORMAL_REPLICATES),
        "arms": list(FORMAL_ARMS),
        "planned_units": len(entries),
        "schedule_mode": "serial_paired; seeded case-pair order and balanced arm order",
        "provider_authorized": False,
        "entries": entries,
    }


def validate_formal_manifest(
    manifest: Mapping[str, Any], cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed on missing, duplicate, reordered, or already-started units."""
    expected = build_formal_manifest(
        cases, seed=int(manifest.get("seed", -1)), experiment_id=str(manifest.get("experiment_id") or ""),
    )
    metadata_keys = {"candidate_sha256", "qualification_sha256", "truth_status_summary"}
    if any(manifest.get(key) != value for key, value in expected.items()) or set(manifest) - set(expected) - metadata_keys:
        raise ValueError("Formal B/G manifest does not match its deterministic seed, arms, and run IDs.")
    entries = list(manifest.get("entries") or [])
    run_ids = [str(row.get("run_id") or "") for row in entries]
    if len(entries) != 360 or len(set(run_ids)) != 360 or any(not value for value in run_ids):
        raise ValueError("Formal B/G schedule must contain 360 unique run IDs.")
    pairs: dict[str, list[Mapping[str, Any]]] = {}
    for row in entries:
        if row.get("status") != "not_started":
            raise ValueError("Frozen formal manifest entries must all begin not_started.")
        pairs.setdefault(str(row.get("pair_id")), []).append(row)
    if len(pairs) != 180 or any(
        len(rows) != 2 or {row.get("arm") for row in rows} != set(FORMAL_ARMS)
        or len({row.get("case_id") for row in rows}) != 1
        or len({row.get("replicate") for row in rows}) != 1
        for rows in pairs.values()
    ):
        raise ValueError("Each case-replicate pair must contain one B and one G run.")
    first_counts = {arm: sum(entries[i]["arm"] == arm for i in range(0, len(entries), 2)) for arm in FORMAL_ARMS}
    return {
        "planned_cases": 60,
        "planned_pairs": 180,
        "planned_units": len(entries),
        "run_id_count": len(set(run_ids)),
        "arm_counts": {arm: sum(row["arm"] == arm for row in entries) for arm in FORMAL_ARMS},
        "first_arm_pair_counts": first_counts,
        "all_not_started": all(row["status"] == "not_started" for row in entries),
    }


def validate_paired_manifest(
    manifest: Mapping[str, Any], cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate a pre-registered paired B/G manifest for any explicit size.

    Unlike ``validate_formal_manifest`` this does not assume the historical
    60-case/three-replicate protocol. The manifest itself fixes the case set
    and replicate count; validation checks every derived count, pair, run ID,
    order, and initial lifecycle state without regenerating or replacing IDs.
    """
    if not cases:
        raise ValueError("At least one selected case is required")
    case_ids = [str(row.get("case_id") or "") for row in cases]
    if any(not value for value in case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("Selected case IDs must be unique and nonempty")
    try:
        replicates = int(manifest.get("replicates_per_case"))
    except (TypeError, ValueError):
        replicates = 0
    if replicates < 1 or isinstance(manifest.get("replicates_per_case"), bool):
        raise ValueError("Manifest must declare a positive replicate count")
    entries = list(manifest.get("entries") or [])
    expected_pairs = len(case_ids) * replicates
    expected_units = expected_pairs * len(FORMAL_ARMS)
    declared_counts = {
        "case_count": len(case_ids),
        "planned_cases": len(case_ids),
        "planned_pairs": expected_pairs,
        "planned_units": expected_units,
    }
    for field, expected in declared_counts.items():
        if field in manifest and manifest[field] != expected:
            raise ValueError(f"Manifest {field} mismatch: expected {expected}, got {manifest[field]!r}")
    if manifest.get("arms") != list(FORMAL_ARMS):
        raise ValueError("Manifest arms must be exactly B and G in the declared order")
    if len(entries) != expected_units:
        raise ValueError(f"Manifest has {len(entries)} entries; expected {expected_units}")
    run_ids = [str(row.get("run_id") or "") for row in entries]
    if any(not value for value in run_ids) or len(set(run_ids)) != expected_units:
        raise ValueError("Manifest run IDs must be unique and nonempty")
    try:
        if any(str(uuid.UUID(value)) != value.lower() for value in run_ids):
            raise ValueError("Manifest run IDs must be canonical UUIDs")
    except (ValueError, AttributeError) as exc:
        raise ValueError("Manifest contains an invalid run ID") from exc
    expected_case_set = set(case_ids)
    if {str(row.get("case_id") or "") for row in entries} != expected_case_set:
        raise ValueError("Manifest entries do not match the selected case IDs")
    allowed_replicates = set(range(1, replicates + 1))
    pairs: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for position, row in enumerate(entries, start=1):
        if row.get("status") != "not_started":
            raise ValueError("Pre-registered manifest entries must all begin not_started")
        if row.get("schedule_index") != position:
            raise ValueError("Manifest schedule_index must be contiguous and match entry order")
        if row.get("arm") not in FORMAL_ARMS:
            raise ValueError("Manifest contains an unsupported arm")
        if row.get("replicate") not in allowed_replicates:
            raise ValueError("Manifest contains an undeclared replicate")
        pair_id = str(row.get("pair_id") or "")
        expected_pair_id = f"{row['case_id']}-r{row['replicate']}"
        if pair_id != expected_pair_id:
            raise ValueError("Manifest pair_id does not match case and replicate")
        pairs.setdefault(pair_id, []).append((position, row))
    if len(pairs) != expected_pairs:
        raise ValueError("Manifest pair count does not match cases × replicates")
    for pair_id, members in pairs.items():
        if len(members) != 2 or {row.get("arm") for _, row in members} != set(FORMAL_ARMS):
            raise ValueError(f"Pair {pair_id} must contain one B and one G run")
        if members[1][0] != members[0][0] + 1:
            raise ValueError(f"Pair {pair_id} must remain adjacent in the frozen schedule")
    first_arm_counts = {arm: 0 for arm in FORMAL_ARMS}
    for members in pairs.values():
        first_arm_counts[str(members[0][1]["arm"])] += 1
    return {
        "planned_cases": len(case_ids),
        "replicates_per_case": replicates,
        "planned_pairs": expected_pairs,
        "planned_units": expected_units,
        "run_id_count": len(set(run_ids)),
        "arm_counts": {arm: sum(row["arm"] == arm for row in entries) for arm in FORMAL_ARMS},
        "first_arm_pair_counts": first_arm_counts,
        "all_not_started": all(row["status"] == "not_started" for row in entries),
    }


def candidate_operation_contract(case: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    """Resolve semantic task fields once; literal tool schemas may differ by arm."""
    if arm not in FORMAL_ARMS:
        raise ValueError(f"Unsupported formal B/G arm: {arm}")
    operation = str(case.get("operation") or "")
    semantic_route = (B_ACTION_BY_OPERATION if arm == "B" else G_SEMANTIC_ROUTE_BY_OPERATION).get(operation)
    if operation == "dimension_contribution" and (case.get("period") or {}).get("kind") == "month_pair":
        semantic_route = "period_metric_contribution" if arm == "B" else "metric_contribution"
    return {
        "operation": case.get("operation"),
        "metric_id": case.get("target_metric"),
        "scope": case.get("scope"),
        "period": case.get("period"),
        "dimension": case.get("dimension"),
        "method_id": case.get("method_id"),
        "dependency_metric_ids": case.get("dependency_metric_ids", []),
        "arm": arm,
        "semantic_route": semantic_route,
    }


def verify_data_insufficiency_precondition(
    expected_baseline_value: Any, observations: Sequence[Mapping[str, Any]], *, tolerance: float = 0.011,
) -> dict[str, Any]:
    """Require a matching valid baseline and explicit missing current side.

    A compare result can correctly have overall status ``not_found`` while still
    carrying one valid side. That status must not erase or invalidate the
    baseline observation.
    """
    baseline_matches = any(
        row.get("status") == "not_found"
        and row.get("baseline_value") is not None
        and expected_baseline_value is not None
        and abs(float(row["baseline_value"]) - float(expected_baseline_value)) <= tolerance
        for row in observations
    )
    current_missing = any(
        row.get("status") == "not_found" and row.get("current_value") is None
        for row in observations
    )
    return {
        "baseline_value_matches_independent_csv": baseline_matches,
        "current_period_has_explicit_not_found": current_missing,
        "verified": baseline_matches and current_missing,
    }


def score_formal_run(
    case: Mapping[str, Any], *, observation: Mapping[str, Any],
    lifecycle_status: str, arm: str = "B", human_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score the same public-result contract for either B or G.

    Governance execution (for example whether G's Critic actually ran) is
    deliberately returned separately and never substitutes for task truth.
    """
    if case.get("truth_status") != "verified":
        return {
            "scorer_version": SCORER_VERSION,
            "task_acceptance_pass": None,
            "task_acceptance_status": "blocked_truth_not_verified",
            "manual_review_required": True,
            "truth_status": case.get("truth_status", "unknown"),
            "reason": "Case truth is not independently verified; no pass/fail inference is allowed.",
            "lifecycle_status": lifecycle_status,
            "experiment_unit_integrity_pass": None,
        }
    expected = case.get("expected") or {}
    checks: dict[str, bool | None] = {}
    checks["lifecycle_completed"] = lifecycle_status == "completed"
    def exact_check(field: str, expected_value: Any) -> bool | None:
        if field not in observation or observation.get(field) is None:
            return None
        return observation.get(field) == expected_value
    expected_operation = case.get("operation")
    expected_operation_by_arm = {
        "B": case.get("b_action_contract"),
        "G": case.get("g_semantic_route"),
    }
    if expected_operation == "expected_data_insufficiency" and expected_operation_by_arm.get(arm):
        # This labels the expected outcome, while the per-arm contract names
        # the query operation that must produce that outcome (for example,
        # metric_compare returning not_found). The observation adapter may
        # expose either the concrete operation or the verified outcome label.
        # Preserve the legacy abstract operation when no explicit route exists.
        actual_operation = observation.get("operation")
        checks["operation_match"] = (
            None if actual_operation is None
            else actual_operation in {expected_operation, expected_operation_by_arm[arm]}
        )
    else:
        checks["operation_match"] = exact_check("operation", expected_operation)
    checks["metric_match"] = exact_check("metric_id", case.get("target_metric"))
    checks["scope_match"] = exact_check("scope", case.get("scope"))
    if "dimension" in case:
        if observation.get("dimension_match") is not None:
            checks["dimension_match"] = observation.get("dimension_match") is True
        else:
            checks["dimension_match"] = exact_check("dimension", case.get("dimension"))
    if observation.get("period_match") is not None:
        checks["period_match"] = observation.get("period_match") is True
    else:
        checks["period_match"] = exact_check("period", case.get("period"))
    checks["evidence_valid"] = observation.get("evidence_valid") if isinstance(observation.get("evidence_valid"), bool) else None
    checks["output_consistent"] = observation.get("answer_card_consistent") if isinstance(observation.get("answer_card_consistent"), bool) else None
    expected_outcome = str(expected.get("outcome") or "")
    if expected_outcome == "safe_stop":
        checks["expected_precondition_verified"] = observation.get("precondition_verified") if isinstance(observation.get("precondition_verified"), bool) else None
        if isinstance(observation.get("expected_stop"), bool):
            checks["expected_stop"] = observation.get("expected_stop")
        else:
            checks["expected_stop"] = exact_check("terminal_code", expected.get("terminal_code"))
        checks["limitation_disclosed"] = observation.get("limitation_disclosed") if isinstance(observation.get("limitation_disclosed"), bool) else None
    elif expected_outcome == "numeric_result":
        checks["numeric_values_match"] = observation.get("numeric_values_match") if isinstance(observation.get("numeric_values_match"), bool) else None
        checks["unit_match"] = observation.get("unit_match") if isinstance(observation.get("unit_match"), bool) else None
    else:
        checks["expected_outcome_defined"] = False
    governance_expected = (case.get("governance_requirements_by_arm") or {}).get(arm)
    governance_actual = observation.get("governance_execution", "not_applicable")
    governance_ok = governance_expected is None or governance_actual == governance_expected
    unknown = [name for name, value in checks.items() if value is None]
    manual = bool(unknown or observation.get("manual_review_required"))
    if human_review is not None:
        manual = False
        human_pass = human_review.get("task_acceptance_pass")
        passed = human_pass is True and all(value is True for value in checks.values() if value is not None)
    else:
        passed = not manual and all(value is True for value in checks.values())
    return {
        "scorer_version": SCORER_VERSION,
        "task_acceptance_pass": passed if not manual or human_review is not None else None,
        "task_acceptance_status": "pass" if passed else ("manual_review_required" if manual else "fail"),
        "manual_review_required": manual,
        "checks": checks,
        "unknown_checks": unknown,
        "lifecycle_status": lifecycle_status,
        "effective_diagnostic_completed": observation.get("effective_diagnostic_completed") is True,
        "correct_safe_stop": (
            checks.get("expected_precondition_verified") is True
            and checks.get("expected_stop") is True
            and checks.get("limitation_disclosed") is True
        ),
        "governance_execution": governance_actual,
        "governance_execution_satisfied": governance_ok,
        "experiment_unit_integrity_pass": bool(passed and governance_ok and lifecycle_status == "completed") if not manual or human_review is not None else None,
        "experiment_unit_status": (
            "pass" if passed and governance_ok and lifecycle_status == "completed"
            else "governance_execution_incomplete" if passed and not governance_ok
            else "manual_review_required" if manual
            else "fail"
        ),
    }


def compare_operation_contracts(
    case: Mapping[str, Any], *, b_contract: Mapping[str, Any], g_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless both arms declare equivalent semantic operation inputs."""
    required = {
        "operation": case.get("operation"),
        "metric_id": case.get("target_metric"),
        "scope": case.get("scope"),
        "period": case.get("period"),
        "dimension": case.get("dimension"),
        "method_id": case.get("method_id"),
        "dependency_metric_ids": case.get("dependency_metric_ids", []),
    }
    mismatches: list[str] = []
    for field, expected in required.items():
        if b_contract.get(field) != expected:
            mismatches.append(f"B.{field}")
        if g_contract.get(field) != expected:
            mismatches.append(f"G.{field}")
        if b_contract.get(field) != g_contract.get(field):
            mismatches.append(f"arms.{field}")
    return {
        "case_id": case.get("case_id"),
        "status": "equivalent" if not mismatches else "blocked_contract_mismatch",
        "mismatches": mismatches,
        "expected_contract": required,
    }
