"""Frozen annotation contract for the Claim-routing benchmark.

This module is the only place that defines claim types, their routing, and the
row shape a human reviewer fills in.  Annotating 30 cases is expensive; the
schema must be frozen before the first human label is recorded, otherwise the
labels become unusable.
"""
from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "claim-annotation-v1"

# Routing is a property of the label, never of the claim's correctness.  A
# deterministic label still requires an actual covering check before it can be
# passed; see claim_routing.plan().
ROUTE_DETERMINISTIC = "deterministic"
ROUTE_SEMANTIC = "semantic"

# `limitation_or_refusal` is semantic on purpose: deciding that an answer
# correctly stopped requires judging what the system could and could not do,
# which the current probes cannot establish from a single evidence bundle.
CLAIM_ROUTES: dict[str, str] = {
    "numeric_fact": ROUTE_DETERMINISTIC,
    "calculation": ROUTE_DETERMINISTIC,
    "comparison": ROUTE_DETERMINISTIC,
    "definition_or_scope": ROUTE_DETERMINISTIC,
    "evidence_reference": ROUTE_DETERMINISTIC,
    "causal_claim": ROUTE_SEMANTIC,
    "recommendation": ROUTE_SEMANTIC,
    "limitation_or_refusal": ROUTE_SEMANTIC,
    "unknown": ROUTE_SEMANTIC,
}

CLAIM_TYPES = frozenset(CLAIM_ROUTES)
CORRECTNESS_LABELS = frozenset({"supported", "unsupported", "uncertain"})
EVIDENCE_REQUIRED_TYPES = frozenset(
    {"numeric_fact", "calculation", "comparison", "evidence_reference"}
)

ROW_STATUSES = frozenset(
    {"pending_human_review", "assistant_annotated_pending_human_spotcheck", "human_verified"}
)
COVERAGE_STATUSES = frozenset({"covered", "missing", "uncertain"})
DECISION_LABELS = frozenset({"pass", "needs_semantic_review", "reject_or_revise"})
PROVENANCE_LABELS = frozenset({"real_correct", "real_failure", "targeted_mutation"})


def route_for(claim_type: str) -> str:
    """Return the routing class, escalating anything unrecognised."""

    return CLAIM_ROUTES.get(claim_type, ROUTE_SEMANTIC)


def validate_annotation_row(row: dict[str, Any]) -> dict[str, Any]:
    """Validate one reviewer row and report whether it may enter the benchmark.

    Structural validity is not correctness and not readiness.  An `uncertain`
    label is a legal annotation, but it must never make a row ready: unresolved
    labels have to be adjudicated before the row can be scored.
    """

    issues: list[str] = []
    unresolved: list[str] = []
    answer = row.get("answer")
    if not isinstance(answer, str) or not answer:
        issues.append("missing_answer")
        answer = ""

    claims = row.get("claims")
    if not isinstance(claims, list) or not claims:
        issues.append("missing_claims")
        claims = []

    spans: list[tuple[int, int]] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            issues.append(f"claim_{index}_not_object")
            continue
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id:
            issues.append(f"claim_{index}_missing_id")
        start, end = claim.get("start"), claim.get("end")
        valid_span = (
            isinstance(start, int) and not isinstance(start, bool)
            and isinstance(end, int) and not isinstance(end, bool)
            and 0 <= start < end <= len(answer)
            and answer[start:end] == claim.get("text_span")
        )
        if not valid_span:
            issues.append(f"claim_{index}_invalid_span")
        else:
            if any(start < prior_end and prior_start < end for prior_start, prior_end in spans):
                issues.append(f"claim_{index}_overlapping_span")
            spans.append((start, end))
        claim_type = claim.get("claim_type")
        if claim_type not in CLAIM_TYPES:
            issues.append(f"claim_{index}_invalid_type")
        elif claim_type in EVIDENCE_REQUIRED_TYPES and not claim.get("evidence_ids"):
            issues.append(f"claim_{index}_missing_evidence_binding")
        correctness = claim.get("correctness")
        if correctness not in CORRECTNESS_LABELS:
            issues.append(f"claim_{index}_invalid_correctness")
        elif correctness == "uncertain":
            unresolved.append(f"claim_correctness_uncertain:{claim_id}")
        if claim_type == "unknown":
            unresolved.append(f"claim_type_unknown:{claim_id}")

    requirements = row.get("task_requirements")
    if not isinstance(requirements, list):
        issues.append("missing_task_requirements")
        requirements = []
    coverage = row.get("requirement_coverage")
    if not isinstance(coverage, dict):
        issues.append("missing_requirement_coverage")
        coverage = {}
    for requirement in requirements:
        status = coverage.get(requirement)
        if status not in COVERAGE_STATUSES:
            issues.append(f"requirement_coverage_invalid:{requirement}")
        elif status == "uncertain":
            unresolved.append(f"requirement_coverage_uncertain:{requirement}")

    if row.get("provenance") not in PROVENANCE_LABELS:
        issues.append("invalid_provenance")
    expected_decision = row.get("expected_decision")
    if expected_decision not in DECISION_LABELS:
        issues.append("invalid_expected_decision")
    elif expected_decision == "pass" and any(
        claim.get("correctness") != "supported" for claim in claims if isinstance(claim, dict)
    ):
        issues.append("pass_decision_with_unsupported_claim")
    verified = row.get("human_verified") is True
    if row.get("annotation_status") not in ROW_STATUSES:
        issues.append("invalid_annotation_status")
    if row.get("annotation_status") == "human_verified" and not verified:
        issues.append("human_verified_flag_mismatch")
    if row.get("oracle_claim_labels") is True and not verified:
        issues.append("oracle_without_human_verification")
    if not verified:
        unresolved.append("human_verification_pending")

    return {
        "schema_version": SCHEMA_VERSION,
        "issues": sorted(set(issues)),
        "unresolved": sorted(set(unresolved)),
        "structurally_valid": not issues,
        "ready_for_benchmark": not issues and not unresolved,
        "oracle_claim_labels": bool(row.get("oracle_claim_labels")) and verified,
    }
