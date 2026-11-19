"""Conservative offline routing of explicitly labelled answer spans.

Routing is not correctness: deterministic eligibility requires an actual
evidence-based check covering the full claim. Incomplete annotation escalates
the whole answer. This prototype is not connected to the workflow.
"""
from __future__ import annotations

from typing import Any

from evaluation.claim_verification.annotation_schema import ROUTE_DETERMINISTIC, route_for


def plan(case: dict[str, Any], annotations: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    answer = case.get("answer") or ""
    claims = annotations.get("claims") or []
    requirements = case.get("task_requirements") or []
    reasons = []
    if annotations.get("oracle_claim_labels") is not True or annotations.get("coverage_verified") is not True:
        reasons.append("claim_coverage_unverified")
    if not claims:
        reasons.append("missing_claims")
    if any((annotations.get("requirement_coverage") or {}).get(req) not in ("covered", "missing") for req in requirements):
        reasons.append("requirement_coverage_unverified")
    decisions = []
    ids = set()
    spans = []
    for claim in claims:
        if not isinstance(claim, dict):
            reasons.append("malformed_claim")
            continue
        claim_id, start, end = claim.get("claim_id"), claim.get("start"), claim.get("end")
        if not isinstance(claim_id, str) or not claim_id or claim_id in ids:
            reasons.append("invalid_or_duplicate_claim_id")
            continue
        ids.add(claim_id)
        if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool) or not 0 <= start < end <= len(answer) or answer[start:end] != claim.get("text_span"):
            reasons.append("invalid_claim_span")
            continue
        if any(start < prior_end and prior_start < end for prior_start, prior_end in spans):
            reasons.append("overlapping_claim_spans")
        spans.append((start, end))
        kind = claim.get("claim_type") or "unknown"
        refs = claim.get("evidence_ids") or []
        # A numerical check must cover the *entire* claim, cite the same source,
        # and have a real Boolean result. A nearby number is not full coverage.
        relevant = [c for c in checks if isinstance(c, dict) and c.get("start") == start and c.get("end") == end and c.get("evidence_id") in refs and type(c.get("matched")) is bool]
        route = ROUTE_DETERMINISTIC if route_for(kind) == ROUTE_DETERMINISTIC and relevant else "semantic"
        decisions.append({"claim_id": claim_id, "route": route, "check_status": ("mismatch" if any(not c["matched"] for c in relevant) else "matched") if route == "deterministic" else "unverified"})
    if any((annotations.get("requirement_coverage") or {}).get(req) == "missing" for req in requirements):
        reasons.append("task_requirement_missing")
    # Claim-level numeric checks do not establish cross-claim consistency or
    # task completeness; these remain mandatory semantic review dimensions.
    return {"claim_decisions": decisions, "semantic_scope": "whole_answer" if reasons else "claim_batch_and_global_checks", "global_checks": ["cross_claim_consistency", "task_completeness"], "escalation_reasons": sorted(set(reasons)), "oracle_claim_labels": annotations.get("oracle_claim_labels") is True, "whole_answer_verified": False}
