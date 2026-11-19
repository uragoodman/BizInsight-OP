"""Oracle-label routing simulator for Claim-verification phase 1.

This is deliberately *not* a replacement Critic.  It answers only whether a
human-labelled claim could be handled by a deterministic verifier or must be
sent to semantic review.  No LLM verdict, token count, or cost is invented.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from evaluation.claim_verification.annotation_schema import (
    CLAIM_TYPES, EVIDENCE_REQUIRED_TYPES, ROUTE_SEMANTIC, route_for,
)


@dataclass(frozen=True)
class RoutingDecision:
    claim_id: str
    route: str
    reason: str


def route_claim(claim: dict[str, Any]) -> RoutingDecision:
    """Route one *human-labelled* atomic claim conservatively.

    An absent type, absent evidence for a factual claim, or an explicitly
    unknown type is never silently treated as deterministic.
    """
    claim_id = str(claim.get("claim_id") or "unknown")
    claim_type = str(claim.get("claim_type") or "unknown")
    evidence_ids = claim.get("evidence_ids") or []
    if claim_type not in CLAIM_TYPES:
        return RoutingDecision(claim_id, "needs_semantic_review", "unrecognised_type")
    if route_for(claim_type) == ROUTE_SEMANTIC:
        return RoutingDecision(claim_id, "needs_semantic_review", f"type:{claim_type}")
    if claim_type in EVIDENCE_REQUIRED_TYPES and not evidence_ids:
        return RoutingDecision(claim_id, "needs_semantic_review", "missing_evidence_binding")
    return RoutingDecision(claim_id, "deterministic", f"type:{claim_type}")


def evaluate_oracle_case(case: dict[str, Any]) -> dict[str, Any]:
    """Produce a transparent routing record without claiming LLM performance."""
    if case.get("oracle_claim_labels") is not True:
        raise ValueError("Oracle routing requires explicitly verified oracle claim labels")
    claims = case.get("claims") or []
    decisions = [route_claim(item) for item in claims if isinstance(item, dict)]
    requirements = case.get("task_requirements") or []
    coverage = case.get("requirement_coverage") or {}
    missing_requirements = [item for item in requirements if coverage.get(item) is not True]
    semantic_required = any(item.route == "needs_semantic_review" for item in decisions) or bool(missing_requirements)
    deterministic_known_errors = [
        item.get("claim_id") for item in claims
        if isinstance(item, dict)
        and item.get("is_correct") is False
        and route_claim(item).route == "deterministic"
    ]
    return {
        "case_id": case.get("case_id"),
        "oracle_claim_labels": True,
        "routing": [item.__dict__ for item in decisions],
        "needs_semantic_review": semantic_required,
        "missing_requirements": missing_requirements,
        "deterministic_known_error_claim_ids": deterministic_known_errors,
        # A/B/C/D model verdicts and model usage are intentionally absent.
        "llm_verdict": "pending_not_executed",
        "llm_token_usage": "pending_not_executed",
        "llm_call_count": "pending_not_executed",
    }


def evaluate_oracle_cases(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [evaluate_oracle_case(case) for case in cases]
