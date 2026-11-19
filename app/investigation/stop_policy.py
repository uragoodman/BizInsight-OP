"""Fail-closed stop rules and investigation progress helpers."""

from __future__ import annotations

from app.investigation.contracts import InvestigationProgress, InvestigationState, StopReason


def soft_stop_blockers(state: InvestigationState) -> list[str]:
    """Return deterministic reasons an ENOUGH_EVIDENCE proposal is premature.

    Reconciliation alone is not evidence of investigation completeness. A soft
    stop requires at least one completed, base-validated action with evidence
    and no unresolved open/investigating branch. Legacy or partial outcomes
    whose validation status is unknown cannot prove sufficient evidence. We
    intentionally do not invent a numeric
    contribution threshold: branch prioritization belongs to the configured
    candidate policy, while an open branch is conservatively treated as
    unresolved.
    """
    blockers: list[str] = []
    evidence_by_id = {evidence.evidence_id: evidence for evidence in state.evidence}
    matching_target_results = [
        result
        for result in state.action_results
        if result.status == "completed"
        and any(
            (evidence := evidence_by_id.get(evidence_id)) is not None
            and evidence.source_action_id == result.candidate_id
            and evidence.metric_id == state.target.metric_id
            and evidence.scope_fingerprint == state.target.scope_fingerprint
            for evidence_id in result.evidence_ids
        )
    ]
    if not matching_target_results:
        blockers.append("no_completed_target_scoped_evidence")
    elif not any(
        result.validation_result.status == "valid"
        for result in matching_target_results
    ):
        blockers.append("target_evidence_not_validated")
    unresolved = sorted(
        branch.branch_id
        for branch in state.branches
        if branch.status in {"open", "investigating"}
    )
    if unresolved:
        blockers.append("unresolved_branches:" + ",".join(unresolved))
    return blockers


def hard_stop_reason(
    *,
    remaining_budget: int,
    candidates_available: int,
    authorization_blocked: bool = False,
    scope_blocked: bool = False,
    required_data_missing: bool = False,
    capability_gap: bool = False,
) -> StopReason | None:
    """Return the highest-priority hard stop; the Agent cannot override it."""
    if authorization_blocked:
        return StopReason(
            code="AUTHORIZATION_BLOCKED",
            detail="所有可执行候选均被授权策略阻止。",
            initiator="policy",
        )
    if scope_blocked:
        return StopReason(
            code="USER_SCOPE_LIMIT",
            detail="候选动作超出用户锁定的分析范围。",
            initiator="policy",
        )
    if required_data_missing:
        return StopReason(
            code="DATA_INSUFFICIENT",
            detail="执行所需的数据字段或样本不足。",
            initiator="backend",
        )
    if capability_gap:
        return StopReason(
            code="CAPABILITY_GAP",
            detail="指标或维度没有已验证的归因能力。",
            initiator="backend",
        )
    if remaining_budget <= 0:
        return StopReason(
            code="BUDGET_EXHAUSTED",
            detail="已达到本次调查的数据动作预算。",
            initiator="budget",
        )
    if candidates_available <= 0:
        return StopReason(
            code="NO_HIGH_VALUE_ACTION",
            detail="当前没有未执行的合法候选动作。",
            initiator="backend",
        )
    return None


def progress_snapshot(
    *,
    open_branch_count: int,
    highest_open_branch_share: float,
    max_remaining_action_score: float,
    remaining_budget: int,
    capability_gap: bool = False,
    authorization_blocked: bool = False,
) -> InvestigationProgress:
    """Build investigation-only progress; numeric quality lives elsewhere."""
    if authorization_blocked or capability_gap:
        status = "blocked"
    elif max_remaining_action_score > 0 and remaining_budget > 0:
        status = "in_progress"
    elif open_branch_count > 0 or remaining_budget <= 0:
        status = "stalled"
    else:
        status = "sufficient"
    return InvestigationProgress(
        open_branch_count=open_branch_count,
        highest_open_branch_share=highest_open_branch_share,
        max_remaining_action_score=max_remaining_action_score,
        remaining_budget=remaining_budget,
        capability_gap=capability_gap,
        authorization_blocked=authorization_blocked,
        investigation_progress=status,
    )


def terminal_progress_status(stop_code: str, open_branch_count: int) -> str | None:
    """Translate an authoritative stop reason into a user/audit progress state."""
    mapping = {
        "ENOUGH_EVIDENCE": "sufficiently_investigated",
        "CAPABILITY_GAP": "capability_limited",
        "AUTHORIZATION_BLOCKED": "authorization_blocked",
        "BUDGET_EXHAUSTED": "budget_exhausted",
        "NO_HIGH_VALUE_ACTION": "no_high_value_action",
        "DATA_INSUFFICIENT": "data_insufficient",
        "INVALID_DATA": "invalid_data",
        "EXECUTION_FAILED": "execution_failed",
        "USER_SCOPE_LIMIT": "user_scope_limit",
    }
    if stop_code == "COMPLETED":
        return "no_high_value_action" if open_branch_count else "sufficiently_investigated"
    return mapping.get(stop_code)
