"""Path-quality evaluation for bounded investigations.

The evaluator checks obligations and evidence provenance, not one golden
action sequence. A case may declare alternative actions for each obligation;
their order is intentionally unconstrained.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from app.investigation.contracts import InvestigationState, StrictModel


class ActionMatcher(StrictModel):
    action_type: str
    metric_id: str | None = None
    dimension: str | None = None
    method_id: str | None = None

    def matches(self, action_result) -> bool:
        return all((expected is None or getattr(action_result, field) == expected) for field, expected in (
            ("action_type", self.action_type),
            ("metric_id", self.metric_id),
            ("dimension", self.dimension),
            ("method_id", self.method_id),
        ))


class EvidenceMatcher(StrictModel):
    evidence_type: str
    metric_id: str | None = None
    method_id: str | None = None

    def matches(self, evidence) -> bool:
        return all((expected is None or getattr(evidence, field) == expected) for field, expected in (
            ("evidence_type", self.evidence_type),
            ("metric_id", self.metric_id),
            ("method_id", self.method_id),
        ))


class BranchMatcher(StrictModel):
    metric_id: str
    dimension: str
    dimension_value: str | None = None
    acceptable_statuses: list[str] = Field(
        default_factory=lambda: ["explained", "low_priority", "blocked", "skipped"],
    )

    def matches(self, branch) -> bool:
        return (
            branch.metric_id == self.metric_id
            and branch.dimension == self.dimension
            and (self.dimension_value is None or branch.dimension_value == self.dimension_value)
        )


class InvestigationEvaluationCase(StrictModel):
    case_id: str = Field(min_length=1)
    target_metric_id: str = Field(min_length=1)
    max_actions: int = Field(ge=0, le=6)
    # Every group is one investigation obligation; any alternative is accepted.
    required_action_groups: list[list[ActionMatcher]] = Field(default_factory=list)
    allowed_action_matchers: list[ActionMatcher] = Field(default_factory=list)
    required_evidence_groups: list[list[EvidenceMatcher]] = Field(default_factory=list)
    required_branch_groups: list[list[BranchMatcher]] = Field(default_factory=list)
    allowed_stop_reasons: list[str] = Field(min_length=1)
    expected_capability_gap: bool | None = None
    require_all_evidence_grounded: bool = True
    require_conclusion_evidence: bool = False


class InvestigationEvaluationResult(StrictModel):
    case_id: str
    passed: bool
    completed_actions: int = Field(ge=0)
    attempted_actions: int = Field(ge=0)
    invalid_action_count: int = Field(ge=0)
    invalid_action_rate: float = Field(ge=0.0, le=1.0)
    fallback_action_count: int = Field(default=0, ge=0)
    fallback_action_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    action_relevance_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    relevant_agent_action_count: int = Field(default=0, ge=0)
    agent_action_count: int = Field(default=0, ge=0)
    action_obligation_coverage: float = Field(ge=0.0, le=1.0)
    evidence_obligation_coverage: float = Field(ge=0.0, le=1.0)
    branch_obligation_coverage: float = Field(ge=0.0, le=1.0)
    evidence_grounding_rate: float = Field(ge=0.0, le=1.0)
    grounded_evidence_count: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    stop_reason_allowed: bool
    stop_reason_code: str
    useful_stop: bool
    capability_gap_recognized: bool | None = None
    within_budget: bool
    duplicate_action_count: int = Field(ge=0)
    issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_relevance_counts(self) -> "InvestigationEvaluationResult":
        if self.relevant_agent_action_count > self.agent_action_count:
            raise ValueError("relevant agent actions cannot exceed agent action count")
        if self.agent_action_count == 0 and self.action_relevance_rate is not None:
            raise ValueError("action relevance is undefined when no Agent action was evaluated")
        if self.agent_action_count > 0 and self.action_relevance_rate is None:
            raise ValueError("action relevance requires an evaluated Agent action")
        if self.grounded_evidence_count > self.evidence_count:
            raise ValueError("grounded evidence count cannot exceed evidence count")
        expected_grounding_rate = (
            self.grounded_evidence_count / self.evidence_count
            if self.evidence_count else 1.0
        )
        if abs(self.evidence_grounding_rate - expected_grounding_rate) > 1e-9:
            raise ValueError("evidence grounding rate must match its evidence counts")
        return self


class InvestigationEvaluationSummary(StrictModel):
    case_count: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    mean_completed_actions: float = Field(ge=0.0)
    invalid_action_rate: float = Field(ge=0.0, le=1.0)
    fallback_action_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_action_relevance_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    relevant_agent_action_count: int = Field(default=0, ge=0)
    agent_action_count: int = Field(default=0, ge=0)
    duplicate_action_rate: float = Field(ge=0.0, le=1.0)
    budget_exhaustion_rate: float = Field(ge=0.0, le=1.0)
    useful_stop_rate: float = Field(ge=0.0, le=1.0)
    capability_gap_recognition_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_grounding_rate: float = Field(ge=0.0, le=1.0)
    grounded_evidence_count: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)


def _group_satisfied(group, values) -> bool:
    return any(matcher.matches(value) for matcher in group for value in values)


def evaluate_investigation_path(
    case: InvestigationEvaluationCase,
    state: InvestigationState,
    *,
    conclusion_evidence_ids: list[str] | None = None,
) -> InvestigationEvaluationResult:
    """Score required evidence/action obligations while allowing alternative paths."""
    issues: list[str] = []
    completed = [result for result in state.action_results if result.status == "completed"]
    attempted = list(state.action_results)
    invalid_count = sum(result.status in {"denied", "failed"} for result in attempted)
    fallback_count = sum(result.decision_source == "workflow_fallback" for result in attempted)
    relevance_rate: float | None = None
    relevant_agent_action_count = 0
    agent_action_count = 0
    evidence_by_id = {item.evidence_id: item for item in state.evidence}

    action_hits = sum(
        _group_satisfied(group, completed)
        for group in case.required_action_groups
    )
    evidence_hits = sum(
        any(matcher.matches(evidence) for matcher in group for evidence in state.evidence)
        for group in case.required_evidence_groups
    )
    action_coverage = action_hits / len(case.required_action_groups) if case.required_action_groups else 1.0
    evidence_coverage = evidence_hits / len(case.required_evidence_groups) if case.required_evidence_groups else 1.0
    branch_hits = sum(
        any(
            matcher.matches(branch) and branch.status in matcher.acceptable_statuses
            for matcher in group for branch in state.branches
        )
        for group in case.required_branch_groups
    )
    branch_coverage = branch_hits / len(case.required_branch_groups) if case.required_branch_groups else 1.0

    grounded_ids: set[str] = set()
    validated_grounded_ids: set[str] = set()
    for result in completed:
        for evidence_id in result.evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if (
                evidence is not None
                and evidence.source_action_id == result.candidate_id
                and evidence.scope_fingerprint == state.target.scope_fingerprint
            ):
                grounded_ids.add(evidence_id)
                if result.validation_result.status == "valid":
                    validated_grounded_ids.add(evidence_id)
    grounding_rate = len(grounded_ids) / len(state.evidence) if state.evidence else 1.0
    if case.require_all_evidence_grounded and len(grounded_ids) != len(state.evidence):
        issues.append("unlinked_or_out_of_scope_evidence")

    fingerprints = [
        result.action_fingerprint for result in state.action_results
        if result.action_fingerprint
    ]
    duplicate_count = len(fingerprints) - len(set(fingerprints))
    if duplicate_count:
        issues.append("duplicate_action_fingerprint")
    if action_coverage < 1.0:
        issues.append("required_action_obligation_missing")
    if evidence_coverage < 1.0:
        issues.append("required_evidence_obligation_missing")
    if branch_coverage < 1.0:
        issues.append("required_branch_not_resolved_or_explicitly_deferred")
    if case.require_conclusion_evidence:
        referenced = set(conclusion_evidence_ids or [])
        if not referenced or not referenced <= grounded_ids:
            issues.append("conclusion_evidence_reference_missing_or_ungrounded")
        elif not referenced <= validated_grounded_ids:
            issues.append("conclusion_evidence_missing_or_not_validated")
    if case.allowed_action_matchers:
        agent_attempts = [
            result for result in attempted if result.decision_source == "agent"
        ]
        agent_action_count = len(agent_attempts)
        relevant_agent_action_count = sum(
            any(matcher.matches(result) for matcher in case.allowed_action_matchers)
            for result in agent_attempts
        )
        relevance_rate = (
            relevant_agent_action_count / agent_action_count
            if agent_action_count else None
        )
        if agent_action_count and relevant_agent_action_count != agent_action_count:
            issues.append("irrelevant_agent_action")

    stop_code = state.stop_reason.code if state.stop_reason else ""
    stop_allowed = stop_code in case.allowed_stop_reasons
    useful_stop = stop_allowed and stop_code not in {
        "BUDGET_EXHAUSTED", "EXECUTION_FAILED",
    }
    if not stop_allowed:
        issues.append("stop_reason_not_allowed")
    within_budget = len(state.action_results) <= case.max_actions
    if not within_budget:
        issues.append("action_budget_exceeded")
    if invalid_count:
        issues.append("invalid_or_denied_action")
    if state.target.metric_id != case.target_metric_id:
        issues.append("target_metric_mismatch")
    capability_gap_recognized: bool | None = None
    if case.expected_capability_gap is not None:
        has_capability_gap = stop_code == "CAPABILITY_GAP" or any(
            result.status == "capability_gap" for result in state.action_results
        )
        capability_gap_recognized = has_capability_gap == case.expected_capability_gap
        if has_capability_gap != case.expected_capability_gap:
            issues.append("capability_gap_expectation_mismatch")

    return InvestigationEvaluationResult(
        case_id=case.case_id,
        passed=not issues,
        completed_actions=len(completed),
        attempted_actions=len(attempted),
        invalid_action_count=invalid_count,
        invalid_action_rate=invalid_count / len(attempted) if attempted else 0.0,
        fallback_action_count=fallback_count,
        fallback_action_rate=fallback_count / len(attempted) if attempted else 0.0,
        action_relevance_rate=relevance_rate,
        relevant_agent_action_count=relevant_agent_action_count,
        agent_action_count=agent_action_count,
        action_obligation_coverage=action_coverage,
        evidence_obligation_coverage=evidence_coverage,
        branch_obligation_coverage=branch_coverage,
        evidence_grounding_rate=grounding_rate,
        grounded_evidence_count=len(grounded_ids),
        evidence_count=len(state.evidence),
        stop_reason_allowed=stop_allowed,
        stop_reason_code=stop_code,
        useful_stop=useful_stop,
        capability_gap_recognized=capability_gap_recognized,
        within_budget=within_budget,
        duplicate_action_count=duplicate_count,
        issues=issues,
    )


def summarize_investigation_evaluations(
    results: list[InvestigationEvaluationResult],
) -> InvestigationEvaluationSummary:
    """Aggregate path-quality rates without assuming one golden sequence."""
    total = len(results)
    attempted = sum(result.attempted_actions for result in results)
    invalid = sum(result.invalid_action_count for result in results)
    relevant_agent_actions = sum(result.relevant_agent_action_count for result in results)
    agent_actions = sum(result.agent_action_count for result in results)
    grounded_evidence = sum(result.grounded_evidence_count for result in results)
    evidence_count = sum(result.evidence_count for result in results)
    capability_results = [result.capability_gap_recognized for result in results
                          if result.capability_gap_recognized is not None]
    return InvestigationEvaluationSummary(
        case_count=total,
        passed_cases=sum(result.passed for result in results),
        pass_rate=(sum(result.passed for result in results) / total) if total else 1.0,
        mean_completed_actions=(sum(result.completed_actions for result in results) / total) if total else 0.0,
        invalid_action_rate=(invalid / attempted) if attempted else 0.0,
        fallback_action_rate=(
            sum(result.fallback_action_count for result in results) / attempted
            if attempted else 0.0
        ),
        mean_action_relevance_rate=(
            relevant_agent_actions / agent_actions if agent_actions else None
        ),
        relevant_agent_action_count=relevant_agent_actions,
        agent_action_count=agent_actions,
        duplicate_action_rate=(
            sum(result.duplicate_action_count for result in results) / attempted
            if attempted else 0.0
        ),
        budget_exhaustion_rate=(
            sum(result.stop_reason_code == "BUDGET_EXHAUSTED" for result in results) / total
            if total else 0.0
        ),
        useful_stop_rate=(sum(result.useful_stop for result in results) / total) if total else 0.0,
        capability_gap_recognition_rate=(
            sum(capability_results) / len(capability_results) if capability_results else None
        ),
        evidence_grounding_rate=(
            grounded_evidence / evidence_count if evidence_count else 1.0
        ),
        grounded_evidence_count=grounded_evidence,
        evidence_count=evidence_count,
    )
