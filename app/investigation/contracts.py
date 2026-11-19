"""Strict contracts for evidence-grounded, bounded investigations."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from app.core.failure_context import ExecutionFailureContext


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class _FrozenDict(dict):
    """JSON-serializable dict that rejects in-place mutation."""

    def _immutable(self, *_args, **_kwargs):
        raise TypeError("analysis target scope is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_thaw(item) for item in sorted(value, key=repr)]
    return value


class AnalysisTarget(StrictModel):
    """Immutable root intent; child investigations never replace this target."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    metric_id: str = Field(min_length=1, max_length=100)
    current_start: date
    current_end: date
    baseline_start: date | None = None
    baseline_end: date | None = None
    canonical_scope: dict[str, Any]
    scope_fingerprint: str = Field(min_length=1)
    requested_analysis_type: Literal["contribution", "diagnostic", "trend", "distribution"]
    metric_definition_version: str = Field(default="1.0", min_length=1)
    method_version: str = Field(default="1.0", min_length=1)

    @model_validator(mode="after")
    def validate_periods(self) -> "AnalysisTarget":
        if self.current_end < self.current_start:
            raise ValueError("analysis target periods must be ordered")
        if (self.baseline_start is None) != (self.baseline_end is None):
            raise ValueError("baseline_start and baseline_end must both be set or both be omitted")
        if self.baseline_start is not None and self.baseline_end < self.baseline_start:
            raise ValueError("analysis target periods must be ordered")
        object.__setattr__(self, "canonical_scope", _freeze(self.canonical_scope))
        return self

    @field_serializer("canonical_scope")
    def serialize_canonical_scope(self, value: dict[str, Any]) -> dict[str, Any]:
        return _thaw(value)


class ComparePeriodAction(StrictModel):
    action_type: Literal["COMPARE_PERIOD"]
    metric_id: str


class DecomposeMetricAction(StrictModel):
    action_type: Literal["DECOMPOSE_METRIC"]
    metric_id: str
    method_id: str
    factor_metric_ids: list[str] = Field(min_length=2, max_length=5)


class BreakdownDimensionAction(StrictModel):
    action_type: Literal["BREAKDOWN_DIMENSION", "DRILL_DOWN"]
    metric_id: str
    dimension: str
    parent_evidence_id: str | None = None
    inherited_filters: dict[str, Any] = Field(default_factory=dict)
    method_id: str


class AnalyzeTrendAction(StrictModel):
    action_type: Literal["ANALYZE_TREND"]
    metric_id: str


class AnalyzeDistributionAction(StrictModel):
    action_type: Literal["ANALYZE_DISTRIBUTION"]
    metric_id: str
    statistic: Literal["quantile_shift", "distribution_summary"]


class AnalyzeEntityDistributionAction(StrictModel):
    """Distribution over registered business entities, not time buckets."""

    action_type: Literal["ANALYZE_ENTITY_DISTRIBUTION"]
    metric_id: str
    entity_type: Literal["seller_order"] = "seller_order"


class AnalyzeAssociationAction(StrictModel):
    """Run one server-registered association pair over the locked scope."""

    action_type: Literal["ANALYZE_ASSOCIATION"]
    metric_id: str
    pair_id: str = Field(min_length=1, max_length=100)


class CheckAnomalyAction(StrictModel):
    action_type: Literal["CHECK_ANOMALY"]
    metric_id: str


class CheckDataQualityAction(StrictModel):
    action_type: Literal["CHECK_DATA_QUALITY"]
    metric_id: str
    check_id: str


AnalysisAction = Annotated[
    Union[
        ComparePeriodAction, DecomposeMetricAction, BreakdownDimensionAction,
        AnalyzeTrendAction, AnalyzeDistributionAction, AnalyzeEntityDistributionAction,
        AnalyzeAssociationAction, CheckAnomalyAction,
        CheckDataQualityAction,
    ],
    Field(discriminator="action_type"),
]


class CandidateFeatures(StrictModel):
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Contribution share is not a probability: offsetting effects can make a
    # single branch exceed 100% of the net root change.
    unresolved_target_share: float = Field(default=0.0, ge=0.0)
    dimension_cardinality: int | None = Field(default=None, ge=0)
    data_availability: Literal["available", "partial", "unknown"] = "unknown"
    estimated_query_cost: float = Field(default=1.0, ge=0.0)
    # Deprecated legacy flag. New candidate generation uses the tri-state field.
    data_available: bool | None = None
    evidence_gap: str = ""
    already_explored: bool = False
    rule_score: float = Field(default=0.0, ge=0.0)


class CandidateAction(StrictModel):
    candidate_id: str = Field(min_length=1)
    action: AnalysisAction
    features: CandidateFeatures = Field(default_factory=CandidateFeatures)
    branch_id: str | None = None
    action_fingerprint: str = ""


class CandidateSet(StrictModel):
    """Server-owned snapshot binding opaque candidate handles to one state."""

    candidate_set_id: str = Field(min_length=1, max_length=128)
    investigation_state_version: int = Field(ge=0)
    scope_fingerprint: str = Field(min_length=1, max_length=200)
    authorization_fingerprint: str | None = Field(default=None, max_length=200)
    candidate_ids: list[str] = Field(max_length=6)

    @model_validator(mode="after")
    def validate_candidate_handles(self) -> "CandidateSet":
        if any(not item for item in self.candidate_ids) or len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("candidate set handles must be non-empty and unique")
        return self


class AgentDecision(StrictModel):
    """Constrained selector response; execution still resolves server candidates."""

    candidate_set_id: str = Field(min_length=1, max_length=128)
    decision: Literal["select_candidate", "soft_stop"]
    candidate_id: str | None = Field(default=None, max_length=100)
    rationale: str = Field(default="", max_length=1000)
    soft_stop_reason: Literal["ENOUGH_EVIDENCE", "NO_HIGH_VALUE_ACTION"] | None = None

    @model_validator(mode="after")
    def validate_decision_shape(self) -> "AgentDecision":
        if self.decision == "select_candidate":
            if not self.candidate_id or self.soft_stop_reason is not None:
                raise ValueError("candidate selection requires candidate_id and forbids soft_stop_reason")
        elif self.candidate_id is not None or self.soft_stop_reason is None:
            raise ValueError("soft stop requires a reason and forbids candidate_id")
        return self


class WorkflowSelection(StrictModel):
    """Trusted server-side fallback selection, never parsed from model output."""

    candidate_id: str = Field(min_length=1, max_length=100)


class CalculationProgress(StrictModel):
    """Quality of one calculation; deliberately separate from investigation status."""

    action_id: str | None = None
    reconciliation_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    residual_value: float | None = None
    residual_type: Literal["coverage", "rounding", "model", "unknown"] = "unknown"
    residual_explainable: bool = False
    residual_tolerance: float | None = Field(default=None, ge=0.0)
    residual_note: str = ""
    data_quality_status: Literal[
        "valid", "empty_result", "invalid_data", "incomplete_result",
        "undefined_denominator", "unknown",
    ] = "unknown"


class DataValidationResult(StrictModel):
    """Per-action executor validation, distinct from business-quality findings."""

    status: Literal[
        "valid", "empty_result", "invalid_data", "incomplete_result",
        "undefined_denominator", "unknown",
    ] = "unknown"
    checks: list[dict[str, Any]] = Field(default_factory=list)
    failed_fields_or_groups: list[str] = Field(default_factory=list)


class ActionResult(StrictModel):
    action_id: str
    candidate_id: str
    candidate_set_id: str | None = None
    decision_source: Literal["agent", "workflow_fallback", "backend", "unknown"] = "unknown"
    # Persist the server-selected action semantics so traces/evaluations do
    # not depend on a candidate list that is replaced on the next round.
    action_type: str | None = None
    metric_id: str | None = None
    dimension: str | None = None
    method_id: str | None = None
    parent_evidence_id: str | None = None
    branch_id: str | None = None
    action_fingerprint: str = ""
    # None is reserved for legacy checkpoints that predate explicit outcome
    # status; never infer that an old result was successful.
    status: Literal["completed", "capability_gap", "denied", "failed"] | None = None
    failure_code: str | None = None
    failure_detail: str = ""
    failure_context: ExecutionFailureContext | None = None
    observed_impact: float | None = None
    concentration: float | None = None
    explained_share: float | None = None
    # Missing residual is unknown, not a measured zero. This is especially
    # important when loading legacy ActionResult snapshots without this field.
    residual_value: float | None = None
    residual_type: Literal["coverage", "rounding", "model", "unknown"] = "unknown"
    residual_explainable: bool = False
    residual_tolerance: float | None = Field(default=None, ge=0.0)
    residual_note: str = ""
    data_quality: dict[str, Any] = Field(default_factory=dict)
    calculation_progress: CalculationProgress | None = None
    validation_result: DataValidationResult = Field(default_factory=DataValidationResult)
    evidence_ids: list[str] = Field(default_factory=list)


class UnresolvedBranch(StrictModel):
    branch_id: str
    # The initial root branch has no parents. New descendants carry both links;
    # parent_branch_id remains optional while loading pre-tree checkpoints.
    parent_branch_id: str | None = None
    parent_evidence_id: str | None = None
    branch_type: Literal[
        "root_target", "metric_effect", "dimension_contribution",
        "business_driver", "data_quality",
    ] = "dimension_contribution"
    metric_id: str
    dimension: str = ""
    dimension_value: str = ""
    effect_value: float | None = None
    effect_unit: str | None = None
    # Root-metric impact is separate from the branch metric's local effect.
    # Only fill it when a registered method can map the contribution to root units.
    root_attributed_effect_value: float | None = None
    root_effect_share: float | None = Field(default=None, ge=0.0)
    # Deprecated checkpoint compatibility field. New code writes it only when
    # root_effect_share is mathematically defined.
    effect_share_of_root: float | None = Field(default=None, ge=0.0)
    depth: int = Field(ge=0, le=3)
    inherited_filters: dict[str, str] = Field(default_factory=dict, max_length=3)
    status: Literal["open", "investigating", "explained", "low_priority", "blocked", "skipped"] = "open"
    # A measured dimension contribution is still open for deeper attribution.
    explanation_level: Literal["statistical_contribution"] | None = None
    label: str = ""

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_branch_type(cls, value: Any) -> Any:
        if not isinstance(value, dict) or value.get("branch_type"):
            return value
        value = dict(value)
        if (
            value.get("depth") == 0
            and value.get("parent_branch_id") is None
            and value.get("parent_evidence_id") is None
            and not value.get("dimension")
        ):
            value["branch_type"] = "root_target"
        elif value.get("dimension") == "__metric_factor__":
            value["branch_type"] = "metric_effect"
        return value

    @model_validator(mode="after")
    def validate_parent_evidence(self) -> "UnresolvedBranch":
        if self.parent_branch_id == self.branch_id:
            raise ValueError("an unresolved branch cannot parent itself")
        if self.parent_branch_id is not None and self.parent_evidence_id is None:
            raise ValueError("a child branch must reference the Evidence that created it")
        if self.parent_evidence_id is None and self.depth != 0:
            raise ValueError("a non-root unresolved branch must reference parent evidence")
        if self.root_effect_share is not None and self.effect_share_of_root is not None:
            if abs(self.root_effect_share - self.effect_share_of_root) > 1e-12:
                raise ValueError("legacy and canonical root effect shares disagree")
        if self.root_effect_share is not None and self.root_attributed_effect_value is None:
            raise ValueError("root effect share requires a root-attributed effect value")
        return self


def branch_share_for_priority(branch: UnresolvedBranch) -> float:
    """Rank unquantified branches below measured positive-impact branches."""
    if branch.root_effect_share is not None:
        return branch.root_effect_share
    # Legacy snapshots may not have the unit/provenance fields required by the
    # current contract. Keep the old value readable, but do not let it outrank
    # typed branches in new investigations.
    if branch.parent_branch_id is None and branch.parent_evidence_id is not None:
        return branch.effect_share_of_root if branch.effect_share_of_root is not None else -1.0
    return -1.0


class InvestigationProgress(StrictModel):
    open_branch_count: int = Field(default=0, ge=0)
    highest_open_branch_share: float = Field(default=0.0, ge=0.0)
    max_remaining_action_score: float = Field(default=0.0, ge=0.0)
    remaining_budget: int = Field(default=0, ge=0)
    capability_gap: bool = False
    authorization_blocked: bool = False
    investigation_progress: Literal[
        "not_started", "in_progress", "stalled", "sufficient", "blocked",
        "sufficiently_investigated", "capability_limited", "authorization_blocked",
        "budget_exhausted", "no_high_value_action", "data_insufficient",
        "invalid_data", "execution_failed", "user_scope_limit",
    ] = "not_started"


class AttributionEvidence(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=200)
    # Optional only for legacy/offline evidence snapshots created before the
    # trace linkage contract existed. Runtime adapters should always populate it.
    trace_id: str | None = Field(default=None, min_length=1, max_length=200)
    evidence_type: str = Field(min_length=1, max_length=100)
    metric_id: str = Field(min_length=1, max_length=100)
    claim: str = ""
    values: dict[str, Any] = Field(default_factory=dict)
    unit: str = ""
    source_action_id: str = Field(min_length=1, max_length=200)
    source_ref: str = Field(min_length=1, max_length=500)
    scope_fingerprint: str = Field(min_length=1, max_length=200)
    method_id: str = Field(min_length=1, max_length=100)
    method_version: str = Field(min_length=1, max_length=50)
    authorized_metric_provenance: str = Field(min_length=1, max_length=100)
    comparison: dict[str, Any] = Field(default_factory=dict)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class StopReason(StrictModel):
    code: Literal[
        "ENOUGH_EVIDENCE", "NO_HIGH_VALUE_ACTION", "CAPABILITY_GAP",
        "DATA_INSUFFICIENT", "BUDGET_EXHAUSTED", "USER_SCOPE_LIMIT",
        "AUTHORIZATION_BLOCKED", "INVALID_DATA", "EXECUTION_FAILED", "COMPLETED",
    ]
    detail: str = ""
    initiator: Literal["backend", "agent", "policy", "budget"]
    failure_context: ExecutionFailureContext | None = None


class InvestigationState(StrictModel):
    target: AnalysisTarget
    state_version: int = Field(default=0, ge=0)
    candidate_set: CandidateSet | None = None
    current_focus_branch_id: str | None = None
    root_effect_value: float | None = None
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[AttributionEvidence] = Field(default_factory=list)
    branches: list[UnresolvedBranch] = Field(default_factory=list)
    candidate_actions: list[CandidateAction] = Field(default_factory=list)
    policy_rejections: list[dict[str, str]] = Field(default_factory=list)
    action_results: list[ActionResult] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    executed_fingerprints: list[str] = Field(default_factory=list)
    remaining_budget: int = Field(default=6, ge=0, le=6)
    drill_depth_limit: int = Field(default=3, ge=0, le=3)
    progress: InvestigationProgress = Field(default_factory=InvestigationProgress)
    calculation_progress: CalculationProgress = Field(default_factory=CalculationProgress)
    stop_reason: StopReason | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_reconciliation_progress(cls, value: Any) -> Any:
        """Move pre-split progress ratios out of InvestigationProgress safely."""
        if not isinstance(value, dict):
            return value
        value = dict(value)
        progress = value.get("progress")
        if isinstance(progress, dict) and "reconciliation_ratio" in progress:
            progress = dict(progress)
            ratio = progress.pop("reconciliation_ratio")
            value["progress"] = progress
            calc = value.get("calculation_progress")
            if not isinstance(calc, dict):
                calc = {}
            if "reconciliation_ratio" not in calc:
                calc["reconciliation_ratio"] = ratio
            if calc.get("data_quality_status") is None:
                calc["data_quality_status"] = "unknown"
            value["calculation_progress"] = calc
        return value

    @model_validator(mode="after")
    def validate_branch_lineage(self) -> "InvestigationState":
        branches = {branch.branch_id: branch for branch in self.branches}
        if len(branches) != len(self.branches):
            raise ValueError("investigation branch IDs must be unique")
        for branch in self.branches:
            if branch.parent_branch_id is not None:
                parent = branches.get(branch.parent_branch_id)
                if parent is None:
                    raise ValueError("unresolved branch references a missing parent branch")
                if branch.parent_evidence_id is None:
                    raise ValueError("child branch must retain its source Evidence reference")
                source_evidence = next((
                    item for item in self.evidence
                    if item.evidence_id == branch.parent_evidence_id
                ), None)
                if source_evidence is None:
                    raise ValueError("child branch references Evidence missing from the investigation")
                source_result = next((
                    item for item in self.action_results
                    if item.status == "completed"
                    and item.validation_result.status == "valid"
                    and branch.parent_evidence_id in item.evidence_ids
                ), None)
                if source_result is None:
                    raise ValueError("child branch source Evidence must belong to a completed validated action")
                if source_evidence.scope_fingerprint != self.target.scope_fingerprint:
                    raise ValueError("child branch source Evidence scope differs from its root target")
                if (
                    source_result.candidate_id != source_evidence.source_action_id
                    or source_result.branch_id != branch.parent_branch_id
                    or source_result.metric_id != source_evidence.metric_id
                ):
                    raise ValueError("child branch lineage does not match its source ActionResult")
                if any(
                    parent.inherited_filters.get(key) != value
                    for key, value in branch.inherited_filters.items()
                    if key in parent.inherited_filters
                ):
                    raise ValueError("child branch filters cannot contradict parent filters")
                visited = {branch.branch_id}
                ancestor = parent
                while ancestor.parent_branch_id is not None:
                    if ancestor.branch_id in visited:
                        raise ValueError("investigation branch lineage cannot contain cycles")
                    visited.add(ancestor.branch_id)
                    ancestor = branches.get(ancestor.parent_branch_id)
                    if ancestor is None:
                        raise ValueError("investigation branch references a missing ancestor")
                if ancestor.branch_id in visited:
                    raise ValueError("investigation branch lineage cannot contain cycles")
        if self.current_focus_branch_id is not None and self.current_focus_branch_id not in branches:
            raise ValueError("current focus must reference an existing investigation branch")
        return self


class InvestigationSummary(StrictModel):
    """Auditable terminal snapshot for a bounded report/chat investigation."""

    metric_id: str
    scope_fingerprint: str
    stop_reason: StopReason
    progress: InvestigationProgress
    calculation_progress: CalculationProgress = Field(default_factory=CalculationProgress)
    actions_used: int = Field(ge=0, le=6)
    remaining_budget: int = Field(ge=0, le=6)
    branches: list[UnresolvedBranch] = Field(default_factory=list)
    action_results: list[ActionResult] = Field(default_factory=list)
    policy_rejections: list[dict[str, str]] = Field(default_factory=list)
    decision_events: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_reconciliation_progress(cls, value: Any) -> Any:
        """Keep previously persisted summary snapshots readable after field split."""
        if not isinstance(value, dict):
            return value
        value = dict(value)
        progress = value.get("progress")
        if isinstance(progress, dict) and "reconciliation_ratio" in progress:
            progress = dict(progress)
            ratio = progress.pop("reconciliation_ratio")
            value["progress"] = progress
            calc = value.get("calculation_progress")
            if not isinstance(calc, dict):
                calc = {}
            calc.setdefault("reconciliation_ratio", ratio)
            calc.setdefault("data_quality_status", "unknown")
            value["calculation_progress"] = calc
        return value


class AttributionComponent(StrictModel):
    component_id: str
    value: float
    unit: str = ""
    component_type: str = "contribution"


class AttributionFactorIdentity(StrictModel):
    """Registry-owned identity for a method component such as a ratio role."""

    role_id: str
    metric_id: str
    display_name: str


class AttributionResult(StrictModel):
    metric_id: str
    method_id: str
    method_version: str
    baseline_value: float
    current_value: float
    total_delta: float
    raw_components: list[AttributionComponent] = Field(default_factory=list)
    display_method: str = "raw"
    display_components: list[AttributionComponent] = Field(default_factory=list)
    reconciliation_ratio: float = Field(ge=0.0, le=1.0)
    residual_value: float = 0.0
    residual_type: Literal["coverage", "rounding", "model", "unknown"] = "unknown"
    residual_explainable: bool = False
    residual_tolerance: float = Field(default=0.0, ge=0.0)
    residual_note: str = ""
    # Backward-compatible optional metadata. Producers of registered ratio
    # decompositions must populate and validate this role-to-metric mapping.
    factor_identities: list[AttributionFactorIdentity] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    causal_claim: Literal[False] = False

    @model_validator(mode="after")
    def validate_factor_identity_shape(self) -> "AttributionResult":
        roles = [item.role_id for item in self.factor_identities]
        if len(roles) != len(set(roles)):
            raise ValueError("factor identity roles must be unique")
        if self.factor_identities and self.method_id != "symmetric_ratio_v1":
            raise ValueError("factor identities are only defined for registered ratio roles")
        if self.factor_identities and set(roles) != {"numerator_effect", "denominator_effect"}:
            raise ValueError("registered ratio factor identities must map both effect roles")
        return self
