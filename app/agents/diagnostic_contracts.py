"""Private contracts for the Root Cause diagnostic agent (Phase 1).

Domain fields stay diagnostic-specific.  Since Phase 10 the structural
protocol (budget shape, observation, policy verdict) is inherited from the
shared runtime; business semantics (which actions exist, what evidence means)
remain private to this domain.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import Field, model_validator

from app.agents.contracts import ContractModel
from app.agents.runtime import Budget as RuntimeBudget
from app.agents.runtime import Observation as RuntimeObservation
from app.agents.runtime import PolicyDecision as RuntimePolicyDecision


ActionType = Literal["CALL_TOOL", "CALL_CANDIDATE", "DECOMPOSE_METRIC", "PARALLEL_FACTS", "REPLAN", "FINISH", "ABSTAIN"]
AllowedAction = Literal["CALL_TOOL", "PARALLEL_FACTS", "REPLAN", "FINISH", "ABSTAIN"]

# Server-enforced evidence categories for an anomaly diagnosis.  The LLM may
# plan additional steps, but a plan that skips one of these categories is
# completed (or repaired) by the server — never executed as-is.
StepKind = Literal["trend", "dimension_contribution", "related_metric"]


class DiagnosticBudget(RuntimeBudget):
    # Domain vocabulary for the shared action ceiling, with the diagnostic
    # domain's original (stricter) bounds preserved.
    max_rounds: int = Field(default=6, ge=1, le=32)
    max_tool_calls: int = Field(default=8, ge=0, le=32)


class DiagnosticStep(ContractModel):
    step_id: str = Field(min_length=1, max_length=80)
    objective: str = Field(min_length=1, max_length=500)
    allowed_actions: list[AllowedAction] = Field(min_length=1)
    completion_condition: str = Field(min_length=1, max_length=500)
    # Evidence category this step covers (declared by the LLM or set by the
    # server when it appends a missing skeleton step).
    kind: Optional[StepKind] = None
    # Server-required steps gate terminal actions: an insufficient-evidence
    # ABSTAIN is denied while any required step is still pending.
    required: bool = False


class DiagnosticPlan(ContractModel):
    goal: str = Field(min_length=1, max_length=500)
    hypotheses: list[str] = Field(min_length=1, max_length=20)
    steps: list[DiagnosticStep] = Field(min_length=1, max_length=32)
    stop_conditions: list[str] = Field(min_length=1, max_length=20)
    budget: DiagnosticBudget


class DiagnosticPeriod(ContractModel):
    start_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    granularity: Literal["day", "week", "month"] = "day"


class DiagnosticComparison(ContractModel):
    current_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    baseline_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    current_start: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    current_end: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    baseline_start: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    baseline_end: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    granularity: Optional[Literal["week", "month"]] = None
    filters: dict[str, str] = Field(default_factory=dict, max_length=3)
    parent_evidence_id: Optional[str] = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def validate_comparison_shape(self) -> "DiagnosticComparison":
        point_fields = (self.current_date, self.baseline_date)
        period_fields = (self.current_start, self.current_end, self.baseline_start, self.baseline_end)
        has_point = all(point_fields)
        has_period = all(period_fields)
        if has_point == has_period:
            raise ValueError("comparison requires either two point dates or two complete periods")
        if any(point_fields) and not has_point:
            raise ValueError("point comparison requires current_date and baseline_date")
        if any(period_fields) and not has_period:
            raise ValueError("period comparison requires all four period boundaries")
        if has_point and self.granularity:
            raise ValueError("point comparison cannot set period granularity")
        if has_period and not self.granularity:
            raise ValueError("period comparison requires week or month granularity")
        if self.filters and (not has_period or self.granularity is None or not self.parent_evidence_id):
            raise ValueError("filtered breakdown requires period boundaries and trusted parent evidence")
        if set(self.filters) - {"product_category"}:
            raise ValueError("only evidence-backed product_category filters are supported")
        return self


class DiagnosticFactQuery(ContractModel):
    tool_name: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    dimension: Optional[str] = None
    period: Optional[DiagnosticPeriod] = None
    comparison: Optional[DiagnosticComparison] = None
    limit: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def validate_query_shape(self) -> "DiagnosticFactQuery":
        if self.period is None and self.comparison is None:
            raise ValueError("Diagnostic fact query requires period or comparison")
        if self.period is not None and self.comparison is not None:
            raise ValueError("Diagnostic fact query cannot contain both period and comparison")
        return self


class CallToolAction(ContractModel):
    action_type: Literal["CALL_TOOL"]
    tool_name: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    dimension: Optional[str] = None
    period: Optional[DiagnosticPeriod] = None
    comparison: Optional[DiagnosticComparison] = None
    statistics: list[Literal["count", "mean", "average", "std", "min", "q1", "median", "q3", "max"]] = Field(default_factory=list, max_length=9)
    association_pair_id: str | None = Field(default=None, min_length=1, max_length=100)
    limit: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def validate_query_shape(self) -> "CallToolAction":
        if self.period is None and self.comparison is None:
            raise ValueError("CALL_TOOL requires period or comparison")
        if self.period is not None and self.comparison is not None:
            raise ValueError("CALL_TOOL cannot contain both period and comparison")
        return self


class CallCandidateAction(ContractModel):
    """LLM may select only a server-issued candidate handle."""

    action_type: Literal["CALL_CANDIDATE"]
    candidate_set_id: str = Field(min_length=1, max_length=128)
    candidate_id: str = Field(min_length=1, max_length=100)


class ExecuteDecompositionAction(ContractModel):
    """Internal marker for a selected, server-issued deterministic action.

    The model cannot supply metric IDs or method parameters here; policy
    execution resolves the opaque candidate_id against InvestigationState.
    """

    action_type: Literal["DECOMPOSE_METRIC"]
    candidate_id: str = Field(min_length=1, max_length=100)


class ParallelFactsAction(ContractModel):
    action_type: Literal["PARALLEL_FACTS"]
    facts: list[DiagnosticFactQuery] = Field(min_length=1, max_length=8)


class ReplanAction(ContractModel):
    action_type: Literal["REPLAN"]
    steps: list[DiagnosticStep] = Field(min_length=1, max_length=32)
    stop_conditions: list[str] = Field(min_length=1, max_length=20)
    hypotheses: Optional[list[str]] = Field(default=None, max_length=20)


class FinishAction(ContractModel):
    action_type: Literal["FINISH"]
    conclusion: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(min_length=1, max_length=50)
    residual_uncertainty: str = Field(default="", max_length=1000)
    # Structured numeric attribution: the metric id or dimension whose
    # registered, validated contribution explains the measured change, when
    # the evidence supports a unique factor. This is not a business-causality
    # claim. Scoring reads this field instead of inferring a factor from prose.
    found_factor: Optional[str] = Field(default=None, max_length=80)


class AbstainAction(ContractModel):
    action_type: Literal["ABSTAIN"]
    reason: Literal["insufficient_evidence", "ambiguous_question", "budget_exhausted"]
    supporting_evidence_ids: list[str] = Field(default_factory=list, max_length=50)


DiagnosticAction = Annotated[
    Union[CallToolAction, CallCandidateAction, ExecuteDecompositionAction, ParallelFactsAction, ReplanAction, FinishAction, AbstainAction],
    Field(discriminator="action_type"),
]


class Observation(RuntimeObservation):
    # Only evidence-producing action types may appear on a diagnostic observation.
    action_type: Literal["CALL_TOOL", "DECOMPOSE_METRIC", "PARALLEL_FACTS"]


class FinishDecision(ContractModel):
    conclusion: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(min_length=1, max_length=50)
    residual_uncertainty: str = Field(default="", max_length=1000)


class PolicyDecision(RuntimePolicyDecision):
    # The diagnostic policy layer only ever verdicts on its own action types.
    action_type: ActionType
