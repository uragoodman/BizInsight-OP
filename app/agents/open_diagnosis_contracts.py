"""Contracts for the open business-diagnosis agent (AGENT_PLAN Phase 6 / A1).

Domain fields stay independent from the Root Cause diagnostic contracts: the
open agent consumes deterministic scan findings, asks which anomalies deserve
diagnostic budget, then drills down through the governed diagnostic loop.

Governance encoded here: scan candidates may only come from the deterministic
scanner, problems may only reference scanned metrics, and a report is always
born as ``pending_review`` — findings are never auto-published.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from app.agents.contracts import ContractModel
from app.agents.diagnostic_contracts import DiagnosticBudget
from app.agents.runtime import AgentRun


class ScanCandidate(ContractModel):
    """One deterministic anomaly finding worth considering for diagnosis."""

    metric_name: str = Field(min_length=1, max_length=80)
    metric_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    direction: Literal["up", "down"]
    score: float
    threshold: float
    value: float
    baseline: float
    evidence_ids: list[str] = Field(default_factory=list, max_length=10)


class DiagnosticProblem(ContractModel):
    """A scan-backed problem selected (and prioritised) for drill-down."""

    problem_id: str = Field(min_length=1, max_length=80)
    metric_name: str = Field(min_length=1, max_length=80)
    metric_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    direction: Literal["up", "down"]
    priority: int = Field(ge=1, le=20)
    hypothesis: str = Field(min_length=1, max_length=500)
    scan_evidence_ids: list[str] = Field(default_factory=list, max_length=10)
    budget: DiagnosticBudget


class OpenDiagnosisPlan(ContractModel):
    """LLM-ranked subset of scan candidates that deserve diagnostic budget."""

    goal: str = Field(min_length=1, max_length=500)
    problems: list[DiagnosticProblem] = Field(min_length=1, max_length=8)


class ProblemDiagnosis(ContractModel):
    """Outcome of drilling down one candidate through the governed loop."""

    problem_id: str = Field(min_length=1, max_length=80)
    metric_name: str = Field(min_length=1, max_length=80)
    metric_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    priority: int = Field(ge=1, le=20)
    status: Literal["diagnosed", "abstained", "skipped_cooldown", "skipped_budget", "failed"]
    conclusion: str = Field(default="", max_length=2000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    # Deterministic anomaly-scan evidence is useful context for an abstention,
    # but it is not itself validated root-cause evidence. Keep the provenance
    # explicit so consumers can distinguish the two evidence sources.
    scan_evidence_ids: list[str] = Field(default_factory=list, max_length=10)
    residual_uncertainty: str = Field(default="", max_length=1000)
    abstain_reason: Optional[Literal[
        "insufficient_evidence", "ambiguous_question", "budget_exhausted",
        "llm_unavailable_workflow_fallback",
    ]] = None
    decision_driver: Literal["llm", "workflow_fallback"] = "llm"
    agent_run: AgentRun | None = None

    @model_validator(mode="after")
    def diagnosed_requires_evidence(self) -> "ProblemDiagnosis":
        if self.status == "diagnosed" and not self.evidence_ids:
            raise ValueError("A diagnosed problem must bind at least one evidence id")
        return self


class OpenDiagnosisReport(ContractModel):
    """Ranked problem list with evidence-bound conclusions.

    ``published`` is structurally False: findings enter the pending-review
    channel only and can never be auto-released by this agent.
    """

    goal: str = Field(min_length=1, max_length=500)
    scope: dict[str, Any] = Field(default_factory=dict)
    generated_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T")
    scan_summary: dict[str, int] = Field(default_factory=dict)
    problems: list[ProblemDiagnosis] = Field(default_factory=list, max_length=8)
    decision_driver: Literal["llm", "workflow_fallback"] = "llm"
    status: Literal["completed", "refused_no_permission", "refused_scope"] = "completed"
    refusal_reason: str = Field(default="", max_length=500)
    published: Literal[False] = False
    review_status: Literal["pending_review"] = "pending_review"
    agent_run: AgentRun | None = None
