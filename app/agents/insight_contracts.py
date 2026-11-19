"""Contracts for the insight evidence agent (AGENT_PLAN Phase 8 / A3).

The agent selects *which* evidence to fetch for a locked report's insight
generation.  It deliberately owns a different action surface from the Root
Cause diagnostic contracts: there are no anomaly hypotheses here, only
evidence gaps over the report's own metrics and dimensions.  Mathematical
attribution (contribution math, multiplicative decomposition) stays
deterministic and is never an LLM action.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import Field

from app.agents.contracts import ContractModel
from app.agents.runtime import Budget as RuntimeBudget
from app.agents.runtime import Observation as RuntimeObservation
from app.agents.runtime import PolicyDecision as RuntimePolicyDecision


class InsightBudget(RuntimeBudget):
    # Domain vocabulary for the shared action ceiling, with the insight
    # domain's original (stricter) bounds preserved.
    max_rounds: int = Field(default=3, ge=1, le=16)
    max_queries: int = Field(default=6, ge=1, le=24)
    max_tokens: int = Field(default=12000, ge=0)
    max_cost: float = Field(default=1.0, ge=0)


class InsightQuery(ContractModel):
    """One (metric, dimension) breakdown the agent wants as extra evidence."""

    candidate_id: str = Field(min_length=1, max_length=100)
    metric_name: str = Field(min_length=1, max_length=80)
    dimension: str = Field(min_length=1, max_length=80)


class InsightPlan(ContractModel):
    goal: str = Field(min_length=1, max_length=500)
    evidence_gaps: list[str] = Field(default_factory=list, max_length=10)
    candidate_ids: list[str] = Field(min_length=1, max_length=24)
    stop_conditions: list[str] = Field(min_length=1, max_length=10)
    budget: InsightBudget


class QueryEvidenceAction(ContractModel):
    action_type: Literal["QUERY_EVIDENCE"]
    candidate_set_id: str = Field(min_length=1, max_length=128)
    candidate_ids: list[str] = Field(min_length=1, max_length=8)


class ReplanInsightAction(ContractModel):
    action_type: Literal["REPLAN"]
    candidate_set_id: str = Field(min_length=1, max_length=128)
    remaining_candidate_ids: list[str] = Field(min_length=1, max_length=24)


class FinishInsightAction(ContractModel):
    action_type: Literal["FINISH"]
    candidate_set_id: str = Field(min_length=1, max_length=128)
    conclusion: str = Field(default="", max_length=1000)


class AbstainInsightAction(ContractModel):
    action_type: Literal["ABSTAIN"]
    candidate_set_id: str = Field(min_length=1, max_length=128)
    reason: Literal["insufficient_gaps", "ambiguous_report", "budget_exhausted"]


InsightAction = Annotated[
    Union[QueryEvidenceAction, ReplanInsightAction, FinishInsightAction, AbstainInsightAction],
    Field(discriminator="action_type"),
]

ObservationKind = Literal["contribution_evidence", "policy_deny", "empty"]


class InsightObservation(RuntimeObservation):
    action_type: Literal["QUERY_EVIDENCE", "REPLAN", "FINISH", "ABSTAIN"]


class PolicyVerdict(RuntimePolicyDecision):
    action_type: Literal["QUERY_EVIDENCE", "REPLAN", "FINISH", "ABSTAIN"]
