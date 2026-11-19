"""Bounded investigation planning and execution contracts."""

from app.investigation.candidates import candidate_id_for, generate_candidate_actions
from app.investigation.engine import (
    prepare_candidates,
    record_action_result,
    run_investigation,
    select_candidate,
)
from app.investigation.contracts import (
    ActionResult,
    AnalysisAction,
    AnalysisTarget,
    AgentDecision,
    AttributionEvidence,
    AttributionResult,
    CandidateAction,
    CandidateSet,
    InvestigationProgress,
    InvestigationSummary,
    InvestigationState,
    StopReason,
    UnresolvedBranch,
    WorkflowSelection,
)
from app.investigation.evaluation import (
    ActionMatcher,
    BranchMatcher,
    EvidenceMatcher,
    InvestigationEvaluationCase,
    InvestigationEvaluationResult,
    evaluate_investigation_path,
    InvestigationEvaluationSummary,
    summarize_investigation_evaluations,
)

__all__ = [
    "ActionResult", "AnalysisAction", "AnalysisTarget", "AgentDecision", "AttributionEvidence", "AttributionResult",
    "CandidateAction", "CandidateSet", "InvestigationProgress", "InvestigationSummary", "InvestigationState", "StopReason",
    "UnresolvedBranch", "WorkflowSelection", "candidate_id_for", "generate_candidate_actions",
    "prepare_candidates", "record_action_result", "select_candidate", "run_investigation",
    "ActionMatcher", "EvidenceMatcher", "InvestigationEvaluationCase",
    "BranchMatcher", "InvestigationEvaluationResult", "InvestigationEvaluationSummary",
    "evaluate_investigation_path", "summarize_investigation_evaluations",
]
