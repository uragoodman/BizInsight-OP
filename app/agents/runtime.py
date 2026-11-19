"""Shared agent runtime protocol (AGENT_PLAN Phase 10).

Extracted only after the Diagnostic (Phase 1-5) and Insight (Phase 8) domains
were both validated.  The protocol carries what every governed loop shares —
budget, observation, policy verdict, trace event, termination, and the run
record — while domain contracts (DiagnosticPlan / InsightPlan /
CodeAnalysisPlan) keep their business fields separate on purpose.

Domain contracts subclass these bases and narrow the generic ``action_type``
string to their own Literal, so cross-domain payload confusion stays a type
error instead of a runtime surprise.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from pydantic import Field

from app.agents.contracts import ContractModel

TerminationInitiator = Literal["agent", "budget", "policy", "critic", "unknown"]


class Budget(ContractModel):
    """Upper bounds a governed loop must respect.  A budget is a ceiling, never
    a target: unspent budget is the normal good case.

    The action-count field deliberately stays domain vocabulary
    (``max_tool_calls`` / ``max_queries``); only rounds, tokens and cost are
    shared, so cross-domain payload confusion stays a type error.
    """

    max_rounds: int = Field(default=4, ge=1, le=64)
    max_tokens: int = Field(default=12000, ge=0)
    max_cost: float = Field(default=1.0, ge=0)


class Observation(ContractModel):
    """Result of executing one action; only observation-producing actions
    create evidence (the domain contract narrows which actions those are)."""

    observation_id: str = Field(min_length=1)
    summary: str = Field(default="", max_length=2000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    payload: dict[str, Any] = Field(default_factory=dict)


class PolicyDecision(ContractModel):
    """Deterministic approval verdict produced by the policy layer."""

    decision: Literal["allow", "deny"]
    reason: str = Field(min_length=1, max_length=500)
    action_type: str = Field(min_length=1)


class TraceEvent(ContractModel):
    node: str = Field(min_length=1, max_length=80)
    summary: str = Field(default="", max_length=500)


class Termination(ContractModel):
    initiator: TerminationInitiator
    reason: str = Field(default="", max_length=500)


class AgentRun(ContractModel):
    """One governed agent execution, recorded for audit and A/B bucketing."""

    run_id: str = Field(min_length=1)
    domain: Literal["diagnostic", "insight", "relation", "open_diagnosis", "alert_triage", "code"]
    decision_driver: Literal["llm", "workflow_fallback"]
    status: Literal["finished", "abstained", "failed"]
    rounds: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost: float = Field(default=0.0, ge=0)
    termination: Termination
    plan: dict[str, Any] = Field(default_factory=dict)
    trace: list[TraceEvent] = Field(default_factory=list, max_length=200)


def make_agent_run(
    *,
    run_id: str,
    domain: Literal["diagnostic", "insight", "relation", "open_diagnosis", "alert_triage", "code"],
    decision_driver: Literal["llm", "workflow_fallback"],
    status: Literal["finished", "abstained", "failed"],
    termination_initiator: TerminationInitiator = "unknown",
    termination_reason: str = "",
    rounds: int = 0,
    tool_calls: int = 0,
    usage: Mapping[str, Any] | None = None,
    plan: Mapping[str, Any] | None = None,
    trace: Sequence[Mapping[str, Any]] = (),
) -> AgentRun:
    """Build the common audit record for every governed Agent entrypoint.

    Keeping construction here prevents individual domains from silently
    dropping usage or termination fields while preserving domain-specific
    plans and action protocols.
    """
    usage = usage or {}
    events = [
        TraceEvent(
            node=str(item.get("node") or item.get("stage") or "agent"),
            summary=str(item.get("summary") or item.get("message") or "")[:500],
        )
        for item in trace
    ]
    return AgentRun(
        run_id=run_id,
        domain=domain,
        decision_driver=decision_driver,
        status=status,
        rounds=max(0, int(rounds)),
        tool_calls=max(0, int(tool_calls)),
        input_tokens=max(0, int(usage.get("input_tokens", 0) or 0)),
        output_tokens=max(0, int(usage.get("output_tokens", 0) or 0)),
        estimated_cost=max(0.0, float(usage.get("estimated_cost", 0) or 0)),
        termination=Termination(initiator=termination_initiator, reason=termination_reason[:500]),
        plan=dict(plan or {}),
        trace=events,
    )
