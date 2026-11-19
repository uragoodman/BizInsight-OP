"""Alert triage decision layer (AGENT_PLAN Phase 9 / A4).

Detection stays fully deterministic (``statistical_anomaly`` + ``AlertAgent``).
This module adds the *decision layer* on top of ``AnomalyDetected``:

  tenant switch → metric whitelist → cooldown / same-root-cause block
    → optionally start a budgeted diagnostic run (the governed loop)
    → results are emitted as ``DiagnosisProposed`` and go to pending review.

Nothing here publishes automatically, and with every switch off the alerting
path is byte-for-byte the deterministic pipeline it was before.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from app.agents.diagnostic_contracts import DiagnosticBudget
from app.agents.runtime import make_agent_run
from app.agents.open_diagnosis_contracts import DiagnosticProblem, ProblemDiagnosis
from app.analysis.contracts import AnomalyFinding
from app.core.config import get_settings


def _load_list(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw)
    except Exception:  # noqa: BLE001 - malformed config denies everything
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _diagnosis_in_cooldown(outbox: Any | None, metric_name: str, tenant_id: str) -> bool:
    """True when a DiagnosisProposed for this metric/tenant is inside the cooldown."""
    if outbox is None:
        return False
    settings = get_settings()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=settings.alert_auto_diagnosis_cooldown_hours)).isoformat()
    try:
        for event in outbox.pending(limit=200):
            occurred = event.occurred_at if isinstance(event.occurred_at, str) else event.occurred_at.isoformat()
            if event.event_type != "DiagnosisProposed" or occurred < cutoff:
                continue
            payload = event.payload or {}
            if payload.get("metric_name") == metric_name and payload.get("tenant_id", tenant_id) == tenant_id:
                return True
    except Exception:  # noqa: BLE001 - outbox unavailability never blocks alerting
        return False
    return False


def _triage_run(trace_id: str, *, allowed: bool, reason: str, diagnosis: ProblemDiagnosis | None = None) -> dict[str, Any]:
    child = diagnosis.agent_run if diagnosis else None
    return make_agent_run(
        run_id=trace_id, domain="alert_triage",
        decision_driver=(child.decision_driver if child else "workflow_fallback"),
        status="finished" if allowed else "abstained",
        termination_initiator="agent" if allowed else "policy",
        termination_reason=reason,
        rounds=1 + (child.rounds if child else 0),
        tool_calls=child.tool_calls if child else 0,
        usage={
            "input_tokens": child.input_tokens if child else 0,
            "output_tokens": child.output_tokens if child else 0,
            "estimated_cost": child.estimated_cost if child else 0,
        },
        trace=[{"node": "alert_triage", "summary": f"allowed={allowed}"}],
    ).model_dump(mode="json")


def triage_anomaly(
    finding: AnomalyFinding,
    *,
    tenant_id: str,
    outbox: Any | None = None,
    trace_id: str = "alert-triage",
) -> dict[str, Any]:
    """Decide whether one anomaly may be escalated into an auto diagnosis.

    Returns a decision dict (never raises): ``allowed`` gates are audited in
    ``reasons`` so the deterministic alert path stays explainable.
    """
    settings = get_settings()
    reasons: list[str] = []
    if not settings.alert_auto_diagnosis_enabled:
        reasons.append("deployment switch alert_auto_diagnosis_enabled is off")
    if tenant_id not in _load_list(settings.alert_auto_diagnosis_tenants_json):
        reasons.append(f"tenant {tenant_id} has no auto-diagnosis switch")
    if finding.metric_name not in _load_list(settings.alert_auto_diagnosis_metrics_json):
        reasons.append(f"metric {finding.metric_name} is not whitelisted")
    if reasons:
        return {
            "allowed": False, "diagnosis_started": False, "reasons": reasons,
            "metric_name": finding.metric_name, "metric_date": finding.metric_date.isoformat(),
            "tenant_id": tenant_id,
            "agent_run": _triage_run(trace_id, allowed=False, reason="; ".join(reasons)),
        }

    if outbox is not None:
        claim_attempt = getattr(outbox, "claim_diagnosis_attempt", None)
        if callable(claim_attempt):
            attempt_allowed = bool(claim_attempt(
                tenant_id, finding.metric_name,
                settings.alert_auto_diagnosis_cooldown_hours,
            ))
        else:
            # Compatibility with outbox adapters that have not yet implemented
            # durable attempt reservations.
            attempt_allowed = not _diagnosis_in_cooldown(outbox, finding.metric_name, tenant_id)
        if not attempt_allowed:
            reasons.append(f"metric {finding.metric_name} was auto-diagnosed within the cooldown window")
            return {
                "allowed": False, "diagnosis_started": False, "reasons": reasons,
                "metric_name": finding.metric_name, "metric_date": finding.metric_date.isoformat(),
                "tenant_id": tenant_id,
                "agent_run": _triage_run(trace_id, allowed=False, reason="; ".join(reasons)),
            }

    problem = DiagnosticProblem(
        problem_id=f"alert-{finding.metric_name}-{finding.metric_date.isoformat()}",
        metric_name=finding.metric_name,
        metric_date=finding.metric_date.isoformat(),
        direction=finding.direction,
        priority=1,
        hypothesis=(
            f"自动诊断：{finding.metric_name} 于 {finding.metric_date} "
            f"{'上升' if finding.direction == 'up' else '下降'}（score={finding.score}），需归因。"
        ),
        scan_evidence_ids=[str(eid) for eid in finding.evidence_ids],
        budget=DiagnosticBudget(
            max_rounds=4,
            max_tool_calls=settings.open_diagnosis_max_tool_calls_per_problem,
            max_tokens=12000, max_cost=1.0,
        ),
    )
    from app.agents.open_diagnosis import diagnose_problem

    # The diagnostic run is scoped to exactly the whitelisted metric that
    # triggered the alert — the triage whitelist is the diagnostic scope.
    diagnosis: ProblemDiagnosis = diagnose_problem(
        problem,
        {"tenant_id": tenant_id, "allowed_metrics": [finding.metric_name]},
        trace_id,
    )
    published_evidence = {
        **diagnosis.model_dump(mode="json"),
        "tenant_id": tenant_id,
        "source": "alert_auto_diagnosis",
        "review_status": "pending_review",
    }
    triage_run = _triage_run(trace_id, allowed=True, reason="Budgeted diagnosis completed.", diagnosis=diagnosis)
    published_evidence["agent_run"] = triage_run
    if outbox is not None and diagnosis.status == "diagnosed":
        outbox.emit(
            "DiagnosisProposed", published_evidence,
            f"alert-auto-diagnosis:{tenant_id}:{diagnosis.metric_name}:{diagnosis.metric_date}",
            "warning",
        )
    return {
        "allowed": True, "diagnosis_started": True,
        "reasons": ["all gates passed; budgeted diagnostic run completed"],
        "metric_name": finding.metric_name, "metric_date": finding.metric_date.isoformat(),
        "tenant_id": tenant_id, "diagnosis": published_evidence,
        "agent_run": triage_run,
    }


def review_diagnosis_proposal(outbox: Any, idempotency_key: str, action: str, actor: str, reason: str = "") -> bool:
    """Human gate for proposed diagnoses: approve (process) or reject.

    Returns True when the proposal was found and closed; the finding is never
    republished automatically either way.
    """
    if action not in {"approve", "reject"}:
        raise ValueError("review action must be approve or reject")
    for event in outbox.pending(limit=200):
        if event.idempotency_key == idempotency_key:
            outbox.mark_processed(event.event_id)
            return True
    return False
