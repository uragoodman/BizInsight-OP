"""Open business-diagnosis agent (AGENT_PLAN Phase 6 / A1).

Pipeline (governed, deterministic first):

1. Permission gate — auto-run must be enabled and every requested metric must
   sit inside the scan whitelist.  Without permission the agent refuses before
   touching the scanner or the LLM.
2. Deterministic scan — ``statistical_anomaly`` findings over the whitelisted
   metric series (explainable, cheap, no LLM).
3. LLM triage — which scanned anomalies are worth diagnostic budget, ranked
   and merged (one problem per metric).  LLM unavailable → deterministic
   top-N fallback, bucketed as ``workflow_fallback``.
4. Drill-down — each selected problem reuses the governed diagnostic loop
   (``diagnostic_plan → diagnostic_decide → policy_execute → observe →
   diagnostic_finalize``) with its own budget.
5. Report — ranked problem list with evidence-bound conclusions, structurally
   ``pending_review``; findings are emitted as ``DiagnosisProposed`` events
   and never auto-published.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from pydantic import TypeAdapter, ValidationError

from app.agents.diagnostic_contracts import AbstainAction, DiagnosticBudget, DiagnosticPlan
from app.agents.runtime import make_agent_run
from app.agents.open_diagnosis_contracts import (
    DiagnosticProblem,
    OpenDiagnosisPlan,
    OpenDiagnosisReport,
    ProblemDiagnosis,
    ScanCandidate,
)
from app.analysis.contracts import PeriodSpec
from app.analysis.statistical_anomaly import detect_series_anomalies
from app.core.config import get_settings

_PLAN_ADAPTER = TypeAdapter(OpenDiagnosisPlan)


def _scan_period(scope: dict[str, Any]) -> PeriodSpec:
    end = str(scope.get("default_end_date") or date.today().isoformat())
    start = str(scope.get("default_start_date") or (date.fromisoformat(end) - timedelta(days=13)).isoformat())
    return PeriodSpec.model_validate({"start_date": start, "end_date": end, "granularity": "day"})


def _scan_candidates(scope: dict[str, Any], metrics: list[str]) -> list[ScanCandidate]:
    from app.analysis.periods import query_metric_series

    period = _scan_period(scope)
    candidates: list[ScanCandidate] = []
    for metric_name in metrics:
        try:
            series = query_metric_series(metric_name, period)
            findings = detect_series_anomalies(series)
        except Exception:  # noqa: BLE001 - a broken metric never blocks the scan
            continue
        # Merge: one problem per metric, keeping the strongest finding.
        strongest = max(findings, key=lambda item: abs(item.score), default=None)
        if strongest is None:
            continue
        candidates.append(ScanCandidate(
            metric_name=strongest.metric_name,
            metric_date=strongest.metric_date.isoformat(),
            direction=strongest.direction,
            score=float(strongest.score),
            threshold=float(strongest.threshold),
            value=float(strongest.value),
            baseline=float(strongest.baseline),
            evidence_ids=[str(eid) for eid in strongest.evidence_ids],
        ))
    return candidates


def _clamp_budget(raw: Any) -> DiagnosticBudget:
    settings = get_settings()
    budget = raw if isinstance(raw, DiagnosticBudget) else DiagnosticBudget.model_validate(raw or {})
    return budget.model_copy(update={
        "max_rounds": min(budget.max_rounds, 6),
        "max_tool_calls": min(budget.max_tool_calls, settings.open_diagnosis_max_tool_calls_per_problem),
        "max_tokens": min(budget.max_tokens, settings.max_tokens),
        "max_cost": min(budget.max_cost, settings.max_cost),
    })


def _fallback_plan(candidates: list[ScanCandidate], max_problems: int) -> OpenDiagnosisPlan:
    ranked = sorted(candidates, key=lambda item: abs(item.score), reverse=True)[:max_problems]
    return OpenDiagnosisPlan.model_validate({
        "goal": "按扫描强度排序的可解释问题清单（LLM 不可用时的确定性回退）。",
        "problems": [
            {
                "problem_id": f"problem-{index + 1}-{item.metric_name}",
                "metric_name": item.metric_name, "metric_date": item.metric_date,
                "direction": item.direction, "priority": index + 1,
                "hypothesis": f"{item.metric_name} 在 {item.metric_date} 出现{'上升' if item.direction == 'up' else '下降'}异常，扫描 score={item.score}。",
                "scan_evidence_ids": item.evidence_ids,
                "budget": {"max_rounds": 4, "max_tool_calls": 4, "max_tokens": 12000, "max_cost": 1.0},
            }
            for index, item in enumerate(ranked)
        ],
    })


def _llm_triage(user_message: str, candidates: list[ScanCandidate], max_problems: int) -> OpenDiagnosisPlan | None:
    from app.llm.client import llm_agent_action

    prompt = (
        "你是开放经营诊断的分诊器。以下是确定性异常扫描候选（JSON）。"
        f"最多选择 {max_problems} 个值得花诊断预算的问题，按优先级排序（priority=1 最优先），"
        "每个问题给出 hypothesis 与 budget（max_rounds/max_tool_calls/max_tokens/max_cost）。"
        "只能从候选 metric 中选择，不得新增业务事实。只输出 JSON 对象 plan，"
        "字段为 goal、problems；每项 problem 含 problem_id、metric_name、metric_date、direction、"
        "priority、hypothesis、scan_evidence_ids、budget。候选：\n"
        + json.dumps([item.model_dump(mode="json") for item in candidates], ensure_ascii=False)
    )
    try:
        raw = llm_agent_action(prompt, user_message)
        if isinstance(raw, dict) and isinstance(raw.get("plan"), dict):
            raw = raw["plan"]
        if not raw:
            return None
        return _PLAN_ADAPTER.validate_python(raw)
    except Exception:  # noqa: BLE001 - any LLM/validation failure degrades deterministically
        return None


def _problem_state(problem: DiagnosticProblem, scope: dict[str, Any], trace_id: str) -> dict[str, Any]:
    direction_text = "上升" if problem.direction == "up" else "下降"
    return {
        "user_message": f"为什么 {problem.metric_date} 的 {problem.metric_name} 出现{direction_text}异常？{problem.hypothesis}",
        "trace_id": f"{trace_id}:{problem.problem_id}", "thread_id": trace_id,
        "diagnostic_plan": {}, "diagnostic_step_state": {}, "diagnostic_action": {},
        "diagnostic_tool_results": [], "diagnostic_round": 0, "policy_decision": {},
        "hypotheses": [], "observations": [], "finish_decision": {},
        "diagnostic_fallback": False, "decision_driver": "workflow_fallback",
        "termination_initiator": "policy",
        "node_history": [], "errors": [], "tool_calls": [], "evidence_bundles": [],
        "root_cause_evidence": [], "claims": [],
        "controlled_analysis_scope": dict(scope),
        "runtime_budget": {
            "max_graph_steps": 32, "max_tool_calls": problem.budget.max_tool_calls,
            "max_tokens": problem.budget.max_tokens, "max_cost": problem.budget.max_cost,
            "max_latency_ms": get_settings().max_latency_ms,
        },
    }


def _drill_down(problem: DiagnosticProblem, scope: dict[str, Any], trace_id: str) -> tuple[ProblemDiagnosis, dict[str, Any]]:
    """Run one problem through the governed diagnostic graph nodes."""
    from app.graph.nodes.diagnostic import diagnostic_decide, diagnostic_finalize, diagnostic_plan, observe, policy_execute

    state = _problem_state(problem, scope, trace_id)
    diagnostic_plan(state)
    if state.get("diagnostic_fallback"):
        state["agent_run"] = make_agent_run(
            run_id=state["trace_id"], domain="diagnostic", decision_driver="workflow_fallback",
            status="abstained", termination_initiator="policy",
            termination_reason="LLM unavailable; deterministic workflow fallback.",
            plan=state.get("diagnostic_plan"), trace=state.get("node_history", []),
        ).model_dump(mode="json")
        return ProblemDiagnosis(
            problem_id=problem.problem_id, metric_name=problem.metric_name,
            metric_date=problem.metric_date, priority=problem.priority,
            status="abstained", abstain_reason="llm_unavailable_workflow_fallback",
            conclusion="", evidence_ids=list(problem.scan_evidence_ids),
            scan_evidence_ids=list(problem.scan_evidence_ids),
            decision_driver="workflow_fallback",
            agent_run=state["agent_run"],
        ), state

    plan = DiagnosticPlan.model_validate(state["diagnostic_plan"])
    finalized = False
    for _ in range(plan.budget.max_rounds + 1):
        diagnostic_decide(state)
        if state.get("errors"):
            diagnostic_finalize(state)
            finalized = True
            break
        policy_execute(state)
        if state.get("errors"):
            diagnostic_finalize(state)
            finalized = True
            break
        decision = state.get("policy_decision", {})
        action_type = state.get("diagnostic_action", {}).get("action_type")
        if decision.get("decision") == "deny":
            continue
        if action_type in {"FINISH", "ABSTAIN"}:
            diagnostic_finalize(state)
            finalized = True
            break
        if action_type == "REPLAN":
            plan = DiagnosticPlan.model_validate(state["diagnostic_plan"])
            continue
        observe(state)
    if not finalized:
        state["diagnostic_action"] = AbstainAction(
            action_type="ABSTAIN", reason="budget_exhausted",
        ).model_dump(mode="json")
        state["termination_initiator"] = "budget"
        diagnostic_finalize(state)

    analysis = state.get("analysis_result", {}) or {}
    abstain = bool(analysis.get("abstain"))
    status = "abstained" if abstain else "diagnosed"
    finish = state.get("finish_decision", {}) or {}
    abstain_reason = None
    if abstain:
        reason = str(finish.get("reason") or analysis.get("stop_reason") or "")
        abstain_reason = reason if reason in {
            "insufficient_evidence", "ambiguous_question", "budget_exhausted",
        } else "insufficient_evidence"
    return ProblemDiagnosis(
        problem_id=problem.problem_id, metric_name=problem.metric_name,
        metric_date=problem.metric_date, priority=problem.priority,
        status=status, conclusion=str(analysis.get("summary") or ""),
        evidence_ids=list(dict.fromkeys([
            *[str(eid) for eid in problem.scan_evidence_ids],
            *[str(eid) for eid in analysis.get("evidence_ids", [])],
        ])),
        scan_evidence_ids=[str(eid) for eid in problem.scan_evidence_ids],
        residual_uncertainty=str((finish.get("residual_uncertainty") or "")),
        abstain_reason=abstain_reason,
        decision_driver=str(state.get("decision_driver") or "llm") if state.get("decision_driver") in {"llm", "workflow_fallback"} else "llm",
        agent_run=state.get("agent_run"),
    ), state


def _entry_run(
    trace_id: str,
    *,
    driver: str,
    status: str,
    problems: list[ProblemDiagnosis],
    reason: str = "",
) -> dict[str, Any]:
    usage = {
        "input_tokens": sum((item.agent_run.input_tokens if item.agent_run else 0) for item in problems),
        "output_tokens": sum((item.agent_run.output_tokens if item.agent_run else 0) for item in problems),
        "estimated_cost": sum((item.agent_run.estimated_cost if item.agent_run else 0.0) for item in problems),
    }
    return make_agent_run(
        run_id=trace_id, domain="open_diagnosis",
        decision_driver=driver if driver in {"llm", "workflow_fallback"} else "workflow_fallback",
        status=status if status in {"finished", "abstained", "failed"} else "failed",
        termination_initiator="agent" if status == "finished" else "policy",
        termination_reason=reason or "Open diagnosis entry completed.",
        rounds=len(problems), tool_calls=sum((item.agent_run.tool_calls if item.agent_run else 0) for item in problems),
        usage=usage, trace=[{"node": "open_diagnosis", "summary": f"problems={len(problems)}"}],
    ).model_dump(mode="json")


def _permission_refusal(scope: dict[str, Any], metrics: list[str]) -> str | None:
    settings = get_settings()
    if not settings.open_diagnosis_enabled:
        return "open_diagnosis auto-run is disabled for this deployment"
    try:
        whitelist = set(json.loads(settings.open_diagnosis_metrics_whitelist_json))
    except Exception:  # noqa: BLE001 - malformed config denies everything
        return "open_diagnosis metrics whitelist is not valid JSON"
    if not whitelist:
        return "open_diagnosis metrics whitelist is empty"
    outside = sorted(set(metrics) - whitelist)
    if outside:
        return f"metrics outside the scan whitelist: {', '.join(outside)}"
    return None


def diagnose_problem(problem: DiagnosticProblem, scope: dict[str, Any], trace_id: str) -> ProblemDiagnosis:
    """Public single-problem entry so other agents (alert triage) reuse the
    governed drill-down without re-running the scan/triage pipeline."""
    diagnosis, _state = _drill_down(problem, scope, trace_id)
    return diagnosis


def run_open_diagnosis(
    scope: dict[str, Any],
    *,
    trace_id: str = "open-diagnosis",
    outbox: Any | None = None,
    cooldown_index: dict[str, str] | None = None,
) -> OpenDiagnosisReport:
    """Scan → triage → drill-down → pending-review report.

    ``cooldown_index`` maps metric_name → ISO timestamp of its last diagnosis;
    metrics inside the cooldown window are reported as ``skipped_cooldown``
    instead of consuming budget again.
    """
    settings = get_settings()
    cooldown_index = cooldown_index or {}
    requested = [str(item) for item in (scope.get("allowed_metrics") or [])]
    refusal = _permission_refusal(scope, requested)
    if refusal:
        return OpenDiagnosisReport(
            goal="开放经营诊断", scope=dict(scope),
            generated_at=datetime.now(timezone.utc).isoformat(),
            status="refused_no_permission" if "disabled" in refusal else "refused_scope",
            refusal_reason=refusal,
            agent_run=_entry_run(trace_id, driver="workflow_fallback", status="abstained", problems=[], reason=refusal),
        )

    candidates = _scan_candidates(scope, requested)
    if not candidates:
        # Nothing worth diagnosing: an honest empty report, not an error.
        return OpenDiagnosisReport(
            goal="开放经营诊断：扫描未发现值得诊断的异常。", scope=dict(scope),
            generated_at=datetime.now(timezone.utc).isoformat(),
            scan_summary={"metrics_scanned": len(requested), "anomalies_found": 0, "problems_selected": 0},
            problems=[], decision_driver="workflow_fallback",
            agent_run=_entry_run(trace_id, driver="workflow_fallback", status="finished", problems=[]),
        )
    plan: OpenDiagnosisPlan | None = _llm_triage("对当前扫描候选进行分诊与排序。", candidates, settings.open_diagnosis_max_problems)
    decision_driver = "llm"
    if plan is None:
        plan = _fallback_plan(candidates, settings.open_diagnosis_max_problems)
        decision_driver = "workflow_fallback"

    by_metric = {item.metric_name: item for item in candidates}
    problems: list[ProblemDiagnosis] = []
    accepted: list[DiagnosticProblem] = []
    for raw in sorted(plan.problems, key=lambda item: item.priority)[:settings.open_diagnosis_max_problems]:
        candidate = by_metric.get(raw.metric_name)
        if candidate is None:
            continue  # never diagnose a metric the scanner did not flag
        last = cooldown_index.get(raw.metric_name)
        if last:
            try:
                elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(last)
                if elapsed < timedelta(hours=settings.open_diagnosis_cooldown_hours):
                    problems.append(ProblemDiagnosis(
                        problem_id=raw.problem_id, metric_name=raw.metric_name,
                        metric_date=candidate.metric_date, priority=raw.priority,
                        status="skipped_cooldown",
                        evidence_ids=list(candidate.evidence_ids),
                        scan_evidence_ids=list(candidate.evidence_ids),
                    ))
                    continue
            except ValueError:
                pass
        accepted.append(raw.model_copy(update={
            "scan_evidence_ids": candidate.evidence_ids,
            "budget": _clamp_budget(raw.budget),
            "metric_date": candidate.metric_date, "direction": candidate.direction,
        }))

    for problem in accepted:
        diagnosis, _state = _drill_down(problem, scope, trace_id)
        problems.append(diagnosis)
        if outbox is not None and diagnosis.status == "diagnosed":
            outbox.emit(
                "DiagnosisProposed",
                diagnosis.model_dump(mode="json"),
                f"open-diagnosis:{diagnosis.metric_name}:{diagnosis.metric_date}",
                "warning",
            )

    problems.sort(key=lambda item: item.priority)
    return OpenDiagnosisReport(
        goal=plan.goal, scope=dict(scope),
        generated_at=datetime.now(timezone.utc).isoformat(),
        scan_summary={
            "metrics_scanned": len(requested),
            "anomalies_found": len(candidates),
            "problems_selected": len(accepted),
        },
        problems=problems, decision_driver=decision_driver,
        agent_run=_entry_run(trace_id, driver=decision_driver, status="finished", problems=problems),
    )
