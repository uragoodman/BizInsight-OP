"""Offline, deterministic quality gate using the checked-in Golden Dataset."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from subprocess import check_output
from typing import Any
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph.workflow import build_chat_graph, run_chat_workflow
from app.harness.evaluator import evaluate_trace_contract
from app.harness.storage import read_trace
from evaluation.scorers import (
    rate,
    score_trajectory,
    score_scenario,
    collect_anomaly_metrics,
    summarize_anomaly_trajectories,
    SCENARIO_SCORER_VERSION,
    DIAGNOSTIC_RUN_SCORER_VERSION,
    score_diagnostic_run,
)
from evaluation.non_root_scorer import summarize_diagnostic_runs
from app.core.config import get_settings
from app.agents.contracts import CONTRACT_SCHEMA_VERSION
from evaluation.fixtures import offline_dependencies
from evaluation.phase6 import run_phase6_acceptance
from evaluation.phase12 import run_phase12_acceptance
from evaluation.phase13 import run_phase13_acceptance
from app.tenancy import PLATFORM_SCOPE, TenantContext, activate_tenant_context


ROOT = Path(__file__).resolve().parents[1]
ROOT_CAUSE_CATEGORIES = {"root_cause", "root_cause_scenario", "root_cause_longtail"}
ROOT_CAUSE_EVAL_CONTEXT = TenantContext(
    tenant_id="golden-evaluation", user_id="golden-rootcause-eval",
    role="admin", scope_type=PLATFORM_SCOPE,
)


def _invoke_golden_case(graph: Any, case: dict[str, Any]) -> dict[str, Any]:
    """Run one Golden case, using a real test identity for root-cause work.

    Provider calls are disabled by ``offline_dependencies``.  Root-cause cases
    therefore measure the normal-authorized deterministic business fallback;
    anonymous rejection is covered by separate security tests.
    """
    thread_id = f"golden-{case['id']}"
    config = {"configurable": {"thread_id": thread_id}}
    root_cause = case.get("category") in ROOT_CAUSE_CATEGORIES
    context = ROOT_CAUSE_EVAL_CONTEXT if root_cause else None
    graph_input = {
        "user_message": case["query"],
        "trace_id": thread_id,
        "thread_id": thread_id,
        "audit_actor": context.user_id if context else "golden-reviewer",
    }
    if context is not None:
        graph_input.update(context.as_state())
    with activate_tenant_context(context) if context is not None else nullcontext():
        result = graph.invoke(graph_input, config=config)
        if result.get("critic_verdict") == "human_review" or result.get("__interrupt__"):
            result = graph.invoke(Command(resume={
                "action": "approve", "actor": "golden-reviewer", "reason": "golden verified",
            }), config=config)
    return result


def _commit_sha() -> str:
    try:
        return check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def run_golden_evaluation(dataset_path: Path | None = None, output_path: Path | None = None) -> dict[str, Any]:
    dataset_path = dataset_path or ROOT / "evaluation" / "datasets" / "golden.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    details: list[dict[str, Any]] = []
    with offline_dependencies():
        graph = build_chat_graph(checkpointer=InMemorySaver())
        for case in dataset["cases"]:
            if case["category"] == "failure":
                try:
                    run_chat_workflow(case["query"])
                    ok, actual = False, None
                except Exception as exc:
                    ok, actual = type(exc).__name__ == case["expected_error"], type(exc).__name__
                details.append({"id": case["id"], "passed": ok, "actual_error": actual})
                continue
            result = _invoke_golden_case(graph, case)
            trajectory = score_trajectory(result)
            if case["category"] == "report":
                history = [item.get("node") for item in result.get("node_history", [])]
                trajectory["structured_periodic_report"] = bool(result.get("periodic_report"))
                trajectory["human_review_resumed"] = "human_review" in history
                trajectory["report_version_published"] = bool(
                    result.get("report_id") and result.get("report_version")
                    and result.get("human_decision", {}).get("action") == "approve"
                )
            entry: dict[str, Any] = {
                "id": case["id"], "passed": False,
                "task_type": result.get("task_type"), "trajectory": trajectory,
            }
            if case["category"] in {"root_cause_scenario", "root_cause_longtail"}:
                # Phase 0 diagnostic cases: pass only when the run actually finds
                # the expected cause / abstains.  The baseline is expected to
                # FAIL these (rules give up) — that is the measured gap.
                expect = case.get("expect", {})
                scenario = score_scenario(result, expect)
                entry["category"] = case["category"]
                entry["expected"] = expect
                entry["scenario_checks"] = scenario
                entry["scenario_metrics"] = collect_anomaly_metrics(result)
                entry["node_history"] = result.get("node_history", [])
                entry["execution_arm"] = "normal_authorized_provider_unavailable_deterministic_fallback"
                expected_outcome = (
                    "safe_stop" if expect.get("abstain") is True
                    else "numeric_decomposition" if expect.get("found_factor")
                    else "cross_metric_link" if expect.get("cross_metric_link")
                    else "statistical_contribution"
                )
                entry["run_score"] = score_diagnostic_run(
                    result,
                    case_id=str(case["id"]),
                    arm=entry["execution_arm"],
                    replicate=1,
                    expected_outcome=expected_outcome,
                    scenario_expectation=expect,
                )
                entry["passed"] = bool(entry["run_score"]["task_acceptance_pass"])
            else:
                entry["passed"] = result.get("task_type") == case["expected_task_type"] and all(trajectory.values())
            details.append(entry)
    details.extend(run_phase6_acceptance())
    details.extend(run_phase12_acceptance())
    details.extend(run_phase13_acceptance())
    # Scenario cases are a Phase 0 diagnostic extension: they must not shift the
    # legacy gate (used by CI and resume evidence until the agent passes Phase 5).
    scenario_details = [item for item in details if item.get("category") == "root_cause_scenario"]
    longtail_details = [item for item in details if item.get("category") == "root_cause_longtail"]
    feature_path_details = [item for item in details if item.get("category") == "feature_path"]
    legacy_details = [item for item in details if item.get("category") not in {"root_cause_scenario", "root_cause_longtail", "feature_path"}]
    scenario_cases = len(scenario_details)
    scenario_passed = sum(item["passed"] for item in scenario_details)
    phase0_metrics = summarize_anomaly_trajectories(scenario_details)
    phase0_run_scores = summarize_diagnostic_runs(
        [item["run_score"] for item in scenario_details if item.get("run_score")]
    )
    longtail_cases = len(longtail_details)
    longtail_passed = sum(item["passed"] for item in longtail_details)
    longtail_metrics = summarize_anomaly_trajectories(longtail_details, category="root_cause_longtail")
    longtail_run_scores = summarize_diagnostic_runs(
        [item["run_score"] for item in longtail_details if item.get("run_score")]
    )
    passed = sum(item["passed"] for item in legacy_details)
    report = {
        "dataset_version": dataset["dataset_version"], "commit_sha": _commit_sha(),
        "scenario_scorer_version": SCENARIO_SCORER_VERSION,
        "diagnostic_run_scorer_version": DIAGNOSTIC_RUN_SCORER_VERSION,
        "model": get_settings().llm_model, "prompt_version": "v1",
        "rag_index_version": "golden-v1", "fixture_version": "golden-v1",
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "root_cause_execution_arm": "normal_authorized_provider_unavailable_deterministic_fallback",
        "root_cause_test_identity": {
            "tenant_id": ROOT_CAUSE_EVAL_CONTEXT.tenant_id,
            "role": ROOT_CAUSE_EVAL_CONTEXT.role,
            "scope_type": ROOT_CAUSE_EVAL_CONTEXT.scope_type,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(), "cases": len(legacy_details),
        "passed": passed, "overall_rate": rate(passed, len(legacy_details)), "details": details,
        "scenario_rate": rate(scenario_passed, scenario_cases), "scenario_cases": scenario_cases,
        "scenario_passed": scenario_passed,
        "phase0_metrics": phase0_metrics,
        "phase0_run_scores": phase0_run_scores,
        "longtail_rate": rate(longtail_passed, longtail_cases),
        "longtail_cases": longtail_cases,
        "longtail_passed": longtail_passed,
        "longtail_metrics": longtail_metrics,
        "longtail_run_scores": longtail_run_scores,
        "feature_path_cases": len(feature_path_details),
        "feature_path_passed": sum(bool(item["passed"]) for item in feature_path_details),
        "feature_path_rate": rate(sum(bool(item["passed"]) for item in feature_path_details), len(feature_path_details)),
        "feature_path_coverage": {
            feature: sorted({str(item.get("path")) for item in feature_path_details if item.get("feature") == feature})
            for feature in sorted({str(item.get("feature")) for item in feature_path_details})
        },
    }
    output_dir = ROOT / "data" / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    # latest.json is the working snapshot.  The Phase 0 control arm is frozen
    # after its first creation; refresh it only with an explicit opt-in.
    report_path = output_path or output_dir / "latest.json"
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    baseline_path = output_dir / "workflow-baseline.json"
    if not baseline_path.exists() or os.environ.get("REFRESH_WORKFLOW_BASELINE") == "1":
        baseline_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def replay_persisted_trace(trace_id: str) -> dict[str, Any]:
    """Score a stored trajectory without rerunning LLM, RAG, or MCP dependencies."""
    trace = read_trace(trace_id)
    if trace is None:
        raise FileNotFoundError(f"Trace {trace_id!r} does not exist.")
    return evaluate_trace_contract(trace)


if __name__ == "__main__":
    report = run_golden_evaluation()
    raise SystemExit(0 if report["overall_rate"] >= 0.85 else 1)
