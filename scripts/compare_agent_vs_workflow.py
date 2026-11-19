"""Phase 5 A/B comparison: governed Agent arm vs frozen Workflow arm.

Two evaluation modes share the same gate:

- default (offline): the Agent arm's plan/action decisions come from the
  deterministic scripted policy in ``evaluation/agent_policy.py``.  This
  validates the typed protocol, policy gates, and scoring wiring without a
  provider request.
- ``--provider``: the same Agent arm is driven by the real LLM through
  ``llm_agent_action`` (data/RAG stay on deterministic fixtures, routing stays
  deterministic).  This is the evidence required by the promotion rule before
  a production rollout decision.

Per the plan's filter rule, only runs whose ``decision_driver`` is ``llm``
count toward the Agent bucket; runs that degraded to ``workflow_fallback``
stay in a separate ``agent_fallback`` bucket and never inflate Agent scores.
"""

from __future__ import annotations

import json
import hashlib
import shutil
import sys
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.contracts import CONTRACT_SCHEMA_VERSION
from app.core.config import get_settings
from app.graph.workflow import build_chat_graph
from app.quality import evaluations as quality_evaluations
from app.tenancy import PLATFORM_SCOPE, TenantContext, activate_tenant_context
from evaluation.agent_policy import scripted_agent_action
from evaluation.fixtures import offline_dependencies
from evaluation.scorers import collect_anomaly_metrics, score_scenario, summarize_anomaly_trajectories


def _invoke(graph: Any, case: dict[str, Any], result_prefix: str, evaluation_run_id: str) -> dict[str, Any]:
    # The offline suite exercises the same governed tools as production. Give
    # each synthetic case an explicit platform-admin identity so authorization
    # and related-metric credentials are derived through the real IAM path.
    context = TenantContext(
        tenant_id="golden-evaluation", user_id="phase5-agent-eval",
        role="admin", scope_type=PLATFORM_SCOPE,
    )
    with activate_tenant_context(context):
        return graph.invoke(
            {
                "user_message": case["query"],
                "trace_id": f"{result_prefix}-{case['id']}",
                "thread_id": f"{result_prefix}-{case['id']}",
                "audit_actor": "phase5", "evaluation_run_id": evaluation_run_id,
                "tenant_id": context.tenant_id, "user_id": context.user_id,
                "user_role": context.role,
            },
            config={"configurable": {"thread_id": f"{result_prefix}-{case['id']}"}},
        )


def _run_cases(graph: Any, cases: list[dict[str, Any]], result_prefix: str, evaluation_run_id: str) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for case in cases:
        result = _invoke(graph, case, result_prefix, evaluation_run_id)
        checks = score_scenario(result, case["expect"])
        details.append({
            "id": case["id"], "category": case["category"], "pattern": case.get("pattern", ""),
            "passed": all(checks.values()),
            "expected": case["expect"], "scenario_checks": checks,
            "scenario_metrics": collect_anomaly_metrics(result),
            "node_history": result.get("node_history", []),
            "decision_driver": result.get("decision_driver"),
            "termination_initiator": result.get("termination_initiator"),
            "llm_failures": result.get("diagnostic_llm_failures", []),
        })
    return details


def _failure_taxonomy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the LLM failure taxonomy across runs (review fix #5/#7).

    Separates output-format failures from real judgment failures:
    - attempts: every failing LLM attempt by class;
    - recovered: later attempt succeeded (schema repair/retry worked);
    - final: no attempt succeeded for that stage (these become fallbacks).
    """
    attempts: dict[str, int] = defaultdict(int)
    recovered: dict[str, int] = defaultdict(int)
    final: dict[str, int] = defaultdict(int)
    for row in rows:
        failures = row.get("llm_failures", [])
        by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for failure in failures:
            cls = str(failure.get("class") or "unknown")
            if cls == "ok":
                continue  # successful attempts are not failures
            attempts[cls] += 1
            by_stage[str(failure.get("stage"))].append(failure)
            if failure.get("recovered"):
                recovered[cls] += 1
        for stage_entries in by_stage.values():
            if stage_entries and not stage_entries[-1].get("recovered"):
                final[str(stage_entries[-1].get("class") or "unknown")] += 1
    plan_attempts = sum(1 for row in rows for f in row.get("llm_failures", []) if f.get("stage") == "plan" and f.get("class") != "ok" and not f.get("recovered"))
    plan_recovered = sum(1 for row in rows for f in row.get("llm_failures", []) if f.get("stage") == "plan" and f.get("class") != "ok" and f.get("recovered"))
    repair_rate = round(plan_recovered / (plan_recovered + plan_attempts), 4) if (plan_recovered + plan_attempts) else 0.0
    return {
        "attempts_by_class": dict(attempts),
        "recovered_by_class": dict(recovered),
        "final_failures_by_class": dict(final),
        "plan_repair_recover_rate": repair_rate,
    }


def run_agent_arm(*, provider_enabled: bool = False, case_ids: set[str] | None = None, evaluation_run_id: str | None = None, dataset_path: Path | None = None) -> dict[str, Any]:
    evaluation_run_id = evaluation_run_id or f"agent_eval_{uuid4().hex}"
    dataset_path = dataset_path or ROOT / "evaluation" / "datasets" / "golden.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = [
        case for case in dataset["cases"]
        if case.get("category") in {"root_cause_scenario", "root_cause_longtail"}
        and (case_ids is None or case.get("id") in case_ids)
    ]
    calls: dict[str, int] = {}

    # Capture the real client functions before offline_dependencies patches the
    # module attributes; provider mode re-patches them back in.  Both must be
    # restored together: llm_agent_action delegates to llm_extract, so restoring
    # only the wrapper would still hit the fixture returning None.
    from app.llm import client as llm_client
    real_extract, real_action = llm_client.llm_extract, llm_client.llm_agent_action

    with offline_dependencies():
        graph = build_chat_graph(checkpointer=InMemorySaver())
        fallback_details = _run_cases(graph, cases, "fallback", evaluation_run_id)

        if provider_enabled:
            # Surgical restore: only the governed agent loop talks to the real
            # provider.  The wrapper calls the captured extract function object
            # directly, so the module-level fixture patch on llm_extract keeps
            # every other caller (routing, planning, critic) deterministic.
            def _real_action(system_prompt: str, user_message: str, **kwargs: Any) -> Any:
                return real_extract(system_prompt, user_message, **kwargs)

            action_patches = (patch("app.llm.client.llm_agent_action", _real_action),)
            evaluation_mode = "provider_enabled"
        else:
            action_patches = (
                patch(
                    "app.llm.client.llm_agent_action",
                    lambda system, user, **kwargs: scripted_agent_action(system, user, _calls=calls),
                ),
            )
            evaluation_mode = "offline_scripted_protocol"
        with ExitStack() as stack:
            for item in action_patches:
                stack.enter_context(item)
            graph = build_chat_graph(checkpointer=InMemorySaver())
            agent_details = _run_cases(graph, cases, "agent", evaluation_run_id)

    # Promotion-rule bucketing: pure LLM-driven runs only; degraded runs are
    # reported separately and never counted as Agent results.  The frozen
    # gate set (root_cause_scenario) and the long-tail set
    # (root_cause_longtail) are aggregated separately — the gate judges the
    # former only, the long-tail report documents where the rule path breaks
    # and whether the model does better (review follow-up #7).
    llm_details = [item for item in agent_details if item.get("decision_driver") == "llm"]
    degraded_details = [item for item in agent_details if item.get("decision_driver") != "llm"]
    scenario_details = [item for item in llm_details if item.get("category") == "root_cause_scenario"]
    longtail_details = [item for item in llm_details if item.get("category") != "root_cause_scenario"]
    fallback_longtail = [item for item in fallback_details if item.get("category") != "root_cause_scenario"]
    metrics = summarize_anomaly_trajectories(scenario_details)
    longtail_metrics = summarize_anomaly_trajectories(longtail_details, category="root_cause_longtail")
    fallback_metrics = summarize_anomaly_trajectories(fallback_details)
    degraded_metrics = summarize_anomaly_trajectories(degraded_details)
    payload: dict[str, Any] = {
        "arm": "agent",
        "run_id": evaluation_run_id,
        "evaluation_mode": evaluation_mode,
        "provider_enabled": provider_enabled,
        "model": get_settings().llm_model if provider_enabled else None,
        "dataset_version": dataset["dataset_version"],
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "details": scenario_details,
        "metrics": metrics,
        "longtail": {
            "details": longtail_details,
            "metrics": longtail_metrics,
            "fallback_metrics": summarize_anomaly_trajectories(fallback_longtail, category="root_cause_longtail"),
            "note": "rule path (offline) vs model judgment (provider) on long-tail scenarios; never merged into the gate",
        },
        "agent_fallback": {"details": degraded_details, "metrics": degraded_metrics},
        "fallback": {"details": fallback_details, "metrics": fallback_metrics},
        "llm_failure_taxonomy": {
            "gate": _failure_taxonomy(scenario_details + [row for row in degraded_details if row.get("category") == "root_cause_scenario"]),
            "longtail": _failure_taxonomy(longtail_details + [row for row in degraded_details if row.get("category") != "root_cause_scenario"]),
        },
    }
    if provider_enabled and not llm_details:
        payload["provider_error"] = "no run completed with decision_driver=llm; the provider arm produced no Agent-scored evidence"
    return payload


def _gate(base: dict[str, Any], agent_metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "completion_rate_improved": agent_metrics["completion_rate"] > base.get("completion_rate", 0),
        "evidence_coverage_not_regressed": agent_metrics["evidence_coverage"] >= base.get("evidence_coverage", 0),
        # Abstention is a safety ceiling. Once the Workflow control is already
        # at 100%, strict improvement is impossible; require non-regression.
        "correct_abstain_not_regressed": agent_metrics["correct_abstain_rate"] >= base.get("correct_abstain_rate", 0),
        "invalid_and_duplicate_calls_not_increased": agent_metrics["invalid_call_count"] <= base.get("invalid_call_count", 0) and agent_metrics["duplicate_call_count"] <= base.get("duplicate_call_count", 0),
        "unauthorized_calls_zero": agent_metrics["unauthorized_call_count"] == 0,
        "required_metrics_recorded": all(
            key in agent_metrics for key in ("p95_latency_ms", "total_tokens", "total_cost", "termination_initiator_counts")
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _persist_unified_run(payload: dict[str, Any], dataset_path: Path, aggregate_path: Path, *, copy_traces: bool = True) -> Path:
    """Archive one evaluation as a filesystem bundle and DB evidence envelope.

    The scoring path remains usable when PostgreSQL is unavailable: the bundle
    is written first, while DB registration is best-effort and can be retried
    from the manifest later.
    """
    run_id = str(payload["run_id"])
    bundle = ROOT / "data" / "evaluations" / "runs" / run_id
    cases_dir, traces_dir = bundle / "cases", bundle / "traces"
    cases_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_path, bundle / "dataset.json")
    shutil.copy2(aggregate_path, bundle / "aggregate.json")

    rows: list[dict[str, Any]] = []
    trace_index: list[dict[str, Any]] = []
    groups = [
        ("agent", "agent", payload.get("details", [])),
        ("agent", "agent", payload.get("longtail", {}).get("details", [])),
        ("agent_fallback", "agent", payload.get("agent_fallback", {}).get("details", [])),
        ("fallback", "fallback", payload.get("fallback", {}).get("details", [])),
    ]
    trace_root = Path(get_settings().trace_dir)
    for arm, prefix, details in groups:
        for item in details or []:
            case_id = str(item.get("id") or "unknown")
            trace_id = f"{prefix}-{case_id}"
            case_file = cases_dir / f"{arm}-{case_id}.json"
            case_file.write_text(json.dumps(item, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            rows.append({
                "arm": arm, "case_id": case_id, "category": item.get("category"),
                "provider_called": bool(payload.get("provider_enabled") and arm == "agent" and item.get("decision_driver") == "llm"),
                "trace_id": trace_id, "result": item,
            })
            source = trace_root / f"{trace_id}.json"
            archived = None
            if copy_traces and source.exists():
                target = traces_dir / source.name
                shutil.copy2(source, target)
                archived = str(target.relative_to(bundle))
            trace_index.append({"case_id": case_id, "arm": arm, "trace_id": trace_id, "path": archived, "storage": "file" if archived else "postgres_or_missing"})

    manifest = {
        "schema_version": "agent-evaluation-bundle-v1",
        "run_id": run_id,
        "dataset": str(dataset_path),
        "dataset_hash": _sha256(dataset_path),
        "aggregate": "aggregate.json",
        "model": payload.get("model"),
        "provider_enabled": bool(payload.get("provider_enabled")),
        "evaluation_mode": payload.get("evaluation_mode"),
        "generated_at": payload.get("generated_at"),
        "total_cases": len(json.loads(dataset_path.read_text(encoding="utf-8")).get("cases", [])),
        "provider_cases": sum(1 for row in rows if row["provider_called"]),
        "passed_gate": payload.get("passed_gate"),
        "case_count": len(rows),
        "trace_count": sum(1 for item in trace_index if item["path"]),
        "trace_index": "trace_index.json",
    }
    (bundle / "trace_index.json").write_text(json.dumps(trace_index, ensure_ascii=False, indent=2), encoding="utf-8")
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Register the same run in the existing evaluation envelope.  A missing
    # database or an older schema must never invalidate a completed score.
    try:
        run = quality_evaluations.create_run(
            "agent_evaluation", run_id, "phase5",
            baseline_ref=payload.get("comparison", {}).get("workflow_baseline", {}),
            candidate_ref={"metrics": payload.get("metrics", {}), "longtail": payload.get("longtail", {}).get("metrics", {})},
            dataset_ref=str(dataset_path), dataset_hash=manifest["dataset_hash"],
            config_snapshot={"model": payload.get("model"), "provider_enabled": payload.get("provider_enabled"), "evaluation_mode": payload.get("evaluation_mode")},
            runner_version="agent-eval-v1",
            run_id=run_id,
        )
        db_run_id = run.get("run_id") if run else None
        if db_run_id:
            quality_evaluations.add_evidence(db_run_id, "golden_artifact", f"dataset:{manifest['dataset_hash']}", "gate", {"path": str(dataset_path), "total_cases": manifest["total_cases"]}, manifest["dataset_hash"])
            quality_evaluations.add_evidence(db_run_id, "benchmark_file", f"bundle:{run_id}", "candidate", {"path": str(bundle), "aggregate": str(aggregate_path), "provider_cases": manifest["provider_cases"]})
            quality_evaluations.add_case_results(db_run_id, rows)
            gate = payload.get("comparison", {}).get("gate", {})
            reasons = [name for name, passed in gate.items() if not passed]
            quality_evaluations.complete_run(db_run_id, status="passed" if payload.get("passed_gate") else "failed", gate_status="pass" if payload.get("passed_gate") else "block", reasons=reasons, metric_deltas={"agent": payload.get("metrics", {}), "longtail": payload.get("longtail", {}).get("metrics", {})}, thresholds=gate)
            manifest["db_run_id"] = db_run_id
            (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - depends on external DB
        (bundle / "persistence_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
    return bundle


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", action="store_true", help="Use the configured live LLM for Agent decisions")
    parser.add_argument("--case-ids", help="Comma-separated case IDs to run")
    parser.add_argument("--dataset", type=Path, help="Golden dataset JSON (defaults to evaluation/datasets/golden.json)")
    parser.add_argument("--output", type=Path, help="Aggregate JSON output path")
    args = parser.parse_args()
    provider_enabled = args.provider
    case_ids = {item for item in args.case_ids.split(",") if item} if args.case_ids else None
    dataset_path = (args.dataset or ROOT / "evaluation" / "datasets" / "golden.json").resolve()
    evaluation_run_id = f"agent_eval_{uuid4().hex}"
    baseline = json.loads((ROOT / "data" / "evaluations" / "workflow-baseline.json").read_text(encoding="utf-8"))
    agent = run_agent_arm(provider_enabled=provider_enabled, case_ids=case_ids, evaluation_run_id=evaluation_run_id, dataset_path=dataset_path)
    dataset_cases = json.loads(dataset_path.read_text(encoding="utf-8")).get("cases", [])
    requested_cases = [
        case for case in dataset_cases
        if case_ids is None or case.get("id") in case_ids
    ]
    core_expected = sum(1 for case in requested_cases if case.get("category") == "root_cause_scenario")
    expected_longtail = sum(1 for case in requested_cases if case.get("category") == "root_cause_longtail")
    base = baseline.get("phase0_metrics", {})
    agent_metrics = agent["metrics"]
    core_gate = _gate(base, agent_metrics) if core_expected else {"enabled": False}
    # A long-tail-only holdout has no core observations. Its applicable gate
    # is computed against the same-case Workflow arm below.
    comparison_gate = dict(core_gate) if core_expected else {}
    longtail = agent.get("longtail", {})
    expanded_longtail: dict[str, Any] = {"enabled": False}
    if expected_longtail:
        core_degraded = sum(
            1 for item in agent.get("agent_fallback", {}).get("details", [])
            if item.get("category") == "root_cause_scenario"
        )
        core_scored = len(agent.get("details", []))
        core_complete_coverage = core_scored + core_degraded == core_expected and core_scored == core_expected
        if core_expected:
            comparison_gate["core_all_cases_scored_by_agent"] = core_complete_coverage
        longtail_baseline = longtail.get("fallback_metrics", {})
        longtail_agent = longtail.get("metrics", {})
        degraded_longtail = sum(
            1 for item in agent.get("agent_fallback", {}).get("details", [])
            if item.get("category") == "root_cause_longtail"
        )
        scored_longtail = len(longtail.get("details", []))
        longtail_gate = _gate(longtail_baseline, longtail_agent)
        complete_coverage = scored_longtail + degraded_longtail == expected_longtail and scored_longtail == expected_longtail
        expanded_longtail = {
            "enabled": True,
            "core_expected_cases": core_expected,
            "core_agent_scored_cases": core_scored,
            "core_degraded_cases": core_degraded,
            "core_all_cases_scored_by_agent": core_complete_coverage,
            "expected_cases": expected_longtail,
            "agent_scored_cases": scored_longtail,
            "degraded_cases": degraded_longtail,
            "workflow_baseline": longtail_baseline,
            "agent": longtail_agent,
            "gate": {**longtail_gate, "all_cases_scored_by_agent": complete_coverage},
        }
        comparison_gate.update({f"expanded_longtail_{key}": value for key, value in longtail_gate.items()})
        comparison_gate["expanded_longtail_all_cases_scored_by_agent"] = complete_coverage
    agent["comparison"] = {
        "workflow_baseline": base,
        "agent": agent_metrics,
        "core_gate": core_gate,
        "expanded_longtail": expanded_longtail,
        "gate": comparison_gate,
    }
    agent["passed_gate"] = all(agent["comparison"]["gate"].values())
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
    elif case_ids:
        output = ROOT / "data" / "evaluations" / ("agent-failed-cases-provider.json" if provider_enabled else "agent-failed-cases-offline.json")
    else:
        output = ROOT / "data" / "evaluations" / ("agent-vs-workflow-provider.json" if provider_enabled else "agent-vs-workflow.json")
    if provider_enabled:
        # Review fix #7: never clobber the pre-normalizer evidence — snapshot
        # the previous artifact once so before/after repair rates stay auditable.
        snapshot = output.with_name("agent-vs-workflow-provider-before-normalizer.json")
        if output.exists() and not snapshot.exists():
            snapshot.write_text(output.read_text(encoding="utf-8"), encoding="utf-8")
    output.write_text(json.dumps(agent, ensure_ascii=False, indent=2), encoding="utf-8")
    bundle = _persist_unified_run(agent, dataset_path, output)
    print(json.dumps({"passed_gate": agent["passed_gate"], "mode": agent["evaluation_mode"], "run_id": evaluation_run_id, "metrics": agent_metrics, "output": str(output), "bundle": str(bundle)}, ensure_ascii=False))
    return 0 if agent["passed_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
