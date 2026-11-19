"""Parameterized executor for the six-case B/G development schedule.

This is separate from the Critic full/off candidate runner. It executes only
``basic_tool_loop`` and ``governed_workflow`` entries from the B/G manifest,
using the B/G scorer and append-only ledgers from provider_candidate_batch.
Importing this module never starts a workflow or contacts a Provider.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from evaluation.provider_candidate_batch import (
    CumulativeCostLimitExceeded,
    CostPolicy,
    EstimatedCostLedger,
    ExperimentConfig,
    RunLedger,
    _reconcile_cost_accounting,
    bind_isolated_experiment_runtime,
    load_isolated_candidate_database,
    drop_isolated_candidate_database,
    token_usage_snapshot,
)
from evaluation.resume_bg_protocol import (
    ARMS,
    PROTOCOL_VERSION,
    SCORER_VERSION,
    TASK_COUNTS,
    build_manifest,
    score_bg_run,
    score_bg_run_v5,
    validate_cases,
)
from evaluation.resume_bg_conditions import validate_case_pre_run_condition

MAX_PLANNED_UNITS = 12
MAX_ESTIMATE_LIMIT_CNY = 30.0
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "kagglehub/datasets/olistbr/brazilian-ecommerce/versions/2"


def resolve_batch_tenant_context(tenant_context: Any = None) -> Any:
    """Resolve one trusted identity for a batch and pass it to both arms."""
    from app.tenancy import TenantContext
    if tenant_context is None:
        return TenantContext(
            tenant_id="platform", user_id="resume-bg-experiment", role="admin", scope_type="platform",
        )
    if not isinstance(tenant_context, TenantContext):
        raise TypeError("tenant_context must be the same trusted TenantContext type for both arms")
    return tenant_context


def validate_bg_schedule(
    cases: list[Mapping[str, Any]], manifest: Mapping[str, Any], *,
    execution_plan: Mapping[str, Any], cost_limit_cny: float = MAX_ESTIMATE_LIMIT_CNY,
) -> dict[str, Any]:
    """Fail closed unless the fixed six-case, 12-unit B/G schedule is intact."""
    validate_cases(cases)
    entries = list(manifest.get("entries") or [])
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("B/G manifest scorer protocol version mismatch.")
    if manifest.get("planned_units") != MAX_PLANNED_UNITS or len(entries) != MAX_PLANNED_UNITS:
        raise ValueError("B/G schedule must contain exactly twelve units.")
    expected = build_manifest(cases, seed=int(manifest.get("seed")))
    if dict(manifest) != expected:
        raise ValueError("Saved B/G schedule differs from its deterministic seed, case order, arms, or run IDs.")
    run_ids = [str(row.get("run_id") or "") for row in entries]
    if any(not value for value in run_ids) or len(set(run_ids)) != MAX_PLANNED_UNITS:
        raise ValueError("B/G schedule requires twelve unique nonempty run IDs.")
    case_ids = {str(row["case_id"]) for row in cases}
    pairs: dict[str, list[Mapping[str, Any]]] = {}
    for entry in entries:
        if entry.get("arm") not in ARMS or entry.get("case_id") not in case_ids:
            raise ValueError("B/G schedule contains an unknown arm or case.")
        pairs.setdefault(str(entry.get("pair_id") or ""), []).append(entry)
        if entry.get("status") != "not_started":
            raise ValueError("Frozen B/G schedule must remain unchanged and unstarted.")
    if len(pairs) != 6 or any(
        len(rows) != 2 or {row.get("arm") for row in rows} != set(ARMS)
        or len({row.get("case_id") for row in rows}) != 1
        for rows in pairs.values()
    ):
        raise ValueError("Each B/G case must have exactly one run in each arm.")
    estimate_policy = CostPolicy(cumulative_estimate_limit_cny=float(cost_limit_cny))
    estimate = estimate_policy.estimate(MAX_PLANNED_UNITS)
    planned_estimate = execution_plan.get("cost_estimate") or {}
    if round(float(planned_estimate.get("estimated_cny") or -1), 8) != estimate["estimated_cny"]:
        raise ValueError("Execution plan cost estimate differs from the shared CostPolicy.")
    if not estimate["within_estimate_limit"] or estimate["estimated_cny"] > MAX_ESTIMATE_LIMIT_CNY:
        raise ValueError("B/G schedule estimate exceeds the authorized CNY 30 estimate limit.")
    return {
        "experiment_id": manifest.get("experiment_id"),
        "scorer_version": SCORER_VERSION,
        "schedule_protocol_version": PROTOCOL_VERSION,
        "planned_units": MAX_PLANNED_UNITS,
        "case_count_by_protocol_type": dict(TASK_COUNTS),
        "arm_counts": {arm: sum(row.get("arm") == arm for row in entries) for arm in ARMS},
        "unique_run_ids": len(set(run_ids)),
        "estimated_cny": estimate["estimated_cny"],
        "cumulative_estimate_limit_cny": estimate["cumulative_estimate_limit_cny"],
        "estimate_is_account_hard_cap": False,
        "execution_blocked": bool(execution_plan.get("execution_blocked")),
        "blocked_case_ids": list(execution_plan.get("blocked_case_ids") or []),
    }


def select_never_started(
    entries: list[Mapping[str, Any]], ledger: RunLedger, *, max_units: int,
) -> list[Mapping[str, Any]]:
    if isinstance(max_units, bool) or not 1 <= int(max_units) <= MAX_PLANNED_UNITS:
        raise ValueError("max_units must be explicitly set from 1 through 12.")
    pending = ledger.resume_candidates(entries)
    return list(pending[:int(max_units)])


def validate_execute_budget(
    *, selected_count: int, policy: CostPolicy, cost_ledger: EstimatedCostLedger,
    estimate_limit_cny: float, provider_authorization_cny: float,
) -> dict[str, Any]:
    if isinstance(estimate_limit_cny, bool) or isinstance(provider_authorization_cny, bool):
        raise ValueError("Estimate limit and Provider authorization must be numeric values.")
    if round(float(estimate_limit_cny), 8) != round(float(provider_authorization_cny), 8):
        raise ValueError("Provider authorization and cumulative estimate limit must match.")
    if not 0 < float(estimate_limit_cny) <= MAX_ESTIMATE_LIMIT_CNY:
        raise ValueError("This B/G batch is limited to its pre-registered CNY 30 estimate cap.")
    required = selected_count * policy.max_per_run_cny()
    remaining = float(estimate_limit_cny) - cost_ledger.reserved_cny
    if required > remaining + 1e-12:
        raise CumulativeCostLimitExceeded("Cumulative estimate reservation is insufficient; no unit was started.")
    return {
        "selected_units": int(selected_count),
        "new_estimate_reservation_cny": round(required, 8),
        "existing_reservations_cny": round(cost_ledger.reserved_cny, 8),
        "cumulative_estimate_limit_cny": float(estimate_limit_cny),
        "remaining_estimate_capacity_cny": round(remaining, 8),
        "estimate_is_account_hard_cap": False,
    }


def run_bg_unit(
    entry: Mapping[str, Any], case: Mapping[str, Any], *, tenant_context: Any,
    trace_dir: Path, ledger: RunLedger, cost_ledger: EstimatedCostLedger,
    experiment_config: ExperimentConfig,
    workflow_fn: Any = None,
    scorer_fn: Any = score_bg_run,
) -> dict[str, Any]:
    """Execute one B/G run; case truth is passed only to the post-run scorer."""
    if scorer_fn not in {score_bg_run, score_bg_run_v5}:
        raise RuntimeError("Only the preserved v3 scorer or explicit opt-in v5 scorer is supported.")
    arm = str(entry.get("arm") or "")
    run_id = str(entry.get("run_id") or "")
    if arm not in ARMS or not run_id or case.get("case_id") != entry.get("case_id"):
        raise ValueError("B/G run entry and candidate case do not match.")
    trace_dir.mkdir(parents=True, exist_ok=True)
    current = ledger.latest().get(run_id)
    if current and current.get("event") not in {"cost_reserved"}:
        raise RuntimeError("Started or terminal B/G run IDs are never retried.")
    reserved = cost_ledger.reserve(run_id)
    ledger.append(run_id, "cost_reserved", reserved_cny=reserved, usage_status="unknown_until_completion")
    ledger.append(run_id, "started", case_id=entry["case_id"], arm=arm, replicate=entry.get("replicate"))
    started = time.perf_counter()
    trace_path = trace_dir / f"{run_id}.json"
    try:
        from app.core.config import get_settings
        from app.graph import workflow as workflow_module
        from app.graph.critic_review_mode import CriticLLMReviewMode
        from app.graph.critic_llm_thinking_mode import CriticLLMThinkingMode
        from app.llm.call_config import LLMCallConfig
        from app.tenancy import TenantContext

        call_config = LLMCallConfig.provider_ab_experiment(
            workflow_module.get_settings(), provider_id=experiment_config.provider_id,
            max_completion_tokens=experiment_config.max_completion_tokens,
        )
        question = str(case["user_question"])
        if arm == "basic_tool_loop":
            from evaluation.resume_bare_tool_loop import run_bare_tool_loop
            raw_state = run_bare_tool_loop(
                question, tenant_context=tenant_context, run_id=run_id,
                llm_call_config=call_config,
                max_rounds=experiment_config.max_graph_steps,
                max_tool_calls=experiment_config.max_tool_calls,
                max_latency_ms=experiment_config.max_latency_ms,
                max_task_tokens=experiment_config.max_task_tokens,
                max_cost_cny=experiment_config.max_cost_cny_per_workflow,
            )
            final_action = raw_state.get("final") or {}
            model_authored_final = raw_state.get("model_authored_final")
            final_adaptation = raw_state.get("final_output_adaptation") or {}
            final_presentation = raw_state.get("presentation") or final_action.get("presentation")
            trace = {
                "trace_id": run_id,
                "state_snapshot": raw_state,
                "workflow_result": {
                    "answer": final_action.get("answer") or "",
                    "presentation": final_presentation,
                    "model_authored_output": model_authored_final,
                    "final_output_adaptation": final_adaptation,
                },
            }
            response = {
                "answer": final_action.get("answer", ""),
                "presentation": final_presentation,
                "model_authored_output": model_authored_final,
                "final_output_adaptation": final_adaptation,
            }
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        else:
            if workflow_fn is None:
                workflow_fn = workflow_module.run_chat_workflow
            response = workflow_fn(
                question, tenant_context=tenant_context, memory_enabled=False,
                trace_id=run_id,
                _critic_llm_review_mode=CriticLLMReviewMode.FULL,
                _critic_llm_thinking_mode=CriticLLMThinkingMode.NON_THINKING_EXPERIMENT,
                _llm_call_config=call_config,
            )
            if not trace_path.is_file():
                raise RuntimeError("Governed workflow returned without saving its trace.")
            trace = json.loads(trace_path.read_text(encoding="utf-8"))

        state = trace.get("state_snapshot") or {}
        if scorer_fn is score_bg_run_v5:
            score = score_bg_run_v5(case, arm=arm, trace=trace, lifecycle_status="completed")
        else:
            score = score_bg_run(case, arm=arm, trace=trace, lifecycle_status="completed")
        model_authored_score = None
        if arm == "basic_tool_loop" and isinstance(state.get("model_authored_final"), Mapping):
            # Apply the same scorer to the unadapted model response as a separate
            # diagnostic result; the official arm score remains for the displayed output.
            raw_trace = json.loads(json.dumps(trace, ensure_ascii=False, default=str))
            raw_state = raw_trace.get("state_snapshot") or {}
            raw_final = json.loads(json.dumps(state["model_authored_final"], ensure_ascii=False, default=str))
            raw_state["final"] = raw_final
            raw_state["presentation"] = raw_final.get("presentation") if isinstance(raw_final, Mapping) else None
            raw_trace["workflow_result"] = {
                "answer": raw_final.get("answer", ""),
                "presentation": raw_final.get("presentation"),
            }
            model_authored_score = scorer_fn(
                case, arm=arm, trace=raw_trace, lifecycle_status="completed",
            )
        try:
            from evaluation.resume_bg_observation import extract_formal_observation
            from evaluation.resume_bg_companion_audit import build_companion_quality_audit
            companion_observation = extract_formal_observation(case, arm, trace)
            companion_quality_audit = build_companion_quality_audit(
                case, arm, trace, companion_observation, expected_run_id=run_id,
            )
            companion_quality_audit_error = None
        except Exception as audit_error:
            companion_quality_audit = None
            companion_quality_audit_error = {
                "error_type": type(audit_error).__name__, "error": str(audit_error),
            }
        events = [row for row in ledger.events if row.get("run_id") == run_id]
        usage = token_usage_snapshot(trace, events, run_id=run_id)
        accounting = cost_ledger.record_usage(run_id, usage)
        reservation_row = next(
            (row for row in cost_ledger.reservations
             if row.get("run_id") == run_id and row.get("reserved_cny") is not None), None,
        )
        _reconcile_cost_accounting(
            usage, accounting, cost_ledger.policy,
            expected_reservation_cny=float(reservation_row["reserved_cny"]) if reservation_row else None,
        )
        events = [row for row in ledger.events if row.get("run_id") == run_id]
        measurements = {
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "logical_model_calls": len(state.get("llm_calls") or []),
            "sdk_sends": sum(row.get("event") == "sdk_send_started" for row in events),
            "tool_calls": len(state.get("tool_calls") or []),
            "sql_attempts": sum(row.get("event") == "sql_attempt_started" for row in events),
            "sql_succeeded": sum(row.get("event") == "sql_attempt_succeeded" for row in events),
            "sql_failed": sum(row.get("event") == "sql_attempt_failed" for row in events),
            "token_usage": usage,
            "estimated_cost_accounting": accounting,
        }
        accounting_failed = usage.get("status") == "metering_error" or bool(accounting.get("reservation_exceeded"))
        transport_unknown = bool(usage.get("unknown_or_failed_usage_requests") or usage.get("provider_failed_sends"))
        if accounting_failed or transport_unknown:
            score.update({
                "lifecycle_status": "failed", "task_acceptance_pass": False,
                "task_acceptance_status": "failed",
                "task_pass_reason": "metering_or_reservation_failure" if accounting_failed else "provider_usage_unknown",
            })
        record = {
            "run_id": run_id, "case_id": entry["case_id"], "arm": arm,
            "replicate": entry.get("replicate"), "response": response,
            "trace": trace, "score": score,
            "companion_quality_audit": companion_quality_audit,
            "companion_quality_audit_error": companion_quality_audit_error,
            "model_authored_score": model_authored_score,
            "output_adaptation": state.get("final_output_adaptation"),
            "measurements": measurements,
            "result_lifecycle_status": "failed" if accounting_failed or transport_unknown else "completed",
        }
        record_path = trace_dir / f"{run_id}.run_record.json"
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        ledger.append(run_id, "usage_summary", **{key: value for key, value in usage.items() if key != "run_id"})
        if accounting_failed or transport_unknown:
            ledger.append(run_id, "failed", error_type="metering_or_reservation_failure" if accounting_failed else "provider_usage_unknown", retry_forbidden=True)
            raise RuntimeError("B/G run metering or estimate reservation requires stopping the batch."
                               if accounting_failed else "B/G Provider usage is unknown; stop the batch without retry.")
        ledger.append(run_id, "completed", provider_sends=measurements["sdk_sends"], task_acceptance_pass=score["task_acceptance_pass"])
        return record
    except BaseException as exc:
        latest = ledger.latest().get(run_id, {})
        if latest.get("event") not in {"failed", "completed", "interrupted_before_send", "interrupted_result_unknown"}:
            failure = {
                "run_id": run_id, "case_id": entry.get("case_id"), "arm": arm,
                "lifecycle_status": "failed", "error_type": type(exc).__name__,
                "provider_sends": sum(
                    row.get("run_id") == run_id and row.get("event") == "sdk_send_started"
                    for row in ledger.events
                ),
                "trace_present": trace_path.is_file(), "retry_forbidden": True,
            }
            if trace_path.is_file():
                try:
                    failed_trace = json.loads(trace_path.read_text(encoding="utf-8"))
                    from evaluation.resume_bg_observation import extract_formal_observation
                    from evaluation.resume_bg_companion_audit import build_companion_quality_audit
                    failed_observation = extract_formal_observation(case, arm, failed_trace)
                    failure["companion_quality_audit"] = build_companion_quality_audit(
                        case, arm, failed_trace, failed_observation, expected_run_id=run_id,
                    )
                except Exception as audit_error:
                    failure["companion_quality_audit_error"] = {
                        "error_type": type(audit_error).__name__, "error": str(audit_error),
                    }
            (trace_dir / f"{run_id}.failure_record.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            ledger.append(run_id, "failed", error_type=type(exc).__name__, retry_forbidden=True)
        raise


def run_bg_batch(
    *, root: Path, cases: list[Mapping[str, Any]], manifest: Mapping[str, Any],
    execution_plan: Mapping[str, Any], max_units: int, estimate_limit_cny: float,
    provider_authorization_cny: float, admin_url_text: str, ledger_path: Path,
    cost_ledger_path: Path, trace_dir: Path, expected_csv_hashes: Mapping[str, str],
    dataset_dir: Path = DEFAULT_DATASET_ROOT,
    condition_probe: Any = None,
    tenant_context: Any = None,
) -> list[dict[str, Any]]:
    """Explicit Phase B API. It refuses a blocked Phase A before DB startup."""
    readiness = validate_bg_schedule(cases, manifest, execution_plan=execution_plan)
    if readiness["execution_blocked"]:
        raise RuntimeError("Phase A execution gates are still blocked; no database or run was started.")
    if execution_plan.get("scorer_version") != SCORER_VERSION:
        raise RuntimeError("Frozen B/G score version does not match the executor.")
    ledger, cost_policy = RunLedger(ledger_path), CostPolicy(
        cumulative_estimate_limit_cny=float(estimate_limit_cny),
    )
    cost_ledger = EstimatedCostLedger(cost_policy, cost_ledger_path)
    selected = select_never_started(list(manifest["entries"]), ledger, max_units=max_units)
    validate_execute_budget(
        selected_count=len(selected), policy=cost_policy, cost_ledger=cost_ledger,
        estimate_limit_cny=estimate_limit_cny,
        provider_authorization_cny=provider_authorization_cny,
    )
    if not selected:
        return []
    engine = None
    database_name = None
    outputs: list[dict[str, Any]] = []
    try:
        engine, database_name, _coverage = load_isolated_candidate_database(
            admin_url_text, expected_csv_hashes=expected_csv_hashes, dataset_dir=dataset_dir,
        )
        context = resolve_batch_tenant_context(tenant_context)
        cases_by_id = {str(row["case_id"]): row for row in cases}
        # Special refusal/insufficiency tasks must prove their precondition from
        # the just-loaded isolated dataset before any run is reserved or started.
        # The probe is required to use the normal authorized query/registry path.
        pre_run_conditions = []
        for entry in selected:
            case = cases_by_id[str(entry["case_id"])]
            task = str(case.get("protocol_task_type") or "")
            if task not in {"expected_data_insufficiency", "expected_capability_limit"}:
                continue
            observation = condition_probe(case, tenant_context=context, engine=engine) if callable(condition_probe) else None
            result = validate_case_pre_run_condition(case, observation=observation)
            pre_run_conditions.append({"case_id": case.get("case_id"), "run_id": entry.get("run_id"), **result})
        blocked_conditions = [item for item in pre_run_conditions if item.get("status") != "ready"]
        if blocked_conditions:
            raise RuntimeError("Special-case pre-run condition was invalid or unverified; no run was started: "
                               + json.dumps(blocked_conditions, ensure_ascii=False, sort_keys=True))
        config = ExperimentConfig()
        with bind_isolated_experiment_runtime(
            engine=engine, trace_dir=trace_dir, ledger=ledger, config=config,
        ):
            for entry in selected:
                case = cases_by_id[str(entry["case_id"])]
                outputs.append(run_bg_unit(
                    entry, case, tenant_context=context, trace_dir=trace_dir,
                    ledger=ledger, cost_ledger=cost_ledger, experiment_config=config,
                ))
                if cost_ledger.reserved_cny > estimate_limit_cny + 1e-12:
                    raise CumulativeCostLimitExceeded("Cumulative B/G estimate limit reached.")
    finally:
        if engine is not None and database_name:
            engine.dispose()
            drop_isolated_candidate_database(admin_url_text, database_name, engine)
        if outputs:
            try:
                from evaluation.resume_bg_companion_audit import summarize_companion_quality_records
                companion_summary = summarize_companion_quality_records(outputs)
                companion_summary.update({
                    "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "batch_run_ids": [str(item.get("run_id")) for item in outputs],
                    "summary_kind": "this_run_bg_batch_only",
                })
                trace_dir.mkdir(parents=True, exist_ok=True)
                summary_path = trace_dir / f"companion_quality_summary_{uuid4()}.json"
                summary_path.write_text(
                    json.dumps(companion_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
                )
            except Exception as summary_error:
                try:
                    error_path = trace_dir / "companion_quality_summary_errors.jsonl"
                    with error_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "error_type": type(summary_error).__name__,
                            "run_ids": [str(item.get("run_id")) for item in outputs],
                        }, ensure_ascii=False) + "\n")
                except OSError:
                    pass
    return outputs
