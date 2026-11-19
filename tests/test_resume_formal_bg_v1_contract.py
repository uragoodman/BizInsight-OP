from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tenancy import TenantContext
from evaluation.resume_formal_bg_v1 import (
    build_formal_manifest,
    compare_operation_contracts,
    candidate_operation_contract,
    score_formal_run,
    validate_candidate_set,
    validate_formal_manifest,
    validate_paired_manifest,
    verify_data_insufficiency_precondition,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "docs/testing/resume_formal_bg_design_20261019_v1"


def test_60_candidate_design_counts_and_limits():
    payload = json.loads((DESIGN / "candidate_cases_60.json").read_text(encoding="utf-8"))
    summary = validate_candidate_set(payload["cases"])
    assert summary["case_count"] == 60
    assert summary["stratum_counts"] == {
        "normal_query": 24,
        "multi_step_diagnostic": 16,
        "expected_limitation": 8,
        "fault_or_budget": 12,
    }
    assert summary["unique_metrics"] >= 20
    assert summary["max_cases_per_metric"] <= 4
    assert summary["max_cases_per_scenario_family"] <= 6
    assert summary["truth_ready_cases"] == 0


def test_offline_fairness_preflight_maps_all_cases_without_db_or_provider():
    result = json.loads((DESIGN / "offline_fairness_preflight.json").read_text(encoding="utf-8"))
    assert result["network_blocked"] is True
    assert result["unexpected_network_attempts"] == []
    assert result["database_started"] is False
    assert result["provider_requests"] == 0
    assert result["case_count"] == 60
    assert result["static_contract_mapped_count"] == 60
    assert result["blocked_count"] == 0
    assert result["formal_execution_ready"] is False
    assert result["tool_equivalence"]["literal_model_visible_schema_identical"] is False


def test_arm_neutral_scorer_does_not_confuse_safe_stop_with_diagnosis():
    case = {
        "truth_status": "verified", "operation": "metric_compare", "target_metric": "gmv",
        "scope": "platform", "period": {"kind": "point_pair", "baseline": "2017-01-05", "current": "2017-01-06"},
        "expected": {"outcome": "safe_stop", "terminal_code": "DATA_INSUFFICIENT"},
        "governance_requirements_by_arm": {"B": "not_applicable", "G": "critic_completed"},
    }
    observation = {
        "operation": "metric_compare", "metric_id": "gmv", "scope": "platform",
        "period": case["period"], "evidence_valid": True, "answer_card_consistent": True,
        "terminal_code": "DATA_INSUFFICIENT", "limitation_disclosed": True,
        "precondition_verified": True,
        "effective_diagnostic_completed": False, "governance_execution": "critic_ran",
    }
    scored = score_formal_run(case, observation=observation, lifecycle_status="completed", arm="B")
    assert scored["task_acceptance_pass"] is True
    assert scored["correct_safe_stop"] is True
    assert scored["effective_diagnostic_completed"] is False
    assert scored["governance_execution"] == "critic_ran"
    assert scored["experiment_unit_integrity_pass"] is False
    g_observation = {**observation, "governance_execution": "critic_completed"}
    g_scored = score_formal_run(case, observation=g_observation, lifecycle_status="completed", arm="G")
    assert g_scored["task_acceptance_pass"] == scored["task_acceptance_pass"]
    assert g_scored["experiment_unit_integrity_pass"] is True


def test_unverified_case_truth_is_not_scored_as_pass_or_failure():
    scored = score_formal_run(
        {"truth_status": "pending_independent_source_verification"},
        observation={}, lifecycle_status="completed",
    )
    assert scored["task_acceptance_pass"] is None
    assert scored["task_acceptance_status"] == "blocked_truth_not_verified"
    assert scored["manual_review_required"] is True


def test_common_contract_rejects_scope_or_period_drift():
    case = {
        "case_id": "x", "operation": "metric_compare", "target_metric": "gmv",
        "scope": "platform", "period": {"baseline": "a", "current": "b"},
        "dimension": None, "method_id": None, "dependency_metric_ids": [],
    }
    good = {
        "operation": case["operation"], "metric_id": "gmv", "scope": "platform",
        "period": case["period"], "dimension": None, "method_id": None,
        "dependency_metric_ids": [],
    }
    bad = {**good, "period": {"baseline": "b", "current": "a"}}
    assert compare_operation_contracts(case, b_contract=good, g_contract=good)["status"] == "equivalent"
    assert compare_operation_contracts(case, b_contract=good, g_contract=bad)["status"] == "blocked_contract_mismatch"


def test_b_value_and_compare_plans_use_same_identity_and_explicit_roles():
    from evaluation.resume_bare_tool_loop import _authorized_plan

    context = TenantContext(
        tenant_id="platform", user_id="provider-candidate-experiment",
        role="admin", scope_type="platform",
    )
    question = "请比较平台 GMV 基线日期 2017-01-06 与当前日期 2017-01-05 的数值。"
    value_question = "请查询平台 GMV 在 2017-01-05 的数值。"
    value = _authorized_plan(
        question=value_question, tool="query_value",
        arguments={"metric_id": "gmv", "metric_date": "2017-01-05"},
        context=context, run_id="offline-value", sequence=1,
    )
    compare = _authorized_plan(
        question=question, tool="metric_compare",
        arguments={"metric_id": "gmv", "baseline_date": "2017-01-06", "current_date": "2017-01-05"},
        context=context, run_id="offline-compare", sequence=1,
    )
    assert (value.query_type, value.metric_date, value.metric_name) == ("metric_value", "2017-01-05", "gmv")
    assert (compare.query_type, compare.baseline_date, compare.current_date) == (
        "metric_compare", "2017-01-06", "2017-01-05",
    )


def test_b_loop_cooperative_total_deadline_keeps_tool_result_and_stops_next_step(monkeypatch):
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop
    from types import SimpleNamespace
    import evaluation.resume_bare_tool_loop as loop_module

    context = TenantContext(
        tenant_id="platform", user_id="provider-candidate-experiment",
        role="admin", scope_type="platform",
    )
    model_actions = iter([
        {"action": "TOOL_CALL", "tool": "query_trend", "arguments": {
            "metric_id": "gmv", "start_date": "2017-01-05", "end_date": "2017-01-06",
        }},
        {"action": "FINAL", "answer": "later", "claims": [], "presentation": {}},
    ])
    calls = {"model": 0, "tool": 0}

    def model_fn(_system: str, _user: str):
        calls["model"] += 1
        return next(model_actions)

    clock_values = iter([10.0, 10.0, 10.0, 10.0, 10.002])
    monkeypatch.setattr(loop_module, "time", SimpleNamespace(monotonic=lambda: next(clock_values)))

    def tool_fn(**kwargs):
        calls["tool"] += 1
        return {
            "tool_name": "query_metric_trend", "request_id": kwargs["request_id"],
            "result": {"status": "ok", "parsed": {"metric_name": "gmv"}, "points": []},
        }

    result = run_bare_tool_loop(
        "查询平台 GMV 从 2017-01-05 到 2017-01-06 的趋势。",
        tenant_context=context, run_id="deadline-case", model_fn=model_fn,
        tool_fn=tool_fn, max_latency_ms=1,
    )
    state = result
    assert calls == {"model": 1, "tool": 1}
    assert state["terminal_code"] == "TIMEOUT"
    assert state["errors"][-1]["code"] == "workflow_deadline_exceeded"
    assert len(state["evidence_bundles"]) == 1
    assert state["runtime_budget"]["max_latency_ms"] == 1


def test_g_graph_node_enforces_same_elapsed_deadline_at_node_boundary():
    import time
    from app.graph.runtime import instrument_node

    entered = {"value": False}
    wrapped = instrument_node("preflight_deadline", lambda state: entered.update(value=True) or state)
    state = {
        "runtime_started_at": time.perf_counter() - 1.0,
        "runtime_steps": 0,
        "runtime_budget": {"max_latency_ms": 1, "max_graph_steps": 48},
        "errors": [],
    }
    result = wrapped(state)
    assert entered["value"] is False
    assert result["errors"][-1]["code"] == "graph_latency_budget_exceeded"


def test_formal_manifest_has_360_unique_paired_units_and_balanced_order():
    payload = json.loads((ROOT / "docs/testing/resume_formal_bg_design_20261019_v2/candidate_cases_60.json").read_text(encoding="utf-8"))
    manifest = build_formal_manifest(payload["cases"], seed=20261019, experiment_id="resume-formal-bg-v2-test")
    summary = validate_formal_manifest(manifest, payload["cases"])
    assert summary["planned_cases"] == 60
    assert summary["planned_pairs"] == 180
    assert summary["planned_units"] == 360
    assert summary["run_id_count"] == 360
    assert summary["arm_counts"] == {"B": 180, "G": 180}
    assert summary["first_arm_pair_counts"] == {"B": 90, "G": 90}
    assert summary["all_not_started"] is True
    assert validate_formal_manifest(manifest, payload["cases"]) == summary


def test_month_pair_task_has_semantically_matched_period_contribution_route():
    case = {
        "case_id": "sample", "operation": "dimension_contribution", "target_metric": "freight_rate",
        "scope": "platform", "period": {"kind": "month_pair", "baseline_month": "2016-12", "current_month": "2017-01"},
        "dimension": "month_position", "method_id": "symmetric_mix_within_v1", "dependency_metric_ids": ["freight_value", "gmv"],
    }
    b_contract = candidate_operation_contract(case, arm="B")
    g_contract = candidate_operation_contract(case, arm="G")
    assert b_contract["semantic_route"] == "period_metric_contribution"
    assert g_contract["semantic_route"] == "metric_contribution"
    assert compare_operation_contracts(case, b_contract=b_contract, g_contract=g_contract)["status"] == "equivalent"


def test_send_time_budget_and_estimate_must_share_the_same_input_bounds():
    from evaluation.provider_candidate_batch import CostPolicy, ExperimentConfig

    config = ExperimentConfig()
    CostPolicy().validate_enforcement_config(config)
    mismatched = CostPolicy(input_token_estimate_cap_per_call=199_999)
    with pytest.raises(ValueError, match="input_token_estimate_cap_per_call"):
        mismatched.validate_enforcement_config(config)


def test_formal_dry_run_reads_frozen_360_schedule_and_shared_fee_policy(tmp_path):
    from evaluation.provider_candidate_batch import CostPolicy, EstimatedCostLedger, ExperimentConfig, RunLedger
    from evaluation.resume_formal_bg_executor import FormalBGBatchExecutor, runtime_freeze_hashes

    directory = ROOT / "docs/testing/resume_formal_bg_design_20261019_v2"
    case_path = directory / "candidate_cases_60.json"
    candidate = json.loads(case_path.read_text(encoding="utf-8"))
    manifest = json.loads((directory / "formal_manifest_360.json").read_text(encoding="utf-8"))
    qualification_path = directory / "offline_candidate_qualification_v2.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    policy = CostPolicy()
    executor = FormalBGBatchExecutor(
        root=ROOT, cases=candidate["cases"], manifest=manifest,
        frozen_hashes={
            **runtime_freeze_hashes(ROOT),
            "docs/testing/resume_formal_bg_design_20261019_v2/candidate_cases_60.json": __import__("hashlib").sha256(case_path.read_bytes()).hexdigest(),
            "docs/testing/resume_formal_bg_design_20261019_v2/offline_candidate_qualification_v2.json": __import__("hashlib").sha256(qualification_path.read_bytes()).hexdigest(),
        },
        qualification=qualification, ledger=RunLedger(tmp_path / "ledger.jsonl"),
        cost_ledger=EstimatedCostLedger(policy, tmp_path / "cost.jsonl"),
        trace_dir=tmp_path / "traces", config=ExperimentConfig(),
    )
    result = executor.dry_run(cost_policy=policy)
    assert result["planned_units"] == 360
    assert result["qualified_case_count"] == 48
    assert result["blocked_case_count"] == 12
    assert result["semantic_contract_pairs_checked"] == 60
    assert result["semantic_contract_pairs_equivalent"] == 60
    assert result["lifecycle_counts"]["not_started"] == 360
    assert result["provider_sends"] == 0
    assert result["database_started"] is False
    assert result["planned_cost_estimate"]["estimated_cny"] == pytest.approx(838.3365432)
    assert result["planned_cost_estimate"]["is_account_hard_cap"] is False


def test_formal_executor_rejects_in_memory_candidate_drift_before_any_run(tmp_path):
    from evaluation.provider_candidate_batch import CostPolicy, EstimatedCostLedger, ExperimentConfig, RunLedger
    from evaluation.resume_formal_bg_executor import FormalBGBatchExecutor, runtime_freeze_hashes

    directory = ROOT / "docs/testing/resume_formal_bg_design_20261019_v2"
    candidate_path = directory / "candidate_cases_60.json"
    qualification_path = directory / "offline_candidate_qualification_v2.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    manifest = json.loads((directory / "formal_manifest_360.json").read_text(encoding="utf-8"))
    candidate["cases"][0]["period"] = {"kind": "point_pair", "baseline": "wrong", "current": "wrong"}
    executor = FormalBGBatchExecutor(
        root=ROOT, cases=candidate["cases"], manifest=manifest,
        frozen_hashes={
            **runtime_freeze_hashes(ROOT),
            "docs/testing/resume_formal_bg_design_20261019_v2/candidate_cases_60.json": __import__("hashlib").sha256(candidate_path.read_bytes()).hexdigest(),
            "docs/testing/resume_formal_bg_design_20261019_v2/offline_candidate_qualification_v2.json": __import__("hashlib").sha256(qualification_path.read_bytes()).hexdigest(),
        }, qualification=qualification, ledger=RunLedger(tmp_path / "ledger.jsonl"),
        cost_ledger=EstimatedCostLedger(CostPolicy(), tmp_path / "cost.jsonl"),
        trace_dir=tmp_path / "traces", config=ExperimentConfig(),
    )
    with pytest.raises(ValueError, match="In-memory candidate rows"):
        executor.verify_inputs()


def test_execute_rejects_incomplete_runtime_freeze_before_database_or_reservation(tmp_path, monkeypatch):
    from evaluation.provider_candidate_batch import CostPolicy, EstimatedCostLedger, ExperimentConfig, RunLedger
    from evaluation.resume_formal_bg_executor import FormalBGBatchExecutor
    from evaluation import resume_formal_bg_executor as executor_module

    directory = ROOT / "docs/testing/resume_formal_bg_design_20261019_v2"
    candidate_path = directory / "candidate_cases_60.json"
    qualification_path = directory / "offline_candidate_qualification_v2.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    manifest = json.loads((directory / "formal_manifest_360.json").read_text(encoding="utf-8"))
    frozen_hashes = {
        candidate_path.relative_to(ROOT).as_posix(): __import__("hashlib").sha256(candidate_path.read_bytes()).hexdigest(),
        qualification_path.relative_to(ROOT).as_posix(): __import__("hashlib").sha256(qualification_path.read_bytes()).hexdigest(),
    }
    executor = FormalBGBatchExecutor(
        root=ROOT, cases=candidate["cases"], manifest=manifest,
        frozen_hashes=frozen_hashes, qualification=qualification,
        ledger=RunLedger(tmp_path / "ledger.jsonl"),
        cost_ledger=EstimatedCostLedger(CostPolicy(), tmp_path / "cost.jsonl"),
        trace_dir=tmp_path / "traces", config=ExperimentConfig(),
    )
    database_started = []
    monkeypatch.setattr(executor_module, "load_isolated_candidate_database", lambda *a, **k: database_started.append(True))
    with pytest.raises(ValueError, match="omits runtime code/config/prompt inputs"):
        executor.execute(
            max_units=1, estimate_limit_cny=5, provider_authorization_cny=5,
            admin_url_text="postgresql://unused", expected_csv_hashes={},
            observation_builder=lambda *args: {},
        )
    assert database_started == []
    assert executor.ledger.events == []
    assert executor.cost_ledger.reservations == []


def test_formal_send_budget_preflight_uses_selected_count_and_existing_reserve():
    from evaluation.provider_candidate_batch import CostPolicy, CumulativeCostLimitExceeded, ExperimentConfig
    from evaluation.resume_formal_bg_executor import validate_formal_budget

    policy, config = CostPolicy(), ExperimentConfig()
    pair = validate_formal_budget(
        planned_units=360, max_units=2, estimate_limit_cny=5, provider_authorization_cny=5,
        cost_policy=policy, experiment_config=config, reserved_cny=0,
    )
    assert pair["new_estimated_reservation_cny"] == pytest.approx(4.65742524)
    assert pair["is_account_hard_cap"] is False
    with pytest.raises(CumulativeCostLimitExceeded):
        validate_formal_budget(
            planned_units=360, max_units=12, estimate_limit_cny=5, provider_authorization_cny=5,
            cost_policy=policy, experiment_config=config, reserved_cny=0,
        )
    with pytest.raises(CumulativeCostLimitExceeded):
        validate_formal_budget(
            planned_units=360, max_units=1, estimate_limit_cny=5, provider_authorization_cny=5,
            cost_policy=policy, experiment_config=config, reserved_cny=3,
        )
    with pytest.raises(ValueError, match="must match"):
        validate_formal_budget(
            planned_units=360, max_units=1, estimate_limit_cny=5, provider_authorization_cny=4,
            cost_policy=policy, experiment_config=config, reserved_cny=0,
        )
    with pytest.raises(ValueError, match="append-only cost ledger"):
        validate_formal_budget(
            planned_units=360, max_units=360, estimate_limit_cny=900, provider_authorization_cny=900,
            cost_policy=policy, experiment_config=config, reserved_cny=0,
        )
    already_reserved = validate_formal_budget(
        planned_units=360, max_units=0, estimate_limit_cny=5, provider_authorization_cny=5,
        cost_policy=policy, experiment_config=config, reserved_cny=policy.max_per_run_cny(),
    )
    assert already_reserved["new_estimated_reservation_cny"] == 0
    assert already_reserved["existing_reservations_cny"] == pytest.approx(policy.max_per_run_cny())
    resumed_pair = validate_formal_budget(
        planned_units=360, max_units=1, estimate_limit_cny=5, provider_authorization_cny=5,
        cost_policy=policy, experiment_config=config, reserved_cny=policy.max_per_run_cny(),
    )
    assert resumed_pair["new_estimated_reservation_cny"] == pytest.approx(policy.max_per_run_cny())
    with pytest.raises(CumulativeCostLimitExceeded):
        validate_formal_budget(
            planned_units=360, max_units=2, estimate_limit_cny=5, provider_authorization_cny=5,
            cost_policy=policy, experiment_config=config, reserved_cny=policy.max_per_run_cny(),
        )


def test_12_case_two_round_manifest_is_accepted_by_generic_executor_and_dry_run(tmp_path):
    """The 12×2 revision is validated from its frozen inputs, not 60×3 constants."""
    from evaluation.provider_candidate_batch import CostPolicy, EstimatedCostLedger, ExperimentConfig, RunLedger
    from evaluation.resume_formal_bg_executor import FormalBGBatchExecutor

    directory = ROOT / "docs/testing/resume_formal_bg_design_20261019_v3_12case_2round"
    selected_path = directory / "selected_cases_12.json"
    manifest_path = directory / "run_manifest_48.json"
    candidate_path = ROOT / "docs/testing/resume_formal_bg_design_20261019_v2/candidate_cases_60.json"
    qualification_path = ROOT / "docs/testing/resume_formal_bg_design_20261019_v2/offline_candidate_qualification_v2.json"
    selected_doc = json.loads(selected_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    hashes = {
        path.relative_to(ROOT).as_posix(): __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        for path in (candidate_path, qualification_path, selected_path, manifest_path)
    }
    from evaluation.resume_formal_bg_executor import runtime_freeze_hashes
    hashes.update(runtime_freeze_hashes(ROOT))
    executor = FormalBGBatchExecutor(
        root=ROOT, cases=selected_doc["cases"], manifest=manifest,
        frozen_hashes=hashes, qualification=qualification,
        ledger=RunLedger(tmp_path / "ledger.jsonl"),
        cost_ledger=EstimatedCostLedger(CostPolicy(), tmp_path / "cost.jsonl"),
        trace_dir=tmp_path / "traces", config=ExperimentConfig(),
    )
    verified = executor.verify_inputs()
    assert verified["planned_cases"] == 12
    assert verified["replicates_per_case"] == 2
    assert verified["planned_pairs"] == 24
    assert verified["planned_units"] == 48
    assert verified["selected_qualified_case_count"] == 12
    assert verified["semantic_contract_pairs_checked"] == 12
    assert verified["semantic_contract_pairs_equivalent"] == 12
    assert verified["same_identity_contract"] == [
        "platform", "provider-candidate-experiment", "admin", "platform",
    ]
    assert verified["same_identity_case_count"] == 12

    summary = validate_paired_manifest(manifest, selected_doc["cases"])
    assert summary["arm_counts"] == {"B": 24, "G": 24}
    assert summary["first_arm_pair_counts"] == {"B": 12, "G": 12}
    assert summary["all_not_started"] is True
    dry_run = executor.dry_run(cost_policy=CostPolicy())
    assert dry_run["lifecycle_counts"]["not_started"] == 48
    assert dry_run["denominator_contract"]["planned_unit_denominator"] == 48
    assert dry_run["denominator_contract"]["failed_timeout_interrupted_and_sent_unknown_usage_remain_in_denominator"] is True
    assert dry_run["provider_sends"] == 0
    assert dry_run["database_started"] is False
    first_run_id = manifest["entries"][0]["run_id"]
    executor.ledger.append(first_run_id, "started")
    executor.ledger.append(first_run_id, "sdk_send_started", attempt_id="offline-attempt")
    executor.ledger.append(first_run_id, "failed", retry_forbidden=True)
    after_failure = executor.dry_run(cost_policy=CostPolicy())
    assert after_failure["lifecycle_counts"]["failed"] == 1
    assert after_failure["lifecycle_counts"]["not_started"] == 47
    assert after_failure["provider_sends"] == 1


def test_formal_executor_uses_frozen_identity_and_refuses_runtime_identity_drift():
    from app.tenancy import TenantContext
    from evaluation.resume_formal_bg_executor import resolve_formal_tenant_context

    cases = [{"case_id": "one", "same_identity": {
        "tenant_id": "platform", "user_id": "frozen-user", "role": "admin", "scope_type": "platform",
    }}]
    context = resolve_formal_tenant_context(cases)
    assert (context.tenant_id, context.user_id, context.role, context.scope_type) == (
        "platform", "frozen-user", "admin", "platform",
    )
    assert resolve_formal_tenant_context(cases, context) is context
    with pytest.raises(ValueError, match="does not match"):
        resolve_formal_tenant_context(cases, TenantContext(
            tenant_id="platform", user_id="different-user", role="admin", scope_type="platform",
        ))


def test_expected_data_insufficiency_requires_both_valid_baseline_and_missing_current():
    case = {
        "truth_status": "verified", "operation": "expected_data_insufficiency", "target_metric": "gmv",
        "scope": "platform", "period": {"kind": "point_pair", "baseline": "d1", "current": "d2"},
        "expected": {"outcome": "safe_stop", "terminal_code": "DATA_INSUFFICIENT"},
    }
    base_observation = {
        "operation": "expected_data_insufficiency", "metric_id": "gmv", "scope": "platform",
        "period": case["period"], "evidence_valid": True, "answer_card_consistent": True,
        "terminal_code": "DATA_INSUFFICIENT", "limitation_disclosed": True,
        "precondition_verified": True,
        "effective_diagnostic_completed": False,
    }
    passed = score_formal_run(case, observation=base_observation, lifecycle_status="completed")
    assert passed["task_acceptance_pass"] is True
    # The expected safe-stop precondition is an independent required check.
    false_positive = {**base_observation, "precondition_verified": False}
    rejected = score_formal_run(case, observation=false_positive, lifecycle_status="completed")
    assert rejected["task_acceptance_pass"] is False
    assert rejected["correct_safe_stop"] is False


def test_expected_data_insufficiency_scores_the_declared_query_operation():
    case = {
        "truth_status": "verified", "operation": "expected_data_insufficiency",
        "b_action_contract": "metric_compare", "g_semantic_route": "metric_compare",
        "target_metric": "avg_delivery_lead_time", "scope": "platform",
        "period": {"kind": "point_pair", "baseline": "d1", "current": "d2"},
        "expected": {"outcome": "safe_stop", "terminal_code": "DATA_INSUFFICIENT"},
    }
    observation = {
        "operation": "metric_compare", "metric_id": "avg_delivery_lead_time",
        "scope": "platform", "period_match": True, "evidence_valid": True,
        "answer_card_consistent": True, "terminal_code": "DATA_INSUFFICIENT",
        "expected_stop": True, "precondition_verified": True,
        "limitation_disclosed": True, "manual_review_required": False,
    }
    score = score_formal_run(case, observation=observation, lifecycle_status="completed", arm="G")
    assert score["checks"]["operation_match"] is True
    assert score["task_acceptance_pass"] is True
    outcome_labeled = {**observation, "operation": "expected_data_insufficiency"}
    score = score_formal_run(case, observation=outcome_labeled, lifecycle_status="completed", arm="G")
    assert score["checks"]["operation_match"] is True


def test_expected_data_gap_accepts_valid_baseline_in_not_found_comparison_but_rejects_weak_evidence():
    verified = verify_data_insufficiency_precondition(
        152.35,
        [{"status": "not_found", "baseline_value": 152.35, "current_value": None}],
    )
    assert verified == {
        "baseline_value_matches_independent_csv": True,
        "current_period_has_explicit_not_found": True,
        "verified": True,
    }
    assert not verify_data_insufficiency_precondition(
        152.35, [{"status": "not_found", "baseline_value": None, "current_value": None}],
    )["verified"]
    assert not verify_data_insufficiency_precondition(
        152.35, [{"status": "ok", "baseline_value": 152.35, "current_value": 0}],
    )["verified"]


def test_batch_identity_resolver_keeps_one_trusted_context_for_either_arm():
    from evaluation.resume_bg_executor import resolve_batch_tenant_context

    context = TenantContext(
        tenant_id="platform", user_id="provider-candidate-experiment",
        role="admin", scope_type="platform",
    )
    assert resolve_batch_tenant_context(context) is context
    assert resolve_batch_tenant_context(context).as_state() == context.as_state()
    with pytest.raises(TypeError):
        resolve_batch_tenant_context({"tenant_id": "platform", "user_id": "caller"})
