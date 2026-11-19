from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def test_critic_ablation_mode_runs_both_arms_through_same_workflow(monkeypatch, tmp_path):
    from evaluation import resume_formal_bg_executor as module
    from evaluation.provider_candidate_batch import CostPolicy, EstimatedCostLedger, ExperimentConfig, RunLedger
    from app.graph.critic_review_mode import CriticLLMReviewMode
    from app.tenancy import TenantContext

    root = Path(__file__).resolve().parents[1]
    case = {
        "case_id": "offline-case", "truth_status": "verified", "user_question": "test request",
        "operation": "metric_compare", "target_metric": "order_count", "scope": "platform",
        "period": {"kind": "point_pair", "baseline": "2017-02-11", "current": "2017-02-12"},
        "expected": {"outcome": "numeric_result"},
    }
    config = ExperimentConfig()
    cost_policy = CostPolicy(cumulative_estimate_limit_cny=5)
    cost_ledger = EstimatedCostLedger(cost_policy, tmp_path / "cost.jsonl")
    executor = module.FormalBGBatchExecutor(
        root=root, cases=[case], manifest={"entries": []}, frozen_hashes={}, qualification={},
        ledger=RunLedger(tmp_path / "events.jsonl"), cost_ledger=cost_ledger,
        trace_dir=tmp_path / "traces", config=config,
        arm_execution_mode="critic_off_vs_full_workflow",
    )
    calls = []

    class FakeCallConfig:
        @staticmethod
        def provider_ab_experiment(*_args, **_kwargs):
            return object()

    monkeypatch.setattr(module, "token_usage_snapshot", lambda *_a, **_k: {})
    monkeypatch.setattr(module, "_reconcile_cost_accounting", lambda *_a, **_k: {})
    monkeypatch.setattr(module, "score_formal_run", lambda *_a, **_k: {
        "scorer_version": "offline-test", "task_acceptance_status": "manual_review_required",
        "manual_review_required": True,
    })
    monkeypatch.setattr(FakeCallConfig, "provider_ab_experiment", FakeCallConfig.provider_ab_experiment)

    from app.graph import workflow
    monkeypatch.setattr(workflow, "get_settings", lambda: SimpleNamespace())

    def fake_run_chat(message, **kwargs):
        calls.append((message, kwargs["_critic_llm_review_mode"], kwargs["_critic_llm_thinking_mode"]))
        trace_id = kwargs["trace_id"]
        (executor.trace_dir / f"{trace_id}.json").write_text(
            json.dumps({"trace_id": trace_id, "state_snapshot": {"final_answer": "offline"}}),
            encoding="utf-8",
        )
        return {"answer": "offline"}

    monkeypatch.setattr(workflow, "run_chat_workflow", fake_run_chat)
    monkeypatch.setattr("app.llm.call_config.LLMCallConfig", FakeCallConfig)
    monkeypatch.setattr(cost_ledger, "record_usage", lambda *_a, **_k: {})

    for arm, run_id in (("B", "offline-b"), ("G", "offline-g")):
        executor._run_one(
            {"run_id": run_id, "arm": arm, "case_id": case["case_id"], "replicate": 1},
            case,
            tenant_context=TenantContext(
                tenant_id="platform", user_id="offline-test", role="admin", scope_type="platform",
            ),
            observation_builder=lambda *_a: {}, fault_context_factory=None,
        )

    assert [row[1] for row in calls] == [
        CriticLLMReviewMode.CRITIC_LLM_REVIEW_OFF,
        CriticLLMReviewMode.FULL,
    ]
    assert [row[0] for row in calls] == ["test request", "test request"]
    assert all(row[2].value == "non_thinking_experiment" for row in calls)


def test_legacy_formal_executor_mode_remains_bare_tool_baseline():
    import inspect
    from evaluation.resume_formal_bg_executor import FormalBGBatchExecutor

    assert inspect.signature(FormalBGBatchExecutor.__init__).parameters[
        "arm_execution_mode"
    ].default == "bare_tool_baseline_vs_governed_workflow"


def test_adaptive_critic_mode_maps_only_the_adaptive_arm(monkeypatch, tmp_path):
    from evaluation import resume_formal_bg_executor as module
    from evaluation.provider_candidate_batch import CostPolicy, EstimatedCostLedger, ExperimentConfig, RunLedger
    from app.graph.critic_review_mode import CriticLLMReviewMode
    from app.tenancy import TenantContext

    root = Path(__file__).resolve().parents[1]
    case = {"case_id": "adaptive-offline", "truth_status": "verified", "user_question": "test request",
            "operation": "metric_compare", "target_metric": "order_count", "scope": "platform",
            "period": {"kind": "point_pair", "baseline": "2017-02-11", "current": "2017-02-12"},
            "expected": {"outcome": "numeric_result"}}
    policy = CostPolicy(cumulative_estimate_limit_cny=5)
    executor = module.FormalBGBatchExecutor(
        root=root, cases=[case], manifest={"entries": []}, frozen_hashes={}, qualification={},
        ledger=RunLedger(tmp_path / "events.jsonl"), cost_ledger=EstimatedCostLedger(policy, tmp_path / "cost.jsonl"),
        trace_dir=tmp_path / "traces", config=ExperimentConfig(),
        arm_execution_mode="critic_adaptive_vs_full_workflow",
    )
    calls = []
    monkeypatch.setattr(module, "token_usage_snapshot", lambda *_a, **_k: {})
    monkeypatch.setattr(module, "_reconcile_cost_accounting", lambda *_a, **_k: {})
    monkeypatch.setattr(module, "score_formal_run", lambda *_a, **_k: {"scorer_version": "offline", "task_acceptance_status": "manual_review_required", "manual_review_required": True})
    from app.graph import workflow
    monkeypatch.setattr(workflow, "get_settings", lambda: type("S", (), {})())
    def fake_run_chat(_message, **kwargs):
        calls.append(kwargs["_critic_llm_review_mode"])
        (executor.trace_dir / f"{kwargs['trace_id']}.json").write_text(
            json.dumps({"trace_id": kwargs["trace_id"], "state_snapshot": {"final_answer": "offline"}}), encoding="utf-8")
    monkeypatch.setattr(workflow, "run_chat_workflow", fake_run_chat)
    monkeypatch.setattr("app.llm.call_config.LLMCallConfig", type("C", (), {"provider_ab_experiment": staticmethod(lambda *_a, **_k: object())}))
    monkeypatch.setattr(executor.cost_ledger, "record_usage", lambda *_a, **_k: {})
    tenant = TenantContext(tenant_id="platform", user_id="offline", role="admin", scope_type="platform")
    for arm in ("B", "G"):
        executor._run_one({"run_id": f"adaptive-{arm}", "arm": arm, "case_id": case["case_id"], "replicate": 1}, case,
                          tenant_context=tenant, observation_builder=lambda *_a: {}, fault_context_factory=None)
    assert calls == [CriticLLMReviewMode.ADAPTIVE, CriticLLMReviewMode.FULL]
