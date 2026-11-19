"""Direct contract tests for the B/G schedule and execution entrypoint."""
from __future__ import annotations

import json
import socket
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.provider_candidate_batch import (
    CostPolicy, EstimatedCostLedger, ExperimentConfig, RunLedger,
    bind_isolated_experiment_runtime,
)
from evaluation.resume_bg_executor import (
    validate_bg_schedule, validate_execute_budget, run_bg_unit, run_bg_batch,
)
from evaluation.resume_bg_protocol import build_manifest, score_bg_run_v5
from test_platform_metric_trend_acceptance import isolated_trend_db  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs/testing/resume_bg_ab_dev_20261016"


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch):
    def denied(sock, address):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise AssertionError("External network is blocked in B/G executor tests.")
        return None
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", lambda address, *a, **k: denied(None, address))


def _inputs():
    case_doc = json.loads((ARTIFACTS / "cases_6.json").read_text(encoding="utf-8"))
    manifest = json.loads((ARTIFACTS / "run_manifest_12.json").read_text(encoding="utf-8"))
    plan = json.loads((ARTIFACTS / "execution_plan.json").read_text(encoding="utf-8"))
    return case_doc["cases"], manifest, plan


def test_frozen_schedule_has_exact_six_cases_two_arms_unique_ids_and_shared_estimate():
    cases, manifest, plan = _inputs()
    result = validate_bg_schedule(cases, manifest, execution_plan=plan)
    assert result["planned_units"] == 12
    assert result["arm_counts"] == {"basic_tool_loop": 6, "governed_workflow": 6}
    assert result["unique_run_ids"] == 12
    assert result["estimated_cny"] == pytest.approx(27.94455144)
    assert result["estimate_is_account_hard_cap"] is False
    assert all(entry["status"] == "not_started" for entry in manifest["entries"])


def test_schedule_arm_mutation_is_rejected_before_execution():
    cases, manifest, plan = _inputs()
    mutated = json.loads(json.dumps(manifest))
    mutated["entries"][0]["arm"] = "full"
    with pytest.raises(ValueError, match="deterministic seed"):
        validate_bg_schedule(cases, mutated, execution_plan=plan)


def test_selected_units_use_existing_reservation_and_require_matching_authorization(tmp_path):
    policy = CostPolicy(cumulative_estimate_limit_cny=30.0)
    ledger = EstimatedCostLedger(policy, tmp_path / "reserve.jsonl")
    ledger.reserve("already-reserved")
    result = validate_execute_budget(
        selected_count=2, policy=policy, cost_ledger=ledger,
        estimate_limit_cny=30.0, provider_authorization_cny=30.0,
    )
    assert result["existing_reservations_cny"] == pytest.approx(policy.max_per_run_cny())
    assert result["new_estimate_reservation_cny"] == pytest.approx(policy.max_per_run_cny() * 2)
    with pytest.raises(ValueError, match="must match"):
        validate_execute_budget(
            selected_count=1, policy=policy, cost_ledger=ledger,
            estimate_limit_cny=30.0, provider_authorization_cny=5.0,
        )
    with pytest.raises(Exception, match="insufficient"):
        validate_execute_budget(
            selected_count=1, policy=policy, cost_ledger=ledger,
            estimate_limit_cny=2.5, provider_authorization_cny=2.5,
        )


def test_b_arm_runner_uses_real_tool_path_writes_trace_score_and_usage(
    monkeypatch, tmp_path, isolated_trend_db,
):
    import app.core.config as core_config
    from app.llm import client as llm
    from app.tenancy import TenantContext
    from test_non_gmv_diagnostic_workflow import seed_facts

    engine, sql_statements, _sql_events, _case_results = isolated_trend_db
    # This six-case development case asks about Jan 5/6 2017. Seed the same
    # fixture shape on those dates instead of returning an unrelated 2024
    # not_found result and mistaking the intentional early stop for a runner bug.
    seed_facts(engine, "order_dimension", start_date=date(2017, 1, 1))
    cases, manifest, _plan = _inputs()
    case = next(row for row in cases if row["case_id"] == "NC01")
    entry = next(row for row in manifest["entries"] if row["case_id"] == "NC01" and row["arm"] == "basic_tool_loop")
    actions = [
        {"action": "TOOL_CALL", "tool": "registered_numeric_decomposition", "arguments": {
            "metric_id": "gmv", "baseline_date": "2017-01-05", "current_date": "2017-01-06",
        }},
        {"action": "FINAL", "answer": "模型原答故意写成相反方向：GMV下降。", "claims": [],
         "presentation": {"metric_id": "gmv", "delta": -999, "unit": "wrong"}},
    ]

    class FakeCompletions:
        def __init__(self):
            self.calls = []
        def create(self, **kwargs):
            self.calls.append(kwargs)
            content = json.dumps(actions.pop(0), ensure_ascii=False)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=23, completion_tokens=11),
            )

    fake = FakeCompletions()
    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=fake)
        def close(self):
            return None

    safe_settings = core_config.get_settings().model_copy(update={
        "deepseek_api_key": "offline-placeholder", "llm_providers_json": "[]",
        "llm_timeout_seconds": 60.0,
    })
    monkeypatch.setattr(core_config, "get_settings", lambda: safe_settings)
    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: FakeClient(**kwargs))
    ledger = RunLedger(tmp_path / "events.jsonl")
    cost = EstimatedCostLedger(CostPolicy(cumulative_estimate_limit_cny=30.0), tmp_path / "cost.jsonl")
    trace_dir = tmp_path / "traces"
    context = TenantContext(tenant_id="platform", user_id="bg-preflight", role="admin", scope_type="platform")

    with bind_isolated_experiment_runtime(
        engine=engine, trace_dir=trace_dir, ledger=ledger,
        config=ExperimentConfig(max_latency_ms=240_000),
    ):
        record = run_bg_unit(
            entry, case, tenant_context=context, trace_dir=trace_dir,
            ledger=ledger, cost_ledger=cost,
            experiment_config=ExperimentConfig(max_latency_ms=240_000),
            scorer_fn=score_bg_run_v5,
        )

    assert len(fake.calls) == 2
    assert sql_statements
    assert record["trace"]["state_snapshot"]["trace_id"] == entry["run_id"]
    assert record["score"]["scorer_version"] == "resume-bg-score-v5"
    assert record["measurements"]["sdk_sends"] == 2
    assert record["measurements"]["token_usage"]["total_tokens"] == 68
    assert record["measurements"]["token_usage"]["status"] == "complete"
    assert record["companion_quality_audit"]["schema_version"] == "resume-bg-companion-audit-v4"
    assert record["companion_quality_audit"]["run_identity"]["match"] is True
    assert record["companion_quality_audit_error"] is None
    assert ledger.latest()[entry["run_id"]]["event"] == "completed"
    assert (trace_dir / f"{entry['run_id']}.run_record.json").is_file()
    raw = record["trace"]["state_snapshot"]["model_authored_final"]
    adapted = record["trace"]["state_snapshot"]["final_display"]
    assert raw["answer"] == "模型原答故意写成相反方向：GMV下降。"
    assert raw["presentation"]["delta"] == -999
    assert adapted["presentation"]["metric_decomposition"] == record["trace"]["state_snapshot"]["registered_decomposition_cards"][-1]
    assert adapted["presentation"]["delta"] != -999
    assert record["response"]["model_authored_output"] == raw
    assert record["response"]["final_output_adaptation"]["applied"] is True
    assert record["model_authored_score"] is not None
    saved = json.loads((trace_dir / f"{entry['run_id']}.run_record.json").read_text(encoding="utf-8"))
    assert saved["companion_quality_audit"] == record["companion_quality_audit"]
    assert saved["response"]["model_authored_output"]["answer"] == raw["answer"]
    assert saved["response"]["answer"] == adapted["answer"]


def test_b_output_snapshot_and_adapted_display_are_both_persisted_without_database(
    monkeypatch, tmp_path,
):
    """Exercise the real executor and bare loop while stubbing only tool data/model output."""
    from uuid import uuid4
    from app.tenancy import TenantContext
    import evaluation.resume_bare_tool_loop as bare
    import evaluation.resume_bg_executor as executor
    from evaluation.provider_candidate_batch import RunLedger

    cases, manifest, _plan = _inputs()
    case = next(row for row in cases if row["case_id"] == "NC01")
    entry = next(row for row in manifest["entries"]
                 if row["case_id"] == "NC01" and row["arm"] == "basic_tool_loop")
    entry = {**entry, "run_id": f"offline-b-output-snapshot-{uuid4().hex}"}
    raw_final = {
        "action": "FINAL",
        "answer": "模型原答写了相反方向：GMV下降。",
        "claims": [],
        "presentation": {"metric_id": "gmv", "delta": -999, "unit": "wrong"},
    }
    actions = [
        {"action": "TOOL_CALL", "tool": "registered_numeric_decomposition", "arguments": {
            "metric_id": "gmv", "baseline_date": "2017-01-05", "current_date": "2017-01-06",
        }},
        raw_final,
    ]
    card = {
        "metric_id": "gmv", "display_name": "GMV", "unit": "BRL",
        "baseline_value": 707.27, "current_value": 988.28, "delta_value": 281.01,
        "baseline_date": "2017-01-05", "current_date": "2017-01-06",
        "method_id": "product_shapley_v1", "factors": [
            {"metric_id": "order_count", "display_name": "订单量", "contribution": -3768.41, "unit": "BRL", "role_id": "order_count"},
        ], "evidence_ids": ["source-evidence"],
        "analysis_nature": "registered_numeric_decomposition", "business_cause_confirmed": False,
    }
    monkeypatch.setattr(bare, "_registered_decomposition_card", lambda _response: card)
    real_loop = bare.run_bare_tool_loop

    def injected_offline_loop(question, **kwargs):
        kwargs["model_fn"] = lambda _system, _user: actions.pop(0)
        kwargs["tool_fn"] = lambda **_kwargs: {
            "tool_name": "registered_numeric_decomposition",
            "request_id": f"{entry['run_id']}:tool:1",
            "result": {"status": "ok", "operation_intent": "registered_numeric_decomposition"},
        }
        return real_loop(question, **kwargs)

    monkeypatch.setattr(bare, "run_bare_tool_loop", injected_offline_loop)
    trace_dir = tmp_path / "trace"
    ledger = RunLedger(tmp_path / "events.jsonl")
    class NoRequestCostLedger:
        policy = SimpleNamespace(input_cny_per_million=0.0, output_cny_per_million=0.0)
        reservations = []
        reserved_cny = 1.0

        def reserve(self, run_id):
            self.reservations.append({"run_id": run_id, "reserved_cny": 1.0})
            return 1.0

        def record_usage(self, run_id, usage):
            return {"run_id": run_id, "usage_status": usage["status"], "reserved_cny_retained": 1.0,
                    "actual_estimated_cny": 0.0, "reservation_exceeded": False}

    monkeypatch.setattr(executor, "token_usage_snapshot", lambda *_args, **_kwargs: {
        "run_id": entry["run_id"], "status": "no_requests", "metering_errors": [],
        "unknown_or_failed_usage_requests": [], "provider_failed_sends": [],
        "known_input_token_subtotal": 0, "known_output_token_subtotal": 0,
        "total_input_tokens": 0, "total_output_tokens": 0, "total_tokens": 0,
    })
    monkeypatch.setattr(executor, "_reconcile_cost_accounting", lambda *_args, **_kwargs: None)
    context = TenantContext(tenant_id="platform", user_id="output-snapshot-test", role="admin", scope_type="platform")
    record = run_bg_unit(
        entry, case, tenant_context=context, trace_dir=trace_dir,
        ledger=ledger, cost_ledger=NoRequestCostLedger(),
        experiment_config=ExperimentConfig(max_latency_ms=240_000),
        scorer_fn=score_bg_run_v5,
    )

    saved = json.loads((trace_dir / f"{entry['run_id']}.run_record.json").read_text(encoding="utf-8"))
    persisted_state = saved["trace"]["state_snapshot"]
    assert persisted_state["model_authored_final"] == raw_final
    assert persisted_state["model_authored_final"]["answer"] == "模型原答写了相反方向：GMV下降。"
    assert persisted_state["final_display"] == persisted_state["final"]
    assert persisted_state["final_output_adaptation"]["applied"] is True
    assert persisted_state["final_output_adaptation"]["operations"] == [
        "registered_decomposition_card_projection", "registered_factor_label_text_supplement",
    ]
    assert saved["response"]["model_authored_output"] == raw_final
    assert saved["response"]["answer"] == persisted_state["final_display"]["answer"]
    assert saved["model_authored_score"] is not None
    assert record["score"] is not saved["model_authored_score"]
    assert saved["companion_quality_audit"]["schema_version"] == "resume-bg-companion-audit-v4"
    assert saved["companion_quality_audit"]["run_identity"]["match"] is True
    assert saved["companion_quality_audit_error"] is None
    assert saved["response"]["model_authored_output"]["presentation"]["delta"] == -999
    assert saved["response"]["presentation"]["metric_decomposition"]["delta_value"] == 281.01


def test_g_arm_runner_uses_normal_workflow_and_three_optional_reviews_reach_sdk_stub(
    monkeypatch, tmp_path, isolated_trend_db,
):
    """Exercise the runner's governed arm through the normal graph, offline."""
    import copy
    from uuid import uuid4
    import app.core.config as core_config
    from app.agents import critic_agent
    from app.graph import workflow
    from app.llm import client as llm
    from app.tenancy import TenantContext
    from test_non_gmv_diagnostic_workflow import seed_facts
    from test_registered_numeric_decomposition_intent import _install_offline_workflow

    engine, _sql_statements, _sql_events, _case_results = isolated_trend_db
    seed_facts(engine, "aov_ratio")
    real_llm_extract = llm.llm_extract
    # Install the normal workflow's isolated DB/tool boundary and offline
    # non-Critic fallbacks; the optional Critic calls below use llm_extract and
    # the real public SDK-send wrapper around a fake SDK client.
    _invoke, captured, _statements, _model_calls, external_attempts = _install_offline_workflow(
        monkeypatch, isolated_trend_db, "aov_ratio",
    )
    monkeypatch.setattr(critic_agent, "llm_extract", real_llm_extract)

    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content='{"verdict":"passed","issues":[]}'),
                    finish_reason="stop",
                )],
                usage=SimpleNamespace(prompt_tokens=23, completion_tokens=11),
            )

    fake = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = SimpleNamespace(completions=fake)

        def close(self):
            return None

    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: FakeOpenAI(**kwargs))
    safe_settings = core_config.get_settings().model_copy(update={
        "deepseek_api_key": "offline-placeholder",
        "deepseek_base_url": "https://offline-critic.invalid/v1",
        "llm_providers_json": "[]",
        "llm_timeout_seconds": 60.0,
    })
    monkeypatch.setattr(core_config, "get_settings", lambda: safe_settings)

    cases, manifest, _plan = _inputs()
    case = copy.deepcopy(next(row for row in cases if row["case_id"] == "NC03"))
    case["user_question"] = (
        "请对平台客单价按注册数值分解方法说明2024-05-03与2024-05-04的变化，"
        "只作数值解释，不推断业务因果。"
    )
    entry = {
        "run_id": f"resume-bg-g-sdk-preflight-{uuid4().hex}",
        "case_id": case["case_id"],
        "arm": "governed_workflow",
        "replicate": 1,
    }
    trace_dir = tmp_path / "g_traces"
    ledger = RunLedger(tmp_path / "g_events.jsonl")
    cost = EstimatedCostLedger(CostPolicy(cumulative_estimate_limit_cny=30.0), tmp_path / "g_cost.jsonl")
    context = TenantContext(tenant_id="platform", user_id="bg-preflight", role="admin", scope_type="platform")

    def persist_trace(trace_id, state):
        captured.clear()
        captured.update(copy.deepcopy(state))
        (trace_dir / f"{trace_id}.json").write_text(
            json.dumps({"trace_id": trace_id, "state_snapshot": state}, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    monkeypatch.setattr(workflow, "write_trace", persist_trace)
    config = ExperimentConfig(max_latency_ms=240_000)
    with bind_isolated_experiment_runtime(
        engine=engine, trace_dir=trace_dir, ledger=ledger, config=config,
    ):
        record = run_bg_unit(
            entry, case, tenant_context=context, trace_dir=trace_dir,
            ledger=ledger, cost_ledger=cost, experiment_config=config, scorer_fn=score_bg_run_v5,
        )

    assert external_attempts == []
    assert len(fake.calls) == 3
    assert all(request.get("max_completion_tokens") == 4096 for request in fake.calls)
    assert all(request.get("extra_body") == {"enable_thinking": False} for request in fake.calls)
    assert [call["node"] for call in captured["llm_calls"]] == ["critic", "critic", "critic"]
    assert [review["verdict"] for review in (
        captured["critic_result"]["llm_review"]["groundedness"],
        captured["critic_result"]["llm_review"]["completeness"],
        captured["critic_result"]["llm_review"]["consistency"],
    )] == ["passed", "passed", "passed"]
    assert record["measurements"]["sdk_sends"] == 3
    assert record["measurements"]["token_usage"]["total_tokens"] == 102
    assert record["score"]["critic_review_complete"] is True
    assert record["score"]["scorer_version"] == "resume-bg-score-v5"
    assert (trace_dir / f"{entry['run_id']}.run_record.json").is_file()


def test_historical_li03_missing_structured_snapshot_is_rejected_before_reservation_or_send(
    monkeypatch, tmp_path,
):
    """The blocked historical case remains blocked; missing proof is not inferred from its prose."""
    from types import SimpleNamespace
    from evaluation.provider_candidate_batch import CostPolicy, EstimatedCostLedger, RunLedger

    cases, manifest, original_plan = _inputs()
    historical = next(row for row in cases if row["case_id"] == "LI03")
    assert historical["readiness"] == "blocked"
    assert "limitation_contract" not in historical
    plan = {**original_plan, "execution_blocked": False, "blocked_case_ids": [],
            "scorer_version": "resume-bg-score-v3"}
    ledger = RunLedger(tmp_path / "li03_blocked_events.jsonl")
    for prior in manifest["entries"][:2]:
        ledger.append(prior["run_id"], "completed", test_only_preexisting=True)
    cost_path = tmp_path / "li03_blocked_cost.jsonl"
    cost_ledger = EstimatedCostLedger(CostPolicy(cumulative_estimate_limit_cny=30.0), cost_path)
    loader_calls, unit_calls, disposed = [], [], []
    dummy_engine = SimpleNamespace(dispose=lambda: disposed.append(True))

    def test_database_boundary(*_args, **_kwargs):
        loader_calls.append("isolated_test_boundary")
        return dummy_engine, "test_only_database", {"fixture": "no_connection"}

    monkeypatch.setattr("evaluation.resume_bg_executor.load_isolated_candidate_database", test_database_boundary)
    monkeypatch.setattr("evaluation.resume_bg_executor.drop_isolated_candidate_database", lambda *_a, **_k: None)
    monkeypatch.setattr("evaluation.resume_bg_executor.run_bg_unit", lambda *a, **k: unit_calls.append((a, k)))

    with pytest.raises(RuntimeError, match="condition was invalid or unverified") as exc:
        run_bg_batch(
            root=ROOT, cases=cases, manifest=manifest, execution_plan=plan,
            max_units=2, estimate_limit_cny=30.0, provider_authorization_cny=30.0,
            admin_url_text="test-only-no-connection", ledger_path=ledger.path,
            cost_ledger_path=cost_path, trace_dir=tmp_path / "li03_blocked_traces",
            expected_csv_hashes={},
        )

    assert "structured_limitation_contract_or_capability_snapshot_missing" in str(exc.value)
    assert loader_calls == ["isolated_test_boundary"]
    assert disposed == [True]
    assert unit_calls == []
    assert cost_ledger.reservations == []
    latest = RunLedger(ledger.path).latest()
    assert set(latest) == {entry["run_id"] for entry in manifest["entries"][:2]}
    assert all(latest[entry["run_id"]]["event"] == "completed" for entry in manifest["entries"][:2])


def test_test_only_qualified_capability_case_business_miss_is_saved_and_next_unit_runs(
    monkeypatch, tmp_path, isolated_trend_db,
):
    """A test-only qualification probe proves the generic batch continuation policy, not LI03 readiness."""
    import copy
    import app.core.config as core_config
    import app.graph.workflow as workflow_module
    from app.llm import client as llm
    from app.tenancy import TenantContext
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop as real_bare_loop
    from test_non_gmv_diagnostic_workflow import seed_facts
    from test_registered_numeric_decomposition_intent import _install_offline_workflow

    engine, sql_statements, _sql_events, _case_results = isolated_trend_db
    seed_facts(engine, "order_dimension")
    real_llm_extract = llm.llm_extract
    _invoke, captured, _statements, _model_calls, external_attempts = _install_offline_workflow(
        monkeypatch, isolated_trend_db, "order_dimension",
    )
    monkeypatch.setattr(llm, "llm_extract", real_llm_extract)

    class FakeCompletions:
        def __init__(self):
            self.calls = []
            self.responses = [
                {"action": "TOOL_CALL", "tool": "period_metric_contribution", "arguments": {
                    "metric_id": "order_count", "dimension": "payment_type",
                    "baseline_start": "2017-05-01", "baseline_end": "2017-05-31",
                    "current_start": "2017-06-01", "current_end": "2017-06-30",
                    "granularity": "month",
                }},
                {"action": "FINAL", "answer": "请求的维度能力受限。", "claims": []},
            ]

        def create(self, **kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            content = json.dumps(self.responses.pop(0), ensure_ascii=False)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=content), finish_reason="stop",
                )],
                usage=SimpleNamespace(prompt_tokens=19, completion_tokens=9),
            )

    fake = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = SimpleNamespace(completions=fake)

        def close(self):
            return None

    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: FakeOpenAI(**kwargs))
    safe_settings = core_config.get_settings().model_copy(update={
        "deepseek_api_key": "offline-placeholder",
        "deepseek_base_url": "https://offline-sdk.invalid/v1",
        "llm_providers_json": "[]",
        "llm_timeout_seconds": 60.0,
    })
    monkeypatch.setattr(core_config, "get_settings", lambda: safe_settings)
    monkeypatch.setattr(
        workflow_module, "write_trace",
        lambda run_id, state: (captured.clear(), captured.update(copy.deepcopy(state)),
                               (tmp_path / "traces" / f"{run_id}.json").write_text(
                                   json.dumps({"trace_id": run_id, "state_snapshot": state},
                                              ensure_ascii=False, default=str), encoding="utf-8")),
    )

    cases, source_manifest, original_plan = _inputs()
    historical_li03 = next(row for row in cases if row["case_id"] == "LI03")
    test_case = copy.deepcopy(historical_li03)
    test_case["case_id"] = "TEST_ONLY_QUALIFIED_CAPABILITY_LIMIT"
    test_case["readiness"] = "test_fixture_qualified"
    test_case["limitation_contract"] = {
        "metric_id": "order_count", "scope": "platform",
        "operation": "metric_contribution", "dimension": "payment_type",
        "qualification_source": "test-only controlled capability/auth boundary",
    }
    # Deliberately expect the wrong terminal code in this new test-only fixture
    # so a correctly executed refusal is recorded as a scored business miss.
    # The historical LI03 expected result is not edited or reused as a success.
    test_case["expected_terminal_code"] = "INVALID_DATA"
    cases = [test_case if row["case_id"] == "LI03" else row for row in cases]
    from evaluation.resume_bg_protocol import build_manifest
    manifest = build_manifest(cases, seed=int(source_manifest["seed"]))
    assert historical_li03["readiness"] == "blocked"
    assert test_case["case_id"].startswith("TEST_ONLY_")
    plan = {**original_plan, "execution_blocked": False, "blocked_case_ids": [],
            "scorer_version": "resume-bg-score-v3"}
    ledger = RunLedger(tmp_path / "batch_events.jsonl")
    target_rows = [row for row in manifest["entries"] if row["case_id"] == test_case["case_id"]]
    first_target_index = manifest["entries"].index(target_rows[0])
    # Keep the frozen scheduling algorithm and select only this new test case's
    # pair by marking preceding synthetic units as already complete.
    for prior in manifest["entries"][:first_target_index]:
        ledger.append(prior["run_id"], "completed", test_only_preexisting=True)
    cost_path = tmp_path / "batch_cost.jsonl"
    trace_dir = tmp_path / "traces"
    cost_policy = CostPolicy(cumulative_estimate_limit_cny=30.0)
    cost_ledger = EstimatedCostLedger(cost_policy, cost_path)
    context = TenantContext(tenant_id="platform", user_id="bg-preflight", role="admin", scope_type="platform")
    loaded = []

    def load_test_database(_admin_url, *, expected_csv_hashes, dataset_dir):
        loaded.append((expected_csv_hashes, dataset_dir))
        return engine, str(engine.url.database), {"isolated_fixture": True}

    # A test-controlled registry boundary and authorization policy generate
    # observations independently of expected_terminal_code / expected text.
    qualification_observations = []
    def condition_probe(case, *, tenant_context, engine):
        contract = case["limitation_contract"]
        assert case["case_id"] == "TEST_ONLY_QUALIFIED_CAPABILITY_LIMIT"
        assert tenant_context.tenant_id == "platform" and tenant_context.role == "admin"
        request = {key: contract[key] for key in ("metric_id", "scope", "operation", "dimension")}
        registry_snapshot = {**request, "supported": False, "observation_source": "controlled_test_registry"}
        authorization_snapshot = {**request, "authorized": True, "observation_source": "controlled_test_authorization_policy"}
        qualification_observations.append({
            "case_id": case["case_id"], "capability": copy.deepcopy(registry_snapshot),
            "authorization": copy.deepcopy(authorization_snapshot),
        })
        return {"capability_snapshot": registry_snapshot, "authorization_snapshot": authorization_snapshot}

    monkeypatch.setattr("evaluation.resume_bg_executor.load_isolated_candidate_database", load_test_database)
    monkeypatch.setattr("evaluation.resume_bg_executor.drop_isolated_candidate_database", lambda *args, **kwargs: None)

    setup_sql_count = len(sql_statements)
    with bind_isolated_experiment_runtime(
        engine=engine, trace_dir=trace_dir, ledger=ledger,
        config=ExperimentConfig(max_latency_ms=240_000),
    ):
        records = run_bg_batch(
            root=ROOT, cases=cases, manifest=manifest, execution_plan=plan,
            max_units=2, estimate_limit_cny=30.0, provider_authorization_cny=30.0,
            admin_url_text="postgresql+psycopg://postgres@127.0.0.1:15434/postgres",
            ledger_path=ledger.path, cost_ledger_path=cost_path, trace_dir=trace_dir,
            expected_csv_hashes={}, condition_probe=condition_probe,
        )

    assert external_attempts == []
    assert len(records) == 2
    assert [row["case_id"] for row in records] == [test_case["case_id"]] * 2
    assert {row["arm"] for row in records} == {"governed_workflow", "basic_tool_loop"}
    assert len(qualification_observations) == 2
    assert all(item["capability"]["supported"] is False for item in qualification_observations)
    assert all(item["authorization"]["authorized"] is True for item in qualification_observations)
    assert all(row["score"]["task_acceptance_pass"] is False for row in records)
    assert all(row["result_lifecycle_status"] == "completed" for row in records)
    assert sum(row["measurements"]["sdk_sends"] for row in records) == 1
    assert sum(row["measurements"]["token_usage"]["total_tokens"] for row in records) == 28
    assert len(fake.calls) == 1
    for record in records:
        state = record["trace"]["state_snapshot"]
        presentation = state.get("analysis_presentation") or state.get("presentation") or {}
        assert state["terminal_code"] == "CAPABILITY_GAP"
        assert presentation["status"] == "unsupported"
        assert state["decision_records"]
        assert all(item["sql_executed"] is False for item in state["decision_records"])
    assert external_attempts == []
    assert all((trace_dir / f"{row['run_id']}.run_record.json").is_file() for row in records)
    latest = RunLedger(ledger.path).latest()
    assert all(latest[row["run_id"]]["event"] == "completed" for row in records)
    assert all(row["trace"]["state_snapshot"].get("terminal_code") != "EXECUTION_FAILED" for row in records)
    assert len(loaded) == 1
    # The fixture itself creates and seeds isolated tables before execution;
    # only SQL added after this marker would be a workflow query.
    assert len(sql_statements) == setup_sql_count


def test_batch_continues_after_completed_business_miss_but_stops_on_execution_error(
    monkeypatch, tmp_path,
):
    """Batch policy distinguishes returned scored misses from raised infrastructure failures."""
    from types import SimpleNamespace

    cases, manifest, original_plan = _inputs()
    plan = {**original_plan, "execution_blocked": False, "blocked_case_ids": [],
            "scorer_version": "resume-bg-score-v3"}
    dummy_engine = SimpleNamespace(dispose=lambda: None)
    monkeypatch.setattr(
        "evaluation.resume_bg_executor.load_isolated_candidate_database",
        lambda *_a, **_k: (dummy_engine, "test_only", {}),
    )
    monkeypatch.setattr("evaluation.resume_bg_executor.drop_isolated_candidate_database", lambda *_a, **_k: None)
    from contextlib import nullcontext
    monkeypatch.setattr("evaluation.resume_bg_executor.bind_isolated_experiment_runtime", lambda **_kwargs: nullcontext())
    policy = CostPolicy(cumulative_estimate_limit_cny=30.0)
    context = SimpleNamespace(tenant_id="platform", user_id="test", role="admin", scope_type="platform")
    selected_ids = [row["run_id"] for row in manifest["entries"][:2]]

    business_calls = []
    def business_miss(entry, *_a, **_k):
        business_calls.append(entry["run_id"])
        return {"run_id": entry["run_id"], "result_lifecycle_status": "completed",
                "score": {"task_acceptance_pass": False}}

    monkeypatch.setattr("evaluation.resume_bg_executor.run_bg_unit", business_miss)
    business_ledger = RunLedger(tmp_path / "business.jsonl")
    business_cost = EstimatedCostLedger(policy, tmp_path / "business_cost.jsonl")
    business_records = run_bg_batch(
        root=ROOT, cases=cases, manifest=manifest, execution_plan=plan,
        max_units=2, estimate_limit_cny=30.0, provider_authorization_cny=30.0,
        admin_url_text="test-only-no-connection", ledger_path=business_ledger.path,
        cost_ledger_path=business_cost.path, trace_dir=tmp_path / "business_traces",
        expected_csv_hashes={},
    )
    assert len(business_records) == 2 and business_calls == selected_ids
    assert all(not row["score"]["task_acceptance_pass"] for row in business_records)

    infra_calls = []
    def infrastructure_error(entry, *_a, **_k):
        infra_calls.append(entry["run_id"])
        raise RuntimeError("metering_error")

    monkeypatch.setattr("evaluation.resume_bg_executor.run_bg_unit", infrastructure_error)
    infra_ledger = RunLedger(tmp_path / "infra.jsonl")
    infra_cost = EstimatedCostLedger(policy, tmp_path / "infra_cost.jsonl")
    with pytest.raises(RuntimeError, match="metering_error"):
        run_bg_batch(
            root=ROOT, cases=cases, manifest=manifest, execution_plan=plan,
            max_units=2, estimate_limit_cny=30.0, provider_authorization_cny=30.0,
            admin_url_text="test-only-no-connection", ledger_path=infra_ledger.path,
            cost_ledger_path=infra_cost.path, trace_dir=tmp_path / "infra_traces",
            expected_csv_hashes={},
        )
    assert infra_calls == [selected_ids[0]]
