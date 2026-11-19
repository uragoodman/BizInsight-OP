from __future__ import annotations

import importlib.util
import socket
from contextlib import nullcontext
from pathlib import Path

import pytest

from evaluation import provider_candidate_batch as batch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/testing/registered_numeric_decomposition_critic_ablation_20261015/run_experiment.py"
spec = importlib.util.spec_from_file_location("numeric_pair_stop_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def pair(tmp_path, monkeypatch):
    # Exercise the real batch loop (including its exception swallowing), with
    # only the workflow and DB/SDK runtime boundary replaced. No real sends.
    def no_network(*args, **kwargs):
        raise AssertionError("Network forbidden in pair-stop regression")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(batch, "bind_isolated_experiment_runtime", lambda **kwargs: nullcontext())
    entries = [
        {"run_id": run_id, "case_id": "dev", "arm": arm, "replicate": 1}
        for run_id, arm in zip(runner.REQUIRED_FIRST_PAIR_RUN_IDS, ("critic_llm_review_off", "full"))
    ]
    ledger = batch.RunLedger(tmp_path / "events.jsonl")
    policy = batch.CostPolicy(cumulative_estimate_limit_cny=5)
    cost = batch.EstimatedCostLedger(policy, tmp_path / "cost.jsonl")
    executor = batch.CandidateBatchExecutor(
        root=tmp_path, manifest={"planned_units": 2, "entries": entries},
        candidates=[{"case_id": "dev"}], frozen_hashes={}, ledger=ledger,
        cost_ledger=cost, trace_dir=tmp_path / "traces", expected_units=2,
    )
    monkeypatch.setattr(executor, "verify_frozen_inputs", lambda: None)
    selection = {"selected_run_ids": list(runner.REQUIRED_FIRST_PAIR_RUN_IDS),
                 "estimate_limit_cny": 5, "provider_authorization_cny": 5}
    called = []
    outputs = []

    def unit(entry, *args, **kwargs):
        run_id = entry["run_id"]
        called.append(run_id)
        cost.reserve(run_id)
        ledger.append(run_id, "started")
        output = {
            "trace": {"state_snapshot": {"trace_id": run_id, "errors": []}},
            "score": {"task_acceptance_pass": False},
            "error": [],
            "measurements": {
                "sql_failed": 0,
                "token_usage": {"status": "no_requests", "metering_errors": []},
                "cost_accounting": {"reconciliation_matches": True, "reservation_exceeded": False},
            },
        }
        outputs.append(output)
        ledger.append(run_id, "completed")
        return output

    monkeypatch.setattr(batch, "run_workflow_unit", unit)
    return executor, selection, called, outputs, unit


def execute(pair):
    executor, selection, *_ = pair
    return runner._run_selected_pair(executor, selection=selection, engine=None, tenant_context=None)


def test_business_failure_with_complete_metering_still_runs_full_once(pair):
    outputs, stop = execute(pair)
    assert stop is None
    assert pair[2] == list(runner.REQUIRED_FIRST_PAIR_RUN_IDS)
    assert len(outputs) == 2
    assert all(row["score"]["task_acceptance_pass"] is False for row in outputs)


def test_exception_swallowed_by_real_batch_loop_never_starts_full(pair, monkeypatch):
    executor, _, called, _, _ = pair

    def failed_unit(entry, *args, **kwargs):
        called.append(entry["run_id"])
        executor.ledger.append(entry["run_id"], "started")
        executor.ledger.append(entry["run_id"], "failed", error_type="RuntimeError")
        raise RuntimeError("local infrastructure failure")

    monkeypatch.setattr(batch, "run_workflow_unit", failed_unit)
    outputs, stop = execute(pair)
    assert outputs == []
    assert stop["reason"] == "unit_result_missing"
    assert called == [runner.REQUIRED_FIRST_PAIR_RUN_IDS[0]]
    assert runner.REQUIRED_FIRST_PAIR_RUN_IDS[1] not in executor.ledger.latest()


@pytest.mark.parametrize("fault", ["metering", "cost", "reservation", "sql", "provider", "workflow", "missing"])
def test_integrity_error_retains_off_result_and_never_starts_full(pair, monkeypatch, fault):
    executor, _, called, _, unit = pair

    def faulty_unit(entry, *args, **kwargs):
        output = unit(entry, *args, **kwargs)
        measurements = output["measurements"]
        if fault == "metering":
            measurements["token_usage"].update(status="metering_error", metering_errors=["broken_trace"])
        elif fault == "cost":
            measurements["cost_accounting"]["reconciliation_matches"] = False
        elif fault == "reservation":
            measurements["cost_accounting"]["reservation_exceeded"] = True
        elif fault == "sql":
            measurements["sql_failed"] = 1
        elif fault == "provider":
            executor.ledger.append(entry["run_id"], "sdk_send_failed")
            executor.ledger.append(entry["run_id"], "completed")
        elif fault == "workflow":
            output["trace"]["state_snapshot"]["errors"] = [{"code": "agent_execution_failed"}]
        else:
            del measurements["cost_accounting"]["reconciliation_matches"]
        return output

    monkeypatch.setattr(batch, "run_workflow_unit", faulty_unit)
    outputs, stop = execute(pair)
    assert stop is not None
    assert len(outputs) == 1  # The original failed measurement is retained.
    assert called == [runner.REQUIRED_FIRST_PAIR_RUN_IDS[0]]
    assert runner.REQUIRED_FIRST_PAIR_RUN_IDS[1] not in executor.ledger.latest()


def test_next_run_id_change_is_not_replaced_with_another_unit(pair, monkeypatch):
    executor, _, called, _, unit = pair

    def changed_schedule(entry, *args, **kwargs):
        result = unit(entry, *args, **kwargs)
        executor.manifest["entries"][1] = dict(executor.manifest["entries"][1], run_id="unapproved")
        return result

    monkeypatch.setattr(batch, "run_workflow_unit", changed_schedule)
    outputs, stop = execute(pair)
    assert len(outputs) == 1
    assert stop["reason"] == "next_run_id_mismatch"
    assert called == [runner.REQUIRED_FIRST_PAIR_RUN_IDS[0]]


def test_cumulative_reservation_limit_stops_pair(pair):
    pair[0].cost_ledger.reserve("earlier-unit")
    pair[1]["estimate_limit_cny"] = 4.0
    outputs, stop = execute(pair)
    # Earlier reservations remain charged; no full retry or alternative unit.
    assert stop is not None
    assert pair[2] == [runner.REQUIRED_FIRST_PAIR_RUN_IDS[0]]
