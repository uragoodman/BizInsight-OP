from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from evaluation.provider_candidate_batch import (
    CostPolicy,
    CandidateBatchExecutor,
    ExperimentConfig,
    EstimatedCostLedger,
    RunLedger,
    _BudgetedCompletions,
    build_run_manifest,
    run_workflow_unit,
    token_usage_snapshot,
    verify_hashes,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/testing/provider_candidate_protocol_17_20261013/candidate_protocol_17.json"


@pytest.fixture(autouse=True)
def _block_external_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("external network is blocked in provider candidate batch tests")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def _cases():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]


def test_schedule_is_102_unique_paired_and_counterbalanced():
    first = build_run_manifest(_cases(), seed=314159)
    second = build_run_manifest(_cases(), seed=314159)
    assert first == second
    entries = first["entries"]
    assert len(entries) == 102
    assert len({row["run_id"] for row in entries}) == 102
    assert {row["arm"] for row in entries} == {"full", "critic_llm_review_off"}
    by_case_arm = {}
    for row in entries:
        by_case_arm[(row["case_id"], row["arm"])] = by_case_arm.get((row["case_id"], row["arm"]), 0) + 1
        assert row["status"] == "not_started"
        assert row["score"] is None
    assert len(by_case_arm) == 34
    assert set(by_case_arm.values()) == {3}
    firsts = [entries[index]["arm"] for index in range(0, len(entries), 2)]
    assert abs(firsts.count("full") - firsts.count("critic_llm_review_off")) == 1
    assert all(entries[i]["pair_id"] == entries[i + 1]["pair_id"] for i in range(0, 102, 2))


def test_full_off_configuration_has_only_review_difference():
    from evaluation.provider_candidate_batch import DEFAULT_EXPERIMENT_CONFIG

    full = DEFAULT_EXPERIMENT_CONFIG.summary("full")
    off = DEFAULT_EXPERIMENT_CONFIG.summary("critic_llm_review_off")
    assert {key: value for key, value in full.items() if key not in {"arm", "critic_review_mode", "optional_critic_reviews", "only_arm_difference"}} == {
        key: value for key, value in off.items() if key not in {"arm", "critic_review_mode", "optional_critic_reviews", "only_arm_difference"}
    }
    assert full["optional_critic_reviews"] == ["groundedness", "completeness", "consistency"]
    assert off["optional_critic_reviews"] == []


def test_cost_estimate_matches_previously_accepted_preflight_and_is_not_hard_cap():
    policy = CostPolicy()
    estimate = policy.estimate(102)
    assert estimate["sdk_calls_upper_bound"] == 1020
    assert estimate["input_token_upper_bound"] == 204_000_000
    assert estimate["output_token_upper_bound_including_variance"] == 4_188_120
    assert estimate["per_run_cny"] == pytest.approx(2.32871262)
    assert estimate["estimated_cny"] == pytest.approx(237.52868724)
    assert estimate["within_estimate_limit"] is True
    assert estimate["is_account_hard_cap"] is False
    assert policy.estimate(12)["estimated_cny"] == pytest.approx(27.94455144)


def test_resume_only_reuses_never_started_units_and_never_retries_failures(tmp_path):
    entries = build_run_manifest(_cases(), seed=1)["entries"]
    ledger = RunLedger(tmp_path / "events.jsonl")
    assert ledger.resume_candidates(entries) == entries

    ledger.append(entries[0]["run_id"], "cost_reserved", reserved_cny=2.3)
    assert entries[0] in ledger.resume_candidates(entries)  # reserved but never started

    ledger.append(entries[0]["run_id"], "started")
    ledger.append(entries[0]["run_id"], "sdk_send_started", send_number=1)
    resumed = ledger.resume_candidates(entries)
    assert entries[0] not in resumed
    # Selection is read-only; interrupted recovery must be requested explicitly.
    assert ledger.latest()[entries[0]["run_id"]]["event"] == "started"
    ledger.reconcile_interrupted()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "interrupted_result_unknown"
    assert events[-1]["provider_sends"] == 1

    ledger.append(entries[1]["run_id"], "started")
    ledger.append(entries[1]["run_id"], "failed", retry_forbidden=True)
    assert entries[1] not in ledger.resume_candidates(entries)


def test_unknown_usage_keeps_worst_case_reservation_and_limit_blocks_before_start(tmp_path):
    policy = CostPolicy(cumulative_estimate_limit_cny=5.0)
    cost = EstimatedCostLedger(policy, tmp_path / "cost.jsonl")
    first = cost.reserve("r1")
    assert first == pytest.approx(2.32871262)
    assert cost.reserved_cny == pytest.approx(first)
    # Unknown actual usage does not release the reservation.
    assert cost.reserve("r1") == pytest.approx(first)
    assert cost.reserved_cny == pytest.approx(first)
    assert cost.reserve("r2") == pytest.approx(first)
    with pytest.raises(RuntimeError, match="estimated cost limit"):
        cost.reserve("r3")

    unknown = cost.record_usage("r1", {"status": "incomplete", "known_input_token_subtotal": 12,
                                       "known_output_token_subtotal": 4, "total_input_tokens": None,
                                       "total_output_tokens": None})
    assert unknown["actual_estimated_cny"] is None
    assert unknown["reserved_cny_retained"] == pytest.approx(first)
    assert cost.reserved_cny == pytest.approx(first * 2)


def test_frozen_hash_check_fails_closed(tmp_path):
    target = tmp_path / "frozen.txt"
    target.write_text("original", encoding="utf-8")
    import hashlib

    expected = {"frozen.txt": hashlib.sha256(b"original").hexdigest()}
    verify_hashes(expected, tmp_path)
    target.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_hashes(expected, tmp_path)


def test_executor_dry_run_checks_freeze_and_never_starts_units(tmp_path):
    import hashlib
    from evaluation.provider_candidate_batch import CandidateBatchExecutor

    candidate_file = tmp_path / "candidate.json"
    candidate_file.write_text("frozen", encoding="utf-8")
    digest = hashlib.sha256(b"frozen").hexdigest()
    entries = build_run_manifest(_cases(), seed=20261013)
    executor = CandidateBatchExecutor(
        root=tmp_path, manifest=entries, candidates=_cases(),
        frozen_hashes={"candidate.json": digest}, ledger=RunLedger(tmp_path / "events.jsonl"),
        cost_ledger=EstimatedCostLedger(CostPolicy(), tmp_path / "cost.jsonl"),
        trace_dir=tmp_path / "traces",
    )
    result = executor.dry_run()
    assert result == {
        "status": "dry_run_only", "planned": 102, "started": 0,
        "completed": 0, "failed": 0, "interrupted_before_send": 0,
        "interrupted_result_unknown": 0, "not_started": 102,
        "provider_sends": 0, "database_started": False, "entrypoints_invoked": 0,
    }
    candidate_file.write_text("modified", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        executor.dry_run()


def test_executor_accepts_explicit_dev_manifest_size_without_changing_formal_102_check(tmp_path):
    rows = build_run_manifest(_cases(), seed=27)["entries"][:2]
    mini = {"planned_units": 2, "entries": rows}
    executor = CandidateBatchExecutor(
        root=tmp_path, manifest=mini, candidates=_cases(), frozen_hashes={},
        ledger=RunLedger(tmp_path / "dev-events.jsonl"),
        cost_ledger=EstimatedCostLedger(CostPolicy(cumulative_estimate_limit_cny=5.0), tmp_path / "dev-cost.jsonl"),
        trace_dir=tmp_path / "dev-traces", expected_units=2,
    )
    result = executor.dry_run()
    assert result["planned"] == 2
    assert result["not_started"] == 2
    with pytest.raises(RuntimeError, match="exactly 102 units"):
        CandidateBatchExecutor(
            root=tmp_path, manifest=mini, candidates=_cases(), frozen_hashes={},
            ledger=RunLedger(tmp_path / "formal-events.jsonl"),
            cost_ledger=EstimatedCostLedger(CostPolicy(), tmp_path / "formal-cost.jsonl"),
            trace_dir=tmp_path / "formal-traces", expected_units=102,
        ).dry_run()


def test_executor_sends_only_question_to_normal_entry_and_scores_after_trace(tmp_path):
    candidate = _cases()[0]
    entry = build_run_manifest(_cases(), seed=3)["entries"][0]
    calls = []

    def normal_entry(*args, **kwargs):
        calls.append((args, kwargs))
        trace = {
            "state_snapshot": {
                "trace_id": entry["run_id"], "answer": "offline response", "llm_calls": [], "tool_calls": [],
                "llm_token_summary": {
                    "sdk_attempt_count": 0, "known_input_tokens_subtotal": 0,
                    "known_output_tokens_subtotal": 0, "input_tokens": 0,
                    "output_tokens": 0, "total_tokens": 0,
                    "local_send_refusal_count": 0, "unknown_usage_attempt_count": 0,
                },
            },
            "workflow_result": {"answer": "offline response", "presentation": None},
        }
        (tmp_path / f"{entry['run_id']}.json").write_text(json.dumps(trace), encoding="utf-8")
        return {"answer": "offline response", "trace_id": entry["run_id"]}

    ledger = RunLedger(tmp_path / "events.jsonl")
    cost = EstimatedCostLedger(CostPolicy(), tmp_path / "cost.jsonl")
    result = run_workflow_unit(
        entry, candidate, tenant_context=object(), trace_dir=tmp_path, ledger=ledger,
        cost_ledger=cost, workflow_fn=normal_entry,
    )
    args, kwargs = calls[0]
    assert args == (candidate["user_question"],)
    assert "expected_result" not in kwargs and "truth_basis" not in kwargs and "scoring_expectation" not in kwargs
    assert kwargs["tenant_context"] is not None and kwargs["memory_enabled"] is False
    assert result["score"]["scorer_version"] == "provider-candidate-run-v4"
    assert result["score"]["task_acceptance_pass"] is False
    assert ledger.latest()[entry["run_id"]]["event"] == "completed"


def test_execution_failure_is_recorded_and_never_automatically_retried(tmp_path):
    candidate = _cases()[0]
    entry = build_run_manifest(_cases(), seed=5)["entries"][0]
    attempts = []

    def failing_workflow(*args, **kwargs):
        attempts.append((args, kwargs))
        raise TimeoutError("stubbed offline failure")

    ledger = RunLedger(tmp_path / "events.jsonl")
    cost = EstimatedCostLedger(CostPolicy(), tmp_path / "cost.jsonl")
    kwargs = dict(tenant_context=object(), trace_dir=tmp_path, ledger=ledger,
                  cost_ledger=cost, workflow_fn=failing_workflow)
    with pytest.raises(TimeoutError):
        run_workflow_unit(entry, candidate, **kwargs)
    failure = json.loads((tmp_path / f"{entry['run_id']}.failure_record.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed" and failure["retry_forbidden"] is True
    with pytest.raises(RuntimeError, match="already started or terminated"):
        run_workflow_unit(entry, candidate, **kwargs)
    assert len(attempts) == 1
    assert ledger.latest()[entry["run_id"]]["event"] == "failed"


def test_sdk_send_cap_counts_attempts_and_records_failure_without_retry(monkeypatch, tmp_path):
    import app.llm.client as llm

    run_id = "dev-run-1"
    state = {"trace_id": run_id, "llm_calls": [
        {"call_id": "call-1", "operation": "chat", "node": "triage", "attempts": [{"attempt_id": "attempt-1"}]}
    ]}
    monkeypatch.setattr(llm, "active_state", lambda: state)

    class FakeCompletions:
        def __init__(self):
            self.sent = 0

        def create(self, **kwargs):
            self.sent += 1
            if self.sent == 1:
                raise TimeoutError("offline simulated provider timeout")
            return type("Response", (), {"usage": None})()

    fake = FakeCompletions()
    ledger = RunLedger(tmp_path / "events.jsonl")
    config = ExperimentConfig(max_sdk_sends_per_workflow=2)
    proxy = _BudgetedCompletions(fake, ledger=ledger, config=config, send_counts={})
    request = {"model": config.model, "max_completion_tokens": config.max_completion_tokens,
               "messages": [{"role": "user", "content": "short"}]}
    with pytest.raises(TimeoutError):
        proxy.create(**request)
    response = proxy.create(**request)
    assert response.usage is None
    assert fake.sent == 2
    with pytest.raises(RuntimeError, match="send limit"):
        proxy.create(**request)
    events = [row["event"] for row in ledger.events]
    assert events.count("sdk_send_started") == 2
    assert events.count("sdk_send_failed") == 1
    assert events.count("sdk_response_received") == 1
    assert all(row.get("usage_status") == "unknown" for row in ledger.events if row["event"] in {"sdk_send_failed", "sdk_response_received"})


def _metering_trace(run_id, *, input_tokens=0, output_tokens=0, sdk_attempt_count=0,
                    known_input=None, known_output=None, unknown_usage=0, local_refusals=0,
                    attempt_ids=(), top_level_id=None):
    state = {
        "trace_id": run_id,
        "llm_calls": [{"attempts": [{"attempt_id": attempt_id, "sdk_request_sent": True} for attempt_id in attempt_ids]}],
        "llm_token_summary": {
            "sdk_attempt_count": sdk_attempt_count,
            "local_send_refusal_count": local_refusals,
            "unknown_usage_attempt_count": unknown_usage,
            "known_input_tokens_subtotal": input_tokens if known_input is None else known_input,
            "known_output_tokens_subtotal": output_tokens if known_output is None else known_output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None,
        },
    }
    result = {"state_snapshot": state}
    if top_level_id is not None:
        result["trace_id"] = top_level_id
    return result


def test_usage_accounting_distinguishes_zero_unknown_no_request_and_local_refusal():
    trace = _metering_trace("r", sdk_attempt_count=1, attempt_ids=("a1",))
    zero = token_usage_snapshot(trace, [
        {"run_id": "r", "event": "sdk_send_started", "attempt_id": "a1"},
        {"run_id": "r", "event": "sdk_response_received", "attempt_id": "a1", "input_tokens": 0, "output_tokens": 0, "usage_status": "complete"},
    ], run_id="r")
    assert zero["status"] == "complete" and zero["total_tokens"] == 0
    assert zero["sdk_sends"] == 1 and zero["verified_usage"] is True

    unknown_trace = _metering_trace("r", input_tokens=None, output_tokens=None, sdk_attempt_count=2, known_input=4, known_output=2, unknown_usage=1, attempt_ids=("a1", "a2"))
    unknown = token_usage_snapshot(unknown_trace, [
        {"run_id": "r", "event": "sdk_send_started", "attempt_id": "a1"},
        {"run_id": "r", "event": "sdk_response_received", "attempt_id": "a1", "input_tokens": 4, "output_tokens": 2, "usage_status": "complete"},
        {"run_id": "r", "event": "sdk_send_started", "attempt_id": "a2"},
        {"run_id": "r", "event": "sdk_send_failed", "attempt_id": "a2", "usage_status": "unknown"},
    ], run_id="r")
    assert unknown["status"] == "sent_usage_unknown"
    assert unknown["provider_failed_sends"] == 1 and unknown["request_status"] == "requested"
    assert unknown["known_input_token_subtotal"] == 4 and unknown["known_output_token_subtotal"] == 2
    assert unknown["total_tokens"] is None

    missing_usage_trace = _metering_trace(
        "r", input_tokens=None, output_tokens=None, sdk_attempt_count=1,
        known_input=9, known_output=0, unknown_usage=1, attempt_ids=("a1",),
    )
    missing_usage = token_usage_snapshot(missing_usage_trace, [
        {"run_id": "r", "event": "sdk_send_started", "attempt_id": "a1"},
        {"run_id": "r", "event": "sdk_response_received", "attempt_id": "a1", "input_tokens": 9, "output_tokens": None, "usage_status": "unknown"},
    ], run_id="r")
    assert missing_usage["status"] == "sent_usage_unknown"
    assert missing_usage["known_input_token_subtotal"] == 9
    assert missing_usage["known_output_token_subtotal"] == 0
    assert missing_usage["total_tokens"] is None

    no_request_trace = _metering_trace("r")
    no_request = token_usage_snapshot(no_request_trace, [], run_id="r")
    assert no_request["status"] == "no_requests" and no_request["total_tokens"] == 0
    refused_trace = _metering_trace("r", local_refusals=1)
    refused = token_usage_snapshot(refused_trace, [
        {"run_id": "r", "event": "sdk_send_refused", "attempt_id": "a0", "send_started": False},
    ], run_id="r")
    assert refused["status"] == "local_send_refused" and refused["sdk_sends"] == 0
    assert refused["local_pre_send_refusals"] == 1 and refused["request_status"] == "not_sent"


def test_init_state_zero_request_summary_satisfies_strict_usage_contract():
    """A real initialized no-call state is complete; malformed summaries stay errors."""
    from app.graph.workflow import init_state

    state = init_state({"trace_id": "zero-request-run"})
    summary = state["llm_token_summary"]
    assert summary["local_send_refusal_count"] == 0

    no_request = token_usage_snapshot(
        {"state_snapshot": state}, [], run_id="zero-request-run",
    )
    assert no_request["status"] == "no_requests"
    assert no_request["sdk_sends"] == 0
    assert no_request["local_pre_send_refusals"] == 0
    assert no_request["total_input_tokens"] == 0
    assert no_request["total_output_tokens"] == 0
    assert no_request["total_tokens"] == 0
    assert no_request["metering_errors"] == []

    damaged_state = dict(state)
    damaged_state["llm_token_summary"] = dict(summary)
    damaged_state["llm_token_summary"].pop("local_send_refusal_count")
    damaged = token_usage_snapshot(
        {"state_snapshot": damaged_state}, [], run_id="zero-request-run",
    )
    assert damaged["status"] == "metering_error"
    assert "trace_llm_token_summary_missing_local_send_refusal_count" in damaged["metering_errors"]


def test_usage_requires_explicit_and_consistent_run_id_and_never_crosses_runs():
    a, b = "run-A", "run-B"
    trace = _metering_trace(a, input_tokens=11, output_tokens=7, sdk_attempt_count=1, attempt_ids=("a1",), top_level_id=None)
    events = [
        {"run_id": a, "event": "sdk_send_started", "attempt_id": "a1"},
        {"run_id": a, "event": "sdk_response_received", "attempt_id": "a1", "input_tokens": 11, "output_tokens": 7, "usage_status": "complete"},
        {"run_id": b, "event": "sdk_send_started", "attempt_id": "b1"},
        {"run_id": b, "event": "sdk_response_received", "attempt_id": "b1", "input_tokens": 100, "output_tokens": 200, "usage_status": "complete"},
    ]
    result = token_usage_snapshot(trace, events, run_id=a)
    assert result["status"] == "complete" and result["sdk_sends"] == 1
    assert result["total_tokens"] == 18 and result["unrelated_run_event_count"] == 2
    assert result["trace_run_id"] == a

    conflict = token_usage_snapshot(trace, events, run_id=b)
    assert conflict["status"] == "metering_error" and conflict["total_tokens"] is None
    assert "execution_unit_run_id_does_not_match_state_snapshot_trace_id" in conflict["metering_errors"]
    missing = token_usage_snapshot({"state_snapshot": {}}, events, run_id=a)
    assert missing["status"] == "metering_error" and missing["total_tokens"] is None
    assert "trace_run_id_missing" in missing["metering_errors"]
    orphan_event = token_usage_snapshot(trace, [
        {"event": "sdk_send_started", "attempt_id": "a1"},
    ], run_id=a)
    assert orphan_event["status"] == "metering_error" and orphan_event["total_tokens"] is None
    assert "run_event_run_id_missing" in orphan_event["metering_errors"]
    no_unit = token_usage_snapshot(trace, events, run_id="")
    assert no_unit["status"] == "metering_error" and no_unit["total_tokens"] is None
    dual_conflict = token_usage_snapshot(_metering_trace(a, input_tokens=11, output_tokens=7, sdk_attempt_count=1, top_level_id=b), events, run_id=a)
    assert dual_conflict["status"] == "metering_error" and dual_conflict["total_tokens"] is None
    assert "trace_id_fields_conflict" in dual_conflict["metering_errors"]


def test_saved_v3_smoke_traces_reconcile_nested_ids_events_state_and_cost(tmp_path):
    from evaluation.provider_candidate_batch import _reconcile_cost_accounting

    run_dir = ROOT / "docs/testing/provider_candidate_freeze_20261014_v3/offline_entry_preflight/runs/20261014T030912169Z/workflow_smoke"
    events = [json.loads(line) for line in (run_dir / "smoke_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {"critic_llm_review_off": (6, 102, 78, 180), "full": (9, 153, 117, 270)}
    for arm, (sends, input_tokens, output_tokens, total_tokens) in expected.items():
        trace_path = next((run_dir / "traces" / arm).glob("*.json"))
        run_record_path = next((run_dir / "traces" / arm).glob("*.run_record.json"))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        record = json.loads(run_record_path.read_text(encoding="utf-8"))
        run_id = trace["state_snapshot"]["trace_id"]
        assert "trace_id" not in trace  # The persisted schema puts it under state_snapshot.
        usage = token_usage_snapshot(trace, events, run_id=run_id)
        assert usage["status"] == "complete", usage["metering_errors"]
        assert usage["sdk_sends"] == sends == record["measurements"]["sdk_sends"]
        assert usage["total_input_tokens"] == input_tokens
        assert usage["total_output_tokens"] == output_tokens
        assert usage["total_tokens"] == total_tokens
        state_usage = trace["state_snapshot"]["llm_token_summary"]
        assert usage["sdk_sends"] == state_usage["sdk_attempt_count"]
        assert usage["total_input_tokens"] == state_usage["input_tokens"]
        assert usage["total_output_tokens"] == state_usage["output_tokens"]
        assert usage["total_tokens"] == state_usage["total_tokens"]
        policy = CostPolicy()
        # The historical cost row repeated the faulty no_requests/zero summary;
        # it must be surfaced as inconsistent, not silently trusted.
        historical = record["measurements"]["cost_accounting"]
        historical_check = dict(usage)
        _reconcile_cost_accounting(
            historical_check, historical, policy,
            expected_reservation_cny=float(historical["reserved_cny_retained"]),
        )
        assert historical_check["status"] == "metering_error"
        assert historical_check["cost_record_matches"] is False
        assert "cost_record_usage_status_mismatch" in historical_check["metering_errors"]

        # Rebuild a non-persisted accounting record from the event-derived usage
        # to prove the reconciler accepts matching values without editing history.
        ledger = EstimatedCostLedger(policy, tmp_path / f"nonpersisted-{arm}.jsonl")
        ledger.reserve(run_id)
        current = ledger.record_usage(run_id, usage)
        _reconcile_cost_accounting(
            usage, current, policy,
            expected_reservation_cny=float(current["reserved_cny_retained"]),
        )
        assert usage["cost_record_matches"] is True
        assert usage["status"] == "complete"

