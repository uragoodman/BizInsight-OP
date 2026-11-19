"""Accounting and stop-boundary checks for the non-provider dev-pair runner."""

from __future__ import annotations

import json

import pytest

from evaluation.provider_candidate_batch import RunLedger
from scripts.run_resume_dev_pair import (
    _append_usage_summary,
    _finalize_run_record,
    _stop_reason_after_run,
)


def test_usage_summary_records_one_explicit_run_id_without_duplicate_keyword(tmp_path):
    run_id = "run-usage-1"
    ledger = RunLedger(tmp_path / "events.jsonl")
    usage = {
        "run_id": run_id,
        "sdk_sends": 2,
        "known_input_token_subtotal": 12,
        "known_output_token_subtotal": 5,
        "total_tokens": 17,
        "status": "complete",
    }

    row = _append_usage_summary(ledger, run_id, usage)

    assert row["run_id"] == run_id
    assert row["event"] == "usage_summary"
    assert len([event for event in ledger.events if event["event"] == "usage_summary"]) == 1
    assert json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))["run_id"] == run_id


@pytest.mark.parametrize("observed", [None, "other-run"])
def test_usage_summary_rejects_missing_or_conflicting_run_id_without_appending(tmp_path, observed):
    run_id = "run-usage-1"
    ledger = RunLedger(tmp_path / "events.jsonl")
    usage = {"run_id": observed, "sdk_sends": 0, "status": "no_requests"}

    with pytest.raises(ValueError, match="usage_summary_run_id_mismatch"):
        _append_usage_summary(ledger, run_id, usage)

    assert ledger.events == []
    assert not (tmp_path / "events.jsonl").exists()


def test_business_tool_rejection_does_not_suppress_the_paired_arm():
    assert _stop_reason_after_run(
        {"errors": [{"code": "tool_execution_error", "exception_type": "ValueError"}]},
        {"status": "complete", "cost_record_matches": True},
        sdk_failed_sends=0,
    ) is None


@pytest.mark.parametrize(
    "failure",
    [
        {"code": "tool_execution_error", "service": "mcp"},
        {"code": "tool_execution_error", "cause_type": "OSError"},
        {"code": "tool_execution_error", "service_error_code": "timeout"},
    ],
)
def test_tool_infrastructure_failure_stops_before_the_paired_arm(failure):
    assert _stop_reason_after_run(
        {"errors": [failure]},
        {"status": "complete", "cost_record_matches": True},
        sdk_failed_sends=0,
    ) == "tool_infrastructure_error"


def test_no_tool_failure_does_not_stop_the_pair():
    assert _stop_reason_after_run(
        {"errors": []},
        {"status": "complete", "cost_record_matches": True},
        sdk_failed_sends=0,
    ) is None


def test_sent_but_unknown_usage_stops_before_the_paired_arm():
    assert _stop_reason_after_run(
        {"errors": []},
        {"status": "sent_usage_unknown", "cost_record_matches": True},
        sdk_failed_sends=0,
    ) == "usage_unknown"


@pytest.mark.parametrize(
    ("stop_reason", "expected_status", "expected_event"),
    [(None, "completed", "completed"), ("tool_infrastructure_error", "failed", "failed")],
)
def test_terminal_run_record_matches_append_only_ledger(
    tmp_path, stop_reason, expected_status, expected_event,
):
    run_id = "run-finalize-1"
    ledger = RunLedger(tmp_path / "events.jsonl")
    path = tmp_path / "trace.run_record.json"
    record = {"run_id": run_id, "case_id": "order_dimension"}
    usage = {"run_id": run_id, "sdk_sends": 0, "status": "no_requests"}

    result = _finalize_run_record(
        ledger=ledger, run_id=run_id, record_path=path, record=record,
        usage=usage, task_pass=False, provider_sends=0, stop_reason=stop_reason,
    )

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert result["lifecycle_status"] == expected_status
    assert saved["lifecycle_status"] == expected_status
    assert ledger.latest()[run_id]["event"] == expected_event
    assert [row["event"] for row in ledger.events] == ["usage_summary", expected_event]
