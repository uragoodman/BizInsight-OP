import json

import pytest

from evaluation.claim_verification.run_review_benchmark import _assert_no_unresolved_prior


def test_prior_started_without_result_is_not_automatically_retried(tmp_path):
    (tmp_path / "ledger.jsonl").write_text(
        json.dumps({"run_id": "run-unknown", "event": "started"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="refusing automatic retry.*run-unknown"):
        _assert_no_unresolved_prior(tmp_path)


def test_prior_durable_terminal_result_allows_resume_skip(tmp_path):
    (tmp_path / "ledger.jsonl").write_text(
        json.dumps({"run_id": "run-complete", "event": "started"}) + "\n" +
        json.dumps({"run_id": "run-complete", "event": "completed"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "results.jsonl").write_text(json.dumps({"run_id": "run-complete"}) + "\n", encoding="utf-8")
    _assert_no_unresolved_prior(tmp_path)


def test_prior_failed_without_result_is_still_not_retried(tmp_path):
    (tmp_path / "ledger.jsonl").write_text(
        json.dumps({"run_id": "run-failed", "event": "started"}) + "\n" +
        json.dumps({"run_id": "run-failed", "event": "failed"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="refusing automatic retry.*run-failed"):
        _assert_no_unresolved_prior(tmp_path)
