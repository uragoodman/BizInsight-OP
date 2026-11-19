from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
import sqlite3
from time import monotonic

import pytest

from app.context.request_contract import seal_request_contract
from app.context.shadow_audit import (
    build_shadow_audit_record,
    persist_shadow_audit_record,
    read_shadow_audit_records,
    sampling_bucket,
    should_sample,
    validate_audit_record,
)
from evaluation.dimension_role_shadow_audit import (
    build_report,
    load_records,
    merge_annotations,
    wilson_interval,
)


def _settings(*, enabled=True, rate=100.0, salt="p4-test-salt"):
    return SimpleNamespace(
        semantic_shadow_enabled=enabled,
        semantic_shadow_sample_rate=rate,
        semantic_shadow_salt=salt,
        semantic_verifier_mode="shadow",
        semantic_verifier_model="test/model-v2",
    )


def _state(trace_id="trace-p4-1", *, dimensions=("product_category", "payment_type")):
    roles = {}
    assignments = []
    offset = 0
    for index, dimension in enumerate(dimensions):
        span_id = f"m{index + 1}"
        roles[span_id] = "grouping" if index == 0 else "filter"
        assignments.append({
            "dimension": dimension,
            "role": roles[span_id],
            "start": offset,
            "end": offset + 1,
            "confidence_class": "closed" if index == 0 else "open",
        })
        offset += 2
    contract = seal_request_contract({
        "parse_status": "resolved",
        "operation_intent": "ordinary_metric_query",
        "metric_id": "gmv",
        "dimension": dimensions[0] if dimensions else None,
        "comparison": {},
    })
    compiled = {
        "needs_clarification": False,
        "dimension_roles": assignments,
        "operation_intent": {"kind": "ordinary_metric_query"},
        "known_slots": {
            "metric": {"value": "gmv"},
            "dimension": {"value": dimensions[0] if dimensions else None},
        },
        "request_contract": contract,
        "request_contract_hash": contract["contract_hash"],
    }
    verifier_roles = dict(roles)
    if verifier_roles:
        verifier_roles[next(iter(verifier_roles))] = "reference"
    observation = {
        "status": "success",
        "mode": "shadow",
        "risk_tier": "open",
        "rules_version": "dimension-role-p2",
        "prompt_version": "semantic-verifier-p3-v2",
        "model": "test/model-v2",
        "span_ids": list(roles),
        "rule_roles": roles,
        "verifier_roles": verifier_roles,
        "disagreement_span_ids": [next(iter(roles))] if roles else [],
    }
    request_id = f"query:{trace_id}:1"
    return {
        "trace_id": trace_id,
        "errors": [],
        "chat_turn_id": f"turn:{trace_id}",
        "lifecycle_execution_id": f"execution:{trace_id}",
        "request_contract_hash": contract["contract_hash"],
        "context_compilation": compiled,
        "semantic_verifier_observation": observation,
        "tool_calls": [{
            "tool_name": "query_metric_value",
            "request_id": request_id,
            "planned_metric_id": "gmv",
            "status": "ok",
            "authorization_snapshot": {
                "request_id": request_id,
                "decision": "allowed",
                "reason_code": "authorization_checks_passed",
                "seller_ids": ["sensitive-seller-22"],
                "date_range": "2024-01-01/2024-03-31",
            },
        }],
        "query_results": [{"request_id": request_id, "result": {"status": "ok", "amount": 123}}],
        "user_message": "private question: seller-secret and date 2024-01-01",
        "controlled_analysis_scope": {
            "seller_id": "sensitive-seller-22",
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "sql": "SELECT sensitive_column",
        },
        "node_history": [],
    }


def _record(trace_id="trace-p4-1", *, confidence="closed"):
    state = _state(trace_id, dimensions=("product_category",))
    state["context_compilation"]["dimension_roles"][0]["confidence_class"] = confidence
    return build_shadow_audit_record(
        state,
        settings=_settings(),
        sampled_at=datetime(2026, 11, 7, tzinfo=timezone.utc),
    )


def test_sampling_requires_explicit_eligibility_and_is_stable():
    settings = _settings(rate=45.0)
    assert sampling_bucket("stable-trace-a", settings=settings) == 2380908551657740507
    assert sampling_bucket("stable-trace-b", settings=settings) == 9176421123835055998
    assert should_sample("stable-trace-a", settings=settings) is False
    assert should_sample("stable-trace-a", settings=settings, eligible=False) is False
    assert should_sample("stable-trace-a", settings=settings, eligible=True) == (
        sampling_bucket("stable-trace-a", settings=settings) < (0.45 * (1 << 64))
    )
    assert sampling_bucket("stable-trace-b", settings=settings) == sampling_bucket(
        "stable-trace-b", settings=settings,
    )


@pytest.mark.parametrize(
    "settings",
    [
        _settings(enabled=False, rate=100.0),
        _settings(enabled=True, rate=100.0, salt=""),
        _settings(enabled=True, rate=0.0),
    ],
)
def test_disabled_missing_salt_or_zero_rate_never_samples(settings):
    assert should_sample("trace-p4", settings=settings, eligible=True) is False


def test_only_completed_contract_anchored_query_is_auditable():
    settings = _settings()
    clarification = _state()
    clarification["context_compilation"]["needs_clarification"] = True
    pending = _state("trace-p4-pending")
    pending["query_results"] = []
    bad_contract = _state("trace-p4-bad-contract")
    bad_contract["request_contract_hash"] = "stale"
    failed_query = _state("trace-p4-failed-query")
    failed_query["query_results"][0]["result"]["status"] = "execution_failed"
    failed_tool = _state("trace-p4-failed-tool")
    failed_tool["tool_calls"][0]["status"] = "failed"
    missing_result_for_failed_call = _state("trace-p4-missing-failed-result")
    failed_batch_call = deepcopy(missing_result_for_failed_call["tool_calls"][0])
    failed_batch_call["request_id"] = "query:trace-p4-missing-failed-result:2"
    failed_batch_call["status"] = "failed"
    failed_batch_call["authorization_snapshot"] = {
        "request_id": failed_batch_call["request_id"],
        "decision": "allowed",
    }
    missing_result_for_failed_call["tool_calls"].append(failed_batch_call)
    mismatched_ids = _state("trace-p4-mismatched-ids")
    mismatched_ids["query_results"][0]["request_id"] = "different-request"
    partial_batch = _state("trace-p4-partial-batch")
    partial_batch["query_results"].append({
        "request_id": "unmatched-failed-query",
        "result": {"status": "execution_failed"},
    })
    partial_batch["metric_batch_execution"] = {"status": "execution_failed"}
    denied_snapshot = _state("trace-p4-denied-snapshot")
    denied_snapshot["tool_calls"][0]["authorization_snapshot"]["decision"] = "denied"
    failed_query = _state("trace-p4-failed-query")
    failed_query["query_results"][0]["result"]["status"] = "execution_failed"
    failed_tool = _state("trace-p4-failed-tool")
    failed_tool["tool_calls"][0]["status"] = "failed"
    non_metric_tool = _state("trace-p4-non-metric-tool")
    non_metric_tool["tool_calls"][0]["planned_metric_id"] = None

    assert build_shadow_audit_record(clarification, settings=settings) is None
    assert build_shadow_audit_record(pending, settings=settings) is None
    assert build_shadow_audit_record(bad_contract, settings=settings) is None
    assert build_shadow_audit_record(failed_query, settings=settings) is None
    assert build_shadow_audit_record(failed_tool, settings=settings) is None
    assert build_shadow_audit_record(missing_result_for_failed_call, settings=settings) is None
    assert build_shadow_audit_record(mismatched_ids, settings=settings) is None
    assert build_shadow_audit_record(partial_batch, settings=settings) is None
    assert build_shadow_audit_record(denied_snapshot, settings=settings) is None
    assert build_shadow_audit_record(failed_query, settings=settings) is None
    assert build_shadow_audit_record(failed_tool, settings=settings) is None
    assert build_shadow_audit_record(non_metric_tool, settings=settings) is None


@pytest.mark.parametrize("status", [None, "running", "abstained", "failed", "suspended"])
def test_only_completed_or_implicit_terminal_status_is_auditable(status):
    state = _state(f"trace-p4-status-{status}")
    state["_status"] = status

    assert build_shadow_audit_record(state, settings=_settings()) is None


def test_audit_requires_empty_errors_and_accepts_completed_terminal_state():
    settings = _settings()
    with_error = _state("trace-p4-has-errors")
    with_error["errors"] = ["critic_failed"]
    missing_errors = _state("trace-p4-missing-errors")
    del missing_errors["errors"]
    explicit_completed = _state("trace-p4-explicit-completed")
    explicit_completed["_status"] = "completed"
    implicit_terminal = _state("trace-p4-implicit-terminal")

    assert build_shadow_audit_record(with_error, settings=settings) is None
    assert build_shadow_audit_record(missing_errors, settings=settings) is None
    assert build_shadow_audit_record(explicit_completed, settings=settings) is not None
    # persist_trace is the graph's final node; normal successful graph states
    # have no durable _status until the workflow wrapper returns.
    assert build_shadow_audit_record(implicit_terminal, settings=settings) is not None


def test_record_is_allowlisted_hashed_and_contains_no_raw_question_or_scope():
    state = _state()
    state["tenant_id"] = "sensitive-tenant-22"
    state["user_id"] = "sensitive-user-33"
    state["semantic_verifier_observation"]["raw_model_output"] = "private model quotation"
    before = deepcopy(state)
    record = build_shadow_audit_record(state, settings=_settings())

    assert record is not None
    assert record["actual_execution"] is True
    assert record["trace_id_hash"] != state["trace_id"]
    assert record["turn_id_hash"] != state["chat_turn_id"]
    assert record["execution_id_hash"] != state["lifecycle_execution_id"]
    packed = repr(record)
    for forbidden in (
        "private question", "seller-secret", "sensitive-seller-22",
        "2024-01-01", "SELECT sensitive_column", "seller_ids", "date_range",
    ):
        assert forbidden not in packed
    assert record["human_review_status"] == "pending"
    assert record["reviewed_label"] is None
    assert record["wrong_contract"] is None
    assert record["wrong_execution"] is None
    assert record["authorization_outcome"] == {
        "status": "passed", "reason_code": "authorization_checks_passed",
    }
    assert record["support_outcome"] == {
        "status": "passed", "reason_code": "query_results_all_ok",
    }
    assert record["review_target"] == "rule_role"
    assert state == before


def test_persist_deduplicates_same_trace_and_roundtrips_only_safe_record(tmp_path):
    state = _state()
    state["tenant_id"] = "tenant-pii-marker"
    state["user_id"] = "user-pii-marker"
    state["semantic_verifier_observation"]["raw_model_output"] = "model-output-pii-marker"
    record = build_shadow_audit_record(state, settings=_settings())
    assert record is not None
    path = tmp_path / "semantic_shadow_audit.sqlite3"
    assert persist_shadow_audit_record(record, path=path) is True
    duplicate = dict(record, sampled_at="2026-11-08T00:00:00+00:00")
    assert persist_shadow_audit_record(duplicate, path=path) is False
    records = read_shadow_audit_records(path=path)
    assert len(records) == 1
    assert records[0]["audit_id"] == record["audit_id"]
    persisted_bytes = path.read_bytes()
    for forbidden in (
        b"private question", b"seller-secret", b"sensitive-seller-22",
        b"tenant-pii-marker", b"user-pii-marker", b"model-output-pii-marker",
        b"SELECT sensitive_column", b"date_range", b"seller_ids",
    ):
        assert forbidden not in persisted_bytes


def test_existing_sqlite_without_audit_table_is_not_reported_as_zero_samples(tmp_path):
    path = tmp_path / "wrong-input.sqlite3"
    sqlite3.connect(path).close()

    with pytest.raises(ValueError, match="missing semantic_shadow_audit table"):
        load_records(path)


def test_record_validator_rejects_unbound_span_ids_and_non_registry_dimensions(tmp_path):
    record = _record()
    assert record is not None

    injected_disagreement = deepcopy(record)
    injected_disagreement["disagreement_span_ids"] = ["seller_private_value"]
    injected_role = deepcopy(record)
    injected_role["verifier_roles"]["m999"] = "grouping"
    injected_dimension = deepcopy(record)
    injected_dimension["span_dimensions"]["m1"] = "seller_12345"

    for invalid in (injected_disagreement, injected_role, injected_dimension):
        assert validate_audit_record(invalid) is False
        assert persist_shadow_audit_record(invalid, path=tmp_path / "must-not-store.sqlite3") is False


def test_unreviewed_rates_are_null_not_treated_as_correct():
    record = _record()
    report = build_report([record], min_sample_size=2)
    for key in ("reviewed_fp_rate", "wrong_contract_rate", "wrong_execution_rate"):
        assert report["global"][key]["value"] is None
        assert report["global"][key]["sample_count"] == 0
        assert report["global"][key]["status"] == "sample_insufficient"


def test_disagreement_metrics_count_requests_and_spans_and_emit_strata():
    record = build_shadow_audit_record(
        _state(dimensions=("product_category", "payment_type")),
        settings=_settings(),
    )
    report = build_report([record], min_sample_size=2)
    disagreement = report["global"]["verifier_disagreement_rate"]
    span_disagreement = report["global"]["verifier_disagreement_span_rate"]
    assert disagreement["numerator"] == 1
    assert disagreement["denominator"] == 1
    assert disagreement["value"] == 1.0
    assert span_disagreement["numerator"] == 1
    assert span_disagreement["denominator"] == 2
    assert span_disagreement["value"] == 0.5
    assert report["global"]["sampled_auto_executed_requests"] == 1
    assert report["global"]["verifier_comparable_requests"] == 1
    categories = {
        (row["dimension"], row["operation"], row["risk_tier"], row["rules_version"])
        for row in report["strata"]
    }
    assert ("product_category", "ordinary_metric_query", "open", "dimension-role-p2") in categories
    assert ("payment_type", "ordinary_metric_query", "open", "dimension-role-p2") in categories
    assert all(row["verifier_disagreement_rate"]["status"] == "sample_insufficient" for row in report["strata"])
    assert disagreement["confidence_interval_95"] is not None
    assert span_disagreement["confidence_interval_95"] is not None


def test_wilson_interval_matches_known_values_and_handles_zero_denominator():
    assert wilson_interval(0, 0) is None
    assert wilson_interval(5, 5) == pytest.approx((0.5655175, 1.0), abs=1e-6)
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.2775328), abs=1e-6)
    with pytest.raises(ValueError, match="0 <= successes <= total"):
        wilson_interval(1, 0)


def test_uncompared_requests_are_not_treated_as_agreement():
    state = _state("trace-p4-no-verifier")
    state["semantic_verifier_observation"]["verifier_roles"] = {}
    state["semantic_verifier_observation"]["disagreement_span_ids"] = []
    record = build_shadow_audit_record(state, settings=_settings())
    report = build_report([record], min_sample_size=1)

    metric = report["global"]["verifier_disagreement_rate"]
    assert report["global"]["sampled_auto_executed_requests"] == 1
    assert report["global"]["verifier_noncomparable_requests"] == 1
    assert metric["value"] is None
    assert metric["status"] == "no_verifier_comparisons"


def test_partial_verifier_coverage_reports_comparable_request_denominator():
    compared = _record("trace-p4-compared")
    no_verifier_state = _state("trace-p4-uncompared")
    no_verifier_state["semantic_verifier_observation"]["verifier_roles"] = {}
    no_verifier_state["semantic_verifier_observation"]["disagreement_span_ids"] = []
    uncompared = build_shadow_audit_record(no_verifier_state, settings=_settings())

    report = build_report([compared, uncompared], min_sample_size=1)
    metric = report["global"]["verifier_disagreement_rate"]
    assert report["global"]["sampled_auto_executed_requests"] == 2
    assert report["global"]["verifier_comparable_requests"] == 1
    assert report["global"]["verifier_noncomparable_requests"] == 1
    assert metric["numerator"] == 1
    assert metric["denominator"] == 2
    assert metric["value"] is None
    assert metric["status"] == "partial_verifier_coverage"
    comparable_rate = report["global"]["verifier_disagreement_comparable_request_rate"]
    assert comparable_rate["numerator"] == 1
    assert comparable_rate["denominator"] == 1
    assert comparable_rate["value"] == 1.0


def test_missing_execution_authorization_snapshot_is_reported_unknown():
    state = _state("trace-p4-no-auth-snapshot")
    state["tool_calls"][0]["authorization_snapshot"] = None
    record = build_shadow_audit_record(state, settings=_settings())
    assert record is not None
    assert record["authorization_outcome"] == {
        "status": "unknown", "reason_code": "authorization_snapshot_incomplete",
    }


def test_review_annotations_gate_contract_and_execution_errors():
    record = _record()
    report = build_report([record])
    assert report["global"]["wrong_contract_rate"]["value"] is None
    assert report["global"]["wrong_execution_rate"]["value"] is None

    with pytest.raises(ValueError, match="eligible review source"):
        merge_annotations([record], [{
            "audit_id": record["audit_id"],
            "wrong_contract": True,
            "contract_review_status": "reviewed",
        }])

    with pytest.raises(ValueError, match="reviewed or adjudicated"):
        merge_annotations([record], [{
            "audit_id": record["audit_id"],
            "human_review_status": "pending",
            "reviewed_label": "fp",
        }])

    reviewed = merge_annotations([record], [{
        "audit_id": record["audit_id"],
        "human_review_status": "adjudicated",
        "reviewed_span_labels": {"m1": "fp"},
        "contract_review_status": "reviewed",
        "wrong_contract": True,
        "contract_review_source": "calibration",
        "execution_review_status": "adjudicated",
        "wrong_execution": False,
        "execution_review_source": "human_gold",
    }])
    annotated = build_report(reviewed)
    assert annotated["global"]["reviewed_fp_rate"]["value"] == 1.0
    assert annotated["global"]["wrong_contract_rate"]["value"] == 1.0
    assert annotated["global"]["wrong_execution_rate"]["value"] == 0.0

    with pytest.raises(ValueError, match="invalid audit record"):
        merge_annotations(reviewed, [{
            "audit_id": record["audit_id"],
            "human_review_status": "pending",
        }])
    with pytest.raises(ValueError, match="duplicate audit_id"):
        merge_annotations([record], [
            {"audit_id": record["audit_id"]},
            {"audit_id": record["audit_id"]},
        ])

    with pytest.raises(ValueError, match="rule-role span"):
        merge_annotations([record], [{
            "audit_id": record["audit_id"],
            "human_review_status": "reviewed",
            "reviewed_span_labels": {"m999": "fp"},
        }])


def test_high_confidence_fp_only_recommends_and_never_mutates_runtime_state():
    states = [_state(f"closed-fp-{index}", dimensions=("product_category",)) for index in range(5)]
    before = deepcopy(states)
    records = [
        build_shadow_audit_record(state, settings=_settings())
        for state in states
    ]
    annotations = [
        {
            "audit_id": record["audit_id"],
            "human_review_status": "adjudicated",
            "reviewed_span_labels": {"m1": "fp"},
        }
        for record in records
    ]
    reviewed = merge_annotations(records, annotations)
    report = build_report(reviewed, fp_budget=0.05, min_sample_size=3)

    assert report["global"]["high_confidence_reviewed_fp_rate"]["value"] == 1.0
    assert report["downgrade_recommendations"]
    recommendation = report["downgrade_recommendations"][0]
    assert recommendation["status"] == "downgrade_recommended"
    assert recommendation["sample_count"] == 5
    assert recommendation["confidence_interval_95"][0] > recommendation["fp_budget"]
    assert recommendation["automatic_change_applied"] is False
    assert states == before


def test_invalid_input_records_are_visible_and_suppress_recommendations():
    states = [_state(f"invalid-input-check-{index}", dimensions=("product_category",)) for index in range(5)]
    records = [build_shadow_audit_record(state, settings=_settings()) for state in states]
    annotations = [{
        "audit_id": record["audit_id"],
        "human_review_status": "adjudicated",
        "reviewed_span_labels": {"m1": "fp"},
    } for record in records]
    reviewed = merge_annotations(records, annotations)
    invalid_record = dict(records[0], forbidden_raw_question="must not be hidden")
    report = build_report([*reviewed, invalid_record], fp_budget=0.05, min_sample_size=3)

    assert report["invalid_record_count"] == 1
    assert report["integrity_status"] == "invalid_records_rejected"
    assert report["downgrade_recommendations"] == []


def test_request_level_error_labels_are_marked_in_each_dimension_stratum():
    record = build_shadow_audit_record(
        _state("trace-p4-request-error-label", dimensions=("product_category", "payment_type")),
        settings=_settings(),
    )
    reviewed = merge_annotations([record], [{
        "audit_id": record["audit_id"],
        "contract_review_status": "reviewed",
        "wrong_contract": True,
        "contract_review_source": "human",
        "execution_review_status": "reviewed",
        "wrong_execution": False,
        "execution_review_source": "human_gold",
    }])
    report = build_report(reviewed, min_sample_size=1)

    rows = {
        row["dimension"]: row
        for row in report["strata"]
    }
    assert rows["product_category"]["wrong_contract_rate"]["value"] == 1.0
    assert rows["payment_type"]["wrong_contract_rate"]["value"] == 1.0
    assert all(
        row["record_level_error_label_attribution"]
        == "request_level_label_repeated_for_each_included_dimension"
        for row in rows.values()
    )

    with pytest.raises(ValueError, match="invalid audit record"):
        merge_annotations([record], [{
            "audit_id": record["audit_id"],
            "human_review_status": "reviewed",
            "reviewed_label": "fp",
        }])


def test_report_deduplicates_repeated_trace_before_counting():
    record = _record("duplicate-trace")
    report = build_report([record, dict(record)])
    assert report["source_record_count"] == 2
    assert report["unique_record_count"] == 1
    assert report["duplicate_records_ignored"] == 1
    assert report["global"]["sampled_auto_executed_requests"] == 1


def test_audit_builder_never_calls_external_provider(monkeypatch):
    from app.context import semantic_verifier

    def forbidden(*_args, **_kwargs):
        raise AssertionError("P4 must not invoke the verifier")

    monkeypatch.setattr(semantic_verifier, "_default_invoke", forbidden)
    assert build_shadow_audit_record(_state(), settings=_settings()) is not None


def test_persist_trace_runs_report_only_sidecar_after_real_terminal_state(monkeypatch, tmp_path):
    from app.context import shadow_audit
    from app.core.config import Settings
    from app.graph import workflow

    settings = Settings(
        semantic_shadow_enabled=True,
        semantic_shadow_sample_rate=100.0,
        semantic_shadow_salt="workflow-test-salt",
        semantic_verifier_mode="shadow",
    )
    state = _state("workflow-p4-trace")
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    monkeypatch.setattr(
        shadow_audit,
        "_default_audit_db_path",
        lambda: tmp_path / "workflow-shadow.sqlite3",
    )
    writes = []
    monkeypatch.setattr(workflow, "write_trace", lambda trace_id, payload: writes.append((trace_id, payload)))

    returned = workflow.persist_trace(state)

    assert returned is state
    assert len(writes) == 1
    assert len(read_shadow_audit_records(path=tmp_path / "workflow-shadow.sqlite3")) == 1


def test_persist_trace_failure_does_not_block_trace_write(monkeypatch):
    from app.context import shadow_audit
    from app.graph import workflow

    state = _state("workflow-p4-sidecar-failure")
    writes = []

    def fail_audit(_state):
        raise OSError("sidecar unavailable")

    monkeypatch.setattr(shadow_audit, "record_completed_shadow_audit", fail_audit)
    monkeypatch.setattr(workflow, "write_trace", lambda trace_id, payload: writes.append(trace_id))

    returned = workflow.persist_trace(state)
    assert returned is state
    assert writes == [state["trace_id"]]


def test_audit_database_lock_wait_is_bounded(monkeypatch, tmp_path):
    from app.context import shadow_audit

    settings = _settings()
    monkeypatch.setattr(shadow_audit, "_settings", lambda: settings)
    path = tmp_path / "locked-audit.sqlite3"
    with sqlite3.connect(path) as setup:
        setup.execute(
            "CREATE TABLE semantic_shadow_audit "
            "(audit_id TEXT PRIMARY KEY, sampled_at TEXT NOT NULL, record_json TEXT NOT NULL)"
        )
    lock = sqlite3.connect(path)
    lock.execute("BEGIN IMMEDIATE")
    try:
        started = monotonic()
        stored = shadow_audit.record_completed_shadow_audit(_state("locked-audit"), path=path)
        elapsed = monotonic() - started
    finally:
        lock.rollback()
        lock.close()

    assert stored is False
    assert elapsed < 2.0
