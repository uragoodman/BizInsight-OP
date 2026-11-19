from __future__ import annotations

from evaluation.resume_bg_companion_audit import (
    build_companion_quality_audit, summarize_companion_quality_records,
)


def _numeric_case():
    return {
        "case_id": "CASE-NUM-1", "target_metric": "gmv", "operation": "metric_value",
        "scope": "platform", "dimension": None,
        "period": {"kind": "point", "date": "2017-01-06"},
        "expected": {"outcome": "numeric_result", "truth": {"value": 12.5}},
    }


def _numeric_trace():
    run_id = "run-a"
    request_id = f"{run_id}:tool:1"
    evidence_id = f"{run_id}:evidence:1"
    return {
        "trace_id": run_id,
        "state_snapshot": {
            "trace_id": run_id, "scope_type": "platform",
            "authorized_metrics": [{"metric_id": "gmv", "authorization_fingerprint": "auth-hash"}],
            "authorization_boundary": {"permission_version": "policy-v1"},
            "tool_calls": [{
                "request_id": request_id, "tool_name": "query_value", "planned_metric_id": "gmv",
                "planned_operation": "metric_value", "arguments": {
                    "metric_id": "gmv", "metric_date": "2017-01-06", "scope": "platform",
                }, "status": "ok", "response": {"status": "ok"}, "evidence_id": evidence_id,
            }],
            "evidence_bundles": [{
                "evidence_id": evidence_id, "request_id": request_id, "trusted": True,
                "content": {"metric_id": "gmv", "request_id": request_id, "scope": "platform",
                "result": {"status": "ok", "parsed": {"authorized_metric": {
                    "metric_id": "gmv", "authorization_fingerprint": "auth-hash",
                    "query_scope_fingerprint": "scope-hash", "catalog_digest": "catalog-hash",
                }}, "metric_result": {"metric_name": "gmv", "value": 12.5}}},
            }],
            "decision_records": [{
                "request_id": request_id, "authorization_checked": True, "sql_executed": True,
                "metric_id": "gmv", "scope": "platform",
            }],
            "presentation": {"status": "completed", "metric_id": "gmv", "evidence_ids": [evidence_id]},
            "claims": [{"claim_type": "metric", "value": "GMV 为 12.5", "source_evidence_ids": [evidence_id]}],
            "claim_evidence_bindings": [{"field_path": "value", "evidence_ids": [evidence_id]}],
        },
    }


def _numeric_observation():
    return {
        "scope": "platform", "terminal_code": None, "precondition_verified": None,
        "numeric_field_comparisons": [{
            "path": "value", "expected": 12.5, "actual": 12.5,
            "actual_field_present": True, "matched": True,
            "absolute_error": 0.0, "absolute_tolerance": 0.02,
        }],
    }


def test_companion_audit_records_field_evidence_auth_citation_and_call_facts():
    audit = build_companion_quality_audit(
        _numeric_case(), "G", _numeric_trace(), _numeric_observation(), expected_run_id="run-a",
    )

    assert audit["schema_version"] == "resume-bg-companion-audit-v4"
    assert audit["frozen_case_contract"]["truth_sha256"]
    assert audit["run_identity"] == {"expected_run_id": "run-a", "observed_run_id": "run-a", "match": True}
    core = audit["core_conclusion_evidence_coverage"]
    assert core["numeric_match"] == {"numerator": 1, "denominator": 1, "rate": 1.0, "fail_count": 0, "unknown_count": 0}
    assert core["field_evidence_binding"]["rate"] == 1.0
    assert core["answer_evidence_binding"]["rate"] == 1.0
    assert audit["valid_evidence_citations"]["numerator"] == 2  # card + Claim are distinct occurrences
    assert audit["valid_evidence_citations"]["denominator"] == 2
    assert audit["authorization_decisions"]["calls"][0]["result"] == "allowed"
    assert audit["authorization_decisions"]["calls"][0]["policy_version"] == "policy-v1"
    assert audit["tool_call_validity"]["calls"][0]["result"] == "valid"
    assert audit["tool_call_validity"]["calls"][0]["execution_outcome"] == "provided"


def test_companion_audit_does_not_turn_missing_or_ambiguous_data_into_zero():
    case = _numeric_case()
    trace = {"trace_id": "run-empty", "state_snapshot": {"trace_id": "run-empty", "tool_calls": [], "evidence_bundles": []}}
    observation = {"scope": None, "numeric_field_comparisons": []}
    audit = build_companion_quality_audit(case, "B", trace, observation, expected_run_id="run-empty")

    assert audit["valid_evidence_citations"]["denominator"] == 0
    assert audit["valid_evidence_citations"]["rate"] is None
    assert audit["authorization_decisions"]["denominator"] == 0
    assert audit["tool_call_validity"]["invalid_rate"] is None
    assert audit["core_conclusion_evidence_coverage"]["numeric_match"]["denominator"] is None


def test_companion_audit_reads_b_model_claim_citations_without_creating_bindings():
    case, trace = _numeric_case(), _numeric_trace()
    state = trace["state_snapshot"]
    evidence_id = state["evidence_bundles"][0]["evidence_id"]
    state.pop("claims")
    state.pop("presentation")
    state.pop("claim_evidence_bindings")
    state["model_authored_final"] = {
        "action": "FINAL",
        "answer": "GMV 为 12.5。",
        "claims": [{"text": "GMV 为 12.5", "evidence_ids": [evidence_id]}],
    }
    state["model_evidence_reference_capture"] = {
        "schema_version": "bare-model-evidence-reference-capture-v1",
        "claims": [{"claim_index": 0, "evidence_ids": [evidence_id]}],
    }

    audit = build_companion_quality_audit(case, "B", trace, _numeric_observation(), expected_run_id="run-a")

    assert audit["valid_evidence_citations"]["numerator"] == 1
    assert audit["valid_evidence_citations"]["denominator"] == 1
    assert audit["valid_evidence_citations"]["occurrences"][0]["evidence_location"] == "/model_authored_final/claims/0/evidence_ids/0"
    # The recorder remains observational; B does not acquire G's bindings.
    assert "claim_evidence_bindings" not in state


def test_companion_audit_separates_misaligned_calls_duplicates_and_execution_errors():
    case, trace = _numeric_case(), _numeric_trace()
    state = trace["state_snapshot"]
    first = state["tool_calls"][0]
    duplicate = {**first, "request_id": "run-a:tool:2", "status": "failed", "response": {"error": "db timeout"}}
    state["tool_calls"].append(duplicate)
    audit = build_companion_quality_audit(case, "B", trace, _numeric_observation(), expected_run_id="run-a")
    calls = audit["tool_call_validity"]

    assert calls["duplicate_signature_count"] == 1
    assert calls["execution_error_count"] == 1
    assert calls["calls"][1]["duplicate_of_request_id"] == "run-a:tool:1"
    assert calls["calls"][1]["result"] == "valid"  # parameters are valid; execution failed separately
    assert calls["calls"][1]["execution_outcome"] == "execution_error"

    first["arguments"]["metric_date"] = "2017-01-07"
    misaligned = build_companion_quality_audit(case, "B", trace, _numeric_observation(), expected_run_id="run-a")
    assert misaligned["tool_call_validity"]["calls"][0]["result"] == "invalid"
    assert misaligned["valid_evidence_citations"]["occurrences"][0]["period_match"] is False


def test_companion_audit_keeps_authorization_unknown_unless_decision_is_explicit():
    case, trace = _numeric_case(), _numeric_trace()
    state = trace["state_snapshot"]
    state["authorized_metrics"] = []
    state["decision_records"] = []
    state["evidence_bundles"][0]["content"]["result"]["parsed"].pop("authorized_metric", None)
    row = build_companion_quality_audit(case, "B", trace, _numeric_observation(), expected_run_id="run-a")["authorization_decisions"]["calls"][0]
    assert row["result"] == "unknown"

    state["tool_calls"][0]["status"] = "unsupported"
    state["decision_records"] = [{
        "request_id": state["tool_calls"][0]["request_id"], "authorization_checked": True,
        "sql_executed": False, "limitation_code": "AUTH_DENIED",
    }]
    row = build_companion_quality_audit(case, "B", trace, _numeric_observation(), expected_run_id="run-a")["authorization_decisions"]["calls"][0]
    assert row["result"] == "denied"


def test_capability_stop_without_query_is_not_an_authorization_or_tool_error():
    case, trace = _numeric_case(), _numeric_trace()
    state = trace["state_snapshot"]
    call = state["tool_calls"][0]
    call.update({
        "tool_name": "metric_clarification", "planned_metric_id": None,
        "planned_operation": None, "arguments": {"scope": "platform"}, "status": "unsupported",
    })
    state["decision_records"] = [{
        "request_id": call["request_id"], "authorization_checked": True,
        "sql_executed": False, "limitation_code": "CAPABILITY_GAP",
    }]
    audit = build_companion_quality_audit(case, "G", trace, _numeric_observation(), expected_run_id="run-a")
    auth = audit["authorization_decisions"]
    calls = audit["tool_call_validity"]
    assert auth["calls"][0]["result"] == "not_applicable"
    assert auth["denominator"] == 0
    assert auth["not_applicable_count"] == 1
    assert calls["calls"][0]["result"] == "not_applicable"
    assert calls["denominator"] == 0
    assert calls["not_applicable_count"] == 1


def test_nested_registry_trend_period_is_audited_as_recorded_not_unknown():
    case = {
        "case_id": "CASE-TREND", "target_metric": "order_count", "operation": "metric_trend",
        "scope": "platform", "dimension": None,
        "period": {"kind": "range", "start": "2017-04-07", "end": "2017-04-08"},
        "expected": {"outcome": "numeric_result", "truth": {}},
    }
    trace = _numeric_trace()
    call = trace["state_snapshot"]["tool_calls"][0]
    call.update({
        "tool_name": "registry_trend", "planned_metric_id": "order_count",
        "planned_operation": "metric_trend",
        "arguments": {"metric_name": "order_count", "scope": "platform", "period": {
            "start_date": "2017-04-07", "end_date": "2017-04-08", "granularity": "day",
        }},
    })
    audit = build_companion_quality_audit(case, "G", trace, {"numeric_field_comparisons": []}, expected_run_id="run-a")
    assert audit["tool_call_validity"]["calls"][0]["result"] == "valid"


def test_authorization_snapshot_is_per_call_and_uses_matching_request_evidence():
    case = {
        "case_id": "CASE-PER-CALL-AUTH", "target_metric": "order_count",
        "operation": "dimension_contribution", "scope": "platform",
        "dimension": "product_category",
        "period": {"kind": "point_pair", "baseline": "2017-02-11", "current": "2017-02-12"},
        "expected": {"outcome": "numeric_result", "truth": {"total_delta": 1}},
    }
    run_id, request_id, evidence_id = "auth-run", "auth-run:tool:1", "auth-run:evidence:1"
    trace = {"trace_id": run_id, "state_snapshot": {
        "trace_id": run_id, "scope_type": "platform", "tenant_id": "t1", "user_id": "u1", "user_role": "analyst",
        "tool_calls": [{
            "request_id": request_id, "tool_name": "compare_breakdown_by_dimension",
            "planned_metric_id": "order_count", "planned_operation": "metric_contribution",
            "arguments": {"metric_id": "order_count", "dimension": "product_category", "scope": "platform",
                "baseline_date": "2017-02-11", "current_date": "2017-02-12"},
            "status": "ok",
        }],
        "decision_records": [],
        "evidence_bundles": [{
            "evidence_id": evidence_id, "request_id": request_id, "trusted": True,
            "content": {"request_id": request_id, "metric_id": "order_count", "result": {
                "status": "ok", "parsed": {"authorized_metric": {
                    "metric_id": "order_count", "provenance_type": "exact",
                    "authorization_fingerprint": "auth-fingerprint",
                    "query_scope_fingerprint": "scope-fingerprint", "catalog_digest": "catalog-digest",
                }},
            }},
        }],
        "presentation": {"status": "completed", "evidence_ids": [evidence_id]},
        "claims": [],
    }}
    audit = build_companion_quality_audit(case, "G", trace, {"numeric_field_comparisons": []}, expected_run_id=run_id)
    row = audit["authorization_decisions"]["calls"][0]
    assert row["request_id"] == request_id
    assert row["result"] == "allowed"
    assert row["snapshot_integrity"] == "valid"
    assert row["snapshot"]["authorized_metric_source"].startswith("evidence_bundles/")
    assert row["snapshot"]["authorized_metric"]["query_scope_fingerprint"]
    assert row["snapshot"]["identity_sha256"]
    assert row["snapshot"]["decision_source"] == "same_request_authorized_metric_and_successful_query_result"


def test_authorization_snapshots_do_not_leak_between_calls_in_same_run():
    case, trace = _numeric_case(), _numeric_trace()
    state = trace["state_snapshot"]
    first = state["tool_calls"][0]
    second = dict(first)
    second["request_id"] = "run-a:tool:2"
    second["status"] = "refused_before_execution"
    second["response"] = {
        "request_id": "run-a:tool:2",
        "result": {"status": "refused_before_execution", "parsed": {}},
    }
    state["tool_calls"].append(second)
    state["decision_records"] = [{
        "request_id": "run-a:tool:2", "authorization_checked": True,
        "authorization_decision": "denied", "limitation_code": "AUTH_DENIED",
        "sql_executed": False,
    }]

    audit = build_companion_quality_audit(case, "G", trace, _numeric_observation(), expected_run_id="run-a")
    calls = audit["authorization_decisions"]["calls"]
    assert [item["request_id"] for item in calls] == ["run-a:tool:1", "run-a:tool:2"]
    assert [item["result"] for item in calls] == ["allowed", "denied"]
    assert [item["snapshot"]["request_id"] for item in calls] == ["run-a:tool:1", "run-a:tool:2"]
    assert calls[1]["snapshot"]["authorized_metric_present"] is False
    assert calls[1]["snapshot"]["decision_source"] == "request_correlated_decision_record"


def test_unrelated_authorized_metric_cannot_authorize_another_call():
    case, trace = _numeric_case(), _numeric_trace()
    state = trace["state_snapshot"]
    state["authorized_metrics"] = [{"metric_id": "another_metric", "authorization_fingerprint": "wrong"}]
    state["tool_calls"][0]["response"] = {"request_id": "run-a:tool:1", "result": {"status": "ok", "parsed": {}}}
    state["evidence_bundles"][0]["content"]["result"].setdefault("parsed", {})["authorized_metric"] = {"metric_id": "another_metric"}
    row = build_companion_quality_audit(case, "B", trace, _numeric_observation(), expected_run_id="run-a")["authorization_decisions"]["calls"][0]
    assert row["result"] == "unknown"
    assert row["snapshot_integrity"] == "invalid"
    assert row["snapshot"]["authorized_metric_present"] is True
    assert row["snapshot"]["metric_match"] is False
    assert row["snapshot"]["authorized_metric_id"] == "another_metric"


def test_unbound_run_level_authorized_metric_does_not_authorize_each_tool_call():
    case, trace = _numeric_case(), _numeric_trace()
    state = trace["state_snapshot"]
    state["tool_calls"][0]["response"] = {"request_id": "run-a:tool:1", "result": {"status": "ok", "parsed": {}}}
    state["evidence_bundles"][0]["content"]["result"]["parsed"].pop("authorized_metric", None)
    row = build_companion_quality_audit(case, "G", trace, _numeric_observation(), expected_run_id="run-a")["authorization_decisions"]["calls"][0]
    assert row["result"] == "unknown"
    assert row["snapshot"]["authorized_metric_source"] is None
    assert row["snapshot"]["authorized_metric_present"] is False
    assert row["snapshot"]["authorization_request_bound"] is False
    assert row["snapshot_integrity"] == "unknown"


def test_mismatched_embedded_authorization_snapshot_is_not_applied_to_call():
    case, trace = _numeric_case(), _numeric_trace()
    call = trace["state_snapshot"]["tool_calls"][0]
    call["authorization_snapshot"] = {
        "request_id": "another-run:tool:9", "decision": "allowed",
        "authorization_checked": True, "authorized_metric": {"metric_id": "gmv"},
    }
    row = build_companion_quality_audit(case, "G", trace, _numeric_observation(), expected_run_id="run-a")["authorization_decisions"]["calls"][0]
    assert row["result"] == "unknown"
    assert row["snapshot"]["embedded_snapshot_request_match"] is False
    assert row["snapshot"]["authorization_checked"] is True
    assert row["snapshot_integrity"] == "invalid"


def test_contribution_fields_bind_to_exact_registered_values_and_card_paths():
    case = {
        "case_id": "CASE-CATEGORY-CONTRIB", "target_metric": "order_count",
        "operation": "dimension_contribution", "scope": "platform", "dimension": "product_category",
        "period": {"kind": "point_pair", "baseline": "2017-02-11", "current": "2017-02-12"},
        "expected": {"outcome": "numeric_result", "truth": {"baseline_groups": {"A": 2},
            "current_groups": {"A": 4}, "baseline_total": 2, "current_total": 4, "total_delta": 2}},
    }
    rid, req, eid = "contrib-run", "contrib-run:tool:1", "contrib-run:evidence:1"
    comparisons = [
        {"path": "baseline_groups.A", "expected": 2, "actual": 2, "actual_field_present": True, "matched": True, "absolute_tolerance": 0},
        {"path": "current_groups.A", "expected": 4, "actual": 4, "actual_field_present": True, "matched": True, "absolute_tolerance": 0},
        {"path": "baseline_total", "expected": 2, "actual": 2, "actual_field_present": True, "matched": True, "absolute_tolerance": 0},
        {"path": "current_total", "expected": 4, "actual": 4, "actual_field_present": True, "matched": True, "absolute_tolerance": 0},
        {"path": "total_delta", "expected": 2, "actual": 2, "actual_field_present": True, "matched": True, "absolute_tolerance": 0},
    ]
    trace = {"trace_id": rid, "state_snapshot": {
        "trace_id": rid, "scope_type": "platform",
        "tool_calls": [{"request_id": req, "tool_name": "compare_breakdown_by_dimension",
            "planned_metric_id": "order_count", "planned_operation": "metric_contribution",
            "arguments": {"metric_id": "order_count", "dimension": "product_category", "scope": "platform",
                "baseline_date": "2017-02-11", "current_date": "2017-02-12"}, "status": "ok"}],
        "evidence_bundles": [{"evidence_id": eid, "request_id": req, "trusted": True, "content": {
            "request_id": req, "metric_id": "order_count", "result": {"status": "ok",
                "parsed": {"authorized_metric": {"metric_id": "order_count"}},
                "contribution_result": {"baseline_items": [{"dimension_value": "A", "value": 2}],
                    "current_items": [{"dimension_value": "A", "value": 4}],
                    "baseline_total": 2, "current_total": 4, "total_delta": 2}},
        }}],
        "presentation": {"status": "completed", "contributions": [{
            "metric_name": "order_count", "dimension": "product_category", "baseline_value": 2,
            "current_value": 4, "total_delta": 2, "evidence_ids": [eid],
            "field_evidence_ids": {
                "baseline_value": [eid], "current_value": [eid], "total_delta": [eid],
            },
            "items": [{"dimension_value": "A", "baseline_value": 2, "current_value": 4,
                "field_evidence_ids": {"baseline_value": [eid], "current_value": [eid]}}],
        }]},
        "claims": [{"claim_type": "analysis", "value": "A: 2 -> 4", "source_evidence_ids": [eid]}],
    }}
    audit = build_companion_quality_audit(case, "G", trace, {"numeric_field_comparisons": comparisons}, expected_run_id=rid)
    core = audit["core_conclusion_evidence_coverage"]
    assert core["numeric_match"]["rate"] == 1.0
    assert core["field_evidence_binding"]["rate"] == 1.0
    assert core["presentation_field_evidence_binding"]["rate"] == 1.0
    assert core["answer_evidence_binding"]["rate"] is None
    assert core["answer_evidence_binding"]["unknown_count"] == len(comparisons)
    assert core["claims"][0]["presentation_field_path"].endswith("items/0/baseline_value")


def test_unrelated_answer_citation_leaves_atom_binding_unknown_not_invalid():
    case, trace = _numeric_case(), _numeric_trace()
    trace["state_snapshot"].pop("claim_evidence_bindings", None)
    core = build_companion_quality_audit(case, "G", trace, _numeric_observation(), expected_run_id="run-a")["core_conclusion_evidence_coverage"]
    assert core["answer_evidence_binding"]["numerator"] == 0
    assert core["answer_evidence_binding"]["invalid_count"] == 0
    assert core["answer_evidence_binding"]["unknown_count"] == 1


def test_all_unknown_presentation_bindings_have_null_rate_not_zero():
    case, trace = _numeric_case(), _numeric_trace()
    trace["state_snapshot"].pop("presentation", None)
    core = build_companion_quality_audit(case, "G", trace, _numeric_observation(), expected_run_id="run-a")["core_conclusion_evidence_coverage"]
    assert core["presentation_field_evidence_binding"]["numerator"] == 0
    assert core["presentation_field_evidence_binding"]["invalid_count"] == 0
    assert core["presentation_field_evidence_binding"]["unknown_count"] == 1
    assert core["presentation_field_evidence_binding"]["rate"] is None


def test_card_level_evidence_does_not_substitute_for_each_numeric_field_binding():
    case = {
        "case_id": "CASE-FIELD-BINDING", "target_metric": "order_count",
        "operation": "dimension_contribution", "scope": "platform",
        "dimension": "product_category",
        "period": {"kind": "point_pair", "baseline": "2017-02-11", "current": "2017-02-12"},
        "expected": {"outcome": "numeric_result", "truth": {"baseline_groups": {"A": 2}}},
    }
    run_id, request_id, evidence_id = "field-run", "field-run:tool:1", "field-run:evidence:1"
    trace = {"trace_id": run_id, "state_snapshot": {
        "trace_id": run_id, "scope_type": "platform",
        "tool_calls": [{"request_id": request_id, "evidence_id": evidence_id,
            "tool_name": "compare_breakdown_by_dimension", "planned_metric_id": "order_count",
            "planned_operation": "metric_contribution", "status": "ok",
            "arguments": {"metric_name": "order_count", "dimension": "product_category",
                "scope": "platform", "baseline_date": "2017-02-11", "current_date": "2017-02-12"}}],
        "evidence_bundles": [{"evidence_id": evidence_id, "trusted": True, "content": {
            "request_id": request_id, "metric_id": "order_count", "result": {"status": "ok",
                "parsed": {"authorized_metric": {"metric_id": "order_count"}},
                "contribution_result": {"metric_name": "order_count", "dimension": "product_category",
                    "baseline_items": [{"dimension_value": "A", "value": 2}],
                    "current_items": [{"dimension_value": "A", "value": 4}],
                    "baseline_total": 2, "current_total": 4, "total_delta": 2}}}}],
        # A card citation must not be inherited by a numeric field.
        "analysis_presentation": {"status": "completed", "contributions": [{
            "metric_name": "order_count", "dimension": "product_category",
            "baseline_value": 2, "current_value": 4, "total_delta": 2,
            "evidence_ids": [evidence_id],
            "items": [{"dimension_value": "A", "baseline_value": 2, "current_value": 4}],
        }]},
    }}
    observation = {"numeric_field_comparisons": [{
        "path": "baseline_groups.A", "expected": 2, "actual": 2,
        "actual_field_present": True, "matched": True, "absolute_tolerance": 0,
    }]}
    core = build_companion_quality_audit(case, "G", trace, observation, expected_run_id=run_id)["core_conclusion_evidence_coverage"]
    assert core["numeric_match"]["rate"] == 1.0
    assert core["presentation_field_evidence_binding"]["numerator"] == 0
    assert core["presentation_field_evidence_binding"]["invalid_count"] == 0
    assert core["presentation_field_evidence_binding"]["unknown_count"] == 1


def test_companion_audit_checks_safe_stop_structure_but_leaves_semantics_for_review():
    case = {
        "case_id": "CASE-STOP", "target_metric": "review_comment_rate", "operation": "expected_data_insufficiency",
        "scope": "platform", "period": {"kind": "point_pair", "baseline": "2017-01-01", "current": "2017-01-02"},
        "expected": {"outcome": "safe_stop", "terminal_code": "DATA_INSUFFICIENT"},
    }
    trace = {"trace_id": "stop-run", "state_snapshot": {
        "trace_id": "stop-run", "terminal_code": "DATA_INSUFFICIENT",
        "presentation": {"status": "partial"}, "tool_calls": [], "evidence_bundles": [],
    }}
    audit = build_companion_quality_audit(case, "G", trace, {"terminal_code": "DATA_INSUFFICIENT", "precondition_verified": True})
    stop = audit["correct_refusal_content"]
    assert stop["automated_status"] == "pass"
    assert stop["semantic_content_status"] == "manual_review_required"


def test_summary_uses_explicit_cohort_and_preserves_zero_denominators():
    case, trace = _numeric_case(), _numeric_trace()
    observation = _numeric_observation()
    first = build_companion_quality_audit(case, "B", trace, observation, expected_run_id="run-a")
    second = build_companion_quality_audit(case, "G", trace, observation, expected_run_id="run-a")
    records = [
        {"run_id": "run-a", "cohort_id": "freeze-1", "arm": "B", "companion_quality_audit": first},
        {"run_id": "run-b", "cohort_id": "freeze-1", "arm": "G", "companion_quality_audit": second},
        {"run_id": "run-c", "arm": "B", "companion_quality_audit": None},
    ]
    summary = summarize_companion_quality_records(records)
    assert [(row["cohort_id"], row["arm"]) for row in summary["groups"]] == [
        ("freeze-1", "B"), ("freeze-1", "G"), ("unversioned:run-c", "B"),
    ]
    assert summary["groups"][0]["core_numeric_fields"]["denominator"] == 1
    assert summary["groups"][2]["core_numeric_fields"]["denominator"] == 0
    assert summary["groups"][2]["core_numeric_fields"]["rate"] is None


def test_companion_audit_marks_cross_run_trace_and_expected_run_mismatch():
    audit = build_companion_quality_audit(
        _numeric_case(), "B", _numeric_trace(), _numeric_observation(), expected_run_id="different-run",
    )
    assert audit["run_identity"]["match"] is False
    assert audit["valid_evidence_citations"]["numerator"] == 0
    assert audit["valid_evidence_citations"]["invalid_count"] == 2


def test_formal_executor_persists_companion_audit_without_database(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from app.tenancy import TenantContext
    from evaluation.provider_candidate_batch import RunLedger
    from evaluation.resume_formal_bg_executor import FormalBGBatchExecutor
    import evaluation.resume_formal_bg_executor as runner
    import evaluation.resume_bare_tool_loop as bare
    from app.graph import workflow

    case = _numeric_case()
    case["user_question"] = "offline test question"
    run_id = "formal-audit-run"
    entry = {"run_id": run_id, "case_id": case["case_id"], "arm": "B", "replicate": 1}
    state = _numeric_trace()["state_snapshot"]
    state["trace_id"] = run_id
    state["tool_calls"][0]["request_id"] = f"{run_id}:tool:1"
    state["tool_calls"][0]["response"]["request_id"] = f"{run_id}:tool:1"
    state["evidence_bundles"][0]["request_id"] = f"{run_id}:tool:1"
    state["evidence_bundles"][0]["content"]["request_id"] = f"{run_id}:tool:1"
    state["decision_records"][0]["request_id"] = f"{run_id}:tool:1"
    state["evidence_bundles"][0]["evidence_id"] = f"{run_id}:evidence:1"
    state["tool_calls"][0]["evidence_id"] = f"{run_id}:evidence:1"
    state["presentation"]["evidence_ids"] = [f"{run_id}:evidence:1"]
    state["claims"][0]["source_evidence_ids"] = [f"{run_id}:evidence:1"]
    state["claim_evidence_bindings"][0]["evidence_ids"] = [f"{run_id}:evidence:1"]

    monkeypatch.setattr(runner, "score_formal_run", lambda *_a, **_k: {
        "scorer_version": "test-score", "task_acceptance_status": "manual_review",
        "manual_review_required": True,
    })
    monkeypatch.setattr(runner, "token_usage_snapshot", lambda *_a, **_k: {"run_id": run_id, "status": "no_requests", "total_tokens": 0})
    monkeypatch.setattr(runner, "_reconcile_cost_accounting", lambda *_a, **_k: None)
    monkeypatch.setattr(bare, "run_bare_tool_loop", lambda *_a, **_k: state)
    monkeypatch.setattr(workflow, "get_settings", lambda: object())
    monkeypatch.setattr("evaluation.resume_bg_tool_surface.build_evaluation_tool_surface", lambda *_a, **_k: {"surface_sha256": "x"})

    class CostLedger:
        policy = SimpleNamespace()
        reserved_cny = 1.0
        reservations = []

        def reserve(self, unit_id):
            self.reservations.append({"run_id": unit_id, "reserved_cny": 1.0})
            return 1.0

        def record_usage(self, unit_id, usage):
            return {"run_id": unit_id, "reservation_exceeded": False}

    class Config:
        max_graph_steps = 8
        max_tool_calls = 4
        max_latency_ms = 1000
        max_task_tokens = 1000
        max_cost_cny_per_workflow = 1.0
        def build_llm_call_config(self, _settings):
            return None

    ledger = RunLedger(tmp_path / "formal-events.jsonl")
    trace_dir = tmp_path / "formal-traces"
    executor = FormalBGBatchExecutor(
        root=tmp_path, cases=[case], manifest={"experiment_id": "exp-audit"},
        frozen_hashes={}, qualification={}, ledger=ledger, cost_ledger=CostLedger(),
        trace_dir=trace_dir, config=Config(),
    )
    record = executor._run_one(
        entry, case,
        tenant_context=TenantContext(tenant_id="platform", user_id="offline", role="admin", scope_type="platform"),
        observation_builder=None, fault_context_factory=None,
    )

    assert record["companion_quality_audit"]["run_identity"]["match"] is True
    assert record["companion_quality_audit"]["authorization_decisions"]["allowed_count"] == 1
    auth_call = record["companion_quality_audit"]["authorization_decisions"]["calls"][0]
    assert auth_call["snapshot"]["request_id"] == f"{run_id}:tool:1"
    assert auth_call["snapshot"]["authorization_request_bound"] is True
    assert auth_call["snapshot"]["decision"] == "allowed"
    assert record["companion_quality_audit_error"] is None
    saved = __import__("json").loads((trace_dir / f"{run_id}.run_record.json").read_text(encoding="utf-8"))
    assert saved["companion_quality_audit"]["schema_version"] == "resume-bg-companion-audit-v4"
    assert saved["cohort_id"] == "exp-audit"
