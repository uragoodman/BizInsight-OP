import hashlib
import json

from app.llm.client import _capture_semantic_eval_parse_failure_output
from scripts.build_semantic_interpreter_evidence import build_summary
from scripts.run_semantic_interpreter_eval import _metric_summary


def test_parse_failure_capture_is_bounded_and_opt_in():
    raw_output = "x" * 700
    call = {
        "operation": "extract.semantic_interpreter",
        "trace_id": "v341-required-011-11",
    }
    state = {
        "_semantic_interpreter_eval_raw_output_case_id": "v341-required-011-11",
    }

    _capture_semantic_eval_parse_failure_output(call, state, raw_output)

    diagnostic = call["semantic_interpreter_eval_raw_output_diagnostic"]
    assert diagnostic["raw_output_chars"] == 700
    assert diagnostic["raw_output_sha256"] == hashlib.sha256(raw_output.encode()).hexdigest()
    assert diagnostic["truncated_text"] == raw_output[:500]
    assert diagnostic["truncated"] is True

    unmarked_call = {
        "operation": "extract.semantic_interpreter",
        "trace_id": "ordinary-request",
    }
    _capture_semantic_eval_parse_failure_output(unmarked_call, {}, raw_output)
    assert "semantic_interpreter_eval_raw_output_diagnostic" not in unmarked_call


def test_safe_summary_keeps_only_allowlisted_parse_diagnostics(tmp_path):
    raw_output = "bad json " + "x" * 600
    digest = hashlib.sha256(raw_output.encode()).hexdigest()
    payload = {
        "results": [
            {
                "case_id": "v341-required-011-11",
                "observation": {
                    "status": "parse_failure",
                    "raw_output_diagnostic": {
                        "capture_status": "captured",
                        "raw_output_chars": len(raw_output),
                        "raw_output_sha256": digest,
                        "truncated_text": raw_output[:500],
                        "truncated": True,
                    },
                },
            },
            {
                "case_id": "other-case",
                "observation": {
                    "status": "parse_failure",
                    "raw_output_diagnostic": {
                        "capture_status": "captured",
                        "raw_output_chars": len(raw_output),
                        "raw_output_sha256": digest,
                        "truncated_text": raw_output,
                        "truncated": True,
                    },
                },
            },
            {
                "case_id": "v341-labeled-sort-basis-01",
                "observation": {
                    "status": "parse_failure",
                    "raw_output_diagnostic": {
                        "capture_status": "captured",
                        "raw_output_chars": len(raw_output),
                        "raw_output_sha256": digest,
                        "truncated_text": raw_output[:500],
                        "truncated": True,
                    },
                },
            },
            {
                "case_id": "schema-case",
                "observation": {
                    "status": "schema_invalid",
                    "reason_code": "candidate_schema_invalid",
                    "schema_errors": [{
                        "loc": ["operations", 0, "target_concept_ids"],
                        "type": "too_short",
                        "msg": "Do not copy arbitrary provider text into the safe summary.",
                    }],
                    "repair": {
                        "attempted": True,
                        "prompt_version": "semantic-interpreter-repair-v1",
                        "attempt_count": 1,
                        "recovered": True,
                        "initial_status": "schema_invalid",
                        "initial_reason_code": "candidate_schema_invalid",
                        "initial_issue_codes": [],
                        "final_status": "ok",
                        "initial_schema_errors": [{"msg": "PRIVATE_RAW_TEXT"}],
                    },
                },
            },
        ],
    }
    source = tmp_path / "raw-eval.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    summary = build_summary(source)

    assert summary["results"][0]["raw_output_diagnostic"]["raw_output_chars"] == len(raw_output)
    assert len(summary["results"][0]["raw_output_diagnostic"]["truncated_text"]) == 500
    assert "raw_output_diagnostic" not in summary["results"][1]
    assert summary["results"][2]["raw_output_diagnostic"]["raw_output_sha256"] == digest
    assert summary["results"][3]["schema_errors"] == [{
        "loc": ["operations", 0, "target_concept_ids"],
        "type": "too_short",
    }]
    assert "Do not copy arbitrary provider text" not in json.dumps(summary)
    assert summary["results"][3]["repair"]["recovered"] is True
    assert summary["results"][3]["repair"]["initial_status"] == "schema_invalid"
    assert "PRIVATE_RAW_TEXT" not in json.dumps(summary)


def test_span_reference_metric_uses_only_results_with_candidates():
    def result(status):
        return {
            "label_status": "draft",
            "observation": {
                "status": status,
                "reason_code": None,
                "issue_codes": [],
                "contract": None,
            },
            "provider_failure": None,
            "scores": {
                "reviewed_status_match": None,
                "draft_status_match": None,
                "comparisons": [],
                "exact_match": None,
            },
        }

    summary = _metric_summary([result("ok"), result("parse_failure")])

    assert summary["schema_valid_count"] == 1
    assert summary["attempted"] == 2
    assert summary["span_reference_valid_count"] == 1
    assert summary["span_reference_result_count"] == 1
    assert summary["span_reference_valid_rate"] == 1.0


def test_span_reference_metric_is_unavailable_without_candidate_results():
    result = {
        "label_status": "draft",
        "observation": {
            "status": "parse_failure",
            "reason_code": "model_output_unparseable",
            "issue_codes": [],
            "contract": None,
        },
        "provider_failure": None,
        "scores": {
            "reviewed_status_match": None,
            "draft_status_match": None,
            "comparisons": [],
            "exact_match": None,
        },
    }

    summary = _metric_summary([result])

    assert summary["span_reference_valid_count"] == 0
    assert summary["span_reference_result_count"] == 0
    assert summary["span_reference_valid_rate"] is None


def test_metrics_preserve_first_attempt_failure_when_repair_recovers():
    result = {
        "label_status": "draft",
        "observation": {
            "status": "ok",
            "reason_code": None,
            "issue_codes": [],
            "contract": {"semantic_status": "CONFIRMED"},
            "repair": {
                "attempted": True,
                "initial_status": "schema_invalid",
                "recovered": True,
            },
        },
        "provider_failure": None,
        "scores": {
            "reviewed_status_match": None,
            "draft_status_match": None,
            "comparisons": [],
            "exact_match": None,
        },
    }

    summary = _metric_summary([result])

    assert summary["schema_valid_count"] == 1
    assert summary["initial_schema_valid_count"] == 0
    assert summary["repair_attempt_count"] == 1
    assert summary["repair_recovered_count"] == 1
    assert summary["repair_recovery_rate"] == 1.0
