from __future__ import annotations

import json

from scripts.check_semantic_interpreter_gate import REQUIRED_PHRASES, main


def _passing_result(case_id: str, text: str) -> dict:
    return {
        "case_id": case_id,
        "text": text,
        "label_status": "reviewed",
        "exception_escaped": False,
        "observation": {
            "status": "validator_issues",
            "issue_codes": ["primary_intent_operation_mismatch"],
            "contract": {"semantic_status": "CONFIRMED", "operations": []},
        },
        "review": {"reviewed_status": "CONFIRMED"},
    }


def _write_evidence(tmp_path, results):
    evaluation = tmp_path / "eval.json"
    dataset = tmp_path / "dataset.jsonl"
    evaluation.write_text(json.dumps({"results": results}, ensure_ascii=False), encoding="utf-8")
    dataset.write_text(
        "".join(json.dumps({"case_id": f"required-{i}", "text": phrase}, ensure_ascii=False) + "\n"
                for i, phrase in enumerate(sorted(REQUIRED_PHRASES))),
        encoding="utf-8",
    )
    return evaluation, dataset


def _all_ok_results():
    results = []
    for i, phrase in enumerate(sorted(REQUIRED_PHRASES)):
        result = _passing_result(f"ok-{i}", phrase)
        result["observation"]["status"] = "ok"
        result["observation"]["issue_codes"] = []
        results.append(result)
    return results


def _fault_injection_payload(*, failed_case: str | None = None):
    expectations = [
        ("timeout", "unavailable", "timeout"),
        ("invalid_json", "parse_failure", "invalid_json"),
        ("schema_invalid", "schema_invalid", "candidate_schema_invalid"),
        ("quota_blocked", "unavailable", "quota_blocked"),
    ]
    cases = [
        {
            "case_id": case_id,
            "status": status,
            "reason_code": reason_code,
            "exception_escaped": False,
            "contract_present": False,
            "issue_codes": [],
            "passed": case_id != failed_case,
        }
        for case_id, status, reason_code in expectations
    ]
    return {
        "required_cases": 4,
        "passed": failed_case is None,
        "cases": cases,
    }


def test_phase1_gate_passes_with_reviewed_metrics_and_full_coverage(tmp_path, capsys):
    evaluation, dataset = _write_evidence(
        tmp_path,
        [_passing_result(f"case-{i}", phrase) for i, phrase in enumerate(sorted(REQUIRED_PHRASES))],
    )

    result = main(["--eval", str(evaluation), "--dataset", str(dataset)])

    output = capsys.readouterr().out
    assert result == 0
    assert "PHASE1_GATE=PASS" in output
    assert "span/reference valid: 100.00% (18/18)" in output
    assert "fail_closed_source=evaluation" in output


def test_phase1_gate_scores_reviewed_dataset_expected_status_without_review_object(tmp_path, capsys):
    results = _all_ok_results()
    for result in results:
        result.pop("review")
        result["expected"] = {"status": "CONFIRMED"}
        result["review_source"] = "codex-ai-20261114"
    evaluation, dataset = _write_evidence(tmp_path, results)
    fault_evidence = tmp_path / "fault-evidence.json"
    fault_evidence.write_text(json.dumps(_fault_injection_payload()), encoding="utf-8")

    result = main([
        "--eval", str(evaluation), "--dataset", str(dataset),
        "--fault-injection-evidence", str(fault_evidence),
    ])

    output = capsys.readouterr().out
    assert result == 0
    assert "reviewed cases: 18" in output
    assert "reviewed semantic-status accuracy: 100.00% (18/18)" in output
    assert "reviewed semantic-status unscored: 0" in output


def test_phase1_gate_reports_nonzero_for_below_threshold_evidence(tmp_path, capsys):
    bad = _passing_result("case-1", "未覆盖短语")
    bad["observation"]["status"] = "validator_issues"
    bad["observation"]["issue_codes"] = ["span_surface_mismatch"]
    evaluation, dataset = _write_evidence(tmp_path, [bad])

    result = main(["--eval", str(evaluation), "--dataset", str(dataset)])

    output = capsys.readouterr().out
    assert result == 1
    assert "PHASE1_GATE=BLOCKED" in output
    assert "schema valid: 100.00% (1/1)" in output
    assert "span/reference valid: 0.00% (0/1)" in output
    assert "required adversarial phrase evaluation coverage must be 18/18" in output


def test_phase1_gate_excludes_fail_closed_without_candidate_from_span_denominator(tmp_path, capsys):
    results = _all_ok_results()
    for index in range(51):
        extra = _passing_result(f"extra-{index}", f"synthetic valid case {index}")
        extra["observation"]["status"] = "ok"
        extra["observation"]["issue_codes"] = []
        results.append(extra)
    results.append({
        "case_id": "provider-empty-response",
        "text": "synthetic provider failure case",
        "label_status": "draft",
        "exception_escaped": False,
        "observation": {
            "status": "parse_failure",
            "reason_code": "model_output_unparseable",
            "issue_codes": [],
            "contract": None,
        },
    })
    evaluation, dataset = _write_evidence(tmp_path, results)

    result = main(["--eval", str(evaluation), "--dataset", str(dataset)])

    output = capsys.readouterr().out
    assert result == 0
    assert "PHASE1_GATE=PASS" in output
    assert "schema valid: 98.57% (69/70)" in output
    assert "span/reference valid: 100.00% (69/69)" in output
    assert "fail-closed: 100.00% (1/1)" in output


def test_phase1_gate_reports_span_metric_unavailable_when_no_candidate_exists(tmp_path, capsys):
    failed = {
        "case_id": "provider-empty-response",
        "text": "未覆盖短语",
        "label_status": "draft",
        "exception_escaped": False,
        "observation": {
            "status": "parse_failure",
            "reason_code": "model_output_unparseable",
            "issue_codes": [],
            "contract": None,
        },
    }
    evaluation, dataset = _write_evidence(tmp_path, [failed])

    result = main(["--eval", str(evaluation), "--dataset", str(dataset)])

    output = capsys.readouterr().out
    assert result == 1
    assert "PHASE1_GATE=BLOCKED" in output
    assert "span/reference valid: n/a (0/0)" in output


def test_phase1_gate_rejects_missing_reviewed_labels(tmp_path, capsys):
    draft = _passing_result("case-1", "不做生命周期分析")
    draft["label_status"] = "draft"
    evaluation, dataset = _write_evidence(tmp_path, [draft])

    result = main(["--eval", str(evaluation), "--dataset", str(dataset)])

    output = capsys.readouterr().out
    assert result == 1
    assert "reviewed cases: 0" in output
    assert "reviewed cases must be greater than 0" in output
    assert "reviewed semantic-status accuracy: n/a (0/0)" in output


def test_phase1_gate_blocks_zero_real_failures_without_fault_injection_evidence(tmp_path, capsys):
    evaluation, dataset = _write_evidence(tmp_path, _all_ok_results())

    result = main(["--eval", str(evaluation), "--dataset", str(dataset)])

    output = capsys.readouterr().out
    assert result == 1
    assert "PHASE1_GATE=BLOCKED" in output
    assert "fail-closed: n/a (0/0); required 100%" in output
    assert "fail_closed_source=none" in output


def test_phase1_gate_passes_zero_real_failures_with_four_case_fault_evidence(tmp_path, capsys):
    evaluation, dataset = _write_evidence(tmp_path, _all_ok_results())
    fault_evidence = tmp_path / "fault-evidence.json"
    fault_evidence.write_text(json.dumps(_fault_injection_payload()), encoding="utf-8")

    result = main([
        "--eval", str(evaluation), "--dataset", str(dataset),
        "--fault-injection-evidence", str(fault_evidence),
    ])

    output = capsys.readouterr().out
    assert result == 0
    assert "PHASE1_GATE=PASS" in output
    assert "fail-closed: n/a (0/0); fault-injection 100.00% (4/4); required 100%" in output
    assert "fail_closed_source=fault_injection" in output


def test_phase1_gate_reports_first_attempt_and_repair_separately(tmp_path, capsys):
    results = _all_ok_results()
    results[0]["observation"]["repair"] = {
        "attempted": True,
        "initial_status": "schema_invalid",
        "recovered": True,
        "final_status": "ok",
    }
    evaluation, dataset = _write_evidence(tmp_path, results)
    fault_evidence = tmp_path / "fault-evidence.json"
    fault_evidence.write_text(json.dumps(_fault_injection_payload()), encoding="utf-8")

    result = main([
        "--eval", str(evaluation), "--dataset", str(dataset),
        "--fault-injection-evidence", str(fault_evidence),
    ])

    output = capsys.readouterr().out
    assert result == 0
    assert "schema valid: 100.00% (18/18)" in output
    assert "initial schema valid before repair: 94.44% (17/18)" in output
    assert "validation repair recovered: 100.00% (1/1)" in output


def test_phase1_gate_blocks_zero_real_failures_with_failed_fault_evidence(tmp_path, capsys):
    evaluation, dataset = _write_evidence(tmp_path, _all_ok_results())
    fault_evidence = tmp_path / "fault-evidence.json"
    fault_evidence.write_text(
        json.dumps(_fault_injection_payload(failed_case="quota_blocked")),
        encoding="utf-8",
    )

    result = main([
        "--eval", str(evaluation), "--dataset", str(dataset),
        "--fault-injection-evidence", str(fault_evidence),
    ])

    output = capsys.readouterr().out
    assert result == 1
    assert "PHASE1_GATE=BLOCKED" in output
    assert "fail_closed_source=none" in output
    assert "fault-injection evidence error:" in output
