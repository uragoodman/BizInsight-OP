from __future__ import annotations

import json

import pytest

from scripts.build_semantic_interpreter_fail_closed_evidence import (
    EXPECTED,
    build_evidence,
    run_fault_injection_case,
)


@pytest.mark.parametrize(
    ("case_id", "status", "reason_code"),
    [
        (case_id, status, reason_code)
        for case_id, (status, reason_code) in EXPECTED.items()
    ],
)
def test_fault_injection_scenarios_fail_closed(case_id, status, reason_code):
    result = run_fault_injection_case(case_id)

    assert result["status"] == status
    assert result["reason_code"] == reason_code
    assert result["exception_escaped"] is False
    assert result["contract_present"] is False or result["issue_codes"]
    assert result["passed"] is True


def test_fault_injection_evidence_contains_four_passing_cases():
    evidence = build_evidence()

    assert evidence["required_cases"] == 4
    assert evidence["passed"] is True
    assert len(evidence["cases"]) == 4
    assert all(item["passed"] for item in evidence["cases"])

    serialized = json.dumps(evidence, ensure_ascii=False)
    assert "API_KEY" not in serialized
    assert "raw_output" not in serialized
