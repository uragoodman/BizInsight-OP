"""Run deterministic fault-injection cases through the shadow interpreter."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.context import semantic_interpreter as interpreter  # noqa: E402


DEFAULT_OUTPUT = "docs/design/evidence/semantic-interpreter-failclosed-20261114.json"
EXPECTED = {
    "timeout": ("unavailable", "timeout"),
    "invalid_json": ("parse_failure", "invalid_json"),
    "schema_invalid": ("schema_invalid", "candidate_schema_invalid"),
    "quota_blocked": ("unavailable", "quota_blocked"),
}


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        semantic_interpreter_mode="shadow",
        semantic_interpreter_send_raw_question=True,
        semantic_interpreter_shadow_sample_rate=1.0,
        semantic_interpreter_model="fault-injection-model",
        semantic_interpreter_timeout_seconds=10.0,
        semantic_interpreter_interaction_policy="stream_fast",
        semantic_interpreter_max_tokens=32,
        semantic_interpreter_span_resolution_enabled=True,
        semantic_interpreter_capture_raw_output=False,
        llm_standard_model="fault-injection-model",
        llm_model="fault-injection-model",
    )


def _fake_provider(case_id: str) -> Callable[..., Any]:
    def invoke(*_args: Any) -> Any:
        if case_id == "timeout":
            raise TimeoutError("injected timeout")
        if case_id == "invalid_json":
            return "{invalid json"
        if case_id == "schema_invalid":
            return {}
        if case_id == "quota_blocked":
            raise interpreter._InvocationFailure("unavailable", "quota_blocked")
        raise ValueError(f"Unknown fault-injection case: {case_id}")

    return invoke


def run_fault_injection_case(case_id: str) -> dict[str, Any]:
    if case_id not in EXPECTED:
        raise ValueError(f"Unknown fault-injection case: {case_id}")

    exception_escaped = False
    with patch.object(interpreter, "get_settings", _settings):
        try:
            observation = interpreter.interpret_semantics(
                "按购买频次给客户分层",
                turn_id=f"fault-injection-{case_id}",
                mode="shadow",
                model="fault-injection-model",
                invoke_model=_fake_provider(case_id),
            )
        except Exception as exc:  # defensive boundary: record, never leak error text
            exception_escaped = True
            observation = {
                "status": "unavailable",
                "reason_code": "uncaught_" + type(exc).__name__,
                "contract": None,
                "issue_codes": [],
            }

    expected_status, expected_reason = EXPECTED[case_id]
    status = observation.get("status")
    reason_code = observation.get("reason_code")
    issue_codes = observation.get("issue_codes") or []
    contract_present = observation.get("contract") is not None
    fail_closed = status in {"parse_failure", "schema_invalid", "validator_issues", "unavailable"}
    safe_result = not exception_escaped and (not contract_present or bool(issue_codes))
    passed = (
        status == expected_status
        and reason_code == expected_reason
        and fail_closed
        and safe_result
    )
    return {
        "case_id": case_id,
        "status": status,
        "reason_code": reason_code,
        "exception_escaped": exception_escaped,
        "contract_present": contract_present,
        "issue_codes": sorted(str(code) for code in issue_codes),
        "passed": passed,
    }


def build_evidence() -> dict[str, Any]:
    cases = [run_fault_injection_case(case_id) for case_id in EXPECTED]
    return {
        "evidence_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "generator": "scripts/build_semantic_interpreter_fail_closed_evidence.py",
        "source": "synthetic invoke_model fault injection; no provider request",
        "required_cases": len(EXPECTED),
        "passed": len(cases) == len(EXPECTED) and all(item["passed"] for item in cases),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Evidence JSON output path")
    args = parser.parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    evidence = build_evidence()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "required_cases": evidence["required_cases"],
        "passed": evidence["passed"],
        "cases": evidence["cases"],
    }, ensure_ascii=False, indent=2))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
