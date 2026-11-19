"""Build a safe, source-verifiable summary of a raw Interpreter evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = "data/evaluations/semantic_interpreter_v2_full52_20261113.json"
DEFAULT_OUTPUT = "docs/design/evidence/semantic-interpreter-v2-full52-summary.json"
PARSE_DIAGNOSTIC_CASE_IDS = {
    "v341-required-011-11",
    "v341-labeled-alias-01",
    "v341-labeled-sort-basis-01",
}
PARSE_DIAGNOSTIC_MAX_CHARS = 500


def build_summary(source_path: Path) -> dict[str, Any]:
    raw_bytes = source_path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("input must be an evaluation object with a results array")

    safe_results: list[dict[str, Any]] = []
    for result in payload["results"]:
        if not isinstance(result, dict) or not isinstance(result.get("observation"), dict):
            raise ValueError("every result must contain an observation object")
        observation = result["observation"]
        safe_result = {
            "case_id": result.get("case_id"),
            "status": observation.get("status"),
            "reason_code": observation.get("reason_code"),
            "schema_errors": [
                {
                    "loc": [
                        part if isinstance(part, int) else str(part)[:100]
                        for part in (error.get("loc") or [])
                        if isinstance(part, (str, int))
                    ],
                    "type": str(error.get("type") or "validation_error")[:100],
                }
                for error in (observation.get("schema_errors") or [])
                if isinstance(error, Mapping)
            ],
            "issue_codes": sorted({
                str(code) for code in (observation.get("issue_codes") or [])
            }),
            "input_tokens": observation.get("input_tokens"),
            "output_tokens": observation.get("output_tokens"),
            "elapsed_ms": observation.get("elapsed_ms"),
            "first_token_ms": observation.get("first_token_ms"),
        }
        repair = observation.get("repair")
        repair = repair if isinstance(repair, Mapping) else {}
        if isinstance(repair, Mapping) and repair.get("attempted") is True:
            safe_result["repair"] = {
                "attempted": True,
                "prompt_version": str(repair.get("prompt_version") or "")[:100],
                "attempt_count": repair.get("attempt_count") if isinstance(repair.get("attempt_count"), int) else None,
                "recovered": bool(repair.get("recovered")),
                "initial_status": str(repair.get("initial_status") or "")[:100],
                "initial_reason_code": str(repair.get("initial_reason_code") or "")[:120],
                "initial_issue_codes": sorted({
                    str(code)[:120] for code in (repair.get("initial_issue_codes") or [])
                }),
                "initial_validator_diagnostics": [
                    {
                        "code": str(item.get("code") or "unknown")[:120],
                        "location": str(item.get("location") or "unknown")[:200],
                        "severity": str(item.get("severity") or "MISSING")[:40],
                    }
                    for item in (repair.get("initial_validator_diagnostics") or [])[:50]
                    if isinstance(item, Mapping)
                ],
                "final_status": str(repair.get("final_status") or "")[:100],
            }
        initial_status = (
            repair.get("initial_status") if repair.get("attempted") is True
            else observation.get("status")
        )
        if result.get("case_id") in PARSE_DIAGNOSTIC_CASE_IDS and initial_status == "parse_failure":
            diagnostic = (
                repair.get("initial_raw_output_diagnostic")
                if repair.get("attempted") is True else observation.get("raw_output_diagnostic")
            )
            if isinstance(diagnostic, Mapping) and diagnostic.get("capture_status") == "captured":
                excerpt = diagnostic.get("truncated_text")
                raw_chars = diagnostic.get("raw_output_chars")
                digest = diagnostic.get("raw_output_sha256")
                safe_result["raw_output_diagnostic"] = {
                    "capture_status": "captured",
                    "raw_output_chars": (
                        raw_chars if isinstance(raw_chars, int) and not isinstance(raw_chars, bool)
                        and raw_chars >= 0 else None
                    ),
                    "raw_output_sha256": (
                        digest if isinstance(digest, str) and len(digest) == 64
                        and all(ch in "0123456789abcdef" for ch in digest.casefold()) else None
                    ),
                    "truncated_text": (
                        excerpt[:PARSE_DIAGNOSTIC_MAX_CHARS] if isinstance(excerpt, str) else ""
                    ),
                    "truncated": bool(diagnostic.get("truncated")),
                }
            else:
                safe_result["raw_output_diagnostic"] = {"capture_status": "unavailable"}
        safe_results.append(safe_result)

    source_label = (
        source_path.resolve().relative_to(ROOT).as_posix()
        if source_path.resolve().is_relative_to(ROOT)
        else source_path.name
    )
    return {
        "source_file": source_label,
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "case_count": len(safe_results),
        "results": safe_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Raw evaluation JSON")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Safe summary JSON")
    args = parser.parse_args()
    source_path = Path(args.input)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    try:
        summary = build_summary(source_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "source_file": summary["source_file"],
        "source_sha256": summary["source_sha256"],
        "case_count": summary["case_count"],
        "output": str(output_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
