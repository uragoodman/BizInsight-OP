"""Replay saved Semantic Interpreter Candidates through deterministic repairs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG
from app.context.semantic_contract_v2 import SemanticContractCandidateV21
from app.context.semantic_contract_validator import validate_semantic_contract
from app.context.semantic_span_resolver import normalize_candidate


DEFAULT_EVAL = Path("data/evaluations/semantic_interpreter_v2_full52_20261113.json")
DEFAULT_OUTPUT = Path("docs/design/evidence/semantic-interpreter-v2-span-replay.json")

# This report measures only deterministic span, alias, and reference validity.
# Semantic routing/interpretation issues do not affect the rate.
SPAN_REFERENCE_ISSUE_CODES = frozenset({
    "argument_concept_reference_mismatch",
    "argument_evidence_outside_operation_clause",
    "argument_negation_cue_outside_source_clause",
    "argument_negation_evidence_outside_source_clause",
    "argument_negation_target_clause_mismatch",
    "empty_inherited_context_reference",
    "empty_text_has_clauses",
    "exact_alias_not_in_catalog",
    "invalid_argument_path",
    "invalid_clause_partition",
    "invalid_context_mention_reference",
    "missing_clause_partition",
    "missing_target_reference",
    "mention_outside_owned_clause",
    "negation_cue_outside_owned_clause",
    "negative_operation_source_missing",
    "negative_target_outside_owned_clause",
    "operation_evidence_outside_owned_clause",
    "operation_outside_owned_clause",
    "role_clause_reference_mismatch",
    "role_concept_reference_mismatch",
    "span_out_of_range",
    "span_surface_mismatch",
    "synonym_not_in_catalog",
    "target_concept_reference_mismatch",
    "unknown_clause_reference",
    "unknown_concept_reference",
    "unknown_intent_reference",
    "unknown_mention_reference",
    "unknown_role_reference",
    "unreferenced_inherited_mention",
    "untrusted_inherited_context_reference",
})
REPORT_ISSUE_CODES = (
    "invalid_clause_partition",
    "span_surface_mismatch",
    "span_out_of_range",
    "exact_alias_not_in_catalog",
    "synonym_not_in_catalog",
)
SEMANTIC_CONFLICT_CODE = "positive_negative_operation_conflict"


def _issue_lists(candidate: SemanticContractCandidateV21, source_text: str) -> tuple[list[str], list[str]]:
    result = validate_semantic_contract(
        candidate,
        source_text=source_text,
        catalog=DEFAULT_SEMANTIC_CATALOG,
    )
    issues = [item.code for item in result.issues]
    audit_issues = [item.code for item in result.audit_issues]
    return issues, audit_issues


def _codes_and_count(issues: list[str], audit_issues: list[str]) -> tuple[list[str], list[str]]:
    return sorted(set(issues)), sorted(set([*issues, *audit_issues]))


def _rate(valid: int, total: int) -> float:
    return round(valid / total, 6) if total else 0.0


def replay(eval_path: Path) -> dict[str, Any]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("evaluation input must contain a results array")

    rows: list[dict[str, Any]] = []
    before_issue_counts: Counter[str] = Counter()
    after_issue_counts: Counter[str] = Counter()
    before_valid = 0
    after_valid = 0
    candidate_count = 0
    normalization_totals: Counter[str] = Counter()
    schema_invalid_case_ids: list[str] = []
    schema_issue_counts: Counter[str] = Counter()
    unresolved_surface_cases: list[dict[str, Any]] = []
    normalization_failure_cases: list[dict[str, str]] = []
    new_semantic_conflict_cases: list[str] = []

    for record in payload["results"]:
        observation = record.get("observation") or {}
        raw_candidate = observation.get("candidate")
        if not isinstance(raw_candidate, dict):
            continue
        candidate_count += 1
        case_id = str(record.get("case_id") or "unknown-case")
        source_text = record.get("text")
        if not isinstance(source_text, str):
            schema_invalid_case_ids.append(case_id)
            rows.append({
                "case_id": case_id,
                "before_issue_codes": ["source_text_missing"],
                "after_issue_codes": ["source_text_missing"],
            })
            continue

        try:
            before_candidate_json = json.dumps(raw_candidate, ensure_ascii=False, separators=(",", ":"))
            before_candidate = SemanticContractCandidateV21.model_validate_json(before_candidate_json)
        except Exception as exc:  # the stored candidate may predate the current schema
            schema_invalid_case_ids.append(case_id)
            error_rows = exc.errors() if hasattr(exc, "errors") else []
            error_codes = sorted({str(item.get("type") or "validation_error") for item in error_rows})
            schema_issue_counts.update(error_codes or ["candidate_schema_invalid"])
            rows.append({
                "case_id": case_id,
                "before_issue_codes": ["candidate_schema_invalid"],
                "after_issue_codes": ["candidate_schema_invalid"],
                "schema_issue_codes": error_codes or ["candidate_schema_invalid"],
            })
            continue

        before_issues, before_audit = _issue_lists(before_candidate, source_text)
        before_codes = sorted(set(before_issues).intersection(SPAN_REFERENCE_ISSUE_CODES))
        before_issue_counts.update(before_codes)
        _before_codes, before_all_codes = _codes_and_count(before_issues, before_audit)
        if not before_codes:
            before_valid += 1

        try:
            normalized_dict, resolution = normalize_candidate(
                before_candidate.model_dump(mode="json"),
                source_text,
                DEFAULT_SEMANTIC_CATALOG,
            )
            after_candidate_json = json.dumps(normalized_dict, ensure_ascii=False, separators=(",", ":"))
            after_candidate = SemanticContractCandidateV21.model_validate_json(after_candidate_json)
        except Exception:  # keep the failed row visible instead of dropping it
            normalization_failure_cases.append({"case_id": case_id, "issue_code": "normalization_failed"})
            rows.append({
                "case_id": case_id,
                "before_issue_codes": before_codes,
                "after_issue_codes": ["normalization_failed"],
            })
            continue

        after_issues, after_audit = _issue_lists(after_candidate, source_text)
        after_codes = sorted(set(after_issues).intersection(SPAN_REFERENCE_ISSUE_CODES))
        after_issue_counts.update(after_codes)
        _after_codes, after_all_codes = _codes_and_count(after_issues, after_audit)
        if not after_codes:
            after_valid += 1
        normalization_totals.update({
            key: value
            for key, value in resolution.items()
            if isinstance(value, int) and not isinstance(value, bool)
        })

        unresolved_count = int(resolution.get("unresolved_surface_count", 0))
        if unresolved_count:
            unresolved_surface_cases.append({
                "case_id": case_id,
                "issue_code": "unresolved_surface",
                "count": unresolved_count,
            })
        if SEMANTIC_CONFLICT_CODE in after_all_codes and SEMANTIC_CONFLICT_CODE not in before_all_codes:
            new_semantic_conflict_cases.append(case_id)

        rows.append({
            "case_id": case_id,
            "before_issue_codes": before_codes,
            "after_issue_codes": after_codes,
        })

    denominator = candidate_count - len(schema_invalid_case_ids)
    issue_deltas = {
        code: {
            "before": before_issue_counts.get(code, 0),
            "after": after_issue_counts.get(code, 0),
            "delta": after_issue_counts.get(code, 0) - before_issue_counts.get(code, 0),
        }
        for code in REPORT_ISSUE_CODES
    }
    return {
        "source_eval_file": eval_path.name,
        "issue_count_unit": "candidate_cases_with_issue",
        "candidate_count": candidate_count,
        "schema_invalid_case_ids": sorted(schema_invalid_case_ids),
        "schema_issue_counts": dict(sorted(schema_issue_counts.items())),
        "span_reference_issue_codes": sorted(SPAN_REFERENCE_ISSUE_CODES),
        "span_reference_validity": {
            "before_valid": before_valid,
            "after_valid": after_valid,
            "denominator": denominator,
            "before_rate": _rate(before_valid, denominator),
            "after_rate": _rate(after_valid, denominator),
        },
        "span_reference_issue_counts_before": dict(sorted(before_issue_counts.items())),
        "span_reference_issue_counts_after": dict(sorted(after_issue_counts.items())),
        "normalization_counts": dict(sorted(normalization_totals.items())),
        "selected_issue_deltas": issue_deltas,
        "unresolved_surfaces": unresolved_surface_cases,
        "normalization_failures": normalization_failure_cases,
        "new_positive_negative_operation_conflicts": [
            {"case_id": case_id, "issue_code": SEMANTIC_CONFLICT_CODE}
            for case_id in sorted(new_semantic_conflict_cases)
        ],
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.eval.is_file():
        parser.error(f"evaluation file does not exist: {args.eval}")
    report = replay(args.eval)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "candidate_count": report["candidate_count"],
        "span_reference_validity": report["span_reference_validity"],
        "selected_issue_deltas": report["selected_issue_deltas"],
        "unresolved_surface_case_count": len(report["unresolved_surfaces"]),
        "normalization_failure_count": len(report["normalization_failures"]),
        "new_semantic_conflict_count": len(report["new_positive_negative_operation_conflicts"]),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
