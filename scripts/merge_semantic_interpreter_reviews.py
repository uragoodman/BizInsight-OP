"""Validate human review decisions and keep reviewed metrics separate from draft."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.context.semantic_contract_v2 import OperationKind


REVIEWED_STATUSES = {"CONFIRMED", "AMBIGUOUS", "UNSUPPORTED", "MISSING", "CONFLICT"}
POLARITIES = {"POSITIVE", "NEGATIVE"}
OPERATION_KINDS = set(OperationKind.__args__)
REQUIRED_DECISION_FIELDS = {
    "case_id", "reviewer", "reviewed_at", "reviewed_status",
    "reviewed_operations", "reviewed_clarification_required", "notes",
}
OPERATION_FIELDS = {"kind", "target_concept_ids", "polarity"}
SCHEMA_VALID_STATUSES = {"ok", "validator_issues"}
FAIL_CLOSED_STATUSES = {"parse_failure", "schema_invalid", "validator_issues", "unavailable"}


def _read_decisions(path: Path, known_case_ids: set[str]) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decision = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid decision JSON at line {line_number}: {exc.msg}") from exc
        if not isinstance(decision, dict):
            raise ValueError(f"Decision at line {line_number} must be an object")
        missing = REQUIRED_DECISION_FIELDS - decision.keys()
        extra = decision.keys() - REQUIRED_DECISION_FIELDS
        if missing or extra:
            details = []
            if missing:
                details.append("missing fields " + ", ".join(sorted(missing)))
            if extra:
                details.append("unknown fields " + ", ".join(sorted(extra)))
            raise ValueError(f"Decision at line {line_number}: " + "; ".join(details))

        case_id = decision["case_id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Decision at line {line_number} has an invalid case_id")
        if case_id not in known_case_ids:
            raise ValueError(f"Decision at line {line_number} references unknown case_id {case_id!r}")
        if case_id in decisions:
            raise ValueError(f"Duplicate decision for case_id {case_id!r}")

        reviewer = decision["reviewer"]
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError(f"Decision for {case_id!r} requires a non-empty reviewer")
        reviewed_at = decision["reviewed_at"]
        if not isinstance(reviewed_at, str) or not reviewed_at.strip():
            raise ValueError(f"Decision for {case_id!r} requires reviewed_at")
        try:
            parsed_time = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Decision for {case_id!r} has an invalid reviewed_at timestamp") from exc
        if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
            raise ValueError(f"Decision for {case_id!r} reviewed_at must include a timezone")
        if not isinstance(decision["reviewed_status"], str) or decision["reviewed_status"] not in REVIEWED_STATUSES:
            raise ValueError(f"Decision for {case_id!r} has an invalid reviewed_status")
        if type(decision["reviewed_clarification_required"]) is not bool:
            raise ValueError(f"Decision for {case_id!r} requires a boolean reviewed_clarification_required")
        if not isinstance(decision["notes"], str):
            raise ValueError(f"Decision for {case_id!r} notes must be a string")

        operations = decision["reviewed_operations"]
        if not isinstance(operations, list):
            raise ValueError(f"Decision for {case_id!r} reviewed_operations must be an array")
        for index, operation in enumerate(operations):
            if not isinstance(operation, dict) or operation.keys() != OPERATION_FIELDS:
                raise ValueError(
                    f"Decision for {case_id!r} operation {index} must contain exactly "
                    "kind, target_concept_ids, and polarity"
                )
            if not isinstance(operation["kind"], str) or operation["kind"] not in OPERATION_KINDS:
                raise ValueError(f"Decision for {case_id!r} operation {index} has an invalid kind")
            targets = operation["target_concept_ids"]
            if (
                not isinstance(targets, list)
                or not targets
                or any(not isinstance(item, str) or not item.strip() for item in targets)
                or len(targets) != len(set(targets))
            ):
                raise ValueError(f"Decision for {case_id!r} operation {index} has invalid target_concept_ids")
            if not isinstance(operation["polarity"], str) or operation["polarity"] not in POLARITIES:
                raise ValueError(f"Decision for {case_id!r} operation {index} has an invalid polarity")

        decisions[case_id] = copy.deepcopy(decision)
    return decisions


def _schema_valid(result: dict[str, Any]) -> bool:
    observation = result.get("observation") or {}
    return observation.get("status") in SCHEMA_VALID_STATUSES


def _observed_operations(result: dict[str, Any]) -> list[dict[str, Any]]:
    observation = result.get("observation") or {}
    contract = observation.get("contract")
    if not isinstance(contract, dict) or not isinstance(contract.get("operations"), list):
        return []
    projected = []
    for operation in contract["operations"]:
        if not isinstance(operation, dict):
            continue
        projected.append({
            "kind": operation.get("kind"),
            "target_concept_ids": sorted(operation.get("target_concept_ids") or []),
            "polarity": operation.get("polarity"),
        })
    return projected


def _reviewed_operation_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    field_correct = 0
    field_total = 0
    exact_cases = 0
    for result in results:
        review = result["review"]
        expected = [
            {
                "kind": item["kind"],
                "target_concept_ids": sorted(item["target_concept_ids"]),
                "polarity": item["polarity"],
            }
            for item in review["reviewed_operations"]
        ]
        actual = _observed_operations(result)
        if expected == actual:
            exact_cases += 1
        for index in range(max(len(expected), len(actual))):
            expected_item = expected[index] if index < len(expected) else {}
            actual_item = actual[index] if index < len(actual) else {}
            for field in ("kind", "target_concept_ids", "polarity"):
                field_total += 1
                field_correct += expected_item.get(field) == actual_item.get(field)
    return {
        "operation_field_correct": field_correct,
        "operation_field_total": field_total,
        "operation_field_accuracy": field_correct / field_total if field_total else None,
        "operation_exact_match_count": exact_cases,
        "operation_exact_match_rate": exact_cases / len(results) if results else None,
    }


def _metrics(results: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reviewed = [item for item in results if item.get("label_status") == "reviewed"]
    draft = [item for item in results if item.get("label_status") == "draft"]

    draft_status_scores = []
    draft_field_correct = 0
    draft_field_total = 0
    for item in draft:
        expected_status = (item.get("expected") or {}).get("status")
        contract = (item.get("observation") or {}).get("contract")
        actual_status = contract.get("semantic_status") if isinstance(contract, dict) else None
        if expected_status is not None:
            draft_status_scores.append(expected_status == actual_status)
        scores = item.get("scores") or {}
        draft_field_correct += int(scores.get("field_correct") or 0)
        draft_field_total += int(scores.get("field_total") or 0)

    reviewed_status_scores = []
    for item in reviewed:
        reviewed_status = item["review"]["reviewed_status"]
        contract = (item.get("observation") or {}).get("contract")
        actual_status = contract.get("semantic_status") if isinstance(contract, dict) else None
        reviewed_status_scores.append(reviewed_status == actual_status)

    reviewed_operation_metrics = _reviewed_operation_metrics(reviewed)
    return {
        "draft": {
            "case_count": len(draft),
            "schema_valid_count": sum(_schema_valid(item) for item in draft),
            "schema_valid_rate": sum(_schema_valid(item) for item in draft) / len(draft) if draft else None,
            "status_scored_count": len(draft_status_scores),
            "status_accuracy": sum(draft_status_scores) / len(draft_status_scores) if draft_status_scores else None,
            "field_correct": draft_field_correct,
            "field_total": draft_field_total,
            "field_accuracy": draft_field_correct / draft_field_total if draft_field_total else None,
        },
        "reviewed": {
            "case_count": len(reviewed),
            "schema_valid_count": sum(_schema_valid(item) for item in reviewed),
            "schema_valid_rate": sum(_schema_valid(item) for item in reviewed) / len(reviewed) if reviewed else None,
            "status_scored_count": len(reviewed_status_scores),
            "status_accuracy": sum(reviewed_status_scores) / len(reviewed_status_scores) if reviewed_status_scores else None,
            **reviewed_operation_metrics,
            "clarification_labels_recorded": sum(
                "reviewed_clarification_required" in item["review"] for item in reviewed
            ),
            "clarification_prediction_scored_count": 0,
        },
    }


def merge_reviews(payload: dict[str, Any], decision_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("evaluation input must contain a results array")
    results = copy.deepcopy(payload["results"])
    by_case_id: dict[str, dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("every evaluation result must be an object")
        case_id = result.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in by_case_id:
            raise ValueError("evaluation results must have unique non-empty case_id values")
        by_case_id[case_id] = result

    unknown = set(decision_map) - set(by_case_id)
    if unknown:
        raise ValueError("decisions reference unknown case_id values: " + ", ".join(sorted(unknown)))

    for result in results:
        result.pop("review", None)
        result["label_status"] = "draft"
        decision = decision_map.get(result["case_id"])
        if decision is not None:
            result["label_status"] = "reviewed"
            result["review"] = decision

    merged = {key: copy.deepcopy(value) for key, value in payload.items() if key not in {"results", "review_metrics", "review_merge"}}
    merged["results"] = results
    merged["review_merge"] = {
        "protocol_version": 1,
        "decision_count": len(decision_map),
        "reviewer_count": len({decision["reviewer"] for decision in decision_map.values()}),
    }
    merged["review_metrics"] = _metrics(results, decision_map)
    return merged


def merge_files(evaluation_path: Path, decisions_path: Path) -> dict[str, Any]:
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("evaluation input must contain a results array")
    ids = [item.get("case_id") for item in payload["results"] if isinstance(item, dict)]
    if len(ids) != len(payload["results"]) or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("evaluation results must have non-empty case_id values")
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation results must have unique case_id values")
    decisions = _read_decisions(decisions_path, set(ids))
    return merge_reviews(payload, decisions)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, help="Raw evaluation JSON")
    parser.add_argument("--decisions", required=True, help="Human decisions JSONL")
    parser.add_argument("--output", required=True, help="Merged evaluation JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)
    evaluation_path = Path(args.evaluation)
    decisions_path = Path(args.decisions)
    output_path = Path(args.output)
    try:
        merged = merge_files(evaluation_path, decisions_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output": str(output_path),
        "review_metrics": merged["review_metrics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
