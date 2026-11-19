"""Build compact, committed evidence from a raw run and review decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.merge_semantic_interpreter_reviews import merge_files  # noqa: E402


DEFAULT_EVALUATION = "data/evaluations/semantic_interpreter_v3_5_full70_20261114.json"
DEFAULT_DECISIONS = "data/evaluations/semantic_interpreter_v3_5_codex_review_decisions_20261114.jsonl"
DEFAULT_OUTPUT = "docs/design/evidence/semantic-interpreter-v3.5-full70-reviewed.json"
DEFAULT_DECISIONS_OUTPUT = "docs/design/evidence/semantic-interpreter-v3.5-review-decisions.jsonl"
DECISION_FIELDS = (
    "case_id", "reviewer", "reviewed_at", "reviewed_status",
    "reviewed_operations", "reviewed_clarification_required", "notes",
)
SAFE_ENVIRONMENT_FIELDS = (
    "model", "interaction_policy", "transport", "thinking", "prompt_version",
    "prompt_sha256", "catalog_snapshot_hash", "provider_host", "max_tokens",
)


def _project_operations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    operations = []
    for operation in value:
        if not isinstance(operation, Mapping):
            continue
        operations.append({
            "kind": operation.get("kind"),
            "target_concept_ids": sorted(operation.get("target_concept_ids") or []),
            "polarity": operation.get("polarity"),
        })
    return operations


def build_reviewed_evidence(evaluation_path: Path, decisions_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_bytes = evaluation_path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    merged = merge_files(evaluation_path, decisions_path)
    meta = payload.get("meta") if isinstance(payload, Mapping) else None
    meta = meta if isinstance(meta, Mapping) else {}
    environment = meta.get("environment") if isinstance(meta.get("environment"), Mapping) else {}

    safe_results: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for result in merged.get("results", []):
        observation = result.get("observation") or {}
        contract = observation.get("contract") or {}
        review = result.get("review")
        safe_result = {
            "case_id": result.get("case_id"),
            "dataset": result.get("dataset"),
            "category": result.get("category"),
            "text": result.get("text"),
            "label_status": result.get("label_status", "draft"),
            "review_source": result.get("review_source"),
            "exception_escaped": bool(result.get("exception_escaped", False)),
            "observation": {
                "status": observation.get("status"),
                "reason_code": observation.get("reason_code"),
                "issue_codes": sorted(observation.get("issue_codes") or []),
                "contract": {
                    "semantic_status": contract.get("semantic_status"),
                    "operations": _project_operations(contract.get("operations")),
                } if isinstance(contract, Mapping) else None,
            },
        }
        if isinstance(review, Mapping):
            safe_result["review"] = {field: review[field] for field in DECISION_FIELDS}
            decisions.append({field: review[field] for field in DECISION_FIELDS})
        elif result.get("label_status") == "reviewed":
            raise ValueError(f"Missing reviewed decision for {result.get('case_id')!r}")
        safe_results.append(safe_result)

    decision_lines = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in decisions
    )
    safe_meta = {
        key: meta.get(key)
        for key in (
            "created_at_utc", "branch", "commit", "evaluation_source_snapshot_commit",
            "runtime_head_at_run", "runtime_worktree_dirty_at_run", "stop_reason",
            "result_count", "token_total_observed", "token_total_reserved", "dataset_sha256",
        )
        if key in meta
    }
    safe_meta["environment"] = {
        key: environment.get(key)
        for key in SAFE_ENVIRONMENT_FIELDS
        if key in environment
    }
    evidence = {
        "evidence_version": 1,
        "evidence_type": "reviewed_semantic_interpreter_evaluation",
        "source_raw_eval_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "committed_decisions_sha256": hashlib.sha256(decision_lines.encode("utf-8")).hexdigest(),
        "meta": safe_meta,
        "review_merge": merged.get("review_merge"),
        "review_metrics": merged.get("review_metrics"),
        "results": safe_results,
    }
    return evidence, decisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default=DEFAULT_EVALUATION)
    parser.add_argument("--decisions", default=DEFAULT_DECISIONS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--decisions-output", default=DEFAULT_DECISIONS_OUTPUT)
    args = parser.parse_args()
    paths = [Path(value) for value in (args.evaluation, args.decisions, args.output, args.decisions_output)]
    evaluation_path, decisions_path, output_path, decisions_output_path = [
        path if path.is_absolute() else ROOT / path for path in paths
    ]
    try:
        evidence, decisions = build_reviewed_evidence(evaluation_path, decisions_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        decisions_output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        decision_lines = "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in decisions
        )
        decisions_output_path.write_text(
            decision_lines,
            encoding="utf-8",
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "output": str(output_path),
        "decisions_output": str(decisions_output_path),
        "case_count": len(evidence["results"]),
        "source_raw_eval_sha256": evidence["source_raw_eval_sha256"],
        "committed_decisions_sha256": evidence["committed_decisions_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
