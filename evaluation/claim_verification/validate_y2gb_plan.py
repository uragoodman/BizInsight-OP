"""Validate the Y2GB design plan and, optionally, a qualified review corpus.

Plan validation is local and makes no model, database, or network calls.
Corpus validation additionally checks that the deterministic probe does not
already detect a case that is intended to measure incremental LLM Critic value.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.claim_verification.deterministic_probe import probe

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "docs/testing/claim_verification_pilot_20261027/y1gb_y2gb_20261028/Y2GB_CASE_PLAN.jsonl"
DEFAULT_Y1 = ROOT / "docs/testing/claim_verification_pilot_20261027/unified_benchmark/repaired_detection_20261028/cases.jsonl"
FORBIDDEN_REVIEW_KEYS = {"mutation_type", "expected_outcome", "expected_truth", "changed_text", "provenance"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_plan(plan_path: Path, y1_path: Path) -> dict[str, Any]:
    plan = _read_jsonl(plan_path)
    y1 = _read_jsonl(y1_path)
    errors: list[str] = []
    ids = [row.get("case_id") for row in plan]
    if len(plan) != 23:
        errors.append(f"expected 23 plan entries, got {len(plan)}")
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        errors.append("case_id values must be non-empty and unique")
    y1_metrics = {(row.get("request_contract") or {}).get("metric_id") for row in y1}
    y2_metrics = {row.get("metric_id") for row in plan}
    overlap = sorted(metric for metric in y2_metrics & y1_metrics if metric)
    if overlap:
        errors.append("Y2 target metric overlaps Y1: " + ", ".join(overlap))
    missing = sorted({row.get("metric_id") for row in plan if not row.get("metric_id")})
    if missing:
        errors.append("plan rows missing metric_id")
    family_counts = dict(sorted(Counter(row.get("family", "<missing>") for row in plan).items()))
    source_failures = []
    for row in plan:
        source = ROOT / str(row.get("source_trace", ""))
        if not source.is_file():
            source_failures.append(f"{row.get('case_id')}: source trace missing: {source}")
            continue
        try:
            saved = json.loads(source.read_text(encoding="utf-8"))
        except Exception as exc:
            source_failures.append(f"{row.get('case_id')}: source trace unreadable ({type(exc).__name__})")
            continue
        if saved.get("case_id") != row.get("source_case_id"):
            source_failures.append(f"{row.get('case_id')}: source case id mismatch")
        if saved.get("run_id") != row.get("source_run_id"):
            source_failures.append(f"{row.get('case_id')}: source run id mismatch")
        observation = saved.get("observation") or {}
        if observation.get("metric_id") != row.get("metric_id"):
            source_failures.append(f"{row.get('case_id')}: source metric mismatch")
    errors.extend(source_failures)
    return {
        "valid": not errors,
        "plan_entries": len(plan),
        "target_metrics": sorted(metric for metric in y2_metrics if metric),
        "y1_metric_overlap": overlap,
        "families": family_counts,
        "source_refs_checked": len(plan) - len(source_failures),
        "errors": errors,
        "mode": "static_plan_integrity_only",
    }


def validate_corpus(corpus_path: Path, plan_path: Path,
                    qualification_ledger_path: Path | None = None) -> dict[str, Any]:
    corpus = _read_jsonl(corpus_path)
    plan = _read_jsonl(plan_path)
    if qualification_ledger_path is None:
        expected_ids = {row["case_id"] for row in plan}
    else:
        ledger = _read_jsonl(qualification_ledger_path)
        plan_ids = {row["case_id"] for row in plan}
        ledger_ids = [row.get("case_id") for row in ledger]
        if len(ledger_ids) != len(set(ledger_ids)) or set(ledger_ids) != plan_ids:
            return {"valid": False, "corpus_entries": len(corpus),
                    "deterministic_probe_status_counts": {},
                    "errors": ["qualification ledger must cover all plan IDs exactly once"],
                    "mode": "offline_corpus_preflight_no_provider"}
        invalid_status = [row.get("case_id") for row in ledger
                          if row.get("qualification_status") not in {"qualified", "blocked"}]
        expected_ids = {row["case_id"] for row in ledger if row.get("qualification_status") == "qualified"}
        if invalid_status:
            return {"valid": False, "corpus_entries": len(corpus),
                    "deterministic_probe_status_counts": {},
                    "errors": [f"qualification ledger has invalid status rows: {invalid_status}"],
                    "mode": "offline_corpus_preflight_no_provider"}
    actual_ids = {row.get("case_id") for row in corpus}
    errors: list[str] = []
    if len(corpus) != len(expected_ids) or actual_ids != expected_ids:
        errors.append(f"qualified corpus IDs must exactly match the qualified set ({len(expected_ids)} cases)")
    probe_counts: Counter[str] = Counter()
    for case in corpus:
        case_id = case.get("case_id")
        if case.get("expected") != "should_be_flagged":
            errors.append(f"{case_id}: positive challenge expected label must be should_be_flagged")
        if FORBIDDEN_REVIEW_KEYS & case.keys():
            errors.append(f"{case_id}: scoring labels must not be passed to the reviewer")
        contract = case.get("request_contract") or {}
        matching_metric = next((item.get("metric_id") for item in plan if item.get("case_id") == case_id), None)
        if contract.get("metric_id") != matching_metric:
            errors.append(f"{case_id}: request contract metric does not match the plan")
        trusted = [b for b in case.get("evidence_bundles", []) if isinstance(b, dict) and b.get("trusted") is True]
        if not trusted and not case.get("operational_context"):
            errors.append(f"{case_id}: requires trusted evidence or a validated operational stop context")
        result = probe(case)
        probe_counts[result.get("status", "unknown")] += 1
        if result.get("status") == "mismatch_found":
            errors.append(f"{case_id}: deterministic probe already catches a mismatch; not an incremental Critic case")
    return {
        "valid": not errors,
        "corpus_entries": len(corpus),
        "qualified_case_ids": sorted(expected_ids),
        "deterministic_probe_status_counts": dict(sorted(probe_counts.items())),
        "errors": errors,
        "mode": "offline_corpus_preflight_no_provider",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--y1-corpus", type=Path, default=DEFAULT_Y1)
    parser.add_argument("--corpus", type=Path, help="Optional qualified Y2 corpus; no reviewer calls are made")
    parser.add_argument("--qualification-ledger", type=Path,
                        help="All 23 source decisions; blocked cases are excluded from a partial corpus")
    args = parser.parse_args()
    result = validate_plan(args.plan, args.y1_corpus)
    if result["valid"] and args.corpus:
        result = validate_corpus(args.corpus, args.plan, args.qualification_ledger)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
