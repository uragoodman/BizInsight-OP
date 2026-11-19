"""Collect small-sample source-need and retrieval metrics.

This is the minimal v1.2 calculator used before expanding the dataset. It does
not call Mem0 or an LLM: intent cases run the real Resolver, and retrieval
metrics are computed from the existing v4 real-result JSON.

Usage:
  python -m evaluation.metrics_collect \
    --cases evaluation/datasets/metric_cases_v1.jsonl \
    --retrieval-result docs/testing/personalization_recall_20261102_v4/result_0_55.json \
    --json-out docs/testing/metric_seed_20261102/result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from app.context.turn_resolver import (
    detect_preference_source_needs,
    resolve_preference_context,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True,
            text=True, check=True,
        ).stdout.strip())
    except Exception:
        commit, dirty = None, None
    return {"code_commit": commit or None, "worktree_dirty": dirty}


def _load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_cases(cases: list[dict[str, Any]], schema_path: Path) -> list[str]:
    """Validate cases against the JSON schema; fall back to required fields."""
    errors: list[str] = []
    try:
        from jsonschema import Draft202012Validator

        schema = _load_schema(schema_path)
        case_schema = schema["definitions"]["case"]
        validator = Draft202012Validator(case_schema)
        for index, case in enumerate(cases):
            for error in validator.iter_errors(case):
                errors.append(f"case[{index}] {case.get('case_id', '?')}: {error.message}")
        return errors
    except ImportError:
        required = (
            "case_id", "category", "input", "expected_mode", "expected_labels",
            "expected_types", "expected_keys", "expected_fields",
        )
        for index, case in enumerate(cases):
            missing = [key for key in required if key not in case]
            if missing:
                errors.append(f"case[{index}]: missing {', '.join(missing)}")
        return errors
    except Exception as exc:  # noqa: BLE001 - schema failure must be reported, not hidden
        return [f"schema validation failed: {type(exc).__name__}: {exc}"]


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _sorted_map(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): sorted(str(item) for item in (values or []))
        for key, values in sorted(value.items())
    }


def _actual_intent(case: dict[str, Any]) -> dict[str, Any]:
    state = {
        "user_message": case["input"],
        "chat_session_id": "metric-seed",
        "context_compilation": {"known_slots": {}},
    }
    needs = detect_preference_source_needs(state)
    resolved = resolve_preference_context(state, provider=None)
    status = str(resolved.get("status") or "unknown")
    decision = resolved.get("intent_decision") or {}
    actual_types = sorted({kind for need in needs for kind in (need.get("memory_types") or [])})
    blocked_categories = {
        key: value
        for need in needs
        for key, value in (need.get("blocked_categories") or {}).items()
    }
    if not blocked_categories:
        blocked_categories = dict(decision.get("blocked_categories") or {})
    actual_labels: set[str] = set()
    for need in needs:
        triggers = set(need.get("trigger_ids") or [])
        types = set(need.get("memory_types") or [])
        if "P1" in triggers:
            if "presentation" in types:
                actual_labels.add("P1_presentation")
            if "analysis_method" in types:
                actual_labels.add("P1_analysis")
            if "scope_default" in types:
                actual_labels.add("P1_scope")
        if "P2" in triggers:
            actual_labels.add("P2_report")
        if "preference_management" in (need.get("other_intents") or []):
            actual_labels.add("P3_management")
    if not needs:
        if "preference_management" in (decision.get("other_intents") or []):
            actual_labels.add("P3_management")
        elif status == "registry_lookup_required":
            actual_labels.add("registry_lookup")
        else:
            actual_labels.add("no_retrieval")
    elif "preference_management" in (decision.get("other_intents") or []):
        actual_labels.add("P3_management")
    lookup = resolved.get("lookup_request")
    return {
        "status": status,
        "labels": sorted(actual_labels),
        "need_count": len(needs),
        "types": actual_types,
        "keys": _sorted_map({
            kind: sorted({key for need in needs for key in (need.get("allowed_keys") or {}).get(kind, [])})
            for kind in actual_types
        }),
        "fields": _sorted_map({
            kind: sorted({field for need in needs for field in (need.get("allowed_fields") or {}).get(kind, [])})
            for kind in actual_types
        }),
        "blocked_categories": dict(sorted(blocked_categories.items())),
        "excluded_values": _sorted_map({
            path: sorted({value for need in needs for value in (need.get("excluded_values") or {}).get(path, [])})
            for need in needs
            for path in (need.get("excluded_values") or {})
        }),
        "negative_query_params": _sorted_map({
            path: sorted({value for need in needs for value in (need.get("negative_query_params") or {}).get(path, [])})
            for need in needs
            for path in (need.get("negative_query_params") or {})
        }),
        "trigger_ids": sorted({trigger for need in needs for trigger in (need.get("trigger_ids") or [])}),
        "lookup": lookup,
        "rules_version": next((str(need.get("rules_version") or "") for need in needs if need.get("rules_version")), ""),
    }


def _mode_matches(expected: str, actual: dict[str, Any]) -> bool:
    if expected == "retrieval":
        return int(actual["need_count"]) > 0
    if expected == "registry_lookup":
        return actual["status"] == "registry_lookup_required"
    return int(actual["need_count"]) == 0 and actual["status"] != "registry_lookup_required"


def _lookup_matches(expected: Any, actual: Any) -> bool:
    if expected in (None, {}):
        return actual in (None, {})
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return False
    for key, value in expected.items():
        if actual.get(key) != value:
            return False
    return True


def _evaluate_intent(case: dict[str, Any]) -> dict[str, Any]:
    actual = _actual_intent(case)
    expected_types = sorted(case.get("expected_types") or [])
    expected_keys = _sorted_map(case.get("expected_keys") or {})
    expected_fields = _sorted_map(case.get("expected_fields") or {})
    expected_blocked = dict(sorted((case.get("expected_blocked_categories") or {}).items()))
    expected_excluded = _sorted_map(case.get("expected_excluded_values") or {})
    expected_negative = _sorted_map(case.get("expected_negative_query_params") or {})
    expected_triggers = sorted(case.get("expected_trigger_ids") or [])
    checks = {
        "mode": _mode_matches(str(case.get("expected_mode")), actual),
        "labels": sorted(case.get("expected_labels") or []) == actual["labels"],
        "types": expected_types == actual["types"],
        "keys": expected_keys == actual["keys"],
        "fields": expected_fields == actual["fields"],
        "blocked_categories": expected_blocked == actual["blocked_categories"],
        "excluded_values": expected_excluded == actual["excluded_values"],
        "negative_query_params": expected_negative == actual["negative_query_params"],
        "trigger_ids": expected_triggers == actual["trigger_ids"],
        "lookup": _lookup_matches(case.get("expected_lookup"), actual["lookup"]),
    }
    return {
        "case_id": case["case_id"],
        "expected_labels": sorted(case.get("expected_labels") or []),
        "actual_labels": actual["labels"],
        "actual": actual,
        "checks": checks,
        "exact_match": all(checks.values()),
    }


def _intent_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({
        label
        for result in results
        for label in (result["expected_labels"] + result["actual_labels"])
    })
    per_label: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = fp = fn = 0
        for result in results:
            expected = label in result["expected_labels"]
            actual = label in result["actual_labels"]
            tp += int(expected and actual)
            fp += int(actual and not expected)
            fn += int(expected and not actual)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }
    exact = sum(1 for result in results if result["exact_match"])
    return {
        "cases": len(results),
        "exact_match": exact,
        "exact_match_rate": exact / len(results) if results else 0.0,
        "checks": {
            name: sum(1 for result in results if result["checks"][name])
            for name in ("mode", "labels", "types", "keys", "fields", "blocked_categories",
                         "excluded_values", "negative_query_params", "trigger_ids", "lookup")
        },
        "labels": per_label,
        "macro_f1": (
            sum(item["f1"] for item in per_label.values()) / len(per_label)
            if per_label else 0.0
        ),
        "micro": (
            sum(item["tp"] for item in per_label.values()),
            sum(item["fp"] for item in per_label.values()),
            sum(item["fn"] for item in per_label.values()),
        ),
    }


def _retrieval_metrics(
    path: Path | None,
    *,
    t3_denominator: int | None = None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    details = payload.get("details") or []
    query_hits = {1: 0, 3: 0, 5: 0}
    key_hits = {1: 0, 3: 0, 5: 0}
    query_mrr = 0.0
    key_mrr = 0.0
    expected_keys = 0
    for result in details:
        expected = set(result.get("expected") or [])
        actual = list(result.get("actual") or [])
        if not expected:
            continue
        for k in query_hits:
            if expected & set(actual[:k]):
                query_hits[k] += 1
        first_rank = next((i + 1 for i, key in enumerate(actual) if key in expected), None)
        query_mrr += 1 / first_rank if first_rank else 0.0
        for key in expected:
            expected_keys += 1
            rank = next((i + 1 for i, hit in enumerate(actual) if hit == key), None)
            for k in key_hits:
                if rank is not None and rank <= k:
                    key_hits[k] += 1
            key_mrr += 1 / rank if rank else 0.0
    count = len([item for item in details if item.get("expected")])
    leaks = payload.get("unauthorized_leaks") or []
    denominator = t3_denominator
    if denominator is None:
        denominator = payload.get("unauthorized_candidate_count")
    leak_count = len(leaks)
    return {
        "source": str(path),
        "queries": count,
        "expected_keys": expected_keys,
        "threshold": payload.get("threshold"),
        "rules_version": payload.get("rules_version"),
        "stage_note": "actual is S1 provider-filtered; accepted is S2; S0 is not recorded",
        "query_hit": {f"@{k}": query_hits[k] / count if count else 0.0 for k in query_hits},
        "query_mrr": query_mrr / count if count else 0.0,
        "key_hit": {f"@{k}": key_hits[k] / expected_keys if expected_keys else 0.0 for k in key_hits},
        "key_mrr": key_mrr / expected_keys if expected_keys else 0.0,
        "precision": payload.get("precision"),
        "recall": payload.get("recall"),
        "accepted_precision": payload.get("accepted_precision"),
        "accepted_recall": payload.get("accepted_recall"),
        "T4_precision_pass": payload.get("T4_precision_pass"),
        "T3_rejection_pass": payload.get("T3_rejection_pass"),
        "unauthorized_leaks": leaks,
        "t3_denominator": denominator,
        "t3_leak_rate": leak_count / denominator if denominator else None,
        "t3_display": (
            f"{leak_count}/{denominator}" if denominator else f"{leak_count}/unknown"
        ),
        "retrieval_variants": payload.get("retrieval_variants"),
        "t3": payload.get("t3"),
        "s0_recorded": bool(payload.get("s0_recorded")),
        "value_accuracy_top1": payload.get("value_accuracy_top1"),
        "value_accuracy_topk": payload.get("value_accuracy_topk"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evaluation/datasets/metric_cases_v1.jsonl")
    parser.add_argument(
        "--schema", default="evaluation/metrics_schema_v1.json",
    )
    parser.add_argument(
        "--retrieval-result",
        default="docs/testing/personalization_recall_20261102_v4/result_0_55.json",
    )
    parser.add_argument("--dataset-version", default="metric-seed-v1")
    parser.add_argument(
        "--t3-denominator", type=int, default=None,
        help="number of unauthorized/expired/inactive candidates in the retrieval evaluation",
    )
    parser.add_argument("--json-out", default="")
    parser.add_argument("--intent-only", action="store_true")
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    schema_errors = _validate_cases(cases, Path(args.schema))
    intent_results = [
        _evaluate_intent(case) for case in cases if case.get("category") == "intent"
    ]
    intent = _intent_summary(intent_results)
    retrieval_path = (
        Path(args.retrieval_result)
        if args.retrieval_result and not args.intent_only else None
    )
    retrieval = _retrieval_metrics(
        retrieval_path,
        t3_denominator=args.t3_denominator,
    )
    metadata = {
        "dataset_version": args.dataset_version,
        "cases_sha256": _sha256(Path(args.cases)),
        "schema_sha256": _sha256(Path(args.schema)),
        "retrieval_sha256": _sha256(retrieval_path) if retrieval_path else None,
        **_git_metadata(),
    }
    gates = {
        "schema_valid": not schema_errors,
        "intent_exact_match": intent["exact_match_rate"] == 1.0,
    }
    if not args.intent_only:
        gates.update({
            "retrieval_present": retrieval is not None,
            "t4_precision_pass": bool(retrieval and retrieval.get("T4_precision_pass")),
            "t3_denominator_present": bool(retrieval and retrieval.get("t3_denominator")),
            "t3_no_leaks": bool(retrieval and not retrieval.get("unauthorized_leaks")),
            "s0_recorded": bool(retrieval and retrieval.get("s0_recorded")),
        })
    report = {
        "schema_version": "metric-seed-v1",
        "cases_file": args.cases,
        "dataset_version": args.dataset_version,
        "rules_version": (
            retrieval.get("rules_version") if retrieval else None
        ) or next(
            (item["actual"].get("rules_version") for item in intent_results
             if item["actual"].get("rules_version")),
            None,
        ),
        "metadata": metadata,
        "schema_errors": schema_errors,
        "intent": intent,
        "intent_details": intent_results,
        "retrieval": retrieval,
        "retrieval_variants": retrieval.get("retrieval_variants") if retrieval else None,
        "t3": retrieval.get("t3") if retrieval else None,
        "gates": gates,
        "all_required_gates_pass": all(gates.values()),
        "s0_recorded": bool(retrieval and retrieval.get("s0_recorded")),
        "limitations": [
            "intent-only contract matrix; not a generalization sample" if args.intent_only else "seed dataset only; not a generalization sample",
            "S0 is a security-filtered raw candidate stage, not an unrestricted dump" if not args.intent_only else "intent-only mode does not run retrieval",
            "intent cases exercise the deterministic Resolver only",
            "T3 reports no leakage; unauthorized records are filtered before S0 by safety checks" if not args.intent_only else "no retrieval or T3 gate in intent-only mode",
        ],
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output + "\n", encoding="utf-8")
    return 0 if report["all_required_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
