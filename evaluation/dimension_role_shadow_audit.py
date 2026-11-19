"""Offline P4 metrics for privacy-bounded Semantic Compiler shadow records.

This module is report-only: it never calls an LLM, changes a route, or writes
back to application configuration. Human annotations are accepted separately
from the immutable runtime evidence and only through a small allowlist.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.context.shadow_audit import (
    _default_audit_db_path,
    validate_audit_record,
)

STRATUM_FIELDS = ("dimension", "operation", "risk_tier", "rules_version")
ANNOTATION_FIELDS = {
    "audit_id", "human_review_status", "reviewed_label", "reviewed_span_labels",
    "contract_review_status", "wrong_contract", "contract_review_source",
    "execution_review_status", "wrong_execution", "execution_review_source",
}


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """Return a two-sided Wilson 95% interval; None when the denominator is 0."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson counts require 0 <= successes <= total")
    if total <= 0:
        return None
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) / total) + z2 / (4.0 * total * total)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _metric(successes: int, total: int, *, min_sample_size: int, unit: str) -> dict[str, Any]:
    interval = wilson_interval(successes, total)
    return {
        "value": round(successes / total, 6) if total else None,
        "numerator": successes,
        "denominator": total,
        "sample_count": total,
        "unit": unit,
        "confidence_interval_95": (
            [round(interval[0], 6), round(interval[1], 6)] if interval is not None else None
        ),
        "status": "sufficient" if total >= min_sample_size else "sample_insufficient",
        "minimum_sample_size": min_sample_size,
    }


def deduplicate_records(
    records: Iterable[Any],
) -> tuple[list[dict[str, Any]], int, int]:
    """Keep first valid records and count duplicate or malformed input rows."""
    unique: dict[str, dict[str, Any]] = {}
    duplicates = 0
    invalid = 0
    for value in records:
        if not isinstance(value, Mapping) or not validate_audit_record(value):
            invalid += 1
            continue
        key = str(value.get("audit_id") or value.get("trace_id_hash") or "")
        if not key:
            invalid += 1
            continue
        if key in unique:
            duplicates += 1
            continue
        unique[key] = dict(value)
    return list(unique.values()), duplicates, invalid


def merge_annotations(
    records: Iterable[Mapping[str, Any]],
    annotations: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply a strict review sidecar without allowing it to rewrite evidence."""
    unique, _duplicates, invalid = deduplicate_records(records)
    if invalid:
        raise ValueError("audit input contains invalid records")
    by_id = {str(item["audit_id"]): item for item in unique}
    seen_annotation_ids: set[str] = set()
    for annotation in annotations:
        if not isinstance(annotation, Mapping) or set(annotation) - ANNOTATION_FIELDS:
            raise ValueError("review annotation contains unsupported fields")
        audit_id = str(annotation.get("audit_id") or "")
        if audit_id in seen_annotation_ids:
            raise ValueError("review annotation contains a duplicate audit_id")
        seen_annotation_ids.add(audit_id)
        if audit_id not in by_id:
            raise ValueError("review annotation references an unknown audit_id")
        record = by_id[audit_id]
        if "human_review_status" in annotation:
            status = annotation["human_review_status"]
            if not isinstance(status, str) or status not in {"pending", "reviewed", "adjudicated"}:
                raise ValueError("invalid human review status")
            record["human_review_status"] = status
        if "reviewed_label" in annotation:
            label = annotation["reviewed_label"]
            if label is not None and (not isinstance(label, str) or label not in {"fp", "not_fp", "unclear"}):
                raise ValueError("invalid reviewed label")
            record["reviewed_label"] = label
        if "reviewed_span_labels" in annotation:
            labels = annotation["reviewed_span_labels"]
            if not isinstance(labels, Mapping) or any(
                span_id not in record["rule_roles"]
                or not isinstance(label, str)
                or label not in {"fp", "not_fp", "unclear"}
                for span_id, label in labels.items()
            ):
                raise ValueError("invalid rule-role span review labels")
            record["reviewed_span_labels"] = dict(labels)
        if (
            ("reviewed_label" in annotation and annotation.get("reviewed_label") is not None)
            or ("reviewed_span_labels" in annotation and bool(annotation.get("reviewed_span_labels")))
        ) and record["human_review_status"] not in {"reviewed", "adjudicated"}:
            raise ValueError("human labels require reviewed or adjudicated status")
        for kind in ("contract", "execution"):
            status_key = f"{kind}_review_status"
            result_key = "wrong_contract" if kind == "contract" else "wrong_execution"
            source_key = f"{kind}_review_source"
            if status_key in annotation:
                status = annotation[status_key]
                if not isinstance(status, str) or status not in {"pending", "reviewed", "adjudicated"}:
                    raise ValueError(f"invalid {kind} review status")
                record[status_key] = status
            if result_key in annotation:
                result = annotation[result_key]
                if result is not None and not isinstance(result, bool):
                    raise ValueError(f"invalid {kind} result")
                record[result_key] = result
            if source_key in annotation:
                source = annotation[source_key]
                allowed_sources = (
                    {"human", "calibration"}
                    if kind == "contract" else {"human_gold", "executable_validator"}
                )
                if not isinstance(source, str) or source not in allowed_sources:
                    raise ValueError(f"invalid {kind} review source")
                record[source_key] = source
            if record[result_key] is not None:
                allowed_sources = (
                    {"human", "calibration"}
                    if kind == "contract" else {"human_gold", "executable_validator"}
                )
                if (
                    record[status_key] not in {"reviewed", "adjudicated"}
                    or record[source_key] not in allowed_sources
                    or (kind == "execution" and record.get("actual_execution") is not True)
                ):
                    raise ValueError(f"{kind} error marker lacks an eligible review source")
        if not validate_audit_record(record):
            raise ValueError("review annotation produced an invalid audit record")
    return list(by_id.values())


def _dimensions(record: Mapping[str, Any]) -> dict[str, str]:
    value = record.get("span_dimensions") or {}
    return dict(value) if isinstance(value, Mapping) else {}


def _base_stratum(record: Mapping[str, Any], dimension: str) -> dict[str, str]:
    return {
        "dimension": dimension,
        "operation": str(record.get("operation") or "unknown"),
        "risk_tier": str(record.get("risk_tier") or "unknown"),
        "rules_version": str(record.get("rules_version") or "unknown"),
    }


def _stratum_key(stratum: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(stratum.get(key) or "unknown") for key in STRATUM_FIELDS)  # type: ignore[return-value]


def _add_label_counts(target: dict[str, int], label: str) -> None:
    target["reviewed_fp"] += int(label == "fp")
    target["reviewed_binary"] += int(label in {"fp", "not_fp"})
    target["reviewed_unclear"] += int(label == "unclear")


def _review_labels_for_dimension(record: Mapping[str, Any], dimension: str) -> list[tuple[str, bool]]:
    """Return rule-role review decisions and their closed-confidence flag."""
    if record.get("human_review_status") not in {"reviewed", "adjudicated"}:
        return []
    confidence = record.get("span_confidence") or {}
    dimensions = _dimensions(record)
    span_labels = record.get("reviewed_span_labels") or {}
    results: list[tuple[str, bool]] = []
    if isinstance(span_labels, Mapping) and span_labels:
        for span_id, label in span_labels.items():
            if (dimension == "__all__" or dimensions.get(str(span_id)) == dimension) and label in {"fp", "not_fp", "unclear"}:
                results.append((str(label), isinstance(confidence, Mapping) and confidence.get(span_id) == "closed"))
        return results
    label = record.get("reviewed_label")
    if label not in {"fp", "not_fp", "unclear"}:
        return []
    rule_role_ids = list((record.get("rule_roles") or {}).keys())
    if len(rule_role_ids) != 1:
        return []
    dims = set(dimensions.values()) or {"other"}
    if dimension != "__all__" and dimension not in dims:
        return []
    # Record-level decisions cannot be attributed to one span when a record has
    # several mentions. Only the single-span case feeds high-confidence FP.
    high = len(rule_role_ids) == 1 and any(
        span_id in rule_role_ids
        and (dimension == "__all__" or dimensions.get(span_id) == dimension)
        and isinstance(confidence, Mapping)
        and confidence.get(span_id) == "closed"
        for span_id in rule_role_ids
    )
    return [(str(label), high)]


def _new_counts() -> dict[str, Any]:
    return {
        "records": set(),
        "comparable_requests": set(),
        "disagree_requests": set(),
        "compared_spans": 0,
        "disagree_spans": 0,
        "reviewed_fp": 0,
        "reviewed_binary": 0,
        "reviewed_unclear": 0,
        "high_conf_fp": 0,
        "high_conf_binary": 0,
        "wrong_contract": 0,
        "reviewed_contracts": 0,
        "wrong_execution": 0,
        "reviewed_executions": 0,
    }


def _apply_review_and_error_counts(counts: dict[str, Any], record: Mapping[str, Any], dimension: str) -> None:
    for label, high_confidence in _review_labels_for_dimension(record, dimension):
        _add_label_counts(counts, label)
        if high_confidence and label in {"fp", "not_fp"}:
            counts["high_conf_fp"] += int(label == "fp")
            counts["high_conf_binary"] += 1

    if (
        record.get("contract_review_status") in {"reviewed", "adjudicated"}
        and record.get("contract_review_source") in {"human", "calibration"}
        and isinstance(record.get("wrong_contract"), bool)
    ):
        counts["reviewed_contracts"] += 1
        counts["wrong_contract"] += int(record["wrong_contract"] is True)
    if (
        record.get("actual_execution") is True
        and record.get("execution_review_status") in {"reviewed", "adjudicated"}
        and record.get("execution_review_source") in {"human_gold", "executable_validator"}
        and isinstance(record.get("wrong_execution"), bool)
    ):
        counts["reviewed_executions"] += 1
        counts["wrong_execution"] += int(record["wrong_execution"] is True)


def _summarize(counts: Mapping[str, Any], *, min_sample_size: int) -> dict[str, Any]:
    sampled_requests = len(counts["records"])
    comparable_requests = len(counts["comparable_requests"])
    request_disagreement = _metric(
        len(counts["disagree_requests"]), sampled_requests,
        min_sample_size=min_sample_size, unit="sampled_auto_executed_requests",
    )
    if comparable_requests == 0:
        request_disagreement["value"] = None
        request_disagreement["confidence_interval_95"] = None
        request_disagreement["status"] = "no_verifier_comparisons"
    elif comparable_requests < sampled_requests:
        # The task denominator is every sampled execution. Since missing
        # verifier output is not agreement, do not publish a numeric rate/CI
        # until all sampled requests have a comparable verifier result.
        request_disagreement["value"] = None
        request_disagreement["confidence_interval_95"] = None
        request_disagreement["status"] = "partial_verifier_coverage"
    comparable_request_rate = _metric(
        len(counts["disagree_requests"]), comparable_requests,
        min_sample_size=min_sample_size, unit="comparable_auto_executed_requests",
    )
    if comparable_requests == 0:
        comparable_request_rate["status"] = "no_verifier_comparisons"
    return {
        "sampled_auto_executed_requests": sampled_requests,
        "verifier_comparable_requests": comparable_requests,
        "verifier_noncomparable_requests": sampled_requests - comparable_requests,
        "verifier_disagreement_rate": request_disagreement,
        "verifier_disagreement_comparable_request_rate": comparable_request_rate,
        "verifier_disagreement_span_rate": _metric(
            counts["disagree_spans"], counts["compared_spans"],
            min_sample_size=min_sample_size, unit="comparable_spans",
        ),
        "reviewed_fp_rate": _metric(
            counts["reviewed_fp"], counts["reviewed_binary"],
            min_sample_size=min_sample_size, unit="binary_reviewed_decisions",
        ),
        "reviewed_unclear_count": counts["reviewed_unclear"],
        "high_confidence_reviewed_fp_rate": _metric(
            counts["high_conf_fp"], counts["high_conf_binary"],
            min_sample_size=min_sample_size, unit="high_confidence_binary_reviewed_spans",
        ),
        "wrong_contract_rate": _metric(
            counts["wrong_contract"], counts["reviewed_contracts"],
            min_sample_size=min_sample_size, unit="human_or_calibration_reviewed_contracts",
        ),
        "wrong_execution_rate": _metric(
            counts["wrong_execution"], counts["reviewed_executions"],
            min_sample_size=min_sample_size, unit="executed_samples_with_gold_or_validator",
        ),
        "comparable_span_count": counts["compared_spans"],
    }


def build_report(
    records: Iterable[Any],
    *,
    fp_budget: float = 0.05,
    min_sample_size: int = 30,
) -> dict[str, Any]:
    """Compute global and dimension × operation × risk × rules strata metrics."""
    if not 0.0 <= fp_budget <= 1.0:
        raise ValueError("fp_budget must be between 0 and 1")
    if min_sample_size < 1:
        raise ValueError("min_sample_size must be positive")
    source_rows = list(records)
    unique, duplicates, invalid_records = deduplicate_records(source_rows)
    global_counts = _new_counts()
    strata: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def counts_for(record: Mapping[str, Any], dimension: str) -> dict[str, Any]:
        key = _stratum_key(_base_stratum(record, dimension))
        return strata.setdefault(key, _new_counts())

    for record in unique:
        dims = _dimensions(record)
        record_dimensions = sorted(set(dims.values()) or {"other"})
        record_id = str(record.get("audit_id") or record.get("trace_id_hash"))
        global_counts["records"].add(record_id)
        for dimension in record_dimensions:
            counts_for(record, dimension)["records"].add(record_id)
            _apply_review_and_error_counts(counts_for(record, dimension), record, dimension)
        _apply_review_and_error_counts(global_counts, record, "__all__")

        rule_roles = record.get("rule_roles") or {}
        verifier_roles = record.get("verifier_roles") or {}
        compared_by_dimension: dict[str, int] = defaultdict(int)
        disagreed_by_dimension: dict[str, int] = defaultdict(int)
        if isinstance(rule_roles, Mapping) and isinstance(verifier_roles, Mapping):
            span_ids = set(rule_roles) & set(verifier_roles)
            for span_id in span_ids:
                dimension = dims.get(str(span_id), "other")
                mismatch = rule_roles[span_id] != verifier_roles[span_id]
                global_counts["compared_spans"] += 1
                global_counts["disagree_spans"] += int(mismatch)
                compared_by_dimension[dimension] += 1
                disagreed_by_dimension[dimension] += int(mismatch)
                target = counts_for(record, dimension)
                target["compared_spans"] += 1
                target["disagree_spans"] += int(mismatch)
        if compared_by_dimension:
            global_counts["comparable_requests"].add(record_id)
            if sum(disagreed_by_dimension.values()) > 0:
                global_counts["disagree_requests"].add(record_id)
            for dimension, count in compared_by_dimension.items():
                target = counts_for(record, dimension)
                target["comparable_requests"].add(record_id)
                if disagreed_by_dimension[dimension] > 0:
                    target["disagree_requests"].add(record_id)

    recommendations: list[dict[str, Any]] = []
    stratum_rows = []
    for key in sorted(strata):
        dimension, operation, risk_tier, rules_version = key
        counts = strata[key]
        metrics = _summarize(counts, min_sample_size=min_sample_size)
        row = {
            "dimension": dimension,
            "operation": operation,
            "risk_tier": risk_tier,
            "rules_version": rules_version,
            "record_level_error_label_attribution": (
                "request_level_label_repeated_for_each_included_dimension"
            ),
            **metrics,
        }
        stratum_rows.append(row)
        high_conf_metric = metrics["high_confidence_reviewed_fp_rate"]
        interval = high_conf_metric["confidence_interval_95"]
        if (
            invalid_records == 0
            and
            high_conf_metric["sample_count"] >= min_sample_size
            and interval is not None
            and interval[0] > fp_budget
        ):
            recommendations.append({
                "category": {
                    "dimension": dimension,
                    "operation": operation,
                    "risk_tier": risk_tier,
                    "rules_version": rules_version,
                },
                "sample_count": high_conf_metric["sample_count"],
                "reviewed_fp_rate": high_conf_metric["value"],
                "confidence_interval_95": interval,
                "fp_budget": fp_budget,
                "status": "downgrade_recommended",
                "suggested_action": "manual_review_before_rule_or_risk_tier_change",
                "automatic_change_applied": False,
            })

    return {
        "schema_version": "semantic-shadow-audit-report-v1",
        "metric_method_version": "p4-wilson-request-span-review-v2",
        "decision_mode": "report_only",
        "source_record_count": len(source_rows),
        "unique_record_count": len(unique),
        "duplicate_records_ignored": duplicates,
        "invalid_record_count": invalid_records,
        "integrity_status": "valid" if invalid_records == 0 else "invalid_records_rejected",
        "minimum_sample_size": min_sample_size,
        "fp_budget": fp_budget,
        "global": _summarize(global_counts, min_sample_size=min_sample_size),
        "strata": stratum_rows,
        "downgrade_recommendations": recommendations,
        "versions": sorted({
            (
                str(record.get("rules_version") or "unknown"),
                str(record.get("prompt_version") or "unknown"),
                str(record.get("model") or "unknown"),
            )
            for record in unique
        }),
        "notes": [
            "verifier_disagreement_rate uses all sampled auto-executed requests as denominator; its value and confidence interval are null when verifier comparison coverage is incomplete.",
            "verifier_disagreement_comparable_request_rate and span-level disagreement rate are separate diagnostics; requests without verifier output are not treated as agreements.",
            "Unclear human labels are reported separately and excluded from binary reviewed_fp_rate denominators.",
            "reviewed_fp_rate labels the executed rule-role decision; verifier-only roles are not eligible for FP labels.",
            "Contract and execution labels count only eligible human/calibration or human-gold/executable-validator sources.",
            "Dimension strata can overlap when one execution contains multiple dimensions; record-level contract/execution labels repeat in every included dimension stratum and do not imply dimension-level causality.",
            "Downgrade recommendations never alter runtime routes or risk tiers.",
            "Invalid input records are counted and excluded; any invalid record suppresses downgrade recommendations.",
        ],
    }


def load_records(path: str | Path) -> list[Any]:
    source = Path(path)
    if source.suffix.casefold() in {".sqlite", ".sqlite3", ".db"}:
        with sqlite3.connect(source, timeout=30.0) as connection:
            try:
                rows = connection.execute(
                    "SELECT record_json FROM semantic_shadow_audit ORDER BY sampled_at, audit_id"
                ).fetchall()
            except sqlite3.OperationalError as exc:
                if "no such table" in str(exc).lower():
                    raise ValueError(
                        "audit input database is missing semantic_shadow_audit table",
                    ) from exc
                raise
        return [json.loads(row[0]) for row in rows]
    records: list[Any] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def load_annotations(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    annotations: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            annotations.append(json.loads(line))
    return annotations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a report-only P4 semantic shadow audit summary")
    parser.add_argument("--input", default=str(_default_audit_db_path()))
    parser.add_argument("--annotations", default=None, help="JSONL review sidecar; must contain only annotation fields")
    parser.add_argument("--output", default="data/quality/semantic_shadow_audit_report.json")
    parser.add_argument("--fp-budget", type=float, default=0.05)
    parser.add_argument("--min-sample-size", type=int, default=30)
    args = parser.parse_args(argv)
    records = load_records(args.input) if Path(args.input).exists() else []
    if args.annotations:
        records = merge_annotations(records, load_annotations(args.annotations))
    report = build_report(records, fp_budget=args.fp_budget, min_sample_size=args.min_sample_size)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(target),
        "sampled_auto_executed_requests": report["global"]["sampled_auto_executed_requests"],
        "recommendation_count": len(report["downgrade_recommendations"]),
        "decision_mode": report["decision_mode"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
