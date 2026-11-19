"""Deterministic analyses over already-authorized metric time series."""

from __future__ import annotations

from datetime import date
from math import isfinite
from typing import Any


QUALITY_CHECK_IDS = frozenset({
    "zero_value_buckets",
    "negative_value_buckets",
    "missing_value_buckets",
})


def extract_query_payload(envelope: dict[str, Any]) -> dict[str, Any]:
    """Unwrap the stable DataQueryAgent result without guessing field semantics."""
    for key in ("trend_result", "statistics_result", "association_result", "metric_result", "contribution_result"):
        payload = envelope.get(key)
        if isinstance(payload, dict):
            return payload
    nested = envelope.get("result")
    if isinstance(nested, dict):
        return extract_query_payload(nested)
    return envelope


def normalize_points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize platform and seller trend point shapes without imputing values."""
    raw_points = payload.get("points")
    if not isinstance(raw_points, list):
        return []
    points: list[dict[str, Any]] = []
    for row in raw_points:
        if not isinstance(row, dict):
            continue
        bucket = row.get("bucket", row.get("metric_date"))
        if bucket is None:
            continue
        value = row.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            value = None
        elif not isfinite(float(value)):
            value = None
        points.append({"bucket": str(bucket)[:10], "value": value})
    return sorted(points, key=lambda item: item["bucket"])


def _validated_points(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], int]:
    """Strictly validate bucket identities and measurement types for analysis."""
    raw_points = payload.get("points")
    if not isinstance(raw_points, list):
        return [], ["points_not_a_list"], 0
    points: list[dict[str, Any]] = []
    errors: set[str] = set()
    seen: set[str] = set()
    for row in raw_points:
        if not isinstance(row, dict):
            errors.add("invalid_point")
            continue
        raw_bucket = row.get("bucket", row.get("metric_date"))
        if not isinstance(raw_bucket, (str, date)):
            errors.add("invalid_bucket")
            continue
        bucket = str(raw_bucket)[:10]
        try:
            date.fromisoformat(bucket)
        except ValueError:
            errors.add("invalid_bucket")
            continue
        if bucket in seen:
            errors.add("duplicate_bucket")
            continue
        seen.add(bucket)
        if "value" not in row:
            errors.add("missing_value_field")
            continue
        value = row["value"]
        try:
            valid_number = (
                value is None
                or (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and isfinite(float(value))
                )
            )
        except (TypeError, ValueError, OverflowError):
            valid_number = False
        if not valid_number:
            errors.add("invalid_value")
            continue
        points.append({"bucket": bucket, "value": float(value) if value is not None else None})
    return sorted(points, key=lambda item: item["bucket"]), sorted(errors), len(raw_points)


def check_series_quality(payload: dict[str, Any], check_id: str) -> dict[str, Any]:
    """Return an auditable bucket-level quality observation, not a causal claim."""
    if check_id not in QUALITY_CHECK_IDS:
        raise ValueError(f"Unsupported time-series quality check: {check_id}")
    points, validation_errors, raw_count = _validated_points(payload)
    if "points_not_a_list" in validation_errors:
        return {
            "status": "invalid_data", "check_id": check_id, "bucket_count": 0,
            "validation_errors": validation_errors,
        }
    if raw_count == 0:
        return {"status": "insufficient_data", "check_id": check_id, "bucket_count": 0}
    if validation_errors:
        return {
            "status": "invalid_data",
            "check_id": check_id,
            "bucket_count": raw_count,
            "validation_errors": validation_errors,
        }

    if check_id == "zero_value_buckets":
        matches = [point for point in points if point["value"] == 0]
    elif check_id == "negative_value_buckets":
        matches = [point for point in points if point["value"] is not None and point["value"] < 0]
    else:
        matches = [point for point in points if point["value"] is None]
    valued = sum(point["value"] is not None for point in points)
    return {
        "status": "ok",
        "check_id": check_id,
        "bucket_count": len(points),
        "valued_bucket_count": valued,
        "affected_bucket_count": len(matches),
        "affected_bucket_rate": round(len(matches) / len(points), 6),
        "examples": matches[:20],
        "limitations": [
            "检查范围仅覆盖查询返回的时间桶；未返回的底层明细无法据此判断。",
            "零值或负值是否异常取决于指标口径，本结果只报告分布，不判定数据错误。",
        ],
    }


def detect_series_anomalies_for_payload(
    payload: dict[str, Any],
    *,
    metric_id: str,
    unit: str,
    start_date: str,
    end_date: str,
    granularity: str,
    definition_version: str,
    source_ref: str,
) -> dict[str, Any]:
    """Run the existing rolling detector on a validated series envelope."""
    from app.analysis.contracts import MetricPoint, MetricSeries, PeriodSpec
    from app.analysis.statistical_anomaly import detect_series_anomalies

    if granularity not in {"day", "week", "month"}:
        return {"status": "unsupported_granularity", "anomalies": []}
    points, validation_errors, raw_count = _validated_points(payload)
    if validation_errors:
        return {
            "status": "invalid_data",
            "numeric_bucket_count": sum(point["value"] is not None for point in points),
            "bucket_count": raw_count,
            "validation_errors": validation_errors,
            "anomalies": [],
        }
    numeric_count = sum(point["value"] is not None for point in points)
    if numeric_count < 8:
        return {
            "status": "insufficient_data",
            "numeric_bucket_count": numeric_count,
            "minimum_required": 8,
            "anomalies": [],
        }
    series = MetricSeries(
        metric_name=metric_id,
        unit=unit or "number",
        period=PeriodSpec(
            start_date=date.fromisoformat(start_date),
            end_date=date.fromisoformat(end_date),
            granularity=granularity,
        ),
        points=[
            MetricPoint(bucket=date.fromisoformat(point["bucket"]), value=point["value"])
            for point in points
        ],
        source_refs=[source_ref or "authorized_metric_trend"],
        definition_version=definition_version,
    )
    findings = detect_series_anomalies(series)
    return {
        "status": "ok",
        "numeric_bucket_count": numeric_count,
        "detector": "rolling_median_mad_iqr_v1",
        "anomaly_count": len(findings),
        "anomalies": [finding.model_dump(mode="json") for finding in findings],
        "limitations": [
            "异常点由历史滚动基线与稳健统计阈值识别，仅表示时间序列异常，不代表业务原因或因果关系。",
        ],
    }
