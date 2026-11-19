"""Consistency checks between immutable report facts and live attribution reads."""

from __future__ import annotations

import math
import hashlib
import json
from datetime import datetime, timezone

from app.analysis.contracts import PeriodicReport
from app.analysis.attribution import AttributionDataInsufficient


def validate_report_breakdown_totals(
    report: PeriodicReport,
    metric,
    baseline_period,
    current_total: float,
    baseline_total: float,
) -> None:
    """Fail closed when live breakdown totals/periods no longer match the report.

    Supplemental dimension queries read mutable source tables. They can only
    support attribution while their complete period totals reconcile to the
    immutable report snapshot.
    """
    comparison = metric.period_over_period
    if (
        comparison is None or metric.value is None
        or comparison.current_value is None or comparison.baseline_value is None
    ):
        raise AttributionDataInsufficient({"comparison": "report_totals_missing"})
    if (
        comparison.current_period is None
        or comparison.current_period.model_dump(mode="json") != report.period.model_dump(mode="json")
        or comparison.baseline_period is None
        or comparison.baseline_period.model_dump(mode="json") != baseline_period.model_dump(mode="json")
    ):
        raise AttributionDataInsufficient({"comparison": "report_period_mismatch"})

    if not math.isclose(current_total, float(metric.value), rel_tol=1e-8, abs_tol=0.02):
        raise AttributionDataInsufficient({"current": "report_snapshot_total_mismatch"})
    if not math.isclose(
        float(comparison.current_value), float(metric.value), rel_tol=1e-8, abs_tol=0.02,
    ):
        raise AttributionDataInsufficient({"comparison": "report_snapshot_current_value_mismatch"})
    if not math.isclose(
        baseline_total, float(comparison.baseline_value), rel_tol=1e-8, abs_tol=0.02,
    ):
        raise AttributionDataInsufficient({"baseline": "report_snapshot_total_mismatch"})
    if comparison.delta_value is not None and not math.isclose(
        current_total - baseline_total, float(comparison.delta_value), rel_tol=1e-8, abs_tol=0.03,
    ):
        raise AttributionDataInsufficient({"comparison": "report_snapshot_delta_mismatch"})


def attribution_snapshot_key(metric_id: str, dimension: str) -> str:
    return f"{metric_id}:{dimension}"


def ratio_attribution_snapshot_key(metric_id: str, dimension: str) -> str:
    return f"{metric_id}:{dimension}:symmetric_mix_within_v1"


def capture_report_attribution_snapshots(report: PeriodicReport) -> PeriodicReport:
    """Freeze bounded dimension facts into the report before it is persisted.

    Insight workers must consume these values, never re-query mutable source
    tables. Every successful snapshot is reconciled to the report's locked
    current/baseline totals before it is retained.
    """
    from app.analysis.attribution import AttributionDataInsufficient
    from app.data.repositories.daily_metrics import (
        SUPPORTED_BREAKDOWN_DIMENSIONS,
        SUPPORTED_RATIO_FACT_DIMENSIONS,
        compare_breakdown_by_dimension_period,
        compare_ratio_facts_by_dimension_period,
    )
    from app.data.repositories.explorer import (
        breakdown_scoped_month_position,
        breakdown_scoped_product_category,
    )
    from app.metrics.registry import get_registry
    from app.schemas.explorer import ControlledAnalysisScope
    from app.analysis.semantic import get_metric_definition

    snapshots: dict[str, dict[str, object]] = {}
    for metric in report.metrics[:2]:
        comparison = metric.period_over_period
        if comparison is None or comparison.baseline_period is None:
            continue
        try:
            definition = get_metric_definition(metric.metric_name, report.scope.scope_type)
            registry_definition = get_registry().get(metric.metric_name)
            capability = registry_definition.attribution.for_scope(
                report.scope.scope_type,
            )
        except Exception:  # noqa: BLE001 - unavailable metadata cannot authorize a snapshot
            continue
        dimensions = []
        if "additive_change_v1" in capability.method_ids:
            if report.scope.scope_type == "seller":
                dimensions = sorted(set(capability.dimensions) & {"product_category", "month_position"})
            else:
                dimensions = sorted(
                    set(capability.dimensions)
                    & set(definition.allowed_dimensions)
                    & SUPPORTED_BREAKDOWN_DIMENSIONS
                )

        for dimension in dimensions[:3]:
            key = attribution_snapshot_key(metric.metric_name, dimension)
            base = {
                "metric_id": metric.metric_name,
                "dimension": dimension,
                "scope_type": report.scope.scope_type,
                "current_period": report.period.model_dump(mode="json"),
                "baseline_period": comparison.baseline_period.model_dump(mode="json"),
                "captured_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                if report.scope.scope_type == "seller":
                    scope = ControlledAnalysisScope(
                        seller_id=report.scope.seller_id,
                        default_start_date=report.period.start_date,
                        default_end_date=report.period.end_date,
                        metric_name=metric.metric_name,
                        granularity="day",
                        allowed_metrics=(metric.metric_name,),
                    )
                    baseline_scope = scope.model_copy(update={
                        "default_start_date": comparison.baseline_period.start_date,
                        "default_end_date": comparison.baseline_period.end_date,
                    })
                    query = (
                        breakdown_scoped_product_category
                        if dimension == "product_category"
                        else breakdown_scoped_month_position
                    )
                    current = query(scope, metric.metric_name, 10) if dimension == "product_category" else query(scope, metric.metric_name)
                    baseline = query(baseline_scope, metric.metric_name, 10) if dimension == "product_category" else query(baseline_scope, metric.metric_name)
                    if current.get("status") != "ok" or baseline.get("status") != "ok":
                        raise AttributionDataInsufficient({
                            "current": str(current.get("status") or "unknown"),
                            "baseline": str(baseline.get("status") or "unknown"),
                        })
                    current_total = float(current.get("population_total") or 0)
                    baseline_total = float(baseline.get("population_total") or 0)
                    current_items = current.get("items", [])
                    baseline_items = baseline.get("items", [])
                    source_table = current.get("source_table")
                else:
                    raw = compare_breakdown_by_dimension_period(
                        metric.metric_name,
                        report.period.start_date, report.period.end_date,
                        comparison.baseline_period.start_date, comparison.baseline_period.end_date,
                        dimension, 10,
                    )
                    if raw.get("status") != "ok":
                        raise AttributionDataInsufficient({"comparison": str(raw.get("status") or "unknown")})
                    current_total = float(raw.get("current_total") or 0)
                    baseline_total = float(raw.get("baseline_total") or 0)
                    current_items = raw.get("current_items", [])
                    baseline_items = raw.get("baseline_items", [])
                    source_table = raw.get("source_table")

                validate_report_breakdown_totals(
                    report, metric, comparison.baseline_period, current_total, baseline_total,
                )
                body = {
                    **base,
                    "status": "ok",
                    "current_total": current_total,
                    "baseline_total": baseline_total,
                    "total_delta": current_total - baseline_total,
                    "current_items": current_items,
                    "baseline_items": baseline_items,
                    "source_table": source_table,
                }
                body["snapshot_digest"] = hashlib.sha256(
                    json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
                ).hexdigest()
            except AttributionDataInsufficient as exc:
                body = {**base, "status": "data_insufficient", "reason_codes": exc.period_statuses}
            except Exception as exc:  # noqa: BLE001 - persist a safe gap; never fall back to a later live query
                body = {**base, "status": "unavailable", "reason_code": type(exc).__name__}
            snapshots[key] = body


    # Capture every eligible ratio metric (ratio metrics need not be among the
    # first two additive metrics selected above).
    ratio_method = "symmetric_mix_within_v1"
    for metric in report.metrics:
        comparison = metric.period_over_period
        if comparison is None or comparison.baseline_period is None:
            continue
        try:
            registry_definition = get_registry().get(metric.metric_name)
            capability = registry_definition.attribution.for_scope(
                report.scope.scope_type,
            )
        except Exception:  # noqa: BLE001 - unavailable metadata cannot authorize a snapshot
            continue
        period_aggregation = registry_definition.period_aggregation
        if (
            report.scope.scope_type != "platform"
            or ratio_method not in capability.method_ids
            or period_aggregation is None
            or period_aggregation.kind != "weighted_ratio"
        ):
            continue
        ratio_dimensions = sorted(
            set(capability.dimensions) & set(SUPPORTED_RATIO_FACT_DIMENSIONS)
        )
        for dimension in ratio_dimensions[:3]:
            key = ratio_attribution_snapshot_key(metric.metric_name, dimension)
            base = {
                "metric_id": metric.metric_name,
                "dimension": dimension,
                "method_id": ratio_method,
                "scope_type": report.scope.scope_type,
                "numerator_metric": period_aggregation.numerator_metric,
                "denominator_metric": period_aggregation.denominator_metric,
                "current_period": report.period.model_dump(mode="json"),
                "baseline_period": comparison.baseline_period.model_dump(mode="json"),
                "captured_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                raw = compare_ratio_facts_by_dimension_period(
                    period_aggregation.numerator_metric,
                    period_aggregation.denominator_metric,
                    report.period.start_date, report.period.end_date,
                    comparison.baseline_period.start_date, comparison.baseline_period.end_date,
                    dimension,
                )
                if raw.get("status") != "ok" or raw.get("complete") is not True:
                    raise AttributionDataInsufficient({
                        "comparison": str(raw.get("status") or "incomplete_group_population"),
                    })
                body = {
                    **base,
                    "status": "ok",
                    "baseline_groups": raw.get("baseline_groups", {}),
                    "current_groups": raw.get("current_groups", {}),
                    "baseline_numerator_total": raw.get("baseline_numerator_total"),
                    "current_numerator_total": raw.get("current_numerator_total"),
                    "baseline_denominator_total": raw.get("baseline_denominator_total"),
                    "current_denominator_total": raw.get("current_denominator_total"),
                    "source_table": raw.get("source_table"),
                    "complete": True,
                }
                body["snapshot_digest"] = hashlib.sha256(
                    json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
                ).hexdigest()
            except AttributionDataInsufficient as exc:
                body = {**base, "status": "data_insufficient", "reason_codes": exc.period_statuses}
            except Exception as exc:  # noqa: BLE001 - retain safe failure, do not retry later
                body = {**base, "status": "unavailable", "reason_code": type(exc).__name__}
            snapshots[key] = body

    return report.model_copy(update={"attribution_breakdowns": snapshots})
