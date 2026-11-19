from __future__ import annotations

import hashlib
import json


def pin_ratio_attribution_snapshot(
    report,
    metric_id: str,
    dimension: str,
    baseline_groups: dict,
    current_groups: dict,
    *,
    source_table: str = "analytics.fact_orders",
):
    """Attach a correctly digested ratio sufficient-statistics snapshot to a report fixture."""
    from app.analysis.report_alignment import ratio_attribution_snapshot_key
    from app.metrics.registry import get_registry

    definition = get_registry().get(metric_id)
    aggregation = definition.period_aggregation
    primary = next(metric for metric in report.metrics if metric.metric_name == metric_id)
    body = {
        "metric_id": metric_id,
        "dimension": dimension,
        "method_id": "symmetric_mix_within_v1",
        "scope_type": report.scope.scope_type,
        "numerator_metric": aggregation.numerator_metric,
        "denominator_metric": aggregation.denominator_metric,
        "current_period": report.period.model_dump(mode="json"),
        "baseline_period": primary.period_over_period.baseline_period.model_dump(mode="json"),
        "status": "ok",
        "baseline_groups": baseline_groups,
        "current_groups": current_groups,
        "baseline_numerator_total": sum(float(group["numerator"]) for group in baseline_groups.values()),
        "current_numerator_total": sum(float(group["numerator"]) for group in current_groups.values()),
        "baseline_denominator_total": sum(float(group["denominator"]) for group in baseline_groups.values()),
        "current_denominator_total": sum(float(group["denominator"]) for group in current_groups.values()),
        "source_table": source_table,
        "complete": True,
    }
    body["snapshot_digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    snapshots = dict(report.attribution_breakdowns)
    snapshots[ratio_attribution_snapshot_key(metric_id, dimension)] = body
    return report.model_copy(update={"attribution_breakdowns": snapshots})
