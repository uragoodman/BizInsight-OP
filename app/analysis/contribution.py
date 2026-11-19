"""Dimension contribution analysis based on change, not current share alone."""

from __future__ import annotations

from datetime import date
from hashlib import sha256
import json
import math

from app.analysis.contracts import ContributionFinding, ContributionItem
from app.analysis.attribution_methods import AttributionDataError, additive_change_attribution


def calculate_change_contribution(
    metric_name: str,
    dimension: str,
    current_date: date,
    baseline_date: date,
    current_items: list[dict],
    baseline_items: list[dict],
    overall_total_delta: float | None = None,
    current_total: float | None = None,
    baseline_total: float | None = None,
    current_period=None,
    baseline_period=None,
) -> ContributionFinding:
    def checked_groups(items: list[dict], label: str) -> dict[str, float]:
        groups: dict[str, float] = {}
        for item in items:
            if not isinstance(item, dict) or item.get("dimension_value") is None:
                raise AttributionDataError(f"{label} breakdown row is missing its dimension value")
            name = str(item["dimension_value"])
            if name in groups:
                raise AttributionDataError(f"{label} breakdown contains duplicate dimension group {name!r}")
            try:
                value = float(item.get("value") or 0)
            except (TypeError, ValueError) as exc:
                raise AttributionDataError(f"{label} breakdown group {name!r} has an invalid value") from exc
            if not math.isfinite(value):
                raise AttributionDataError(f"{label} breakdown group {name!r} is non-finite")
            groups[name] = value
        return groups

    current = checked_groups(current_items, "current")
    baseline = checked_groups(baseline_items, "baseline")
    total_delta = overall_total_delta if overall_total_delta is not None else sum(current.values()) - sum(baseline.values())
    if current_total is not None and baseline_total is not None:
        if not math.isfinite(float(current_total)) or not math.isfinite(float(baseline_total)):
            raise AttributionDataError("current or baseline total is non-finite")
        totals_delta = float(current_total) - float(baseline_total)
        if abs(float(total_delta) - totals_delta) > 0.01:
            raise AttributionDataError(
                "reported overall change does not match current and baseline totals"
            )
    if not math.isfinite(float(total_delta)):
        raise AttributionDataError("overall change is non-finite")
    # Breakdown APIs may return only top groups. Preserve omitted mass in an
    # explicit Other bucket, then use the shared versioned additive method.
    if current_total is not None and baseline_total is not None:
        residual_current = current_total - sum(current.values())
        residual_baseline = baseline_total - sum(baseline.values())
        if abs(residual_current) > 0.005 or abs(residual_baseline) > 0.005:
            current["Other"] = current.get("Other", 0.0) + residual_current
            baseline["Other"] = baseline.get("Other", 0.0) + residual_baseline
    elif overall_total_delta is not None:
        residual_delta = total_delta - sum(
            current.get(key, 0.0) - baseline.get(key, 0.0)
            for key in current.keys() | baseline.keys()
        )
        if abs(residual_delta) > 0.005:
            current["Other"] = current.get("Other", 0.0) + residual_delta

    method_result = additive_change_attribution(
        metric_id=metric_name,
        baseline_value=float(baseline_total if baseline_total is not None else sum(baseline.values())),
        current_value=float(current_total if current_total is not None else sum(current.values())),
        baseline_groups=baseline,
        current_groups=current,
        tolerance=0.03,
    )
    items = []
    for component in method_result.display_components:
        key = component.component_id
        delta = component.value
        ratio = None if total_delta == 0 else delta / total_delta
        items.append(ContributionItem(
            dimension_value=key,
            current_value=round(current.get(key, 0.0), 2),
            baseline_value=round(baseline.get(key, 0.0), 2),
            delta_value=round(delta, 2),
            contribution_ratio=None if ratio is None else round(ratio, 4),
        ))
    items.sort(key=lambda item: abs(item.delta_value), reverse=True)
    raw = json.dumps({
        "metric": metric_name, "dimension": dimension,
        "baseline_date": baseline_date.isoformat(), "current_date": current_date.isoformat(),
        "current": current_items, "baseline": baseline_items, "total_delta": total_delta,
    }, sort_keys=True, default=str)
    return ContributionFinding(
        metric_name=metric_name, dimension=dimension, current_date=current_date,
        baseline_date=baseline_date, current_period=current_period, baseline_period=baseline_period,
        total_delta=round(total_delta, 2), items=items,
        residual_value=method_result.residual_value,
        residual_type=method_result.residual_type,
        residual_explainable=method_result.residual_explainable,
        residual_tolerance=method_result.residual_tolerance,
        residual_note=method_result.residual_note,
        evidence_ids=[f"ev_{sha256(raw.encode()).hexdigest()[:16]}"],
    )
