"""Period aggregation and comparison services."""

from __future__ import annotations

from collections.abc import Callable
from calendar import monthrange
from datetime import date, datetime, time, timedelta, timezone
from statistics import fmean
from typing import Literal

from app.analysis.contracts import MetricPoint, MetricSeries, PeriodComparison, PeriodSpec
from app.analysis.semantic import get_metric_definition
from app.data.repositories.daily_metrics import query_metric
from app.data.repositories.daily_metrics import query_metric_period as repository_query_metric_period
from app.data.repositories.daily_metrics import query_metric_trend as repository_query_metric_trend


MetricFetcher = Callable[[str, date], dict]
PeriodFetcher = Callable[[str, date, date], dict]

RATIO_COMPONENTS = {
    "avg_order_value": ("gmv", "order_count", 1.0),
    "repeat_purchase_rate": ("repeat_customer_count", "customer_count", 100.0),
    "freight_rate": ("freight_value", "gmv", 100.0),
}


def _registry_ratio_components(metric_name: str):
    """Read composite-period metadata from the Registry with old-code fallback.

    The fallback is deliberately retained during migration so an older or
    partially loaded registry cannot silently change a calculation.  The YAML
    values currently mirror ``RATIO_COMPONENTS``; a mismatch is treated as a
    migration signal and the legacy definition remains authoritative for that
    request until the configuration is corrected.
    """

    try:
        from app.metrics.registry import get_registry

        definition = get_registry().get(metric_name)
        period = definition.period_aggregation
        registry_value = (
            period.numerator_metric,
            period.denominator_metric,
            period.multiplier,
        ) if period is not None else None
    except Exception:
        registry_value = None
    legacy_value = RATIO_COMPONENTS.get(metric_name)
    if registry_value is None:
        return legacy_value
    if legacy_value is not None and registry_value != legacy_value:
        return legacy_value
    return registry_value


def previous_period(period: PeriodSpec) -> PeriodSpec:
    if period.granularity == "month":
        previous_end = period.start_date - timedelta(days=1)
        previous_start = previous_end.replace(day=1)
        return PeriodSpec(
            start_date=previous_start, end_date=previous_end,
            granularity="month", timezone=period.timezone,
        )
    days = (period.end_date - period.start_date).days + 1
    end = period.start_date - timedelta(days=1)
    return PeriodSpec(
        start_date=end - timedelta(days=days - 1), end_date=end,
        granularity=period.granularity, timezone=period.timezone,
    )


def year_ago_period(period: PeriodSpec) -> PeriodSpec:
    def shift(value: date) -> date:
        try:
            return value.replace(year=value.year - 1)
        except ValueError:
            return value.replace(year=value.year - 1, day=28)
    return PeriodSpec(start_date=shift(period.start_date), end_date=shift(period.end_date), granularity=period.granularity)


def validate_natural_period(period: PeriodSpec) -> None:
    if period.granularity == "week":
        if period.start_date.weekday() != 0 or period.end_date.weekday() != 6 or (period.end_date - period.start_date).days != 6:
            raise ValueError("week period must be a complete Monday-Sunday natural week")
    elif period.granularity == "month":
        last_day = monthrange(period.start_date.year, period.start_date.month)[1]
        if period.start_date.day != 1 or period.end_date != period.start_date.replace(day=last_day):
            raise ValueError("month period must be a complete natural month")


def query_metric_series(metric_name: str, period: PeriodSpec, fetcher: MetricFetcher = query_metric) -> MetricSeries:
    definition = get_metric_definition(metric_name)
    if fetcher is query_metric:
        trend = repository_query_metric_trend(
            metric_name, period.start_date, period.end_date, "day",
        )
        repository_points = [
            MetricPoint(bucket=date.fromisoformat(item["bucket"]), value=item.get("value"))
            for item in trend["points"]
        ]
        freshness = (
            datetime.combine(
                max(point.bucket for point in repository_points if point.value is not None),
                time.max,
                tzinfo=timezone.utc,
            )
            if any(point.value is not None for point in repository_points) else None
        )
        component_totals: dict[str, float] = {}
        ratio = _registry_ratio_components(metric_name)
        if ratio:
            numerator_name, denominator_name, _ = ratio
            numerator = repository_query_metric_period(
                numerator_name, period.start_date, period.end_date,
            ).get("value")
            denominator = repository_query_metric_period(
                denominator_name, period.start_date, period.end_date,
            ).get("value")
            if numerator is not None:
                component_totals["numerator"] = float(numerator)
            if denominator is not None:
                component_totals["denominator"] = float(denominator)
        return MetricSeries(
            metric_name=metric_name, unit=definition.unit, period=period, points=repository_points,
            source_refs=[str(trend.get("source_table", definition.source_table))],
            definition_version=definition.version, freshness_at=freshness,
            component_totals=component_totals,
        )
    points: list[MetricPoint] = []
    components: dict[str, float] = {}
    current = period.start_date
    source_refs: set[str] = set()
    while current <= period.end_date:
        result = fetcher(metric_name, current)
        source_refs.add(str(result.get("source_table", definition.source_table)))
        points.append(MetricPoint(bucket=current, value=result.get("value")))
        ratio = _registry_ratio_components(metric_name)
        if ratio:
            numerator_name, denominator_name, _ = ratio
            numerator = fetcher(numerator_name, current).get("value")
            denominator = fetcher(denominator_name, current).get("value")
            if numerator is not None:
                components["numerator"] = components.get("numerator", 0.0) + float(numerator)
            if denominator is not None:
                components["denominator"] = components.get("denominator", 0.0) + float(denominator)
        current += timedelta(days=1)
    return MetricSeries(
        metric_name=metric_name, unit=definition.unit, period=period, points=points,
        source_refs=sorted(source_refs), definition_version=definition.version, component_totals=components,
        freshness_at=(
            datetime.combine(max(point.bucket for point in points if point.value is not None), time.max, tzinfo=timezone.utc)
            if any(point.value is not None for point in points) else None
        ),
    )


def aggregate_series(series: MetricSeries) -> float | None:
    values = [point.value for point in series.points if point.value is not None]
    if not values:
        return None
    aggregation = get_metric_definition(series.metric_name).aggregation
    if aggregation == "sum":
        return round(sum(values), 2)
    if aggregation == "weighted_ratio":
        numerator = series.component_totals.get("numerator")
        denominator = series.component_totals.get("denominator")
        if numerator is None or denominator is None or denominator == 0:
            return None
        multiplier = (_registry_ratio_components(series.metric_name) or ("", "", 1.0))[2]
        return round(numerator / denominator * multiplier, 2)
    if aggregation == "distinct_count":
        # Daily distinct counts cannot be combined without the underlying entity
        # identifiers; returning unavailable prevents double counting across days.
        return None
    return round(fmean(values), 2)


def aggregate_metric_period(
    metric_name: str,
    period: PeriodSpec,
    *,
    daily_fetcher: MetricFetcher | None = None,
    period_fetcher: PeriodFetcher = repository_query_metric_period,
) -> float | None:
    """Use exact entity-level aggregation in production and deterministic fixtures in tests."""

    if daily_fetcher is not None:
        return aggregate_series(query_metric_series(metric_name, period, daily_fetcher))
    return period_fetcher(metric_name, period.start_date, period.end_date).get("value")


def compare_periods(
    metric_name: str,
    current_period: PeriodSpec,
    baseline_period: PeriodSpec,
    comparison_type: Literal["period_over_period", "year_over_year", "target"] = "period_over_period",
    fetcher: MetricFetcher | None = None,
    period_fetcher: PeriodFetcher = repository_query_metric_period,
) -> PeriodComparison:
    current = aggregate_metric_period(metric_name, current_period, daily_fetcher=fetcher, period_fetcher=period_fetcher)
    baseline = aggregate_metric_period(metric_name, baseline_period, daily_fetcher=fetcher, period_fetcher=period_fetcher)
    if current is None or baseline is None:
        status: Literal["ok", "not_found", "zero_baseline"] = "not_found"
        delta, ratio = None, None
    else:
        delta = round(current - baseline, 2)
        ratio = None if baseline == 0 else round(delta * 100 / baseline, 2)
        status = "zero_baseline" if baseline == 0 else "ok"
    return PeriodComparison(
        comparison_type=comparison_type, current_period=current_period, baseline_period=baseline_period,
        current_value=current, baseline_value=baseline, delta_value=delta, delta_ratio=ratio, status=status,
    )
