"""Portable Phase 6 benchmark; reports distributions instead of machine-specific gates."""

from __future__ import annotations

from datetime import date, timedelta
import os
from math import ceil
from statistics import median
from time import perf_counter

from app.analysis.contracts import PeriodSpec
from app.analysis.contribution import calculate_change_contribution
from app.analysis.periodic_reports import build_periodic_report
from app.analysis.periods import query_metric_series
from app.analysis.semantic import METRIC_REGISTRY


def fixture_fetcher(metric_name: str, metric_date: date) -> dict:
    return {"value": float(metric_date.toordinal() % 100 + 1), "source_table": "benchmark.fixture"}


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, ceil(len(ordered) * ratio) - 1))]


def measure(operation, repetitions: int) -> list[float]:
    durations = []
    for _ in range(repetitions):
        started = perf_counter()
        operation()
        durations.append((perf_counter() - started) * 1000)
    return durations


def report_result(name: str, durations: list[float], threshold_variable: str) -> bool:
    p50, p95 = median(durations), percentile(durations, 0.95)
    threshold = float(os.getenv(threshold_variable, "inf"))
    print(f"workload={name} runs={len(durations)} p50_ms={p50:.3f} p95_ms={p95:.3f} limit_ms={threshold}")
    return p95 <= threshold


def main() -> None:
    end = date(2025, 3, 31)
    period = PeriodSpec(start_date=end - timedelta(days=89), end_date=end)
    metric_durations = []
    for metric_name in METRIC_REGISTRY:
        metric_durations.extend(measure(lambda name=metric_name: query_metric_series(name, period, fixture_fetcher), 3))
    current = [{"dimension_value": f"item-{index}", "value": 100 - index} for index in range(100)]
    baseline = [{"dimension_value": f"item-{index}", "value": 90 - index} for index in range(100)]
    contribution_durations = measure(
        lambda: calculate_change_contribution("gmv", "customer_city", end, end - timedelta(days=7), current, baseline),
        50,
    )
    week = PeriodSpec(start_date=date(2025, 3, 24), end_date=date(2025, 3, 30), granularity="week")
    month = PeriodSpec(start_date=date(2025, 3, 1), end_date=date(2025, 3, 31), granularity="month")
    report_durations = measure(
        lambda: build_periodic_report(
            week, ["gmv", "order_count", "avg_order_value"], fetcher=fixture_fetcher,
            include_charts=False, auto_enrich=False,
        ),
        10,
    ) + measure(
        lambda: build_periodic_report(
            month, ["gmv", "order_count", "avg_order_value"], fetcher=fixture_fetcher,
            include_charts=False, auto_enrich=False,
        ),
        10,
    )
    passed = all([
        report_result("metric-series-90d", metric_durations, "PHASE6_BENCHMARK_SERIES_P95_MS"),
        report_result("topk-contribution", contribution_durations, "PHASE6_BENCHMARK_CONTRIBUTION_P95_MS"),
        report_result("weekly-monthly-report", report_durations, "PHASE6_BENCHMARK_REPORT_P95_MS"),
    ])
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
