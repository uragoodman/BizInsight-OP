"""Pure descriptive statistics for metric time-series results.

The metric query layer supplies one value per selected time bucket.  This
module deliberately does not query data or choose a metric; it only produces a
deterministic summary from already scoped numeric points.
"""

from __future__ import annotations

from math import isfinite, sqrt
from statistics import fmean, median, quantiles
from typing import Iterable


DEFAULT_STATISTICS = ("count", "mean", "std", "min", "q1", "median", "q3", "max")


def summarize_points(
    points: Iterable[dict[str, object]],
    *,
    requested: Iterable[str] | None = None,
) -> dict[str, object]:
    """Summarize finite metric values while retaining their time buckets."""

    rows: list[tuple[str, float]] = []
    for point in points:
        value = point.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value_float = float(value)
        if isfinite(value_float):
            rows.append((str(point.get("bucket") or ""), value_float))

    names = tuple(dict.fromkeys(requested or DEFAULT_STATISTICS))
    values = [value for _, value in rows]
    result: dict[str, object] = {
        "count": len(values),
        "statistics": {name: None for name in names},
        "min_bucket": None,
        "max_bucket": None,
    }
    if not values:
        result["status"] = "not_found"
        return result

    stats = result["statistics"]
    assert isinstance(stats, dict)
    if "count" in stats:
        stats["count"] = len(values)
    if "mean" in stats or "average" in stats:
        mean = round(fmean(values), 6)
        if "mean" in stats:
            stats["mean"] = mean
        if "average" in stats:
            stats["average"] = mean
    if "std" in stats:
        mean = fmean(values)
        stats["std"] = round(sqrt(fmean([(value - mean) ** 2 for value in values])), 6)
    if "min" in stats:
        stats["min"] = min(values)
    if "max" in stats:
        stats["max"] = max(values)
    if "median" in stats:
        stats["median"] = median(values)
    if "q1" in stats or "q3" in stats:
        quartiles = quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
        if "q1" in stats:
            stats["q1"] = quartiles[0]
        if "q3" in stats:
            stats["q3"] = quartiles[2]

    min_value = min(values)
    max_value = max(values)
    result["min_bucket"] = next(bucket for bucket, value in rows if value == min_value)
    result["max_bucket"] = next(bucket for bucket, value in rows if value == max_value)
    result["status"] = "ok"
    return result
