"""Small-sample-aware statistics for registered binary association pairs."""

from __future__ import annotations

import math


_Z_95 = 1.959963984540054


def _wilson(events: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("group sample size must be positive")
    p = events / total
    z2 = _Z_95 * _Z_95
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half = _Z_95 * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def summarize_binary_association(
    *, exposed_events: int, exposed_total: int,
    comparison_events: int, comparison_total: int,
    min_total: int = 50,
) -> dict[str, object]:
    """Summarize a pre-registered 2x2 association with effect and uncertainty.

    The effect is risk difference (exposed minus comparison) with Newcombe's
    Wilson-score interval. The chi-square p value is supplementary, never the
    sole evidence. Percentages are returned as fractions here for stable APIs.
    """
    values = (exposed_events, exposed_total, comparison_events, comparison_total, min_total)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("2x2 counts must be integers")
    if min_total < 1 or exposed_total < 0 or comparison_total < 0:
        raise ValueError("sample sizes must be non-negative and min_total positive")
    if not 0 <= exposed_events <= exposed_total or not 0 <= comparison_events <= comparison_total:
        raise ValueError("event counts must be within their group totals")
    total = exposed_total + comparison_total
    result: dict[str, object] = {
        "status": "insufficient_data" if total < min_total or min(exposed_total, comparison_total) == 0 else "ok",
        "method_id": "binary_risk_difference_newcombe_v1",
        "sample_size": total,
        "min_group_sample_size": min(exposed_total, comparison_total),
        "exposed": {"n": exposed_total, "events": exposed_events, "rate": None, "rate_ci_95": None},
        "comparison": {"n": comparison_total, "events": comparison_events, "rate": None, "rate_ci_95": None},
        "risk_difference": None,
        "risk_difference_ci_95": None,
        "risk_ratio": None,
        "chi_square": {"statistic": None, "p_value": None},
        "claims": "statistical association only; not a causal effect",
    }
    if exposed_total == 0 or comparison_total == 0:
        return result
    p1 = exposed_events / exposed_total
    p0 = comparison_events / comparison_total
    lo1, hi1 = _wilson(exposed_events, exposed_total)
    lo0, hi0 = _wilson(comparison_events, comparison_total)
    # Newcombe hybrid score interval for the difference of two independent proportions.
    lower = (p1 - p0) - math.sqrt((p1 - lo1) ** 2 + (hi0 - p0) ** 2)
    upper = (p1 - p0) + math.sqrt((hi1 - p1) ** 2 + (p0 - lo0) ** 2)
    a, b, c, d = exposed_events, exposed_total - exposed_events, comparison_events, comparison_total - comparison_events
    cells = (a, b, c, d)
    row1, row0, col1, col0 = a + b, c + d, a + c, b + d
    expected = (
        row1 * col1 / total, row1 * col0 / total,
        row0 * col1 / total, row0 * col0 / total,
    )
    chi2 = sum((observed - exp) ** 2 / exp for observed, exp in zip(cells, expected) if exp > 0)
    result.update({
        "exposed": {"n": exposed_total, "events": exposed_events, "rate": p1, "rate_ci_95": [lo1, hi1]},
        "comparison": {"n": comparison_total, "events": comparison_events, "rate": p0, "rate_ci_95": [lo0, hi0]},
        "risk_difference": p1 - p0,
        "risk_difference_ci_95": [max(-1.0, lower), min(1.0, upper)],
        "risk_ratio": p1 / p0 if p0 else None,
        "chi_square": {"statistic": chi2, "p_value": math.erfc(math.sqrt(chi2 / 2.0)) if chi2 > 0 else 1.0},
    })
    return result


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for index in range(cursor, end):
            ranks[ordered[index][0]] = average_rank
        cursor = end
    return ranks


def summarize_spearman_association(
    pairs: list[tuple[float, float]], *, population_n: int,
    missing_n: int, min_total: int = 50,
) -> dict[str, object]:
    """Summarize a pre-registered continuous pair using Spearman's rho.

    The 95% interval and two-sided p value use Fisher/normal approximations;
    both are explicitly identified as approximations. The method is less
    sensitive than Pearson correlation to extreme magnitudes, but it does not
    adjust for confounding or establish causality.
    """
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (population_n, missing_n, min_total)):
        raise ValueError("sample counts must be integers")
    if population_n < 0 or missing_n < 0 or min_total < 4 or missing_n + len(pairs) != population_n:
        raise ValueError("population, missing and eligible sample counts are inconsistent")
    clean: list[tuple[float, float]] = []
    for x, y in pairs:
        if (isinstance(x, bool) or isinstance(y, bool)
                or not isinstance(x, (int, float)) or not isinstance(y, (int, float))
                or not math.isfinite(float(x)) or not math.isfinite(float(y))):
            raise ValueError("association pairs must be finite numeric observations")
        clean.append((float(x), float(y)))
    n = len(clean)
    result: dict[str, object] = {
        "status": "insufficient_data" if n < min_total else "ok",
        "method_id": "spearman_rank_correlation_fisher_approx_v1",
        "sample_size": n,
        "population_size": population_n,
        "excluded_missing": missing_n,
        "minimum_sample_size": min_total,
        "spearman_rho": None,
        "confidence_level": 0.95,
        "spearman_rho_ci_95_approx": None,
        "p_value_approx": None,
        "claims": "statistical association only; not a causal effect",
        "limitations": [
            "Spearman rank correlation captures monotonic association, not arbitrary nonlinear dependence.",
            "Confidence interval and p value use large-sample Fisher/normal approximations.",
            "No adjustment for confounders; association does not establish causality.",
        ],
    }
    if n < min_total:
        return result
    rank_x, rank_y = _average_ranks([item[0] for item in clean]), _average_ranks([item[1] for item in clean])
    mean_x, mean_y = sum(rank_x) / n, sum(rank_y) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(rank_x, rank_y))
    sum_x = sum((x - mean_x) ** 2 for x in rank_x)
    sum_y = sum((y - mean_y) ** 2 for y in rank_y)
    if sum_x == 0 or sum_y == 0:
        result["status"] = "undefined"
        result["undefined_reason"] = "one variable is constant in the eligible sample"
        return result
    rho = max(-1.0, min(1.0, covariance / math.sqrt(sum_x * sum_y)))
    clipped = max(-0.999999999, min(0.999999999, rho))
    z = math.atanh(clipped)
    half_width = _Z_95 / math.sqrt(n - 3)
    lower, upper = math.tanh(z - half_width), math.tanh(z + half_width)
    # Floating-point clipping near +/-1 must not produce an interval that
    # excludes the observed effect at the boundary.
    lower, upper = min(lower, rho), max(upper, rho)
    z_score = abs(z) * math.sqrt(n - 3)
    result.update({
        "status": "ok",
        "spearman_rho": rho,
        "spearman_rho_ci_95_approx": [lower, upper],
        "p_value_approx": math.erfc(z_score / math.sqrt(2.0)),
    })
    return result
