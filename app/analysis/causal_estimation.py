"""Fail-closed intent-to-treat estimator for verified randomized experiments.

This is deliberately a calculation primitive, not an ingestion or authorization
surface. Callers must obtain the rows from a trusted experiment repository and
verify assignment provenance before constructing the input contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


_BETA_EPSILON = 3.0e-14
_BETA_FLOOR = 1.0e-300
_BETA_MAX_ITERATIONS = 300
_NORMAL_95_CRITICAL = 1.959963984540054


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the regularized beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _BETA_FLOOR:
        d = _BETA_FLOOR
    d = 1.0 / d
    result = d

    for iteration in range(1, _BETA_MAX_ITERATIONS + 1):
        twice = 2.0 * iteration
        coefficient = iteration * (b - iteration) * x / (
            (qam + twice) * (a + twice)
        )
        d = 1.0 + coefficient * d
        if abs(d) < _BETA_FLOOR:
            d = _BETA_FLOOR
        c = 1.0 + coefficient / c
        if abs(c) < _BETA_FLOOR:
            c = _BETA_FLOOR
        d = 1.0 / d
        result *= d * c

        coefficient = -((a + iteration) * (qab + iteration) * x) / (
            (a + twice) * (qap + twice)
        )
        d = 1.0 + coefficient * d
        if abs(d) < _BETA_FLOOR:
            d = _BETA_FLOOR
        c = 1.0 + coefficient / c
        if abs(c) < _BETA_FLOOR:
            c = _BETA_FLOOR
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= _BETA_EPSILON:
            return result

    raise ArithmeticError("regularized beta continued fraction did not converge")


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a,b), implemented with stdlib math."""
    if not (a > 0.0 and b > 0.0) or not math.isfinite(a + b):
        raise ValueError("beta parameters must be positive and finite")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_beta_factor = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    beta_factor = math.exp(log_beta_factor)
    if x < (a + 1.0) / (a + b + 2.0):
        result = beta_factor * _beta_continued_fraction(a, b, x) / a
    else:
        result = 1.0 - beta_factor * _beta_continued_fraction(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, result))


def _student_t_two_sided_p(statistic: float, degrees_of_freedom: float) -> float:
    if statistic == 0.0:
        return 1.0
    if math.isinf(degrees_of_freedom):
        return math.erfc(abs(statistic) / math.sqrt(2.0))
    x = degrees_of_freedom / (degrees_of_freedom + statistic * statistic)
    return _regularized_incomplete_beta(x, degrees_of_freedom / 2.0, 0.5)


def _student_t_critical_975(degrees_of_freedom: float) -> float:
    """Return t_(0.975, df) by monotone bisection; no SciPy runtime dependency."""
    if math.isinf(degrees_of_freedom):
        return _NORMAL_95_CRITICAL
    if not (degrees_of_freedom > 0.0 and math.isfinite(degrees_of_freedom)):
        raise ValueError("degrees_of_freedom must be positive and finite")

    low, high = 0.0, 1.0
    while _student_t_two_sided_p(high, degrees_of_freedom) > 0.05:
        high *= 2.0
        if not math.isfinite(high):
            raise ArithmeticError("could not bracket Student t critical value")
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if _student_t_two_sided_p(midpoint, degrees_of_freedom) > 0.05:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


@dataclass(frozen=True)
class ExperimentObservation:
    unit_id: str
    assigned_group: Literal["treatment", "control"]
    outcome: float


@dataclass(frozen=True)
class VerifiedRandomizedExperiment:
    experiment_id: str
    metric_id: str
    outcome_unit: str
    assignment_unit: str
    assignment_method: Literal["randomized"]
    randomization_verified: bool
    assignment_source: str
    observations: tuple[ExperimentObservation, ...]


def estimate_randomized_itt(
    experiment: VerifiedRandomizedExperiment,
    *,
    minimum_per_arm: int = 30,
) -> dict[str, object]:
    """Estimate treatment-assignment ITT as a difference in means.

    The interval and two-sided p-value use the Welch standard error and the
    Welch–Satterthwaite t reference distribution. This does not establish
    universal causality: validity depends on the verified randomization design,
    outcome completeness, no interference, and adherence to the stated window.
    The estimator remains inappropriate for paired, stratified, cluster-randomized,
    or strongly heavy-tailed outcomes without a design-specific method.
    """
    if not experiment.experiment_id or not experiment.metric_id or not experiment.outcome_unit:
        raise ValueError("experiment identity, metric, and outcome unit are required")
    if experiment.assignment_method != "randomized" or not experiment.randomization_verified:
        raise PermissionError("Verified randomized assignment is required for causal estimation")
    if experiment.assignment_source != "governance.experiment_registry+analytics.experiment_observations":
        raise PermissionError("Experiment rows must come from the verified server-side source")
    if experiment.assignment_unit not in {"user", "customer", "order", "session"}:
        raise ValueError("unsupported randomization unit")
    if minimum_per_arm < 30:
        raise ValueError("minimum_per_arm cannot be lower than the governed minimum of 30")
    if not experiment.observations:
        raise ValueError("experiment has no observations")

    seen: set[str] = set()
    outcomes: dict[str, list[float]] = {"treatment": [], "control": []}
    for row in experiment.observations:
        if not row.unit_id or row.unit_id in seen:
            raise ValueError("randomization unit IDs must be present and unique")
        seen.add(row.unit_id)
        if row.assigned_group not in outcomes:
            raise ValueError("each unit must have a valid randomized assignment")
        if isinstance(row.outcome, bool) or not isinstance(row.outcome, (int, float)) or not math.isfinite(float(row.outcome)):
            raise ValueError("missing and non-finite outcomes are not accepted by this estimator")
        outcomes[row.assigned_group].append(float(row.outcome))

    treatment = outcomes["treatment"]
    control = outcomes["control"]
    n_t, n_c = len(treatment), len(control)
    if min(n_t, n_c) < minimum_per_arm:
        return {
            "status": "insufficient_data",
            "estimand": "intention_to_treat_difference_in_means",
            "experiment_id": experiment.experiment_id,
            "metric_id": experiment.metric_id,
            "outcome_unit": experiment.outcome_unit,
            "sample_size": n_t + n_c,
            "treatment_n": n_t,
            "control_n": n_c,
            "minimum_per_arm": minimum_per_arm,
        }

    def mean_variance(values: list[float]) -> tuple[float, float]:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return mean, variance

    mean_t, var_t = mean_variance(treatment)
    mean_c, var_c = mean_variance(control)
    effect = mean_t - mean_c
    treatment_variance_term = var_t / n_t
    control_variance_term = var_c / n_c
    standard_error = math.sqrt(treatment_variance_term + control_variance_term)
    df_denominator = (
        treatment_variance_term**2 / (n_t - 1)
        + control_variance_term**2 / (n_c - 1)
    )
    degrees_of_freedom = (
        (treatment_variance_term + control_variance_term) ** 2 / df_denominator
        if df_denominator > 0.0
        else None
    )
    reference_df = degrees_of_freedom if degrees_of_freedom is not None else math.inf
    if standard_error:
        statistic = effect / standard_error
        critical_value = _student_t_critical_975(reference_df)
        p_value = _student_t_two_sided_p(statistic, reference_df)
    else:
        critical_value = _NORMAL_95_CRITICAL
        p_value = 1.0 if effect == 0.0 else 0.0
    lower = effect - critical_value * standard_error
    upper = effect + critical_value * standard_error
    return {
        "status": "ok",
        "method_id": "randomized_itt_welch_satterthwaite_t_v2",
        "estimand": "intention_to_treat_difference_in_means",
        "experiment_id": experiment.experiment_id,
        "metric_id": experiment.metric_id,
        "assignment_unit": experiment.assignment_unit,
        "outcome_unit": experiment.outcome_unit,
        "assignment_method": "randomized",
        "sample_size": n_t + n_c,
        "treatment_n": n_t,
        "control_n": n_c,
        "treatment_mean": mean_t,
        "control_mean": mean_c,
        "effect": effect,
        "standard_error": standard_error,
        "degrees_of_freedom": degrees_of_freedom,
        "critical_value": critical_value,
        "confidence_level": 0.95,
        "confidence_interval": [lower, upper],
        "p_value": p_value,
        "causal_claim": "randomized treatment-assignment effect estimate under the recorded design; not proof beyond these assumptions",
        "assumptions": [
            "assignment was randomized and independently verified",
            "each randomization unit appears exactly once",
            "outcomes are complete for all assigned units",
            "no interference between randomization units",
            "the reported outcome and analysis window were pre-specified",
        ],
        "limitations": [
            "Welch–Satterthwaite t approximation assumes independent randomization units and a suitable mean-based model.",
            "The estimate is ITT; it is not a treatment-on-treated estimate.",
            "Paired, stratified, cluster-randomized, and strongly heavy-tailed outcomes require a design-specific estimator.",
        ],
    }
