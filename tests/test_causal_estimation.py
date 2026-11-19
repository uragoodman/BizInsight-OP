import pytest

from app.analysis.causal_estimation import (
    ExperimentObservation,
    VerifiedRandomizedExperiment,
    _student_t_critical_975,
    _student_t_two_sided_p,
    estimate_randomized_itt,
)


def _experiment(treatment_value=2.0, control_value=1.0, n=40, **updates):
    observations = tuple(
        [ExperimentObservation(f"t-{i}", "treatment", treatment_value) for i in range(n)]
        + [ExperimentObservation(f"c-{i}", "control", control_value) for i in range(n)]
    )
    values = {
        "experiment_id": "exp-1",
        "metric_id": "conversion_rate",
        "outcome_unit": "conversion",
        "assignment_unit": "user",
        "assignment_method": "randomized",
        "randomization_verified": True,
        "assignment_source": "governance.experiment_registry+analytics.experiment_observations",
        "observations": observations,
    }
    values.update(updates)
    return VerifiedRandomizedExperiment(**values)


def test_randomized_itt_estimates_effect_and_uncertainty():
    result = estimate_randomized_itt(_experiment())
    assert result["status"] == "ok"
    assert result["estimand"] == "intention_to_treat_difference_in_means"
    assert result["effect"] == pytest.approx(1.0)
    assert result["sample_size"] == 80
    assert result["confidence_interval"][0] <= result["effect"] <= result["confidence_interval"][1]
    assert "not proof" in result["causal_claim"]
    assert result["degrees_of_freedom"] is None


def test_welch_satterthwaite_t_reference_values_match_published_quantiles():
    # Reference values for the two-sided 95% Student t interval and its tail
    # probability; these keep the implementation independent of SciPy.
    assert _student_t_critical_975(1) == pytest.approx(12.7062047364, abs=1e-9)
    assert _student_t_critical_975(58) == pytest.approx(2.0017174841, abs=1e-9)
    assert _student_t_critical_975(1000) == pytest.approx(1.9623390808, abs=1e-9)
    assert _student_t_two_sided_p(2.0, 58) == pytest.approx(0.0501904680, abs=1e-10)


def test_randomized_itt_reports_welch_satterthwaite_df_and_t_interval():
    observations = tuple(
        [ExperimentObservation(f"t-{i}", "treatment", float(i % 7)) for i in range(40)]
        + [ExperimentObservation(f"c-{i}", "control", float(i % 3)) for i in range(45)]
    )
    result = estimate_randomized_itt(_experiment(observations=observations))
    assert result["method_id"] == "randomized_itt_welch_satterthwaite_t_v2"
    assert result["degrees_of_freedom"] < result["treatment_n"] + result["control_n"] - 2
    half_width = result["critical_value"] * result["standard_error"]
    assert result["confidence_interval"] == pytest.approx([
        result["effect"] - half_width,
        result["effect"] + half_width,
    ])
    assert 0.0 <= result["p_value"] <= 1.0
    assert any("Welch–Satterthwaite" in item for item in result["limitations"])


@pytest.mark.parametrize(("treatment", "control", "sign"), [(2.0, 1.0, 1), (1.0, 2.0, -1), (1.0, 1.0, 0)])
def test_randomized_itt_handles_positive_negative_and_null_effect(treatment, control, sign):
    result = estimate_randomized_itt(_experiment(treatment, control))
    assert result["status"] == "ok"
    assert (result["effect"] > 0) - (result["effect"] < 0) == sign


def test_randomized_itt_rejects_unverified_or_nonrandomized_data():
    with pytest.raises(PermissionError, match="randomized assignment"):
        estimate_randomized_itt(_experiment(randomization_verified=False))
    with pytest.raises(PermissionError, match="randomized assignment"):
        estimate_randomized_itt(_experiment(assignment_method="observational"))
    with pytest.raises(PermissionError, match="server-side source"):
        estimate_randomized_itt(_experiment(assignment_source="user_upload"))


def test_randomized_itt_fails_closed_on_small_or_incomplete_sample():
    result = estimate_randomized_itt(_experiment(n=20))
    assert result["status"] == "insufficient_data"
    observations = list(_experiment().observations)
    observations[0] = ExperimentObservation("t-0", "treatment", float("nan"))
    with pytest.raises(ValueError, match="missing and non-finite"):
        estimate_randomized_itt(_experiment(observations=tuple(observations)))


def test_randomized_itt_rejects_duplicate_randomization_units():
    observations = list(_experiment().observations)
    observations[-1] = ExperimentObservation("t-0", "control", 1.0)
    with pytest.raises(ValueError, match="unique"):
        estimate_randomized_itt(_experiment(observations=tuple(observations)))
