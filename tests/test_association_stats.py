import pytest

from app.analysis.association_stats import summarize_binary_association, summarize_spearman_association


@pytest.mark.parametrize(
    ("exposed", "comparison", "expected_sign"),
    [
        ((40, 100), (10, 100), 1),
        ((10, 100), (40, 100), -1),
        ((20, 100), (20, 100), 0),
    ],
)
def test_registered_binary_association_reports_effect_and_uncertainty(exposed, comparison, expected_sign):
    result = summarize_binary_association(
        exposed_events=exposed[0], exposed_total=exposed[1],
        comparison_events=comparison[0], comparison_total=comparison[1],
    )

    assert result["status"] == "ok"
    assert result["sample_size"] == 200
    assert result["exposed"]["n"] == 100
    assert result["comparison"]["n"] == 100
    assert result["exposed"]["rate_ci_95"][0] < result["exposed"]["rate"] < result["exposed"]["rate_ci_95"][1]
    assert result["risk_difference_ci_95"][0] <= result["risk_difference"] <= result["risk_difference_ci_95"][1]
    assert (result["risk_difference"] > 0) - (result["risk_difference"] < 0) == expected_sign
    assert result["chi_square"]["p_value"] is not None
    assert "not a causal" in result["claims"]


def test_registered_binary_association_marks_small_sample_as_insufficient():
    result = summarize_binary_association(
        exposed_events=5, exposed_total=20,
        comparison_events=2, comparison_total=20,
    )
    assert result["status"] == "insufficient_data"
    assert result["sample_size"] == 40
    assert result["risk_difference"] is not None


def test_registered_binary_association_validates_counts_and_zero_group():
    with pytest.raises(ValueError, match="within their group"):
        summarize_binary_association(
            exposed_events=3, exposed_total=2,
            comparison_events=1, comparison_total=2,
        )
    result = summarize_binary_association(
        exposed_events=0, exposed_total=0,
        comparison_events=2, comparison_total=60,
    )
    assert result["status"] == "insufficient_data"
    assert result["risk_difference"] is None


@pytest.mark.parametrize(
    ("exposed_events", "comparison_events", "expected_difference", "expected_rr"),
    [
        (0, 0, 0.0, None),       # both groups have zero events
        (1000, 0, 1.0, None),   # exposed group is 100%; comparison is 0%
        (0, 1000, -1.0, 0.0),   # exposed group is 0%; comparison is 100%
        (1, 999, -0.998, 1 / 999), # extreme but valid proportions
    ],
)
def test_registered_binary_association_handles_zero_and_extreme_cells(
    exposed_events, comparison_events, expected_difference, expected_rr,
):
    result = summarize_binary_association(
        exposed_events=exposed_events,
        exposed_total=1000,
        comparison_events=comparison_events,
        comparison_total=1000,
    )

    assert result["status"] == "ok"
    assert result["risk_difference"] == pytest.approx(expected_difference)
    if expected_rr is None:
        assert result["risk_ratio"] is None
    else:
        assert result["risk_ratio"] == pytest.approx(expected_rr)
    lower, upper = result["risk_difference_ci_95"]
    assert -1.0 <= lower <= expected_difference <= upper <= 1.0
    assert result["chi_square"]["p_value"] is not None


def test_registered_binary_association_rejects_boolean_counts():
    with pytest.raises(ValueError, match="integers"):
        summarize_binary_association(
            exposed_events=True, exposed_total=100,
            comparison_events=0, comparison_total=100,
        )


@pytest.mark.parametrize(("slope", "expected_sign"), [(1.0, 1), (-1.0, -1), (0.0, 0)])
def test_spearman_reports_direction_and_approximate_uncertainty(slope, expected_sign):
    pairs = [
        (float(index), float((index * 7) % 11) if slope == 0 else float(index * slope))
        for index in range(60)
    ]
    result = summarize_spearman_association(pairs, population_n=60, missing_n=0)
    assert result["status"] == "ok"
    assert result["method_id"] == "spearman_rank_correlation_fisher_approx_v1"
    if expected_sign:
        assert (result["spearman_rho"] > 0) - (result["spearman_rho"] < 0) == expected_sign
        interval = result["spearman_rho_ci_95_approx"]
        assert interval[0] <= result["spearman_rho"] <= interval[1]
    else:
        assert result["spearman_rho"] == pytest.approx(0.0, abs=0.05)
    assert "not a causal" in result["claims"]


def test_spearman_handles_ties_and_extreme_magnitude_outliers():
    pairs = [(float(index), float(index // 2)) for index in range(60)]
    ordinary = summarize_spearman_association(pairs, population_n=60, missing_n=0)
    pairs[-1] = (1e100, 1e100)
    outlier = summarize_spearman_association(pairs, population_n=60, missing_n=0)
    assert ordinary["status"] == outlier["status"] == "ok"
    assert ordinary["spearman_rho"] > 0.95
    assert outlier["spearman_rho"] > 0.95


def test_spearman_fails_closed_on_small_constant_nonfinite_or_inconsistent_data():
    assert summarize_spearman_association(
        [(float(i), float(i)) for i in range(20)], population_n=22, missing_n=2,
    )["status"] == "insufficient_data"
    constant = summarize_spearman_association(
        [(float(i), 1.0) for i in range(50)], population_n=50, missing_n=0,
    )
    assert constant["status"] == "undefined"
    with pytest.raises(ValueError, match="finite"):
        summarize_spearman_association(
            [(float(i), float("nan")) for i in range(50)], population_n=50, missing_n=0,
        )
    with pytest.raises(ValueError, match="inconsistent"):
        summarize_spearman_association([(1.0, 2.0)], population_n=5, missing_n=0)
