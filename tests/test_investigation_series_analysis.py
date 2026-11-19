from datetime import date, timedelta

import pytest

from app.investigation.series_analysis import (
    check_series_quality,
    detect_series_anomalies_for_payload,
    extract_query_payload,
    normalize_points,
)
from app.investigation.causal_readiness import assess_causal_readiness


def test_normalize_points_handles_platform_and_seller_buckets_without_imputation():
    payload = {
        "points": [
            {"metric_date": "2025-01-02", "value": None},
            {"bucket": "2025-01-01", "value": 0},
            {"bucket": "2025-01-03", "value": float("inf")},
        ],
    }
    assert normalize_points(payload) == [
        {"bucket": "2025-01-01", "value": 0},
        {"bucket": "2025-01-02", "value": None},
        {"bucket": "2025-01-03", "value": None},
    ]


@pytest.mark.parametrize(
    ("check_id", "expected"),
    [
        ("zero_value_buckets", 1),
        ("negative_value_buckets", 1),
        ("missing_value_buckets", 1),
    ],
)
def test_time_series_quality_checks_return_scoped_bucket_evidence(check_id, expected):
    result = check_series_quality({
        "points": [
            {"bucket": "2025-01-01", "value": 0},
            {"bucket": "2025-01-02", "value": -2},
            {"bucket": "2025-01-03", "value": None},
        ],
    }, check_id)
    assert result["status"] == "ok"
    assert result["affected_bucket_count"] == expected
    assert result["bucket_count"] == 3
    assert result["limitations"]


def test_unknown_quality_check_fails_closed():
    with pytest.raises(ValueError, match="Unsupported"):
        check_series_quality({"points": []}, "duplicate_orders")


@pytest.mark.parametrize(
    ("points", "error_code"),
    [
        ([{"bucket": "2025-01-01", "value": 1}, {"bucket": "2025-01-01", "value": 2}], "duplicate_bucket"),
        ([{"bucket": "not-a-date", "value": 1}], "invalid_bucket"),
        ([{"bucket": "2025-01-01", "value": "1"}], "invalid_value"),
        ([{"bucket": "2025-01-01", "value": float("inf")}], "invalid_value"),
        ([{"bucket": "2025-01-01", "value": True}], "invalid_value"),
        ([{"bucket": "2025-01-01"}], "missing_value_field"),
    ],
)
def test_quality_checks_reject_malformed_or_duplicate_buckets(points, error_code):
    result = check_series_quality({"points": points}, "zero_value_buckets")

    assert result["status"] == "invalid_data"
    assert error_code in result["validation_errors"]


def test_quality_checks_distinguish_null_measurements_from_invalid_values():
    result = check_series_quality({
        "points": [
            {"bucket": "2025-01-01", "value": None},
            {"bucket": "2025-01-02", "value": 0},
        ],
    }, "missing_value_buckets")

    assert result["status"] == "ok"
    assert result["bucket_count"] == 2
    assert result["valued_bucket_count"] == 1
    assert result["affected_bucket_count"] == 1
    assert result["affected_bucket_rate"] == 0.5


def test_anomaly_analysis_abstains_below_minimum_history():
    result = detect_series_anomalies_for_payload(
        {"points": [{"bucket": f"2025-01-{day:02d}", "value": day} for day in range(1, 8)]},
        metric_id="gmv",
        unit="BRL",
        start_date="2025-01-01",
        end_date="2025-01-07",
        granularity="day",
        definition_version="1.0",
        source_ref="fixture",
    )
    assert result["status"] == "insufficient_data"
    assert result["minimum_required"] == 8
    assert result["anomalies"] == []


def test_anomaly_analysis_rejects_duplicate_bucket_instead_of_counting_it_as_history():
    points = [
        {"bucket": "2025-01-01", "value": 1},
        {"bucket": "2025-01-01", "value": 2},
        *[
            {"bucket": (date(2025, 1, 2) + timedelta(days=index)).isoformat(), "value": 3}
            for index in range(7)
        ],
    ]

    result = detect_series_anomalies_for_payload(
        {"points": points}, metric_id="gmv", unit="BRL",
        start_date="2025-01-01", end_date="2025-01-09", granularity="day",
        definition_version="1.0", source_ref="fixture",
    )

    assert result["status"] == "invalid_data"
    assert result["validation_errors"] == ["duplicate_bucket"]
    assert result["anomalies"] == []


def test_anomaly_analysis_uses_existing_deterministic_detector_and_evidence():
    points = [
        {"bucket": (date(2025, 1, 1) + timedelta(days=index)).isoformat(), "value": 10}
        for index in range(14)
    ]
    points.append({"bucket": "2025-01-15", "value": 100})
    result = detect_series_anomalies_for_payload(
        {"points": points},
        metric_id="gmv",
        unit="BRL",
        start_date="2025-01-01",
        end_date="2025-01-15",
        granularity="day",
        definition_version="1.0",
        source_ref="fixture",
    )
    assert result["status"] == "ok"
    assert result["detector"] == "rolling_median_mad_iqr_v1"
    assert result["anomaly_count"] == 1
    assert result["anomalies"][0]["evidence_ids"]


def test_extract_query_payload_unwraps_typed_query_result():
    assert extract_query_payload({
        "status": "ok",
        "trend_result": {"points": [{"bucket": "2025-01-01", "value": 3}]},
    }) == {"points": [{"bucket": "2025-01-01", "value": 3}]}


def test_causal_readiness_fails_closed_without_trusted_experiment_design():
    result = assess_causal_readiness([
        {"investigation_evidence": {"values": {"claim": "campaign caused the increase"}}},
    ])
    assert result["status"] == "not_ready"
    assert result["ready_for_estimation"] is False
    assert "experiment_id" in result["missing_fields"]
    assert result["causal_effect_estimated"] is False


def test_causal_readiness_requires_valid_randomization_and_distinct_groups():
    design = {
        "experiment_id": "exp-1",
        "assignment_unit": "customer",
        "treatment_group": "variant",
        "control_group": "control",
        "outcome_metric": "gmv",
        "observation_window": {"start": "2025-01-01", "end": "2025-01-31"},
        "randomization_validation": True,
    }
    ready = assess_causal_readiness([{
        "investigation_evidence": {"values": {"experiment_design": design}},
    }])
    assert ready["status"] == "ready_for_estimator"
    assert ready["ready_for_estimation"] is True
    assert ready["causal_effect_estimated"] is False

    design["randomization_validation"] = False
    not_ready = assess_causal_readiness([{
        "investigation_evidence": {"values": {"experiment_design": design}},
    }])
    assert not_ready["status"] == "not_ready"
    assert "randomization_validation_passed" in not_ready["missing_fields"]


def test_causal_readiness_rejects_malformed_experiment_window_and_blank_identifiers():
    design = {
        "experiment_id": " ",
        "assignment_unit": "customer",
        "treatment_group": "treatment",
        "control_group": "control",
        "outcome_metric": "gmv",
        "observation_window": {"start": "2025-02-01", "end": "2025-01-01"},
        "randomization_validation": True,
    }
    result = assess_causal_readiness([{
        "investigation_evidence": {"values": {"experiment_design": design}},
    }])
    assert result["ready_for_estimation"] is False
    assert "experiment_id" in result["missing_fields"]
    assert "valid_observation_window" in result["missing_fields"]
