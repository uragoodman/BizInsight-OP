from app.graph.nodes.diagnostic import _observed_values_summary


def test_observed_values_keep_measured_trend_points_when_cause_is_unresolved():
    state = {
        "root_cause_evidence": [{
            "purpose": "trend_baseline",
            "metric_id": "delay_rate",
            "result": {"result": {"trend_result": {"points": [
                {"date": "2018-08-06", "value": 0, "unit": "%"},
                {"date": "2018-08-07", "value": 75, "unit": "%"},
                {"date": "2018-08-08", "value": 0, "unit": "%"},
            ]}}},
        }],
    }

    summary = _observed_values_summary(state)

    assert "delay_rate" in summary
    assert "2018-08-06=0%" in summary
    assert "2018-08-07=75%" in summary
    assert "2018-08-08=0%" in summary


def test_observed_values_report_missing_numeric_points():
    state = {
        "root_cause_evidence": [{
            "purpose": "trend_baseline",
            "metric_id": "review_score_avg",
            "result": {"result": {"trend_result": {"points": []}}},
        }],
    }

    assert _observed_values_summary(state) == (
        "Observed values: review_score_avg: no numeric observations."
    )
