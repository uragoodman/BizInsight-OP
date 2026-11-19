from app.graph.nodes.answer import _summarize_metric_query


def test_trend_summary_reports_change_and_extrema() -> None:
    answer = _summarize_metric_query({
        "result": {
            "trend_result": {
                "metric_name": "gmv", "start_date": "2018-01-01", "end_date": "2018-03-31",
                "source_table": "analytics.daily_metrics",
                "points": [
                    {"bucket": "2018-01", "value": 100.0},
                    {"bucket": "2018-02", "value": 120.0},
                    {"bucket": "2018-03", "value": 90.0},
                ],
            },
        },
    })
    assert "整体下降" in answer
    assert "最高点" in answer and "2018-02" in answer
    assert "最低点" in answer and "2018-03" in answer
    assert "最近一期较上一期下降" in answer
