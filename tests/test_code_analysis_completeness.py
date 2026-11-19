from app.code_analysis.completeness import validate_result_completeness


def _result(**overrides):
    result = {
        "conclusion": "GMV 月度趋势和订单数环比变化已汇总，最高月份与最低月份已列出。",
        "method": "按月计算 GMV、订单数和 MoM。",
        "summary": {"monthly_gmv": [], "monthly_order_count": [], "gmv_mom": [], "order_mom": []},
        "evidence": [{"month": "2026-01", "gmv": 10, "order_count": 2, "mom": 0}],
        "charts": [{"title": "GMV trend"}],
        "limitations": ["示例限制"],
        "recommendations": ["示例建议"],
    }
    result.update(overrides)
    return result


def test_result_completeness_rejects_adjacent_analysis_missing_requested_metrics():
    result = _result(
        conclusion="GMV 月度峰值和谷值已计算，并补充了国家排名。",
        method="按月计算 GMV。",
        summary={"peak_month": "2026-01", "trough_month": "2026-02"},
        evidence=[{"peak_month": "2026-01", "trough_month": "2026-02", "gmv": 10}],
    )

    checked = validate_result_completeness(
        "请按月汇总 GMV 和订单数，计算环比变化，指出最高和最低月份。",
        result,
        ["conclusion", "method", "evidence", "charts", "limitations", "recommendations", "reproducibility"],
    )

    assert checked.valid is False
    assert "订单数" in checked.missing
    assert "环比变化" in checked.missing


def test_result_completeness_accepts_requested_metrics_and_worker_owned_reproducibility():
    checked = validate_result_completeness(
        "请按月汇总 GMV 和订单数，计算环比变化，指出最高和最低月份。",
        _result(),
        ["conclusion", "method", "evidence", "charts", "limitations", "recommendations", "reproducibility"],
    )

    assert checked.valid is True
    assert checked.missing == ()


def test_result_completeness_checks_required_sections_but_skips_worker_owned_field():
    checked = validate_result_completeness(
        "总结 GMV。",
        _result(recommendations=[]),
        ["conclusion", "method", "evidence", "recommendations", "reproducibility"],
    )

    assert checked.valid is False
    assert "输出部分:recommendations" in checked.missing
    assert "输出部分:reproducibility" not in checked.missing
