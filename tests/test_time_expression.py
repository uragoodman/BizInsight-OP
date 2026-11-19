from datetime import date

import pytest

from app.core.time_expression import ambiguous_year_list, resolve_expression, resolve_preset
from app.agents.data_query_agent import apply_trusted_time_range
from app.mcp.schemas import MetricQueryPlan


def _bounds():
    return date(2016, 9, 4), date(2018, 10, 17)


def test_quarter_is_clipped_to_available_data(monkeypatch):
    monkeypatch.setattr("app.core.time_expression.data_time_bounds", _bounds)
    value = resolve_expression("2018 Q4 的 GMV 趋势")
    assert value and value.start_date == date(2018, 10, 1)
    assert value.end_date == date(2018, 10, 17) and value.truncated


def test_relative_months_anchor_to_dataset_latest(monkeypatch):
    monkeypatch.setattr("app.core.time_expression.data_time_bounds", _bounds)
    value = resolve_preset("recent_months", value=3)
    assert (value.start_date, value.end_date) == (date(2018, 8, 1), date(2018, 10, 17))


def test_range_over_three_years_is_rejected(monkeypatch):
    monkeypatch.setattr("app.core.time_expression.data_time_bounds", _bounds)
    with pytest.raises(ValueError, match="three calendar years"):
        resolve_expression("2015-01-01 至 2018-10-17")


def test_scoped_relative_time_uses_supplied_bounds_without_global_database(monkeypatch):
    def unexpected_global_lookup():
        raise AssertionError("scoped time resolution must not query global data bounds")

    monkeypatch.setattr("app.core.time_expression.data_time_bounds", unexpected_global_lookup)
    value = resolve_expression(
        "最近3个月的 GMV 趋势",
        data_bounds=(date(2017, 9, 1), date(2018, 8, 31)),
    )
    assert value
    assert (value.start_date, value.end_date) == (date(2018, 6, 1), date(2018, 8, 31))


def test_scoped_calendar_expression_is_clipped_to_authorized_range(monkeypatch):
    monkeypatch.setattr(
        "app.core.time_expression.data_time_bounds",
        lambda: (_ for _ in ()).throw(AssertionError("must use scoped bounds")),
    )
    value = resolve_expression(
        "分析2018年1月到12月 GMV",
        data_bounds=(date(2018, 3, 1), date(2018, 8, 31)),
    )
    assert value
    assert (value.start_date, value.end_date) == (date(2018, 3, 1), date(2018, 8, 31))
    assert value.truncated


@pytest.mark.parametrize(("question", "expected"), [
    ("分析 2018年1月到5月 GMV 趋势", (date(2018, 1, 1), date(2018, 5, 31))),
    ("分析 2018年1-5月 GMV 趋势", (date(2018, 1, 1), date(2018, 5, 31))),
    ("分析 1-5月 GMV 趋势", (date(2018, 1, 1), date(2018, 5, 31))),
    ("Analyze Jan-May 2018 GMV trend", (date(2018, 1, 1), date(2018, 5, 31))),
    ("Analyze Jan-May GMV trend", (date(2018, 1, 1), date(2018, 5, 31))),
    ("分析 2018年5月 GMV", (date(2018, 5, 1), date(2018, 5, 31))),
    ("分析 2018年1月1日到5月31日 GMV", (date(2018, 1, 1), date(2018, 5, 31))),
    ("分析 2018年5月31日 GMV", (date(2018, 5, 31), date(2018, 5, 31))),
    ("Analyze Jan 1, 2018 to May 31, 2018 GMV", (date(2018, 1, 1), date(2018, 5, 31))),
    ("从 2018年5月1日至2018年5月31日分析 GMV", (date(2018, 5, 1), date(2018, 5, 31))),
    ("分析 2018年5月1日-30日 GMV", (date(2018, 5, 1), date(2018, 5, 30))),
    ("分析 5月1日至30号 GMV", (date(2018, 5, 1), date(2018, 5, 30))),
    ("Analyze May 1-30, 2018 GMV", (date(2018, 5, 1), date(2018, 5, 30))),
])
def test_resolve_chinese_and_english_month_expressions(monkeypatch, question, expected):
    monkeypatch.setattr("app.core.time_expression.data_time_bounds", _bounds)
    value = resolve_expression(question)
    assert value and (value.start_date, value.end_date) == expected


def test_month_range_sets_monthly_trend_granularity(monkeypatch):
    monkeypatch.setattr("app.core.time_expression.data_time_bounds", _bounds)
    plan = apply_trusted_time_range(
        MetricQueryPlan(query_type="metric_trend", metric_name="gmv"),
        "分析 2018年1月到5月 GMV 趋势",
    )
    assert (plan.start_date, plan.end_date, plan.granularity, plan.tool_name) == (
        "2018-01-01", "2018-05-31", "month", "query_metric_trend",
    )


def test_short_month_range_sets_monthly_trend_granularity(monkeypatch):
    monkeypatch.setattr("app.core.time_expression.data_time_bounds", _bounds)
    plan = apply_trusted_time_range(
        MetricQueryPlan(query_type="metric_trend", metric_name="gmv"),
        "分析 1-5月 GMV 趋势",
    )
    assert (plan.start_date, plan.end_date, plan.granularity) == ("2018-01-01", "2018-05-31", "month")


def test_chinese_category_followup_keeps_explicit_date():
    from app.agents.data_query_agent import (
        _extract_dates,
        _extract_dimension,
        build_metric_query_plan,
    )

    question = (
        "\u6210\u4ea4\u603b\u989d\u6309\u54c1\u7c7b\u62c6\u5f00\uff0c"
        "\u65e5\u671f\u4ecd\u662f2018-08-29"
    )
    assert _extract_dates(question) == ["2018-08-29"]
    assert _extract_dimension(question) == "product_category"
    plan = apply_trusted_time_range(build_metric_query_plan(question), question)
    assert (plan.query_type, plan.metric_name, plan.metric_date, plan.dimension) == (
        "metric_breakdown", "gmv", "2018-08-29", "product_category",
    )
    assert plan.tool_name == "breakdown_by_dimension"


@pytest.mark.parametrize(("question", "expected"), [
    ("分析 2017年至2018年 GMV", (date(2017, 1, 1), date(2018, 10, 17))),
    ("Analyze from 2017 to 2018 GMV", (date(2017, 1, 1), date(2018, 10, 17))),
])
def test_year_ranges_are_recognized(monkeypatch, question, expected):
    monkeypatch.setattr("app.core.time_expression.data_time_bounds", _bounds)
    value = resolve_expression(question)
    assert value and (value.start_date, value.end_date) == expected


def test_enumerated_year_trend_is_a_single_yearly_range(monkeypatch):
    monkeypatch.setattr("app.core.time_expression.data_time_bounds", _bounds)
    value = resolve_expression("评估2016年、2017年、2018年的最终交付率趋势")
    assert value and (value.start_date, value.end_date) == (date(2016, 9, 4), date(2018, 10, 17))


def test_enumerated_year_trend_uses_yearly_granularity(monkeypatch):
    monkeypatch.setattr("app.core.time_expression.data_time_bounds", _bounds)
    plan = apply_trusted_time_range(
        MetricQueryPlan(query_type="metric_trend", metric_name="order_final_delivery_rate"),
        "评估2016年、2017年、2018年的最终交付率趋势",
    )
    assert (plan.start_date, plan.end_date, plan.granularity) == ("2016-09-04", "2018-10-17", "year")


def test_bare_enumerated_years_require_human_intent_confirmation():
    assert ambiguous_year_list("查看2016年、2018年的最终交付率") == (2016, 2018)
    assert ambiguous_year_list("查看2016年、2018年的最终交付率趋势") is None


@pytest.mark.parametrize("question", ["查询 1-30号 GMV", "GMV 从 1 到 30 下降"])
def test_ambiguous_bare_numbers_are_not_parsed_as_dates(monkeypatch, question):
    monkeypatch.setattr("app.core.time_expression.data_time_bounds", _bounds)
    assert resolve_expression(question) is None
