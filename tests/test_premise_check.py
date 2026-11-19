from datetime import date

import pytest

from app.core.time_expression import extract_iso_dates, resolve_expression
from app.graph.nodes.premise import (
    _claimed_direction,
    _premise_plan,
    _trend_values_for_question,
    premise_check,
    premise_confirmation,
    premise_query_hint,
)


def test_directional_premise_recognizes_below_comparison_word():
    assert _claimed_direction("为什么 8 月 GMV 低于 7 月？") == "down"


def _trend_result(start: float, end: float) -> dict:
    return {
        "server": "registry-mcp",
        "tool_name": "query_metric_trend",
        "result": {
            "status": "ok",
            "trend_result": {
                "metric_name": "gmv",
                "start_date": "2018-08-19",
                "end_date": "2018-08-20",
                "points": [
                    {"bucket": "2018-08-19", "value": start},
                    {"bucket": "2018-08-20", "value": end},
                ],
            },
        },
    }


def _period_comparison_result(current: float, baseline: float) -> dict:
    parsed = {
        "query_type": "metric_compare",
        "metric_name": "gmv",
        "authorized_metric": {"metric_id": "gmv"},
        "current_start": "2018-08-01",
        "current_end": "2018-08-31",
        "baseline_start": "2018-07-01",
        "baseline_end": "2018-07-31",
    }
    comparison = {
        "status": "ok",
        "metric_name": "gmv",
        "current_date": "2018-08-01",
        "left_end_date": "2018-08-31",
        "baseline_date": "2018-07-01",
        "right_end_date": "2018-07-31",
        "current_value": current,
        "baseline_value": baseline,
    }
    content = {
        "request_id": "compare-request:1",
        "result": {"status": "ok", "parsed": parsed, "compare_result": comparison},
    }
    return {
        "task_type": "anomaly_analysis",
        "tool_result": {
            "server": "metric-registry",
            "tool_name": "registry_compare",
            "request_id": "compare-request:1",
            "result": {"status": "ok", "parsed": parsed, "compare_result": comparison},
        },
        "request_id": "compare-request:1",
        "evidence_bundles": [{
            "evidence_id": "ev-period-compare",
            "source_type": "mcp",
            "trusted": True,
            "content": content,
        }],
    }


def test_premise_check_corrects_wrong_declared_drop_from_planned_trend_result():
    state = {
        "task_type": "anomaly_analysis",
        "user_message": "为什么 2018-08-20 的 GMV 下降了？",
        "trace_id": "premise-test",
        "tool_result": _trend_result(100.0, 130.0),
        "query_results": [],
        "tool_calls": [],
        "node_history": [],
    }
    result = premise_check(state)
    assert result["premise_check"]["status"] == "mismatch"
    assert result["premise_check"]["actual_direction"] == "up"
    assert result["awaiting_user_confirmation"] is True
    assert "实际是上升" in result["answer"]


def test_premise_check_allows_matching_direction():
    state = {
        "task_type": "anomaly_analysis",
        "user_message": "为什么 2018-08-20 的 GMV 下降了？",
        "tool_result": _trend_result(130.0, 100.0),
        "query_results": [],
        "tool_calls": [],
        "node_history": [],
    }
    result = premise_check(state)
    assert result["premise_check"]["status"] == "confirmed"
    assert "answer" not in result


def test_premise_check_accepts_matching_trusted_period_comparison():
    state = _period_comparison_result(current=985491.64, baseline=1027807.28)
    state.update({
        "user_message": "平台范围内，2018 年 8 月 GMV 为什么比 2018 年 7 月下降？",
        "trace_id": "premise-period-compare",
        "query_results": [],
        "tool_calls": [],
        "node_history": [],
    })

    result = premise_check(state)

    assert result["premise_check"]["status"] == "confirmed"
    assert result["premise_check"]["actual_direction"] == "down"
    assert result["premise_check"]["baseline_value"] == 1027807.28
    assert result["premise_check"]["current_value"] == 985491.64
    assert result["premise_check"]["evidence_ids"] == ["ev-period-compare"]
    assert "terminal_code" not in result


def test_premise_check_flags_mismatching_trusted_period_comparison():
    state = _period_comparison_result(current=130.0, baseline=100.0)
    state.update({
        "user_message": "平台范围内，2018 年 8 月 GMV 为什么比 2018 年 7 月下降？",
        "trace_id": "premise-period-mismatch",
        "query_results": [],
        "tool_calls": [],
        "node_history": [],
    })

    result = premise_check(state)

    assert result["premise_check"]["status"] == "mismatch"
    assert result["premise_check"]["actual_direction"] == "up"
    assert result["awaiting_user_confirmation"] is True
    assert "实际是上升" in result["answer"]


def test_premise_check_still_fails_closed_without_trusted_period_comparison():
    state = _period_comparison_result(current=985491.64, baseline=1027807.28)
    state["evidence_bundles"][0]["trusted"] = False
    state.update({
        "user_message": "平台范围内，2018 年 8 月 GMV 为什么比 2018 年 7 月下降？",
        "trace_id": "premise-period-untrusted",
        "query_results": [],
        "tool_calls": [],
        "node_history": [],
    })

    result = premise_check(state)

    assert result["premise_check"]["status"] == "unavailable"
    assert result["terminal_code"] == "DATA_INSUFFICIENT"
    assert "不会继续" in result["answer"]


def test_confirmed_period_comparison_does_not_resume_root_cause_search(monkeypatch):
    from app.graph.workflow import _route_after_premise_confirmation

    monkeypatch.setattr("app.graph.nodes.premise.interrupt", lambda *_args, **_kwargs: {"action": "confirm"})
    state = _period_comparison_result(current=130.0, baseline=100.0)
    state.update({
        "user_message": "平台范围内，2018 年 8 月 GMV 为什么比 2018 年 7 月下降？",
        "trace_id": "premise-period-confirmation",
        "awaiting_user_confirmation": True,
        "premise_check": {
            "status": "mismatch",
            "claimed_direction": "down",
            "actual_direction": "up",
            "metric_name": "gmv",
            "metric_display_name": "成交总额",
            "baseline_date": "2018-07-01 至 2018-07-31",
            "current_date": "2018-08-01 至 2018-08-31",
            "baseline_value": 100.0,
            "current_value": 130.0,
        },
        "node_history": [],
    })

    result = premise_confirmation(state)

    assert result["premise_correction_finalized"] is True
    assert not result.get("errors")
    assert result["analysis_result"]["evidence_ids"] == ["ev-period-compare"]
    assert result["analysis_presentation"]["changes"][0]["baseline_period"] == "2018-07-01 至 2018-07-31"
    assert _route_after_premise_confirmation(result) == "persist_trace"


def test_premise_query_binds_direction_to_the_metric_named_with_that_claim(monkeypatch):
    query = "2018-08-25 GMV下降，延迟送达率也上升，GMV为什么下降？"
    monkeypatch.setattr(
        "app.core.time_expression.data_time_bounds",
        lambda: (date(2017, 1, 1), date(2018, 8, 31)),
    )

    hint = premise_query_hint(query)

    assert hint is not None
    assert "gmv" in hint
    assert "delivery_delay_rate" not in hint


def test_premise_check_resumes_after_confirmation(monkeypatch):
    monkeypatch.setattr("app.graph.nodes.premise.interrupt", lambda *_args, **_kwargs: {"action": "confirm"})
    state = {
        "task_type": "anomaly_analysis", "user_message": "Why did GMV decrease?",
        "awaiting_user_confirmation": True,
        "premise_check": {"status": "mismatch", "actual_direction": "up"},
        "answer": "premise mismatch", "node_history": [],
    }
    result = premise_confirmation(state)
    assert result["premise_check"]["status"] == "confirmed_by_user"
    assert result["awaiting_user_confirmation"] is False


def test_confirmed_mismatch_completes_from_verified_trend_without_root_cause_search(monkeypatch):
    monkeypatch.setattr("app.graph.nodes.premise.interrupt", lambda *_args, **_kwargs: {"action": "confirm"})
    state = {
        "task_type": "anomaly_analysis",
        "user_message": "为什么 2018-08-20 的 GMV 下降了？",
        "trace_id": "premise-correction",
        "awaiting_user_confirmation": True,
        "premise_check": {
            "status": "mismatch", "claimed_direction": "down", "actual_direction": "up",
            "metric_name": "gmv", "metric_display_name": "成交总额",
            "baseline_date": "2018-08-19", "current_date": "2018-08-20",
            "baseline_value": 100.0, "current_value": 125.0, "delta": 25.0,
            "delta_ratio": 25.0,
        },
        "evidence_bundles": [{
            "evidence_id": "ev-premise-trend", "source_type": "mcp", "trusted": True,
            "content": {"result": {"status": "ok", "trend_result": {
                "status": "ok", "metric_name": "gmv", "unit": "BRL",
                "points": [
                    {"bucket": "2018-08-19", "value": 100.0},
                    {"bucket": "2018-08-20", "value": 125.0},
                ],
            }}},
        }],
        "node_history": [],
    }

    result = premise_confirmation(state)

    assert result["premise_check"]["status"] == "confirmed_by_user"
    assert result["premise_correction_finalized"] is True
    assert result["awaiting_user_confirmation"] is False
    assert not result.get("errors")
    assert "实际上升" in result["answer"]
    assert "不足以解释" in result["answer"]
    assert result["analysis_result"]["abstain"] is True
    assert result["analysis_result"]["evidence_ids"] == ["ev-premise-trend"]
    assert result["analysis_presentation"]["changes"][0]["delta_value"] == 25.0
    assert "diagnostic_plan" not in result

    from app.graph.workflow import _route_after_premise_confirmation

    assert _route_after_premise_confirmation(result) == "persist_trace"


def test_premise_check_abstains_when_trend_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "app.graph.nodes.data_query.query_business_data",
        lambda **_: pytest.fail("premise_check must not issue an unsigned fallback query"),
    )
    state = {
        "task_type": "anomaly_analysis",
        "user_message": "为什么 2018-08-20 的 GMV 下降了？",
        "trace_id": "premise-unavailable",
        "tool_result": {
            "tool_name": "registry_trend",
            "evidence_id": "trend-evidence",
            "result": {"status": "ok", "trend_result": {
                "metric_name": "gmv", "points": [{"bucket": "2018-08-20", "value": 9.0}],
            }},
        },
        "query_results": [], "tool_calls": [], "node_history": [],
        "evidence_bundles": [{
            "evidence_id": "trend-evidence", "source_type": "mcp", "trusted": True,
            "content": {"question": "查询2018-08-20 GMV趋势"},
        }],
    }
    result = premise_check(state)
    assert result["premise_check"]["status"] == "unavailable"
    assert result["premise_check"]["required_valid_points"] == 2
    assert result["premise_check"]["observed_valid_points"] == 1
    assert result["terminal_code"] == "DATA_INSUFFICIENT"
    assert result["analysis_presentation"]["status"] == "partial"
    assert result["analysis_presentation"]["evidence"][0]["evidence_id"] == "trend-evidence"
    assert "不会继续" in result["answer"]
    from app.graph.workflow import _route_after_premise_check

    assert _route_after_premise_check(result) == "persist_trace"
    assert result["tool_calls"] == []


def test_premise_query_uses_primary_gmv_claim_when_secondary_factor_moves_oppositely(monkeypatch):
    query = "2017-05-15 GMV\u4e0b\u6ed1\u4f46\u8ba2\u5355\u91cf\u589e\u957f\uff0c\u5f52\u56e0\u4e8e\u4ec0\u4e48"
    monkeypatch.setattr(
        "app.core.time_expression.data_time_bounds",
        lambda: (date(2017, 1, 1), date(2018, 8, 31)),
    )

    hint = premise_query_hint(query)

    assert hint is not None
    assert "gmv" in hint
    assert "order_count" not in hint


def test_iso_dates_and_premise_hint_accept_chinese_adjacency_and_spaces(monkeypatch):
    monkeypatch.setattr(
        "app.core.time_expression.data_time_bounds",
        lambda: (date(2024, 4, 28), date(2024, 5, 4)),
    )
    adjacent = "为什么2024-05-04的订单量下降了？"
    spaced = "为什么 2024-05-04 的订单量下降了？"

    assert extract_iso_dates(adjacent) == ["2024-05-04"]
    assert extract_iso_dates(spaced) == ["2024-05-04"]
    adjacent_plan = _premise_plan(adjacent)
    spaced_plan = _premise_plan(spaced)
    assert adjacent_plan is not None and spaced_plan is not None
    assert (adjacent_plan.query_type, adjacent_plan.metric_name, adjacent_plan.start_date, adjacent_plan.end_date) == (
        "metric_trend", "order_count", "2024-05-03", "2024-05-04",
    )
    assert adjacent_plan.model_dump() == spaced_plan.model_dump()


def test_explicit_comparison_roles_are_preserved_while_query_window_is_chronological(monkeypatch):
    bounds = (date(2024, 4, 28), date(2024, 5, 4))
    monkeypatch.setattr(
        "app.core.time_expression.data_time_bounds",
        lambda: bounds,
    )
    monkeypatch.setattr(
        "app.graph.nodes.premise.data_time_bounds",
        lambda: bounds,
    )
    question = "基线日期：2024-05-04，当前日期：2024-05-03，订单量下降了"
    plan = _premise_plan(question)
    assert plan is not None
    assert (plan.start_date, plan.end_date) == ("2024-05-03", "2024-05-04")
    values = _trend_values_for_question({
        "points": [
            {"bucket": "2024-05-03", "value": 6.0},
            {"bucket": "2024-05-04", "value": 10.0},
        ],
    }, question)
    assert values == (10.0, 6.0, "2024-05-04", "2024-05-03")


def test_missing_premise_date_does_not_invent_a_window(monkeypatch):
    monkeypatch.setattr(
        "app.core.time_expression.data_time_bounds",
        lambda: (date(2024, 4, 28), date(2024, 5, 4)),
    )
    assert _premise_plan("为什么订单量下降了？") is None


def test_ordinary_single_day_value_query_is_not_rewritten_as_a_trend(monkeypatch):
    from app.agents.data_query_agent import build_metric_query_plan

    monkeypatch.setattr(
        "app.core.time_expression.data_time_bounds",
        lambda: (date(2024, 4, 28), date(2024, 5, 4)),
    )
    question = "查询2024-05-04的订单量"

    assert premise_query_hint(question) is None
    plan = build_metric_query_plan(question)
    assert plan.query_type == "metric_value"
    assert plan.metric_name == "order_count"
    assert plan.metric_date == "2024-05-04"


def test_shared_resolver_recognizes_iso_date_adjacent_to_chinese_text():
    resolved = resolve_expression(
        "为什么2024-05-04的订单量下降了？",
        data_bounds=(date(2024, 4, 28), date(2024, 5, 4)),
    )
    assert resolved is not None
    assert resolved.start_date == date(2024, 5, 4)
    assert resolved.end_date == date(2024, 5, 4)


def test_diagnostic_target_uses_metric_of_decline_not_first_mentioned_factor():
    from app.graph.nodes.diagnostic import _target_metric

    state = {"user_message": "2017-07-31 \u5c0f\u5e45\u8ba2\u5355\u53d8\u5316\u662f\u5426\u89e3\u91ca GMV \u4e0b\u964d\uff1f"}

    assert _target_metric(state) == "gmv"


def test_reverse_opposing_fixture_query_matches_its_gmv_direction():
    from evaluation.scenario_library import generate_scenarios

    case = next(item for item in generate_scenarios() if item.scenario_id == "gen-reverse_opposing-58")

    assert case.values["gmv"] < 1000.0
    assert "\u4e0b\u964d" in case.query
