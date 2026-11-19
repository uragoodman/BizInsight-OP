from __future__ import annotations

from app.agents.data_query_agent import is_explicit_standard_metric_request
from app.context.compiler import compile_turn_context
from app.metrics.registry import get_registry


def _compiled(question: str) -> dict:
    return compile_turn_context({"user_message": question})["context_compilation"]


def test_generic_business_terms_are_not_exact_metric_requests() -> None:
    assert is_explicit_standard_metric_request("按购买频次和金额给客户分层") is False
    assert is_explicit_standard_metric_request("统计订单") is False


def test_specific_order_and_customer_count_phrases_remain_exact() -> None:
    assert is_explicit_standard_metric_request("按客户统计订单量") is True
    assert is_explicit_standard_metric_request("统计客户数量") is True


def test_customer_word_does_not_bind_customer_count_during_compilation() -> None:
    compiled = _compiled("按购买频次和金额给客户分层")
    assert compiled["request_contract"].get("metric_id") != "customer_count"
    assert compiled.get("known_slots", {}).get("metric", {}).get("value") != "customer_count"


def test_customer_grouping_still_binds_order_count() -> None:
    compiled = _compiled("按客户统计订单量")
    assert compiled["request_contract"].get("metric_id") == "order_count"


def test_weekly_seller_average_delay_remains_a_metric_query() -> None:
    compiled = _compiled("按周统计店铺平均延迟送达天数")
    assert compiled["request_contract"].get("metric_id") == "seller_avg_late_delivery_days"
    assert compiled["request_contract"].get("parse_status") == "resolved"


def test_generic_terms_remain_planning_semantics_without_becoming_exact_aliases() -> None:
    expected = {
        "customer_count": "客户",
        "order_count": "订单",
        "repeat_purchase_rate": "复购",
        "delivery_delay_rate": "延迟",
        "review_score_avg": "评分",
        "cancel_rate": "取消",
    }
    registry = get_registry()
    for metric_id, term in expected.items():
        planning = registry.get(metric_id).planning
        assert term in planning.semantic_terms
        assert term not in planning.exact_aliases
