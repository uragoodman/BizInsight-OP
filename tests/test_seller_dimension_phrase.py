"""Seller grouping phrase detection for the registry planner."""

from app.agents.data_query_agent import _extract_dimension


def test_each_seller_grouping_phrase_is_recognized() -> None:
    assert _extract_dimension("各店铺的 GMV") == "seller"
    assert _extract_dimension("每个店铺的订单量") == "seller"


def test_ordinary_seller_metric_wording_is_not_a_grouping() -> None:
    assert _extract_dimension("店铺平均商品售价") is None
