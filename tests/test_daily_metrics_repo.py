from datetime import date
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.data.database import engine
from app.data.repositories.daily_metrics import (
    SUPPORTED_METRICS,
    breakdown_by_dimension_period,
    breakdown_by_dimension,
    compare_metric,
    query_metric,
    query_metric_period,
    query_metric_trend,
)


def test_supported_metrics_contains_all_business_metrics() -> None:
    assert {
        "gmv",
        "order_count",
        "avg_order_value",
        "cancel_rate",
        "freight_value",
        "review_score_avg",
        "delivery_delay_rate",
        "customer_count",
        "repeat_customer_count",
        "repeat_purchase_rate",
        "avg_items_per_order",
        "avg_delivery_lead_time",
        "on_time_delivery_rate",
        "bad_review_rate",
        "freight_rate",
    }.issubset(SUPPORTED_METRICS)


def test_query_metric_rejects_unknown_metric() -> None:
    try:
        query_metric("unknown_metric", "2018-08-10")
    except ValueError as exc:
        assert "Unsupported metric" in str(exc)
    else:
        raise AssertionError("query_metric should reject unsupported metric names")


def test_query_metric_supports_second_batch_metrics() -> None:
    metrics = [
        "customer_count",
        "repeat_customer_count",
        "repeat_purchase_rate",
        "avg_items_per_order",
        "avg_delivery_lead_time",
        "on_time_delivery_rate",
        "bad_review_rate",
        "freight_rate",
    ]
    for metric_name in metrics:
        result = query_metric(metric_name, "2018-08-10")
        assert result["metric_name"] == metric_name
        assert result["metric_date"] == "2018-08-10"
        assert result["status"] in {"ok", "not_found"}


def test_query_metric_trend_supports_second_batch_metric() -> None:
    result = query_metric_trend("customer_count", "2018-08-08", "2018-08-10")
    assert result["metric_name"] == "customer_count"
    assert result["granularity"] == "day"
    assert len(result["points"]) == 3
    assert result["status"] in {"ok", "not_found"}


def test_compare_metric_supports_second_batch_metric() -> None:
    result = compare_metric("repeat_purchase_rate", "2018-08-09", "2018-08-10")
    assert result["metric_name"] == "repeat_purchase_rate"
    assert result["left_date"] == "2018-08-09"
    assert result["right_date"] == "2018-08-10"
    assert result["status"] in {"ok", "not_found"}


def test_breakdown_supports_seller_state_dimension() -> None:
    result = breakdown_by_dimension("gmv", "2018-08-10", "seller_state", 5)
    assert result["metric_name"] == "gmv"
    assert result["dimension"] == "seller_state"
    assert result["status"] in {"ok", "not_found"}
    assert len(result["items"]) > 0


def test_breakdown_supports_product_category_dimension() -> None:
    result = breakdown_by_dimension("order_count", "2018-08-10", "product_category", 5)
    assert result["metric_name"] == "order_count"
    assert result["dimension"] == "product_category"
    assert result["status"] in {"ok", "not_found"}
    assert len(result["items"]) > 0


def test_breakdown_supports_payment_type_dimension() -> None:
    result = breakdown_by_dimension("customer_count", "2018-08-10", "payment_type", 5)
    assert result["metric_name"] == "customer_count"
    assert result["dimension"] == "payment_type"
    assert result["status"] in {"ok", "not_found"}
    assert len(result["items"]) > 0


def test_gmv_payment_type_allocation_reconciles_to_period_gmv() -> None:
    start_date, end_date = "2016-01-01", "2020-01-01"
    result = breakdown_by_dimension_period(
        "gmv", start_date, end_date, "payment_type", limit=20,
    )
    gmv = query_metric_period("gmv", start_date, end_date)["value"]

    assert result["status"] == "ok"
    assert result["allocation_method"] == "order_payment_value_ratio"
    assert "payment_value" in result["lineage_note"]
    assert any(item["dimension_value"] == "UNKNOWN" for item in result["items"])
    assert round(sum(item["value"] for item in result["items"]), 2) == round(gmv, 2)


@pytest.fixture
def payment_allocation_edge_orders():
    """Seed isolated delivered orders for missing, zero, and rounded payments."""
    token = uuid4().hex
    cases = [
        (f"pay_edge_{token}_missing", date(2099, 12, 1), 10.00, []),
        (f"pay_edge_{token}_zero", date(2099, 12, 2), 20.00, [("voucher", 0.00)]),
        (
            f"pay_edge_{token}_rounding", date(2099, 12, 3), 0.05,
            [("boleto", 1.00), ("credit_card", 1.00)],
        ),
    ]
    with engine.begin() as connection:
        connection.execute(text("RESET ROLE"))
        connection.execute(text("""
            INSERT INTO analytics.fact_orders(order_id, order_status, purchase_date)
            VALUES (:order_id, 'delivered', :purchase_date)
        """), [
            {"order_id": order_id, "purchase_date": purchase_date}
            for order_id, purchase_date, _, _ in cases
        ])
        connection.execute(text("""
            INSERT INTO analytics.fact_order_items(
                order_id, order_item_id, seller_id, order_status, price, freight_value
            ) VALUES (:order_id, 1, :seller_id, 'delivered', :price, 0)
        """), [
            {"order_id": order_id, "seller_id": f"test_seller_{token}", "price": gmv}
            for order_id, _, gmv, _ in cases
        ])
        payment_rows = [
            {
                "order_id": order_id,
                "payment_sequential": index,
                "payment_type": payment_type,
                "payment_value": payment_value,
            }
            for order_id, _, _, payments in cases
            for index, (payment_type, payment_value) in enumerate(payments, start=1)
        ]
        if payment_rows:
            connection.execute(text("""
                INSERT INTO raw.order_payments(
                    order_id, payment_sequential, payment_type, payment_value
                ) VALUES (:order_id, :payment_sequential, :payment_type, :payment_value)
            """), payment_rows)
    try:
        yield cases
    finally:
        with engine.begin() as connection:
            connection.execute(text("RESET ROLE"))
            for order_id, _, _, _ in cases:
                connection.execute(text(
                    "DELETE FROM raw.order_payments WHERE order_id=:order_id"
                ), {"order_id": order_id})
                connection.execute(text(
                    "DELETE FROM analytics.fact_order_items WHERE order_id=:order_id"
                ), {"order_id": order_id})
                connection.execute(text(
                    "DELETE FROM analytics.fact_orders WHERE order_id=:order_id"
                ), {"order_id": order_id})


def test_gmv_payment_type_allocation_reconciles_each_edge_order(payment_allocation_edge_orders) -> None:
    from app.data.repositories.daily_metrics import breakdown_by_dimension

    expected_by_date = {
        date(2099, 12, 1): {"UNKNOWN": 10.00},
        date(2099, 12, 2): {"UNKNOWN": 20.00},
        # The one-cent rounding residual is assigned deterministically to the
        # largest payment type; ties are broken by payment_type ascending.
        date(2099, 12, 3): {"boleto": 0.02, "credit_card": 0.03},
    }

    for order_id, purchase_date, expected_gmv, _ in payment_allocation_edge_orders:
        result = breakdown_by_dimension("gmv", purchase_date, "payment_type", limit=20)
        values = {item["dimension_value"]: item["value"] for item in result["items"]}
        expected = expected_by_date[purchase_date]

        assert result["status"] == "ok", f"{order_id}: {result}"
        assert values == pytest.approx(expected)
        assert sum(values.values()) == pytest.approx(expected_gmv, abs=0.001)
