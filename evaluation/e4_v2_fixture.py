# -*- coding: utf-8 -*-
"""Deterministic E4.1 fixture and 39-case seed matrix.

The fixture is intentionally small and explicit. Every order has one item, one
payment row and one delivered status so value, trend, compare, breakdown and
behavior expectations can be derived without calling the product planner.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any

# order_id, date, customer_id, customer_unique_id, customer_state, customer_city,
# seller_id, seller_state, product_id, product_category, price, freight, payment_type
ORDERS: list[dict[str, Any]] = [
    {"order_id": "o01", "date": "2024-01-02", "customer_id": "c01", "customer_unique_id": "customer-01", "customer_state": "SP", "customer_city": "Sao Paulo", "seller_id": "s01", "seller_state": "SP", "product_id": "p01", "product_category": "books", "price": 100.0, "freight": 10.0, "payment_type": "credit_card"},
    {"order_id": "o02", "date": "2024-01-03", "customer_id": "c02", "customer_unique_id": "customer-02", "customer_state": "RJ", "customer_city": "Rio de Janeiro", "seller_id": "s02", "seller_state": "RJ", "product_id": "p02", "product_category": "toys", "price": 200.0, "freight": 20.0, "payment_type": "boleto"},
    {"order_id": "o03", "date": "2024-01-04", "customer_id": "c03", "customer_unique_id": "customer-03", "customer_state": "MG", "customer_city": "Belo Horizonte", "seller_id": "s03", "seller_state": "MG", "product_id": "p03", "product_category": "electronics", "price": 300.0, "freight": 30.0, "payment_type": "voucher"},
    {"order_id": "o04", "date": "2024-01-05", "customer_id": "c01", "customer_unique_id": "customer-01", "customer_state": "SP", "customer_city": "Sao Paulo", "seller_id": "s01", "seller_state": "SP", "product_id": "p01", "product_category": "books", "price": 50.0, "freight": 5.0, "payment_type": "credit_card"},
    {"order_id": "o05", "date": "2024-01-08", "customer_id": "c04", "customer_unique_id": "customer-04", "customer_state": "SP", "customer_city": "Campinas", "seller_id": "s04", "seller_state": "SP", "product_id": "p04", "product_category": "books", "price": 150.0, "freight": 15.0, "payment_type": "boleto"},
    {"order_id": "o06", "date": "2024-01-09", "customer_id": "c05", "customer_unique_id": "customer-05", "customer_state": "RJ", "customer_city": "Niteroi", "seller_id": "s02", "seller_state": "RJ", "product_id": "p05", "product_category": "toys", "price": 80.0, "freight": 8.0, "payment_type": "credit_card"},
    {"order_id": "o07", "date": "2024-01-10", "customer_id": "c06", "customer_unique_id": "customer-06", "customer_state": "MG", "customer_city": "Uberlandia", "seller_id": "s03", "seller_state": "MG", "product_id": "p06", "product_category": "electronics", "price": 120.0, "freight": 12.0, "payment_type": "voucher"},
    {"order_id": "o08", "date": "2024-01-11", "customer_id": "c02", "customer_unique_id": "customer-02", "customer_state": "RJ", "customer_city": "Rio de Janeiro", "seller_id": "s02", "seller_state": "RJ", "product_id": "p02", "product_category": "toys", "price": 60.0, "freight": 6.0, "payment_type": "boleto"},
    {"order_id": "o09", "date": "2024-01-12", "customer_id": "c03", "customer_unique_id": "customer-03", "customer_state": "MG", "customer_city": "Belo Horizonte", "seller_id": "s03", "seller_state": "MG", "product_id": "p03", "product_category": "electronics", "price": 400.0, "freight": 40.0, "payment_type": "credit_card"},
    {"order_id": "o10", "date": "2024-01-15", "customer_id": "c01", "customer_unique_id": "customer-01", "customer_state": "SP", "customer_city": "Sao Paulo", "seller_id": "s01", "seller_state": "SP", "product_id": "p01", "product_category": "books", "price": 90.0, "freight": 9.0, "payment_type": "voucher"},
    {"order_id": "o11", "date": "2024-01-16", "customer_id": "c04", "customer_unique_id": "customer-04", "customer_state": "SP", "customer_city": "Campinas", "seller_id": "s04", "seller_state": "SP", "product_id": "p04", "product_category": "books", "price": 210.0, "freight": 21.0, "payment_type": "credit_card"},
    {"order_id": "o12", "date": "2024-01-17", "customer_id": "c05", "customer_unique_id": "customer-05", "customer_state": "RJ", "customer_city": "Niteroi", "seller_id": "s02", "seller_state": "RJ", "product_id": "p05", "product_category": "toys", "price": 30.0, "freight": 3.0, "payment_type": "boleto"},
    {"order_id": "o13", "date": "2024-01-18", "customer_id": "c06", "customer_unique_id": "customer-06", "customer_state": "MG", "customer_city": "Uberlandia", "seller_id": "s03", "seller_state": "MG", "product_id": "p06", "product_category": "electronics", "price": 70.0, "freight": 7.0, "payment_type": "voucher"},
    {"order_id": "o14", "date": "2024-01-19", "customer_id": "c01", "customer_unique_id": "customer-01", "customer_state": "SP", "customer_city": "Sao Paulo", "seller_id": "s01", "seller_state": "SP", "product_id": "p01", "product_category": "books", "price": 120.0, "freight": 12.0, "payment_type": "credit_card"},
    {"order_id": "o15", "date": "2024-01-22", "customer_id": "c02", "customer_unique_id": "customer-02", "customer_state": "RJ", "customer_city": "Rio de Janeiro", "seller_id": "s02", "seller_state": "RJ", "product_id": "p02", "product_category": "toys", "price": 250.0, "freight": 25.0, "payment_type": "boleto"},
    {"order_id": "o16", "date": "2024-01-23", "customer_id": "c03", "customer_unique_id": "customer-03", "customer_state": "MG", "customer_city": "Belo Horizonte", "seller_id": "s03", "seller_state": "MG", "product_id": "p03", "product_category": "electronics", "price": 180.0, "freight": 18.0, "payment_type": "credit_card"},
    {"order_id": "o17", "date": "2024-01-24", "customer_id": "c04", "customer_unique_id": "customer-04", "customer_state": "SP", "customer_city": "Campinas", "seller_id": "s04", "seller_state": "SP", "product_id": "p04", "product_category": "books", "price": 40.0, "freight": 4.0, "payment_type": "voucher"},
    {"order_id": "o18", "date": "2024-01-25", "customer_id": "c05", "customer_unique_id": "customer-05", "customer_state": "RJ", "customer_city": "Niteroi", "seller_id": "s02", "seller_state": "RJ", "product_id": "p05", "product_category": "toys", "price": 95.0, "freight": 9.5, "payment_type": "credit_card"},
    {"order_id": "o19", "date": "2024-01-26", "customer_id": "c06", "customer_unique_id": "customer-06", "customer_state": "MG", "customer_city": "Uberlandia", "seller_id": "s03", "seller_state": "MG", "product_id": "p06", "product_category": "electronics", "price": 60.0, "freight": 6.0, "payment_type": "boleto"},
    {"order_id": "o20", "date": "2024-01-29", "customer_id": "c01", "customer_unique_id": "customer-01", "customer_state": "SP", "customer_city": "Sao Paulo", "seller_id": "s01", "seller_state": "SP", "product_id": "p01", "product_category": "books", "price": 300.0, "freight": 30.0, "payment_type": "voucher"},
    {"order_id": "o21", "date": "2024-01-30", "customer_id": "c02", "customer_unique_id": "customer-02", "customer_state": "RJ", "customer_city": "Rio de Janeiro", "seller_id": "s02", "seller_state": "RJ", "product_id": "p02", "product_category": "toys", "price": 130.0, "freight": 13.0, "payment_type": "credit_card"},
    {"order_id": "o22", "date": "2024-01-31", "customer_id": "c03", "customer_unique_id": "customer-03", "customer_state": "MG", "customer_city": "Belo Horizonte", "seller_id": "s03", "seller_state": "MG", "product_id": "p03", "product_category": "electronics", "price": 220.0, "freight": 22.0, "payment_type": "boleto"},
    {"order_id": "o23", "date": "2024-01-10", "customer_id": "c07", "customer_unique_id": "customer-07", "customer_state": "SP", "customer_city": "Santos", "seller_id": "s01", "seller_state": "SP", "product_id": "p05", "product_category": "toys", "price": 100.0, "freight": 10.0, "payment_type": "credit_card"},
    {"order_id": "o24", "date": "2024-01-16", "customer_id": "c08", "customer_unique_id": "customer-08", "customer_state": "RJ", "customer_city": "Petropolis", "seller_id": "s02", "seller_state": "RJ", "product_id": "p03", "product_category": "electronics", "price": 80.0, "freight": 8.0, "payment_type": "boleto"},
    {"order_id": "o25", "date": "2024-01-22", "customer_id": "c08", "customer_unique_id": "customer-08", "customer_state": "RJ", "customer_city": "Petropolis", "seller_id": "s02", "seller_state": "RJ", "product_id": "p01", "product_category": "books", "price": 25.0, "freight": 2.5, "payment_type": "voucher"},
    {"order_id": "o26", "date": "2024-01-31", "customer_id": "c07", "customer_unique_id": "customer-07", "customer_state": "SP", "customer_city": "Santos", "seller_id": "s01", "seller_state": "SP", "product_id": "p06", "product_category": "electronics", "price": 58.0, "freight": 5.8, "payment_type": "credit_card"},
    {"order_id": "o27", "date": "2024-01-16", "customer_id": "c09", "customer_unique_id": "customer-09", "customer_state": "SP", "customer_city": "Sao Paulo", "seller_id": "s01", "seller_state": "SP", "product_id": "p02", "product_category": "toys", "price": 90.0, "freight": 9.0, "payment_type": "credit_card"},
    {"order_id": "o28", "date": "2024-01-10", "customer_id": "c10", "customer_unique_id": "customer-10", "customer_state": "MG", "customer_city": "Contagem", "seller_id": "s03", "seller_state": "MG", "product_id": "p04", "product_category": "books", "price": 100.0, "freight": 10.0, "payment_type": "boleto"},
]

CUSTOMERS = {
    "c01": ("customer-01", "SP", "Sao Paulo"),
    "c02": ("customer-02", "RJ", "Rio de Janeiro"),
    "c03": ("customer-03", "MG", "Belo Horizonte"),
    "c04": ("customer-04", "SP", "Campinas"),
    "c05": ("customer-05", "RJ", "Niteroi"),
    "c06": ("customer-06", "MG", "Uberlandia"),
    "c07": ("customer-07", "SP", "Santos"),
    "c08": ("customer-08", "RJ", "Petropolis"),
    "c09": ("customer-09", "SP", "Sao Paulo"),
    "c10": ("customer-10", "MG", "Contagem"),
}

SELLERS = {
    "s01": "SP",
    "s02": "RJ",
    "s03": "MG",
    "s04": "SP",
}

PRODUCTS = {
    "p01": "books",
    "p02": "toys",
    "p03": "electronics",
    "p04": "books",
    "p05": "toys",
    "p06": "electronics",
}

SCHEMA_STATEMENTS = (
    "CREATE SCHEMA analytics",
    "CREATE SCHEMA raw",
    """
    CREATE TABLE analytics.daily_metrics (
        metric_date date PRIMARY KEY,
        gmv numeric(12, 2) NOT NULL,
        order_count integer NOT NULL,
        avg_order_value numeric(12, 2) NOT NULL
    )
    """,
    """
    CREATE TABLE analytics.fact_orders (
        order_id text PRIMARY KEY,
        customer_unique_id text,
        purchase_date date NOT NULL,
        order_status text NOT NULL DEFAULT 'delivered',
        item_gross_value numeric(12, 2) NOT NULL DEFAULT 0,
        freight_value numeric(12, 2) NOT NULL DEFAULT 0,
        item_count integer NOT NULL DEFAULT 1,
        customer_state text,
        customer_city text
    )
    """,
    """
    CREATE TABLE raw.orders (
        order_id text PRIMARY KEY,
        customer_id text,
        order_purchase_timestamp timestamp NOT NULL,
        order_delivered_customer_date timestamp,
        order_estimated_delivery_date timestamp,
        order_status text NOT NULL
    )
    """,
    """
    CREATE TABLE raw.order_items (
        order_id text NOT NULL,
        product_id text NOT NULL,
        seller_id text,
        price numeric(12, 2) NOT NULL,
        freight_value numeric(12, 2) NOT NULL
    )
    """,
    "CREATE TABLE raw.customers (customer_id text PRIMARY KEY, customer_unique_id text)",
    "CREATE TABLE raw.sellers (seller_id text PRIMARY KEY, seller_state text)",
    "CREATE TABLE raw.products (product_id text PRIMARY KEY, product_category_name text)",
    "CREATE TABLE raw.order_reviews (order_id text, review_score integer)",
    "CREATE TABLE raw.order_payments (order_id text, payment_type text, payment_value numeric(12, 2))",
    """
    CREATE VIEW analytics.v_delivered_orders AS
    SELECT order_id, purchase_date, item_gross_value, freight_value
    FROM analytics.fact_orders
    WHERE order_status = 'delivered'
    """,
    """
    CREATE VIEW analytics.v_delivered_order_items AS
    SELECT order_id, item_gross_value AS price, freight_value
    FROM analytics.fact_orders
    WHERE order_status = 'delivered'
    """,
)


def _money(value: float) -> float:
    return round(float(value), 2)


def _orders_on(day: str) -> list[dict[str, Any]]:
    return [order for order in ORDERS if order["date"] == day]


def _orders_between(start: str, end: str) -> list[dict[str, Any]]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    return [
        order for order in ORDERS
        if first <= date.fromisoformat(order["date"]) <= last
    ]


def _gmv(orders: list[dict[str, Any]]) -> float:
    return _money(sum(order["price"] + order["freight"] for order in orders))


def _freight(orders: list[dict[str, Any]]) -> float:
    return _money(sum(order["freight"] for order in orders))


def value(day: str, metric: str) -> float | None:
    orders = _orders_on(day)
    if not orders:
        return None
    if metric == "gmv":
        return _gmv(orders)
    if metric == "order_count":
        return float(len(orders))
    if metric == "avg_order_value":
        return _money(_gmv(orders) / len(orders))
    if metric == "freight_value":
        return _freight(orders)
    raise ValueError(f"Unsupported fixture metric: {metric}")


def trend(start: str, end: str, metric: str) -> list[dict[str, Any]]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    points = []
    current = first
    while current <= last:
        points.append({
            "bucket": current.isoformat(),
            "value": value(current.isoformat(), metric),
        })
        current += timedelta(days=1)
    return points


def comparison(metric: str, baseline: str, current: str) -> dict[str, Any]:
    baseline_value = value(baseline, metric)
    current_value = value(current, metric)
    delta_value = None
    delta_ratio = None
    if baseline_value is not None and current_value is not None:
        delta_value = _money(current_value - baseline_value)
        if baseline_value != 0:
            delta_ratio = _money((current_value - baseline_value) * 100.0 / baseline_value)
    return {
        "baseline_date": baseline,
        "current_date": current,
        "baseline_value": baseline_value,
        "current_value": current_value,
        "delta_value": delta_value,
        "delta_ratio": delta_ratio,
    }


_MONTH_SEGMENTS = (
    (1, 10, "月初(1-10日)"),
    (11, 20, "月中(11-20日)"),
    (21, 31, "月末(21-31日)"),
)
_WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def _dimension_value(order: dict[str, Any], dimension: str) -> str:
    if dimension == "product_category":
        return order["product_category"]
    if dimension == "customer_state":
        return order["customer_state"]
    if dimension == "customer_city":
        return order["customer_city"]
    if dimension == "seller_state":
        return order["seller_state"]
    if dimension == "payment_type":
        return order["payment_type"]
    if dimension == "month_position":
        day = date.fromisoformat(order["date"]).day
        return next(label for start, end, label in _MONTH_SEGMENTS if start <= day <= end)
    if dimension == "day_of_week":
        return _WEEKDAYS[date.fromisoformat(order["date"]).weekday()]
    raise ValueError(f"Unsupported fixture dimension: {dimension}")


def breakdown(day: str, metric: str, dimension: str) -> dict[str, Any]:
    if dimension in {"month_position", "day_of_week"}:
        target = date.fromisoformat(day)
        orders = [
            order for order in ORDERS
            if date.fromisoformat(order["date"]).year == target.year
            and date.fromisoformat(order["date"]).month == target.month
        ]
    else:
        orders = _orders_on(day)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for order in orders:
        grouped[_dimension_value(order, dimension)].append(order)
    items: dict[str, float] = {}
    for label, group in grouped.items():
        if metric == "gmv":
            items[label] = _gmv(group)
        elif metric == "freight_value":
            items[label] = _freight(group)
        elif metric == "order_count":
            items[label] = float(len(group))
        elif metric == "avg_order_value":
            items[label] = _money(_gmv(group) / len(group))
        else:
            raise ValueError(f"Unsupported fixture metric: {metric}")
    return items


def _seed_sql() -> tuple[str, ...]:
    statements: list[str] = []
    daily: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for order in ORDERS:
        daily[order["date"]].append(order)
    daily_values = ", ".join(
        f"('{day}', {_gmv(orders):.2f}, {len(orders)}, {_money(_gmv(orders) / len(orders)):.2f})"
        for day, orders in sorted(daily.items())
    )
    statements.append(f"INSERT INTO analytics.daily_metrics VALUES {daily_values}")
    statements.append(
        "INSERT INTO analytics.fact_orders "
        "(order_id, customer_unique_id, purchase_date, order_status, item_gross_value, freight_value, item_count, customer_state, customer_city) VALUES "
        + ", ".join(
            f"('{order['order_id']}', '{order['customer_unique_id']}', '{order['date']}', 'delivered', "
            f"{order['price']:.2f}, {order['freight']:.2f}, 1, '{order['customer_state']}', '{order['customer_city']}')"
            for order in ORDERS
        )
    )
    statements.append(
        "INSERT INTO raw.orders VALUES "
        + ", ".join(
            f"('{order['order_id']}', '{order['customer_id']}', '{order['date']} 09:00:00', "
            f"'{order['date']} 12:00:00', '{order['date']} 18:00:00', 'delivered')"
            for order in ORDERS
        )
    )
    statements.append(
        "INSERT INTO raw.order_items VALUES "
        + ", ".join(
            f"('{order['order_id']}', '{order['product_id']}', '{order['seller_id']}', "
            f"{order['price']:.2f}, {order['freight']:.2f})"
            for order in ORDERS
        )
    )
    statements.append(
        "INSERT INTO raw.customers VALUES "
        + ", ".join(f"('{cid}', '{unique}')" for cid, (unique, _, _) in CUSTOMERS.items())
    )
    statements.append(
        "INSERT INTO raw.sellers VALUES "
        + ", ".join(f"('{seller_id}', '{state}')" for seller_id, state in SELLERS.items())
    )
    statements.append(
        "INSERT INTO raw.products VALUES "
        + ", ".join(f"('{product_id}', '{category}')" for product_id, category in PRODUCTS.items())
    )
    statements.append(
        "INSERT INTO raw.order_payments VALUES "
        + ", ".join(
            f"('{order['order_id']}', '{order['payment_type']}', {_money(order['price'] + order['freight']):.2f})"
            for order in ORDERS
        )
    )
    return tuple(statements)


SEED_STATEMENTS = _seed_sql()


def _display_metric(metric: str) -> str:
    return {
        "gmv": "GMV",
        "order_count": "订单量",
        "avg_order_value": "客单价",
        "freight_value": "运费",
    }[metric]


def _case(
    case_id: str,
    question: str,
    expected_behavior: str,
    expected: dict[str, Any],
    *,
    allowed_metrics: list[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "question": question,
        "allowed_metrics": allowed_metrics or ["gmv", "order_count", "avg_order_value", "freight_value"],
        "expected_behavior": expected_behavior,
        "expected": expected,
        "tolerance": 0.01,
        "snapshot": "e4-fixture-v2",
        "notes": notes,
    }


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    allowed = ["gmv", "order_count", "avg_order_value", "freight_value"]

    def add_value(metric: str, day: str) -> None:
        cases.append(_case(
            f"e4-value-{metric}-{day}",
            f"查询 {day} 的 {_display_metric(metric)}",
            "result",
            {
                "decision_status": "confirmed",
                "query_type": "metric_value",
                "metric_name": metric,
                "metric_date": day,
                "value": value(day, metric),
            },
            allowed_metrics=allowed,
            notes=f"daily_metrics {metric}",
        ))

    def add_trend(metric: str, start: str, end: str) -> None:
        cases.append(_case(
            f"e4-trend-{metric}-{start}-{end}",
            f"{start} 到 {end} 的 {_display_metric(metric)} 趋势",
            "result",
            {
                "decision_status": "confirmed",
                "query_type": "metric_trend",
                "metric_name": metric,
                "start_date": start,
                "end_date": end,
                "points": trend(start, end, metric),
            },
            allowed_metrics=allowed,
            notes="daily series, including missing-date nulls",
        ))

    def add_compare(metric: str, baseline: str, current: str) -> None:
        expected = comparison(metric, baseline, current)
        cases.append(_case(
            f"e4-compare-{metric}-{baseline}-{current}",
            f"对比 {baseline} 和 {current} 的 {_display_metric(metric)}",
            "result",
            {
                "decision_status": "confirmed",
                "query_type": "metric_compare",
                "metric_name": metric,
                **expected,
            },
            allowed_metrics=allowed,
            notes="current minus baseline, ratio in percent",
        ))

    def add_breakdown(metric: str, day: str, dimension: str, question: str) -> None:
        cases.append(_case(
            f"e4-breakdown-{dimension}-{metric}-{day}",
            question,
            "result",
            {
                "decision_status": "confirmed",
                "query_type": "metric_breakdown",
                "metric_name": metric,
                "metric_date": day,
                "dimension": dimension,
                "items": breakdown(day, metric, dimension),
            },
            allowed_metrics=allowed,
            notes=f"{dimension} via fixture rows",
        ))

    for metric, day in (
        ("gmv", "2024-01-10"),
        ("gmv", "2024-01-12"),
        ("gmv", "2024-01-31"),
        ("order_count", "2024-01-10"),
        ("order_count", "2024-01-16"),
        ("avg_order_value", "2024-01-10"),
        ("avg_order_value", "2024-01-16"),
        ("freight_value", "2024-01-10"),
    ):
        add_value(metric, day)
    for metric, start, end in (
        ("gmv", "2024-01-08", "2024-01-12"),
        ("order_count", "2024-01-08", "2024-01-12"),
        ("avg_order_value", "2024-01-08", "2024-01-12"),
        ("gmv", "2024-01-22", "2024-01-31"),
    ):
        add_trend(metric, start, end)
    for metric, baseline, current in (
        ("gmv", "2024-01-09", "2024-01-10"),
        ("gmv", "2024-01-10", "2024-01-12"),
        ("order_count", "2024-01-10", "2024-01-16"),
        ("avg_order_value", "2024-01-10", "2024-01-16"),
        ("freight_value", "2024-01-10", "2024-01-12"),
    ):
        add_compare(metric, baseline, current)
    for dimension, day, question in (
        ("product_category", "2024-01-10", "按品类拆解 2024-01-10 的 GMV"),
        ("product_category", "2024-01-16", "按品类拆解 2024-01-16 的 GMV"),
        ("customer_state", "2024-01-10", "按州拆解 2024-01-10 的 GMV"),
        ("customer_city", "2024-01-10", "按城市拆解 2024-01-10 的 GMV"),
        ("seller_state", "2024-01-10", "按卖家州拆解 2024-01-10 的 GMV"),
        ("payment_type", "2024-01-10", "按支付方式拆解 2024-01-10 的 GMV"),
    ):
        add_breakdown("gmv", day, dimension, question)
    cases.append(_case(
        "e4-breakdown-customer_state-avg_order_value-2024-01-16",
        "按州拆解 2024-01-16 的客单价",
        "result",
        {
            "decision_status": "confirmed",
            "query_type": "metric_breakdown",
            "metric_name": "avg_order_value",
            "metric_date": "2024-01-16",
            "dimension": "customer_state",
            "items": breakdown("2024-01-16", "avg_order_value", "customer_state"),
        },
        allowed_metrics=allowed,
        notes="ratio metric by customer_state",
    ))

    # Behavior cases: empty, clarification, unauthorized.
    for case_id, question, behavior, expected in (
        (
            "e4-empty-gmv-2024-01-13",
            "查询 2024-01-13 的 GMV",
            "empty",
            {"decision_status": "confirmed", "query_type": "metric_value", "metric_name": "gmv", "metric_date": "2024-01-13", "status": "not_found", "value": None},
        ),
        (
            "e4-empty-gmv-2024-02-01",
            "查询 2024-02-01 的 GMV",
            "empty",
            {"decision_status": "confirmed", "query_type": "metric_value", "metric_name": "gmv", "metric_date": "2024-02-01", "status": "not_found", "value": None},
        ),
        (
            "e4-empty-breakdown-2024-01-13",
            "按品类拆解 2024-01-13 的 GMV",
            "empty",
            {"decision_status": "confirmed", "query_type": "metric_breakdown", "metric_name": "gmv", "metric_date": "2024-01-13", "dimension": "product_category", "status": "not_found", "items": {}},
        ),
        (
            "e4-clarify-invalid-date",
            "查询 2024-01-32 的 GMV",
            "clarify",
            {"decision_status": "clarification_required"},
        ),
        (
            "e4-clarify-missing-date",
            "查询 GMV",
            "clarify",
            {"decision_status": "confirmed"},
        ),
        (
            "e4-clarify-unknown-metric",
            "查询 2024-01-10 的转化率",
            "clarify",
            {"decision_status": "clarification_required"},
        ),
        (
            "e4-unauthorized-order-count",
            "查询 2024-01-10 的订单量",
            "unauthorized",
            {"decision_status": "clarification_required", "reason_code": "unsupported_metric"},
        ),
        (
            "e4-unauthorized-aov",
            "查询 2024-01-10 的客单价",
            "unauthorized",
            {"decision_status": "clarification_required", "reason_code": "unsupported_metric"},
        ),
        (
            "e4-unauthorized-mixed-metric",
            "查询 2024-01-10 的 GMV 和订单量",
            "unauthorized",
            {"decision_status": "clarification_required", "reason_code": "scope_rejected"},
        ),
        (
            "e4-unauthorized-dimension",
            "按商品评分拆解 2024-01-10 的 GMV",
            "unauthorized",
            {"decision_status": "clarification_required", "reason_code": "scope_rejected"},
        ),
        (
            "e4-unsupported-month-position",
            "按月内日期位置拆解 2024-01-10 的 GMV",
            "unsupported",
            {"decision_status": "confirmed", "query_type": "metric_breakdown", "metric_name": "gmv", "metric_date": "2024-01-10", "dimension": "month_position"},
        ),
        (
            "e4-unsupported-day-of-week",
            "按周几拆解 2024-01-10 的 GMV",
            "unsupported",
            {"decision_status": "confirmed", "query_type": "metric_breakdown", "metric_name": "gmv", "metric_date": "2024-01-10", "dimension": "day_of_week"},
        ),
    ):
        cases.append(_case(
            case_id,
            question,
            behavior,
            expected,
            allowed_metrics=["gmv"] if behavior == "unauthorized" else allowed,
            notes="behavior matrix",
        ))

    return cases


REPAIR_PROBES = (
    {
        "case_id": "e4-repair-value-transient",
        "tool": "query_gmv",
        "repository_attr": "repository_query_gmv",
        "args": ("2024-01-10",),
        "failures": 1,
        "expected": "repaired",
    },
    {
        "case_id": "e4-repair-trend-transient",
        "tool": "query_metric_trend",
        "repository_attr": "repository_query_metric_trend",
        "args": ("gmv", "2024-01-08", "2024-01-12", "day"),
        "failures": 1,
        "expected": "repaired",
    },
    {
        "case_id": "e4-repair-permanent-control",
        "tool": "query_gmv",
        "repository_attr": "repository_query_gmv",
        "args": ("2024-01-10",),
        "failures": 99,
        "expected": "failed",
    },
)
