"""Static whitelist: handler_key → callable.

Every key referenced in YAML ``implementations.*.handler_key`` must be
registered here.  Adapter handlers in Phase 1 delegate to existing
repository functions; domain-specific handlers are introduced in later phases.
"""

from __future__ import annotations

from app.metrics.contracts import (
    ControlledAnalysisScope,
    MetricQueryRequest,
    PlatformQueryContext,
)

# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------


def _call_value(metric_name: str, request: MetricQueryRequest) -> dict:
    if isinstance(request.context, ControlledAnalysisScope):
        from app.data.repositories import seller_extensions
        if seller_extensions.supports(metric_name):
            return seller_extensions.query(request.context, metric_name, request.metric_date)
        from app.data.repositories.explorer import query_scoped_metric

        return query_scoped_metric(
            request.context, metric_name, metric_date=request.metric_date,
        )
    from app.data.repositories.daily_metrics import query_metric

    date_str = (
        request.metric_date.isoformat()
        if request.metric_date
        else request.start_date.isoformat()
    )
    return query_metric(metric_name, date_str)


def _call_trend(metric_name: str, request: MetricQueryRequest) -> dict:
    if isinstance(request.context, ControlledAnalysisScope):
        from app.data.repositories import seller_extensions
        if seller_extensions.supports(metric_name):
            return seller_extensions.trend(request.context, metric_name, request.start_date, request.end_date)
        from app.data.repositories.explorer import query_scoped_trend

        return query_scoped_trend(
            request.context, metric_name,
            start_date=request.start_date, end_date=request.end_date,
        )
    from app.data.repositories.daily_metrics import query_metric_trend

    return query_metric_trend(
        metric_name,
        request.start_date.isoformat(),
        request.end_date.isoformat(),
        request.granularity,
    )


def _call_compare(metric_name: str, request: MetricQueryRequest) -> dict:
    from app.data.repositories.daily_metrics import compare_metric

    return compare_metric(
        metric_name,
        request.left_date.isoformat(),
        request.right_date.isoformat(),
    )


def _call_breakdown(metric_name: str, request: MetricQueryRequest) -> dict:
    if isinstance(request.context, ControlledAnalysisScope):
        from app.metrics.exceptions import CapabilityNotSupported
        raise CapabilityNotSupported(
            f"Scoped breakdown for {metric_name} must use the dedicated "
            f"scoped MCP path (query_scoped_business_data), not the generic handler"
        )
    from app.data.repositories.daily_metrics import breakdown_by_dimension

    date_str = (
        request.metric_date.isoformat()
        if request.metric_date
        else request.start_date.isoformat()
    )
    return breakdown_by_dimension(
        metric_name, date_str, request.dimension or "product_category",
        request.limit,
    )


def _call_period(metric_name: str, request: MetricQueryRequest) -> dict:
    from app.data.repositories.daily_metrics import query_metric_period

    return query_metric_period(
        metric_name,
        request.start_date.isoformat(),
        request.end_date.isoformat(),
    )


def _call_statistics(metric_name: str, request: MetricQueryRequest) -> dict:
    requested = getattr(request, "statistics", None)
    if isinstance(request.context, ControlledAnalysisScope):
        from app.data.repositories.explorer import query_scoped_statistics

        return query_scoped_statistics(
            request.context,
            metric_name,
            start_date=request.start_date,
            end_date=request.end_date,
            granularity=request.granularity,
            requested=requested,
        )
    from app.data.repositories.daily_metrics import query_metric_statistics

    return query_metric_statistics(
        metric_name,
        request.start_date,
        request.end_date,
        request.granularity,
        requested=requested,
    )


# ---------------------------------------------------------------------------
# Adapter: dispatch by operation
# ---------------------------------------------------------------------------


def _adapter(metric_name: str) -> callable:
    """Return a handler function that delegates to existing repositories."""

    def handler(request: MetricQueryRequest) -> dict:
        dispatch = {
            "value": _call_value,
            "trend": _call_trend,
            "compare": _call_compare,
            "breakdown": _call_breakdown,
            "period": _call_period,
            "statistics": _call_statistics,
        }
        fn = dispatch.get(request.operation)
        if fn is None:
            from app.metrics.exceptions import CapabilityNotSupported
            raise CapabilityNotSupported(
                f"{metric_name}: operation {request.operation!r} not supported"
            )
        return fn(metric_name, request)

    return handler


# ---------------------------------------------------------------------------
# Static handler registration
# ---------------------------------------------------------------------------
# Every key below matches a YAML handler_key across all scope implementations.

HANDLER_REGISTRY: dict[str, callable] = {
    # Sales
    "sales.gmv": _adapter("gmv"),
    "sales.gmv_seller": _adapter("gmv"),  # canonical name; sales_gmv is historical alias
    "sales.order_count": _adapter("order_count"),
    "sales.avg_order_value": _adapter("avg_order_value"),
    "sales.freight_value": _adapter("freight_value"),
    "sales.freight_rate": _adapter("freight_rate"),
    # Customer
    "customer.customer_count": _adapter("customer_count"),
    "customer.repeat_customer_count": _adapter("repeat_customer_count"),
    "customer.repeat_purchase_rate": _adapter("repeat_purchase_rate"),
    "customer.avg_items_per_order": _adapter("avg_items_per_order"),
    # Quality
    "quality.review_score_avg": _adapter("review_score_avg"),
    "quality.bad_review_rate": _adapter("bad_review_rate"),
    "quality.review_comment_rate": _adapter("review_comment_rate"),
    # Payment
    "payment.payment_value": _adapter("payment_value"),
    "payment.installment_order_share": _adapter("installment_order_share"),
    # Fulfillment
    "fulfillment.on_time_delivery_rate": _adapter("on_time_delivery_rate"),
    "fulfillment.delivery_delay_rate": _adapter("delivery_delay_rate"),
    "fulfillment.avg_delivery_lead_time": _adapter("avg_delivery_lead_time"),
    "fulfillment.approval_lead_hours": _adapter("avg_approval_lead_hours"),
    "fulfillment.dispatch_lead_hours": _adapter("avg_dispatch_lead_hours"),
    "fulfillment.delivery_transit_hours": _adapter("avg_delivery_transit_hours"),
    # Other
    "other.cancel_rate": _adapter("cancel_rate"),
    # Explicit seller-level Olist metrics.
    "seller.bad_review_rate": _adapter("seller_order_bad_review_rate"),
    "seller.items_per_order": _adapter("seller_items_per_order"),
    "seller.freight_rate": _adapter("seller_freight_to_gmv_rate"),
    "seller.late_delivery_rate": _adapter("seller_late_delivery_rate"),
    "seller.delivery_days": _adapter("seller_order_to_delivery_days"),
    "seller.attributable_cancel_rate": _adapter("seller_attributable_cancel_rate"),
    "seller.item_gross_value": _adapter("seller_item_gross_value"),
    "seller.avg_item_price": _adapter("seller_avg_item_price"),
    "seller.avg_freight_per_order": _adapter("seller_avg_freight_per_order"),
    "seller.free_shipping_order_rate": _adapter("seller_free_shipping_order_rate"),
    "seller.avg_payment_installments": _adapter("seller_avg_payment_installments"),
    "seller.credit_card_payment_share": _adapter("seller_credit_card_payment_share"),
    "seller.boleto_payment_share": _adapter("seller_boleto_payment_share"),
    "seller.active_product_count": _adapter("seller_active_product_count"),
    "seller.active_category_count": _adapter("seller_active_category_count"),
    "seller.top1_product_gross_share": _adapter("seller_top1_product_gross_share"),
    "seller.top3_category_gross_share": _adapter("seller_top3_category_gross_share"),
    "seller.new_product_gross_share": _adapter("seller_new_product_gross_share"),
    "seller.returning_customer_count": _adapter("seller_historical_returning_customer_count"),
    "seller.returning_customer_rate": _adapter("seller_historical_returning_customer_rate"),
    "seller.new_customer_count": _adapter("seller_new_customer_count"),
    "seller.new_customer_rate": _adapter("seller_new_customer_rate"),
    "seller.single_item_review_score": _adapter("seller_single_item_order_product_review_score_avg"),
    "seller.single_item_bad_review_rate": _adapter("seller_single_item_order_product_bad_review_rate"),
    "seller.product_attributed_review_score": _adapter("seller_product_attributed_review_score_avg"),
    "seller.product_attributed_bad_review_rate": _adapter("seller_product_attributed_bad_review_rate"),
    "seller.avg_early_delivery_days": _adapter("seller_avg_early_delivery_days"),
    "seller.avg_late_delivery_days": _adapter("seller_avg_late_delivery_days"),
    "seller.delivery_estimate_abs_error_days": _adapter("seller_avg_delivery_estimate_abs_error_days"),
    "seller.p90_order_to_delivery_days": _adapter("seller_p90_order_to_delivery_days"),
    "seller.missing_delivery_timestamp_rate": _adapter("seller_missing_delivery_timestamp_rate"),
    # Order funnel (Phase 6 M1b)
    "funnel.order_approval_rate": _adapter("order_approval_rate"),
    "funnel.order_shipment_rate": _adapter("order_shipment_rate"),
    "funnel.order_final_delivery_rate": _adapter("order_final_delivery_rate"),
}


def get_handler(handler_key: str) -> callable:
    """Look up a handler by key.  Raises ``KeyError`` if not found."""
    if handler_key not in HANDLER_REGISTRY:
        raise KeyError(f"Handler {handler_key!r} is not registered")
    return HANDLER_REGISTRY[handler_key]
