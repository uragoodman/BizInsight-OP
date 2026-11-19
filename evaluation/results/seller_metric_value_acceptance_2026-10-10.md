# Seller scope value acceptance

Registered seller value metrics: **46**. Item score: **40 pass / 6 fail**. Test run: **41 passed, 6 failed, 2 warnings** (the additional test checks exact inventory).

Execution used MetricRegistry.execute with a server-built ControlledAnalysisScope, an active tenant identity and a valid seller grant. Each measured query was checked for bound seller ID and date; successful query traces contained tenant, user, membership and seller-grant checks. Other-seller rows, including a shared order and large allocated payments, were present.

| Metric | Expected | Actual | Registry unit | Query date filter | SQL sources | Status | Score / failure |
|---|---:|---:|---|---|---|---|---|
| `avg_approval_lead_hours` | 3.0 | 3.0 | hours | 2024-01-02 bound | analytics.v_seller_fulfillment | ok | PASS |
| `avg_delivery_transit_hours` | 24.0 | 24.0 | hours | 2024-01-02 bound | analytics.v_seller_fulfillment | ok | PASS |
| `avg_dispatch_lead_hours` | 6.0 | 6.0 | hours | 2024-01-02 bound | analytics.v_seller_fulfillment | ok | PASS |
| `avg_order_value` | 135.0 | 135.0 | BRL/order | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | FAIL: source — returned source label differs from executed SQL — lineage metadata differs: analytics.tenant_daily_metrics |
| `bad_review_rate` | 50.0 | 50.0 | % | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders, raw.order_reviews | ok | PASS |
| `customer_count` | 3.0 | 3.0 | customers | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | FAIL: source — returned source label differs from executed SQL — lineage metadata differs: analytics.tenant_fact_orders |
| `freight_value` | 40.0 | 40.0 | BRL | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | FAIL: source — returned source label differs from executed SQL |
| `gmv` | 540.0 | 540.0 | BRL | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | FAIL: source — returned source label differs from executed SQL — lineage metadata differs: analytics.tenant_daily_metrics |
| `installment_order_share` | 82.65 | 82.65306122448979 | % | 2024-01-02 bound | analytics.fact_order_payments, analytics.fact_orders | ok | PASS |
| `on_time_delivery_rate` | 66.67 | 66.66666666666667 | % | 2024-01-02 bound | analytics.v_seller_fulfillment | ok | PASS |
| `order_approval_rate` | 75.0 | 75.0 | % | 2024-01-02 bound | raw.order_items, raw.orders | ok | PASS |
| `order_count` | 4.0 | 4.0 | orders | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | FAIL: source — returned source label differs from executed SQL — lineage metadata differs: analytics.tenant_fact_orders |
| `order_final_delivery_rate` | 100.0 | 100.0 | % | 2024-01-02 bound | raw.order_items, raw.orders | ok | PASS |
| `order_shipment_rate` | 50.0 | 50.0 | % | 2024-01-02 bound | raw.order_items, raw.orders | ok | PASS |
| `payment_value` | 490.0 | 490.0 | BRL | 2024-01-02 bound | analytics.fact_order_payments, analytics.fact_orders | ok | PASS |
| `repeat_purchase_rate` | 33.33 | — | % | no query | — | exception | FAIL: date_filter, seller_filter, source, status, value — KeyError: 'repeat_purchase_rate' — lineage metadata differs: analytics.tenant_daily_metrics |
| `review_comment_rate` | 50.0 | 50.0 | % | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders, raw.order_reviews | ok | PASS — lineage metadata differs: analytics.v_order_unified |
| `review_score_avg` | 3.0 | 3.0 | score | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders, raw.order_reviews | ok | PASS |
| `seller_active_category_count` | 4.0 | 4.0 | categories | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders, analytics.v_product_category | ok | PASS — lineage metadata differs: raw.products |
| `seller_active_product_count` | 4.0 | 4.0 | products | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | PASS |
| `seller_avg_delivery_estimate_abs_error_days` | 1.0 | 1.0 | days | 2024-01-02 bound | analytics.v_seller_fulfillment | ok | PASS |
| `seller_avg_early_delivery_days` | 2.0 | 2.0 | days | 2024-01-02 bound | analytics.v_seller_fulfillment | ok | PASS |
| `seller_avg_freight_per_order` | 10.0 | 10.0 | BRL/order | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | PASS |
| `seller_avg_item_price` | 100.0 | 100.0 | BRL/item | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | PASS |
| `seller_avg_late_delivery_days` | 1.0 | 1.0 | days | 2024-01-02 bound | analytics.v_seller_fulfillment | ok | PASS |
| `seller_avg_payment_installments` | 2.2755102040816326 | 2.2755102040816326 | installments | 2024-01-02 bound | analytics.fact_order_payments, analytics.fact_orders | ok | PASS |
| `seller_boleto_payment_share` | 17.346938775510203 | 17.346938775510203 | % | 2024-01-02 bound | analytics.fact_order_payments, analytics.fact_orders | ok | PASS |
| `seller_credit_card_payment_share` | 82.65306122448979 | 82.65306122448979 | % | 2024-01-02 bound | analytics.fact_order_payments, analytics.fact_orders | ok | PASS |
| `seller_free_shipping_order_rate` | 25.0 | 25.0 | % | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | PASS |
| `seller_freight_to_gmv_rate` | 7.407407407407407 | 7.407407407407407 | % | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | PASS |
| `seller_historical_returning_customer_count` | 1.0 | 1.0 | customers | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | PASS |
| `seller_historical_returning_customer_rate` | 33.33 | 33.333333333333336 | % | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | PASS |
| `seller_item_gross_value` | 500.0 | 500.0 | BRL | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | PASS |
| `seller_items_per_order` | 1.25 | 1.25 | items/order | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | PASS |
| `seller_missing_delivery_timestamp_rate` | 25.0 | 25.0 | % | 2024-01-02 bound | analytics.v_seller_fulfillment | ok | PASS |
| `seller_new_customer_count` | 2.0 | 2.0 | customers | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | PASS |
| `seller_new_customer_rate` | 66.67 | 66.66666666666667 | % | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | PASS |
| `seller_new_product_gross_share` | 40.0 | 40.0 | % | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | PASS |
| `seller_order_bad_review_rate` | 50.0 | 50.0 | % | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders, raw.order_reviews | ok | PASS |
| `seller_p90_order_to_delivery_days` | 7.4 | 7.4 | days | 2024-01-02 bound | analytics.v_seller_fulfillment | ok | PASS |
| `seller_product_attributed_bad_review_rate` | 60.0 | 60.0 | % | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders, raw.order_reviews | ok | PASS |
| `seller_product_attributed_review_score_avg` | 2.8 | 2.8 | score | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders, raw.order_reviews | ok | PASS |
| `seller_single_item_order_product_bad_review_rate` | 33.333333333333336 | 33.333333333333336 | % | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders, raw.order_reviews | ok | PASS |
| `seller_single_item_order_product_review_score_avg` | 3.3333333333333335 | 3.3333333333333335 | score | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders, raw.order_reviews | ok | PASS |
| `seller_top1_product_gross_share` | 60.0 | 60.0 | % | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders | ok | PASS |
| `seller_top3_category_gross_share` | 90.0 | 90.0 | % | 2024-01-02 bound | analytics.v_delivered_order_items, analytics.v_delivered_orders, analytics.v_product_category | ok | PASS — lineage metadata differs: raw.products |

Full handler records, SQL parameters and authorization statements are in the JSON artifact.
