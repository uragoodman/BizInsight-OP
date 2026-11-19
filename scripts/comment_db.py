from pathlib import Path
import sys

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data.database import engine


TABLE_COMMENTS = {
    "raw.customers": "Olist 原始客户表，记录客户主键、唯一客户标识及所属城市、州、邮编前缀等信息。",
    "raw.geolocation": "Olist 原始地理位置表，记录邮编前缀与经纬度、城市、州之间的映射关系。",
    "raw.order_items": "Olist 原始订单商品明细表，记录每个订单中的商品、卖家、成交价格和运费。",
    "raw.order_payments": "Olist 原始订单支付表，记录订单支付方式、分期次数和支付金额。",
    "raw.order_reviews": "Olist 原始订单评价表，记录订单评分、评价标题、评价内容和评价时间。",
    "raw.orders": "Olist 原始订单主表，记录订单状态及购买、审批、发货、签收、预计送达等关键时间。",
    "raw.product_category_name_translation": "Olist 商品品类翻译表，记录葡萄牙语品类名与英文品类名之间的对应关系。",
    "raw.products": "Olist 原始商品表，记录商品品类、名称长度、描述长度、图片数量和体积重量属性。",
    "raw.sellers": "Olist 原始卖家表，记录卖家主键及所属城市、州、邮编前缀等信息。",
    "analytics.fact_orders": "分析层订单事实表，按已交付订单整合客户、订单、支付和商品汇总信息，作为经营指标分析基础表。",
    "analytics.daily_metrics": "分析层日级指标表，按购买日期聚合 GMV、订单量和客单价，用于最小经营指标查询闭环。",
}


COLUMN_COMMENTS = {
    "raw.customers.customer_id": "客户实例主键，同一用户在不同订单中可能对应不同的 customer_id。",
    "raw.customers.customer_unique_id": "客户唯一标识，用于识别同一真实客户的多次下单行为。",
    "raw.customers.customer_zip_code_prefix": "客户邮编前缀。",
    "raw.customers.customer_city": "客户所在城市。",
    "raw.customers.customer_state": "客户所在州。",
    "raw.geolocation.geolocation_zip_code_prefix": "地理位置对应的邮编前缀。",
    "raw.geolocation.geolocation_lat": "邮编前缀对应纬度。",
    "raw.geolocation.geolocation_lng": "邮编前缀对应经度。",
    "raw.geolocation.geolocation_city": "邮编前缀对应城市。",
    "raw.geolocation.geolocation_state": "邮编前缀对应州。",
    "raw.order_items.order_id": "订单主键，关联 raw.orders.order_id。",
    "raw.order_items.order_item_id": "订单内商品行号，同一订单可能有多条明细。",
    "raw.order_items.product_id": "商品主键，关联 raw.products.product_id。",
    "raw.order_items.seller_id": "卖家主键，关联 raw.sellers.seller_id。",
    "raw.order_items.shipping_limit_date": "卖家承诺最晚发货时间。",
    "raw.order_items.price": "订单商品成交价格，不含运费。",
    "raw.order_items.freight_value": "该商品对应运费金额。",
    "raw.order_payments.order_id": "订单主键，关联 raw.orders.order_id。",
    "raw.order_payments.payment_sequential": "同一订单的支付顺序编号。",
    "raw.order_payments.payment_type": "支付方式，如 credit_card、voucher 等。",
    "raw.order_payments.payment_installments": "支付分期次数。",
    "raw.order_payments.payment_value": "该笔支付金额。",
    "raw.order_reviews.review_id": "评价主键。",
    "raw.order_reviews.order_id": "订单主键，关联 raw.orders.order_id。",
    "raw.order_reviews.review_score": "评价分数，通常为 1 到 5。",
    "raw.order_reviews.review_comment_title": "评价标题。",
    "raw.order_reviews.review_comment_message": "评价正文内容。",
    "raw.order_reviews.review_creation_date": "评价创建日期。",
    "raw.order_reviews.review_answer_timestamp": "评价答复时间。",
    "raw.orders.order_id": "订单主键。",
    "raw.orders.customer_id": "客户实例主键，关联 raw.customers.customer_id。",
    "raw.orders.order_status": "订单状态，如 delivered、shipped、canceled 等。",
    "raw.orders.order_purchase_timestamp": "订单购买时间。",
    "raw.orders.order_approved_at": "订单审批通过时间。",
    "raw.orders.order_delivered_carrier_date": "订单交付承运商时间。",
    "raw.orders.order_delivered_customer_date": "订单签收时间。",
    "raw.orders.order_estimated_delivery_date": "预计送达时间。",
    "raw.product_category_name_translation.product_category_name": "葡萄牙语商品品类名。",
    "raw.product_category_name_translation.product_category_name_english": "英文商品品类名。",
    "raw.products.product_id": "商品主键。",
    "raw.products.product_category_name": "商品品类名，通常为葡萄牙语。",
    "raw.products.product_name_lenght": "商品名称长度，原始数据字段拼写保持与源数据一致。",
    "raw.products.product_description_lenght": "商品描述长度，原始数据字段拼写保持与源数据一致。",
    "raw.products.product_photos_qty": "商品图片数量。",
    "raw.products.product_weight_g": "商品重量，单位为克。",
    "raw.products.product_length_cm": "商品长度，单位为厘米。",
    "raw.products.product_height_cm": "商品高度，单位为厘米。",
    "raw.products.product_width_cm": "商品宽度，单位为厘米。",
    "raw.sellers.seller_id": "卖家主键。",
    "raw.sellers.seller_zip_code_prefix": "卖家邮编前缀。",
    "raw.sellers.seller_city": "卖家所在城市。",
    "raw.sellers.seller_state": "卖家所在州。",
    "analytics.fact_orders.order_id": "订单主键，分析层订单事实表粒度键。",
    "analytics.fact_orders.customer_id": "客户实例主键。",
    "analytics.fact_orders.customer_unique_id": "客户唯一标识，用于识别同一真实客户。",
    "analytics.fact_orders.customer_city": "客户所在城市，用于地域分析。",
    "analytics.fact_orders.customer_state": "客户所在州，用于地域分析。",
    "analytics.fact_orders.order_status": "订单状态，当前事实表仅保留 delivered 订单。",
    "analytics.fact_orders.order_purchase_timestamp": "订单购买时间戳。",
    "analytics.fact_orders.purchase_date": "订单购买日期，用于日级指标聚合。",
    "analytics.fact_orders.payment_value": "订单总支付金额，用于计算 GMV。",
    "analytics.fact_orders.item_count": "订单商品件数。",
    "analytics.fact_orders.item_gross_value": "订单商品金额汇总，不含运费。",
    "analytics.fact_orders.freight_value": "订单运费汇总。",
    "analytics.daily_metrics.metric_date": "指标日期，对应订单购买日期。",
    "analytics.daily_metrics.order_count": "当日已交付订单量。",
    "analytics.daily_metrics.gmv": "当日 GMV，当前口径为已交付订单 payment_value 汇总。",
    "analytics.daily_metrics.avg_order_value": "当日客单价，当前口径为 GMV / 订单量。",
}


def main() -> None:
    with engine.begin() as conn:
        for table_name, comment in TABLE_COMMENTS.items():
            escaped = comment.replace("'", "''")
            conn.execute(text(f"COMMENT ON TABLE {table_name} IS '{escaped}'"))
            print(f"Commented table {table_name}", flush=True)

        for column_path, comment in COLUMN_COMMENTS.items():
            escaped = comment.replace("'", "''")
            conn.execute(text(f"COMMENT ON COLUMN {column_path} IS '{escaped}'"))
            print(f"Commented column {column_path}", flush=True)


if __name__ == "__main__":
    main()
