"""数据库初始化与数仓构建工具。

该文件负责把原始 CSV 数据装载到数据库，并在此基础上生成
`raw`、`analytics` 两层结构：

- `raw` 层保存与原始文件一一对应的明细表。
- `analytics` 层构建适合经营分析查询的事实表与日聚合表。

处理能力包括：

- 自动发现本地数据集目录。
- 创建所需 schema。
- 读取并导入多份 CSV 文件。
- 构建订单事实表和每日指标聚合表。
- 提供一键式数据库引导入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from app.data.database import engine


@dataclass(frozen=True)
class CSVTableSpec:
    """描述单个 CSV 文件如何映射到数据库表。

    作用：
    - 统一声明文件名、目标表名和需要按日期解析的列。
    - 让批量导入逻辑以配置驱动，减少重复代码。
    """

    filename: str
    table_name: str
    parse_dates: tuple[str, ...] = ()


RAW_TABLE_SPECS: tuple[CSVTableSpec, ...] = (
    CSVTableSpec(
        filename="olist_orders_dataset.csv",
        table_name="orders",
        parse_dates=(
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ),
    ),
    CSVTableSpec(
        filename="olist_order_items_dataset.csv",
        table_name="order_items",
        parse_dates=("shipping_limit_date",),
    ),
    CSVTableSpec(
        filename="olist_order_payments_dataset.csv",
        table_name="order_payments",
    ),
    CSVTableSpec(
        filename="olist_customers_dataset.csv",
        table_name="customers",
    ),
    CSVTableSpec(
        filename="olist_products_dataset.csv",
        table_name="products",
    ),
    CSVTableSpec(
        filename="olist_sellers_dataset.csv",
        table_name="sellers",
    ),
    CSVTableSpec(
        filename="olist_order_reviews_dataset.csv",
        table_name="order_reviews",
        parse_dates=("review_creation_date", "review_answer_timestamp"),
    ),
    CSVTableSpec(
        filename="olist_geolocation_dataset.csv",
        table_name="geolocation",
    ),
    CSVTableSpec(
        filename="product_category_name_translation.csv",
        table_name="product_category_name_translation",
    ),
)


def discover_dataset_dir(root: str = "kagglehub") -> Path:
    """在数据根目录下定位 Olist 数据集所在文件夹。

    Args:
        root: 数据集搜索起点目录，默认从 `kagglehub` 开始递归查找。

    Returns:
        Path: 包含 `olist_orders_dataset.csv` 的目录路径。

    Raises:
        FileNotFoundError: 当根目录不存在，或未找到必需的 CSV 文件时抛出。
    """

    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"Dataset root not found: {base}")

    matches = sorted(base.rglob("olist_orders_dataset.csv"))
    if not matches:
        raise FileNotFoundError("Could not find olist_orders_dataset.csv under dataset root.")
    return matches[0].parent


def ensure_schemas(target_engine: Engine = engine) -> None:
    """确保数据库中存在 `raw` 和 `analytics` schema。

    Args:
        target_engine: 用于执行建库语句的 SQLAlchemy Engine。

    Returns:
        None: 仅执行建 schema 操作，不返回业务数据。
    """

    statements = (
        "CREATE SCHEMA IF NOT EXISTS raw",
        "CREATE SCHEMA IF NOT EXISTS analytics",
    )
    with target_engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


def load_raw_tables(dataset_dir: Path, target_engine: Engine = engine) -> None:
    """将数据集目录中的原始 CSV 文件导入到 `raw` schema。

    处理能力：
    - 校验每份必需 CSV 是否存在。
    - 按配置解析日期列。
    - 统一清洗列名格式。
    - 以覆盖方式写入数据库原始表。

    Args:
        dataset_dir: 已定位到的 Olist 数据集目录。
        target_engine: 目标数据库连接引擎。

    Returns:
        None: 导入完成后不返回值。

    Raises:
        FileNotFoundError: 任意必需 CSV 缺失时抛出。
    """

    for spec in RAW_TABLE_SPECS:
        csv_path = dataset_dir / spec.filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing required CSV: {csv_path}")

        dataframe = pd.read_csv(csv_path, parse_dates=list(spec.parse_dates), low_memory=False)
        dataframe.columns = [column.strip().lower() for column in dataframe.columns]
        dataframe.to_sql(
            spec.table_name,
            target_engine,
            schema="raw",
            if_exists="replace",
            index=False,
            chunksize=5000,
            method="multi",
        )


def build_analytics_tables(target_engine: Engine = engine) -> None:
    """基于 `raw` 层表构建分析用事实表和日指标聚合表。

    处理能力：
    - 重建 `analytics.fact_orders` 事实表。
    - 重建 `analytics.daily_metrics` 日粒度指标表。
    - 为常用查询字段创建索引，提升查询性能。

    Args:
        target_engine: 用于执行建表 SQL 的数据库引擎。

    Returns:
        None: 只负责建表与计算，不返回结果。
    """

    """Refresh marts whose schema is owned by Alembic.

    Do not add ``DROP TABLE / CREATE TABLE AS`` here.  Bootstrap owns only the
    raw CSV load; the migration owns the analytics contract and this function
    merely asks PostgreSQL to refill it idempotently.
    """
    with target_engine.begin() as conn:
        has_refresh_function = conn.execute(text("""
            SELECT EXISTS (
              SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
              WHERE n.nspname = 'analytics' AND p.proname = 'refresh_olist_marts'
            )
        """)).scalar()
        if not has_refresh_function:
            raise RuntimeError(
                "analytics.refresh_olist_marts is missing; run `alembic upgrade head` before bootstrap."
            )
        conn.execute(text("SELECT analytics.refresh_olist_marts()"))
    return

    statements = (
        """
        DROP TABLE IF EXISTS analytics.fact_order_items
        """,
        """
        CREATE TABLE analytics.fact_order_items AS
        SELECT
            oi.order_id,
            oi.order_item_id,
            oi.seller_id,
            c.customer_unique_id,
            c.customer_state,
            DATE(o.order_purchase_timestamp) AS purchase_date,
            oi.price::numeric AS price,
            oi.freight_value::numeric AS freight_value,
            o.order_status
        FROM raw.order_items oi
        JOIN raw.orders o ON o.order_id = oi.order_id
        LEFT JOIN raw.customers c ON c.customer_id = o.customer_id
        WHERE o.order_status = 'delivered'
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_fact_order_items_pk ON analytics.fact_order_items (order_id, order_item_id)",
        "CREATE INDEX IF NOT EXISTS idx_fact_order_items_seller_date ON analytics.fact_order_items (seller_id, purchase_date)",
        """
        DROP TABLE IF EXISTS analytics.fact_orders
        """,
        """
        CREATE TABLE analytics.fact_orders AS
        WITH payment_summary AS (
            SELECT
                order_id,
                SUM(payment_value) AS payment_value
            FROM raw.order_payments
            GROUP BY order_id
        ),
        item_summary AS (
            SELECT
                order_id,
                COUNT(*) AS item_count,
                SUM(price) AS item_gross_value,
                SUM(freight_value) AS freight_value
            FROM raw.order_items
            GROUP BY order_id
        )
        SELECT
            o.order_id,
            o.customer_id,
            c.customer_unique_id,
            c.customer_city,
            c.customer_state,
            o.order_status,
            o.order_purchase_timestamp,
            DATE(o.order_purchase_timestamp) AS purchase_date,
            COALESCE(p.payment_value, 0) AS payment_value,
            COALESCE(i.item_count, 0) AS item_count,
            COALESCE(i.item_gross_value, 0) AS item_gross_value,
            COALESCE(i.freight_value, 0) AS freight_value
        FROM raw.orders o
        LEFT JOIN raw.customers c
            ON o.customer_id = c.customer_id
        LEFT JOIN payment_summary p
            ON o.order_id = p.order_id
        LEFT JOIN item_summary i
            ON o.order_id = i.order_id
        WHERE o.order_status = 'delivered'
        """,
        "CREATE INDEX IF NOT EXISTS idx_fact_orders_purchase_date ON analytics.fact_orders (purchase_date)",
        "CREATE INDEX IF NOT EXISTS idx_fact_orders_customer_state ON analytics.fact_orders (customer_state)",
        """
        DROP TABLE IF EXISTS analytics.daily_metrics
        """,
        """
        CREATE TABLE analytics.daily_metrics AS
        WITH delivered_metrics AS (
            SELECT
                purchase_date AS metric_date,
                COUNT(*) AS order_count,
                SUM(item_gross_value + freight_value) AS gmv,
                CASE
                    WHEN COUNT(*) = 0 THEN 0
                    ELSE ROUND((SUM(item_gross_value + freight_value) / COUNT(*))::numeric, 2)
                END AS avg_order_value,
                SUM(freight_value) AS freight_value,
                COUNT(DISTINCT customer_unique_id) AS customer_count,
                AVG(item_count)::numeric(12, 2) AS avg_items_per_order
            FROM analytics.fact_orders
            GROUP BY purchase_date
        ),
        first_purchase AS (
            SELECT
                customer_unique_id,
                MIN(purchase_date) AS first_purchase_date
            FROM analytics.fact_orders
            WHERE customer_unique_id IS NOT NULL
            GROUP BY customer_unique_id
        ),
        repeat_metrics AS (
            SELECT
                fo.purchase_date AS metric_date,
                COUNT(
                    DISTINCT CASE
                        WHEN fp.first_purchase_date < fo.purchase_date THEN fo.customer_unique_id
                        ELSE NULL
                    END
                ) AS repeat_customer_count
            FROM analytics.fact_orders fo
            LEFT JOIN first_purchase fp
                ON fo.customer_unique_id = fp.customer_unique_id
            GROUP BY fo.purchase_date
        ),
        order_metrics AS (
            SELECT
                DATE(order_purchase_timestamp) AS metric_date,
                COUNT(*) AS total_order_count,
                SUM(CASE WHEN order_status = 'canceled' THEN 1 ELSE 0 END) AS canceled_order_count,
                SUM(CASE WHEN order_status = 'delivered' THEN 1 ELSE 0 END) AS delivered_order_count,
                SUM(
                    CASE
                        WHEN order_status = 'delivered'
                             AND order_delivered_customer_date IS NOT NULL
                             AND order_estimated_delivery_date IS NOT NULL
                             AND order_delivered_customer_date <= order_estimated_delivery_date
                        THEN 1
                        ELSE 0
                    END
                ) AS on_time_order_count,
                SUM(
                    CASE
                        WHEN order_status = 'delivered'
                             AND order_delivered_customer_date IS NOT NULL
                             AND order_estimated_delivery_date IS NOT NULL
                             AND order_delivered_customer_date > order_estimated_delivery_date
                        THEN 1
                        ELSE 0
                    END
                ) AS delayed_order_count,
                AVG(
                    CASE
                        WHEN order_status = 'delivered'
                             AND order_delivered_customer_date IS NOT NULL
                        THEN EXTRACT(EPOCH FROM (order_delivered_customer_date - order_purchase_timestamp)) / 86400.0
                        ELSE NULL
                    END
                )::numeric(12, 2) AS avg_delivery_lead_time
            FROM raw.orders
            GROUP BY DATE(order_purchase_timestamp)
        ),
        review_metrics AS (
            SELECT
                DATE(o.order_purchase_timestamp) AS metric_date,
                ROUND(AVG(r.review_score)::numeric, 2) AS review_score_avg,
                COUNT(r.review_score) AS reviewed_order_count,
                SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS bad_review_order_count
            FROM raw.orders o
            LEFT JOIN raw.order_reviews r
                ON o.order_id = r.order_id
            GROUP BY DATE(o.order_purchase_timestamp)
        )
        SELECT
            om.metric_date,
            COALESCE(dm.order_count, 0) AS order_count,
            COALESCE(dm.gmv, 0) AS gmv,
            COALESCE(dm.avg_order_value, 0) AS avg_order_value,
            CASE
                WHEN om.total_order_count = 0 THEN 0
                ELSE ROUND((om.canceled_order_count * 100.0 / om.total_order_count)::numeric, 2)
            END AS cancel_rate,
            COALESCE(dm.freight_value, 0) AS freight_value,
            rm.review_score_avg,
            COALESCE(dm.customer_count, 0) AS customer_count,
            COALESCE(rpm.repeat_customer_count, 0) AS repeat_customer_count,
            CASE
                WHEN COALESCE(dm.customer_count, 0) = 0 THEN 0
                ELSE ROUND((COALESCE(rpm.repeat_customer_count, 0) * 100.0 / dm.customer_count)::numeric, 2)
            END AS repeat_purchase_rate,
            COALESCE(dm.avg_items_per_order, 0) AS avg_items_per_order,
            om.avg_delivery_lead_time,
            CASE
                WHEN om.delivered_order_count = 0 THEN 0
                ELSE ROUND((om.on_time_order_count * 100.0 / om.delivered_order_count)::numeric, 2)
            END AS on_time_delivery_rate,
            CASE
                WHEN om.delivered_order_count = 0 THEN 0
                ELSE ROUND((om.delayed_order_count * 100.0 / om.delivered_order_count)::numeric, 2)
            END AS delivery_delay_rate,
            CASE
                WHEN COALESCE(rm.reviewed_order_count, 0) = 0 THEN 0
                ELSE ROUND((rm.bad_review_order_count * 100.0 / rm.reviewed_order_count)::numeric, 2)
            END AS bad_review_rate,
            CASE
                WHEN COALESCE(dm.gmv, 0) = 0 THEN 0
                ELSE ROUND((COALESCE(dm.freight_value, 0) * 100.0 / dm.gmv)::numeric, 2)
            END AS freight_rate
        FROM order_metrics om
        LEFT JOIN delivered_metrics dm
            ON om.metric_date = dm.metric_date
        LEFT JOIN repeat_metrics rpm
            ON om.metric_date = rpm.metric_date
        LEFT JOIN review_metrics rm
            ON om.metric_date = rm.metric_date
        ORDER BY om.metric_date
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_metrics_metric_date ON analytics.daily_metrics (metric_date)",
    )

    with target_engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
        # Phase 7 tenant marts are materialized only for explicitly onboarded
        # seller accounts. Platform analytics remains untouched.
        has_tenant_function = conn.execute(text("""
            SELECT EXISTS (SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
            WHERE n.nspname='analytics' AND p.proname='refresh_tenant_marts')
        """)).scalar()
        if has_tenant_function:
            conn.execute(text("SELECT analytics.refresh_tenant_marts()"))


def bootstrap_database(root: str = "kagglehub", target_engine: Engine = engine) -> Path:
    """完成从发现数据集到构建分析表的整套数据库引导流程。

    Args:
        root: 数据集搜索根目录。
        target_engine: 目标数据库连接引擎。

    Returns:
        Path: 实际被加载的数据集目录，便于上层记录或打印。
    """

    dataset_dir = discover_dataset_dir(root)
    ensure_schemas(target_engine)
    load_raw_tables(dataset_dir, target_engine)
    build_analytics_tables(target_engine)
    return dataset_dir
