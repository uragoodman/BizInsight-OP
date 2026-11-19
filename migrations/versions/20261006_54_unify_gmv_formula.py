"""Use delivered item value plus freight as the single platform GMV formula."""

from alembic import op


revision = "20261006_54"
down_revision = "20260925_53"
branch_labels = None
depends_on = None


_ALIGN_FUNCTION = r"""
DO $$
DECLARE
    function_definition text;
    old_formula constant text := 'SUM(payment_value) gmv';
    new_formula constant text := 'SUM(item_gross_value + freight_value) gmv';
BEGIN
    IF to_regprocedure('analytics.refresh_olist_marts()') IS NULL THEN
        RAISE EXCEPTION 'analytics.refresh_olist_marts() is required for GMV alignment';
    END IF;

    SELECT pg_get_functiondef('analytics.refresh_olist_marts()'::regprocedure)
      INTO function_definition;
    IF position(old_formula IN function_definition) > 0 THEN
        function_definition := replace(function_definition, old_formula, new_formula);
        EXECUTE function_definition;
    ELSIF position(new_formula IN function_definition) = 0 THEN
        RAISE EXCEPTION 'could not identify the existing GMV expression in refresh_olist_marts()';
    END IF;

    IF to_regclass('analytics.daily_metrics') IS NOT NULL
       AND to_regclass('analytics.v_delivered_orders') IS NOT NULL THEN
        WITH daily_gmv AS (
            SELECT dm.metric_date,
                   COUNT(fo.order_id) AS order_count,
                   COALESCE(SUM(fo.item_gross_value + fo.freight_value), 0) AS gmv,
                   COALESCE(SUM(fo.freight_value), 0) AS freight_value
            FROM analytics.daily_metrics dm
            LEFT JOIN analytics.v_delivered_orders fo
              ON fo.purchase_date = dm.metric_date
            GROUP BY dm.metric_date
        )
        UPDATE analytics.daily_metrics dm
           SET gmv = daily_gmv.gmv,
               avg_order_value = CASE WHEN daily_gmv.order_count = 0 THEN 0
                   ELSE ROUND((daily_gmv.gmv / daily_gmv.order_count)::numeric, 2) END,
               freight_rate = CASE WHEN daily_gmv.gmv = 0 THEN 0
                   ELSE ROUND((daily_gmv.freight_value * 100.0 / daily_gmv.gmv)::numeric, 2) END
          FROM daily_gmv
         WHERE daily_gmv.metric_date = dm.metric_date;
    END IF;
END $$;
"""


def upgrade() -> None:
    op.execute(_ALIGN_FUNCTION)


def downgrade() -> None:
    # Keep the corrected metric contract when rolling application code back.
    # Reverting to payment_value would silently reintroduce the definition bug.
    pass
