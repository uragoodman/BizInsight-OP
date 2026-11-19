"""Create two real seller tenants from the Phase 6 CI Olist-shaped dataset."""
from sqlalchemy import text

from app.data.database import engine


def main() -> None:
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO iam.tenants(tenant_id,tenant_name,scope_type) VALUES
            ('seller-tenant-1','Seller 1','seller'),('seller-tenant-2','Seller 2','seller'),('platform','Platform','platform')
            ON CONFLICT (tenant_id) DO NOTHING"""))
        conn.execute(text("""INSERT INTO iam.users(user_id,display_name,email) VALUES
            ('seller-user-1','Seller User 1','seller-user-1@ci.local'),
            ('seller-user-2','Seller User 2','seller-user-2@ci.local'),
            ('platform-admin','Platform Admin','platform-admin@ci.local')
            ON CONFLICT (user_id) DO NOTHING"""))
        conn.execute(text("""INSERT INTO iam.tenant_memberships(tenant_id,user_id,role) VALUES
            ('seller-tenant-1','seller-user-1','analyst'),('seller-tenant-2','seller-user-2','analyst'),('platform','platform-admin','admin')
            ON CONFLICT (tenant_id,user_id) DO NOTHING"""))
        conn.execute(text("""INSERT INTO iam.tenant_seller_accounts(tenant_id,seller_id) VALUES
            ('seller-tenant-1','seller-1'),('seller-tenant-2','seller-2') ON CONFLICT (tenant_id) DO NOTHING"""))
        conn.execute(text("""INSERT INTO analytics.fact_order_items
            (order_id,order_item_id,seller_id,product_id,customer_unique_id,customer_state,
             purchase_date,order_status,price,freight_value)
            SELECT oi.order_id,oi.order_item_id,oi.seller_id,oi.product_id,c.customer_unique_id,
                   c.customer_state,DATE(o.order_purchase_timestamp),o.order_status,
                   oi.price,oi.freight_value
            FROM raw.order_items oi JOIN raw.orders o ON o.order_id=oi.order_id JOIN raw.customers c ON c.customer_id=o.customer_id
            WHERE o.order_status='delivered' ON CONFLICT DO NOTHING"""))
        # Seeder runs as the CI database owner; RLS is intentionally verified by
        # the tests using a non-owner application role/session setting.
        conn.execute(text("SELECT analytics.refresh_tenant_marts()"))


if __name__ == "__main__":
    main()
