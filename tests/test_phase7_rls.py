"""Database-level isolation: deliberately omit tenant predicates."""
import os

import pytest
from sqlalchemy import text

from app.data.database import engine


pytestmark = pytest.mark.skipif(not os.getenv("POSTGRES_URL"), reason="requires PostgreSQL")


def _seed_phase7_tenants() -> None:
    """Make the RLS proof self-contained instead of relying on bootstrap data."""
    with engine.begin() as conn:
        for tenant_id in ("seller-tenant-1", "seller-tenant-2"):
            conn.execute(text("""INSERT INTO iam.tenants(tenant_id,tenant_name,scope_type,status)
              VALUES(:id,:name,'seller','active') ON CONFLICT(tenant_id) DO UPDATE SET status='active'"""), {"id": tenant_id, "name": tenant_id})
            conn.execute(text("""INSERT INTO iam.tenant_seller_accounts(tenant_id,seller_id)
              VALUES(:id,:seller) ON CONFLICT DO NOTHING"""), {"id": tenant_id, "seller": f"{tenant_id}-seller"})
            conn.execute(text("""INSERT INTO analytics.tenant_fact_orders(tenant_id,order_id,purchase_date,sales_gmv)
              VALUES(:id,:order,'2018-08-01',1) ON CONFLICT DO NOTHING"""), {"id": tenant_id, "order": f"{tenant_id}-order"})


def _rows_for(tenant_id: str) -> list[str]:
    _seed_phase7_tenants()
    with engine.begin() as conn:
        conn.execute(text("DO $$ BEGIN CREATE ROLE phase7_reader NOINHERIT; EXCEPTION WHEN duplicate_object THEN NULL; END $$"))
        conn.execute(text("GRANT USAGE ON SCHEMA analytics TO phase7_reader"))
        conn.execute(text("GRANT SELECT ON analytics.tenant_fact_orders TO phase7_reader"))
        conn.execute(text("SET LOCAL ROLE phase7_reader"))
        conn.execute(text("SELECT set_config('app.tenant_id', :tenant_id, true)"), {"tenant_id": tenant_id})
        conn.execute(text("SELECT set_config('app.scope_type', 'seller', true)"))
        # No WHERE clause: RLS, rather than repository code, is the proof.
        return [row[0] for row in conn.execute(text("SELECT DISTINCT tenant_id FROM analytics.tenant_fact_orders"))]


def test_rls_cannot_return_another_seller_tenant():
    assert _rows_for("seller-tenant-1") == ["seller-tenant-1"]
    assert _rows_for("seller-tenant-2") == ["seller-tenant-2"]


def test_rls_without_context_returns_no_rows():
    _seed_phase7_tenants()
    with engine.begin() as conn:
        conn.execute(text("SET LOCAL ROLE phase7_reader"))
        assert conn.execute(text("SELECT count(*) FROM analytics.tenant_fact_orders")).scalar_one() == 0
