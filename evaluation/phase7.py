"""Phase 7 seller-scope Golden checks against raw Olist item facts."""
from sqlalchemy import text

from app.data.database import engine
from app.data.repositories.tenant_metrics import query_seller_metric
from app.tenancy import SELLER_SCOPE, TenantContext


def main() -> None:
    checks=[]
    for tenant_id, user_id, seller_id in (("seller-tenant-1","seller-user-1","seller-1"),("seller-tenant-2","seller-user-2","seller-2")):
        context=TenantContext(tenant_id,user_id,"analyst",SELLER_SCOPE,seller_id)
        actual=query_seller_metric(context,"sales_gmv","2018-08-01","2018-08-31")["value"]
        with engine.connect() as conn:
            expected=conn.execute(text("""SELECT COALESCE(SUM(oi.price+oi.freight_value),0) FROM raw.order_items oi
              JOIN raw.orders o ON o.order_id=oi.order_id WHERE oi.seller_id=:seller AND o.order_status='delivered'
              AND DATE(o.order_purchase_timestamp) BETWEEN '2018-08-01' AND '2018-08-31'"""),{"seller":seller_id}).scalar_one()
        checks.append({"tenant_id":tenant_id,"actual":actual,"expected":float(expected)})
    print({"phase7_seller_golden":checks})
    if any(abs(item["actual"]-item["expected"])>0.01 for item in checks): raise SystemExit("seller Golden mismatch")


if __name__ == "__main__": main()
