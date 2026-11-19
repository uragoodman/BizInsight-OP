"""Isolated PostgreSQL acceptance for registered seller daily trend metrics."""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url

from app.data.repositories import explorer, seller_extensions
from app.metrics.registry import get_registry
from app.schemas.explorer import ControlledAnalysisScope
from app.tenancy import SellerGrant, TenantContext, activate_tenant_context


SELLER = "trend-target"
OTHER = "trend-other"
START = date(2024, 1, 10)
END = date(2024, 1, 12)
DATES = [date(2024, 1, day).isoformat() for day in (10, 11, 12)]

# Independent daily calculations from the seed rows below. repeat_purchase_rate
# is deliberately retained in the matrix but has no numeric expectation: its
# required complete-history boundary is not represented by current coverage
# metadata, so scoring it as a normal rate would overclaim certainty.
EXPECTED = {
    "customer_count": ([2, 3, 4], "customers"),
    "repeat_purchase_rate": (None, "%"),
    "on_time_delivery_rate": ([50, 50, 66.67], "%"),
    "avg_approval_lead_hours": ([2, 4.67, 7.5], "hours"),
    "avg_dispatch_lead_hours": ([3, 6, 10], "hours"),
    "avg_delivery_transit_hours": ([15, 11.67, 10], "hours"),
    # From raw.order_items joined to raw.orders: Jan 10 has 1 approved / 3
    # seller orders; Jan 11 has 2 / 4; Jan 12 has 3 / 5, including the 20:00
    # cancellation in that day's funnel population.
    "order_approval_rate": ([33.33, 50, 60], "%"),
    "order_shipment_rate": ([33.33, 50, 60], "%"),
    "order_final_delivery_rate": ([66.67, 75, 80], "%"),
    "payment_value": ([335, 350, 370], "BRL"),
    "installment_order_share": ([89.55, 57.14, 54.05], "%"),
    "review_score_avg": ([3, 3.67, 3.25], "score"),
    "bad_review_rate": ([50, 33.33, 25], "%"),
    "review_comment_rate": ([50, 66.67, 75], "%"),
    "gmv": ([420, 303, 501], "BRL"),
    "order_count": ([2, 3, 4], "orders"),
    "avg_order_value": ([210, 101, 125.25], "BRL/order"),
    "freight_value": ([40, 23, 31], "BRL"),
    "seller_free_shipping_order_rate": ([0, 33.33, 25], "%"),
    "seller_avg_early_delivery_days": ([2, 3, 2.5], "days"),
    "seller_avg_late_delivery_days": ([1, 2, 4], "days"),
    "seller_avg_delivery_estimate_abs_error_days": ([1.5, 2.5, 3], "days"),
    "seller_p90_order_to_delivery_days": ([3.8, 2.8, 5.6], "days"),
    "seller_missing_delivery_timestamp_rate": ([0, 33.33, 25], "%"),
    "seller_single_item_order_product_review_score_avg": ([5, 4.5, 2.67], "score"),
    "seller_single_item_order_product_bad_review_rate": ([0, 0, 33.33], "%"),
    "seller_product_attributed_review_score_avg": ([2, 3.25, 3.6], "score"),
    "seller_product_attributed_bad_review_rate": ([75, 50, 20], "%"),
    "seller_historical_returning_customer_count": ([2, 2, 1], "customers"),
    "seller_historical_returning_customer_rate": ([100, 66.67, 25], "%"),
    "seller_new_customer_count": ([0, 1, 3], "customers"),
    "seller_new_customer_rate": ([0, 33.33, 75], "%"),
    "seller_top1_product_gross_share": ([52.63, 35.71, 34.04], "%"),
    # Jan 12 category totals: F=160, B=120, G=80, A=70, E=40; top three
    # are 360 / total 470 = 76.5957%, rounded here to 76.60 for acceptance.
    "seller_top3_category_gross_share": ([92.11, 100, 76.60], "%"),
    "seller_new_product_gross_share": ([100, 71.43, 74.47], "%"),
    "seller_item_gross_value": ([380, 280, 470], "BRL"),
    "seller_avg_item_price": ([95, 70, 94], "BRL/item"),
    "seller_avg_freight_per_order": ([20, 7.67, 7.75], "BRL/order"),
    "seller_avg_payment_installments": ([2.55, 2.14, 1.92], "installments"),
    "seller_credit_card_payment_share": ([89.55, 85.71, 62.16], "%"),
    "seller_boleto_payment_share": ([10.45, 14.29, 37.84], "%"),
    "seller_active_product_count": ([4, 4, 5], "products"),
    "seller_active_category_count": ([4, 3, 5], "categories"),
    "seller_order_bad_review_rate": ([50, 33.33, 25], "%"),
    "seller_items_per_order": ([2, 1.33, 1.25], "items/order"),
    "seller_freight_to_gmv_rate": ([9.52, 7.59, 6.19], "%"),
}

VIEW_LINEAGE = {
    "analytics.v_delivered_order_items": {"analytics.fact_order_items"},
    "analytics.v_delivered_orders": {"analytics.fact_orders"},
    "analytics.v_seller_fulfillment": {"analytics.fact_fulfillment"},
    "analytics.v_product_category": {"raw.products"},
}
AUTH_TABLES = ("iam.tenants", "iam.users", "iam.tenant_memberships", "iam.membership_seller_grants")
FUNNEL_SINGLE_DAY_EXPECTED = {
    "order_approval_rate": 60.0,  # 3 approved / 5 target-seller orders on Jan 12
    "order_shipment_rate": 60.0,  # 3 shipped / 5 target-seller orders on Jan 12
    "order_final_delivery_rate": 80.0,  # 4 delivered / 5 target-seller orders on Jan 12
}


@pytest.fixture(scope="module")
def isolated_seller_trend_db():
    admin_text = os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL")
    if not admin_text:
        pytest.skip("Set ISOLATED_TEST_POSTGRES_ADMIN_URL to an isolated-test PostgreSQL maintenance URL")
    admin_url = make_url(admin_text)
    db_name = f"seller_trend_acceptance_{uuid4().hex}"
    admin = create_engine(admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT", pool_pre_ping=True)
    db = None
    trace: list[dict] = []
    cases: list[dict] = []
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{db_name}"'))
        db = create_engine(admin_url.set(database=db_name), pool_pre_ping=True)
        with db.begin() as conn:
            conn.execute(text("CREATE SCHEMA iam"))
            conn.execute(text("CREATE SCHEMA analytics"))
            conn.execute(text("CREATE SCHEMA raw"))
            conn.execute(text("CREATE TABLE iam.tenants (tenant_id text PRIMARY KEY, scope_type text NOT NULL, status text NOT NULL)"))
            conn.execute(text("CREATE TABLE iam.users (user_id text PRIMARY KEY, status text NOT NULL)"))
            conn.execute(text("CREATE TABLE iam.tenant_memberships (tenant_id text NOT NULL, user_id text NOT NULL, role text NOT NULL, revoked_at timestamptz)"))
            conn.execute(text("CREATE TABLE iam.membership_seller_grants (tenant_id text NOT NULL, user_id text NOT NULL, seller_id text NOT NULL, access_start_date date, access_end_date date, revoked_at timestamptz)"))
            conn.execute(text("""CREATE TABLE analytics.fact_orders (
                order_id text PRIMARY KEY, order_status text NOT NULL, purchase_date date NOT NULL,
                customer_unique_id text, item_count integer NOT NULL)"""))
            conn.execute(text("""CREATE TABLE analytics.fact_order_items (
                order_id text NOT NULL, order_item_id integer NOT NULL, seller_id text NOT NULL,
                product_id text NOT NULL, purchase_date date NOT NULL, order_status text NOT NULL,
                price numeric NOT NULL, freight_value numeric NOT NULL)"""))
            conn.execute(text("""CREATE VIEW analytics.v_delivered_orders AS
                SELECT order_id, customer_unique_id, purchase_date, item_count, order_status
                FROM analytics.fact_orders WHERE order_status='delivered'"""))
            conn.execute(text("""CREATE VIEW analytics.v_delivered_order_items AS
                SELECT * FROM analytics.fact_order_items WHERE order_status='delivered'"""))
            conn.execute(text("""CREATE TABLE analytics.fact_order_payments (
                seller_id text NOT NULL, order_id text NOT NULL, payment_installments integer,
                allocated_payment_value numeric NOT NULL, payment_type text)"""))
            conn.execute(text("""CREATE TABLE analytics.fact_fulfillment (
                seller_id text NOT NULL, order_id text NOT NULL, purchase_date date NOT NULL,
                order_status text NOT NULL, purchase_at timestamp, delivered_customer_at timestamp,
                estimated_delivery_at timestamp, approval_lead_hours numeric,
                dispatch_lead_hours numeric, delivery_transit_hours numeric, is_on_time boolean)"""))
            conn.execute(text("""CREATE VIEW analytics.v_seller_fulfillment AS
                SELECT * FROM analytics.fact_fulfillment"""))
            conn.execute(text("CREATE TABLE raw.products (product_id text PRIMARY KEY, product_category_name_english text)"))
            conn.execute(text("CREATE VIEW analytics.v_product_category AS SELECT product_id, product_category_name_english FROM raw.products"))
            conn.execute(text("""CREATE TABLE raw.orders (
                order_id text PRIMARY KEY, order_purchase_timestamp timestamp NOT NULL,
                order_approved_at timestamp, order_delivered_carrier_date timestamp,
                order_delivered_customer_date timestamp, order_estimated_delivery_date timestamp,
                order_status text NOT NULL)"""))
            conn.execute(text("""CREATE TABLE raw.order_items (
                order_id text NOT NULL, order_item_id integer NOT NULL, seller_id text NOT NULL,
                product_id text NOT NULL, shipping_limit_date timestamp, price numeric, freight_value numeric)"""))
            conn.execute(text("CREATE TABLE raw.order_reviews (order_id text NOT NULL, review_score integer, review_comment_message text)"))
            conn.execute(text("""INSERT INTO iam.tenants VALUES ('seller-trend-tenant','seller','active')"""))
            conn.execute(text("""INSERT INTO iam.users VALUES ('seller-trend-user','active')"""))
            conn.execute(text("""INSERT INTO iam.tenant_memberships VALUES ('seller-trend-tenant','seller-trend-user','analyst',NULL)"""))
            conn.execute(text("""INSERT INTO iam.membership_seller_grants VALUES
                ('seller-trend-tenant','seller-trend-user',:seller,'2024-01-09','2024-01-13',NULL)"""), {"seller": SELLER})

            # Each tuple is (order_id, customer, items[(product, price, freight)],
            # review score, non-empty review comment, approval/dispatch/transit,
            # delivered-day-offset, estimated-vs-delivered delta-days, on-time).
            daily = [
                [
                    ("d1a", "c1", [("p1",100,10),("p2",50,5),("p3",200,20)], 1, True, 1,2,10,2,2,True),
                    ("d1b", "c2", [("p4",30,5)], 5, False, 3,4,20,4,-1,False),
                ],
                [
                    ("d2a", "c1", [("p1",80,8),("p6",100,10)], 2, True, 2,3,5,1,3,True),
                    ("d2b", "c2", [("p5",40,0)], 4, False, 4,6,10,3,-2,False),
                    ("d2c", "c3", [("p7",60,5)], 5, True, 8,9,20,None,None,None),
                ],
                [
                    ("d3a", "c1", [("p5",120,12)], 4, True, 3,4,3,2,3,True),
                    ("d3b", "c4", [("p8",70,7)], 1, False, 6,8,7,4,2,True),
                    ("d3c", "c5", [("p9",40,0)], 3, True, 9,12,11,6,-4,False),
                    ("d3d", "c6", [("p10",160,8),("p11",80,4)], 5, True, 12,16,19,None,None,None),
                ],
            ]
            order_rows, item_rows, raw_order_rows, raw_item_rows = [], [], [], []
            payment_rows, fulfillment_rows, review_rows = [], [], []
            for day_index, orders in enumerate(daily, start=1):
                d = date(2024, 1, 9 + day_index)
                for order_index, row in enumerate(orders, start=1):
                    (oid, customer, items, score, has_comment, approval, dispatch, transit,
                     delivery_offset, estimate_delta, on_time) = row
                    order_rows.append({"oid":oid,"status":"delivered","day":d,"customer":customer,"item_count":len(items)})
                    for item_index, (product, price, freight) in enumerate(items, start=1):
                        item_rows.append({"oid":oid,"item":item_index,"seller":SELLER,"product":product,"day":d,"status":"delivered","price":price,"freight":freight})
                        raw_item_rows.append({"oid":oid,"item":item_index,"seller":SELLER,"product":product,"price":price,"freight":freight,"shipping":datetime.combine(d,datetime.min.time())})
                    purchase = datetime.combine(d, datetime.min.time())
                    delivered = purchase + timedelta(days=delivery_offset) if delivery_offset is not None else None
                    estimated = delivered + timedelta(days=estimate_delta) if delivered is not None else purchase + timedelta(days=8)
                    approved = purchase + timedelta(hours=1) if (day_index == 1 and order_index == 1) or (day_index == 2 and order_index <= 2) or (day_index == 3 and order_index <= 3) else None
                    shipped = purchase + timedelta(hours=4) if (day_index == 1 and order_index == 1) or (day_index == 2 and order_index <= 2) or (day_index == 3 and order_index <= 3) else None
                    raw_order_rows.append({"oid":oid,"purchase":purchase,"approved":approved,"shipped":shipped,"delivered":delivered,"estimated":estimated,"status":"delivered"})
                    fulfillment_rows.append({"seller":SELLER,"oid":oid,"day":d,"status":"delivered","purchase":purchase,"delivered":delivered,"estimated":estimated,"approval":approval,"dispatch":dispatch,"transit":transit,"on_time":on_time})
                    review_rows.append({"oid":oid,"score":score,"comment":"useful" if has_comment else "   "})
                    if oid == "d1a":
                        allocations = [(80,2,"credit_card"),(35,1,"boleto")]
                    elif oid == "d1b":
                        allocations = [(220,3,"credit_card")]
                    elif oid == "d2a":
                        allocations = [(100,1,"credit_card")]
                    elif oid == "d2b":
                        allocations = [(200,3,"credit_card")]
                    elif oid == "d2c":
                        allocations = [(50,1,"boleto")]
                    else:
                        allocations = {"d3a":[(90,1,"credit_card")],"d3b":[(60,2,"boleto")],"d3c":[(140,3,"credit_card")],"d3d":[(80,1,"boleto")]}[oid]
                    payment_rows.extend({"seller":SELLER,"oid":oid,"value":v,"installments":n,"type":t} for v,n,t in allocations)
                # One cancelled target order per date belongs to the seller's
                # funnel population, but not delivered-item metrics.
                cancel_id = f"d{day_index}cancel"
                cancel_purchase = datetime.combine(d, datetime.min.time()) + timedelta(hours=20)
                order_rows.append({"oid":cancel_id,"status":"canceled","day":d,"customer":None,"item_count":1})
                raw_order_rows.append({"oid":cancel_id,"purchase":cancel_purchase,"approved":None,"shipped":None,"delivered":None,"estimated":None,"status":"canceled"})
                item_rows.append({"oid":cancel_id,"item":1,"seller":SELLER,"product":"cancel-product","day":d,"status":"canceled","price":9999,"freight":999})
                raw_item_rows.append({"oid":cancel_id,"item":1,"seller":SELLER,"product":"cancel-product","price":9999,"freight":999,"shipping":cancel_purchase})

            # Authorized prior history, outside the queried trend range.
            history_day = date(2024, 1, 9)
            order_rows.extend([
                {"oid":"history-c1","status":"delivered","day":history_day,"customer":"c1","item_count":1},
                {"oid":"history-c2","status":"delivered","day":history_day,"customer":"c2","item_count":1},
            ])
            item_rows.extend([
                {"oid":"history-c1","item":1,"seller":SELLER,"product":"history-p1","day":history_day,"status":"delivered","price":10,"freight":1},
                {"oid":"history-c2","item":1,"seller":SELLER,"product":"history-p2","day":history_day,"status":"delivered","price":10,"freight":1},
            ])
            # Target-seller out-of-range noise after the period.
            outside_day = date(2024, 1, 13)
            order_rows.append({"oid":"outside-target","status":"delivered","day":outside_day,"customer":"outside","item_count":1})
            item_rows.append({"oid":"outside-target","item":1,"seller":SELLER,"product":"outside-product","day":outside_day,"status":"delivered","price":99999,"freight":9999})
            # Other-seller data includes both a shared target order and a
            # seller-exclusive order with large values. Only the shared order
            # identity is shared; customer/payment/item ownership stays scoped.
            item_rows.append({"oid":"d1a","item":99,"seller":OTHER,"product":"other-shared-product","day":date.fromisoformat(DATES[0]),"status":"delivered","price":50000,"freight":5000})
            order_rows.append({"oid":"other-only","status":"delivered","day":date(2024,1,11),"customer":"other-customer","item_count":1})
            item_rows.append({"oid":"other-only","item":1,"seller":OTHER,"product":"other-only-product","day":date(2024,1,11),"status":"delivered","price":80000,"freight":8000})
            raw_item_rows.extend([
                {"oid":"d1a","item":99,"seller":OTHER,"product":"other-shared-product","price":50000,"freight":5000,"shipping":datetime(2024,1,10)},
                {"oid":"other-only","item":1,"seller":OTHER,"product":"other-only-product","price":80000,"freight":8000,"shipping":datetime(2024,1,11)},
            ])
            raw_order_rows.append({"oid":"other-only","purchase":datetime(2024,1,11),"approved":datetime(2024,1,11,1),"shipped":datetime(2024,1,11,2),"delivered":datetime(2024,1,12),"estimated":datetime(2024,1,13),"status":"delivered"})
            raw_order_rows.append({"oid":"outside-target","purchase":datetime(2024,1,13),"approved":datetime(2024,1,13,1),"shipped":datetime(2024,1,13,2),"delivered":datetime(2024,1,14),"estimated":datetime(2024,1,15),"status":"delivered"})
            raw_item_rows.append({"oid":"outside-target","item":1,"seller":SELLER,"product":"outside-product","price":99999,"freight":9999,"shipping":datetime(2024,1,13)})
            # Fact orders are order-grain; shared orders exist once in this table.
            unique_orders = {r["oid"]: r for r in order_rows}
            conn.execute(text("INSERT INTO analytics.fact_orders VALUES (:oid,:status,:day,:customer,:item_count)"), list(unique_orders.values()))
            conn.execute(text("INSERT INTO analytics.fact_order_items VALUES (:oid,:item,:seller,:product,:day,:status,:price,:freight)"), item_rows)
            conn.execute(text("INSERT INTO analytics.fact_order_payments VALUES (:seller,:oid,:installments,:value,:type)"), payment_rows)
            conn.execute(text("INSERT INTO analytics.fact_fulfillment VALUES (:seller,:oid,:day,:status,:purchase,:delivered,:estimated,:approval,:dispatch,:transit,:on_time)"), fulfillment_rows)
            categories = {"p1":"cat_a","p2":"cat_b","p3":"cat_c","p4":"cat_d","p5":"cat_b","p6":"cat_c","p7":"cat_c","p8":"cat_a","p9":"cat_e","p10":"cat_f","p11":"cat_g","history-p1":"history_a","history-p2":"history_b","outside-product":"outside_cat","other-shared-product":"other_cat","other-only-product":"other_cat"}
            conn.execute(text("INSERT INTO raw.products VALUES (:product,:category)"), [{"product":p,"category":c} for p,c in categories.items()])
            conn.execute(text("INSERT INTO raw.orders VALUES (:oid,:purchase,:approved,:shipped,:delivered,:estimated,:status)"), raw_order_rows)
            conn.execute(text("INSERT INTO raw.order_items VALUES (:oid,:item,:seller,:product,:shipping,:price,:freight)"), raw_item_rows)
            conn.execute(text("INSERT INTO raw.order_reviews VALUES (:oid,:score,:comment)"), review_rows)

        @event.listens_for(db, "before_cursor_execute")
        def capture(_conn, _cursor, statement, parameters, _context, _many):
            trace.append({"statement":statement,"parameters":parameters})
        yield db, trace, cases
    finally:
        from _artifact_safety import isolated_output_path
        output_path = isolated_output_path("SELLER_TREND_RESULT_PATH")
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            registry = get_registry()
            inventory = [m for m in registry.list(scope="seller") if "trend" in m.implementations["seller"].capabilities]
            Path(output_path).write_text(json.dumps({"inventory_count":len(inventory),"inventory":[{"metric_id":m.metric_id,"unit":m.unit,"lineage_source":m.implementations["seller"].lineage_source} for m in inventory],"cases":cases},ensure_ascii=False,indent=2,default=str),encoding="utf-8")
        if db is not None:
            db.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture(scope="module")
def authorized_identity(isolated_seller_trend_db):
    return TenantContext(tenant_id="seller-trend-tenant", user_id="seller-trend-user", role="analyst", scope_type="seller", seller_id=SELLER, seller_grants=(SellerGrant(SELLER,"2024-01-09","2024-01-13"),))


def _relations(value: str) -> list[str]:
    return sorted(set(re.findall(r"\b(?:analytics|raw)\.[a-z_][a-z_0-9]*\b",value.lower())))


def test_seller_trend_registry_inventory_is_exactly_46():
    registry=get_registry()
    registered={m.metric_id for m in registry.list(scope="seller") if "trend" in m.implementations["seller"].capabilities}
    assert registered == set(EXPECTED)
    assert len(registered) == 46


@pytest.mark.parametrize("metric_id",sorted(EXPECTED))
def test_seller_daily_trend_via_authorized_registry(monkeypatch,isolated_seller_trend_db,authorized_identity,metric_id):
    db,trace,cases=isolated_seller_trend_db
    monkeypatch.setattr(explorer,"engine",db)
    monkeypatch.setattr(seller_extensions,"engine",db)
    registry=get_registry(); definition=registry.get(metric_id)
    scope=ControlledAnalysisScope(seller_id=SELLER,allowed_metrics=tuple(sorted(EXPECTED)),default_start_date=START,default_end_date=END,metric_name=metric_id,granularity="day",max_date_span_days=10,max_trend_points=10)
    trace_start=len(trace); exception=None
    try:
        with activate_tenant_context(authorized_identity):
            actual=registry.execute(metric_id,operation="trend",scope="seller",seller_scope=scope,start_date=START,end_date=END,granularity="day")
    except Exception as exc:
        exception=f"{type(exc).__name__}: {exc}"
        actual={"metric_name":metric_id,"points":[],"source_table":None,"status":"exception"}
    authorized_trace=trace[trace_start:]
    boundary_check=None
    boundary_trace=[]
    if metric_id in FUNNEL_SINGLE_DAY_EXPECTED:
        boundary_start=len(trace)
        try:
            with activate_tenant_context(authorized_identity):
                single_day=registry.execute(metric_id,operation="trend",scope="seller",seller_scope=scope,start_date=END,end_date=END,granularity="day")
        except Exception as exc:
            single_day={"metric_name":metric_id,"points":[],"status":"exception"}
            boundary_exception=f"{type(exc).__name__}: {exc}"
        else:
            boundary_exception=None
        boundary_trace=trace[boundary_start:]
        boundary_sql=" ".join(event["statement"].lower() for event in boundary_trace if "select" in event["statement"].lower() and " from iam." not in event["statement"].lower())
        boundary_params=[event["parameters"] for event in boundary_trace if (" from analytics." in event["statement"].lower() or " from raw." in event["statement"].lower()) and isinstance(event["parameters"],dict)]
        boundary_points=single_day.get("points",[])
        boundary_values=[point.get("value") for point in boundary_points]
        boundary_dates=[str(point.get("bucket")) for point in boundary_points]
        boundary_check={
            "expected_value_from_fixture":FUNNEL_SINGLE_DAY_EXPECTED[metric_id],
            "actual_value":boundary_values,
            "dates":boundary_dates,
            "status":single_day.get("status"),
            "inclusive_start_day_midnight_and_end_day_evening":len(boundary_values)==1 and boundary_dates==[END.isoformat()] and boundary_values[0] == pytest.approx(FUNNEL_SINGLE_DAY_EXPECTED[metric_id],abs=0.01),
            "next_day_midnight_excluded":boundary_dates==[END.isoformat()],
            "half_open_timestamp_predicate":">= %(start_date)s" in boundary_sql and "< %(end_date_exclusive)s" in boundary_sql,
            "exclusive_bound_is_next_calendar_day":any(str(params.get("end_date_exclusive"))=="2024-01-13" for params in boundary_params),
            "exception":boundary_exception,
            "sql_trace":boundary_trace,
        }
        assert boundary_check["status"]=="ok" and all(boundary_check[key] for key in ("inclusive_start_day_midnight_and_end_day_evening","next_day_midnight_excluded","half_open_timestamp_predicate","exclusive_bound_is_next_calendar_day")),boundary_check
        authorized_trace += boundary_trace
    # Probe the same registered call under the same tenant/membership but a
    # seller ID with no grant; this must be denied before any data SELECT.
    untrusted_scope=scope.model_copy(update={"seller_id":OTHER})
    deny_start=len(trace); deny_exception=None
    try:
        with activate_tenant_context(authorized_identity):
            registry.execute(metric_id,operation="trend",scope="seller",seller_scope=untrusted_scope,start_date=START,end_date=END,granularity="day")
    except Exception as exc:
        deny_exception=f"{type(exc).__name__}: {exc}"
    deny_trace=trace[deny_start:]
    point_dates=[str(point.get("bucket")) for point in actual.get("points",[])]
    actual_values=[point.get("value") for point in actual.get("points",[])]
    expected_values,expected_unit=EXPECTED[metric_id]
    business_sql=" ".join(event["statement"].lower() for event in authorized_trace if "select" in event["statement"].lower() and " from iam." not in event["statement"].lower())
    sql_relations=_relations(business_sql)
    business_params=[event["parameters"] for event in authorized_trace if (" from analytics." in event["statement"].lower() or " from raw." in event["statement"].lower()) and isinstance(event["parameters"],dict)]
    auth_sql=" ".join(event["statement"].lower() for event in authorized_trace)
    auth_checks={
        "tenant_membership_and_seller_grant_checked":all(table in auth_sql for table in AUTH_TABLES),
        "authorized_query_binds_target_seller":any(p.get("seller_id")==SELLER for p in business_params),
        "unauthorized_seller_denied":deny_exception is not None and "grant" in deny_exception.lower(),
        "denied_probe_has_no_business_select":not any(" from analytics." in e["statement"].lower() or " from raw." in e["statement"].lower() for e in deny_trace),
        "authorized_context_matches_tenant_and_user":authorized_identity.tenant_id=="seller-trend-tenant" and authorized_identity.user_id=="seller-trend-user" and authorized_identity.seller_grant(SELLER) is not None,
    }
    lineage=definition.implementations["seller"].lineage_source
    trace_plus_views=set(sql_relations)
    for relation in list(trace_plus_views): trace_plus_views.update(VIEW_LINEAGE.get(relation,set()))
    lineage_relations=_relations(lineage)
    source_checks={
        "returned_source_present":bool(actual.get("source_table")),
        "returned_source_relations_in_sql_or_view_lineage":all(token in trace_plus_views for token in _relations(str(actual.get("source_table") or ""))),
        "registered_lineage_covered_by_sql_or_view_lineage":all(token in trace_plus_views for token in lineage_relations),
    }
    if expected_values is None:
        numeric_checks={"status_and_trace":actual.get("status")!="exception" and bool(authorized_trace)}
        numeric_status="unscored_history_contract"
        expected={"values":None,"dates":DATES,"unit":expected_unit,"reason":"repeat_purchase_rate requires proven complete seller history coverage; registry/query scope exposes no history coverage start or completeness status"}
    else:
        numeric_checks={
            "values":len(actual_values)==3 and all(v is not None and float(v)==pytest.approx(float(e),abs=0.01) for v,e in zip(actual_values,expected_values)),
            "date_range_and_sorted_daily_buckets":point_dates==DATES,
            "status":actual.get("status")=="ok",
            "registered_unit_matches_expected_contract":definition.unit==expected_unit,
        }
        numeric_status="pass" if all(numeric_checks.values()) else "fail"
        expected={"values":expected_values,"dates":DATES,"unit":expected_unit}
    case={"metric_id":metric_id,"expected":expected,"actual":actual,"boundary_check":boundary_check,"registered_unit":definition.unit,"registered_lineage_source":lineage,"actual_sql_relations":sql_relations,"numeric_checks":numeric_checks,"numeric_status":numeric_status,"numeric_pass":numeric_status=="pass","authorization_checks":auth_checks,"authorization_pass":all(auth_checks.values()),"source_metadata_checks":source_checks,"source_metadata_pass":all(source_checks.values()),"exception":exception,"unauthorized_probe_exception":deny_exception,"sql_trace":authorized_trace,"authorization_trace":[{"phase":"authorized","events":authorized_trace},{"phase":"ungranted seller probe","events":deny_trace}]}
    cases.append(case)
    print("SELLER_TREND_RESULT "+json.dumps(case,ensure_ascii=False,sort_keys=True,default=str))
    if expected_values is not None:
        assert all(auth_checks.values()),case
        assert all(numeric_checks.values()),case
