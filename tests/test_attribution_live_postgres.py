"""Opt-in live PostgreSQL checks for attribution data paths.

Run with RUN_ATTRIBUTION_DB_INTEGRATION=1 and DB_USE_RLS_ROLE=true. The
checks reconcile registered association and freight results against scoped
source rows. The inventory test inserts one uniquely keyed synthetic
observation and removes only that exact row in a finally block.
"""

from __future__ import annotations

import math
import hashlib
import json
import os
import threading
from datetime import date, datetime, time, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.data.database import engine
from app.data.inventory_contracts import (
    InventorySnapshotBatch,
    InventorySnapshotInput,
    InventorySnapshotQuery,
)
from app.data.repositories.inventory import (
    ingest_inventory_snapshot_batch,
    query_inventory_snapshot,
)
from app.data.rls import set_local_context
from app.schemas.explorer import ControlledAnalysisScope
from app.tenancy import SellerGrant, TenantContext, activate_tenant_context


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_ATTRIBUTION_DB_INTEGRATION") != "1",
    reason="set RUN_ATTRIBUTION_DB_INTEGRATION=1 to run live PostgreSQL checks",
)


def _seed_live_test_identity(
    connection,
    *,
    tenant_id: str,
    user_id: str,
    seller_id: str,
    start_date: date,
    end_date: date,
    role: str = "analyst",
    create_tenant: bool = False,
) -> None:
    if create_tenant:
        connection.execute(text("""
            INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
            VALUES (:tenant, 'live authorization integration test', 'seller', 'active')
        """), {"tenant": tenant_id})
    connection.execute(text("""
        INSERT INTO iam.users(user_id, display_name, email, status)
        VALUES (:user, 'live authorization integration test', :email, 'active')
    """), {"user": user_id, "email": f"{user_id}@example.invalid"})
    connection.execute(text("""
        INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
        VALUES (:tenant, :user, :role)
    """), {"tenant": tenant_id, "user": user_id, "role": role})
    connection.execute(text("""
        INSERT INTO iam.membership_seller_grants(
            tenant_id, user_id, seller_id, access_start_date, access_end_date
        ) VALUES (:tenant, :user, :seller, :start_date, :end_date)
    """), {
        "tenant": tenant_id, "user": user_id, "seller": seller_id,
        "start_date": start_date, "end_date": end_date,
    })


def _mapped_tenant_for_seller(seller_id: str) -> str | None:
    with engine.connect() as connection:
        value = connection.execute(text("""
            SELECT tenant_id FROM iam.tenant_seller_accounts
            WHERE seller_id=:seller_id
        """), {"seller_id": seller_id}).scalar_one_or_none()
    return str(value) if value is not None else None


def _cleanup_live_test_identity(tenant_id: str, user_id: str, *, delete_tenant: bool = False) -> None:
    with engine.begin() as connection:
        connection.execute(text("RESET ROLE"))
        connection.execute(text("DELETE FROM iam.auth_sessions WHERE tenant_id=:tenant AND user_id=:user"), {"tenant": tenant_id, "user": user_id})
        connection.execute(text("DELETE FROM iam.api_credentials WHERE tenant_id=:tenant AND user_id=:user"), {"tenant": tenant_id, "user": user_id})
        connection.execute(text("DELETE FROM iam.membership_seller_grants WHERE tenant_id=:tenant AND user_id=:user"), {"tenant": tenant_id, "user": user_id})
        connection.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant AND user_id=:user"), {"tenant": tenant_id, "user": user_id})
        connection.execute(text("DELETE FROM iam.users WHERE user_id=:user"), {"user": user_id})
        if delete_tenant:
            connection.execute(text("DELETE FROM iam.tenants WHERE tenant_id=:tenant"), {"tenant": tenant_id})


def test_seller_grant_revocation_serializes_with_live_query_transaction() -> None:
    """A seller revoke waits for an already-authorized bounded query transaction."""
    tenant_id = f"it_lease_t_{uuid4().hex}"
    user_id = f"it_lease_u_{uuid4().hex}"
    seller_id = f"it_lease_s_{uuid4().hex}"
    session_id = uuid4().hex
    today = date.today()
    context = TenantContext(
        tenant_id=tenant_id,
        user_id=user_id,
        role="analyst",
        scope_type="seller",
        seller_id=seller_id,
        seller_grants=(SellerGrant(seller_id=seller_id),),
        auth_session_id=session_id,
    )
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
            VALUES (:tenant, 'authorization lease integration test', 'seller', 'active')
        """), {"tenant": tenant_id})
        connection.execute(text("""
            INSERT INTO iam.users(user_id, display_name, email, status)
            VALUES (:user, 'authorization lease integration test', :email, 'active')
        """), {"user": user_id, "email": f"{user_id}@example.invalid"})
        connection.execute(text("""
            INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
            VALUES (:tenant, :user, 'analyst')
        """), {"tenant": tenant_id, "user": user_id})
        connection.execute(text("""
            INSERT INTO iam.membership_seller_grants(
                tenant_id, user_id, seller_id, access_start_date, access_end_date
            ) VALUES (:tenant, :user, :seller, :start_date, :end_date)
        """), {
            "tenant": tenant_id, "user": user_id, "seller": seller_id,
            "start_date": today, "end_date": today,
        })
        connection.execute(text("""
            INSERT INTO iam.auth_sessions(session_id, tenant_id, user_id, expires_at)
            VALUES (:session, :tenant, :user, now() + interval '1 hour')
        """), {"session": session_id, "tenant": tenant_id, "user": user_id})

    revoke_started = threading.Event()
    revoke_update_completed = threading.Event()
    revoke_errors: list[BaseException] = []

    def revoke_grant() -> None:
        try:
            revoke_started.set()
            with engine.begin() as connection:
                connection.execute(text("""
                    UPDATE iam.membership_seller_grants
                    SET revoked_at=now()
                    WHERE tenant_id=:tenant AND user_id=:user AND seller_id=:seller
                """), {"tenant": tenant_id, "user": user_id, "seller": seller_id})
                revoke_update_completed.set()
        except BaseException as exc:  # surfaced in the test thread below
            revoke_errors.append(exc)

    try:
        with engine.begin() as query_connection:
            set_local_context(
                query_connection,
                context,
                scope_type="seller",
                seller_access={seller_id: (today, today)},
            )
            worker = threading.Thread(target=revoke_grant, daemon=True)
            worker.start()
            assert revoke_started.wait(timeout=2)
            # UPDATE must wait behind the FOR SHARE lease held by this query
            # transaction, rather than revoking authority midway through SQL.
            assert not revoke_update_completed.wait(timeout=0.25)
            assert query_connection.execute(text("SELECT 1")).scalar_one() == 1
        worker.join(timeout=5)
        assert not worker.is_alive(), "seller grant revocation did not resume after query commit"
        assert not revoke_errors, revoke_errors
        assert revoke_update_completed.is_set()

        with engine.begin() as connection:
            with pytest.raises(PermissionError, match="Seller grant was revoked"):
                set_local_context(
                    connection,
                    context,
                    scope_type="seller",
                    seller_access={seller_id: (today, today)},
                )
    finally:
        with engine.begin() as connection:
            connection.execute(text("RESET ROLE"))
            connection.execute(text("DELETE FROM iam.auth_sessions WHERE session_id=:session"), {"session": session_id})
            connection.execute(text("DELETE FROM iam.membership_seller_grants WHERE tenant_id=:tenant"), {"tenant": tenant_id})
            connection.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant"), {"tenant": tenant_id})
            connection.execute(text("DELETE FROM iam.tenants WHERE tenant_id=:tenant"), {"tenant": tenant_id})
            connection.execute(text("DELETE FROM iam.users WHERE user_id=:user"), {"user": user_id})


def test_inventory_snapshot_roundtrip_and_rls_with_app_role() -> None:
    tenant_id = f"it_inv_t_{uuid4().hex}"
    seller_id = f"it_inv_s_{uuid4().hex}"
    warehouse_id = f"it_inv_w_{uuid4().hex}"
    product_id = f"it_inv_p_{uuid4().hex}"
    source_system = f"it_inv_src_{uuid4().hex}"
    batch_id = f"it_inv_batch_{uuid4().hex}"
    today = date.today()
    snapshot_at = datetime.combine(today, time(23, 0), tzinfo=timezone.utc)
    tenant = TenantContext(
        tenant_id=tenant_id,
        user_id=f"it_inv_u_{uuid4().hex}",
        role="owner",
        scope_type="seller",
        seller_id=seller_id,
        seller_grants=(SellerGrant(seller_id=seller_id),),
    )
    batch = InventorySnapshotBatch.model_validate({
        "source_system": source_system,
        "source_batch_id": batch_id,
        "rows": [InventorySnapshotInput.model_validate({
            "seller_id": seller_id,
            "warehouse_id": warehouse_id,
            "product_id": product_id,
            "snapshot_at": snapshot_at,
            "snapshot_semantics": "daily_close",
            "on_hand_quantity": "12",
            "reserved_quantity": "3",
            "available_quantity": "9",
            "quantity_unit": "unit",
        })],
    })
    scope = ControlledAnalysisScope.model_validate({
        "seller_id": seller_id,
        "allowed_metrics": ["gmv"],
        "default_start_date": today,
        "default_end_date": today,
        "metric_name": "gmv",
        "granularity": "day",
        "max_rows": 10,
    })
    query = InventorySnapshotQuery.model_validate({
        "seller_id": seller_id,
        "as_of": datetime.combine(today, time(23, 59, 59), tzinfo=timezone.utc),
        "quantity_basis": "available",
        "limit": 10,
    })

    try:
        with engine.begin() as connection:
            _seed_live_test_identity(
                connection,
                tenant_id=tenant_id,
                user_id=tenant.user_id,
                seller_id=seller_id,
                start_date=today,
                end_date=today,
                role="owner",
                create_tenant=True,
            )
        with activate_tenant_context(tenant):
            inserted = ingest_inventory_snapshot_batch(batch)
            assert inserted["inserted_rows"] == 1
            result = query_inventory_snapshot(scope, query)
        assert result["status"] == "ok"
        assert result["coverage_status"] == "unknown_without_authoritative_product_warehouse_catalog"
        assert result["items"] == [{
            "warehouse_id": warehouse_id,
            "product_id": product_id,
            "snapshot_at": snapshot_at.isoformat(),
            "snapshot_semantics": "daily_close",
            "quantity": 9.0,
            "quantity_unit": "unit",
            "source_system": source_system,
            "measurement_status": "observed",
        }]

        with engine.begin() as connection:
            assert connection.execute(text("SELECT current_user")).scalar_one() == "bizinsight_app"
            # The table's tenant RLS policy must deny reads without transaction context.
            assert connection.execute(text(
                "SELECT count(*) FROM analytics.inventory_snapshots WHERE seller_id=:seller"
            ), {"seller": seller_id}).scalar_one() == 0
            set_local_context(connection, tenant, scope_type="seller")
            assert connection.execute(text(
                "SELECT count(*) FROM analytics.inventory_snapshots WHERE seller_id=:seller"
            ), {"seller": seller_id}).scalar_one() == 1
    finally:
        # The local bootstrap connection is a superuser; return to it only for
        # exact-key cleanup, then the pool checkout hook restores app role.
        with engine.begin() as connection:
            connection.execute(text("RESET ROLE"))
            connection.execute(text("""
                DELETE FROM analytics.inventory_snapshots
                WHERE tenant_id=:tenant AND seller_id=:seller
                  AND warehouse_id=:warehouse AND product_id=:product
                  AND snapshot_at=:snapshot AND source_system=:source
                  AND source_batch_id=:batch
            """), {
                "tenant": tenant_id,
                "seller": seller_id,
                "warehouse": warehouse_id,
                "product": product_id,
                "snapshot": snapshot_at,
                "source": source_system,
                "batch": batch_id,
            })
            connection.execute(text("DELETE FROM iam.membership_seller_grants WHERE tenant_id=:tenant"), {"tenant": tenant_id})
            connection.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant"), {"tenant": tenant_id})
            connection.execute(text("DELETE FROM iam.tenants WHERE tenant_id=:tenant"), {"tenant": tenant_id})
            connection.execute(text("DELETE FROM iam.users WHERE user_id=:user"), {"user": tenant.user_id})


def test_registered_association_queries_execute_against_live_order_view(request) -> None:
    from app.analysis.association_catalog import get_association_pair
    from app.data.repositories.explorer import query_scoped_delivery_review_association

    with engine.connect() as connection:
        row = connection.execute(text("""
            SELECT seller_id, MIN(purchase_date) AS start_date,
                   MAX(purchase_date) AS end_date,
                   COUNT(DISTINCT order_id) FILTER (
                     WHERE review_score IS NOT NULL AND is_on_time IS NOT NULL
                   ) AS binary_n,
                   COUNT(DISTINCT order_id) FILTER (
                     WHERE review_score IS NOT NULL
                       AND delivery_transit_hours IS NOT NULL
                       AND delivery_transit_hours >= 0
                   ) AS delivery_n,
                   COUNT(DISTINCT order_id) FILTER (
                     WHERE review_score IS NOT NULL
                       AND freight_sum IS NOT NULL
                       AND freight_sum >= 0
                   ) AS freight_n,
                   COUNT(DISTINCT order_id) FILTER (
                     WHERE review_score IS NOT NULL AND item_count IS NOT NULL AND item_count >= 0
                   ) AS item_line_n,
                   COUNT(DISTINCT order_id) FILTER (
                     WHERE review_score IS NOT NULL AND price_sum IS NOT NULL AND price_sum >= 0
                   ) AS merchandise_value_n
            FROM analytics.v_order_unified
            WHERE is_single_seller_order
            GROUP BY seller_id
            HAVING COUNT(DISTINCT order_id) FILTER (
                     WHERE review_score IS NOT NULL AND is_on_time IS NOT NULL
                   ) >= 50
               AND COUNT(DISTINCT order_id) FILTER (
                     WHERE review_score IS NOT NULL
                       AND delivery_transit_hours IS NOT NULL
                       AND delivery_transit_hours >= 0
                   ) >= 50
               AND COUNT(DISTINCT order_id) FILTER (
                     WHERE review_score IS NOT NULL
                       AND freight_sum IS NOT NULL
                       AND freight_sum >= 0
                   ) >= 50
               AND COUNT(DISTINCT order_id) FILTER (
                     WHERE review_score IS NOT NULL AND item_count IS NOT NULL AND item_count >= 0
                   ) >= 50
               AND COUNT(DISTINCT order_id) FILTER (
                     WHERE review_score IS NOT NULL AND price_sum IS NOT NULL AND price_sum >= 0
                   ) >= 50
            ORDER BY LEAST(
              COUNT(DISTINCT order_id) FILTER (
                WHERE review_score IS NOT NULL AND is_on_time IS NOT NULL
              ),
              COUNT(DISTINCT order_id) FILTER (
                WHERE review_score IS NOT NULL
                  AND delivery_transit_hours IS NOT NULL
                  AND delivery_transit_hours >= 0
              ),
              COUNT(DISTINCT order_id) FILTER (
                WHERE review_score IS NOT NULL
                  AND freight_sum IS NOT NULL
                  AND freight_sum >= 0
              ),
              COUNT(DISTINCT order_id) FILTER (
                WHERE review_score IS NOT NULL AND item_count IS NOT NULL AND item_count >= 0
              ),
              COUNT(DISTINCT order_id) FILTER (
                WHERE review_score IS NOT NULL AND price_sum IS NOT NULL AND price_sum >= 0
              )
            )
            LIMIT 1
        """)).mappings().one_or_none()
    if row is None:
        pytest.skip("the live database has no seller with 50 eligible observations for all registered pairs")

    seller_id = str(row["seller_id"])
    start, end = row["start_date"], row["end_date"]
    tenant_id = _mapped_tenant_for_seller(seller_id)
    if tenant_id is None:
        pytest.skip("the observed live seller has no tenant mapping for an authorization-backed query")
    user_id = f"attribution_live_{uuid4().hex}"
    with engine.begin() as connection:
        _seed_live_test_identity(
            connection,
            tenant_id=tenant_id,
            user_id=user_id,
            seller_id=seller_id,
            start_date=start,
            end_date=end,
        )
    request.addfinalizer(lambda: _cleanup_live_test_identity(tenant_id, user_id))
    tenant = TenantContext(
        tenant_id=tenant_id,
        user_id=user_id,
        role="analyst",
        scope_type="seller",
        seller_id=seller_id,
        seller_grants=(SellerGrant(seller_id=seller_id),),
    )
    for pair_id in (
        "delivery_timing_bad_review_v1",
        "delivery_hours_review_score_spearman_v1",
        "order_freight_review_score_spearman_v1",
        "order_item_line_count_review_score_spearman_v1",
        "order_merchandise_value_review_score_spearman_v1",
    ):
        pair = get_association_pair(pair_id)
        assert pair is not None
        scope = ControlledAnalysisScope.model_validate({
            "seller_id": seller_id,
            "allowed_metrics": [pair.target_metric_id],
            "default_start_date": start,
            "default_end_date": end,
            "metric_name": pair.target_metric_id,
            "granularity": "day",
            "max_rows": 10_000,
        })
        with activate_tenant_context(tenant):
            # Independently read the scoped source observations, then compare
            # the repository result to an in-process recomputation. This checks
            # SQL filters/counts as well as the registered result contract; it
            # does not treat production observations as synthetic test data.
            with engine.begin() as connection:
                set_local_context(
                    connection, tenant, scope_type="seller",
                    seller_access={seller_id: (start, end)},
                )
                source_rows = connection.execute(text("""
                    SELECT order_id, review_score, is_on_time,
                           delivery_transit_hours, freight_sum, item_count, price_sum
                    FROM analytics.v_order_unified
                    WHERE seller_id=:seller_id
                      AND purchase_date BETWEEN :start_date AND :end_date
                      AND is_single_seller_order
                    ORDER BY purchase_date, order_id
                """), {
                    "seller_id": seller_id,
                    "start_date": start,
                    "end_date": end,
                }).mappings().all()
            result = query_scoped_delivery_review_association(
                scope, start_date=start, end_date=end, pair_id=pair_id,
            )
        assert result["pair_id"] == pair_id
        assert result["population_orders"] >= result["eligible_orders"] >= 50
        assert result["excluded_missing_pair_fields"] == (
            result["population_orders"] - result["eligible_orders"]
        )
        assert result["source_table"] == "analytics.v_order_unified"
        assert "order_id" not in result

        # The source view is order-level, but assert uniqueness so a future
        # view change cannot silently make the independent comparison invalid.
        source_by_order = {str(item["order_id"]): item for item in source_rows}
        assert len(source_by_order) == len(source_rows)
        assert len(source_rows) == result["population_orders"]
        if pair_id == "delivery_timing_bad_review_v1":
            from app.analysis.association_stats import summarize_binary_association

            eligible = [
                item for item in source_rows
                if item["review_score"] is not None and item["is_on_time"] is not None
            ]
            delayed = [item for item in eligible if item["is_on_time"] is False]
            on_time = [item for item in eligible if item["is_on_time"] is True]
            expected = summarize_binary_association(
                exposed_events=sum(item["review_score"] <= 2 for item in delayed),
                exposed_total=len(delayed),
                comparison_events=sum(item["review_score"] <= 2 for item in on_time),
                comparison_total=len(on_time),
                min_total=pair.minimum_sample_size,
            )
            assert result["eligible_orders"] == len(eligible)
            assert result["exposed"]["n"] == len(delayed)
            assert result["comparison"]["n"] == len(on_time)
            assert result["exposed"]["events"] == expected["exposed"]["events"]
            assert result["comparison"]["events"] == expected["comparison"]["events"]
            assert result["risk_difference"] == pytest.approx(expected["risk_difference"])
            assert result["risk_difference_ci_95"] == pytest.approx(
                expected["risk_difference_ci_95"]
            )
        else:
            from app.analysis.association_stats import summarize_spearman_association

            exposure_column = {
                "delivery_hours_review_score_spearman_v1": "delivery_transit_hours",
                "order_freight_review_score_spearman_v1": "freight_sum",
                "order_item_line_count_review_score_spearman_v1": "item_count",
                "order_merchandise_value_review_score_spearman_v1": "price_sum",
            }[pair_id]
            eligible = [
                item for item in source_rows
                if item["review_score"] is not None
                and item[exposure_column] is not None
                and item[exposure_column] >= 0
            ]
            expected = summarize_spearman_association(
                [
                    (float(item[exposure_column]), float(item["review_score"]))
                    for item in eligible
                ],
                population_n=len(source_rows),
                missing_n=len(source_rows) - len(eligible),
                min_total=pair.minimum_sample_size,
            )
            assert result["eligible_orders"] == len(eligible)
            assert result["spearman_rho"] == pytest.approx(expected["spearman_rho"])
            assert result["spearman_rho_ci_95_approx"] == pytest.approx(
                expected["spearman_rho_ci_95_approx"]
            )


def test_binary_association_sql_matches_isolated_extreme_fixtures(monkeypatch) -> None:
    """Exercise the registered SQL on unique rows and clean them up exactly.

    Fixtures are inserted only into the shared local test database under
    unique keys, queried through the non-owner RLS role, and deleted in
    ``finally``. They are never loaded into or presented as production data.
    """
    from app.data.repositories.explorer import query_scoped_delivery_review_association
    from app.auth import context as auth_context

    monkeypatch.setattr(
        auth_context,
        "_policy_tools",
        lambda *_args: ("query_scoped_registered_association",),
    )

    suffix = uuid4().hex
    fixture_token = suffix[:10]
    order_prefix = f"ia_{fixture_token}_"
    start = date(2024, 1, 1)
    end = date(2024, 1, 31)
    seller_specs = {
        # 30 on-time + 30 delayed; neither group has a low-score event.
        "zero_events": [(True, 5)] * 30 + [(False, 5)] * 30,
        # All delayed orders are low-score; no on-time order is low-score.
        "separated_extremes": [(True, 5)] * 30 + [(False, 1)] * 30,
        # Only the on-time group exists, so the risk difference is undefined.
        "one_exposure_group": [(True, 1 if i < 15 else 5) for i in range(60)],
        # Population exists, but every review is missing.
        "all_reviews_missing": [(True, None)] * 30 + [(False, None)] * 30,
        # Highly unequal but non-degenerate rates: 1/30 versus 29/30.
        "extreme_rates": (
            [(False, 1)] + [(False, 5)] * 29
            + [(True, 1)] * 29 + [(True, 5)]
        ),
        # Both groups exist, but the total sample is below the registered 50.
        "below_minimum_sample": [(True, 5)] * 10 + [(False, 1)] * 10,
    }
    order_ids: list[str] = []
    item_rows = []
    order_rows = []
    allocation_rows = []
    fulfillment_rows = []
    review_rows = []
    for scenario, observations in seller_specs.items():
        seller_id = f"ias_{fixture_token}_{scenario}"
        for index, (is_on_time, review_score) in enumerate(observations):
            order_id = f"{order_prefix}{scenario}_{index:03d}"
            order_ids.append(order_id)
            purchased_at = datetime(2024, 1, 10, 12, 0)
            order_rows.append({
                "order_id": order_id, "order_status": "delivered",
                "purchase_at": purchased_at, "purchase_date": start,
            })
            item_rows.append({
                "order_id": order_id, "seller_id": seller_id,
                "order_item_id": 1, "product_id": f"iap_{fixture_token}",
                "shipping_limit_date": purchased_at, "price": 1, "freight_value": 0,
            })
            allocation_rows.append({
                "order_id": order_id, "seller_id": seller_id,
            })
            fulfillment_rows.append({
                "order_id": order_id, "purchase_date": start,
                "order_status": "delivered", "is_on_time": is_on_time,
            })
            if review_score is not None:
                review_rows.append({"review_id": order_id, "order_id": order_id, "review_score": review_score})

    tenant_id = f"it_assoc_boundary_t_{suffix}"
    user_id = f"it_assoc_boundary_u_{suffix}"
    tenant = TenantContext(
        tenant_id=tenant_id,
        user_id=user_id,
        role="analyst",
        scope_type="seller",
        seller_id=f"ias_{fixture_token}_zero_events",
        seller_grants=tuple(
            SellerGrant(
                seller_id=f"ias_{fixture_token}_{scenario}",
                start_date=start.isoformat(), end_date=end.isoformat(),
            )
            for scenario in seller_specs
        ),
    )
    try:
        with engine.begin() as connection:
            connection.execute(text("RESET ROLE"))
            _seed_live_test_identity(
                connection,
                tenant_id=tenant_id,
                user_id=user_id,
                seller_id=f"ias_{fixture_token}_zero_events",
                start_date=start,
                end_date=end,
                create_tenant=True,
            )
            for scenario in tuple(seller_specs)[1:]:
                connection.execute(text("""
                    INSERT INTO iam.membership_seller_grants(
                      tenant_id, user_id, seller_id, access_start_date, access_end_date
                    ) VALUES (:tenant, :user, :seller, :start_date, :end_date)
                """), {
                    "tenant": tenant_id,
                    "user": user_id,
                    "seller": f"ias_{fixture_token}_{scenario}",
                    "start_date": start,
                    "end_date": end,
                })
            connection.execute(text("""
                INSERT INTO analytics.fact_orders (
                  order_id, order_status, order_purchase_timestamp, purchase_date,
                  payment_value, item_count, item_gross_value, freight_value
                ) VALUES (
                  :order_id, :order_status, :purchase_at, :purchase_date,
                  1, 1, 1, 0
                )
            """), order_rows)
            connection.execute(text("""
                INSERT INTO raw.order_items (
                  order_id, order_item_id, product_id, seller_id,
                  shipping_limit_date, price, freight_value
                ) VALUES (
                  :order_id, :order_item_id, :product_id, :seller_id,
                  :shipping_limit_date, :price, :freight_value
                )
            """), item_rows)
            connection.execute(text("""
                INSERT INTO analytics.fact_order_seller_allocation (
                  order_id, seller_id, seller_gross_freight_value,
                  order_gross_freight_value, seller_allocation_ratio,
                  seller_count, is_single_seller_order
                ) VALUES (:order_id, :seller_id, 0, 0, 1, 1, TRUE)
            """), allocation_rows)
            connection.execute(text("""
                INSERT INTO analytics.fact_fulfillment (
                  order_id, purchase_date, order_status, is_on_time
                ) VALUES (:order_id, :purchase_date, :order_status, :is_on_time)
            """), fulfillment_rows)
            if review_rows:
                connection.execute(text("""
                    INSERT INTO raw.order_reviews (review_id, order_id, review_score)
                    VALUES (:review_id, :order_id, :review_score)
                """), review_rows)

        for scenario in seller_specs:
            seller_id = f"ias_{fixture_token}_{scenario}"
            scope = ControlledAnalysisScope.model_validate({
                "seller_id": seller_id,
                "allowed_metrics": ["seller_order_bad_review_rate"],
                "default_start_date": start,
                "default_end_date": end,
                "metric_name": "seller_order_bad_review_rate",
                "granularity": "day",
                "max_rows": 1000,
            })
            with activate_tenant_context(tenant):
                result = query_scoped_delivery_review_association(
                    scope, start_date=start, end_date=end,
                    pair_id="delivery_timing_bad_review_v1",
                )
            observations = seller_specs[scenario]
            eligible = [(on_time, score) for on_time, score in observations if score is not None]
            delayed = [(on_time, score) for on_time, score in eligible if on_time is False]
            on_time = [(flag, score) for flag, score in eligible if flag is True]
            from app.analysis.association_stats import summarize_binary_association

            recomputed = summarize_binary_association(
                exposed_events=sum(score <= 2 for _, score in delayed),
                exposed_total=len(delayed),
                comparison_events=sum(score <= 2 for _, score in on_time),
                comparison_total=len(on_time),
                min_total=50,
            )
            assert result["population_orders"] == len(observations)
            assert result["eligible_orders"] == len(eligible)
            assert result["excluded_missing_pair_fields"] == len(observations) - len(eligible)
            assert result["exposed"] == recomputed["exposed"]
            assert result["comparison"] == recomputed["comparison"]
            assert result["risk_difference"] == recomputed["risk_difference"]
            assert result["risk_difference_ci_95"] == recomputed["risk_difference_ci_95"]
            assert result["chi_square"] == recomputed["chi_square"]
            assert result["status"] == recomputed["status"]
            assert "order_id" not in result
    finally:
        with engine.begin() as connection:
            connection.execute(text("RESET ROLE"))
            for table in (
                "analytics.fact_orders",
                "raw.order_items",
                "analytics.fact_order_seller_allocation",
                "analytics.fact_fulfillment",
                "raw.order_reviews",
            ):
                connection.execute(text(
                    f"DELETE FROM {table} WHERE order_id LIKE :prefix"
                ), {"prefix": f"{order_prefix}%"})
                assert connection.execute(text(
                    f"SELECT count(*) FROM {table} WHERE order_id LIKE :prefix"
                ), {"prefix": f"{order_prefix}%"}).scalar_one() == 0
            connection.execute(text("DELETE FROM iam.membership_seller_grants WHERE tenant_id=:tenant"), {"tenant": tenant_id})
            connection.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant"), {"tenant": tenant_id})
            connection.execute(text("DELETE FROM iam.tenants WHERE tenant_id=:tenant"), {"tenant": tenant_id})
            connection.execute(text("DELETE FROM iam.users WHERE user_id=:user"), {"user": user_id})


def test_verified_experiment_repository_executes_synthetic_rows_under_rls(monkeypatch) -> None:
    """Exercise the real SQL/read/ITT path without representing fixtures as real experiments."""
    from app.auth import context as auth_context
    from app.data.repositories.experiments import estimate_scoped_randomized_experiment

    suffix = uuid4().hex
    tenant_id = f"it_exp_t_{suffix}"
    seller_id = f"it_exp_s_{suffix}"
    user_id = f"it_exp_u_{suffix}"
    experiment_id = f"it_exp_{suffix}"
    source_batch_id = f"it_exp_batch_{suffix}"
    starts_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ends_at = datetime(2024, 1, 31, 23, 59, 59, tzinfo=timezone.utc)
    observed_at = datetime(2024, 1, 15, 12, tzinfo=timezone.utc)
    tenant = TenantContext(
        tenant_id=tenant_id,
        user_id=user_id,
        role="analyst",
        scope_type="seller",
        seller_id=seller_id,
        seller_grants=(SellerGrant(
            seller_id=seller_id, start_date="2024-01-01", end_date="2024-01-31",
        ),),
    )
    scope = ControlledAnalysisScope.model_validate({
        "seller_id": seller_id,
        "allowed_metrics": ["gmv"],
        "default_start_date": date(2024, 1, 1),
        "default_end_date": date(2024, 1, 31),
        "metric_name": "gmv",
        "granularity": "day",
        "max_rows": 100,
    })
    # This is an explicit test-only grant. Production remains default-deny
    # until a governed policy intentionally exposes the experiment action.
    monkeypatch.setattr(
        auth_context,
        "_policy_tools",
        lambda *_args: ("estimate_scoped_randomized_experiment",),
    )

    try:
        with engine.begin() as connection:
            connection.execute(text("RESET ROLE"))
            _seed_live_test_identity(
                connection,
                tenant_id=tenant_id,
                user_id=user_id,
                seller_id=seller_id,
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                create_tenant=True,
            )
            connection.execute(text("""
                INSERT INTO governance.experiment_registry (
                  tenant_id, experiment_id, seller_id, outcome_metric_id,
                  assignment_unit, assignment_method, status, starts_at, ends_at,
                  randomization_verified, verification_ref, verified_by,
                  verified_at, created_by
                ) VALUES (
                  :tenant_id, :experiment_id, :seller_id, 'gmv',
                  'user', 'randomized', 'closed', :starts_at, :ends_at,
                  TRUE, :verification_ref, 'test-governance-reviewer',
                  :verified_at, :created_by
                )
            """), {
                "tenant_id": tenant_id,
                "experiment_id": experiment_id,
                "seller_id": seller_id,
                "starts_at": starts_at,
                "ends_at": ends_at,
                "verification_ref": f"synthetic-test-only:{suffix}",
                "verified_at": ends_at,
                "created_by": user_id,
            })
            connection.execute(text("""
                INSERT INTO analytics.experiment_observations (
                  tenant_id, experiment_id, seller_id, randomization_unit_hash,
                  assigned_group, outcome_metric_id, outcome_value, observed_at,
                  source_batch_id
                ) VALUES (
                  :tenant_id, :experiment_id, :seller_id, :unit_hash,
                  :assigned_group, 'gmv', :outcome_value, :observed_at,
                  :source_batch_id
                )
            """), [{
                "tenant_id": tenant_id,
                "experiment_id": experiment_id,
                "seller_id": seller_id,
                "unit_hash": f"synthetic-unit-{index:03d}",
                "assigned_group": "treatment" if index < 40 else "control",
                "outcome_value": 2.0 if index < 40 else 1.0,
                "observed_at": observed_at,
                "source_batch_id": source_batch_id,
            } for index in range(80)])

        with activate_tenant_context(tenant):
            with engine.begin() as connection:
                assert connection.execute(text("SELECT current_user")).scalar_one() == "bizinsight_app"
                # Forced RLS exposes no rows until this transaction is scoped.
                assert connection.execute(text("""
                    SELECT count(*) FROM governance.experiment_registry
                    WHERE experiment_id=:experiment_id
                """), {"experiment_id": experiment_id}).scalar_one() == 0
                set_local_context(connection, tenant, scope_type="seller")
                assert connection.execute(text("""
                    SELECT count(*) FROM analytics.experiment_observations
                    WHERE experiment_id=:experiment_id
                """), {"experiment_id": experiment_id}).scalar_one() == 80
            result = estimate_scoped_randomized_experiment(
                scope, experiment_id=experiment_id,
            )

        assert result["status"] == "ok"
        assert result["sample_size"] == 80
        assert result["treatment_n"] == result["control_n"] == 40
        assert result["effect"] == pytest.approx(1.0)
        assert result["source_table"] == "governance.experiment_registry+analytics.experiment_observations"
        assert result["unit_identifiers_returned"] is False
        assert "synthetic-unit" not in repr(result)
    finally:
        with engine.begin() as connection:
            connection.execute(text("RESET ROLE"))
            connection.execute(text("""
                DELETE FROM governance.experiment_registry
                WHERE tenant_id=:tenant_id AND experiment_id=:experiment_id
            """), {"tenant_id": tenant_id, "experiment_id": experiment_id})
            connection.execute(text("DELETE FROM iam.membership_seller_grants WHERE tenant_id=:tenant"), {"tenant": tenant_id})
            connection.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant"), {"tenant": tenant_id})
            connection.execute(text("DELETE FROM iam.tenants WHERE tenant_id=:tenant"), {"tenant": tenant_id})
            connection.execute(text("DELETE FROM iam.users WHERE user_id=:user"), {"user": user_id})


def test_order_freight_distribution_matches_live_item_rows_under_rls(request) -> None:
    from app.data.repositories.explorer import query_scoped_order_freight_distribution

    with engine.connect() as connection:
        seller = connection.execute(text("""
            SELECT oi.seller_id, MIN(fo.purchase_date) AS start_date,
                   MAX(fo.purchase_date) AS end_date, COUNT(DISTINCT oi.order_id) AS n
            FROM analytics.v_delivered_order_items oi
            JOIN analytics.v_delivered_orders fo USING (order_id)
            GROUP BY oi.seller_id
            HAVING COUNT(DISTINCT oi.order_id) >= 20
            ORDER BY COUNT(DISTINCT oi.order_id) DESC
            LIMIT 1
        """)).mappings().one_or_none()
    if seller is None:
        pytest.skip("the live database has no seller with 20 delivered orders")

    seller_id = str(seller["seller_id"])
    start, end = seller["start_date"], seller["end_date"]
    tenant_id = _mapped_tenant_for_seller(seller_id)
    if tenant_id is None:
        pytest.skip("the observed live seller has no tenant mapping for an authorization-backed query")
    user_id = f"freight_live_{uuid4().hex}"
    with engine.begin() as connection:
        _seed_live_test_identity(
            connection,
            tenant_id=tenant_id,
            user_id=user_id,
            seller_id=seller_id,
            start_date=start,
            end_date=end,
        )
    request.addfinalizer(lambda: _cleanup_live_test_identity(tenant_id, user_id))
    tenant = TenantContext(
        tenant_id=tenant_id,
        user_id=user_id,
        role="analyst",
        scope_type="seller",
        seller_id=seller_id,
        seller_grants=(SellerGrant(seller_id=seller_id),),
    )
    scope = ControlledAnalysisScope.model_validate({
        "seller_id": seller_id,
        "allowed_metrics": ["seller_avg_freight_per_order"],
        "default_start_date": start,
        "default_end_date": end,
        "metric_name": "seller_avg_freight_per_order",
        "granularity": "day",
        "max_rows": 10_000,
    })

    with activate_tenant_context(tenant):
        with engine.begin() as connection:
            set_local_context(connection, tenant, scope_type="seller")
            item_rows = connection.execute(text("""
                SELECT oi.order_id, oi.freight_value
                FROM analytics.v_delivered_order_items oi
                JOIN analytics.v_delivered_orders fo USING (order_id)
                WHERE oi.seller_id=:seller_id
                  AND fo.purchase_date BETWEEN :start_date AND :end_date
                ORDER BY oi.order_id
            """), {
                "seller_id": seller_id, "start_date": start, "end_date": end,
            }).mappings().all()
        actual = query_scoped_order_freight_distribution(
            scope, start_date=start, end_date=end,
        )

    per_order: dict[str, list[float | None]] = {}
    for row in item_rows:
        per_order.setdefault(str(row["order_id"]), []).append(
            float(row["freight_value"]) if row["freight_value"] is not None else None
        )
    complete = [
        sum(value for value in values if value is not None)
        for values in per_order.values()
        if all(value is not None for value in values)
    ]
    complete.sort()

    def percentile_cont(values: list[float], p: float) -> float | None:
        if not values:
            return None
        position = p * (len(values) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    n = len(complete)
    mean = sum(complete) / n if n else None
    expected_statistics = {
        "mean": mean,
        "std": math.sqrt(sum((value - mean) ** 2 for value in complete) / n) if n else None,
        "min": complete[0] if n else None,
        "q1": percentile_cont(complete, 0.25),
        "median": percentile_cont(complete, 0.50),
        "q3": percentile_cont(complete, 0.75),
        "max": complete[-1] if n else None,
    }
    assert actual["sample_size"] == n
    assert actual["excluded_missing_freight_orders"] == len(per_order) - n
    assert actual["zero_freight_orders"] == sum(value == 0 for value in complete)
    for name, expected in expected_statistics.items():
        if expected is None:
            assert actual["statistics"][name] is None
        else:
            assert actual["statistics"][name] == pytest.approx(expected, abs=1e-8)


def test_chat_http_uses_real_rls_database_query_and_authorized_explorer_scope(monkeypatch) -> None:
    """Exercise scoped HTTP→Graph→MCP→PostgreSQL with a synthetic seller.

    Model-facing triage/review are deterministic doubles; Explorer scope
    reconstruction, metric authorization, MCP validation, repository SQL,
    answer assembly and trace capture remain real. This is not live-provider
    or browser acceptance.
    """
    from copy import deepcopy

    from fastapi.testclient import TestClient
    from langgraph.checkpoint.memory import MemorySaver

    from app.core.config import get_settings
    from app.graph import workflow
    from app.iam.service import hash_password
    from app.main import app
    import app.api.auth as api_auth
    import app.api.routes.auth as auth_routes

    suffix = uuid4().hex
    fixture_token = suffix[:10]
    order_prefix = f"chat_e2e_{fixture_token}_"
    seller_id = f"chat_e2e_s_{fixture_token}"
    # Keep the originally selected seller first in IAM's canonical grant sort;
    # this fixture tests revocation with an alternate grant without changing
    # the pre-existing selected-seller request setup.
    other_seller_id = f"zz_chat_e2e_other_s_{fixture_token}"
    tenant_id = f"chat-e2e-tenant-{suffix}"
    user_id = f"chat-e2e-user-{suffix}"
    other_user_id = f"chat-e2e-other-user-{suffix}"
    policy_id = f"chat-e2e-policy-{suffix}"
    api_credential_id = f"chat-e2e-credential-{suffix}"
    api_key = f"chat-e2e-api-key-{suffix}"
    chat_session_id = f"chat-e2e-session-{suffix}"
    trend_client_turn_id = f"chat-e2e-trend-{suffix}"
    distractor_client_turn_id = f"chat-e2e-distractor-{suffix}"
    followup_client_turn_id = f"chat-e2e-followup-{suffix}"
    association_client_turn_id = f"chat-e2e-association-{suffix}"
    email = f"{user_id}@example.test"
    other_email = f"{other_user_id}@example.test"
    password = f"test-only-{suffix}"
    other_password = f"other-test-only-{suffix}"
    auth_secret = f"test-signing-secret-{uuid4().hex}"
    test_settings = get_settings().model_copy(update={
        "tenant_auth_secret": auth_secret,
        "tenant_auth_required": True,
        "abac_database_enabled": True,
    })
    monkeypatch.setattr(api_auth, "get_settings", lambda: test_settings)
    monkeypatch.setattr(auth_routes, "get_settings", lambda: test_settings)
    monkeypatch.setattr("app.security.governance.get_settings", lambda: test_settings)
    monkeypatch.setattr("app.security.abac.get_settings", lambda: test_settings)
    password_hash = hash_password(password)
    other_password_hash = hash_password(other_password)
    start, end = date(2024, 1, 1), date(2024, 1, 31)
    purchased_at = datetime(2024, 1, 15, 12, 0)
    order_rows, item_rows, allocation_rows, fulfillment_rows, review_rows = [], [], [], [], []
    for index in range(60):
        order_id = f"{order_prefix}{index:03d}"
        freight = float(index % 7)
        price = float(20 + index)
        purchase_date = start + timedelta(days=14)
        order_rows.append({
            "order_id": order_id, "order_status": "delivered",
            "purchase_at": purchased_at, "purchase_date": purchase_date,
            "payment_value": price + freight, "item_count": 1,
            "item_gross_value": price, "freight_value": freight,
        })
        item_rows.append({
            "order_id": order_id, "order_item_id": 1,
            "product_id": f"chat_e2e_p_{fixture_token}", "seller_id": seller_id,
            "shipping_limit_date": purchased_at, "price": price,
            "freight_value": freight,
        })
        allocation_rows.append({"order_id": order_id, "seller_id": seller_id})
        fulfillment_rows.append({
            "order_id": order_id, "purchase_date": purchase_date,
            "order_status": "delivered", "is_on_time": index % 2 == 0,
        })
        if index % 2 == 0:
            review_score = 3 if index % 4 == 0 else 4
        elif index < 24:
            review_score = 1
        else:
            review_score = 3 if index % 4 == 1 else 4
        review_rows.append({
            "review_id": f"{order_id}_review", "order_id": order_id,
            "review_score": review_score,
        })

    traces: dict[str, dict] = {}

    def fake_triage(state):
        is_association_question = "关联" in str(state.get("user_message") or "")
        state["task_type"] = "anomaly_analysis" if is_association_question else "metric_query"
        state["next_agent"] = "data_query"
        state.setdefault("handoffs", []).append({
            "from": "triage", "to": "data_query",
            "reason": "live PostgreSQL association test" if is_association_question else "live PostgreSQL test",
        })
        state.setdefault("node_history", []).append({
            "node": "triage",
            "summary": "live PostgreSQL association test" if is_association_question else "live PostgreSQL scoped-query test",
        })
        return state

    def fake_critic(state):
        state["critic_verdict"] = "approved"
        state["critic_revision_target"] = None
        state.setdefault("critic_decisions", []).append({
            "llm_issues": [], "revision_target": None,
        })
        state.setdefault("node_history", []).append({
            "node": "critic", "summary": "fixture approved",
        })
        return state

    def fake_premise_check(state):
        state["premise_check"] = {"status": "aligned"}
        state.setdefault("node_history", []).append({
            "node": "premise_check", "summary": "no directional premise to correct",
        })
        return state

    def fake_diagnostic_plan(state):
        from app.agents.diagnostic_contracts import DiagnosticPlan
        from app.graph.nodes import diagnostic as diagnostic_node

        state["diagnostic_target_metric"] = "seller_order_bad_review_rate"
        diagnostic_node._ensure_graph_investigation_state(state)
        plan, _ = diagnostic_node._ensure_required_skeleton(
            DiagnosticPlan.model_validate({
                "goal": "检查订单差评率与准时状态的统计关联",
                "hypotheses": ["准时状态与差评率可能存在统计关联"],
                "steps": [{
                    "step_id": "target-trend", "objective": "确认当前目标指标趋势",
                    "kind": "trend", "required": True,
                    "allowed_actions": ["CALL_TOOL", "FINISH", "ABSTAIN"],
                    "completion_condition": "已查询目标趋势",
                }],
                "stop_conditions": ["取得注册关联证据或达到安全停止条件"],
                "budget": {"max_rounds": 4, "max_tool_calls": 4},
            }),
            supported_kinds=diagnostic_node._required_evidence_kinds_for(state),
        )
        state["diagnostic_plan"] = plan.model_dump(mode="json")
        state["diagnostic_step_state"] = {step.step_id: "pending" for step in plan.steps}
        state.setdefault("node_history", []).append({
            "node": "diagnostic_plan", "summary": "deterministic live association test plan",
        })
        return state

    def select_registered_association(prompt, _question):
        import json
        import re

        set_match = re.search(r"当前候选集 ID：([^。]+)。", prompt)
        assert set_match, prompt
        candidate_text = prompt.split("候选如下：\n", 1)[1].split("\nFINISH", 1)[0]
        candidates = json.loads(candidate_text)
        candidate = next(
            item for item in candidates
            if item.get("tool_name") == "query_scoped_registered_association"
        )
        return {
            "action_type": "CALL_CANDIDATE",
            "candidate_set_id": set_match.group(1),
            "candidate_id": candidate["candidate_id"],
        }

    def capture_trace(trace_id, payload):
        traces[str(trace_id)] = deepcopy(payload)

    try:
        with engine.begin() as connection:
            connection.execute(text("RESET ROLE"))
            connection.execute(text("""
                INSERT INTO iam.tenants (tenant_id, tenant_name, scope_type)
                VALUES (:tenant_id, :tenant_name, 'seller')
            """), {"tenant_id": tenant_id, "tenant_name": f"HTTP test {fixture_token}"})
            connection.execute(text("""
                INSERT INTO iam.users (user_id, display_name, email, password_hash)
                VALUES
                  (:user_id, :display_name, :email, :password_hash),
                  (:other_user_id, :other_display_name, :other_email, :other_password_hash)
            """), {
                "user_id": user_id, "display_name": f"HTTP test {fixture_token}",
                "email": email, "password_hash": password_hash,
                "other_user_id": other_user_id,
                "other_display_name": f"HTTP other user test {fixture_token}",
                "other_email": other_email,
                "other_password_hash": other_password_hash,
            })
            connection.execute(text("""
                INSERT INTO iam.tenant_memberships (tenant_id, user_id, role)
                VALUES
                  (:tenant_id, :user_id, 'admin'),
                  (:tenant_id, :other_user_id, 'admin')
            """), {
                "tenant_id": tenant_id, "user_id": user_id,
                "other_user_id": other_user_id,
            })
            connection.execute(text("""
                INSERT INTO iam.membership_seller_grants (
                  tenant_id, user_id, seller_id, access_start_date, access_end_date
                ) VALUES
                  (:tenant_id, :user_id, :seller_id, :start_date, :end_date),
                  (:tenant_id, :user_id, :other_seller_id, :start_date, :end_date),
                  (:tenant_id, :other_user_id, :seller_id, :start_date, :end_date)
            """), {
                "tenant_id": tenant_id, "user_id": user_id, "seller_id": seller_id,
                "other_seller_id": other_seller_id,
                "other_user_id": other_user_id,
                "start_date": start, "end_date": end,
            })
            connection.execute(text("""
                INSERT INTO iam.api_credentials (
                  credential_id, tenant_id, user_id, key_hash
                ) VALUES (:credential_id, :tenant_id, :user_id, :key_hash)
            """), {
                "credential_id": api_credential_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "key_hash": hashlib.sha256(api_key.encode("utf-8")).hexdigest(),
            })
            connection.execute(text("""
                INSERT INTO governance.abac_policy_rules (
                  tenant_id, policy_id, policy_version, status, rule_json,
                  created_by, updated_by
                ) VALUES (
                  :tenant_id, :policy_id, 1, 'active', CAST(:rule_json AS jsonb),
                  :user_id, :user_id
                )
            """), {
                "tenant_id": tenant_id,
                "policy_id": policy_id,
                "user_id": user_id,
                "rule_json": json.dumps({
                    "role": "admin",
                    "agent": "data_query",
                    "allow": [
                        "query_metric_trend",
                        "query_scoped_registered_association",
                    ],
                    "deny": [],
                    "conditions": {"scope_types": ["seller"]},
                }),
            })
            connection.execute(text("""
                INSERT INTO analytics.fact_orders (
                  order_id, order_status, order_purchase_timestamp, purchase_date,
                  payment_value, item_count, item_gross_value, freight_value
                ) VALUES (
                  :order_id, :order_status, :purchase_at, :purchase_date,
                  :payment_value, :item_count, :item_gross_value, :freight_value
                )
            """), order_rows)
            connection.execute(text("""
                INSERT INTO raw.order_items (
                  order_id, order_item_id, product_id, seller_id,
                  shipping_limit_date, price, freight_value
                ) VALUES (
                  :order_id, :order_item_id, :product_id, :seller_id,
                  :shipping_limit_date, :price, :freight_value
                )
            """), item_rows)
            connection.execute(text("""
                INSERT INTO analytics.fact_order_items (
                  order_id, order_item_id, seller_id, product_id, purchase_date,
                  order_status, price, freight_value, shipping_limit_date
                ) VALUES (
                  :order_id, :order_item_id, :seller_id, :product_id, :purchase_date,
                  'delivered', :price, :freight_value, :shipping_limit_date
                )
            """), [
                {**item, "purchase_date": start + timedelta(days=14)}
                for item in item_rows
            ])
            connection.execute(text("""
                INSERT INTO analytics.fact_order_seller_allocation (
                  order_id, seller_id, seller_gross_freight_value,
                  order_gross_freight_value, seller_allocation_ratio,
                  seller_count, is_single_seller_order
                ) VALUES (:order_id, :seller_id, 0, 0, 1, 1, TRUE)
            """), allocation_rows)
            connection.execute(text("""
                INSERT INTO analytics.fact_fulfillment (
                  order_id, purchase_date, order_status, is_on_time
                ) VALUES (:order_id, :purchase_date, :order_status, :is_on_time)
            """), fulfillment_rows)
            connection.execute(text("""
                INSERT INTO raw.order_reviews (review_id, order_id, review_score)
                VALUES (:review_id, :order_id, :review_score)
            """), review_rows)

        with engine.connect() as connection:
            visible_fixture_rows = connection.execute(text("""
                SELECT COUNT(*)
                FROM analytics.v_delivered_order_items oi
                JOIN analytics.v_delivered_orders fo USING (order_id)
                JOIN raw.order_reviews r USING (order_id)
                WHERE oi.seller_id=:seller_id
                  AND fo.purchase_date BETWEEN :start_date AND :end_date
            """), {
                "seller_id": seller_id, "start_date": start, "end_date": end,
            }).scalar_one()
            assert visible_fixture_rows == 60, visible_fixture_rows

        monkeypatch.setattr(workflow, "get_checkpointer", lambda: MemorySaver())
        monkeypatch.setattr(workflow, "triage", fake_triage)
        monkeypatch.setattr(workflow, "critic", fake_critic)
        monkeypatch.setattr(workflow, "persist_memory_p0", lambda state: state)
        monkeypatch.setattr(workflow, "write_trace", capture_trace)
        monkeypatch.setattr(
            "app.api.routes.chat.relation",
            lambda *_args, **_kwargs: {"action": "none"},
        )

        with TestClient(app) as client:
            login_response = client.post("/auth/login", json={
                "email": email, "password": password,
            })
            assert login_response.status_code == 200, login_response.text
            login_body = login_response.json()
            session_id = api_auth._decode_hs256_jwt(
                login_body["access_token"], auth_secret,
            )["sid"]
            assert login_body["profile"]["tenant_id"] == tenant_id
            assert login_body["profile"]["user_id"] == user_id
            assert {
                grant["seller_id"] for grant in login_body["profile"]["seller_grants"]
            } == {seller_id, other_seller_id}
            headers = {"Authorization": f"Bearer {login_body['access_token']}"}
            me_response = client.get("/auth/me", headers=headers)
            assert me_response.status_code == 200, me_response.text
            assert me_response.json()["tenant_id"] == tenant_id

            # Match the frontend flow: ExplorerView first posts the selected
            # seller/date/metric scope, stores the server-validated preview
            # context, then ScopedChatPanel submits that context to chat.
            explorer_query = {
                "seller_id": seller_id,
                "metric_name": "review_score_avg",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "granularity": "month",
            }
            preview_response = client.post(
                "/data-explorer/preview", headers=headers, json=explorer_query,
            )
            assert preview_response.status_code == 200, preview_response.text
            preview_body = preview_response.json()
            assert preview_body["summary"]["status"] == "ok", preview_body
            assert preview_body["summary"]["value"] == pytest.approx(3.0)
            assert preview_body["context"]["locked"] is True
            frontend_scope = {
                key: preview_body["context"][key]
                for key in ("seller_id", "metric_name", "start_date", "end_date", "granularity")
            }
            response = client.post("/chat/stream", headers=headers, json={
                "message": "请按月汇总本店2024年1月的平均评分，并指出最高和最低月份。",
                "memory_enabled": False,
                "chat_session_id": chat_session_id,
                "chat_turn_id": trend_client_turn_id,
                "use_current_explorer_context": True,
                "explorer_context": frontend_scope,
            })
            assert response.status_code == 200, response.text
            stream_events = [
                json.loads(line[len("data: "):])
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            body = next(
                (event for event in reversed(stream_events) if event.get("event") in {"completed", "failed"}),
                {},
            )
            assert body.get("event") == "completed", stream_events

            additional_metric_turn_ids = []
            for metric_name, metric_label, expected_value in (
                ("gmv", "GMV", 3144.0),
                ("order_count", "订单量", 60.0),
                ("freight_value", "运费", 174.0),
            ):
                metric_query = {
                    **explorer_query,
                    "metric_name": metric_name,
                }
                metric_preview_response = client.post(
                    "/data-explorer/preview", headers=headers, json=metric_query,
                )
                assert metric_preview_response.status_code == 200, metric_preview_response.text
                metric_preview = metric_preview_response.json()
                assert metric_preview["summary"]["status"] == "ok", metric_preview
                assert metric_preview["summary"]["value"] == pytest.approx(expected_value)
                assert metric_preview["context"]["locked"] is True
                metric_scope = {
                    key: metric_preview["context"][key]
                    for key in ("seller_id", "metric_name", "start_date", "end_date", "granularity")
                }
                metric_turn_id = f"chat-e2e-{metric_name}-{suffix}"
                additional_metric_turn_ids.append(metric_turn_id)
                metric_response = client.post("/chat/stream", headers=headers, json={
                    "message": f"请查询当前锁定范围内 {metric_label} 的月度趋势。",
                    "memory_enabled": False,
                    "chat_session_id": chat_session_id,
                    "chat_turn_id": metric_turn_id,
                    "use_current_explorer_context": True,
                    "explorer_context": metric_scope,
                })
                assert metric_response.status_code == 200, metric_response.text
                metric_events = [
                    json.loads(line[len("data: "):])
                    for line in metric_response.text.splitlines()
                    if line.startswith("data: ")
                ]
                metric_body = next(
                    (event for event in reversed(metric_events) if event.get("event") in {"completed", "failed"}),
                    {},
                )
                assert metric_body.get("event") == "completed", metric_events
                assert metric_body.get("error") is False, metric_body
                metric_trace = traces[metric_body["trace_id"]]
                assert metric_trace["controlled_analysis_scope"]["seller_id"] == seller_id
                selected_metric = next(
                    item for item in metric_trace["authorized_metrics"]
                    if item["metric_id"] == metric_name
                )
                assert selected_metric["provenance_type"] == "explorer_selection"
                metric_result = metric_trace["tool_result"]["result"]["trend_result"]
                assert metric_result["metric_name"] == metric_name
                assert metric_result["granularity"] == "month"
                assert len(metric_result["points"]) == 1
                assert metric_result["points"][0]["value"] == pytest.approx(expected_value)

            # A follow-up must inherit the explicitly named completed Turn,
            # not whichever same-scope result happens to be most recent.
            identity_context = TenantContext(
                tenant_id=tenant_id, user_id=user_id, role="admin",
                scope_type="seller",
                seller_grants=(
                    SellerGrant(seller_id=seller_id),
                    SellerGrant(seller_id=other_seller_id),
                ),
            )
            # Changing the IAM role changes the effective permission version.
            # The same still-valid login session must reload that role and
            # must not replay a result created under the former role.
            with engine.begin() as connection:
                set_local_context(connection, identity_context, scope_type="seller")
                changed_role = connection.execute(text("""
                    UPDATE iam.tenant_memberships SET role='reviewer'
                    WHERE tenant_id=:tenant_id AND user_id=:user_id
                      AND revoked_at IS NULL
                """), {"tenant_id": tenant_id, "user_id": user_id})
                assert changed_role.rowcount == 1, "test IAM role update must affect one membership"
            try:
                changed_role_profile = client.get("/auth/me", headers=headers)
                assert changed_role_profile.status_code == 200, changed_role_profile.text
                assert changed_role_profile.json()["role"] == "reviewer"
                api_key_profile = client.get(
                    "/auth/me", headers={"X-API-Key": api_key},
                )
                assert api_key_profile.status_code == 200, api_key_profile.text
                assert api_key_profile.json()["role"] == "reviewer"
                traces_before_role_stale_replay = set(traces)
                role_stale_replay = client.post("/chat", headers=headers, json={
                    "message": "请按月汇总本店2024年1月的平均评分，并指出最高和最低月份。",
                    "memory_enabled": False,
                    "chat_session_id": chat_session_id,
                    "chat_turn_id": trend_client_turn_id,
                    "use_current_explorer_context": True,
                    "explorer_context": {
                        "seller_id": seller_id,
                        "metric_name": "review_score_avg",
                        "start_date": start.isoformat(),
                        "end_date": end.isoformat(),
                        "granularity": "month",
                    },
                })
                assert role_stale_replay.status_code == 400, role_stale_replay.text
                assert "stale authorization or scope" in role_stale_replay.text
                assert set(traces) == traces_before_role_stale_replay

                # Simulate an in-flight request which authenticated as admin
                # before the role change but reaches the MCP boundary after
                # the database now says reviewer. MCP must reload the session
                # and reject the already-issued admin AuthorizedMetric.
                from app.mcp.client import query_scoped_business_data
                from app.mcp.schemas import MetricQueryPlan

                original_trace = traces[body["trace_id"]]
                old_authorized_metric = next(
                    item for item in original_trace["authorized_metrics"]
                    if item["metric_id"] == "review_score_avg"
                )
                old_trend_plan = MetricQueryPlan.model_validate({
                    "query_type": "metric_trend",
                    "metric_name": "review_score_avg",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "granularity": "month",
                    "authorized_metric": old_authorized_metric,
                })
                stale_request_context = TenantContext(
                    tenant_id=tenant_id, user_id=user_id, role="admin",
                    scope_type="seller",
                    seller_grants=(
                        SellerGrant(seller_id=seller_id),
                        SellerGrant(seller_id=other_seller_id),
                    ),
                    auth_session_id=str(session_id),
                )
                with activate_tenant_context(stale_request_context):
                    role_revoked_mcp = query_scoped_business_data(
                        "验证进入 MCP 前的 IAM 角色撤权。",
                        original_trace["controlled_analysis_scope"],
                        plan=old_trend_plan,
                        request_id=f"role-revoked-mcp-{suffix}",
                    )
                assert role_revoked_mcp["tool_name"] is None, role_revoked_mcp
                assert role_revoked_mcp["result"]["status"] == "unsupported"
                assert "scope fingerprint is stale" in role_revoked_mcp["result"]["reason"]

                stale_api_key_context = TenantContext(
                    tenant_id=tenant_id, user_id=user_id, role="admin",
                    scope_type="seller",
                    seller_grants=(
                        SellerGrant(seller_id=seller_id),
                        SellerGrant(seller_id=other_seller_id),
                    ),
                    auth_credential_id=api_credential_id,
                )
                with activate_tenant_context(stale_api_key_context):
                    api_role_revoked_mcp = query_scoped_business_data(
                        "验证 API Key 进入 MCP 前的 IAM 角色撤权。",
                        original_trace["controlled_analysis_scope"],
                        plan=old_trend_plan,
                        request_id=f"api-role-revoked-mcp-{suffix}",
                    )
                assert api_role_revoked_mcp["tool_name"] is None, api_role_revoked_mcp
                assert api_role_revoked_mcp["result"]["status"] == "unsupported"
                assert "scope fingerprint is stale" in api_role_revoked_mcp["result"]["reason"]
            finally:
                with engine.begin() as connection:
                    set_local_context(connection, identity_context, scope_type="seller")
                    restored_role = connection.execute(text("""
                        UPDATE iam.tenant_memberships SET role='admin'
                        WHERE tenant_id=:tenant_id AND user_id=:user_id
                          AND revoked_at IS NULL
                    """), {"tenant_id": tenant_id, "user_id": user_id})
                    assert restored_role.rowcount == 1, "test IAM role restoration must affect one membership"
            restored_profile = client.get("/auth/me", headers=headers)
            assert restored_profile.status_code == 200, restored_profile.text
            assert restored_profile.json()["role"] == "admin"

            with engine.begin() as connection:
                set_local_context(connection, identity_context, scope_type="seller")
                parent_turn = connection.execute(text("""
                    SELECT turn_id, status, scope_fingerprint, authz_digest
                    FROM governance.chat_turns
                    WHERE tenant_id=:tenant_id AND user_id=:user_id
                      AND chat_session_id=:session_id
                      AND client_turn_id=:client_turn_id
                """), {
                    "tenant_id": tenant_id, "user_id": user_id,
                    "session_id": chat_session_id,
                    "client_turn_id": trend_client_turn_id,
            }).mappings().one()
            assert parent_turn["status"] == "completed"

            distractor_response = client.post("/chat", headers=headers, json={
                "message": "查询本店的平均评分。",
                "memory_enabled": False,
                "chat_session_id": chat_session_id,
                "chat_turn_id": distractor_client_turn_id,
                "use_current_explorer_context": True,
                "explorer_context": {
                    "seller_id": seller_id,
                    "metric_name": "review_score_avg",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "granularity": "month",
                },
            })
            assert distractor_response.status_code == 200, distractor_response.text
            assert distractor_response.json()["error"] is False

            real_data_query = workflow.data_query
            inherited_contexts = []

            def capture_inherited_context(state):
                if state.get("user_message") == "为什么？":
                    inherited_contexts.append(deepcopy(state.get("previous_turn_context") or {}))
                return real_data_query(state)

            monkeypatch.setattr(workflow, "data_query", capture_inherited_context)
            followup_response = client.post("/chat", headers=headers, json={
                "message": "为什么？",
                "memory_enabled": False,
                "chat_session_id": chat_session_id,
                "chat_turn_id": followup_client_turn_id,
                "inherited_from_turn_id": parent_turn["turn_id"],
                "use_current_explorer_context": True,
                "explorer_context": {
                    "seller_id": seller_id,
                    "metric_name": "review_score_avg",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "granularity": "month",
                },
            })
            assert followup_response.status_code == 200, followup_response.text
            followup_body = followup_response.json()
            assert followup_body["error"] is False, followup_body
            assert inherited_contexts
            assert inherited_contexts[0]["source_turn_id"] == parent_turn["turn_id"]
            assert inherited_contexts[0]["authz_digest"] == parent_turn["authz_digest"]
            assert any(
                item["metric_id"] == "review_score_avg"
                for item in inherited_contexts[0]["authorized_metrics"]
            )
            assert "上一轮服务端授权指标" in followup_body["answer"] or "review_score_avg" in str(
                traces.get(followup_body["trace_id"], {}).get("metric_planning_decision", {})
            )

            trace_ids_before_stale_parent = set(traces)
            stale_parent_response = client.post("/chat", headers=headers, json={
                "message": "为什么？",
                "memory_enabled": False,
                "chat_session_id": chat_session_id,
                "chat_turn_id": f"chat-e2e-stale-parent-{suffix}",
                "inherited_from_turn_id": parent_turn["turn_id"],
                "use_current_explorer_context": True,
                "explorer_context": {
                    "seller_id": seller_id,
                    "metric_name": "review_score_avg",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "granularity": "day",
                },
            })
            assert stale_parent_response.status_code == 400, stale_parent_response.text
            assert "inherited turn has stale authorization or scope" in stale_parent_response.text
            assert set(traces) == trace_ids_before_stale_parent

            # A Registry metric-definition version change is part of the
            # canonical query-scope fingerprint. The predecessor must be
            # rejected at lifecycle creation, before the graph can reuse its
            # AuthorizedMetric or run any tool.
            from dataclasses import replace as dataclass_replace
            import app.metrics.registry as metric_registry_module

            original_get_registry = metric_registry_module.get_registry
            base_registry = original_get_registry()

            class _BumpedMetricRegistry:
                def __getattr__(self, name):
                    return getattr(base_registry, name)

                @property
                def catalog_digest(self):
                    return f"{base_registry.catalog_digest}:test-definition-change"

                def get(self, metric_id):
                    definition = base_registry.get(metric_id)
                    if metric_id == "review_score_avg":
                        return dataclass_replace(
                            definition, version=f"{definition.version}-test-change",
                        )
                    return definition

            monkeypatch.setattr(
                metric_registry_module, "get_registry", lambda: _BumpedMetricRegistry(),
            )
            changed_definition_parent = client.post("/chat", headers=headers, json={
                "message": "继续解释上一轮结果。",
                "memory_enabled": False,
                "chat_session_id": chat_session_id,
                "chat_turn_id": f"chat-e2e-metric-definition-change-{suffix}",
                "inherited_from_turn_id": parent_turn["turn_id"],
                "use_current_explorer_context": True,
                "explorer_context": {
                    "seller_id": seller_id,
                    "metric_name": "review_score_avg",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "granularity": "month",
                },
            })
            monkeypatch.setattr(
                metric_registry_module, "get_registry", original_get_registry,
            )
            assert changed_definition_parent.status_code == 400, changed_definition_parent.text
            assert "inherited turn has stale authorization or scope" in changed_definition_parent.text
            assert set(traces) == trace_ids_before_stale_parent

            cross_session_parent = client.post("/chat", headers=headers, json={
                "message": "继续分析上一轮。",
                "memory_enabled": False,
                "chat_session_id": f"{chat_session_id}-other",
                "chat_turn_id": f"chat-e2e-cross-session-{suffix}",
                "inherited_from_turn_id": parent_turn["turn_id"],
                "use_current_explorer_context": True,
                "explorer_context": {
                    "seller_id": seller_id,
                    "metric_name": "review_score_avg",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "granularity": "month",
                },
            })
            assert cross_session_parent.status_code == 400, cross_session_parent.text
            assert "does not belong to the current user and session" in cross_session_parent.text
            assert set(traces) == trace_ids_before_stale_parent

            # Another authenticated user in the same tenant may use the same
            # session identifier and have access to the same seller, but must
            # not inherit this user's private completed turn.
            other_login = client.post("/auth/login", json={
                "email": other_email, "password": other_password,
            })
            assert other_login.status_code == 200, other_login.text
            other_headers = {
                "Authorization": f"Bearer {other_login.json()['access_token']}"
            }
            cross_user_parent = client.post("/chat", headers=other_headers, json={
                "message": "继续分析上一轮。",
                "memory_enabled": False,
                "chat_session_id": chat_session_id,
                "chat_turn_id": f"chat-e2e-cross-user-{suffix}",
                "inherited_from_turn_id": parent_turn["turn_id"],
                "use_current_explorer_context": True,
                "explorer_context": {
                    "seller_id": seller_id,
                    "metric_name": "review_score_avg",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "granularity": "month",
                },
            })
            assert cross_user_parent.status_code == 400, cross_user_parent.text
            assert "does not belong to the current user and session" in cross_user_parent.text
            assert set(traces) == trace_ids_before_stale_parent
            assert client.post("/auth/logout", headers=other_headers).status_code == 200

            # Restore the regular graph node before the independent association
            # turn; the wrapper above only observes the explicit follow-up.
            monkeypatch.setattr(workflow, "data_query", real_data_query)

            from app.graph.nodes import diagnostic as diagnostic_node
            monkeypatch.setattr(workflow, "premise_check", fake_premise_check)
            monkeypatch.setattr(workflow, "diagnostic_plan", fake_diagnostic_plan)
            monkeypatch.setattr(
                "app.llm.client.llm_agent_action", select_registered_association,
            )
            association_response = client.post("/chat", headers=headers, json={
                "message": "分析店铺差评率与准时状态的统计关联，只报告统计关联，不作因果推断。",
                "memory_enabled": False,
                "chat_session_id": chat_session_id,
                "chat_turn_id": association_client_turn_id,
                "use_current_explorer_context": True,
                "explorer_context": {
                    "seller_id": seller_id,
                    "metric_name": "seller_order_bad_review_rate",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "granularity": "month",
                },
            })
            assert association_response.status_code == 200, association_response.text
            association_body = association_response.json()
            association_trace = traces[association_body["trace_id"]]

            # Replaying the same client turn must return its stored terminal
            # result, without re-running the graph or creating another turn.
            association_replay = client.post("/chat", headers=headers, json={
                "message": "分析店铺差评率与准时状态的统计关联，只报告统计关联，不作因果推断。",
                "memory_enabled": False,
                "chat_session_id": chat_session_id,
                "chat_turn_id": association_client_turn_id,
                "use_current_explorer_context": True,
                "explorer_context": {
                    "seller_id": seller_id,
                    "metric_name": "seller_order_bad_review_rate",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "granularity": "month",
                },
            })
            assert association_replay.status_code == 200, association_replay.text
            assert association_replay.json()["trace_id"] == association_body["trace_id"]
            assert association_replay.json()["answer"] == association_body["answer"]

            # The same client-turn key cannot replay a result under a changed
            # scope. This must be rejected before a second graph invocation.
            stale_scope_replay = client.post("/chat", headers=headers, json={
                "message": "分析店铺差评率与准时状态的统计关联，只报告统计关联，不作因果推断。",
                "memory_enabled": False,
                "chat_session_id": chat_session_id,
                "chat_turn_id": association_client_turn_id,
                "use_current_explorer_context": True,
                "explorer_context": {
                    "seller_id": seller_id,
                    "metric_name": "seller_order_bad_review_rate",
                    "start_date": start.isoformat(),
                    "end_date": (end - timedelta(days=1)).isoformat(),
                    "granularity": "month",
                },
            })
            assert stale_scope_replay.status_code == 400, stale_scope_replay.text
            assert "stale authorization or scope" in stale_scope_replay.text

            # Changing the tenant tool policy invalidates a previously
            # completed turn's authorization snapshot: an identical client
            # retry must not replay an answer produced under the old policy.
            with engine.begin() as connection:
                set_local_context(connection, identity_context, scope_type="seller")
                updated_policy = connection.execute(text("""
                    UPDATE governance.abac_policy_rules
                    SET policy_version=policy_version+1,
                        rule_json=CAST(:rule_json AS jsonb), updated_at=now()
                    WHERE tenant_id=:tenant_id AND policy_id=:policy_id
                      AND status='active'
                """), {
                    "tenant_id": tenant_id, "policy_id": policy_id,
                    "rule_json": json.dumps({
                        "role": "admin",
                        "agent": "data_query",
                        "allow": ["query_metric_trend"],
                        "deny": [],
                        "conditions": {"scope_types": ["seller"]},
                    }),
                })
                assert updated_policy.rowcount == 1, "test policy update must pass tenant RLS"
            traces_before_policy_stale_replay = set(traces)
            policy_stale_replay = client.post("/chat", headers=headers, json={
                "message": "分析店铺差评率与准时状态的统计关联，只报告统计关联，不作因果推断。",
                "memory_enabled": False,
                "chat_session_id": chat_session_id,
                "chat_turn_id": association_client_turn_id,
                "use_current_explorer_context": True,
                "explorer_context": {
                    "seller_id": seller_id,
                    "metric_name": "seller_order_bad_review_rate",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "granularity": "month",
                },
            })
            assert policy_stale_replay.status_code == 400, policy_stale_replay.text
            assert "stale authorization or scope" in policy_stale_replay.text
            assert set(traces) == traces_before_policy_stale_replay

            # Re-submit the already planned association QueryPlan through the
            # real scoped MCP client after revocation. The plan carries the old
            # authz fingerprint and must be denied before repository SQL runs.
            from app.mcp.client import query_scoped_business_data
            from app.mcp.schemas import MetricQueryPlan

            old_association_authorization = next(
                item for item in association_trace["authorized_metrics"]
                if item["metric_id"] == "seller_order_bad_review_rate"
            )
            old_association_plan = MetricQueryPlan.model_validate({
                "query_type": "metric_association",
                "metric_name": "seller_order_bad_review_rate",
                "association_pair_id": "delivery_timing_bad_review_v1",
                "authorized_metric": old_association_authorization,
            })
            with activate_tenant_context(identity_context):
                stale_mcp_result = query_scoped_business_data(
                    "执行旧授权下的关联查询应被拒绝。",
                    association_trace["controlled_analysis_scope"],
                    plan=old_association_plan,
                    request_id=f"stale-authz-{suffix}",
                )
            assert stale_mcp_result["tool_name"] is None, stale_mcp_result
            assert stale_mcp_result["result"]["status"] == "unsupported"
            assert "scope fingerprint is stale" in stale_mcp_result["result"]["reason"]

            # Revoke only the selected seller grant while leaving the login
            # session and a second seller grant active. The same JWT must be
            # re-evaluated and denied before creating a Turn or invoking Graph.
            with engine.begin() as connection:
                connection.execute(text("""
                    UPDATE iam.membership_seller_grants
                    SET revoked_at=now()
                    WHERE tenant_id=:tenant_id AND user_id=:user_id
                      AND seller_id=:seller_id AND revoked_at IS NULL
                """), {
                    "tenant_id": tenant_id, "user_id": user_id,
                    "seller_id": seller_id,
                })
            active_session_profile = client.get("/auth/me", headers=headers)
            assert active_session_profile.status_code == 200, active_session_profile.text
            assert {
                grant["seller_id"]
                for grant in active_session_profile.json()["seller_grants"]
            } == {other_seller_id}

            # A plan authorized before seller-grant revocation must also fail
            # at the MCP boundary when evaluated under the reloaded, reduced
            # authorization context—not merely at the HTTP scope preview.
            revoked_context = TenantContext(
                tenant_id=tenant_id, user_id=user_id, role="admin",
                scope_type="seller",
                seller_grants=(SellerGrant(
                    other_seller_id, start.isoformat(), end.isoformat(),
                ),),
            )
            with activate_tenant_context(revoked_context):
                revoked_mcp_result = query_scoped_business_data(
                    "使用已撤销店铺授权的旧计划应被拒绝。",
                    association_trace["controlled_analysis_scope"],
                    plan=old_association_plan,
                    request_id=f"revoked-seller-stale-plan-{suffix}",
                )
            assert revoked_mcp_result["tool_name"] is None, revoked_mcp_result
            assert revoked_mcp_result["result"]["status"] == "unsupported"
            assert "Seller scope is not authorized" in revoked_mcp_result["result"]["reason"]

            traces_before_revoked_scope = set(traces)
            revoked_seller_chat = client.post("/chat", headers=headers, json={
                "message": "查询刚刚被撤销授权的店铺，必须拒绝。",
                "memory_enabled": False,
                "chat_session_id": chat_session_id,
                "chat_turn_id": f"chat-e2e-revoked-seller-{suffix}",
                "use_current_explorer_context": True,
                "explorer_context": {
                    "seller_id": seller_id,
                    "metric_name": "review_score_avg",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "granularity": "month",
                },
            })
            assert revoked_seller_chat.status_code == 400, revoked_seller_chat.text
            assert set(traces) == traces_before_revoked_scope

            logout_response = client.post("/auth/logout", headers=headers)
            assert logout_response.status_code == 200, logout_response.text
            revoked_session_response = client.get("/auth/me", headers=headers)
            assert revoked_session_response.status_code == 401
            with activate_tenant_context(stale_request_context):
                revoked_session_mcp = query_scoped_business_data(
                    "验证进入 MCP 前的会话撤销。",
                    original_trace["controlled_analysis_scope"],
                    plan=old_trend_plan,
                    request_id=f"revoked-session-mcp-{suffix}",
                )
            assert revoked_session_mcp["tool_name"] is None, revoked_session_mcp
            assert revoked_session_mcp["result"]["status"] == "unsupported"
            assert "revoked, expired" in revoked_session_mcp["result"]["reason"]
            with engine.begin() as connection:
                connection.execute(text("""
                    UPDATE iam.api_credentials SET revoked_at=now()
                    WHERE credential_id=:credential_id AND revoked_at IS NULL
                """), {"credential_id": api_credential_id})
            with activate_tenant_context(stale_api_key_context):
                revoked_api_key_mcp = query_scoped_business_data(
                    "验证 API Key 撤销后的 MCP 拒绝。",
                    original_trace["controlled_analysis_scope"],
                    plan=old_trend_plan,
                    request_id=f"revoked-api-key-mcp-{suffix}",
                )
            assert revoked_api_key_mcp["tool_name"] is None, revoked_api_key_mcp
            assert revoked_api_key_mcp["result"]["status"] == "unsupported"
            assert "revoked, expired" in revoked_api_key_mcp["result"]["reason"]
            revoked_chat_response = client.post("/chat", headers=headers, json={
                "message": "再次查询，不应通过已撤销的登录会话。",
                "memory_enabled": False,
                "chat_session_id": chat_session_id,
                "chat_turn_id": f"chat-e2e-revoked-{suffix}",
                "use_current_explorer_context": True,
                "explorer_context": {
                    "seller_id": seller_id,
                    "metric_name": "review_score_avg",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "granularity": "month",
                },
            })
            assert revoked_chat_response.status_code == 401

        lifecycle_tenant = TenantContext(
            tenant_id=tenant_id, user_id=user_id, role="admin",
            scope_type="seller", seller_id=seller_id,
            seller_grants=(SellerGrant(seller_id=seller_id),),
        )
        with engine.begin() as connection:
            set_local_context(connection, lifecycle_tenant, scope_type="seller")
            lifecycle_rows = connection.execute(text("""
                SELECT t.client_turn_id, t.status AS turn_status,
                       t.final_trace_id, e.status AS execution_status,
                       e.trace_id AS execution_trace_id
                FROM governance.chat_turns t
                JOIN governance.conversation_executions e ON e.turn_id=t.turn_id
                WHERE t.tenant_id=:tenant_id
                  AND t.user_id=:user_id
                  AND t.chat_session_id=:session_id
                ORDER BY t.client_turn_id
            """), {
                "tenant_id": tenant_id, "user_id": user_id,
                "session_id": chat_session_id,
            }).mappings().all()
        lifecycle_by_client_id = {row["client_turn_id"]: row for row in lifecycle_rows}
        assert set(lifecycle_by_client_id) == {
            trend_client_turn_id, distractor_client_turn_id,
            followup_client_turn_id,
            association_client_turn_id,
            *additional_metric_turn_ids,
        }
        assert all(row["turn_status"] == "completed" for row in lifecycle_rows), {
            "rows": lifecycle_rows,
            "association_errors": association_body.get("errors"),
            "trace_errors": traces.get(association_body.get("trace_id"), {}).get("errors"),
            "trace_analysis": traces.get(association_body.get("trace_id"), {}).get("analysis_result"),
        }
        assert all(row["execution_status"] == "completed" for row in lifecycle_rows), lifecycle_rows
        assert all(row["final_trace_id"] == row["execution_trace_id"] for row in lifecycle_rows)
        assert len(traces) == 7, "terminal retry/stale-scope replay must not re-run the graph"
        assert body["error"] is False, {
            "errors": body.get("errors"),
            "trace_errors": traces.get(body.get("trace_id"), {}).get("errors"),
            "tool_result": traces.get(body.get("trace_id"), {}).get("tool_result"),
            "authorized_metrics": traces.get(body.get("trace_id"), {}).get("authorized_metrics"),
        }
        assert body["trace_id"] in traces

        trace = traces[body["trace_id"]]
        assert trace["controlled_analysis_scope"]["seller_id"] == seller_id
        assert trace["metric_planning_decision"]["status"] == "confirmed"
        authorized = next(
            item for item in trace["authorized_metrics"]
            if item["metric_id"] == "review_score_avg"
        )
        assert authorized["provenance_type"] == "explorer_selection"
        tool_result = trace["tool_result"]
        assert tool_result["server"] == "scoped-mcp"
        assert tool_result["tool_name"] == "query_scoped_trend"
        trend_result = tool_result["result"]["trend_result"]
        assert trend_result["metric_name"] == "review_score_avg"
        assert trend_result["granularity"] == "month"
        assert len(trend_result["points"]) == 1
        assert trend_result["points"][0]["value"] == pytest.approx(3.0)
        assert "3" in body["answer"]
        with engine.connect() as connection:
            assert connection.execute(text("SELECT current_user")).scalar_one() == "bizinsight_app"

        assert association_body["error"] is False, {
            "errors": association_body.get("errors"),
            "tool_names": [
                item.get("tool_name")
                for item in traces.get(association_body.get("trace_id"), {}).get("tool_calls", [])
            ],
        }
        assert association_body["trace_id"] in traces
        assert "root_cause_evidence" in association_trace, {
            "gaps": [(item.get("kind"), item.get("reason")) for item in association_trace.get("diagnostic_capability_gaps", [])],
            "stop": (association_trace.get("investigation_state") or {}).get("stop_reason"),
            "required": [(item.get("kind"), item.get("required")) for item in (association_trace.get("diagnostic_plan") or {}).get("steps", [])],
            "action": (association_trace.get("diagnostic_action") or {}).get("action_type"),
            "authorized": [item.get("metric_id") for item in association_trace.get("authorized_metrics", [])],
            "tools": [item.get("tool_name") for item in association_trace.get("tool_calls", [])],
            "llm_failures": association_trace.get("diagnostic_llm_failures"),
        }
        association_record = next(
            item for item in association_trace["root_cause_evidence"]
            if item.get("purpose") == "registered_association:delivery_timing_bad_review_v1"
        )
        association_result = association_record["result"]["result"]["association_result"]
        assert association_result["pair_id"] == "delivery_timing_bad_review_v1"
        assert association_result["population_orders"] == 60
        assert association_result["eligible_orders"] == 60
        assert association_result["risk_difference"] == pytest.approx(0.4)
        assert association_trace["controlled_analysis_scope"]["seller_id"] == seller_id
        assert any(
            item.get("tool_name") == "query_scoped_registered_association"
            for item in association_trace["tool_calls"]
        )
        association_authorized_metric = next(
            item for item in association_trace["authorized_metrics"]
            if item["metric_id"] == "seller_order_bad_review_rate"
        )
        assert association_authorized_metric["provenance_type"] == "explorer_selection"
    finally:
        with engine.begin() as connection:
            connection.execute(text("RESET ROLE"))
            connection.execute(text("""
                DELETE FROM governance.chat_turns
                WHERE tenant_id=:tenant_id AND user_id=:user_id
            """), {"tenant_id": tenant_id, "user_id": user_id})
            connection.execute(text("""
                DELETE FROM governance.chat_sessions
                WHERE tenant_id=:tenant_id AND user_id=:user_id
            """), {"tenant_id": tenant_id, "user_id": user_id})
            connection.execute(text("""
                DELETE FROM governance.authorization_snapshots
                WHERE tenant_id=:tenant_id AND user_id=:user_id
            """), {"tenant_id": tenant_id, "user_id": user_id})
            connection.execute(text("""
                DELETE FROM governance.abac_policy_rules
                WHERE tenant_id=:tenant_id AND policy_id=:policy_id
            """), {"tenant_id": tenant_id, "policy_id": policy_id})
            connection.execute(text("DELETE FROM iam.auth_sessions WHERE tenant_id=:tenant_id"), {"tenant_id": tenant_id})
            connection.execute(text("DELETE FROM iam.api_credentials WHERE credential_id=:credential_id"), {"credential_id": api_credential_id})
            connection.execute(text("DELETE FROM iam.membership_seller_grants WHERE tenant_id=:tenant_id"), {"tenant_id": tenant_id})
            connection.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant_id"), {"tenant_id": tenant_id})
            connection.execute(text("DELETE FROM iam.tenants WHERE tenant_id=:tenant_id"), {"tenant_id": tenant_id})
            connection.execute(text("DELETE FROM iam.users WHERE user_id=:user_id"), {"user_id": user_id})
            connection.execute(text("DELETE FROM iam.users WHERE user_id=:user_id"), {"user_id": other_user_id})
            for table in (
                "analytics.fact_orders", "analytics.fact_order_items", "raw.order_items",
                "analytics.fact_order_seller_allocation", "analytics.fact_fulfillment",
                "raw.order_reviews",
            ):
                connection.execute(text(
                    f"DELETE FROM {table} WHERE order_id LIKE :prefix"
                ), {"prefix": f"{order_prefix}%"})
