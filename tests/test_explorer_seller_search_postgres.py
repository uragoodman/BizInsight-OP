"""PostgreSQL acceptance for the seeded Explorer seller catalog query."""

from __future__ import annotations

from app.data.repositories.explorer import search_sellers
from app.tenancy import TenantContext


FIXED_SELLER = "6560211a19b47992c3666cc44a7e94c0"


def test_seeded_seller_search_returns_city_label_state_and_order_count() -> None:
    context = TenantContext(
        tenant_id="seller-catalog-acceptance",
        user_id="platform-admin-acceptance",
        role="admin",
        scope_type="platform",
    )

    rows = search_sellers(context, query="", limit=5)

    assert rows
    assert len(rows) <= 5
    for row in rows:
        assert {"seller_id", "label", "city", "state", "order_count"} <= row.keys()
        assert row["city"]
        assert row["state"]
        assert row["label"]
        assert int(row["order_count"]) > 0

    fixed_seller = next(row for row in rows if row["seller_id"] == FIXED_SELLER)
    assert fixed_seller["city"] == "sao-paulo"
    assert fixed_seller["state"] == "SP"
    assert FIXED_SELLER[:8] in str(fixed_seller["label"])
