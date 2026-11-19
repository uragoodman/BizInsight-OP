from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from app.analysis.contracts import PeriodSpec
from app.analysis import insights
from app.analysis.attribution import AttributionDataInsufficient
from app.analysis.report_alignment import validate_report_breakdown_totals
from app.tenancy import SellerGrant, TenantContext, current_tenant_context


def test_async_insight_worker_restores_authenticated_seller_context(monkeypatch):
    tenant = TenantContext(
        tenant_id="tenant-a", user_id="user-a", role="analyst", scope_type="seller",
        seller_grants=(SellerGrant("seller-a"),), auth_session_id="session-a",
    )

    class Store:
        finalized = None

        def get(self, *_args):
            return {"report_json": {}}

        def finalize_insight(self, *_args):
            self.finalized = _args[-1]

        def fail_insight(self, *_args):
            pytest.fail(f"worker unexpectedly failed: {_args[-1]}")

    store = Store()
    monkeypatch.setattr(insights, "_report_from_row", lambda _row: object())

    def generate(*_args, **_kwargs):
        assert current_tenant_context() == tenant
        return "artifact"

    monkeypatch.setattr(insights, "run_insight_generation", generate)
    insights._background_insight_task(
        store, "report-a", 1, 1, object(), "actor-a",
        {"tenant_id": "tenant-a", "user_id": "user-a"}, tenant,
    )

    assert store.finalized == "artifact"
    assert current_tenant_context() is None


def test_live_breakdown_must_reconcile_with_report_snapshot():
    current = PeriodSpec(start_date=date(2025, 2, 1), end_date=date(2025, 2, 28), granularity="month")
    baseline = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 31), granularity="month")
    comparison = SimpleNamespace(
        current_period=current, baseline_period=baseline,
        current_value=100.0, baseline_value=80.0, delta_value=20.0,
    )
    report = SimpleNamespace(period=current)
    metric = SimpleNamespace(value=100.0, period_over_period=comparison)

    with pytest.raises(AttributionDataInsufficient) as error:
        validate_report_breakdown_totals(report, metric, baseline, 105.0, 80.0)
    assert error.value.period_statuses == {"current": "report_snapshot_total_mismatch"}


def test_live_breakdown_rejects_period_misalignment():
    report_period = PeriodSpec(start_date=date(2025, 2, 1), end_date=date(2025, 2, 28), granularity="month")
    wrong_period = PeriodSpec(start_date=date(2025, 2, 1), end_date=date(2025, 2, 28), granularity="day")
    baseline = PeriodSpec(start_date=date(2025, 1, 1), end_date=date(2025, 1, 31), granularity="month")
    comparison = SimpleNamespace(
        current_period=wrong_period, baseline_period=baseline,
        current_value=100.0, baseline_value=80.0, delta_value=20.0,
    )
    report = SimpleNamespace(period=report_period)
    metric = SimpleNamespace(value=100.0, period_over_period=comparison)

    with pytest.raises(AttributionDataInsufficient) as error:
        validate_report_breakdown_totals(report, metric, baseline, 100.0, 80.0)
    assert error.value.period_statuses == {"comparison": "report_period_mismatch"}
