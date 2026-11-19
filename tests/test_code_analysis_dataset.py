"""P2 授权数据预取测试：RLS 范围校验、指标白名单、Parquet 工件返回。"""

from __future__ import annotations

from datetime import date

import pytest

from app.code_analysis.dataset import prefetch_business_dataset
from app.tenancy import SellerGrant, TenantContext


class _FakeRegistry:
    def supports(self, metric: str, capability: str, scope: str) -> bool:
        return True


def _seller_context() -> TenantContext:
    return TenantContext(
        tenant_id="tenant-a", user_id="user-1", role="analyst",
        scope_type="seller", seller_id="s1",
        seller_grants=(SellerGrant(seller_id="s1"),),
    )


def test_prefetch_rejects_unauthorized_seller(monkeypatch) -> None:
    monkeypatch.setattr("app.code_analysis.dataset.get_registry", lambda: _FakeRegistry())
    with pytest.raises(PermissionError):
        prefetch_business_dataset(
            _seller_context(), seller_id="s2",
            start_date=date(2018, 8, 1), end_date=date(2018, 8, 2),
            metric_names=["gmv"],
        )


def test_prefetch_rejects_non_whitelisted_metric(monkeypatch) -> None:
    class Registry:
        def supports(self, metric, capability, scope):
            return False
    monkeypatch.setattr("app.code_analysis.dataset.get_registry", lambda: Registry())
    with pytest.raises(ValueError):
        prefetch_business_dataset(
            _seller_context(), seller_id="s1",
            start_date=date(2018, 8, 1), end_date=date(2018, 8, 2),
            metric_names=["gmv"],
        )


def test_prefetch_builds_governed_parquet_artifact(monkeypatch) -> None:
    monkeypatch.setattr("app.code_analysis.dataset.get_registry", lambda: _FakeRegistry())

    def fake_trend(scope, metric, *, start_date=None, end_date=None):
        return {
            "points": [{"bucket": "2018-08-01", "value": 100.0}, {"bucket": "2018-08-02", "value": 120.0}],
            "source_table": "fact_orders", "status": "ok",
        }
    monkeypatch.setattr("app.code_analysis.dataset.query_scoped_trend", fake_trend)

    def fake_upload(context, filename, content_type, raw):
        # Parquet 工件确实被写入 upload_file 生命周期（这里是桩）。
        return {"file_id": "file_abc123", "sha256": "a" * 64, "size_bytes": len(raw),
                "expires_at": "2026-08-03T00:00:00Z", "data_contract": {"file_type": "parquet", "rows": 2}}
    monkeypatch.setattr("app.code_analysis.dataset.upload_file", fake_upload)

    result = prefetch_business_dataset(
        _seller_context(), seller_id="s1",
        start_date=date(2018, 8, 1), end_date=date(2018, 8, 2),
        metric_names=["gmv", "order_count"],
    )
    assert result["file_id"] == "file_abc123"
    assert result["rows"] == 4  # 2 指标 × 2 天
    assert result["scope_fingerprint"]
    assert result["query_version"] == "dataset-v1"
    assert result["metrics"] == ["gmv", "order_count"]
    assert "fact_orders" in result["source_tables"]
