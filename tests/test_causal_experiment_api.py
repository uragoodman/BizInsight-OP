from datetime import date

from fastapi.testclient import TestClient

from app.api.auth import require_tenant_context
from app.main import app
from app.tenancy import SellerGrant, TenantContext


def _tenant():
    return TenantContext(
        tenant_id="tenant-test", user_id="user-test", role="analyst",
        scope_type="seller", seller_id="seller-test",
        seller_grants=(SellerGrant("seller-test", "2026-01-01", "2026-01-31"),),
    )


def test_eligible_endpoint_returns_only_server_repo_experiments(monkeypatch):
    from app.api.routes import causal_experiments

    monkeypatch.setattr(causal_experiments, "require_permission", lambda *_: None)
    monkeypatch.setattr(
        "app.data.repositories.experiments.list_scoped_randomized_experiments",
        lambda scope: {
            "items": [{"experiment_id": "registered-1", "outcome_metric_id": "gmv"}],
            "truncated": False,
            "scope": {"seller_id": scope.seller_id},
        },
    )
    app.dependency_overrides[require_tenant_context] = _tenant
    try:
        response = TestClient(app).post("/causal-experiments/eligible", json={
            "seller_id": "seller-test", "start_date": "2026-01-01", "end_date": "2026-01-31",
        })
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200, response.text
    assert response.json()["items"][0]["experiment_id"] == "registered-1"


def test_estimate_endpoint_uses_opaque_id_and_rejects_unauthorized_action(monkeypatch):
    from app.api.routes import causal_experiments

    monkeypatch.setattr(causal_experiments, "require_permission", lambda *_: (_ for _ in ()).throw(PermissionError("analysis denied")))
    app.dependency_overrides[require_tenant_context] = _tenant
    try:
        response = TestClient(app).post("/causal-experiments/estimate", json={
            "seller_id": "seller-test", "start_date": "2026-01-01", "end_date": "2026-01-31",
            "experiment_id": "registered-1", "outcome_metric_id": "forged_metric",
        })
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 403
    assert "analysis denied" in response.text


def test_estimate_endpoint_executes_the_registered_experiment_contract(monkeypatch):
    from app.api.routes import causal_experiments

    captured = {}
    monkeypatch.setattr(causal_experiments, "require_permission", lambda *_: None)

    def estimate(scope, *, experiment_id):
        captured["scope"] = scope
        captured["experiment_id"] = experiment_id
        return {"status": "ok", "causal_effect_estimated": True, "effect": 1.0}

    monkeypatch.setattr("app.data.repositories.experiments.estimate_scoped_randomized_experiment", estimate)
    app.dependency_overrides[require_tenant_context] = _tenant
    try:
        response = TestClient(app).post("/causal-experiments/estimate", json={
            "seller_id": "seller-test", "start_date": "2026-01-01", "end_date": "2026-01-31",
            "experiment_id": "registered-1", "outcome_metric_id": "forged_metric",
        })
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200, response.text
    assert response.json()["effect"] == 1.0
    assert captured["experiment_id"] == "registered-1"
    assert captured["scope"].seller_id == "seller-test"


def test_causal_api_date_scope_is_bounded():
    from app.schemas.experiments import CausalExperimentScopeRequest
    import pytest

    with pytest.raises(ValueError, match="three calendar years"):
        CausalExperimentScopeRequest(
            seller_id="seller-test", start_date=date(2020, 1, 1), end_date=date(2024, 1, 1),
        )
