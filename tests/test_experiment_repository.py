from contextlib import contextmanager
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from app.data.repositories import experiments
from app.schemas.explorer import ControlledAnalysisScope
from app.tenancy import SellerGrant, TenantContext


def _scope():
    return ControlledAnalysisScope.model_validate({
        "seller_id": "seller-1",
        "allowed_metrics": ["gmv"],
        "default_start_date": date(2024, 1, 1),
        "default_end_date": date(2024, 1, 31),
        "metric_name": "gmv",
        "granularity": "day",
    })


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _Connection:
    def __init__(self, rows):
        self.rows = rows
        self.statement = ""
        self.params = None

    def execute(self, statement, params):
        self.statement = str(statement)
        self.params = params
        return _Result(self.rows)


def _row(unit_id, group, outcome):
    starts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ends = datetime(2024, 1, 31, 23, 59, tzinfo=timezone.utc)
    return {
        "experiment_id": "exp-1",
        "seller_id": "seller-1",
        "outcome_metric_id": "gmv",
        "assignment_unit": "user",
        "assignment_method": "randomized",
        "status": "closed",
        "starts_at": starts,
        "ends_at": ends,
        "randomization_verified": True,
        "verification_ref": "audit-123",
        "verified_by": "governance-service",
        "verified_at": ends,
        "randomization_unit_hash": unit_id,
        "assigned_group": group,
        "outcome_value": outcome,
    }


def _prepare(monkeypatch, rows):
    connection = _Connection(rows)

    @contextmanager
    def begin():
        yield connection

    class _Authz:
        allowed_metric_ids = ("gmv",)

        def require_sellers(self, sellers):
            assert sellers == ["seller-1"]

        def require_date_range(self, seller_id, start, end):
            assert (seller_id, start, end) == ("seller-1", date(2024, 1, 1), date(2024, 1, 31))

        def require_metrics(self, metrics):
            if "gmv" not in metrics:
                raise PermissionError("metric denied")

        def require_tool(self, tool_name):
            if tool_name != "estimate_scoped_randomized_experiment":
                raise PermissionError("tool denied")

    tenant = TenantContext(
        tenant_id="tenant-1", user_id="user-1", role="analyst",
        scope_type="seller", seller_id="seller-1",
        seller_grants=(SellerGrant(seller_id="seller-1"),),
    )
    monkeypatch.setattr(experiments.engine, "begin", begin)
    monkeypatch.setattr(experiments, "current_tenant_context", lambda: tenant)
    monkeypatch.setattr(experiments, "set_local_context", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "app.auth.context.build_authorization_context",
        lambda *_a, **_kw: _Authz(),
    )
    monkeypatch.setattr(
        "app.metrics.registry.get_registry",
        lambda: SimpleNamespace(
            resolve=lambda metric: metric,
            get=lambda _metric: SimpleNamespace(unit="BRL"),
        ),
    )
    return connection


def test_verified_randomized_experiment_is_authorized_and_unit_ids_are_not_returned(monkeypatch):
    rows = [
        _row(f"unit-{i}", "treatment" if i < 40 else "control", 2.0 if i < 40 else 1.0)
        for i in range(80)
    ]
    connection = _prepare(monkeypatch, rows)

    result = experiments.estimate_scoped_randomized_experiment(
        _scope(), experiment_id="exp-1",
    )

    assert result["status"] == "ok"
    assert result["effect"] == pytest.approx(1.0)
    assert result["source_table"] == "governance.experiment_registry+analytics.experiment_observations"
    assert result["unit_identifiers_returned"] is False
    assert "r.status = 'closed'" in connection.statement
    assert "r.randomization_verified IS TRUE" in connection.statement
    assert connection.params["tenant_id"] == "tenant-1"
    assert connection.params["seller_id"] == "seller-1"
    assert connection.params["row_limit"] == 100_001
    assert "unit-0" not in repr(result)


def test_experiment_not_verified_or_outside_scope_returns_no_estimable_data(monkeypatch):
    connection = _prepare(monkeypatch, [])
    result = experiments.estimate_scoped_randomized_experiment(
        _scope(), experiment_id="unavailable",
    )
    assert result["status"] == "not_available"
    assert result["causal_effect_estimated"] is False
    assert "verified randomized experiment" in result["reason"]
    assert "r.starts_at::date >= :scope_start" in connection.statement
    assert "r.ends_at::date <= :scope_end" in connection.statement


def test_experiment_repository_requires_trusted_context_and_bounded_id(monkeypatch):
    monkeypatch.setattr(experiments, "current_tenant_context", lambda: None)
    with pytest.raises(PermissionError, match="Trusted tenant"):
        experiments.estimate_scoped_randomized_experiment(_scope(), experiment_id="exp-1")

    with pytest.raises(ValueError, match="valid registered"):
        experiments.estimate_scoped_randomized_experiment(_scope(), experiment_id="x" * 129)


def test_experiment_estimation_requires_separate_tool_authorization(monkeypatch):
    rows = [_row(f"unit-{i}", "treatment" if i < 40 else "control", float(i % 3)) for i in range(80)]
    connection = _prepare(monkeypatch, rows)

    from app.auth.context import AuthorizationDenied

    class _DeniedToolAuthz:
        def require_sellers(self, _sellers): pass
        def require_date_range(self, *_args): pass
        def require_metrics(self, _metrics): pass
        def require_tool(self, _tool): raise AuthorizationDenied("TOOL_DENIED", "not granted")

    monkeypatch.setattr(
        "app.auth.context.build_authorization_context",
        lambda *_a, **_kw: _DeniedToolAuthz(),
    )
    with pytest.raises(PermissionError, match="not granted"):
        experiments.estimate_scoped_randomized_experiment(
            _scope(), experiment_id="exp-1",
        )
    assert connection.statement == ""


def test_list_experiments_returns_only_verified_scope_metadata(monkeypatch):
    from datetime import datetime, timezone

    starts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ends = datetime(2024, 1, 31, tzinfo=timezone.utc)
    row = {
        "experiment_id": "exp-1", "seller_id": "seller-1", "outcome_metric_id": "gmv",
        "assignment_unit": "user", "starts_at": starts, "ends_at": ends,
        "observation_count": 80,
    }
    connection = _prepare(monkeypatch, [row])

    result = experiments.list_scoped_randomized_experiments(_scope())

    assert result["items"][0]["experiment_id"] == "exp-1"
    assert result["items"][0]["observation_count"] == 80
    assert "r.randomization_verified IS TRUE" in connection.statement
    assert "r.status = 'closed'" in connection.statement
    assert connection.params["tenant_id"] == "tenant-1"
