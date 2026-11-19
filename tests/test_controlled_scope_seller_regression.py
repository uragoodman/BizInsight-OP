from __future__ import annotations

import hashlib
import json
from datetime import date

import pytest

from app.auth.context import build_authorization_context
from app.core.config import Settings
from app.data.repositories.explorer import controlled_scope
from app.graph.nodes.data_query import _validate_lifecycle_boundary
from app.metrics.registry import get_registry
from app.schemas.explorer import (
    CONTROLLED_ANALYSIS_DEFAULT_DIMENSIONS,
    ControlledAnalysisScope,
)
from app.tenancy import SellerGrant, TenantContext, activate_tenant_context


def _snapshot():
    return {
        "context": {
            "seller_id": "A",
            "start_date": date(2024, 1, 1),
            "end_date": date(2024, 1, 31),
            "metric_name": "gmv",
            "granularity": "day",
        },
    }


def _tenant():
    return TenantContext(
        "tenant-test", "user-test", "analyst", "seller", seller_id="A",
        seller_grants=(SellerGrant("A", "2024-01-01", "2024-01-31"),),
    )


def _controlled_payload(scope: ControlledAnalysisScope):
    return scope.model_dump(mode="json")


def test_flags_off_default_scope_does_not_authorize_seller_dimension(monkeypatch):
    monkeypatch.setattr("app.core.config.get_settings", lambda: Settings())
    tenant = _tenant()
    issued = controlled_scope(_snapshot())
    assert "seller" not in issued.allowed_dimensions
    assert issued.allowed_dimensions == CONTROLLED_ANALYSIS_DEFAULT_DIMENSIONS
    scope = ControlledAnalysisScope(
        seller_id="A", default_start_date=date(2024, 1, 1),
        default_end_date=date(2024, 1, 31), metric_name="gmv", granularity="day",
        allowed_metrics=issued.allowed_metrics,
    )
    assert "seller" not in scope.allowed_dimensions
    assert issued.allowed_dimensions == scope.allowed_dimensions
    assert issued.model_dump(mode="json") == scope.model_dump(mode="json")
    authz = build_authorization_context(tenant, metric_scope_type="seller")
    assert "seller" not in authz.allowed_dimensions
    assert all(
        "seller" not in item["dimensions"]
        for item in get_registry().planning_catalog(scope="seller")
    )
    state = {
        "controlled_analysis_scope": _controlled_payload(scope),
        "authorization_boundary": None,
    }
    with activate_tenant_context(tenant):
        current = _validate_lifecycle_boundary(state, state["controlled_analysis_scope"])
    assert current["query_scope_fingerprint"]


def test_enabled_server_issuer_adds_seller_and_lifecycle_accepts_it(monkeypatch):
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(seller_dimension_enabled=True, seller_multi_scope_enabled=False),
    )
    tenant = _tenant()
    scope = controlled_scope(_snapshot())
    assert "seller" in scope.allowed_dimensions
    state = {
        "controlled_analysis_scope": _controlled_payload(scope),
        "authorization_boundary": None,
    }
    with activate_tenant_context(tenant):
        current = _validate_lifecycle_boundary(state, state["controlled_analysis_scope"])
    assert current["query_scope_fingerprint"]


def test_enabled_flag_does_not_expand_explicit_scope_without_seller(monkeypatch):
    from app.metrics.authorization import authorize_metric
    from app.mcp.client import query_scoped_business_data
    from app.mcp.schemas import MetricQueryPlan

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(seller_dimension_enabled=True, seller_multi_scope_enabled=False),
    )
    tenant = _tenant()
    scope = ControlledAnalysisScope(
        seller_id="A", default_start_date=date(2024, 1, 1),
        default_end_date=date(2024, 1, 31), metric_name="gmv", granularity="day",
        allowed_metrics=("gmv",), allowed_dimensions=("product_category",),
    )
    with activate_tenant_context(tenant):
        _validate_lifecycle_boundary({"authorization_boundary": None}, _controlled_payload(scope))
    metric = authorize_metric(
        "gmv", provenance_type="exact", provenance_id="test:gmv", scope="seller",
        scope_data=scope.model_dump(mode="json"),
    )
    plan = MetricQueryPlan(
        query_type="metric_breakdown", metric_name="gmv", dimension="seller",
        authorized_metric=metric,
    )
    with activate_tenant_context(tenant):
        response = query_scoped_business_data(
            "按店铺拆解 GMV", scope.model_dump(mode="json"), plan=plan,
        )
    assert response["result"]["status"] == "unsupported"
    assert "Seller dimension is outside" in response["result"]["reason"]


def test_legacy_mcp_scope_without_seller_dimension_fails_before_query(monkeypatch):
    from app.metrics.authorization import authorize_metric
    from app.mcp.client import query_scoped_business_data
    from app.mcp.schemas import MetricQueryPlan

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(seller_dimension_enabled=True, seller_multi_scope_enabled=True),
    )
    tenant = _tenant()
    scope = ControlledAnalysisScope(
        seller_id="A", default_start_date=date(2024, 1, 1),
        default_end_date=date(2024, 1, 31), metric_name="gmv", granularity="day",
        allowed_metrics=("gmv",), allowed_dimensions=("product_category",),
    )
    metric = authorize_metric(
        "gmv", provenance_type="exact", provenance_id="test:gmv", scope="seller",
        scope_data=scope.model_dump(mode="json"),
    )
    plan = MetricQueryPlan(
        query_type="metric_breakdown", metric_name="gmv", dimension="seller",
        authorized_metric=metric,
    )
    calls = []
    monkeypatch.setattr(
        "app.mcp.client._query_scoped_seller_breakdown_with_context",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"result": {"status": "ok"}},
    )
    with activate_tenant_context(tenant):
        result = query_scoped_business_data(
            "按店铺拆解 GMV", scope.model_dump(mode="json"), plan=plan,
        )
    assert result["result"]["status"] == "unsupported"
    assert "server-locked Explorer scope" in result["result"]["reason"]
    assert calls == []


def test_flag_off_catalog_digest_matches_pre_seller_registry(monkeypatch):
    registry = get_registry()
    registry._require_loaded()
    monkeypatch.setattr("app.core.config.get_settings", lambda: Settings())
    payload = []
    for metric_id in sorted(registry._metrics):
        metric = registry._metrics[metric_id]
        payload.append({
            "metric_id": metric.metric_id,
            "display_name": metric.display_name,
            "aliases": list(metric.aliases),
            "unit": metric.unit,
            "definition": metric.documentation,
            "version": metric.version,
            "surfaces": list(metric.surfaces),
            "planning": {
                "exact_aliases": list(metric.planning.exact_aliases),
                "semantic_terms": list(metric.planning.semantic_terms),
                "phrases": list(metric.planning.phrases),
                "examples": list(metric.planning.examples),
                "intents": list(metric.planning.intents),
                "limitations": metric.planning.limitations,
            },
            "implementations": {
                scope_name: {
                    "handler_key": implementation.handler_key,
                    "capabilities": sorted(implementation.capabilities),
                    "dimensions": [
                        dimension for dimension in implementation.dimensions
                        if dimension != "seller"
                    ],
                    "lineage_source": implementation.lineage_source,
                }
                for scope_name, implementation in sorted(metric.implementations.items())
            },
            "attribution": {
                "scopes": {
                    scope_name: {
                        "methods": list(capability.method_ids),
                        "dimensions": list(capability.dimensions),
                        "required_data": {
                            method: list(fields)
                            for method, fields in sorted(capability.required_data.items())
                        },
                        "factor_specs": {
                            method: [
                                {
                                    "factor_id": factor.factor_id,
                                    "kind": factor.kind,
                                    "metric_id": factor.metric_id,
                                    "numerator_metric": factor.numerator_metric,
                                    "denominator_metric": factor.denominator_metric,
                                }
                                for factor in factors
                            ]
                            for method, factors in sorted(capability.factor_specs.items())
                        },
                    }
                    for scope_name, capability in sorted(metric.attribution.scopes.items())
                },
                "limitation": metric.attribution.limitation,
            },
        })
    old_payload_digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert old_payload_digest == "a9785cee6bd9dde7ca53365bf77ee73fb63b2dafbd489432e4a39eb15fd218bc"
    assert registry.catalog_digest == old_payload_digest

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(seller_dimension_enabled=True),
    )
    assert registry.catalog_digest != old_payload_digest


def test_seller_three_year_cap_matches_explorer_including_leap_day():
    from app.schemas.explorer import ExplorerQuery
    from app.scope.seller_scope import SellerScopeIntent

    assert Settings().seller_scope_max_date_span_days == 366 * 3
    # Explorer validates the elapsed-day difference, so this leap-containing
    # three-year range is accepted at the shared 1098-day limit.
    scope = ControlledAnalysisScope(
        seller_id="A", default_start_date=date(2020, 1, 1),
        default_end_date=date(2023, 1, 3), metric_name="gmv", granularity="day",
        allowed_metrics=("gmv",),
    )
    request = SellerScopeIntent(
        seller_scope_type="single", seller_ids=("A",),
        start_date=scope.default_start_date, end_date=scope.default_end_date,
        metric_ids=("gmv",), dimensions=("seller",),
    )
    assert (request.end_date - request.start_date).days == 1098
    ExplorerQuery(
        seller_id="A", start_date=request.start_date, end_date=request.end_date,
        metric_name="gmv", granularity="day",
    )

