from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace

import pytest

from app.data.repositories import explorer
from app.investigation.candidates import generate_candidate_actions
from app.investigation.contracts import AnalysisTarget
from app.mcp.client import query_scoped_business_data
from app.mcp.schemas import MetricQueryPlan
from app.schemas.explorer import ControlledAnalysisScope


def _scope(**updates):
    values = {
        "seller_id": "seller-1",
        "allowed_metrics": ("seller_avg_freight_per_order",),
        "default_start_date": date(2024, 1, 1),
        "default_end_date": date(2024, 1, 31),
        "metric_name": "seller_avg_freight_per_order",
        "granularity": "day",
    }
    values.update(updates)
    return ControlledAnalysisScope.model_validate(values)


class _Rows:
    def mappings(self):
        return self

    def one(self):
        return {
            "sample_size": 4,
            "excluded_missing": 1,
            "mean": 12.5,
            "std": 8.5,
            "min": 0.0,
            "q1": 3.0,
            "median": 10.0,
            "q3": 20.0,
            "max": 30.0,
            "zero_freight_orders": 1,
        }


class _AssociationRows:
    def mappings(self):
        return self

    def one(self):
        return {
            "population_orders": 120,
            "eligible_orders": 110,
            "on_time_n": 80,
            "on_time_bad": 8,
            "delayed_n": 30,
            "delayed_bad": 9,
        }


class _Connection:
    def __init__(self):
        self.statement = None
        self.params = None

    def execute(self, statement, params):
        self.statement = str(statement)
        self.params = params
        return _Rows()


def test_order_freight_distribution_uses_entity_population_and_keeps_zero(monkeypatch):
    connection = _Connection()

    @contextmanager
    def begin():
        yield connection

    monkeypatch.setattr(explorer.engine, "begin", begin)
    monkeypatch.setattr("app.tenancy.current_tenant_context", lambda: object())
    monkeypatch.setattr("app.data.rls.set_local_context", lambda *_args, **_kwargs: None)

    result = explorer.query_scoped_order_freight_distribution(_scope())

    assert result["basis"] == "order_entity"
    assert result["entity"] == "delivered_seller_order"
    assert result["sample_size"] == 4
    assert result["zero_freight_orders"] == 1
    assert result["excluded_missing_freight_orders"] == 1
    assert result["statistics"]["median"] == 10.0
    assert "order_id" not in result
    assert "percentile_cont(0.5)" in connection.statement
    assert "purchase_date" in connection.statement
    assert connection.params["seller_id"] == "seller-1"


def test_order_freight_distribution_fails_closed_outside_metric_and_date_scope(monkeypatch):
    scope = _scope(allowed_metrics=("gmv",))
    with pytest.raises(PermissionError):
        explorer.query_scoped_order_freight_distribution(scope)

    with pytest.raises(ValueError, match="outside the locked scope"):
        explorer.query_scoped_order_freight_distribution(
            _scope(), start_date=date(2023, 12, 31), end_date=date(2024, 1, 10),
        )


def test_scoped_chat_routes_entity_statistics_to_order_repository(monkeypatch):
    scope_payload = {
        "seller_id": "seller-1",
        "allowed_metrics": ["seller_avg_freight_per_order"],
        "default_start_date": "2024-01-01",
        "default_end_date": "2024-01-31",
        "metric_name": "seller_avg_freight_per_order",
        "granularity": "day",
    }
    scope_model = ControlledAnalysisScope.model_validate(scope_payload)
    from app.metrics.authorization import authorize_metric
    from app.tenancy import SellerGrant, TenantContext
    monkeypatch.setattr(
        "app.mcp.client.current_tenant_context",
        lambda: TenantContext(
            tenant_id="public", user_id="test-user", role="analyst",
            scope_type="seller", seller_id="seller-1",
            seller_grants=(SellerGrant(seller_id="seller-1"),),
        ),
    )

    plan = MetricQueryPlan(
        query_type="metric_entity_statistics",
        metric_name="seller_avg_freight_per_order",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    plan = plan.model_copy(update={
        "authorized_metric": authorize_metric(
            "seller_avg_freight_per_order",
            provenance_type="explorer_selection",
            provenance_id="entity-distribution-test",
            scope="seller",
            scope_data={**scope_model.model_dump(mode="json"), "tenant_id": None, "user_id": None, "permissions": None},
        ),
    })
    monkeypatch.setattr(
        "app.data.repositories.explorer.query_scoped_order_freight_distribution",
        lambda *_args, **_kwargs: {
            "metric_name": "seller_avg_freight_per_order",
            "basis": "order_entity",
            "entity": "delivered_seller_order",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "sample_size": 2,
            "statistics": {"mean": 4.0, "std": 2.0, "min": 2.0, "q1": 2.5, "median": 3.0, "q3": 3.5, "max": 6.0},
            "zero_freight_orders": 0,
            "excluded_missing_freight_orders": 0,
            "population_definition": "delivered orders with complete freight values",
            "source_table": "analytics.v_delivered_order_items + analytics.v_delivered_orders",
            "status": "ok",
        },
    )

    result = query_scoped_business_data("按订单看运费分布", scope_payload, plan=plan)

    assert result["tool_name"] == "query_scoped_order_freight_distribution", result["result"].get("reason")
    assert result["result"]["status"] == "ok"
    assert result["result"]["statistics_result"]["basis"] == "order_entity"


def test_entity_distribution_candidate_is_opt_in_by_metric_and_action_capability():
    target = AnalysisTarget(
        metric_id="seller_avg_freight_per_order",
        current_start=date(2024, 1, 1),
        current_end=date(2024, 1, 31),
        canonical_scope={"seller_id": "seller-1"},
        scope_fingerprint="scope-1",
        requested_analysis_type="distribution",
    )

    candidates = generate_candidate_actions(
        target,
        allowed_methods=(),
        allowed_dimensions=(),
        supported_action_types=("ANALYZE_ENTITY_DISTRIBUTION",),
        entity_distribution_metric_ids=("seller_avg_freight_per_order",),
    )

    assert len(candidates) == 1
    assert candidates[0].action.action_type == "ANALYZE_ENTITY_DISTRIBUTION"

    denied = generate_candidate_actions(
        target,
        allowed_methods=(),
        allowed_dimensions=(),
        supported_action_types=("ANALYZE_ENTITY_DISTRIBUTION",),
        entity_distribution_metric_ids=("gmv",),
    )
    assert denied == []


def test_registered_association_query_returns_effect_sample_missing_and_limitations(monkeypatch):
    connection = _Connection()

    @contextmanager
    def begin():
        yield connection

    monkeypatch.setattr(explorer.engine, "begin", begin)
    monkeypatch.setattr("app.tenancy.current_tenant_context", lambda: object())
    monkeypatch.setattr("app.data.rls.set_local_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(connection, "execute", lambda statement, params: _AssociationRows())
    class _Authz:
        def require_sellers(self, _sellers): pass
        def require_metrics(self, _metrics): pass
        def require_date_range(self, *_args): pass
        def require_tool(self, _tool): pass
    monkeypatch.setattr("app.auth.context.build_authorization_context", lambda *_a, **_kw: _Authz())
    scope = _scope(
        allowed_metrics=("seller_order_bad_review_rate",),
        metric_name="seller_order_bad_review_rate",
    )

    result = explorer.query_scoped_delivery_review_association(
        scope, pair_id="delivery_timing_bad_review_v1",
    )

    assert result["status"] == "ok"
    assert result["population_orders"] == 120
    assert result["eligible_orders"] == 110
    assert result["excluded_missing_pair_fields"] == 10
    assert result["sample_size"] == 110
    assert result["risk_difference"] == pytest.approx(0.3 - 0.1)
    assert result["risk_difference_ci_95"]
    assert result["limitations"]
    assert result["analysis_contract"]["analysis_unit"] == "single_seller_order"
    assert result["analysis_contract"]["minimum_sample_size"] == 50


@pytest.mark.parametrize(
    ("pair_id", "expected_column"),
    [
        ("delivery_hours_review_score_spearman_v1", "delivery_transit_hours"),
        ("order_freight_review_score_spearman_v1", "freight_sum"),
        ("order_item_line_count_review_score_spearman_v1", "item_count"),
        ("order_merchandise_value_review_score_spearman_v1", "price_sum"),
    ],
)
def test_registered_continuous_association_uses_bounded_rows_and_returns_spearman(
    monkeypatch, pair_id, expected_column,
):
    class _QueryResult:
        def __init__(self, rows): self.rows = rows
        def mappings(self): return self
        def one(self): return self.rows[0]
        def all(self): return self.rows

    class _AssociationConnection:
        def __init__(self): self.statements = []
        def execute(self, statement, params):
            sql = str(statement)
            self.statements.append((sql, params))
            if "COUNT(DISTINCT order_id) AS population_orders" in sql:
                return _QueryResult([{"population_orders": 60, "eligible_orders": 58}])
            return _QueryResult([
                {"exposure_value": float(i), "review_score": float(i % 5 + 1)}
                for i in range(58)
            ])

    connection = _AssociationConnection()
    @contextmanager
    def begin(): yield connection
    monkeypatch.setattr(explorer.engine, "begin", begin)
    monkeypatch.setattr("app.tenancy.current_tenant_context", lambda: object())
    monkeypatch.setattr("app.data.rls.set_local_context", lambda *_a, **_kw: None)
    class _Authz:
        def require_sellers(self, _sellers): pass
        def require_metrics(self, _metrics): pass
        def require_date_range(self, *_args): pass
        def require_tool(self, _tool): pass
    monkeypatch.setattr("app.auth.context.build_authorization_context", lambda *_a, **_kw: _Authz())
    scope = _scope(allowed_metrics=("review_score_avg",), metric_name="review_score_avg")

    result = explorer.query_scoped_delivery_review_association(
        scope, pair_id=pair_id,
    )

    assert result["status"] == "ok"
    assert result["method_id"] == "spearman_rank_correlation_fisher_approx_v1"
    assert result["population_orders"] == 60
    assert result["eligible_orders"] == 58
    assert result["excluded_missing_pair_fields"] == 2
    assert len(result["spearman_rho_ci_95_approx"]) == 2
    assert result["analysis_contract"]["method_id"] == result["method_id"]
    assert result["analysis_contract"]["exposure_definition"]
    assert result["analysis_contract"]["exclusion_policy"] == result["missing_policy"]
    assert "LIMIT :row_limit" in connection.statements[1][0]
    assert expected_column in connection.statements[0][0]
    assert expected_column in connection.statements[1][0]


def test_freight_review_association_is_registered_for_the_canonical_metric():
    from app.analysis.association_catalog import get_association_pair, registered_pair_ids_for_metric

    pair = get_association_pair("order_freight_review_score_spearman_v1")
    assert pair is not None
    assert pair.target_metric_id == "review_score_avg"
    assert pair.method_id == "spearman_rank_correlation_fisher_approx_v1"
    assert "order_freight_review_score_spearman_v1" in registered_pair_ids_for_metric("review_score_avg")


@pytest.mark.parametrize(
    ("pair_id", "query_handler", "field_name"),
    [
        ("order_item_line_count_review_score_spearman_v1", "order_item_line_count_review_score", "item_count"),
        ("order_merchandise_value_review_score_spearman_v1", "order_merchandise_value_review_score", "price_sum"),
    ],
)
def test_added_review_association_pairs_have_explicit_units_and_definitions(pair_id, query_handler, field_name):
    from app.analysis.association_catalog import get_association_pair, registered_pair_ids_for_metric

    pair = get_association_pair(pair_id)
    assert pair is not None
    assert pair.target_metric_id == "review_score_avg"
    assert pair.kind == "continuous"
    assert pair.query_handler == query_handler
    assert pair.analysis_unit == "single_seller_order"
    assert field_name in pair.exposure_label
    assert pair.minimum_sample_size == 50
    assert any("association does not establish causality" in item.lower() for item in pair.limitations)
    assert pair_id in registered_pair_ids_for_metric("review_score_avg")


def test_freight_review_association_over_limit_fails_closed_with_complete_evidence_contract(monkeypatch):
    class _QueryResult:
        def mappings(self): return self
        def one(self): return {"population_orders": 120, "eligible_orders": 110}

    class _AssociationConnection:
        def __init__(self): self.statements = []
        def execute(self, statement, params):
            self.statements.append(str(statement))
            return _QueryResult()

    connection = _AssociationConnection()
    @contextmanager
    def begin(): yield connection
    monkeypatch.setattr(explorer.engine, "begin", begin)
    monkeypatch.setattr("app.tenancy.current_tenant_context", lambda: object())
    monkeypatch.setattr("app.data.rls.set_local_context", lambda *_a, **_kw: None)

    class _Authz:
        def require_sellers(self, _sellers): pass
        def require_metrics(self, _metrics): pass
        def require_date_range(self, *_args): pass
        def require_tool(self, _tool): pass

    monkeypatch.setattr("app.auth.context.build_authorization_context", lambda *_a, **_kw: _Authz())
    scope = _scope(
        allowed_metrics=("review_score_avg",), metric_name="review_score_avg", max_rows=100,
    )
    result = explorer.query_scoped_delivery_review_association(
        scope, pair_id="order_freight_review_score_spearman_v1",
    )

    assert result["status"] == "over_limit"
    assert result["eligible_orders"] == 110
    assert result["row_limit"] == 100
    assert result["missing_policy"]
    assert result["limitations"]
    assert result["analysis_contract"]["source_table"] == result["source_table"]
    assert "spearman_rho" not in result
    assert len(connection.statements) == 1  # never fetch or estimate a truncated sample


def test_association_pair_fails_closed_for_unregistered_pair_or_metric():
    with pytest.raises(ValueError, match="not registered"):
        explorer.query_scoped_delivery_review_association(
            _scope(allowed_metrics=("seller_order_bad_review_rate",)),
            pair_id="arbitrary_llm_pair",
        )
    with pytest.raises(PermissionError, match="outside the locked scope"):
        explorer.query_scoped_delivery_review_association(
            _scope(), pair_id="delivery_timing_bad_review_v1",
        )


def test_association_action_is_only_generated_from_server_pair_allowlist():
    target = AnalysisTarget(
        metric_id="seller_order_bad_review_rate",
        current_start=date(2024, 1, 1),
        current_end=date(2024, 1, 31),
        canonical_scope={"seller_id": "seller-1"},
        scope_fingerprint="scope-2",
        requested_analysis_type="diagnostic",
    )
    generated = generate_candidate_actions(
        target,
        allowed_methods=(),
        allowed_dimensions=(),
        supported_action_types=("ANALYZE_ASSOCIATION",),
        association_pair_ids_by_metric={
            "seller_order_bad_review_rate": ("delivery_timing_bad_review_v1",),
        },
    )
    assert len(generated) == 1
    assert generated[0].action.action_type == "ANALYZE_ASSOCIATION"
    assert generated[0].action.pair_id == "delivery_timing_bad_review_v1"

    review_target = target.model_copy(update={"metric_id": "review_score_avg"})
    freight_review = generate_candidate_actions(
        review_target,
        allowed_methods=(),
        allowed_dimensions=(),
        supported_action_types=("ANALYZE_ASSOCIATION",),
        association_pair_ids_by_metric={
            "review_score_avg": ("order_freight_review_score_spearman_v1",),
        },
    )
    assert len(freight_review) == 1
    assert freight_review[0].action.pair_id == "order_freight_review_score_spearman_v1"

    arbitrary = generate_candidate_actions(
        target,
        allowed_methods=(),
        allowed_dimensions=(),
        supported_action_types=("ANALYZE_ASSOCIATION",),
        association_pair_ids_by_metric={"seller_order_bad_review_rate": ("freeform_llm_pair",)},
    )
    assert arbitrary == []


def test_continuous_association_evidence_requires_registered_effect_and_interval():
    from app.investigation.contracts import AnalyzeAssociationAction, AttributionEvidence
    from app.investigation.engine import _evidence_failure

    action = AnalyzeAssociationAction(
        action_type="ANALYZE_ASSOCIATION", metric_id="review_score_avg",
        pair_id="delivery_hours_review_score_spearman_v1",
    )
    from app.analysis.association_catalog import get_association_pair
    pair = get_association_pair("delivery_hours_review_score_spearman_v1")
    assert pair is not None
    evidence = AttributionEvidence(
        evidence_id="ev-spearman", evidence_type="registered_association",
        metric_id="review_score_avg", source_action_id="candidate-1",
        source_ref="analytics.v_order_unified", scope_fingerprint="scope-1",
        method_id="spearman_rank_correlation_fisher_approx_v1", method_version="1",
        authorized_metric_provenance="explorer_selection", limitations=["association only"],
        values={"series_analysis": {
            "pair_id": "delivery_hours_review_score_spearman_v1",
            "target_metric": "review_score_avg", "status": "ok",
            "analysis_contract": pair.evidence_contract(),
            "population_orders": 80, "eligible_orders": 70,
            "excluded_missing_pair_fields": 10, "sample_size": 70,
            "spearman_rho": -0.25,
            "spearman_rho_ci_95_approx": [-0.45, -0.02],
            "missing_policy": "count and exclude missing fields",
            "limitations": ["association is not causality"],
        }},
    )
    assert _evidence_failure(action, evidence) is None

    tampered_contract = evidence.model_copy(update={
        "values": {"series_analysis": {
            **evidence.values["series_analysis"],
            "analysis_contract": {
                **pair.evidence_contract(), "minimum_sample_size": 1,
            },
        }},
    })
    assert _evidence_failure(action, tampered_contract)[0] == "INVALID_DATA"

    invalid = evidence.model_copy(update={
        "values": {"series_analysis": {
            **evidence.values["series_analysis"], "spearman_rho_ci_95_approx": [0.1, 0.3],
        }},
    })
    assert _evidence_failure(action, invalid)[0] == "INVALID_DATA"


def test_scoped_chat_executes_only_the_registered_association_pair(monkeypatch):
    scope_payload = {
        "seller_id": "seller-1",
        "allowed_metrics": ["seller_order_bad_review_rate"],
        "default_start_date": "2024-01-01",
        "default_end_date": "2024-01-31",
        "metric_name": "seller_order_bad_review_rate",
        "granularity": "day",
    }
    scope_model = ControlledAnalysisScope.model_validate(scope_payload)
    from app.analysis.association_catalog import get_association_pair
    pair = get_association_pair("delivery_timing_bad_review_v1")
    assert pair is not None
    pair_contract = pair.evidence_contract()
    from app.metrics.authorization import authorize_metric
    from app.tenancy import SellerGrant, TenantContext

    monkeypatch.setattr(
        "app.mcp.client.current_tenant_context",
        lambda: TenantContext(
            tenant_id="public", user_id="test-user", role="analyst",
            scope_type="seller", seller_id="seller-1",
            seller_grants=(SellerGrant(seller_id="seller-1"),),
        ),
    )
    plan = MetricQueryPlan(
        query_type="metric_association",
        metric_name="seller_order_bad_review_rate",
        association_pair_id="delivery_timing_bad_review_v1",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    plan = plan.model_copy(update={
        "authorized_metric": authorize_metric(
            "seller_order_bad_review_rate",
            provenance_type="explorer_selection",
            provenance_id="association-test-selection",
            scope="seller",
            scope_data={**scope_model.model_dump(mode="json"), "tenant_id": None, "user_id": None, "permissions": None},
        ),
    })
    monkeypatch.setattr(
        "app.data.repositories.explorer.query_scoped_delivery_review_association",
        lambda *_args, **_kwargs: {
            "pair_id": "delivery_timing_bad_review_v1",
            "target_metric": "seller_order_bad_review_rate",
            "analysis_contract": pair_contract,
            "method_id": pair_contract["method_id"],
            "status": "ok",
            "sample_size": 110,
            "risk_difference": 0.2,
            "risk_difference_ci_95": [0.05, 0.35],
            "population_orders": 120,
            "eligible_orders": 110,
            "excluded_missing_pair_fields": 10,
            "missing_policy": "mock contract",
            "source_table": "analytics.v_order_unified",
            "limitations": ["association is not causation"],
        },
    )

    result = query_scoped_business_data("延迟与低评分的关联", scope_payload, plan=plan)

    assert result["tool_name"] == "query_scoped_registered_association"
    assert result["result"]["status"] == "ok"
    assert result["result"]["association_result"]["pair_id"] == "delivery_timing_bad_review_v1"
