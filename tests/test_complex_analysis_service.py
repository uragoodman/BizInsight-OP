import pytest
from types import SimpleNamespace
from datetime import datetime, timezone

from app.complex_analysis import service as service_module
from app.complex_analysis.service import _fingerprint, _legacy_fingerprint, _preferences, _recommendations, focus_options
from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG
from app.context.semantic_contract_v2 import CapabilityDecisionBasis, CapabilityReference, SealedRouteDecision


def _sealed_cohort_decision() -> SealedRouteDecision:
    capability = next(
        item for item in DEFAULT_SEMANTIC_CATALOG.capabilities
        if item.capability_id == "customer_cohort_retention"
    )
    catalog_hash = DEFAULT_SEMANTIC_CATALOG.snapshot_hash()
    operation_ids = ("op-trend",)
    clause_ids = ("clause-1",)
    arguments_digest = "a" * 64
    basis = CapabilityDecisionBasis(
        contract_version="semantic-contract-v2.1",
        catalog_snapshot_hash=catalog_hash,
        operation_ids=operation_ids,
        clause_ids=clause_ids,
        capability_ids=(CapabilityReference(capability_id=capability.capability_id),),
        effective_arguments_digest=arguments_digest,
    )
    return SealedRouteDecision.issue(
        contract_version="semantic-contract-v2.1",
        catalog_snapshot_hash=catalog_hash,
        route_decision_status="SELECTED",
        route_id=capability.route_id,
        capability_id=capability.capability_id,
        template_id=capability.template_id,
        operation_ids=operation_ids,
        clause_ids=clause_ids,
        effective_arguments_digest=arguments_digest,
        decision_basis=basis,
    )


def test_correlation_action_requires_significance_and_sample_size():
    weak = {"status": "ok", "sample_size": 50, "chi_square": {"p_value": 0.2}}
    strong = {"status": "ok", "sample_size": 50, "chi_square": {"p_value": 0.01}}
    assert _recommendations("delay_review_correlation", weak, "e") == []
    assert _recommendations("delay_review_correlation", strong, "e")


def test_pareto_action_requires_multiple_nonzero_entities():
    assert _recommendations("pareto_analysis", {"status": "ok", "items": [{}]}, "e") == []
    assert _recommendations("pareto_analysis", {"status": "ok", "items": [{}, {}]}, "e")


def test_human_preferences_are_template_controlled_and_note_limited():
    options = focus_options("rfm_segmentation")
    assert {item["key"] for item in options} >= {"high_value", "at_risk", "growth"}
    assert _preferences("rfm_segmentation", {"focus_key": "at_risk", "note": "优先看流失风险"})["focus_key"] == "at_risk"
    with pytest.raises(ValueError, match="not available"):
        _preferences("rfm_segmentation", {"focus_key": "unregistered"})
    with pytest.raises(ValueError, match="280"):
        _preferences("rfm_segmentation", {"note": "x" * 281})


def test_plan_fingerprint_binds_preferences_but_old_plan_hash_remains_distinct():
    scope, params = {"seller_id": "s"}, {"start_date": "2018-01-01", "end_date": "2018-01-31"}
    assert _fingerprint("rfm_segmentation", scope, params, {"focus_key": "at_risk"}) != _fingerprint("rfm_segmentation", scope, params, {"focus_key": "growth"})
    assert _legacy_fingerprint("rfm_segmentation", scope, params) != _fingerprint("rfm_segmentation", scope, params, {})


def test_typed_sealed_plan_creation_reuses_pending_plan_for_same_request(monkeypatch):
    pending_plans = []
    scope = {
        "seller_id": "seller-1",
        "default_start_date": "2018-01-01",
        "default_end_date": "2018-01-31",
        "metric_name": "gmv",
        "granularity": "day",
    }

    class _Result:
        def __init__(self, rows):
            self._rows = rows

        def mappings(self):
            return self

        def all(self):
            return [{"plan_data": dict(plan)} for plan in self._rows]

    class _Connection:
        def execute(self, statement, _params=None):
            if "SELECT plan_data" in str(statement):
                return _Result(pending_plans)
            return None

    class _Begin:
        def __enter__(self):
            return _Connection()

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(service_module, "_scope", lambda _context, _payload: scope)
    monkeypatch.setattr(service_module.engine, "begin", lambda: _Begin())
    monkeypatch.setattr(
        service_module,
        "_insert_plan_v2",
        lambda _conn, _context, plan, *_args, **_kwargs: pending_plans.append(plan),
    )
    context = SimpleNamespace(tenant_id="tenant-1", user_id="user-1")
    payload = {
        "seller_id": "seller-1",
        "start_date": "2018-01-01",
        "end_date": "2018-01-31",
        "metric_name": "gmv",
        "granularity": "day",
    }
    route_decision = _sealed_cohort_decision()

    first = service_module.create_plan_v2(
        context, "Cohort 留存分析", payload,
        chat_turn_id="turn-1",
        template_name="cohort_retention",
        route_decision=route_decision,
    )
    second = service_module.create_plan_v2(
        context, "Cohort 留存分析", payload,
        chat_turn_id="turn-1",
        template_name="cohort_retention",
        route_decision=route_decision,
    )

    assert first["plan_id"] == second["plan_id"]
    assert len(pending_plans) == 1
    assert first["contract_version"] == route_decision.contract_version
    assert first["catalog_snapshot_hash"] == route_decision.catalog_snapshot_hash
    assert first["route_decision_hash"] == service_module._digest(route_decision.model_dump(mode="json"))
    assert first["idempotency_key"] == pending_plans[0]["idempotency_key"]


def test_plan_service_rejects_missing_or_untyped_route_decisions_before_database(monkeypatch):
    monkeypatch.setattr(
        service_module.engine,
        "begin",
        lambda: (_ for _ in ()).throw(AssertionError("must reject before database access")),
    )
    with pytest.raises(ValueError, match="sealed route_decision is required"):
        service_module.create_plan_v2(None, "any question", {}, route_decision=None)
    with pytest.raises(ValueError, match="typed server-sealed"):
        service_module.create_plan_v2(
            None,
            "any question",
            {},
            route_decision={"route_decision_status": "SELECTED"},
        )
    with pytest.raises(ValueError, match="Direct natural-language plan creation is disabled"):
        service_module.create_plan(None, "any question", {})


def test_get_plan_v2_includes_top_level_trace_id(monkeypatch):
    row = {
        "plan_id": "plan-1",
        "plan_data": {"plan_id": "plan-1", "template_name": "cohort_retention"},
        "status": "pending_confirmation",
        "expires_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "parent_plan_id": None,
        "revision": 1,
        "preferences_data": {},
        "template_name": "cohort_retention",
        "trace_id": "trace-from-row",
    }

    class Result:
        def __init__(self, value):
            self.value = value

        def mappings(self):
            return self

        def one_or_none(self):
            return self.value

    class Connection:
        def execute(self, statement, _params=None):
            sql = str(statement)
            if "FROM governance.complex_analysis_plans WHERE plan_id" in sql:
                return Result(row)
            if "FROM governance.analysis_execution_audits" in sql:
                return Result(None)
            return Result(None)

    class Begin:
        def __enter__(self):
            return Connection()

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(service_module.engine, "begin", lambda: Begin())
    context = SimpleNamespace(tenant_id="tenant-1", user_id="user-1", scope_type="seller")

    plan = service_module.get_plan_v2(context, "plan-1")

    assert plan is not None
    assert plan["trace_id"] == "trace-from-row"
