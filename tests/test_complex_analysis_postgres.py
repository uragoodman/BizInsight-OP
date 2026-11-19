from __future__ import annotations

import os
from functools import wraps
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.api.auth import require_tenant_context
from app.analysis.templates import registry as template_registry
from app.complex_analysis import service
from app.core.config import Settings
from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG
from app.context.semantic_contract_v2 import CapabilityDecisionBasis, CapabilityReference, SealedRouteDecision
from app.data.database import engine
from app.graph import workflow
from app.graph.nodes import complex_plan as complex_plan_node
from app.graph.nodes import intent_router as intent_router_node
from app.graph.nodes import resolve_explorer_context as resolve_explorer_context_node
from app.graph.nodes import semantic_parse as semantic_parse_node
from app.graph.nodes import triage as triage_node
from app.main import app
from app.tenancy import SellerGrant, TenantContext


pytestmark = pytest.mark.skipif(
    os.environ.get("BIZINSIGHT_TEST_SHARD") != "postgres",
    reason="requires the isolated PostgreSQL shard",
)

_TEST_SUFFIX = uuid4().hex[:12]
_SELLER_ID = f"phase21-seller-{_TEST_SUFFIX}"
_TENANT_ID = f"phase21-complex-test-{_TEST_SUFFIX}"
_ANALYST_ID = f"phase21-complex-analyst-{_TEST_SUFFIX}"
_START_DATE = "2018-01-01"
_END_DATE = "2018-01-31"


@pytest.fixture(scope="module", autouse=True)
def _phase21_database_identity():
    """Seed the database authorization boundary for the generated test principal."""
    with engine.begin() as conn:
        # The isolated shard connects as its explicit admin account. Reset a
        # configured app role before seeding IAM rows and during teardown.
        conn.execute(text("RESET ROLE"))
        conn.execute(text("""INSERT INTO iam.tenants
            (tenant_id,tenant_name,scope_type,status)
            VALUES(:tenant,'Phase 2.1 integration test','seller','active')"""), {
            "tenant": _TENANT_ID,
        })
        conn.execute(text("""INSERT INTO iam.users
            (user_id,display_name,email,status)
            VALUES(:user,'Phase 2.1 integration analyst',:email,'active')"""), {
            "user": _ANALYST_ID,
            "email": f"{_ANALYST_ID}@example.invalid",
        })
        conn.execute(text("""INSERT INTO iam.tenant_memberships
            (tenant_id,user_id,role)
            VALUES(:tenant,:user,'analyst')"""), {
            "tenant": _TENANT_ID,
            "user": _ANALYST_ID,
        })
        conn.execute(text("""INSERT INTO iam.tenant_seller_accounts
            (tenant_id,seller_id)
            VALUES(:tenant,:seller)"""), {
            "tenant": _TENANT_ID,
            "seller": _SELLER_ID,
        })
        conn.execute(text("""INSERT INTO iam.membership_seller_grants
            (tenant_id,user_id,seller_id,access_start_date,access_end_date,granted_by)
            VALUES(:tenant,:user,:seller,'2018-01-01','2018-12-31',:actor)"""), {
            "tenant": _TENANT_ID,
            "user": _ANALYST_ID,
            "seller": _SELLER_ID,
            "actor": _ANALYST_ID,
        })
    try:
        yield
    finally:
        with engine.begin() as conn:
            conn.execute(text("RESET ROLE"))
            conn.execute(text("""DELETE FROM governance.action_plans
                WHERE tenant_id=:tenant AND user_id=:user"""), {
                "tenant": _TENANT_ID, "user": _ANALYST_ID,
            })
            conn.execute(text("""DELETE FROM governance.analysis_execution_audits
                WHERE tenant_id=:tenant AND user_id=:user"""), {
                "tenant": _TENANT_ID, "user": _ANALYST_ID,
            })
            conn.execute(text("""DELETE FROM governance.complex_analysis_plans
                WHERE tenant_id=:tenant AND user_id=:user"""), {
                "tenant": _TENANT_ID, "user": _ANALYST_ID,
            })
            conn.execute(text("""DELETE FROM governance.chat_turns
                WHERE tenant_id=:tenant AND user_id=:user"""), {
                "tenant": _TENANT_ID, "user": _ANALYST_ID,
            })
            conn.execute(text("""DELETE FROM governance.chat_sessions
                WHERE tenant_id=:tenant AND user_id=:user"""), {
                "tenant": _TENANT_ID, "user": _ANALYST_ID,
            })
            conn.execute(text("DELETE FROM iam.membership_seller_grants WHERE tenant_id=:tenant"), {
                "tenant": _TENANT_ID,
            })
            conn.execute(text("DELETE FROM iam.tenant_seller_accounts WHERE tenant_id=:tenant"), {
                "tenant": _TENANT_ID,
            })
            conn.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant"), {
                "tenant": _TENANT_ID,
            })
            conn.execute(text("DELETE FROM iam.tenants WHERE tenant_id=:tenant"), {
                "tenant": _TENANT_ID,
            })
            conn.execute(text("DELETE FROM iam.users WHERE user_id=:user"), {
                "user": _ANALYST_ID,
            })


def _context(*, role: str = "analyst", user_id: str = _ANALYST_ID, seller_id: str = _SELLER_ID) -> TenantContext:
    return TenantContext(
        tenant_id=_TENANT_ID,
        user_id=user_id,
        role=role,
        scope_type="seller",
        seller_id=seller_id,
        seller_grants=(SellerGrant(seller_id, "2018-01-01", "2018-12-31"),),
    )


def _scope_payload(seller_id: str = _SELLER_ID) -> dict[str, str]:
    return {
        "seller_id": seller_id,
        "metric_name": "gmv",
        "start_date": _START_DATE,
        "end_date": _END_DATE,
        "granularity": "day",
    }


def _route_decision(template_id: str = "cohort_retention") -> SealedRouteDecision:
    capability = next(
        item for item in DEFAULT_SEMANTIC_CATALOG.capabilities
        if item.template_id == template_id
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
        template_id=template_id,
        operation_ids=operation_ids,
        clause_ids=clause_ids,
        effective_arguments_digest=arguments_digest,
        decision_basis=basis,
    )


def _make_plan(context: TenantContext, *, turn_id: str, question: str = "Cohort 留存分析") -> dict:
    return service.create_plan_v2(
        context,
        question,
        _scope_payload(context.seller_id or _SELLER_ID),
        chat_turn_id=turn_id,
        template_name="cohort_retention",
        route_decision=_route_decision(),
    )


def _db_rows(context: TenantContext, query: str, params: dict) -> list[dict]:
    with engine.begin() as conn:
        service._set_scope(conn, context)
        return [dict(row) for row in conn.execute(text(query), params).mappings().all()]


def _patch_active_chat(monkeypatch) -> Settings:
    import app.core.config as core_config

    settings = Settings(
        _env_file=None,
        complex_route_mode="active",
        complex_route_canary_enabled=True,
        complex_route_canary_tenants=["*"],
        complex_route_canary_templates=[
            "rfm_segmentation", "cohort_retention", "customer_lifecycle",
        ],
        complex_route_canary_sample_rate=1.0,
        semantic_interpreter_mode="active",
        semantic_interpreter_send_raw_question=True,
        complex_chat_visible_templates=[],
        data_versioning_enabled=False,
    )
    for module in (
        core_config,
        workflow,
        intent_router_node,
        complex_plan_node,
        resolve_explorer_context_node,
        semantic_parse_node,
        template_registry,
    ):
        monkeypatch.setattr(module, "get_settings", lambda settings=settings: settings)
    monkeypatch.setattr(
        semantic_parse_node,
        "interpret_semantics",
        lambda *_args, **_kwargs: {
            "status": "unavailable",
            "mode": "active",
            "reason_code": "provider_unavailable",
        },
    )
    monkeypatch.setattr(triage_node, "llm_classify", lambda *_args, **_kwargs: "metric_query")
    return settings


def test_active_chat_fails_closed_without_a_v2_route_or_plan(monkeypatch) -> None:
    context = _context()
    _patch_active_chat(monkeypatch)
    previous = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        with TestClient(app) as client:
            response = client.post('/chat', json={
                'message': 'Cohort 留存分析',
                'use_current_explorer_context': True,
                'explorer_context': _scope_payload(),
                'chat_turn_id': 'phase0-no-route-turn',
            })
        assert response.status_code == 200, response.text
        body = response.json()
        assert body['clarification']['status'] == 'complex_analysis_unavailable'
        assert body['clarification']['reason_code'] == 'semantic_interpreter_failed'
        assert body.get('complex_plan') is None
        assert _db_rows(context, '''SELECT plan_id FROM governance.complex_analysis_plans
            WHERE tenant_id=:tenant AND user_id=:user''', {
            'tenant': context.tenant_id, 'user': context.user_id,
        }) == []
    finally:
        if previous is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous

def test_sealed_plan_creation_is_idempotent_per_turn_decision_and_scope() -> None:
    context = _context()
    first = _make_plan(context, turn_id="phase21-idempotency-turn")
    second = service.create_plan_v2(
        context,
        "Cohort 留存分析",
        _scope_payload(),
        chat_turn_id="phase21-idempotency-turn",
        template_name="cohort_retention",
        route_decision=_route_decision(),
    )
    assert first["plan_id"] == second["plan_id"]

    different_scope = _scope_payload()
    different_scope["granularity"] = "week"
    third = service.create_plan_v2(
        context,
        "Cohort 留存分析",
        different_scope,
        chat_turn_id="phase21-idempotency-turn",
        template_name="cohort_retention",
        route_decision=_route_decision(),
    )
    assert third["plan_id"] != first["plan_id"]
    assert third["idempotency_key"] != first["idempotency_key"]

    rows = _db_rows(context, """SELECT plan_id FROM governance.complex_analysis_plans
        WHERE chat_turn_id=:turn AND status='pending_confirmation'""", {
        "turn": "phase21-idempotency-turn",
    })
    assert {row["plan_id"] for row in rows} == {first["plan_id"], third["plan_id"]}


def test_plan_persists_contract_provenance() -> None:
    context = _context()
    route_decision = _route_decision()
    plan = service.create_plan_v2(
        context,
        "Cohort 留存分析",
        _scope_payload(),
        chat_turn_id="phase21-provenance-turn",
        template_name="cohort_retention",
        route_decision=route_decision,
    )

    [row] = _db_rows(context, """SELECT plan_data FROM governance.complex_analysis_plans
        WHERE plan_id=:plan""", {"plan": plan["plan_id"]})
    stored_plan = row["plan_data"]
    assert stored_plan["contract_version"] == "semantic-contract-v2.1"
    assert stored_plan["catalog_snapshot_hash"] == DEFAULT_SEMANTIC_CATALOG.snapshot_hash()
    assert stored_plan["route_decision_hash"] == service._digest(route_decision.model_dump(mode="json"))
    assert stored_plan["route_decision_hash"]
    assert stored_plan["idempotency_key"] == plan["idempotency_key"]


def test_revision_inherits_provenance_and_parent() -> None:
    context = _context()
    parent = _make_plan(context, turn_id="phase21-revision-provenance-turn")
    revision = service.revise_plan(context, parent["plan_id"], _scope_payload())

    assert revision["revision"] == 2
    assert revision["parent_plan_id"] == parent["plan_id"]
    [parent_row] = _db_rows(context, """SELECT status, plan_data
        FROM governance.complex_analysis_plans WHERE plan_id=:plan""", {
        "plan": parent["plan_id"],
    })
    [revision_row] = _db_rows(context, """SELECT status, revision, parent_plan_id, plan_data
        FROM governance.complex_analysis_plans WHERE plan_id=:plan""", {
        "plan": revision["plan_id"],
    })
    parent_data = parent_row["plan_data"]
    revision_data = revision_row["plan_data"]

    assert parent_row["status"] == "superseded"
    assert revision_row["revision"] == 2
    assert revision_row["parent_plan_id"] == parent["plan_id"]
    for field in ("contract_version", "catalog_snapshot_hash", "route_decision_hash"):
        assert revision_data[field] == parent_data[field]


def test_revising_an_expired_plan_persists_expired_status() -> None:
    context = _context()
    plan = _make_plan(context, turn_id="phase21-revise-expired-turn")
    with engine.begin() as conn:
        service._set_scope(conn, context)
        conn.execute(text("""UPDATE governance.complex_analysis_plans
            SET expires_at=now()-interval '1 minute' WHERE plan_id=:plan"""), {
            "plan": plan["plan_id"],
        })

    with pytest.raises(ValueError, match="expired"):
        service.revise_plan(context, plan["plan_id"], _scope_payload())

    [stored] = _db_rows(context, """SELECT status FROM governance.complex_analysis_plans
        WHERE plan_id=:plan""", {"plan": plan["plan_id"]})
    assert stored["status"] == "expired"


def test_direct_plan_api_requires_an_internal_sealed_route() -> None:
    context = _context()
    previous = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context
    payload = {
        "question": "Cohort 留存分析",
        "explorer_context": _scope_payload(),
    }
    try:
        with TestClient(app) as client:
            first = client.post("/complex-analysis/plans", json=payload)
            second = client.post("/complex-analysis/plans", json=payload)
        assert first.status_code == 409, first.text
        assert second.status_code == 409, second.text
        assert first.json()["detail"] == "sealed_route_decision_required"
    finally:
        if previous is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous


def test_execute_rejects_hidden_template_without_audit_or_template_call(monkeypatch) -> None:
    context = _context()
    plan = _make_plan(context, turn_id="phase21-hidden-template-turn")
    import app.analysis.templates.registry as registry

    monkeypatch.setattr(
        registry,
        "get_settings",
        lambda: Settings(_env_file=None, complex_chat_visible_templates=["rfm_segmentation"]),
    )
    calls: list[str] = []
    original = service.get_template

    def tracked_get_template(name: str):
        fn, display, required, keywords, description = original(name)
        @wraps(fn)
        def tracked(**kwargs):
            calls.append(name)
            return fn(**kwargs)
        return tracked, display, required, keywords, description

    monkeypatch.setattr(service, "get_template", tracked_get_template)
    with pytest.raises(ValueError, match="no longer visible"):
        service.execute_plan(context, plan["plan_id"])
    assert calls == []
    assert _db_rows(context, """SELECT execution_id FROM governance.analysis_execution_audits
        WHERE complex_plan_id=:plan""", {"plan": plan["plan_id"]}) == []


@pytest.mark.parametrize(
    "mutation",
    ["expired", "plan_hash", "code_hash", "legacy_baseline_missing", "scope_revoked", "wrong_owner"],
)
def test_execute_rechecks_plan_integrity_code_and_scope(monkeypatch, mutation: str) -> None:
    context = _context()
    plan = _make_plan(context, turn_id=f"phase21-{mutation}-turn")
    calls: list[str] = []
    original = service.get_template

    def tracked_get_template(name: str):
        fn, display, required, keywords, description = original(name)
        @wraps(fn)
        def tracked(**kwargs):
            calls.append(name)
            return fn(**kwargs)
        return tracked, display, required, keywords, description

    monkeypatch.setattr(service, "get_template", tracked_get_template)
    if mutation == "expired":
        with engine.begin() as conn:
            service._set_scope(conn, context)
            conn.execute(text("UPDATE governance.complex_analysis_plans SET expires_at=now()-interval '1 minute' WHERE plan_id=:plan"), {"plan": plan["plan_id"]})
        execute_context = context
        expected = "expired"
    elif mutation == "plan_hash":
        with engine.begin() as conn:
            service._set_scope(conn, context)
            conn.execute(text("UPDATE governance.complex_analysis_plans SET plan_hash='tampered' WHERE plan_id=:plan"), {"plan": plan["plan_id"]})
        execute_context = context
        expected = "integrity validation failed"
    elif mutation == "code_hash":
        monkeypatch.setattr(service, "_template_code_hash", lambda _fn: "f" * 64)
        execute_context = context
        expected = "code changed"
    elif mutation == "legacy_baseline_missing":
        with engine.begin() as conn:
            service._set_scope(conn, context)
            conn.execute(text("""UPDATE governance.complex_analysis_plans
                SET plan_data=plan_data - 'template_code_hash' WHERE plan_id=:plan"""), {
                "plan": plan["plan_id"],
            })
        execute_context = context
        expected = "code baseline is missing"
    elif mutation == "wrong_owner":
        execute_context = _context(user_id="phase21-other-user")
        expected = "not found"
    else:
        execute_context = _context(seller_id="phase21-other-seller")
        expected = "not authorized"

    with pytest.raises((PermissionError, ValueError), match=expected):
        service.execute_plan(execute_context, plan["plan_id"])
    assert calls == []
    assert _db_rows(context, """SELECT execution_id FROM governance.analysis_execution_audits
        WHERE complex_plan_id=:plan""", {"plan": plan["plan_id"]}) == []
    if mutation in {"expired", "plan_hash"}:
        [stored] = _db_rows(context, """SELECT status FROM governance.complex_analysis_plans
            WHERE plan_id=:plan""", {"plan": plan["plan_id"]})
        assert stored["status"] == ("expired" if mutation == "expired" else "failed")
