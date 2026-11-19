from __future__ import annotations

from copy import deepcopy

from app.auth.context import AuthorizationContext
from app.context.compiler import compile_turn_context
from app.context.semantic_verifier import risk_tier_for_compilation, trusted_authorization_preflight
from app.core.config import Settings
from app.tenancy import SellerGrant, TenantContext, activate_tenant_context


def _state(question: str, **extra):
    return {
        "user_message": question,
        "trace_id": "p3-shadow-test",
        "preference_context": {"source_needs": [], "candidate_analysis_plan": None},
        **extra,
    }


def _tenant(*, seller_grants=(), seller_id=None, scope_type="platform", role="admin"):
    return TenantContext(
        tenant_id="p3-tenant",
        user_id="p3-user",
        role=role,
        scope_type=scope_type,
        seller_id=seller_id,
        seller_grants=tuple(seller_grants),
    )


def _authz(tenant: TenantContext, *, dimensions=("product_category",)):
    return AuthorizationContext(
        tenant_id=tenant.tenant_id,
        user_id=tenant.user_id,
        role=tenant.role,
        scope_type=tenant.scope_type,
        permission_version="p3-permissions-v1",
        policy_set_version="p3-policy-v1",
        authz_ref="authz:p3-test",
        authz_digest="p3-test-digest",
        allowed_seller_ids=tenant.allowed_seller_ids,
        seller_grants=tenant.seller_grants,
        allowed_metric_ids=("gmv", "order_count"),
        allowed_dimensions=tuple(dimensions),
        allowed_actions=("analyze",),
        allowed_tools=("query_business_data", "query_scoped_business_data"),
    )


def test_shadow_verifier_does_not_change_compiler_route_answer_or_contract(monkeypatch):
    question = "按商品品类范围的 GMV"
    off_settings = Settings(semantic_verifier_mode="off")
    shadow_settings = Settings(
        semantic_verifier_mode="shadow",
        semantic_verifier_send_raw_question=True,
    )
    tenant = _tenant()
    authz = _authz(tenant)
    monkeypatch.setattr("app.core.config.get_settings", lambda: off_settings)
    with activate_tenant_context(tenant):
        off_state = compile_turn_context(_state(
            question, answer="stable answer", authorization_context=authz,
        ))
    off_compiled = deepcopy(off_state["context_compilation"])

    calls = []

    def fake_model(_system, request, _model):
        calls.append(request)
        assert "question" not in request
        return {
            "roles": [{
                "id": "m1", "start": 1, "end": 5,
                "role": "reference", "reason_code": "scope_reference",
            }],
            "abstain": False,
        }, {"input_tokens": 12, "output_tokens": 4, "model": "offline-stub"}

    monkeypatch.setattr("app.core.config.get_settings", lambda: shadow_settings)
    monkeypatch.setattr("app.context.semantic_verifier._default_invoke", fake_model)
    with activate_tenant_context(tenant):
        shadow_state = compile_turn_context(_state(
            question, answer="stable answer", authorization_context=authz,
        ))

    assert len(calls) == 1
    assert set(calls[0]) == {
        "mentions", "proposed_roles", "eligible_mention_ids", "allowed_roles", "reason_codes",
    }
    assert not {"scope", "seller", "seller_id", "seller_ids", "date", "metric", "metric_id", "sql", "gold_roles"} & set(calls[0])
    shadow_compiled = shadow_state["context_compilation"]
    assert shadow_compiled == off_compiled
    observation = shadow_state["semantic_verifier_observation"]
    assert observation["status"] == "success"
    assert observation["applied"] is False
    assert observation["input_tokens"] == 12
    assert observation["output_tokens"] == 4
    assert shadow_state["answer"] == off_state["answer"] == "stable answer"


def test_off_mode_keeps_p2_route_and_makes_no_provider_call(monkeypatch):
    settings = Settings(semantic_verifier_mode="off", semantic_verifier_send_raw_question=True)
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    monkeypatch.setattr(
        "app.context.semantic_verifier._default_invoke",
        lambda *_: (_ for _ in ()).throw(AssertionError("verifier must not be called")),
    )
    result = compile_turn_context(_state("按店铺范围的 GMV"))
    assert result["context_compilation"]["clarification_reason"] == "role_conflict"
    assert "semantic_verifier_observation" not in result


def test_authorization_rejection_skips_verifier(monkeypatch):
    settings = Settings(
        semantic_verifier_mode="shadow",
        semantic_verifier_send_raw_question=True,
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)

    tenant = _tenant()
    authz = _authz(tenant, dimensions=())
    calls = []
    monkeypatch.setattr(
        "app.context.semantic_verifier._default_invoke",
        lambda *args: calls.append(args),
    )
    with activate_tenant_context(tenant):
        result = compile_turn_context(_state(
            "按商品品类范围的 GMV", authorization_context=authz,
        ))
    assert calls == []
    observation = result["semantic_verifier_observation"]
    assert observation["risk_tier"] == "high"
    assert observation["reason_code"] == "risk_not_eligible"


def test_scope_out_of_bounds_skips_verifier(monkeypatch):
    settings = Settings(
        semantic_verifier_mode="shadow",
        semantic_verifier_send_raw_question=True,
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    calls = []
    monkeypatch.setattr(
        "app.context.semantic_verifier._default_invoke",
        lambda *args: calls.append(args),
    )
    result = compile_turn_context(_state(
        "按商品品类范围的 GMV",
        controlled_analysis_scope={"allowed_dimensions": ["seller"]},
    ))
    assert calls == []
    assert result["semantic_verifier_observation"]["risk_tier"] == "high"


def test_locked_date_boundary_skips_verifier(monkeypatch):
    settings = Settings(
        semantic_verifier_mode="shadow",
        semantic_verifier_send_raw_question=True,
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    calls = []
    monkeypatch.setattr(
        "app.context.semantic_verifier._default_invoke",
        lambda *args: calls.append(args),
    )
    result = compile_turn_context(_state(
        "按商品品类拆解 2024-01-01 GMV",
        controlled_analysis_scope={
            "allowed_dimensions": ["product_category"],
            "default_start_date": "2018-08-01",
            "default_end_date": "2018-08-31",
        },
    ))
    assert calls == []
    assert result["context_compilation"]["clarification_reason"] == "date_outside_locked_scope"
    assert result["semantic_verifier_observation"]["risk_tier"] == "high"


def _seller_compiled(*, date_value=None):
    compiled = {
        "dimension_roles": [{
            "dimension": "product_category",
            "start": 1,
            "end": 5,
            "role": "grouping",
            "confidence_class": "open",
            "evidence": ["group_prefix:按", "group_suffix:拆解"],
        }],
        "known_slots": {"metric": {"value": None}},
        "operation_intent": {"kind": "ordinary_metric_query"},
    }
    if date_value is not None:
        compiled["known_slots"]["date"] = {"value": date_value}
    return compiled


def test_seller_not_in_real_grant_skips_verifier(monkeypatch):
    from app.context.semantic_verifier import observe_compilation

    tenant = _tenant(
        scope_type="seller",
        role="analyst",
        seller_id="seller-a",
        seller_grants=(SellerGrant("seller-a", "2024-01-01", "2024-12-31"),),
    )
    authz = _authz(tenant)
    state = {
        "authorization_context": authz,
        "controlled_analysis_scope": {
            "seller_id": "seller-b",
            "default_start_date": "2024-01-01",
            "default_end_date": "2024-12-31",
            "allowed_dimensions": ["product_category"],
        },
        "semantic_verifier_risk_tier": "elevated",
    }
    calls = []
    monkeypatch.setattr("app.core.config.get_settings", lambda: Settings(
        semantic_verifier_mode="shadow", semantic_verifier_send_raw_question=True,
    ))

    monkeypatch.setattr("app.context.semantic_verifier._default_invoke", lambda *args: calls.append(args))

    with activate_tenant_context(tenant):
        observed = observe_compilation(state, _seller_compiled(), "按商品品类拆解 GMV")

    assert calls == []
    assert observed["risk_tier"] == "high"
    assert observed["reason_code"] == "risk_not_eligible"


def test_seller_date_outside_real_grant_skips_verifier(monkeypatch):
    from app.context.semantic_verifier import observe_compilation

    tenant = _tenant(
        scope_type="seller",
        role="analyst",
        seller_id="seller-a",
        seller_grants=(SellerGrant("seller-a", "2024-02-01", "2024-12-31"),),
    )
    authz = _authz(tenant)
    state = {
        "authorization_context": authz,
        "controlled_analysis_scope": {
            "seller_id": "seller-a",
            "default_start_date": "2024-01-01",
            "default_end_date": "2024-12-31",
            "allowed_dimensions": ["product_category"],
        },
        "semantic_verifier_risk_tier": "elevated",
    }
    calls = []
    monkeypatch.setattr("app.core.config.get_settings", lambda: Settings(
        semantic_verifier_mode="shadow", semantic_verifier_send_raw_question=True,
    ))

    monkeypatch.setattr("app.context.semantic_verifier._default_invoke", lambda *args: calls.append(args))

    with activate_tenant_context(tenant):
        observed = observe_compilation(state, _seller_compiled(), "按商品品类拆解 GMV")

    assert calls == []
    assert observed["risk_tier"] == "high"
    assert observed["reason_code"] == "risk_not_eligible"


def test_valid_real_seller_grant_and_date_pass_shared_preflight(monkeypatch):
    from app.context.semantic_verifier import observe_compilation

    tenant = _tenant(
        scope_type="seller",
        role="analyst",
        seller_id="seller-a",
        seller_grants=(SellerGrant("seller-a", "2024-01-01", "2024-12-31"),),
    )
    authz = _authz(tenant)
    scope = {
        "seller_id": "seller-a",
        "default_start_date": "2024-03-01",
        "default_end_date": "2024-03-31",
        "allowed_dimensions": ["product_category"],
    }
    state = {
        "authorization_context": authz,
        "controlled_analysis_scope": scope,
        "semantic_verifier_risk_tier": "elevated",
    }
    compiled = _seller_compiled(date_value="2024-03-15")
    calls = []

    with activate_tenant_context(tenant):
        allowed, reason, resolved_authz = trusted_authorization_preflight(compiled, state)
    assert allowed is True, reason
    assert reason is None
    assert resolved_authz is authz

    monkeypatch.setattr("app.core.config.get_settings", lambda: Settings(
        semantic_verifier_mode="shadow", semantic_verifier_send_raw_question=True,
    ))

    def fake_model(_system, request, _model):
        calls.append(request)
        return {
            "roles": [{
                "id": "m1", "start": 1, "end": 5,
                "role": "grouping", "reason_code": "explicit_grouping",
            }],
            "abstain": False,
        }

    monkeypatch.setattr("app.context.semantic_verifier._default_invoke", fake_model)
    with activate_tenant_context(tenant):
        observed = observe_compilation(state, compiled, "按商品品类拆解 GMV")

    assert len(calls) == 1
    assert observed["status"] == "success"


def test_signed_seller_scope_payload_rechecks_real_grant_and_date(monkeypatch):
    from dataclasses import replace
    from datetime import date

    from app.context.semantic_verifier import observe_compilation
    from app.scope.seller_scope import (
        SellerScopeIntent,
        issue_seller_scope_payload,
        resolve_seller_scope,
        verify_seller_scope_payload,
    )

    tenant = _tenant(
        scope_type="seller",
        role="analyst",
        seller_id="seller-a",
        seller_grants=(SellerGrant("seller-a", "2024-01-01", "2024-12-31"),),
    )
    authz = replace(
        _authz(tenant, dimensions=("seller", "product_category")),
        allowed_actions=("read", "analyze"),
    )
    settings = Settings(
        semantic_verifier_mode="shadow",
        semantic_verifier_send_raw_question=True,
        seller_dimension_enabled=True,
        tenant_auth_secret="p3-test-only-scope-signing-secret",
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    payload = issue_seller_scope_payload(
        SellerScopeIntent(
            seller_scope_type="single",
            seller_ids=("seller-a",),
            start_date=date(2024, 3, 1),
            end_date=date(2024, 3, 31),
            metric_ids=("gmv",),
            dimensions=("seller",),
            granularity="day",
        ),
        tenant=tenant,
    )
    state = {
        "authorization_context": authz,
        "seller_scope_payload": payload,
        "semantic_verifier_risk_tier": "elevated",
    }
    compiled = {
        "dimension_roles": [{
            "dimension": "seller", "role": "grouping", "start": 1, "end": 3,
            "confidence_class": "open", "evidence": ["group_prefix:按"],
        }],
        "known_slots": {
            "metric": {"value": "gmv"},
            "date": {"value": "2024-03-15"},
        },
        "operation_intent": {"kind": "ordinary_metric_query"},
    }
    calls = []
    monkeypatch.setattr(
        "app.context.semantic_verifier._default_invoke",
        lambda *_args: calls.append(_args) or {
            "roles": [{
                "id": "m1", "start": 1, "end": 3,
                "role": "grouping", "reason_code": "explicit_grouping",
            }],
            "abstain": False,
        },
    )

    with activate_tenant_context(tenant):
        intent = verify_seller_scope_payload(payload, tenant=tenant)
        resolve_seller_scope(intent, tenant=tenant, authz=authz)
        allowed, reason, resolved_authz = trusted_authorization_preflight(compiled, state)
        observed = observe_compilation(state, compiled, "按店铺拆解 GMV")

    assert allowed is True, reason
    assert reason is None
    assert resolved_authz is authz
    assert len(calls) == 1
    assert observed["status"] == "success"


def test_signed_seller_scope_outside_real_grant_skips_verifier(monkeypatch):
    from dataclasses import replace
    from datetime import date

    from app.context.semantic_verifier import observe_compilation
    from app.scope.seller_scope import SellerScopeIntent, issue_seller_scope_payload

    tenant = _tenant(
        scope_type="seller",
        role="analyst",
        seller_id="seller-a",
        seller_grants=(SellerGrant("seller-a", "2024-04-01", "2024-12-31"),),
    )
    authz = replace(
        _authz(tenant, dimensions=("seller", "product_category")),
        allowed_actions=("read", "analyze"),
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: Settings(
        semantic_verifier_mode="shadow",
        semantic_verifier_send_raw_question=True,
        seller_dimension_enabled=True,
        tenant_auth_secret="p3-test-only-scope-signing-secret",
    ))
    payload = issue_seller_scope_payload(
        SellerScopeIntent(
            seller_scope_type="single",
            seller_ids=("seller-a",),
            start_date=date(2024, 3, 1),
            end_date=date(2024, 3, 31),
            metric_ids=("gmv",),
            dimensions=("seller",),
            granularity="day",
        ),
        tenant=tenant,
    )
    state = {
        "authorization_context": authz,
        "seller_scope_payload": payload,
        "semantic_verifier_risk_tier": "elevated",
    }
    compiled = {
        "dimension_roles": [{
            "dimension": "seller", "role": "grouping", "start": 1, "end": 3,
            "confidence_class": "open",
        }],
        "known_slots": {
            "metric": {"value": "gmv"},
            "date": {"value": "2024-03-15"},
        },
        "operation_intent": {"kind": "ordinary_metric_query"},
    }
    calls = []
    monkeypatch.setattr(
        "app.context.semantic_verifier._default_invoke",
        lambda *args: calls.append(args),
    )

    with activate_tenant_context(tenant):
        observed = observe_compilation(state, compiled, "按店铺拆解 GMV")

    assert calls == []
    assert observed["risk_tier"] == "high"
    assert observed["reason_code"] == "risk_not_eligible"


def test_explicit_date_outside_controlled_seller_scope_skips_verifier(monkeypatch):
    from app.context.semantic_verifier import observe_compilation

    tenant = _tenant(
        scope_type="seller",
        role="analyst",
        seller_id="seller-a",
        seller_grants=(SellerGrant("seller-a", "2024-01-01", "2024-12-31"),),
    )
    state = {
        "authorization_context": _authz(tenant),
        "controlled_analysis_scope": {
            "seller_id": "seller-a",
            "default_start_date": "2024-03-01",
            "default_end_date": "2024-03-31",
            "allowed_dimensions": ["product_category"],
        },
        "semantic_verifier_risk_tier": "elevated",
    }
    calls = []
    monkeypatch.setattr("app.core.config.get_settings", lambda: Settings(
        semantic_verifier_mode="shadow", semantic_verifier_send_raw_question=True,
    ))
    monkeypatch.setattr("app.context.semantic_verifier._default_invoke", lambda *args: calls.append(args))

    with activate_tenant_context(tenant):
        observed = observe_compilation(
            state,
            _seller_compiled(date_value="2024-04-01"),
            "按商品品类拆解 GMV",
        )

    assert calls == []
    assert observed["risk_tier"] == "high"
    assert observed["reason_code"] == "risk_not_eligible"


def test_platform_admin_controlled_single_seller_keeps_existing_bypass():
    tenant = _tenant()
    authz = _authz(tenant)
    state = {
        "authorization_context": authz,
        "controlled_analysis_scope": {
            "seller_id": "seller-without-individual-grant",
            "default_start_date": "2024-03-01",
            "default_end_date": "2024-03-31",
            "allowed_dimensions": ["product_category"],
        },
    }
    with activate_tenant_context(tenant):
        allowed, reason, resolved_authz = trusted_authorization_preflight(
            _seller_compiled(date_value="2024-03-15"), state,
        )

    assert allowed is True
    assert reason is None
    assert resolved_authz is authz


def test_missing_trusted_tenant_does_not_call_verifier(monkeypatch):
    from app.context.semantic_verifier import observe_compilation

    state = {
        "authorization_context": _authz(_tenant()),
        "semantic_verifier_risk_tier": "elevated",
    }
    calls = []
    monkeypatch.setattr("app.core.config.get_settings", lambda: Settings(
        semantic_verifier_mode="shadow", semantic_verifier_send_raw_question=True,
    ))
    monkeypatch.setattr("app.context.semantic_verifier._default_invoke", lambda *args: calls.append(args))

    observed = observe_compilation(state, _seller_compiled(), "按商品品类拆解 GMV")

    assert calls == []
    assert observed["risk_tier"] == "high"
    assert observed["reason_code"] == "risk_not_eligible"


def test_high_risk_tier_wins_over_earlier_open_assignments():
    tenant = _tenant()
    authz = _authz(tenant)
    compiled = {
        "dimension_roles": [
            {"dimension": "product_category", "role": "grouping", "risk_tier": "open"},
            {"dimension": "product_category", "role": "reference", "risk_tier": "high"},
        ],
    }
    with activate_tenant_context(tenant):
        assert risk_tier_for_compilation(compiled, {
            "authorization_context": authz,
            "semantic_verifier_risk_tier": "open",
        }) == "high"


def test_low_risk_closed_role_skips_verifier(monkeypatch):
    settings = Settings(
        semantic_verifier_mode="shadow",
        semantic_verifier_send_raw_question=True,
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    calls = []
    monkeypatch.setattr(
        "app.context.semantic_verifier._default_invoke",
        lambda *args: calls.append(args),
    )
    tenant = _tenant()
    with activate_tenant_context(tenant):
        result = compile_turn_context(_state(
            "按商品品类拆解 GMV", authorization_context=_authz(tenant),
        ))
    assert calls == []
    assert result["semantic_verifier_observation"]["risk_tier"] == "low"
    assert result["context_compilation"]["known_slots"]["dimension"]["value"] == "product_category"


def test_shadow_provider_failure_keeps_deterministic_route(monkeypatch):
    from app.context import semantic_verifier

    semantic_verifier._CACHE.clear()
    settings = Settings(
        semantic_verifier_mode="shadow",
        semantic_verifier_send_raw_question=True,
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)

    monkeypatch.setattr(
        "app.context.semantic_verifier._default_invoke",
        lambda *_: (_ for _ in ()).throw(TimeoutError("provider timeout")),
    )
    tenant = _tenant()
    authz = _authz(tenant)
    with activate_tenant_context(tenant):
        result = compile_turn_context(_state(
            "按商品品类范围的 GMV", authorization_context=authz,
        ))
    assert result["semantic_verifier_observation"]["status"] == "provider_error"
    assert result["semantic_verifier_observation"]["applied"] is False
    assert result["context_compilation"]["clarification_reason"] == "role_conflict"


def test_open_only_never_routes_conflict_or_elevated_to_model(monkeypatch):
    settings = Settings(
        semantic_verifier_mode="open_only",
        semantic_verifier_send_raw_question=True,
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    calls = []
    monkeypatch.setattr(
        "app.context.semantic_verifier._default_invoke",
        lambda *args: calls.append(args),
    )
    result = compile_turn_context(_state("按店铺范围的 GMV"))
    assert calls == []
    assert result["semantic_verifier_observation"]["risk_tier"] == "high"
    assert result["semantic_verifier_observation"]["reason_code"] == "risk_not_eligible"
    assert result["context_compilation"]["clarification_reason"] == "role_conflict"


def test_elevated_closed_mention_is_sent_to_shadow_verifier(monkeypatch):
    settings = Settings(
        semantic_verifier_mode="shadow",
        semantic_verifier_send_raw_question=True,
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)

    calls = []

    def fake_model(_system, request, _model):
        calls.append(request)
        roles = [
            {
                "id": mention["id"],
                "start": mention["start"],
                "end": mention["end"],
                "role": "grouping",
                "reason_code": "explicit_grouping",
            }
            for mention in request["mentions"]
            if mention["id"] in request["eligible_mention_ids"]
        ]
        return {"roles": roles, "abstain": False}, {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr("app.context.semantic_verifier._default_invoke", fake_model)
    state = _state(
        "按商品品类拆解 GMV",
        semantic_verifier_risk_tier="elevated",
    )
    tenant = _tenant()
    state["authorization_context"] = _authz(tenant)
    with activate_tenant_context(tenant):
        result = compile_turn_context(state)
    observation = result["semantic_verifier_observation"]

    assert len(calls) == 1
    assert observation["status"] == "success"
    assert observation["risk_tier"] == "elevated"
    assert observation["span_ids"]
    assert observation["applied"] is False


def test_shadow_does_not_change_p0_p1_p2_core_compilations(monkeypatch):
    from pathlib import Path

    from evaluation.dimension_role_audit import _load_cases, _state_for

    dataset = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "dimension_role_cases_v1.jsonl"
    cases = [case for case in _load_cases(dataset) if case.get("tier") == "core"]
    off = Settings(semantic_verifier_mode="off")
    shadow = Settings(
        semantic_verifier_mode="shadow",
        semantic_verifier_send_raw_question=False,
    )

    monkeypatch.setattr("app.core.config.get_settings", lambda: off)
    off_compiled = {
        case["case_id"]: compile_turn_context(_state_for(case))["context_compilation"]
        for case in cases
    }

    monkeypatch.setattr("app.core.config.get_settings", lambda: shadow)
    for case in cases:
        shadow_state = compile_turn_context(_state_for(case))
        assert shadow_state["context_compilation"] == off_compiled[case["case_id"]]
