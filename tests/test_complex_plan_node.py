from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.complex_analysis import service
from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG
from app.context.semantic_contract_v2 import CapabilityDecisionBasis, CapabilityReference, SealedRouteDecision
from app.graph import workflow
from app.graph.nodes import complex_clarification as clarification_node
from app.graph.nodes import complex_plan as plan_node


def _decision() -> SealedRouteDecision:
    capability = next(
        item for item in DEFAULT_SEMANTIC_CATALOG.capabilities
        if item.capability_id == "customer_rfm_segmentation"
    )
    catalog_hash = DEFAULT_SEMANTIC_CATALOG.snapshot_hash()
    operation_ids = ("op-segment",)
    clause_ids = ("clause-1",)
    arguments_digest = "b" * 64
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


def _authorize_canary_plan(monkeypatch, state: dict) -> list[str]:
    decision = state["complex_route_decision"]
    settings = SimpleNamespace(
        complex_route_mode="active",
        complex_route_canary_enabled=True,
        complex_route_canary_tenants=["tenant-1"],
        complex_route_canary_templates=["rfm_segmentation"],
        complex_route_canary_sample_rate=1.0,
        complex_route_canary_salt="s" * 32,
        semantic_interpreter_mode="active",
        semantic_interpreter_send_raw_question=True,
    )
    monkeypatch.setattr(plan_node, "get_settings", lambda: settings)
    monkeypatch.setattr(
        plan_node,
        "current_tenant_context",
        lambda: SimpleNamespace(tenant_id="tenant-1", user_id="user-1"),
    )
    validated_json: list[str] = []
    monkeypatch.setattr(
        plan_node,
        "SemanticContractV21",
        SimpleNamespace(
            model_validate_json=lambda payload: validated_json.append(payload) or object(),
        ),
    )
    monkeypatch.setattr(plan_node, "route_semantic_contract", lambda _contract: object())
    monkeypatch.setattr(
        plan_node,
        "issue_sealed_route_decision",
        lambda *_args, **_kwargs: decision,
    )
    state["complex_route_canary_selected"] = True
    state["complex_route_canary_observation"] = {"eligible": True}
    state["semantic_interpreter_observation"] = {"status": "ok"}
    state["semantic_contract"] = {"fixture": "validated"}
    return validated_json


def test_active_template_plan_creates_only_a_pending_plan(monkeypatch) -> None:
    context = SimpleNamespace(tenant_id="tenant-1", user_id="user-1")
    calls = []
    plan = {"plan_id": "plan-1", "template_name": "rfm_segmentation", "status": "pending_confirmation"}
    monkeypatch.setattr(plan_node, "create_plan_v2", lambda *args, **kwargs: calls.append((args, kwargs)) or plan)
    monkeypatch.setattr(service, "execute_plan", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not execute")))
    state = {
        "user_message": "按购买频次和金额给客户分层",
        "chat_session_id": "session-1",
        "chat_turn_id": "turn-1",
        "use_current_explorer_context": True,
        "controlled_analysis_scope": {
            "seller_id": "seller-1",
            "default_start_date": "2025-01-01",
            "default_end_date": "2025-03-31",
            "metric_name": "gmv",
            "granularity": "day",
        },
        "complex_route_decision": _decision(),
        "node_history": [],
    }
    validated_json = _authorize_canary_plan(monkeypatch, state)
    monkeypatch.setattr(plan_node, "current_tenant_context", lambda: context)

    result = plan_node.complex_plan(state)

    assert result["complex_plan"] == plan
    assert result["complex_clarification"]["status"] == "confirmation_required"
    assert result["task_type"] == "complex_analysis"
    assert result["semantic_contract"] == {}
    assert json.loads(validated_json[0]) == {"fixture": "validated"}
    args, kwargs = calls[0]
    assert args[0] is context
    assert args[2] == {
        "seller_id": "seller-1",
        "metric_name": "gmv",
        "start_date": "2025-01-01",
        "end_date": "2025-03-31",
        "granularity": "day",
    }
    assert isinstance(kwargs["route_decision"], SealedRouteDecision)
    assert kwargs["route_decision"].route_decision_status == "SELECTED"


def test_missing_server_locked_scope_clarifies_without_creating_a_plan(monkeypatch) -> None:
    monkeypatch.setattr(plan_node, "create_plan_v2", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not create")))
    state = {
        "user_message": "按购买频次和金额给客户分层",
        "use_current_explorer_context": False,
        "complex_route_decision": _decision(),
        "node_history": [],
    }
    _authorize_canary_plan(monkeypatch, state)
    result = plan_node.complex_plan(state)
    assert result.get("complex_plan") is None
    assert result["complex_route_decision"]["kind"] == "clarify"
    assert result["complex_clarification"]["reason_code"] == "scope_required"
    clarification_node.complex_clarification(result)
    assert result["complex_clarification"]["status"] == "complex_scope_required"


def test_multiple_capability_clarification_offers_registered_choices_without_plan():
    state = {
        "complex_route_decision": {
            "kind": "clarify",
            "evidence": {"clarification_reason": "multiple_capabilities"},
        },
        "complex_clarification": {
            "reason_code": "multiple_capabilities",
            "candidate_templates": [
                {"template_id": "cohort_retention", "display_name": "Cohort 留存分析"},
                {"template_id": "customer_lifecycle", "display_name": "客户生命周期分析"},
            ],
        },
        "node_history": [],
    }

    result = clarification_node.complex_clarification(state)

    assert result["complex_clarification"]["status"] == "complex_template_required"
    assert len(result["complex_clarification"]["candidate_templates"]) == 2
    assert "待确认计划" in result["answer"]
    assert result.get("complex_plan") is None


def test_typed_sealed_route_is_required_before_service_plan_creation(monkeypatch) -> None:
    class _Connection:
        def execute(self, *_args, **_kwargs):
            return None

    class _Begin:
        def __enter__(self):
            return _Connection()

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(service, "_scope", lambda _context, payload: payload)
    monkeypatch.setattr(service, "_insert_plan_v2", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service.engine, "begin", lambda: _Begin())
    plan = service.create_plan_v2(
        SimpleNamespace(tenant_id="tenant-1", user_id="user-1"),
        "语义路由选择的分析问题",
        {
            "seller_id": "seller-1",
            "default_start_date": "2025-01-01",
            "default_end_date": "2025-03-31",
            "metric_name": "gmv",
            "granularity": "day",
        },
        template_name="rfm_segmentation",
        route_decision=_decision(),
    )
    assert plan["template_name"] == "rfm_segmentation"
    assert plan["status"] == "pending_confirmation"


def test_untyped_route_dict_fails_closed_before_plan_creation(monkeypatch) -> None:
    monkeypatch.setattr(plan_node, "get_settings", lambda: SimpleNamespace(complex_route_mode="active"))
    monkeypatch.setattr(plan_node, "create_plan_v2", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not create")))
    result = plan_node.complex_plan({
        "complex_route_decision": {"kind": "template", "route_decision_status": "SELECTED"},
        "node_history": [],
    })
    assert result.get("complex_plan") is None
    assert result["complex_clarification"]["reason_code"] == "semantic_router_not_ready"


def test_typed_decision_without_canary_enrollment_cannot_create_plan(monkeypatch) -> None:
    monkeypatch.setattr(
        plan_node,
        "get_settings",
        lambda: SimpleNamespace(complex_route_mode="active", complex_route_canary_enabled=False),
    )
    monkeypatch.setattr(plan_node, "current_tenant_context", lambda: None)
    monkeypatch.setattr(
        plan_node,
        "create_plan_v2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("canary must gate plan creation")),
    )

    result = plan_node.complex_plan({
        "complex_route_decision": _decision(),
        "complex_route_canary_selected": False,
        "semantic_interpreter_observation": {"status": "ok"},
        "node_history": [],
    })

    assert result.get("complex_plan") is None
    assert result["complex_clarification"]["reason_code"] == "canary_not_authorized"


@pytest.mark.parametrize(
    ("error", "reason", "status"),
    [
        (ValueError("invalid scope"), "plan_creation_rejected", "complex_plan_rejected"),
        (PermissionError("scope denied"), "scope_rejected", "complex_scope_rejected"),
    ],
)
def test_plan_creation_rejections_route_to_clear_clarification(
    monkeypatch, error: Exception, reason: str, status: str,
) -> None:
    monkeypatch.setattr(plan_node, "get_settings", lambda: SimpleNamespace(complex_route_mode="active"))
    monkeypatch.setattr(workflow, "get_settings", lambda: SimpleNamespace(complex_route_mode="active"))
    monkeypatch.setattr(plan_node, "current_tenant_context", lambda: object())
    monkeypatch.setattr(
        plan_node,
        "create_plan_v2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    state = {
        "user_message": "Cohort 留存分析",
        "use_current_explorer_context": True,
        "controlled_analysis_scope": {
            "seller_id": "seller-1",
            "default_start_date": "2025-01-01",
            "default_end_date": "2025-01-31",
        },
        "complex_route_decision": _decision(),
        "node_history": [],
    }
    _authorize_canary_plan(monkeypatch, state)

    result = plan_node.complex_plan(state)

    assert result.get("complex_plan") is None
    assert result["complex_route_decision"]["kind"] == "clarify"
    assert result["complex_clarification"]["reason_code"] == reason
    assert workflow._route_after_complex_plan(result) == "complex_clarification"
    clarification_node.complex_clarification(result)
    assert result["complex_clarification"]["status"] == status


def test_specific_router_comparison_reason_replaces_generic_clarification() -> None:
    state = {
        "complex_route_decision": {
            "kind": "clarify",
            "evidence": {"clarification_reason": "comparison_dimension_not_supported"},
        },
        "complex_clarification": {"status": "needs_context", "reason_code": "needs_context"},
        "node_history": [],
    }

    result = clarification_node.complex_clarification(state)

    assert result["complex_clarification"] == {
        "status": "needs_context",
        "reason_code": "comparison_dimension_not_supported",
    }
    assert "不能同时按商品品类分组并按支付方式比较" in result["answer"]
    assert "本轮没有执行数据查询" in result["answer"]


def test_dimension_required_uses_specific_complex_clarification_copy() -> None:
    state = {
        "complex_route_decision": {
            "kind": "clarify",
            "route_decision_status": "NEEDS_CONTEXT",
            "evidence": {"clarification_reason": "dimension_required"},
        },
        "complex_clarification": {"reason_code": "dimension_required"},
        "node_history": [],
    }

    result = clarification_node.complex_clarification(state)

    assert result["task_type"] == "complex_analysis"
    assert result["complex_clarification"] == {
        "status": "needs_context",
        "reason_code": "dimension_required",
    }
    assert "请明确要按哪个维度展示统计贡献" in result["answer"]


def test_unsupported_contribution_dimension_keeps_specific_reason_and_action() -> None:
    state = {
        "complex_route_decision": {
            "kind": "clarify",
            "evidence": {"clarification_reason": "dimension_not_supported_by_metric"},
        },
        "complex_clarification": {"status": "needs_context", "reason_code": "needs_context"},
        "node_history": [],
    }

    result = clarification_node.complex_clarification(state)

    assert result["complex_clarification"] == {
        "status": "needs_context",
        "reason_code": "dimension_not_supported_by_metric",
    }
    assert "不支持请求的统计贡献维度" in result["answer"]
    assert "改选该指标支持的统计贡献维度" in result["answer"]
    assert "没有执行数据查询" in result["answer"]
