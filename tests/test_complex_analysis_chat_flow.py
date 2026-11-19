from __future__ import annotations

import json

import app.api.routes.chat as chat_routes
import pytest
from fastapi.testclient import TestClient

from app.api.auth import require_tenant_context
from app.core.config import Settings
from app.graph.nodes import intent_router as intent_router_node
from app.graph.nodes import data_query as data_query_node
from app.graph.nodes import semantic_parse as semantic_parse_node
from app.graph.nodes import triage as triage_node
from app.graph import workflow
from app.mcp.schemas import MetricPlanningDecision, MetricQueryBatchPlan
from app.main import app
from app.tenancy import TenantContext


def _active_settings() -> Settings:
    return Settings(
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
    )


def _patch_active_no_provider_route(monkeypatch, settings: Settings) -> None:
    monkeypatch.setattr(semantic_parse_node, "get_settings", lambda: settings)
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
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

    # The compiler owns this as an unresolved ordinary metric request, so the
    # intended fallback is a metric clarification, not a complex-plan error.
    # Keep this HTTP flow test deterministic and prevent a real planner call.
    monkeypatch.setattr(
        data_query_node,
        "plan_confirmed_metric_queries",
        lambda *_args, **_kwargs: MetricQueryBatchPlan(decision=MetricPlanningDecision(
            status="clarification_required",
            reason="请确认要分析的指标。",
            reason_code="semantic_confirmation_required",
        )),
    )


def test_active_chat_clarifies_unresolved_ordinary_metric_when_interpreter_is_unavailable(monkeypatch) -> None:
    settings = _active_settings()
    context = TenantContext(
        tenant_id="phase0-chat-tenant",
        user_id="phase0-chat-user",
        role="analyst",
        scope_type="seller",
        seller_id="phase0-chat-seller",
    )
    _patch_active_no_provider_route(monkeypatch, settings)
    previous = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"message": "客户留存分析"})
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["clarification"]["status"] == "clarification_required"
            assert payload.get("complex_plan") is None

            legacy_choice = client.post("/chat", json={
                "message": "客户留存分析",
                "complex_template_confirmation": "cohort_retention",
            })
            assert legacy_choice.status_code == 200, legacy_choice.text
            assert legacy_choice.json().get("complex_plan") is None
            assert legacy_choice.json()["clarification"]["status"] == "clarification_required"
            assert legacy_choice.json().get("complex_plan") is None
    finally:
        if previous is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous


def test_active_stream_chat_clarifies_unresolved_ordinary_metric_when_interpreter_is_unavailable(monkeypatch) -> None:
    settings = _active_settings()
    context = TenantContext(
        tenant_id="phase0-stream-tenant",
        user_id="phase0-stream-user",
        role="analyst",
        scope_type="seller",
        seller_id="phase0-stream-seller",
    )
    _patch_active_no_provider_route(monkeypatch, settings)
    previous = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        with TestClient(app) as client:
            response = client.post("/chat/stream", json={"message": "客户留存分析"})
            assert response.status_code == 200, response.text
            events = [
                json.loads(line.removeprefix("data: "))
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            terminal = next(event for event in reversed(events) if event.get("event") in {"completed", "failed", "suspended"})
            assert terminal["event"] in {"completed", "suspended"}
            assert terminal.get("error") is not True
            assert terminal["clarification"]["status"] == "clarification_required"
            assert terminal.get("complex_plan") is None
    finally:
        if previous is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous


@pytest.mark.parametrize("route_mode", ["off", "shadow"])
def test_off_and_shadow_sync_http_keep_ordinary_path_without_plan(monkeypatch, route_mode: str) -> None:
    settings = Settings(_env_file=None, complex_route_mode=route_mode)
    context = TenantContext(
        tenant_id="phase0-http-tenant",
        user_id="phase0-http-user",
        role="analyst",
        scope_type="seller",
        seller_id="phase0-http-seller",
    )
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    observed_routes: list[str] = []

    def fake_workflow(message, *args, **kwargs):
        state = intent_router_node.intent_router({"user_message": message, "node_history": []})
        observed_routes.append(workflow._route_after_intent_router(state))
        return {
            "answer": "ordinary triage response",
            "task_type": "general_chat",
            "complex_plan": None,
            "trace_id": kwargs.get("trace_id", "phase0-sync-trace"),
        }

    monkeypatch.setattr(chat_routes, "run_chat_workflow", fake_workflow)
    previous = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"message": "客户留存分析"})
        assert response.status_code == 200, response.text
        assert response.json()["answer"] == "ordinary triage response"
        assert response.json().get("complex_plan") is None
        assert observed_routes == ["triage"]
    finally:
        if previous is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous


@pytest.mark.parametrize("route_mode", ["off", "shadow"])
def test_off_and_shadow_stream_http_keep_ordinary_path_without_plan(monkeypatch, route_mode: str) -> None:
    settings = Settings(_env_file=None, complex_route_mode=route_mode)
    context = TenantContext(
        tenant_id="phase0-stream-http-tenant",
        user_id="phase0-stream-http-user",
        role="analyst",
        scope_type="seller",
        seller_id="phase0-stream-http-seller",
    )
    monkeypatch.setattr(intent_router_node, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    observed_routes: list[str] = []

    def fake_stream(message, *args, **kwargs):
        state = intent_router_node.intent_router({"user_message": message, "node_history": []})
        observed_routes.append(workflow._route_after_intent_router(state))
        yield {
            "event": "completed",
            "answer": "ordinary triage response",
            "task_type": "general_chat",
            "complex_plan": None,
            "trace_id": "phase0-stream-trace",
        }

    monkeypatch.setattr(chat_routes, "stream_chat_workflow", fake_stream)
    previous = app.dependency_overrides.get(require_tenant_context)
    app.dependency_overrides[require_tenant_context] = lambda: context
    try:
        with TestClient(app) as client:
            response = client.post("/chat/stream", json={"message": "客户留存分析"})
        assert response.status_code == 200, response.text
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        terminal = next(event for event in reversed(events) if event.get("event") == "completed")
        assert terminal["answer"] == "ordinary triage response"
        assert terminal.get("complex_plan") is None
        assert observed_routes == ["triage"]
    finally:
        if previous is None:
            app.dependency_overrides.pop(require_tenant_context, None)
        else:
            app.dependency_overrides[require_tenant_context] = previous
