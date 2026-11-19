"""Scope locking: use_current_explorer_context flag must survive init_state,
and data_query must not fall back to global MCP when scope is enabled but
controlled_analysis_scope is missing.
"""

from app.graph.nodes.data_query import data_query
from app.graph.nodes.resolve_explorer_context import resolve_explorer_context
from app.graph.workflow import init_state
from app.tenancy import SellerGrant, TenantContext, activate_tenant_context


def test_init_state_does_not_override_true_flag() -> None:
    """use_current_explorer_context=True 传入后，init_state 不能设回 False。"""
    state = init_state({"use_current_explorer_context": True})
    assert state["use_current_explorer_context"] is True


def test_init_state_defaults_to_false() -> None:
    """不传入时默认 False。"""
    state = init_state({})
    assert state["use_current_explorer_context"] is False


def test_resolve_explorer_context_disabled_records_trace(monkeypatch) -> None:
    """禁用范围时写入 node_history 说明未启用，不生成 scope。"""
    state = resolve_explorer_context({"use_current_explorer_context": False})
    assert state.get("controlled_analysis_scope") is None
    summaries = [entry["summary"] for entry in state.get("node_history", [])]
    assert any("disabled" in s.lower() for s in summaries)


def test_resolve_explorer_context_enabled_missing_payload(monkeypatch) -> None:
    """启用但缺筛选条件时写结构化错误，不报裸 ValueError。"""
    state = resolve_explorer_context({
        "use_current_explorer_context": True,
        "tenant_id": "t", "user_id": "u",
    })
    codes = [e["code"] for e in state.get("errors", [])]
    assert "controlled_scope_missing" in codes


def test_resolve_explorer_context_enabled_builds_selection_id(monkeypatch) -> None:
    """启用范围锁定时，即使未启用数据版本，也必须正常计算范围指纹。"""

    class FakeScope:
        def model_dump(self, mode="json"):
            return {
                "seller_id": "seller-a",
                "start_date": "2018-08-01",
                "end_date": "2018-08-31",
                "metric_name": "seller_avg_payment_installments",
                "granularity": "day",
            }

    snapshot = {
        "context": {
            "seller_id": "seller-a",
            "start_date": "2018-08-01",
            "end_date": "2018-08-31",
            "metric_name": "seller_avg_payment_installments",
            "granularity": "day",
        },
        "summary": {"status": "ok", "value": 2.6},
        "lineage": {"source_table": "test"},
    }
    monkeypatch.setattr("app.graph.nodes.resolve_explorer_context.preview_explorer", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr("app.graph.nodes.resolve_explorer_context.controlled_scope", lambda _snapshot: FakeScope())

    context = TenantContext(
        tenant_id="t", user_id="u", role="analyst", scope_type="seller",
        seller_id="seller-a", seller_grants=(SellerGrant("seller-a"),),
    )
    with activate_tenant_context(context):
        state = resolve_explorer_context({
            "use_current_explorer_context": True,
            "tenant_id": "t",
            "user_id": "u",
            "user_role": "analyst",
            "scope_type": "seller",
            "seller_id": "seller-a",
            "explorer_context": snapshot["context"],
            "trace_id": "scope-selection-regression",
        })

    assert state.get("errors") in (None, [])
    assert state["explorer_selection_id"].startswith("explorer:")


def test_resolve_explorer_context_requires_trusted_context(monkeypatch) -> None:
    """Graph-state identity fields alone cannot authorize a controlled query."""
    monkeypatch.setattr(
        "app.graph.nodes.resolve_explorer_context.current_tenant_context",
        lambda: None,
    )
    state = resolve_explorer_context({
        "use_current_explorer_context": True,
        "tenant_id": "t",
        "user_id": "u",
        "user_role": "analyst",
        "scope_type": "seller",
        "seller_id": "seller-a",
        "explorer_context": {
            "seller_id": "seller-a", "metric_name": "review_score_avg",
            "start_date": "2018-08-01", "end_date": "2018-08-31",
            "granularity": "month",
        },
    })

    assert state["errors"][-1]["code"] == "tenant_context_missing"
    assert "controlled_analysis_scope" not in state


def test_resolve_explorer_context_uses_full_server_seller_grants(monkeypatch) -> None:
    """A multi-seller identity is read from trusted context, not graph state."""
    seen = {}
    context = TenantContext(
        tenant_id="t", user_id="u", role="analyst", scope_type="seller",
        seller_grants=(SellerGrant("seller-a"), SellerGrant("seller-b")),
    )
    snapshot = {
        "context": {
            "seller_id": "seller-a", "start_date": "2018-08-01",
            "end_date": "2018-08-31", "metric_name": "review_score_avg",
            "granularity": "month",
        },
        "summary": {"status": "ok", "value": 4.2},
        "lineage": {"source_table": "test"},
    }

    def fake_preview(received_context, _query):
        seen["context"] = received_context
        assert received_context.seller_grant("seller-a") is not None
        assert received_context.seller_grant("seller-b") is not None
        return snapshot

    class FakeScope:
        def model_dump(self, mode="json"):
            return snapshot["context"]

    monkeypatch.setattr(
        "app.graph.nodes.resolve_explorer_context.preview_explorer", fake_preview,
    )
    monkeypatch.setattr(
        "app.graph.nodes.resolve_explorer_context.controlled_scope",
        lambda _snapshot: FakeScope(),
    )

    with activate_tenant_context(context):
        state = resolve_explorer_context({
            "use_current_explorer_context": True,
            "tenant_id": "t", "user_id": "u", "user_role": "analyst",
            "scope_type": "seller", "seller_id": None,
            "explorer_context": snapshot["context"],
            "trace_id": "multi-seller-scope",
        })

    assert state.get("errors") in (None, [])
    assert seen["context"] is context
    assert state["controlled_analysis_scope"]["seller_id"] == "seller-a"


def test_resolve_explorer_context_rejects_graph_identity_mismatch(monkeypatch) -> None:
    context = TenantContext(
        tenant_id="trusted-tenant", user_id="u", role="analyst",
        scope_type="seller", seller_id="seller-a",
        seller_grants=(SellerGrant("seller-a"),),
    )
    with activate_tenant_context(context):
        state = resolve_explorer_context({
            "use_current_explorer_context": True,
            "tenant_id": "other-tenant", "user_id": "u",
            "user_role": "analyst", "scope_type": "seller",
            "explorer_context": {
                "seller_id": "seller-a", "metric_name": "review_score_avg",
                "start_date": "2018-08-01", "end_date": "2018-08-31",
                "granularity": "month",
            },
        })

    assert state["errors"][-1]["code"] == "tenant_context_mismatch"
    assert "controlled_analysis_scope" not in state


def test_data_query_refuses_global_mcp_when_scope_missing(monkeypatch) -> None:
    """use_current_explorer_context=True 但 scope 缺失时，
    data_query 不调 query_business_data，直接写 controlled_scope_missing。"""
    calls = []

    def fake_global(**kwargs):
        calls.append(kwargs)
        return {"server": "local-mcp", "tool_name": "gmv", "arguments": {}, "request_id": "test", "result": {"status": "ok"}}

    monkeypatch.setattr("app.graph.nodes.data_query.query_business_data", fake_global)
    state = data_query({
        "user_message": "查询 GMV",
        "trace_id": "scope-missing-guard",
        "use_current_explorer_context": True,
        # 故意不传 controlled_analysis_scope
    })
    codes = [e["code"] for e in state.get("errors", [])]
    assert "controlled_scope_missing" in codes
    assert len(calls) == 0, "must NOT fall back to global query_business_data"


def test_data_query_calls_scoped_when_scope_present(monkeypatch) -> None:
    """use_current_explorer_context=True 且 scope 存在时走 scoped MCP。"""
    calls = []

    def fake_scoped(question, scope_payload, *, request_id=None, manifest_id=None):
        calls.append((question, scope_payload))
        return {"server": "scoped-mcp", "tool_name": "query_scoped_metric", "arguments": {}, "request_id": "test", "result": {"status": "ok"}}

    monkeypatch.setattr("app.graph.nodes.data_query.query_scoped_business_data", fake_scoped)
    state = data_query({
        "user_message": "查询 GMV",
        "trace_id": "scope-present",
        "use_current_explorer_context": True,
        "controlled_analysis_scope": {
            "seller_id": "seller-a", "start_date": "2018-08-01", "end_date": "2018-08-10",
            "metric_name": "gmv", "granularity": "day",
        },
    })
    assert len(calls) == 1, "must call scoped MCP when scope present"
    assert state.get("errors") is None or len(state["errors"]) == 0


def test_data_query_unrestricted_when_flag_false(monkeypatch) -> None:
    """use_current_explorer_context=False 且无 scope 时走全局 query_business_data。"""
    calls = []

    def fake_global(**kwargs):
        calls.append(kwargs)
        return {"server": "local-mcp", "tool_name": "gmv", "arguments": {}, "request_id": "test", "result": {"status": "ok"}}

    monkeypatch.setattr("app.graph.nodes.data_query.query_business_data", fake_global)
    state = data_query({
        "user_message": "查询 GMV",
        "trace_id": "unrestricted",
        "use_current_explorer_context": False,
    })
    assert len(calls) == 1, "must call global MCP when scope is not active"
    assert state.get("errors") is None or len(state["errors"]) == 0
