from app.api.routes.traces import _trace_view
from app.tenancy import SellerGrant, TenantContext


def test_trace_view_redacts_business_payload_for_analyst():
    trace = {
        "recorded_at": "now",
        "trace_version": {"schema_version": "2.0"},
        "graph_path": ["data_query", "answer"],
        "state_snapshot": {
            "tenant_id": "tenant-a",
            "question": "secret question",
            "answer": "secret answer",
            "status": "completed",
            "error_code": None,
        },
        "events": [{"node": "data_query", "status": "completed", "duration_ms": 2, "payload": "secret"}],
    }
    context = TenantContext(tenant_id="tenant-a", user_id="user-a", role="analyst", scope_type="seller", seller_grants=(SellerGrant("seller-a"),))
    result = _trace_view(trace, context)
    assert result["view"] == "operational_summary"
    assert "question" not in result["state_snapshot"]
    assert "answer" not in result["state_snapshot"]
    assert result["events"][0] == {"node": "data_query", "status": "completed", "duration_ms": 2}


def test_trace_view_keeps_full_payload_for_admin():
    trace = {"state_snapshot": {"answer": "visible"}, "events": []}
    context = TenantContext(tenant_id="platform", user_id="admin", role="admin", scope_type="platform")
    assert _trace_view(trace, context) is trace
