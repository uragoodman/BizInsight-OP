import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.mcp.client import call_local_mcp_tool


client = TestClient(app)

MSG_GMV_DEF = "GMV \u662f\u4ec0\u4e48\uff1f"
MSG_GMV_VALUE = "\u67e5\u8be2 2018-08-10 \u7684 GMV"
MSG_GMV_TREND = "\u67e5\u8be2 2018-08-01 \u5230 2018-08-03 \u7684 GMV \u8d8b\u52bf"
MSG_GMV_COMPARE = "\u5bf9\u6bd4 2018-08-09 \u548c 2018-08-10 \u7684 GMV"
MSG_GMV_BREAKDOWN = "\u67e5\u8be2 2018-08-10 \u7684 GMV \u6309\u5dde\u62c6\u89e3"
MSG_CUSTOMER_COUNT = "\u67e5\u8be2 2018-08-10 \u7684\u6210\u4ea4\u5ba2\u6237\u6570"
MSG_CUSTOMER_COUNT_TREND = "\u67e5\u8be2 2018-08-08 \u5230 2018-08-10 \u7684\u6210\u4ea4\u5ba2\u6237\u6570\u8d8b\u52bf"
MSG_REPEAT_PURCHASE_RATE_COMPARE = "\u5bf9\u6bd4 2018-08-09 \u548c 2018-08-10 \u7684\u590d\u8d2d\u7387"
MSG_GMV_BY_SELLER_STATE = "\u67e5\u8be2 2018-08-10 \u7684 GMV \u6309\u5356\u5bb6\u5dde\u62c6\u89e3"


def test_chat_route_returns_trace() -> None:
    response = client.post("/chat", json={"message": MSG_GMV_DEF})
    body = response.json()
    assert response.status_code == 200
    assert body["task_type"] == "metric_definition"
    assert "trace_id" in body


def test_chat_route_queries_gmv_metric() -> None:
    response = client.post("/chat", json={"message": MSG_GMV_VALUE})
    body = response.json()
    assert response.status_code == 200
    assert body["task_type"] == "metric_query"
    assert "2018-08-10" in body["answer"]
    assert "gmv" in body["answer"].lower()
    assert "analytics.daily_metrics" in body["answer"]


def test_chat_route_queries_metric_trend() -> None:
    response = client.post("/chat", json={"message": MSG_GMV_TREND})
    body = response.json()
    assert response.status_code == 200
    assert body["task_type"] == "metric_query"
    # The answer wording and list formatting are LLM-generated; assert the
    # underlying dates and value survive either supported presentation.
    for expected in ("2018-08-01", "2018-08-02", "2018-08-03", "48066.12"):
        assert expected in body["answer"]


def test_chat_route_queries_metric_compare() -> None:
    response = client.post("/chat", json={"message": MSG_GMV_COMPARE})
    body = response.json()
    assert response.status_code == 200
    assert body["task_type"] == "metric_query"
    assert "\u5bf9\u6bd4\u7ed3\u679c" in body["answer"]
    assert "2018-08-09" in body["answer"]
    assert "2018-08-10" in body["answer"]


def test_chat_route_queries_metric_breakdown() -> None:
    response = client.post("/chat", json={"message": MSG_GMV_BREAKDOWN})
    body = response.json()
    assert response.status_code == 200
    assert body["task_type"] == "metric_query"
    assert "customer_state" in body["answer"]


def test_chat_route_queries_customer_count_metric() -> None:
    response = client.post("/chat", json={"message": MSG_CUSTOMER_COUNT})
    body = response.json()
    assert response.status_code == 200
    assert body["task_type"] == "metric_query"
    assert "\u6210\u4ea4\u5ba2\u6237\u6570" in body["answer"]
    assert "2018-08-10" in body["answer"]


def test_chat_route_queries_customer_count_trend() -> None:
    response = client.post("/chat", json={"message": MSG_CUSTOMER_COUNT_TREND})
    body = response.json()
    assert response.status_code == 200
    assert body["task_type"] == "metric_query"
    assert "\u6574\u4f53" in body["answer"]
    assert "\u6210\u4ea4\u5ba2\u6237\u6570" in body["answer"]


def test_chat_route_queries_repeat_purchase_rate_compare() -> None:
    response = client.post("/chat", json={"message": MSG_REPEAT_PURCHASE_RATE_COMPARE})
    body = response.json()
    assert response.status_code == 200
    assert body["task_type"] == "metric_query"
    assert "\u5bf9\u6bd4\u7ed3\u679c" in body["answer"]
    assert "\u590d\u8d2d\u7387" in body["answer"]


def test_chat_route_queries_gmv_by_seller_state_breakdown() -> None:
    response = client.post("/chat", json={"message": MSG_GMV_BY_SELLER_STATE})
    body = response.json()
    assert response.status_code == 200
    assert body["task_type"] == "metric_query"
    assert "seller_state" in body["answer"]


def test_mcp_metric_value_tool_can_be_called_independently() -> None:
    result = call_local_mcp_tool("query_gmv", {"metric_date": "2018-08-10"})
    assert result["metric_name"] == "gmv"
    assert result["metric_date"] == "2018-08-10"
    assert result["source_table"] == "analytics.daily_metrics"
    assert result["status"] in {"ok", "not_found"}


def test_mcp_metric_trend_tool_can_be_called_independently() -> None:
    result = call_local_mcp_tool(
        "query_metric_trend",
        {
            "metric_name": "gmv",
            "start_date": "2018-08-01",
            "end_date": "2018-08-03",
            "granularity": "day",
        },
    )
    assert result["metric_name"] == "gmv"
    assert len(result["points"]) == 3
    assert result["status"] == "ok"


def test_mcp_metric_compare_tool_can_be_called_independently() -> None:
    result = call_local_mcp_tool(
        "compare_metric",
        {
            "metric_name": "gmv",
            "left_date": "2018-08-09",
            "right_date": "2018-08-10",
        },
    )
    assert result["metric_name"] == "gmv"
    assert result["left_date"] == "2018-08-09"
    assert result["right_date"] == "2018-08-10"
    assert result["status"] == "ok"


def test_mcp_metric_breakdown_tool_can_be_called_independently() -> None:
    result = call_local_mcp_tool(
        "breakdown_by_dimension",
        {
            "metric_name": "gmv",
            "metric_date": "2018-08-10",
            "dimension": "customer_state",
            "limit": 5,
        },
    )
    assert result["metric_name"] == "gmv"
    assert result["dimension"] == "customer_state"
    assert len(result["items"]) > 0
    assert result["status"] == "ok"


def test_mcp_customer_count_tool_can_be_called_independently() -> None:
    result = call_local_mcp_tool("query_customer_count", {"metric_date": "2018-08-10"})
    assert result["metric_name"] == "customer_count"
    assert result["metric_date"] == "2018-08-10"
    assert result["status"] in {"ok", "not_found"}


def test_mcp_new_metric_trend_tool_can_be_called_independently() -> None:
    result = call_local_mcp_tool(
        "query_metric_trend",
        {
            "metric_name": "customer_count",
            "start_date": "2018-08-08",
            "end_date": "2018-08-10",
            "granularity": "day",
        },
    )
    assert result["metric_name"] == "customer_count"
    assert len(result["points"]) == 3
    assert result["status"] in {"ok", "not_found"}


def test_mcp_new_metric_compare_tool_can_be_called_independently() -> None:
    result = call_local_mcp_tool(
        "compare_metric",
        {
            "metric_name": "repeat_purchase_rate",
            "left_date": "2018-08-09",
            "right_date": "2018-08-10",
        },
    )
    assert result["metric_name"] == "repeat_purchase_rate"
    assert result["status"] in {"ok", "not_found"}


def test_mcp_breakdown_supports_seller_state_dimension() -> None:
    result = call_local_mcp_tool(
        "breakdown_by_dimension",
        {
            "metric_name": "gmv",
            "metric_date": "2018-08-10",
            "dimension": "seller_state",
            "limit": 5,
        },
    )
    assert result["metric_name"] == "gmv"
    assert result["dimension"] == "seller_state"
    assert len(result["items"]) > 0
    assert result["status"] in {"ok", "not_found"}


def test_mcp_breakdown_supports_product_category_dimension() -> None:
    result = call_local_mcp_tool(
        "breakdown_by_dimension",
        {
            "metric_name": "order_count",
            "metric_date": "2018-08-10",
            "dimension": "product_category",
            "limit": 5,
        },
    )
    assert result["metric_name"] == "order_count"
    assert result["dimension"] == "product_category"
    assert len(result["items"]) > 0
    assert result["status"] in {"ok", "not_found"}


def test_mcp_breakdown_supports_payment_type_dimension() -> None:
    result = call_local_mcp_tool(
        "breakdown_by_dimension",
        {
            "metric_name": "customer_count",
            "metric_date": "2018-08-10",
            "dimension": "payment_type",
            "limit": 5,
        },
    )
    assert result["metric_name"] == "customer_count"
    assert result["dimension"] == "payment_type"
    assert len(result["items"]) > 0
    assert result["status"] in {"ok", "not_found"}


def test_mcp_gmv_payment_type_breakdown_uses_registered_allocation() -> None:
    from app.metrics.registry import get_registry

    registry = get_registry()
    assert registry.supports("gmv", "breakdown", scope="platform")
    assert registry.is_dimension_allowed("gmv", "payment_type", scope="platform")

    breakdown = call_local_mcp_tool(
        "breakdown_by_dimension",
        {
            "metric_name": "gmv",
            "metric_date": "2018-08-10",
            "dimension": "payment_type",
            "limit": 20,
        },
    )
    gmv = call_local_mcp_tool("query_gmv", {"metric_date": "2018-08-10"})

    assert breakdown["status"] == "ok"
    assert breakdown["allocation_method"] == "order_payment_value_ratio"
    assert "payment_value" in breakdown["lineage_note"]
    assert sum(item["value"] for item in breakdown["items"]) == pytest.approx(gmv["value"], abs=0.01)

    # Payment amount is already grouped by payment type and must not be marked
    # as GMV allocated by the payment-value ratio.
    payment_value = call_local_mcp_tool(
        "breakdown_by_dimension",
        {
            "metric_name": "payment_value",
            "metric_date": "2018-08-10",
            "dimension": "payment_type",
            "limit": 20,
        },
    )
    assert payment_value["status"] == "ok"
    assert "allocation_method" not in payment_value
    assert "lineage_note" not in payment_value


def test_trace_records_mcp_invocation_details() -> None:
    response = client.post("/chat", json={"message": MSG_GMV_VALUE})
    body = response.json()
    assert response.status_code == 200

    trace_response = client.get(f"/traces/{body['trace_id']}")
    trace = trace_response.json()
    assert trace_response.status_code == 200
    sn = trace["state_snapshot"]
    assert sn["tool_result"]["server"] == "metric-registry"
    assert sn["tool_result"]["tool_name"] == "registry_value"
    assert sn["tool_result"]["arguments"] == {"metric_date": "2018-08-10"}
    assert "result" in sn["tool_result"]
