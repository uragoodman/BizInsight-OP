"""Opt-in black-box tests against the running HTTP API.

Run only when Docker/API is already running:
    $env:RUN_HTTP_SMOKE = "1"
    pytest -q tests/test_http_smoke.py

These tests deliberately do not use FastAPI TestClient.  They exercise the
same network boundary as the browser/Postman client.
"""

from __future__ import annotations

import os

import pytest

from scripts.api_smoke import DEFAULT_METRIC, _headers


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_HTTP_SMOKE") != "1",
    reason="set RUN_HTTP_SMOKE=1 to call the running API",
)


def _client():
    import httpx

    return httpx.Client(
        base_url=os.getenv("BIZINSIGHT_API_BASE_URL", "http://127.0.0.1:28080").rstrip("/"),
        headers=_headers(),
        timeout=float(os.getenv("BIZINSIGHT_HTTP_SMOKE_TIMEOUT", "180")),
    )


def test_live_health_http() -> None:
    with _client() as client:
        response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_docker_serves_built_frontend_over_http() -> None:
    with _client() as client:
        response = client.get("/explorer")
    assert response.status_code == 200
    assert "<html" in response.text.lower()
    assert "/assets/" in response.text


def test_explorer_preview_http() -> None:
    seller_id = os.getenv("BIZINSIGHT_SMOKE_SELLER_ID", "1f50f920176fa81dab994f9023523100")
    body = {
        "seller_id": seller_id,
        "metric_name": DEFAULT_METRIC,
        "start_date": os.getenv("BIZINSIGHT_SMOKE_START_DATE", "2017-08-31"),
        "end_date": os.getenv("BIZINSIGHT_SMOKE_END_DATE", "2018-08-31"),
        "granularity": "day",
    }
    with _client() as client:
        response = client.post("/data-explorer/preview", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["context"]["metric_name"] == DEFAULT_METRIC


def test_randomized_experiment_api_http():
    seller_id = os.getenv("BIZINSIGHT_SMOKE_SELLER_ID", "1f50f920176fa81dab994f9023523100")
    body = {
        "seller_id": seller_id,
        "start_date": os.getenv("BIZINSIGHT_SMOKE_START_DATE", "2017-08-31"),
        "end_date": os.getenv("BIZINSIGHT_SMOKE_END_DATE", "2018-08-31"),
    }
    with _client() as client:
        response = client.post("/causal-experiments/eligible", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload.get("items"), list)
    assert payload["scope"]["seller_id"] == seller_id


def test_scoped_statistics_chat_http() -> None:
    seller_id = os.getenv("BIZINSIGHT_SMOKE_SELLER_ID", "1f50f920176fa81dab994f9023523100")
    body = {
        "message": "帮我计算平均每单运费的均值、最大值和最小值，并说明差异。",
        "use_current_explorer_context": True,
        "explorer_context": {
            "seller_id": seller_id,
            "metric_name": DEFAULT_METRIC,
            "start_date": os.getenv("BIZINSIGHT_SMOKE_START_DATE", "2017-08-31"),
            "end_date": os.getenv("BIZINSIGHT_SMOKE_END_DATE", "2018-08-31"),
            "granularity": "day",
        },
        "memory_enabled": False,
        "chat_session_id": "pytest-http-smoke",
    }
    with _client() as client:
        response = client.post("/chat", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload.get("trace_id")
    assert "内部异常" not in str(payload.get("answer") or "")
