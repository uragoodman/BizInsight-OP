from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from app.api.routes import metrics as metrics_route
from app.api.routes import ui


def _request(accept: str) -> Request:
    return Request({
        "type": "http",
        "method": "GET",
        "path": "/metrics",
        "headers": [(b"accept", accept.encode("ascii"))],
    })


def test_browser_metrics_route_serves_spa_entry(monkeypatch):
    page = Response("<html>frontend</html>", media_type="text/html")
    monkeypatch.setattr(ui, "index", lambda: page)

    assert metrics_route.metrics(_request("text/html,application/xhtml+xml")) is page


def test_prometheus_scrape_keeps_text_exposition_response(monkeypatch):
    monkeypatch.setattr(metrics_route, "prometheus_payload", lambda: "# TYPE test_counter counter\ntest_counter 1\n")

    response = metrics_route.metrics(_request("text/plain"))

    assert response.body == b"# TYPE test_counter counter\ntest_counter 1\n"
    assert response.media_type == "text/plain; version=0.0.4"


def test_spa_roots_coexisting_with_api_prefixes_remain_refreshable(monkeypatch):
    page = Response("<html>frontend</html>", media_type="text/html")
    monkeypatch.setattr(ui, "index", lambda: page)

    assert ui.spa_fallback("knowledge") is page
    assert ui.spa_fallback("chat") is page
    try:
        ui.spa_fallback("knowledge/documents")
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("a backend knowledge API path must not fall back to the SPA")
