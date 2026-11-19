from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_serves_console() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "BizInsight Agent Console" in response.text
    assert '<div id="app">' in response.text
