from fastapi.testclient import TestClient

from app.api.auth import require_tenant_context
from app.main import app
from app.rag import indexing
from app.tenancy import TenantContext


client = TestClient(app)


def _admin_context() -> TenantContext:
    return TenantContext("platform-test", "admin-test", "admin", "platform")


def test_list_knowledge_documents(monkeypatch) -> None:
    monkeypatch.setitem(app.dependency_overrides, require_tenant_context, _admin_context)
    response = client.get("/knowledge/documents")
    body = response.json()

    assert response.status_code == 200
    assert body["total"] > 0
    assert body["index_version"]
    assert any(item["metric_name"] == "gmv" for item in body["documents"])
    assert "text" not in body["documents"][0]


def test_list_knowledge_documents_can_include_text(monkeypatch) -> None:
    monkeypatch.setitem(app.dependency_overrides, require_tenant_context, _admin_context)
    response = client.get("/knowledge/documents?include_text=true")
    body = response.json()

    assert response.status_code == 200
    assert "text" in body["documents"][0]


def test_update_knowledge_index(monkeypatch) -> None:
    monkeypatch.setitem(app.dependency_overrides, require_tenant_context, _admin_context)
    def fake_build_qdrant_index(*, collection_name: str, dry_run: bool) -> dict[str, object]:
        return {
            "backend": "qdrant",
            "status": "skipped",
            "collection": "bizinsight_knowledge",
            "documents": 20,
            "changed_documents": 0,
            "index_version": "v1",
        }

    monkeypatch.setattr(indexing, "build_qdrant_index", fake_build_qdrant_index)

    response = client.post("/knowledge/index?collection=bizinsight_knowledge_v2")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "skipped"
    assert body["collection"] == "bizinsight_knowledge"


def test_update_knowledge_index_requires_manage_knowledge(monkeypatch) -> None:
    context = TenantContext("tenant-a", "viewer-a", "viewer", "seller", seller_id="seller-a")
    monkeypatch.setitem(app.dependency_overrides, require_tenant_context, lambda: context)
    monkeypatch.setattr(
        indexing,
        "build_qdrant_index",
        lambda **_: (_ for _ in ()).throw(AssertionError("unauthorized indexing must not run")),
    )

    response = client.post("/knowledge/index?collection=bizinsight_knowledge")

    assert response.status_code == 403
