from __future__ import annotations

from app.rag.retriever import (
    TenantPrivateKnowledgeRead,
    _retrieve_qdrant_by_policy,
    _qdrant_routes,
    _resolve_metric_name,
    retrieve_knowledge,
)
from app.tenancy import SELLER_SCOPE, TenantContext


def _seller_context() -> TenantContext:
    return TenantContext(
        tenant_id="seller-a",
        user_id="user-a",
        role="analyst",
        scope_type=SELLER_SCOPE,
        seller_id="seller-a",
    )


def test_registry_metric_resolution_is_recorded_and_ambiguity_is_safe() -> None:
    assert _resolve_metric_name("GMV 是什么？") == ("gmv", "inferred", False)
    resolved, source, conflict = _resolve_metric_name("客单价和订单量怎么比较？")
    assert resolved is None
    assert source == "conflict"
    assert conflict is True


def test_metric_analysis_policy_has_dedicated_and_generic_routes() -> None:
    assert _qdrant_routes({"metric", "sop"}, "gmv") == [
        ({"metric"}, "gmv"),
        ({"sop"}, None),
    ]


def test_private_knowledge_unavailable_is_not_reported_as_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.rag.retriever._tenant_private_chunks",
        lambda _context: TenantPrivateKnowledgeRead(
            [], "unavailable", "tenant_knowledge_query_failed", "OperationalError"
        ),
    )

    result = retrieve_knowledge(
        "GMV 是什么？",
        doc_types={"metric"},
        tenant_context=_seller_context(),
        backend_override="local_hybrid",
    )

    assert result.diagnostics is not None
    assert result.diagnostics.tenant_private_knowledge_status == "unavailable"
    assert result.diagnostics.context_incomplete is True


def test_qdrant_fallback_has_controlled_diagnostics_and_event(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "app.rag.retriever.record_event",
        lambda name, attributes: events.append((name, attributes)),
    )

    def unavailable(*_args, diagnostics, **_kwargs):
        diagnostics.update({
            "reason": "qdrant_query_failed",
            "exception_type": "ConnectionError",
        })
        return None

    monkeypatch.setattr("app.rag.retriever._retrieve_qdrant_by_policy", unavailable)
    result = retrieve_knowledge(
        "GMV 是什么？",
        doc_types={"metric"},
        backend_override="qdrant_vector",
    )

    assert result.backend == "local"
    assert result.diagnostics is not None
    assert result.diagnostics.degraded is True
    assert result.diagnostics.fallback_from == "qdrant_vector"
    assert result.diagnostics.degraded_reason == "qdrant_query_failed"
    assert result.diagnostics.exception_type == "ConnectionError"
    assert all("password" not in str(attributes).lower() for _, attributes in events)


def test_qdrant_success_is_explicitly_not_degraded(monkeypatch) -> None:
    from app.rag.schemas import RagSearchResult

    monkeypatch.setattr(
        "app.rag.retriever._retrieve_qdrant_by_policy",
        lambda *_args, **_kwargs: RagSearchResult(
            query="GMV 是什么？", backend="qdrant", strategy="qdrant_vector", top_k=5,
        ),
    )
    result = retrieve_knowledge("GMV 是什么？", backend_override="qdrant_vector")
    assert result.backend == "qdrant"
    assert result.diagnostics is not None
    assert result.diagnostics.degraded is False


def test_inferred_metric_is_forwarded_to_qdrant_policy(monkeypatch) -> None:
    from app.rag.schemas import RagSearchResult
    captured: dict[str, object] = {}

    def retrieve_policy(*_args, **kwargs):
        captured.update(kwargs)
        return RagSearchResult(
            query="GMV 是什么？", backend="qdrant", strategy="qdrant_vector", top_k=5,
        )

    monkeypatch.setattr("app.rag.retriever._retrieve_qdrant_by_policy", retrieve_policy)
    result = retrieve_knowledge(
        "GMV 是什么？", doc_types={"metric"}, backend_override="qdrant_vector",
    )
    assert captured["metric_name"] == "gmv"
    assert result.diagnostics.resolved_metric_name == "gmv"
    assert result.diagnostics.metric_resolution_source == "inferred"


def test_local_candidate_pool_is_trimmed_to_requested_top_k(monkeypatch) -> None:
    result = retrieve_knowledge(
        "GMV 是什么？", top_k=2, backend_override="local_hybrid",
    )
    assert result.top_k == 2
    assert len(result.hits) <= 2
    assert result.filters["candidate_k"] >= result.top_k
