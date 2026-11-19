"""Qdrant-backed metric semantic recall.

Qdrant is deliberately only a recall layer.  The Registry and the mapping
state machine remain authoritative and every failure returns no candidates.
"""

from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from app.core.config import get_settings
from app.metrics.registry import get_registry


COLLECTION_NAME = "metric_semantic_catalog_v1"


def _semantic_text(item: dict[str, object]) -> str:
    fields = [
        str(item.get("display_name") or ""),
        str(item.get("definition") or ""),
        " ".join(str(value) for value in item.get("semantic_terms", [])),
        " ".join(str(value) for value in item.get("examples", [])),
        " ".join(str(value) for value in item.get("intents", [])),
        str(item.get("limitations") or ""),
    ]
    return "\n".join(value for value in fields if value)


def index_metric_semantic_catalog(*, scope: str = "platform", client: Any | None = None) -> dict[str, object]:
    """Create/update the independent catalog collection from Registry metadata."""
    from qdrant_client import QdrantClient, models
    from app.rag.indexing import make_dense_embed_fn

    settings = get_settings()
    collection = settings.metric_semantic_qdrant_collection or COLLECTION_NAME
    registry = get_registry()
    catalog = registry.planning_catalog(scope=scope, surface="chat")
    if not catalog:
        return {"collection": collection, "count": 0, "catalog_digest": registry.catalog_digest}
    client = client or QdrantClient(url=settings.qdrant_url, timeout=1.0)
    embed = make_dense_embed_fn()
    vectors = [embed(_semantic_text(item)) for item in catalog]
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
        )
    points = [
        models.PointStruct(
            # The same collection stores platform and seller catalogs.  Scope
            # must be part of the point identity; using a per-call integer
            # would make the second indexing pass overwrite the first one.
            id=str(uuid5(NAMESPACE_URL, f"{scope}:{item['metric_id']}")),
            vector=vectors[index],
            payload={
                "metric_id": item["metric_id"],
                "display_name": item["display_name"],
                "catalog_digest": registry.catalog_digest,
                "scope": scope,
                "semantic_text": _semantic_text(item),
            },
        )
        for index, item in enumerate(catalog)
    ]
    client.upsert(collection_name=collection, points=points, wait=True)
    return {"collection": collection, "count": len(points), "catalog_digest": registry.catalog_digest}


def retrieve_metric_candidates(
    question: str,
    *,
    scope: str,
    allowed_metrics: set[str] | None = None,
    limit: int = 5,
    client: Any | None = None,
) -> list[dict[str, object]]:
    """Recall at most five current-digest candidates; fail closed otherwise."""
    from app.rag.indexing import make_dense_embed_fn
    try:
        from qdrant_client import QdrantClient, models

        settings = get_settings()
        collection = settings.metric_semantic_qdrant_collection or COLLECTION_NAME
        registry = get_registry()
        catalog = {str(item["metric_id"]): item for item in registry.planning_catalog(scope=scope, surface="chat")}
        if allowed_metrics is not None:
            catalog = {metric_id: item for metric_id, item in catalog.items() if metric_id in allowed_metrics}
        if not catalog:
            return []
        client = client or QdrantClient(url=settings.qdrant_url, timeout=1.0)
        if not client.collection_exists(collection):
            return []
        result = client.query_points(
            collection_name=collection,
            query=make_dense_embed_fn()(question),
            limit=min(max(limit, 1), 5),
            with_payload=True,
            query_filter=models.Filter(must=[models.FieldCondition(
                key="catalog_digest", match=models.MatchValue(value=registry.catalog_digest),
            ), models.FieldCondition(
                key="scope", match=models.MatchValue(value=scope),
            )]),
        )
        candidates: list[dict[str, object]] = []
        seen_metric_ids: set[str] = set()
        for hit in result.points:
            payload = hit.payload or {}
            metric_id = str(payload.get("metric_id") or "")
            if metric_id not in catalog:
                continue
            # Incremental indexing can leave multiple points for one metric.
            # Candidate handles must represent unique server-owned metrics.
            if metric_id in seen_metric_ids:
                continue
            seen_metric_ids.add(metric_id)
            item = catalog[metric_id]
            candidates.append({
                # Re-number after the authoritative registry/scope filter so
                # the LLM-facing handle is a dense, stable index into the
                # server-owned candidate array.
                "candidate_no": len(candidates),
                "metric_id": metric_id,
                "display_name": item.get("display_name", metric_id),
                "reason": "Qdrant 召回的相关指标，统计口径仍需确认。",
                "evidence": str(payload.get("semantic_text") or ""),
                "score": float(getattr(hit, "score", 0.0) or 0.0),
                "relation": "related_but_different",
            })
        return candidates[:5]
    except Exception:
        return []
