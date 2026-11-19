"""RAG 检索质量评测 runner：Doc/Chunk Hit@K、MRR@5、Hit@5/8/12/20、P50/P95。

P3.3 用途：量化"规则融合"基线，对比 Cross-Encoder 精排 / RRF 排名融合，
决定是否启用 Reranker。示例：python -m evaluation.retrieval.runner

语料可扩展性：
- expected_doc_ids 用完整 source_path（稳定、无歧义）；加载时校验引用文档存在，
  缺失则评测失败（不静默）——新增/重命名 markdown 后能立即暴露断链。
- expected_chunk_snippets 用内容锚点，chunk 重排/新增文档后仍能匹配正确证据。
"""

from __future__ import annotations

import json
import math
import statistics
import time
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

from app.rag.indexing import load_knowledge_chunks
from app.rag.config import get_rag_settings
from app.core.config import get_settings
from app.rag.retriever import retrieve_knowledge
from evaluation.retrieval.metrics import chunk_hit_at_k, doc_hit_at_k, mrr_at_k, ndcg_at_k

QUERIES = Path(__file__).parent / "queries.jsonl"
ROOT = Path(__file__).resolve().parents[2]
_HIT_KS = (5, 8, 12, 20)


def _load_cases() -> list[dict]:
    """读取评测集并做扩展性守卫：引用的文档必须存在于当前语料。"""
    cases = [json.loads(line) for line in QUERIES.read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus_paths = {chunk.source_path for chunk in load_knowledge_chunks()}
    for case in cases:
        for path in case.get("expected_doc_ids", []):
            if path not in corpus_paths:
                raise ValueError(f"Eval case {case['query_id']} references missing doc: {path}")
    return cases


def _evaluate_once(strategy: str = "rule", top_k: int = 20, *, backend_override: str = "local_hybrid") -> dict:
    """对指定重排策略跑一轮评测。

    strategy: rule / cross_encoder / rrf。top_k=20 一次检索即可统计 Hit@5/8/12/20。
    """
    cases = _load_cases()
    main = [c for c in cases if c.get("query_type") != "unanswerable"]
    unanswerable = [c for c in cases if c.get("query_type") == "unanswerable"]
    chunk_cases = [c for c in main if c.get("expected_chunk_snippets")]

    k_hits = {k: 0 for k in _HIT_KS}
    doc_hit5 = chunk_hit5 = mrr_sum = ndcg_sum = 0
    timeout_count = 0
    latencies: list[float] = []
    details: list[dict] = []

    for case in main:
        start = time.perf_counter()
        # Offline strategy experiments must not repeatedly initialize an
        # unavailable remote Qdrant/embedding client for every Golden query.
        # Qdrant release validation remains a separate release-contract test.
        result = retrieve_knowledge(case["query"], top_k=top_k, reranker_strategy=strategy, backend_override=backend_override)
        latencies.append((time.perf_counter() - start) * 1000)
        hits = result.hits
        for k in _HIT_KS:
            if chunk_hit_at_k(hits, case, k):
                k_hits[k] += 1
        doc_hit5 += int(doc_hit_at_k(hits, case, 5))
        # chunk_hit@5 只统计 chunk 级用例（有内容锚点的），避免与非 chunk 用例混算。
        if case.get("expected_chunk_snippets"):
            chunk_hit5 += int(chunk_hit_at_k(hits, case, 5))
        mrr_sum += mrr_at_k(hits, case, 5)
        ndcg_sum += ndcg_at_k(hits, case, 5)
        if result.diagnostics and result.diagnostics.reranker_reason == "timeout":
            timeout_count += 1
        details.append({"query_id": case["query_id"], "type": case.get("query_type"), "doc_hit@5": doc_hit_at_k(hits, case, 5), "mrr@5": round(mrr_at_k(hits, case, 5), 3), "ndcg@5": round(ndcg_at_k(hits, case, 5), 3)})
    unanswerable_count = len(unanswerable)

    n = len(main)
    chunk_n = len(chunk_cases)
    sorted_lat = sorted(latencies)
    return {
        "strategy": strategy, "backend": backend_override, "cases": n,
        "doc_hit@5": round(doc_hit5 / n, 4) if n else None,
        "chunk_hit@5": round(chunk_hit5 / chunk_n, 4) if chunk_n else None,
        "mrr@5": round(mrr_sum / n, 4) if n else None,
        "ndcg@5": round(ndcg_sum / n, 4) if n else None,
        "timeout_rate": round(timeout_count / n, 4) if n else None,
        "hit@k": {str(k): round(k_hits[k] / n, 4) if n else None for k in _HIT_KS},
        "p50_ms": round(statistics.median(sorted_lat), 1) if sorted_lat else None,
        "p95_ms": round(sorted_lat[int(n * 0.95) - 1], 1) if n else None,
        "unanswerable": {"cases": unanswerable_count, "note": "无答案问题不计入主指标；当前系统无'无可回答'闸门，会返回最相关 chunk。"},
        "details": details,
    }


@contextmanager
def _temporary_runtime_overrides(
    *, candidate_k: int | None = None, weight_overrides: dict[str, float] | None = None,
):
    settings = get_settings()
    fields = ["rag_candidate_k"]
    original = {field: getattr(settings, field) for field in fields}
    if weight_overrides:
        for field in _WEIGHT_FIELDS:
            original[field] = getattr(settings, f"rag_{field}")
    try:
        if candidate_k is not None:
            if candidate_k < 1:
                raise ValueError("candidate_k must be at least 1")
            settings.rag_candidate_k = int(candidate_k)
        for field, value in (weight_overrides or {}).items():
            if field not in _WEIGHT_FIELDS:
                raise ValueError(f"unknown RAG weight field: {field}")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError("RAG weights must be finite and non-negative")
            setattr(settings, f"rag_{field}", numeric)
        yield
    finally:
        settings.rag_candidate_k = original["rag_candidate_k"]
        for field in _WEIGHT_FIELDS:
            if field in original:
                setattr(settings, f"rag_{field}", original[field])


def evaluate(
    strategy: str = "rule", top_k: int = 20, *, backend_override: str = "local_hybrid",
    candidate_k: int | None = None, weight_overrides: dict[str, float] | None = None,
) -> dict:
    """Evaluate a concrete backend/runtime configuration without persisting it."""

    with _temporary_runtime_overrides(
        candidate_k=candidate_k, weight_overrides=weight_overrides,
    ):
        return _evaluate_once(strategy, top_k=top_k, backend_override=backend_override)


def _summary(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "details"}


def _runtime_config(
    backend_override: str | None = None, *, top_k: int | None = None,
    candidate_k: int | None = None,
) -> dict:
    """Capture every runtime parameter that can change ranking behavior."""

    settings = get_rag_settings()
    app_settings = get_settings()
    effective_top_k = top_k if top_k is not None else settings.top_k
    effective_candidate_k = max(
        effective_top_k,
        candidate_k if candidate_k is not None else settings.candidate_k,
    )
    return {
        "backend": backend_override or settings.backend,
        "collection": settings.collection_name,
        "index_version": settings.index_version,
        "embedding_model": settings.embedding_model,
        "embedding_model_revision": settings.embedding_model_revision,
        "chunk_pipeline_version": settings.chunk_pipeline_version,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": effective_top_k,
        "candidate_k": effective_candidate_k,
        "reranker_top_k": settings.reranker_top_k,
        "reranker_strategy": settings.reranker_strategy,
        "reranker_enabled": app_settings.rag_reranker_enabled,
        "reranker_model": app_settings.rag_reranker_model,
        "reranker_timeout_ms": app_settings.rag_reranker_timeout_ms,
        "vector_weight": settings.vector_weight,
        "sparse_weight": settings.sparse_weight,
        "semantic_weight": settings.semantic_weight,
        "metadata_weight": settings.metadata_weight,
    }


def evaluate_all(top_k: int = 20, *, backend_override: str = "local_hybrid") -> dict:
    """Run the versioned retrieval matrix and retain strategy-specific detail."""
    # This is a deterministic offline evaluation, not a model download path.
    # A missing reranker must expose its structured fallback, rather than make
    # the evidence run depend on network availability.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return {strategy: evaluate(strategy, top_k=top_k, backend_override=backend_override) for strategy in ("rule", "cross_encoder", "rrf")}


_WEIGHT_FIELDS = ("vector_weight", "sparse_weight", "semantic_weight", "metadata_weight")


def _runtime_config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def evaluate_weight_grid(
    weight_sets: list[dict[str, float]],
    *,
    strategy: str = "rule",
    top_k: int = 20,
    backend_override: str = "local_hybrid",
) -> list[dict]:
    """Evaluate explicit runtime weight candidates without changing defaults.

    Each candidate is applied only for the duration of its evaluation and the
    original settings are restored in a ``finally`` block.  The returned
    record contains the complete runtime configuration and hash, so a winner
    can be reproduced and bound to a release manifest instead of being copied
    into production by accident.
    """

    settings = get_settings()
    original = {field: getattr(settings, f"rag_{field}") for field in _WEIGHT_FIELDS}
    results: list[dict] = []
    try:
        for index, candidate in enumerate(weight_sets, start=1):
            unknown = set(candidate) - set(_WEIGHT_FIELDS)
            if unknown:
                raise ValueError(f"unknown RAG weight fields: {sorted(unknown)}")
            weights = dict(original)
            weights.update({key: float(value) for key, value in candidate.items()})
            if any(not math.isfinite(value) or value < 0 for value in weights.values()):
                raise ValueError("RAG weights must be finite and non-negative")
            for field, value in weights.items():
                setattr(settings, f"rag_{field}", value)
            runtime = _runtime_config(backend_override, top_k=top_k)
            evaluation = evaluate(
                strategy=strategy,
                top_k=top_k,
                backend_override=backend_override,
            )
            results.append({
                "candidate_id": f"weights-{index}",
                "weights": weights,
                "runtime_config": runtime,
                "runtime_config_hash": _runtime_config_hash(runtime),
                "evaluation": evaluation,
            })
    finally:
        for field, value in original.items():
            setattr(settings, f"rag_{field}", value)
    return results


def write_weight_grid(
    path: Path,
    weight_sets: list[dict[str, float]],
    *,
    strategy: str = "rule",
    top_k: int = 20,
    backend_override: str = "local_hybrid",
) -> tuple[Path, dict]:
    """Write an immutable-input weight experiment artifact."""

    dataset_bytes = QUERIES.read_bytes()
    payload = {
        "runner": "evaluation.retrieval.runner",
        "dataset_ref": str(QUERIES.relative_to(ROOT)),
        "dataset_hash": hashlib.sha256(dataset_bytes).hexdigest(),
        "strategy": strategy,
        "top_k": top_k,
        "backend": backend_override,
        "results": evaluate_weight_grid(
            weight_sets,
            strategy=strategy,
            top_k=top_k,
            backend_override=backend_override,
        ),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target, payload


def write_results(path: Path | None = None, *, top_k: int = 20, backend_override: str = "local_hybrid") -> tuple[Path, dict]:
    """Persist an immutable-input retrieval comparison artifact for evidence."""
    target = path or ROOT / "data" / "quality" / "rag-retrieval-latest.json"
    dataset_bytes = QUERIES.read_bytes()
    payload = {
        "runner": "evaluation.retrieval.runner",
        "dataset_ref": str(QUERIES.relative_to(ROOT)),
        "dataset_hash": hashlib.sha256(dataset_bytes).hexdigest(),
        "top_k": top_k,
        "backend": backend_override,
        "runtime_config": _runtime_config(backend_override, top_k=top_k),
        "results": evaluate_all(top_k=top_k, backend_override=backend_override),
    }
    payload["runtime_config_hash"] = hashlib.sha256(
        json.dumps(payload["runtime_config"], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target, payload


if __name__ == "__main__":
    output, payload = write_results()
    for strategy, result in payload["results"].items():
        print(f"{strategy.upper()}:", _summary(result))
    print(f"evidence={output}")
