"""P3.3 重排策略单测：CE 排序、RRF 融合、降级、候选集合不变（安全边界）、配置关闭。

全部 mock 模型分数，不依赖真实模型/网络/数据库。
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from app.rag.reranker import RerankResult
from app.rag.retriever import _apply_rerank_strategy
from app.rag.schemas import RagCitation, RagRetrievalSignals, RagSearchHit


def _hit(chunk_id: str, text: str, score: float) -> RagSearchHit:
    return RagSearchHit(
        chunk_id=chunk_id, text=text, score=score,
        citation=RagCitation(source_path=f"docs/knowledge/metrics/{chunk_id}.md", title="", doc_type="metric",
                             metric_name=None, chunk_id=chunk_id, section=None, index_version="v1"),
        signals=RagRetrievalSignals(),
    )


def _result(hits: list[RagSearchHit]):
    return SimpleNamespace(hits=hits, context="", filters={}, top_k=len(hits), diagnostics=None)


def _mock_rerank(monkeypatch, scores, applied=True, degraded=False, reason=None):
    monkeypatch.setattr("app.rag.reranker.rerank", lambda query, candidates: RerankResult(
        scores=[float(s) for s in scores] if applied else None, applied=applied,
        degraded=degraded, reason=reason, latency_ms=1.0,
    ))


def test_cross_encoder_reorders_by_score(monkeypatch) -> None:
    hits = [_hit("a", "A 文本", 0.5), _hit("b", "B 文本", 0.5), _hit("c", "C 文本", 0.5)]
    _mock_rerank(monkeypatch, [0.2, 0.9, 0.5])  # 对应 a/b/c
    result = _apply_rerank_strategy(_result(hits), "q", "cross_encoder")
    assert [h.chunk_id for h in result.hits] == ["b", "c", "a"]
    assert result.diagnostics.reranker_degraded is False
    assert result.diagnostics.reranker_reason is None


def test_rrf_fusion_blends_ranks(monkeypatch) -> None:
    """RRF：规则排名 [a,b,c]（1,2,3）与 CE 排名 [b,c,a]（1,2,3）融合 → b 最稳。"""
    hits = [_hit("a", "A", 3.0), _hit("b", "B", 2.0), _hit("c", "C", 1.0)]  # 规则序 a>b>c
    _mock_rerank(monkeypatch, [0.2, 0.9, 0.5])  # CE 序 b>c>a
    result = _apply_rerank_strategy(_result(hits), "q", "rrf")
    # b 在两个列表都靠前 → RRF 最高
    assert result.hits[0].chunk_id == "b"
    assert result.diagnostics.reranker_degraded is False


def test_reranker_does_not_change_candidate_set(monkeypatch) -> None:
    """安全边界：reranker 只重排，不新增/删除候选——越权 chunk 不可能被'加'进来。"""
    hits = [_hit("a", "A", 1.0), _hit("b", "B", 0.5)]
    _mock_rerank(monkeypatch, [0.9, 0.1])
    before = {h.chunk_id for h in hits}
    result = _apply_rerank_strategy(_result(hits), "q", "cross_encoder")
    assert {h.chunk_id for h in result.hits} == before


def test_degradation_falls_back_to_rule(monkeypatch) -> None:
    hits = [_hit("a", "A", 3.0), _hit("b", "B", 2.0)]
    _mock_rerank(monkeypatch, None, applied=False, degraded=True, reason="model_unavailable")
    result = _apply_rerank_strategy(_result(hits), "q", "cross_encoder")
    # 顺序保持规则排序，且降级原因被记录
    assert [h.chunk_id for h in result.hits] == ["a", "b"]
    assert result.diagnostics.reranker_degraded is True
    assert result.diagnostics.reranker_reason == "model_unavailable"


def test_score_count_mismatch_is_a_structured_degradation(monkeypatch) -> None:
    hits = [_hit("a", "A", 3.0), _hit("b", "B", 2.0)]
    _mock_rerank(monkeypatch, [0.9])
    result = _apply_rerank_strategy(_result(hits), "q", "cross_encoder")
    assert [h.chunk_id for h in result.hits] == ["a", "b"]
    assert result.diagnostics.reranker_degraded is True
    assert result.diagnostics.reranker_reason == "score_count_mismatch"


def test_rule_strategy_is_noop(monkeypatch) -> None:
    hits = [_hit("a", "A", 3.0), _hit("b", "B", 2.0)]
    result = _apply_rerank_strategy(_result(hits), "q", "rule")
    assert [h.chunk_id for h in result.hits] == ["a", "b"]
    assert result.diagnostics is None


def test_reranker_not_loaded_when_disabled(monkeypatch) -> None:
    """配置关闭（enabled=False，未显式覆盖策略）时不加载/调用模型。"""
    hits = [_hit("a", "A", 3.0), _hit("b", "B", 2.0)]
    called = {"n": 0}
    monkeypatch.setattr("app.rag.reranker.rerank", lambda *a, **k: called.update(n=called["n"] + 1) or RerankResult(None, False, True, "disabled"))
    from app.core.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "rag_reranker_enabled", False)
    result = _apply_rerank_strategy(_result(hits), "q", None)
    assert called["n"] == 0  # 未调用模型
    assert [h.chunk_id for h in result.hits] == ["a", "b"]


def test_default_enabled_strategy_invokes_cross_encoder(monkeypatch) -> None:
    """默认配置启用 CE，检索没有显式覆盖 strategy 时会实际调用模型。"""
    hits = [_hit("a", "A", 3.0), _hit("b", "B", 2.0)]
    from app.core.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "rag_reranker_enabled", True)
    monkeypatch.setattr(settings, "rag_reranker_strategy", "cross_encoder")
    _mock_rerank(monkeypatch, [0.1, 0.9])

    result = _apply_rerank_strategy(_result(hits), "q", None)

    assert [h.chunk_id for h in result.hits] == ["b", "a"]
    assert result.diagnostics.reranker_degraded is False


def test_cached_pinned_model_loads_from_local_snapshot(monkeypatch, tmp_path) -> None:
    """A model cached before another HF model initializes still loads offline."""
    from app.core.config import get_settings
    from app.rag.reranker import Reranker

    settings = get_settings()
    model_id = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    revision = "test-revision"
    snapshot = tmp_path / "models--cross-encoder--mmarco-mMiniLMv2-L12-H384-v1" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(settings, "rag_cache_dir", str(tmp_path))
    monkeypatch.setattr(settings, "rag_reranker_model", model_id)
    monkeypatch.setattr(settings, "rag_reranker_model_revision", revision)
    calls = {}
    sentinel = object()

    def cross_encoder(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return sentinel

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=cross_encoder))
    reranker = Reranker()

    assert reranker._load() is sentinel
    assert calls == {"path": str(snapshot.resolve()), "kwargs": {}}


def test_reranker_uses_candidate_pool_before_final_top_k(monkeypatch) -> None:
    hits = [_hit("a", "A", 3.0), _hit("b", "B", 2.0), _hit("c", "C", 1.0)]
    result = _result(hits)
    result.top_k = 1
    from app.core.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "rag_reranker_top_k", 2)
    seen = {"count": 0}

    def rerank(_query, candidates):
        seen["count"] = len(candidates)
        return RerankResult([0.1, 0.9], True, False, None, 1.0)

    monkeypatch.setattr("app.rag.reranker.rerank", rerank)
    output = _apply_rerank_strategy(result, "q", "cross_encoder")
    assert seen["count"] == 2
    assert len(output.hits) == 1
    assert output.hits[0].chunk_id == "b"
