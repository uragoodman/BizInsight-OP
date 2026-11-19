"""检索评测指标单测：Doc/Chunk Hit@K、MRR@5、语料扩展性守卫。"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.retrieval import runner
from evaluation.retrieval.metrics import chunk_hit_at_k, doc_hit_at_k, mrr_at_k, ndcg_at_k


def _hit(source_path: str, text: str):
    return type("H", (), {"citation": type("C", (), {"source_path": source_path})(), "text": text})()


DOC = "docs/knowledge/metrics/gmv.md"


def test_doc_hit_at_k() -> None:
    hits = [_hit("docs/knowledge/metrics/order_count.md", "x"), _hit(DOC, "y")]
    case = {"expected_doc_ids": [DOC]}
    assert doc_hit_at_k(hits, case, 5) is True
    assert doc_hit_at_k(hits[:1], case, 5) is False


def test_chunk_hit_requires_snippet() -> None:
    hits = [_hit(DOC, "GMV 为支付订单商品金额之和")]
    case = {"expected_doc_ids": [DOC], "expected_chunk_snippets": ["取消"]}
    assert chunk_hit_at_k(hits, case, 5) is False  # 文档对但内容锚点不匹配
    hits2 = [_hit(DOC, "取消订单是否计入 GMV")]
    assert chunk_hit_at_k(hits2, case, 5) is True


def test_mrr_rank() -> None:
    hits = [_hit("a.md", "x"), _hit(DOC, "y")]
    case = {"expected_doc_ids": [DOC]}
    assert mrr_at_k(hits, case, 5) == 0.5  # 第二个命中 → 1/2
    assert mrr_at_k(hits[:1], case, 5) == 0.0


def test_ndcg_rewards_correct_evidence_at_the_top() -> None:
    case = {"expected_doc_ids": [DOC], "expected_chunk_snippets": ["取消"]}
    ranked = [_hit(DOC, "取消订单"), _hit("other.md", "x")]
    delayed = [_hit("other.md", "x"), _hit(DOC, "取消订单")]
    assert ndcg_at_k(ranked, case, 5) > ndcg_at_k(delayed, case, 5)


def test_eval_set_docs_exist_in_corpus(monkeypatch) -> None:
    """扩展性守卫：评测集引用的文档必须存在于语料，缺失则报错（不静默）。"""
    monkeypatch.setattr(runner, "QUERIES", Path(__file__).parent.parent / "evaluation" / "retrieval" / "queries.jsonl")
    cases = runner._load_cases()
    assert len(cases) >= 30
    # 无答案用例 expected_doc_ids 为空，不触发守卫
    with_snippets = [c for c in cases if c.get("expected_chunk_snippets")]
    assert len(with_snippets) >= 8  # 至少有 8 条 chunk 级硬用例


def test_write_results_persists_the_strategy_matrix(monkeypatch, tmp_path: Path) -> None:
    matrix = {"rule": {"mrr@5": 0.5}, "cross_encoder": {"mrr@5": 0.5}, "rrf": {"mrr@5": 0.5}}
    monkeypatch.setattr(runner, "evaluate_all", lambda **_kwargs: matrix)
    output, payload = runner.write_results(tmp_path / "retrieval.json")
    assert output.exists()
    assert payload["results"] == matrix
    assert len(payload["dataset_hash"]) == 64
    assert payload["runtime_config"]["candidate_k"] >= payload["runtime_config"]["top_k"]
    assert len(payload["runtime_config_hash"]) == 64


def test_weight_grid_restores_defaults_and_records_runtime_hash(monkeypatch) -> None:
    settings = runner.get_settings()
    original = settings.rag_vector_weight
    monkeypatch.setattr(
        runner,
        "evaluate",
        lambda strategy, top_k, backend_override: {
            "strategy": strategy, "top_k": top_k, "backend": backend_override,
        },
    )
    results = runner.evaluate_weight_grid(
        [{"vector_weight": 0.8}, {"sparse_weight": 0.9}], top_k=10,
    )
    assert len(results) == 2
    assert all(len(item["runtime_config_hash"]) == 64 for item in results)
    assert settings.rag_vector_weight == original
