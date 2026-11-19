"""RAG 检索评测指标：Doc/Chunk Hit@K、MRR/nDCG、P95 延迟。

- Doc 级：只看是否命中正确文档（source_path）。
- Chunk 级：命中正确文档且文本包含内容锚点（expected_chunk_snippets）——
  内容锚点对 chunk 重排/新增文档健壮，符合语料可扩展目标。
"""

from __future__ import annotations

from typing import Any


def _doc_matches(hit, expected_doc_ids: list[str]) -> bool:
    return any(hit.citation.source_path.endswith(path) for path in expected_doc_ids)


def _is_hit(hit, case: dict[str, Any]) -> bool:
    """一条命中是否"可回答"：文档正确，且（若给了内容锚点）文本包含之。"""
    if not _doc_matches(hit, case.get("expected_doc_ids", [])):
        return False
    snippets = case.get("expected_chunk_snippets") or []
    if not snippets:
        return True
    return any(snippet in hit.text for snippet in snippets)


def doc_hit_at_k(hits: list, case: dict[str, Any], k: int) -> bool:
    return any(_doc_matches(hit, case.get("expected_doc_ids", [])) for hit in hits[:k])


def chunk_hit_at_k(hits: list, case: dict[str, Any], k: int) -> bool:
    return any(_is_hit(hit, case) for hit in hits[:k])


def mrr_at_k(hits: list, case: dict[str, Any], k: int = 5) -> float:
    """第一个可回答命中的倒数排名；前 k 内未命中为 0。"""
    for index, hit in enumerate(hits[:k]):
        if _is_hit(hit, case):
            return 1.0 / (index + 1)
    return 0.0


def ndcg_at_k(hits: list, case: dict[str, Any], k: int = 5) -> float:
    """Binary/graded evidence nDCG for a ranked retrieval list.

    A chunk satisfying the expected content anchor has relevance 2, while a
    merely correct document has relevance 1.  The ideal ranking places the
    strongest available evidence first; this keeps the metric meaningful for
    both document-only and chunk-anchored cases.
    """

    import math

    if not case.get("expected_doc_ids"):
        return 0.0
    snippets = case.get("expected_chunk_snippets") or []

    def relevance(hit) -> float:
        if _is_hit(hit, case):
            return 2.0 if snippets else 1.0
        elif _doc_matches(hit, case.get("expected_doc_ids", [])):
            return 1.0
        return 0.0

    observed = [relevance(hit) for hit in hits]
    dcg = sum(
        value / math.log2(index + 2)
        for index, value in enumerate(observed[:k])
    )
    ideal = sum(
        value / math.log2(index + 2)
        for index, value in enumerate(sorted(observed, reverse=True)[:k])
    )
    return round(dcg / ideal, 6) if ideal else 0.0
