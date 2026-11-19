"""BM25 风格的稀疏特征生成，供 Qdrant native sparse 向量使用。

Qdrant 支持在同一个 collection 里同时保存 dense 向量与 sparse 向量，并在查询时用
RRF/权重融合做 native hybrid 检索。本模块只负责把 chunk/query 文本转换成稀疏特征
``{"indices": [...], "values": [...]}``，不依赖任何外部稀疏模型，因此离线可复现。

词表在索引构建阶段从全部 chunk 中收集（按词频排序并截断），IDF 随之固定并持久化；
查询阶段必须加载同一份词表，才能保证稀疏向量的维度语义一致。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    """把文本拆成检索词项：英文/数字/下划线整词 + 单个中文字符。"""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def chunk_search_text(chunk: object) -> str:
    """Return the versioned text surface shared by native sparse and local search."""

    return "\n".join(
        [
            str(getattr(chunk, "title", "") or ""),
            str(getattr(chunk, "section", "") or ""),
            str(getattr(chunk, "text", "") or ""),
            " ".join(str(tag) for tag in (getattr(chunk, "tags", None) or [])),
        ]
    )


def build_vocab(
    chunks: list[object],
    *,
    min_df: int = 1,
    max_vocab: int = 10_000,
) -> tuple[dict[str, int], dict[int, float]]:
    """从全部 chunk 收集全局稀疏词表与 IDF。

    参数：
    - `chunks`：可迭代的 chunk，每个对象必须有 `.text` 属性（如 `KnowledgeChunk`）。
    - `min_df`：词项至少出现的文档数，低于该值不进词表。
    - `max_vocab`：词表大小上限，按词频降序截断。

    返回值：
    - `vocab`：词项 → 全局索引的映射。
    - `idf`：全局索引 → 逆文档频率权重。
    """
    doc_freq: Counter[str] = Counter()
    doc_count = 0
    for chunk in chunks:
        doc_count += 1
        doc_freq.update(set(tokenize(chunk_search_text(chunk))))

    terms = sorted(
        (term for term, freq in doc_freq.items() if freq >= min_df),
        key=lambda term: (-doc_freq[term], term),
    )
    vocab = {term: index for index, term in enumerate(terms[:max_vocab])}
    idf: dict[int, float] = {}
    for term, index in vocab.items():
        freq = doc_freq[term]
        idf[index] = math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
    return vocab, idf


def sparse_embed(
    text: str,
    vocab: dict[str, int],
    idf: dict[int, float],
) -> dict[str, list[int] | list[float]]:
    """把文本编码为 Qdrant 稀疏向量 ``{"indices": [...], "values": [...]}``。

    权重为 ``(1 + log(tf)) * idf``，未出现在词表中的词项被忽略。
    """
    tf = Counter(tokenize(text))
    indices: list[int] = []
    values: list[float] = []
    for term, count in tf.items():
        index = vocab.get(term)
        if index is None:
            continue
        weight = (1.0 + math.log(count)) * idf[index]
        if weight > 0:
            indices.append(index)
            values.append(round(weight, 6))
    return {"indices": indices, "values": values}


def save_sparse_model(
    vocab: dict[str, int],
    idf: dict[int, float],
    path: Path | str,
) -> None:
    """持久化词表与 IDF，供查询阶段加载使用。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "vocab": vocab,
        "idf": {str(index): weight for index, weight in idf.items()},
    }
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_sparse_model(
    path: Path | str,
) -> tuple[dict[str, int], dict[int, float]] | None:
    """加载之前持久化的稀疏模型；文件缺失或损坏时返回 ``None``。"""
    source = Path(path)
    if not source.exists():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    vocab = payload.get("vocab") or {}
    idf = {int(index): float(weight) for index, weight in (payload.get("idf") or {}).items()}
    return vocab, idf
