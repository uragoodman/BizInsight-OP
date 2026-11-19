"""记忆嵌入：bge-small-zh 单例嵌入器，带优雅降级。

- 语义召回依赖真实 embedding；模型不可用时返回 None，调用方降级到确定性过滤 + 时间排序
  （与 RAG local_hybrid 的降级哲学一致）。
- 模型以模块级单例懒加载，避免每次检索都重新加载。
- 离线模式复用 RAG 的 llama_index 缓存目录（data/cache/llama_index）。
"""

from __future__ import annotations

import os
import threading
from typing import Any

from app.core.config import get_settings

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_VERSION = "v1"
EMBEDDING_DIM = 512

_lock = threading.Lock()
_embedder: Any | None = None
_embedder_failed = False


def _load_embedder() -> Any | None:
    """懒加载 SentenceTransformer 单例；加载失败后不再重试（避免每次检索都尝试）。"""
    global _embedder, _embedder_failed
    if _embedder is not None or _embedder_failed:
        return _embedder
    with _lock:
        if _embedder is not None or _embedder_failed:
            return _embedder
        try:
            # 复用 RAG 的缓存目录：模型快照已存在（llama-index 与 HF hub 同布局）。
            os.environ.setdefault("HF_HUB_CACHE", str(get_settings().rag_cache_dir))
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer(get_settings().rag_embedding_model or EMBEDDING_MODEL)
        except Exception:  # noqa: BLE001 - 嵌入不可用时降级，不阻塞主链
            _embedder_failed = True
            _embedder = None
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """把文本列表编码为归一化向量（unit-length → 余弦 = 点积）。

    返回值：`list[list[float]] | None`——模型不可用时返回 None，调用方降级到
    确定性过滤 + 时间排序；空列表返回空列表。
    """
    if not texts:
        return []
    embedder = _load_embedder()
    if embedder is None:
        return None
    try:
        vectors = embedder.encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in vectors]
    except Exception:  # noqa: BLE001 - 单次编码失败降级，不重试
        return None
