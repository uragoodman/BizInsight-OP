from __future__ import annotations
"""RAG 配置与运行时环境初始化。

本文件负责把 `.env` / `app.core.config.Settings` 中的配置转换成 RAG 层内部使用的
`RagSettings`，并在调用 LlamaIndex、HuggingFace embedding、Qdrant 前设置必要的
运行时环境变量。

典型使用场景：
- `app.rag.indexing.build_qdrant_index` 在写入 Qdrant 前调用 `configure_rag_runtime`。
- `app.rag.retriever._retrieve_from_qdrant` 在连接 Qdrant 检索前调用 `configure_rag_runtime`。
- 本地开发时，如果 embedding 模型已经下载到 `data/cache/llama_index`，
  本文件会开启离线优先模式，减少 HuggingFace 网络探测导致的不稳定。
"""

from dataclasses import dataclass
import os
from pathlib import Path

from app.core.config import get_settings


@dataclass(frozen=True)
class RagSettings:
    """RAG 层运行配置。

    该类是只读 dataclass，用于集中表达 RAG 检索和索引所需的配置。
    它不直接读取环境变量，而是由 `get_rag_settings` 从全局 Settings 中构造。

    字段说明：
    - `knowledge_root`：知识库 markdown 根目录。
    - `collection_name`：Qdrant collection 名。
    - `embedding_model`：HuggingFace embedding 模型名。
    - `top_k`：默认返回命中条数。
    - `cache_dir`：LlamaIndex/HuggingFace 模型缓存目录。
    - `offline_embeddings`：是否在已有缓存时优先使用离线 embedding。
    - `index_version`：当前知识库索引版本，会写入 Qdrant metadata 和 trace。
    - `manifest_path`：增量索引 manifest 路径，用于记录文档 hash 和索引状态。
    - `candidate_k`：本地 hybrid 检索候选规模预留参数。
    - `vector_weight`：Qdrant 向量相似度分在重排中的权重。
    - `sparse_weight`：稀疏关键词分权重。
    - `semantic_weight`：轻量语义分权重。
    - `metadata_weight`：元数据加权分权重。
    - `backend`：当前 RAG 后端，如 `local_hybrid`、`qdrant_vector`。
    """

    knowledge_root: str
    collection_name: str
    active_alias: str
    embedding_model: str
    embedding_model_revision: str
    top_k: int
    cache_dir: str
    offline_embeddings: bool
    index_version: str
    chunk_pipeline_version: str
    chunk_size: int
    chunk_overlap: int
    manifest_path: str
    candidate_k: int
    vector_weight: float
    sparse_weight: float
    semantic_weight: float
    metadata_weight: float
    backend: str
    reranker_top_k: int
    reranker_strategy: str

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("rag_top_k must be at least 1")
        if self.candidate_k < 1:
            raise ValueError("rag_candidate_k must be at least 1")
        if self.chunk_size < 1:
            raise ValueError("rag_chunk_size must be at least 1")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("rag_chunk_overlap must be >= 0 and smaller than rag_chunk_size")
        if self.chunk_pipeline_version not in {"section-v1", "sentence-v2"}:
            raise ValueError(f"unsupported rag_chunk_pipeline_version: {self.chunk_pipeline_version}")
        if self.reranker_top_k < 0:
            raise ValueError("rag_reranker_top_k must be 0 or greater")
        if self.reranker_strategy not in {"rule", "cross_encoder", "rrf"}:
            raise ValueError(f"unsupported rag_reranker_strategy: {self.reranker_strategy}")


def get_rag_settings() -> RagSettings:
    """读取全局配置并转换成 RAG 专用配置。

    参数：无。
    返回值：`RagSettings`，包含知识库路径、Qdrant collection、embedding 模型、
    检索权重和后端策略。

    使用场景：
    - 轻量函数只需要读取配置时调用它，例如本地检索读取 `top_k` 和权重。
    - 如果函数还需要初始化 LlamaIndex/HuggingFace 运行环境，应调用
      `configure_rag_runtime`，而不是直接调用本函数。
    """

    settings = get_settings()
    return RagSettings(
        knowledge_root=settings.rag_knowledge_root,
        collection_name=settings.qdrant_collection,
        active_alias=settings.rag_qdrant_active_alias.strip(),
        embedding_model=settings.rag_embedding_model,
        embedding_model_revision=settings.rag_embedding_model_revision,
        top_k=settings.rag_top_k,
        cache_dir=settings.rag_cache_dir,
        offline_embeddings=settings.rag_offline_embeddings,
        index_version=settings.rag_index_version,
        chunk_pipeline_version=settings.rag_chunk_pipeline_version,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        manifest_path=settings.rag_manifest_path,
        candidate_k=settings.rag_candidate_k,
        vector_weight=settings.rag_vector_weight,
        sparse_weight=settings.rag_sparse_weight,
        semantic_weight=settings.rag_semantic_weight,
        metadata_weight=settings.rag_metadata_weight,
        backend=settings.rag_backend,
        reranker_top_k=settings.rag_reranker_top_k,
        reranker_strategy=settings.rag_reranker_strategy,
    )


def configure_rag_runtime() -> RagSettings:
    """配置 RAG 运行时环境，并返回 RAG 配置。

    参数：无。
    返回值：`RagSettings`。

    具体操作：
    1. 读取 RAG 配置。
    2. 设置 `LLAMA_INDEX_CACHE_DIR`，让 LlamaIndex/HuggingFace 缓存落在项目目录。
    3. 如果 `offline_embeddings=True` 且缓存目录已存在，设置
       `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，优先使用本地模型缓存。

    使用场景：
    - 建立 Qdrant 索引前调用，避免模型缓存写到系统用户目录。
    - Qdrant 检索前调用，避免每次请求都尝试联网探测 HuggingFace。
    """

    rag_settings = get_rag_settings()
    os.environ.setdefault("LLAMA_INDEX_CACHE_DIR", rag_settings.cache_dir)
    if rag_settings.offline_embeddings and Path(rag_settings.cache_dir).exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return rag_settings
