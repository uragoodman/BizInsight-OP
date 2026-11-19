from __future__ import annotations
"""RAG 层结构化数据模型。

本文件只定义数据契约，不包含索引、检索或数据库访问逻辑。
它的意义是让 RAG 层、Graph 节点、Critic Agent、Harness 评测和 trace 存储
都消费同一套稳定结构，避免在不同模块之间传递不可解释的普通字符串。

典型使用场景：
- `app.rag.indexing` 加载知识文档时，会生成 `KnowledgeDocument` 和 `KnowledgeChunk`。
- `app.rag.retriever` 检索知识库时，会返回 `RagSearchResult`。
- `app.graph.nodes.metric_rag` 会把 `RagSearchResult` 写入 workflow state 和 trace。
- `app.agents.critic_agent` 会读取 `RagSearchResult` 的 citation 和 signals 做证据审核。
"""

from typing import Literal

from pydantic import BaseModel, Field


KnowledgeDocType = Literal["metric", "sop", "template", "unknown"]
RagRetrievalStrategy = Literal["local_hybrid", "qdrant_vector", "qdrant_hybrid", "fallback"]


class KnowledgeDocument(BaseModel):
    """知识库原始文档模型。

    一个 `KnowledgeDocument` 对应 `docs/knowledge` 下的一篇 markdown 文件。
    它用于描述“整篇文档”的元数据和原文内容，是切分 chunk 之前的输入结构。

    字段说明：
    - `doc_id`：文档唯一标识，当前使用文档路径。
    - `title`：文档标题，通常来自 markdown 一级标题。
    - `text`：文档完整文本。
    - `source_path`：文档来源路径，用于 citation 和 trace。
    - `doc_type`：知识类型，如指标、SOP、报告模板。
    - `metric_name`：指标文档对应的标准指标名，非指标文档通常为空。
    - `tags`：辅助检索标签，用于本地 hybrid 检索和后续 metadata filter。
    - `document_hash`：文档内容 hash，用于增量索引判断文档是否变化。
    - `index_version`：知识库索引版本，用于 trace 回放和未来版本切换。
    """

    doc_id: str
    title: str
    text: str
    source_path: str
    doc_type: KnowledgeDocType = "unknown"
    metric_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    document_hash: str | None = None
    index_version: str | None = None
    visibility: Literal["global", "tenant"] = "global"
    tenant_id: str | None = None
    classification: Literal["public", "internal", "confidential"] = "internal"
    allowed_roles: list[str] = Field(default_factory=list)


class KnowledgeChunk(BaseModel):
    """知识库检索 chunk 模型。

    一个 `KnowledgeChunk` 对应一篇 markdown 文档中的一个可检索片段。
    当前按 markdown section 切分，用于提高检索粒度，使 citation 能定位到更具体的证据。

    主要被 `app.rag.indexing.load_knowledge_chunks` 生成，
    再被 `app.rag.indexing.build_qdrant_index` 写入 Qdrant，
    或被 `app.rag.retriever._retrieve_local` 用作本地降级检索的候选集。

    字段说明：
    - `chunk_id`：chunk 唯一标识，格式为 `文档路径#chunk-N`。
    - `doc_id`：所属原始文档 ID。
    - `title`：所属文档标题。
    - `text`：chunk 文本内容。
    - `source_path`：所属文档路径。
    - `doc_type`：知识类型。
    - `metric_name`：关联指标名。
    - `tags`：辅助检索标签。
    - `section`：markdown 二级标题，便于引用时说明证据位置。
    - `document_hash`：所属文档内容 hash。
    - `index_version`：chunk 写入的知识库索引版本。
    """

    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_path: str
    doc_type: KnowledgeDocType = "unknown"
    metric_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    section: str | None = None
    document_hash: str | None = None
    index_version: str | None = None
    # ``pipeline_version`` identifies the chunking contract.  ``content_hash``
    # detects a changed final chunk but is deliberately not used as its ID;
    # citations need a stable identifier within a pipeline version.
    pipeline_version: str | None = None
    content_hash: str | None = None
    source_start: int | None = None
    source_end: int | None = None
    visibility: Literal["global", "tenant"] = "global"
    tenant_id: str | None = None
    classification: Literal["public", "internal", "confidential"] = "internal"
    allowed_roles: list[str] = Field(default_factory=list)


class RagCitation(BaseModel):
    """RAG 引用信息模型。

    `RagCitation` 是最终回答和 trace 中最重要的证据结构。
    它不保存完整 chunk 文本，只保存“证据来自哪里”，用于审计、展示和 Critic 校验。

    字段说明：
    - `source_path`：来源文档路径。
    - `title`：来源文档标题。
    - `doc_type`：来源知识类型。
    - `metric_name`：关联指标名。
    - `chunk_id`：命中的 chunk 标识。
    - `section`：命中的文档章节。
    - `index_version`：命中证据所属的知识库索引版本。
    """

    source_path: str
    title: str
    doc_type: KnowledgeDocType
    metric_name: str | None = None
    chunk_id: str | None = None
    section: str | None = None
    index_version: str | None = None
    source_start: int | None = None
    source_end: int | None = None


class RagRetrievalSignals(BaseModel):
    """RAG 检索质量信号模型。

    该模型用于记录一次命中的评分构成。
    本地 `local_hybrid` 检索会填充 sparse、semantic、metadata 和 rerank 分数；
    Qdrant 向量检索当前主要填充 `rerank_score`，后续接入 dense+sparse hybrid 后可继续扩展。

    字段说明：
    - `sparse_score`：关键词/词项匹配分。
    - `semantic_score`：别名、字符 n-gram 等轻量语义匹配分。
    - `metadata_score`：doc_type、metric_name、section 等元数据加权分。
    - `vector_score`：Qdrant 向量相似度分，用于 Qdrant 召回后的重排。
    - `rerank_score`：融合后的最终排序分。
    - `matched_terms`：命中的查询词项，用于 trace 和调试。
    """

    sparse_score: float = 0.0
    semantic_score: float = 0.0
    metadata_score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float = 0.0
    cross_encoder_score: float = 0.0  # P3.3：cross-encoder 重排相关分（无模型时为 0）
    matched_terms: list[str] = Field(default_factory=list)


class RagSearchHit(BaseModel):
    """RAG 单条命中结果模型。

    一个 `RagSearchHit` 表示检索返回的一个 chunk。
    它同时包含 chunk 文本、最终得分、引用信息和质量信号，
    是 `RagSearchResult.hits` 列表中的基本单位。

    字段说明：
    - `chunk_id`：命中的 chunk ID。
    - `text`：命中的 chunk 文本。
    - `score`：最终排序分。
    - `citation`：引用来源。
    - `signals`：检索质量信号。
    """

    chunk_id: str
    text: str
    score: float
    citation: RagCitation
    signals: RagRetrievalSignals = Field(default_factory=RagRetrievalSignals)


class RagRetrievalDiagnostics(BaseModel):
    """一次检索的执行诊断信息。

    诊断字段与 ``filters`` 分离：``filters`` 描述查询约束，
    ``RagRetrievalDiagnostics`` 描述请求实际如何执行以及是否发生降级。
    所有字段都提供安全默认值，保证旧调用方可以继续消费 ``RagSearchResult``。
    """

    degraded: bool = False
    fallback_from: str | None = None
    degraded_reason: str | None = None
    exception_type: str | None = None
    tenant_private_knowledge_status: str | None = None
    context_incomplete: bool = False
    requested_metric_name: str | None = None
    resolved_metric_name: str | None = None
    metric_resolution_source: str = "none"
    metric_resolution_conflict: bool = False
    reranker_degraded: bool = False
    reranker_reason: str | None = None
    reranker_latency_ms: float | None = None
    # ``legacy_unverified`` is a controlled compatibility state for an old
    # collection whose manifest predates fingerprints.  A real mismatch is a
    # hard retrieval failure, never a silent fallback to an incompatible index.
    index_compatibility: str = "verified"
    index_fingerprint_digest: str | None = None


class RagSearchResult(BaseModel):
    """RAG 检索总结果模型。

    `retrieve_knowledge` 的标准返回类型。
    workflow 节点会把它转换为 dict 写入 state 和 trace，
    answer/analysis 节点消费其中的 `context`，
    Critic/Harness 消费其中的 `hits`、`citations`、`strategy` 和 `filters`。

    字段说明：
    - `query`：原始查询文本。
    - `backend`：实际使用的后端，如 `qdrant` 或 `local`。
    - `strategy`：检索策略，如 `qdrant_vector` 或 `local_hybrid`。
    - `top_k`：最多返回的命中条数。
    - `filters`：本次检索使用的过滤条件和权重信息。
    - `index_version`：本次检索面向的知识库索引版本。
    - `hits`：命中的 chunk 列表。
    - `context`：拼接后的上下文文本，供下游分析/回答使用。
    """

    query: str
    backend: str
    strategy: RagRetrievalStrategy = "local_hybrid"
    top_k: int
    filters: dict[str, object] = Field(default_factory=dict)
    diagnostics: RagRetrievalDiagnostics | None = None
    index_version: str | None = None
    hits: list[RagSearchHit] = Field(default_factory=list)
    context: str = ""

    @property
    def citations(self) -> list[RagCitation]:
        """返回所有命中结果中的 citation 列表。

        参数：无。
        返回值：`list[RagCitation]`，顺序与 `hits` 保持一致。
        使用场景：`metric_rag` 节点把 citation 单独写入 `state["rag_citations"]`，
        方便 trace 页面和 Critic Agent 直接读取。
        """

        return [hit.citation for hit in self.hits]
