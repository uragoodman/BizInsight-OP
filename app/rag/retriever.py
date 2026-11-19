from __future__ import annotations
"""RAG 检索实现。

本文件是业务侧调用 RAG 的主入口，负责根据用户问题从知识库中召回相关 chunk，
并返回带 citation、检索策略、过滤条件和质量信号的结构化结果。

结构与职责：
- 本地 hybrid 检索：在 Qdrant 未启动或检索失败时使用，融合关键词、轻量语义和元数据分数。
- Qdrant 向量检索：在 `RAG_BACKEND=qdrant_vector` 时优先使用，读取已经入库的 Qdrant collection。
- 统一结果封装：无论后端是 Qdrant 还是本地降级，都返回 `RagSearchResult`。

使用者：
- `app.graph.nodes.metric_definition.retrieve_metric_definition` 在每次 chat workflow 中调用。
- `app.harness.evaluator` 调用它评估 RAG 命中率。
- 旧接口 `retrieve_metric_notes` 保留给兼容代码使用。
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal

from app.core.config import get_settings
from app.auth.context import AuthorizationContext, build_authorization_context
from app.auth.enforcement import (
    AuthorizationBoundary,
    StaleAuthorization,
    assert_authorization_boundary,
    coerce_authorization_boundary,
)
from app.rag.config import configure_rag_runtime, get_rag_settings
from app.rag.indexing import (
    _load_dense_embedding,
    _sparse_model_path,
    check_collection_fingerprint,
    load_knowledge_chunks,
    make_dense_embed_fn,
    resolve_collection_alias,
)
from app.rag.schemas import (
    KnowledgeChunk,
    KnowledgeDocType,
    RagCitation,
    RagRetrievalSignals,
    RagRetrievalDiagnostics,
    RagSearchHit,
    RagSearchResult,
)
from app.rag import sparse as sparse_features
from app.data.database import engine
from app.data.rls import set_local_context
from app.tenancy import TenantContext, current_tenant_context
from app.observability import record_event
from sqlalchemy import text


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
METRIC_ALIASES = {
    "gmv": {"gmv", "gross", "merchandise", "volume", "成交金额", "成交额", "销售额"},
    "order_count": {"order_count", "订单量", "订单数", "订单"},
    "avg_order_value": {"avg_order_value", "aov", "客单价", "平均订单金额"},
    "cancel_rate": {"cancel_rate", "取消率", "取消"},
    "customer_count": {"customer_count", "成交客户数", "客户数", "客户"},
    "repeat_purchase_rate": {"repeat_purchase_rate", "复购率", "回购率", "复购"},
    "delivery_delay_rate": {"delivery_delay_rate", "延迟送达率", "延期送达率", "延迟"},
    "on_time_delivery_rate": {"on_time_delivery_rate", "准时送达率", "按时送达率"},
    "bad_review_rate": {"bad_review_rate", "差评率", "低分评论率"},
    "freight_rate": {"freight_rate", "运费率", "运费占比"},
}


@dataclass(frozen=True)
class TenantPrivateKnowledgeRead:
    """Tenant-private knowledge read outcome.

    ``empty`` means the query succeeded and returned no rows; ``unavailable``
    means the application could not establish a trustworthy read result.
    """

    chunks: list[KnowledgeChunk]
    status: str
    error_code: str | None = None
    exception_type: str | None = None


@lru_cache(maxsize=1)
def _registry_metric_aliases() -> dict[str, set[str]]:
    """Build the runtime alias view from Metric Registry metadata.

    The legacy table remains a compatibility source while P4.1 is migrated,
    but Registry definitions are preferred whenever they are available.
    """

    aliases: dict[str, set[str]] = {}
    try:
        from app.metrics.registry import get_registry

        for definition in get_registry().list():
            aliases[definition.metric_id] = {
                str(item).strip().lower()
                for item in (
                    definition.metric_id,
                    definition.display_name,
                    *definition.aliases,
                )
                if str(item).strip()
            }
    except Exception:
        return {}
    return aliases


def _compact_metric_phrase(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value.casefold())


def _registry_metric_candidates(question: str) -> set[str]:
    folded = _compact_metric_phrase(question)
    candidates: set[str] = set()
    for metric_name, aliases in _registry_metric_aliases().items():
        if any(_compact_metric_phrase(alias) in folded for alias in aliases):
            candidates.add(metric_name)
    return candidates


def _legacy_infer_query_metric(question: str) -> str | None:
    lowered = question.lower()
    compact = lowered.replace(" ", "")
    matches: list[tuple[int, str]] = []
    for metric_name, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower in lowered or alias_lower in compact or alias in question:
                matches.append((len(alias_lower), metric_name))
    return max(matches)[1] if matches else None


def _resolve_metric_name(
    question: str,
    explicit_metric_name: str | None = None,
) -> tuple[str | None, str, bool]:
    """Resolve a metric and expose ambiguity between Registry and legacy aliases."""

    if explicit_metric_name:
        try:
            from app.metrics.registry import get_registry

            return get_registry().resolve(explicit_metric_name), "explicit", False
        except Exception:
            if explicit_metric_name in METRIC_ALIASES:
                return explicit_metric_name, "explicit", False
            return explicit_metric_name, "explicit", True

    registry_candidates = _registry_metric_candidates(question)
    legacy_metric = _legacy_infer_query_metric(question)
    if len(registry_candidates) > 1:
        return None, "conflict", True
    registry_metric = next(iter(registry_candidates), None)
    if registry_metric and legacy_metric and registry_metric != legacy_metric:
        return None, "conflict", True
    if registry_metric:
        return registry_metric, "inferred", False
    if legacy_metric:
        return legacy_metric, "inferred_legacy", False
    return None, "none", False


def _tokenize(text: str) -> list[str]:
    """将文本拆成检索词项。

    参数：
    - `text`：待分词文本，可以是用户问题、文档标题或 chunk 正文。

    返回值：
    - `list[str]`：小写后的英文/数字/下划线 token，以及单个中文字符 token。

    使用场景：
    - 本地 sparse 检索计算词项匹配分。
    - 生成 matched_terms 供 trace 调试。
    """

    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    """生成字符 n-gram 集合。

    参数：
    - `text`：输入文本。
    - `n`：n-gram 长度，默认 2。

    返回值：
    - `set[str]`：文本压缩空白并小写后的字符 n-gram 集合。

    使用场景：
    - 本地 semantic score 的轻量相似度计算。
    - 对中文短问句和指标名称相近表达提供弱语义召回能力。
    """

    compact = re.sub(r"\s+", "", text.lower())
    if len(compact) <= n:
        return {compact} if compact else set()
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def _infer_query_metric(question: str) -> str | None:
    """从用户问题中推断标准指标名。

    参数：
    - `question`：用户原始问题。

    返回值：
    - `str | None`：识别出的标准指标名，如 `gmv`、`order_count`；
      无法识别时返回 `None`。

    使用场景：
    - 本地检索时作为 metadata boost。
    - `retrieve_knowledge` 在调用方未显式传入 metric_name 时自动补全。
    """

    return _resolve_metric_name(question)[0]


def _expand_query_terms(question: str) -> set[str]:
    """扩展查询词项。

    参数：
    - `question`：用户原始问题。

    返回值：
    - `set[str]`：原始 token 加上指标别名扩展后的词项集合。

    具体操作：
    - 先调用 `_tokenize` 得到基础词项。
    - 如果能识别指标名，则加入标准指标名和英文别名。

    使用场景：
    - 本地 hybrid 检索提升“GMV/成交额/销售额”等同义表达的召回稳定性。
    """

    terms = set(_tokenize(question))
    metric_name = _infer_query_metric(question)
    if metric_name:
        terms.add(metric_name)
        aliases = _registry_metric_aliases().get(metric_name, set())
        aliases = aliases or METRIC_ALIASES.get(metric_name, set())
        for alias in aliases:
            if alias.isascii():
                terms.update(_tokenize(alias))
    return terms


def _sparse_score(query_terms: Counter[str], chunk: KnowledgeChunk) -> tuple[float, list[str]]:
    """计算关键词稀疏匹配分。

    参数：
    - `query_terms`：查询词项计数字典。
    - `chunk`：候选知识 chunk。

    返回值：
    - `tuple[float, list[str]]`：
      - 第一个元素是 sparse 分数。
      - 第二个元素是实际命中的查询词项列表。

    具体操作：
    - 对 chunk 的标题、section、正文、tags 做 token 统计。
    - 使用近似 TF 权重计算词项命中分。
    - 对标题命中和 metric_name 命中额外加权。

    使用场景：
    - 本地 `local_hybrid` 检索的 sparse 子分数。
    """

    text_terms = Counter(_tokenize(f"{chunk.title}\n{chunk.section or ''}\n{chunk.text}\n{' '.join(chunk.tags)}"))
    if not text_terms:
        return 0.0, []

    score = 0.0
    matched_terms: list[str] = []
    for term, query_count in query_terms.items():
        if term in text_terms:
            score += (1.0 + math.log(text_terms[term])) * query_count
            matched_terms.append(term)

    title_terms = set(_tokenize(chunk.title))
    score += sum(1.5 for term in query_terms if term in title_terms)
    if chunk.metric_name and chunk.metric_name.lower() in query_terms:
        score += 3.0
    return score, matched_terms


def _semantic_score(question: str, query_terms: set[str], chunk: KnowledgeChunk) -> float:
    """计算轻量语义匹配分。

    参数：
    - `question`：用户原始问题。
    - `query_terms`：扩展后的查询词项集合。
    - `chunk`：候选知识 chunk。

    返回值：
    - `float`：轻量语义分数。

    具体操作：
    - 如果问题推断出的指标名与 chunk 的 metric_name 一致，则加权。
    - 统计指标别名在 chunk 文本中的命中情况。
    - 使用字符 n-gram Jaccard 相似度补充中文短文本相似度。

    使用场景：
    - 在没有向量库或 Qdrant 失败时，让本地检索仍具备一定语义召回能力。
    """

    metric_name = _infer_query_metric(question)
    score = 0.0
    if metric_name and chunk.metric_name == metric_name:
        score += 3.0

    alias_hits = 0
    haystack = f"{chunk.title}\n{chunk.section or ''}\n{chunk.text}".lower()
    alias_catalog = dict(_registry_metric_aliases())
    for metric_name_key, aliases in METRIC_ALIASES.items():
        alias_catalog.setdefault(metric_name_key, aliases)
    for aliases in alias_catalog.values():
        if any(alias.lower() in query_terms for alias in aliases if alias.isascii()):
            alias_hits += sum(1 for alias in aliases if alias.lower() in haystack)
    score += min(alias_hits, 4) * 0.4

    query_ngrams = _char_ngrams(question)
    chunk_ngrams = _char_ngrams(f"{chunk.title}{chunk.section or ''}{chunk.text}")
    if query_ngrams and chunk_ngrams:
        score += len(query_ngrams & chunk_ngrams) / len(query_ngrams | chunk_ngrams)
    return score


def _metadata_score(chunk: KnowledgeChunk, doc_types: set[KnowledgeDocType] | None, metric_name: str | None) -> float:
    """计算元数据加权分。

    参数：
    - `chunk`：候选知识 chunk。
    - `doc_types`：调用方指定的知识类型过滤条件。
    - `metric_name`：调用方指定或自动识别出的指标名。

    返回值：
    - `float`：元数据加权分。

    使用场景：
    - Triage/Agent 已知道任务类型时，通过 doc_type 提升对应知识类型。
    - 指标已明确时，通过 metric_name 提升对应指标文档。
    """

    score = 0.0
    if doc_types and chunk.doc_type in doc_types:
        score += 1.0
    if metric_name and chunk.metric_name == metric_name:
        score += 2.0
    if chunk.section:
        score += 0.2
    return score


def _filter_chunks(
    chunks: Iterable[KnowledgeChunk],
    doc_types: set[KnowledgeDocType] | None,
    metric_name: str | None,
    tenant_id: str | None = None,
    authorization_context: AuthorizationContext | None = None,
) -> list[KnowledgeChunk]:
    """按 metadata 过滤候选 chunk。

    参数：
    - `chunks`：候选 chunk 可迭代对象。
    - `doc_types`：允许的知识类型集合。为空表示不过滤 doc_type。
    - `metric_name`：指定指标名。为空表示不过滤指标。

    返回值：
    - `list[KnowledgeChunk]`：过滤后的候选 chunk。

    使用场景：
    - 本地 `local_hybrid` 检索前缩小候选集。
    - 模拟 Qdrant metadata filter 的行为，保持两种后端语义一致。
    """

    filtered = []
    for chunk in chunks:
        if chunk.visibility == "tenant":
            if not tenant_id or chunk.tenant_id != tenant_id:
                continue
        elif chunk.visibility != "global":
            # Unknown visibility is not treated as public.  This is the local
            # equivalent of Qdrant's tenant payload filter.
            continue
        allowed_classifications = set(
            getattr(authorization_context, "allowed_classifications", ())
            or {"public", "internal"}
        )
        if chunk.classification not in allowed_classifications:
            continue
        if chunk.allowed_roles and (
            authorization_context is None
            or authorization_context.role not in chunk.allowed_roles
        ):
            continue
        if doc_types and chunk.doc_type not in doc_types:
            continue
        if metric_name and chunk.metric_name and chunk.metric_name != metric_name:
            continue
        filtered.append(chunk)
    return filtered


def _tenant_private_chunks(context: TenantContext | None) -> TenantPrivateKnowledgeRead:
    """Retrieve private knowledge through RLS with an explicit outcome state."""
    if context is None or context.scope_type == "platform":
        return TenantPrivateKnowledgeRead([], "not_applicable")
    try:
        with engine.begin() as connection:
            # Tenant-private knowledge must use the same transaction-local
            # context as every other governed database read.  In particular,
            # do not use connection-level SET here: pooled connections must
            # never carry the previous tenant into the next request.
            set_local_context(connection, context, scope_type="seller")
            rows = connection.execute(text("SELECT chunk_id,title,content,source_ref,doc_type,metric_name,classification,allowed_roles FROM governance.tenant_knowledge_chunks")).mappings().all()
    except Exception as exc:
        record_event("rag.private_knowledge.unavailable", {
            "rag.error_code": "tenant_knowledge_query_failed",
            "rag.exception_type": type(exc).__name__,
        })
        return TenantPrivateKnowledgeRead(
            [],
            "unavailable",
            error_code="tenant_knowledge_query_failed",
            exception_type=type(exc).__name__,
        )
    chunks = [
        KnowledgeChunk(
            chunk_id=f"tenant:{context.tenant_id}:{row['chunk_id']}",
            doc_id=str(row["chunk_id"]),
            title=row["title"],
            text=row["content"],
            source_path=row["source_ref"],
            doc_type=row["doc_type"],
            metric_name=row["metric_name"],
            tags=["tenant"],
            visibility="tenant",
            tenant_id=context.tenant_id,
            classification=row.get("classification") or "internal",
            allowed_roles=list(row.get("allowed_roles") or []),
        )
        for row in rows
    ]
    return TenantPrivateKnowledgeRead(chunks, "ok" if chunks else "empty")


def _hit_from_chunk(chunk: KnowledgeChunk, score: float, signals: RagRetrievalSignals) -> RagSearchHit:
    """将候选 chunk 转换为标准命中结果。

    参数：
    - `chunk`：命中的知识 chunk。
    - `score`：最终排序分。
    - `signals`：检索质量信号。

    返回值：
    - `RagSearchHit`：包含文本、citation 和 signals 的标准命中结果。

    使用场景：
    - 本地检索排序结束后构造 `RagSearchResult.hits`。
    """

    return RagSearchHit(
        chunk_id=chunk.chunk_id,
        text=chunk.text.strip(),
        score=round(score, 4),
        citation=RagCitation(
            source_path=chunk.source_path,
            title=chunk.title,
            doc_type=chunk.doc_type,
            metric_name=chunk.metric_name,
            chunk_id=chunk.chunk_id,
            section=chunk.section,
            index_version=chunk.index_version,
            source_start=chunk.source_start,
            source_end=chunk.source_end,
        ),
        signals=signals,
    )


def _rerank_chunk(
    question: str,
    chunk: KnowledgeChunk,
    *,
    doc_types: set[KnowledgeDocType] | None,
    metric_name: str | None,
    vector_score: float = 0.0,
) -> tuple[float, RagRetrievalSignals]:
    """对单个 chunk 计算统一重排分。

    参数：
    - `question`：用户原始问题。
    - `chunk`：候选知识 chunk。
    - `doc_types`：知识类型过滤/加权条件。
    - `metric_name`：指标名过滤/加权条件。
    - `vector_score`：Qdrant 返回的向量相似度分。本地检索没有该分数，默认 0。

    返回值：
    - `tuple[float, RagRetrievalSignals]`：
      - 第一个元素是最终 rerank_score。
      - 第二个元素记录各类子分数，供 trace、Critic 和调试使用。

    为什么需要：
    - 原来 Qdrant 路径直接使用向量相似度排序，业务术语、缩写、metadata 约束不够稳定。
    - 统一 rerank 后，Qdrant 候选会再次结合关键词、别名和 metadata 排序，
      相当于在 dense retrieval 后补了一层轻量 hybrid rerank。
    """

    rag_settings = get_rag_settings()
    expanded_terms = _expand_query_terms(question)
    query_terms = Counter(expanded_terms)
    sparse_score, matched_terms = _sparse_score(query_terms, chunk)
    semantic_score = _semantic_score(question, expanded_terms, chunk)
    metadata_score = _metadata_score(chunk, doc_types, metric_name)
    # The component functions intentionally expose raw diagnostic scores, but
    # their ranges differ: sparse TF can grow with repeated terms while
    # metadata and n-gram scores are bounded.  Convert each component to a
    # bounded relevance signal before applying runtime weights.
    sparse_component = sparse_score / (sparse_score + 5.0) if sparse_score > 0 else 0.0
    semantic_component = semantic_score / (semantic_score + 4.0) if semantic_score > 0 else 0.0
    metadata_component = min(max(metadata_score / 3.2, 0.0), 1.0)
    vector_component = min(max(vector_score, 0.0), 1.0)
    rerank_score = (
        rag_settings.vector_weight * vector_component
        + rag_settings.sparse_weight * sparse_component
        + rag_settings.semantic_weight * semantic_component
        + rag_settings.metadata_weight * metadata_component
    )
    signals = RagRetrievalSignals(
        sparse_score=round(sparse_component, 4),
        semantic_score=round(semantic_component, 4),
        metadata_score=round(metadata_component, 4),
        vector_score=round(vector_component, 4),
        rerank_score=round(rerank_score, 4),
        matched_terms=sorted(set(matched_terms)),
    )
    return rerank_score, signals


def _build_context(hits: list[RagSearchHit]) -> str:
    """把命中结果拼接成下游可消费的上下文文本。

    参数：
    - `hits`：RAG 命中结果列表。

    返回值：
    - `str`：包含编号、来源路径和 chunk 文本的上下文。

    使用场景：
    - Agent answer/report 节点消费 `metric_context`，并结合 citation 生成 grounded answer。
    """

    sections = []
    for index, hit in enumerate(hits, start=1):
        citation = hit.citation
        sections.append(
            "\n".join(
                [
                    f"[{index}] {citation.title}",
                    f"source={citation.source_path}",
                    hit.text,
                ]
            )
        )
    return "\n\n".join(sections)


_RRF_CONSTANT = 60


def _rrf_score(rule_rank: int, cross_encoder_rank: int) -> float:
    """RRF 排名融合：按排名而非原始分数融合，避免规则分与模型分不同量纲导致权重失真。"""
    return 1.0 / (_RRF_CONSTANT + rule_rank) + 1.0 / (_RRF_CONSTANT + cross_encoder_rank)


def _apply_rerank_strategy(result: RagSearchResult, question: str, strategy: str | None) -> RagSearchResult:
    """Apply a second-stage reranker to a sufficiently large candidate set.

    - `strategy`：None 时取配置（`rag_reranker_enabled` 关闭 → rule）。
    - `rule`：保持规则融合排序（基线）。
    - `cross_encoder`：按 CE 分数重新排序（纯 CE 排序能力）。
    - `rrf`：规则排名 + CE 排名做 RRF 融合（推荐，避免量纲失真）。
    - CE 不可用/异常/超时 → 结构化降级到规则排序，并在 diagnostics 记录 reason。
    - 安全边界：本函数只作用于已通过 RLS/租户/权限过滤的候选，模型无法引入越权 chunk。
    """
    settings = get_settings()
    resolved = strategy or (settings.rag_reranker_strategy if settings.rag_reranker_enabled else "rule")
    final_k = max(1, int(getattr(result, "top_k", len(result.hits))))
    if resolved == "rule" or not result.hits:
        result.hits = result.hits[:final_k]
        result.context = _build_context(result.hits)
        return result
    from app.rag.reranker import RerankCandidate, rerank

    configured_rerank_k = get_rag_settings().reranker_top_k
    rerank_count = (
        len(result.hits)
        if configured_rerank_k <= 0
        else min(len(result.hits), max(final_k, configured_rerank_k))
    )
    rerank_hits = result.hits[:rerank_count]
    tail_hits = result.hits[rerank_count:]
    candidates = [RerankCandidate(chunk_id=hit.chunk_id, text=hit.text) for hit in rerank_hits]
    rr = rerank(question, candidates)
    existing_diagnostics = getattr(result, "diagnostics", None)
    if not isinstance(existing_diagnostics, RagRetrievalDiagnostics):
        existing_diagnostics = RagRetrievalDiagnostics()
    result.diagnostics = existing_diagnostics.model_copy(update={
        "reranker_degraded": bool(rr.degraded),
        "reranker_reason": rr.reason,
        "reranker_latency_ms": rr.latency_ms,
    })
    if rr.degraded or rr.scores is None:
        result.hits = result.hits[:final_k]
        result.context = _build_context(result.hits)
        return result  # 降级：保持原规则排序
    scores = rr.scores
    if len(scores) != len(rerank_hits):
        result.diagnostics = existing_diagnostics.model_copy(update={
            "reranker_degraded": True,
            "reranker_reason": "score_count_mismatch",
            "reranker_latency_ms": rr.latency_ms,
        })
        result.hits = result.hits[:final_k]
        result.context = _build_context(result.hits)
        return result
    if resolved == "cross_encoder":
        for hit, score in zip(rerank_hits, scores):
            hit.signals.cross_encoder_score = round(score, 4)
            hit.score = round(score, 4)
        rerank_hits = sorted(rerank_hits, key=lambda hit: hit.score, reverse=True)
    elif resolved == "rrf":
        ce_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ce_rank = {original: rank + 1 for rank, original in enumerate(ce_order)}
        for index, hit in enumerate(rerank_hits):
            hit.signals.cross_encoder_score = round(scores[index], 4)
            hit.score = round(_rrf_score(index + 1, ce_rank[index]), 4)
        rerank_hits = sorted(rerank_hits, key=lambda hit: hit.score, reverse=True)
    result.hits = (rerank_hits + tail_hits)[:final_k]
    result.context = _build_context(result.hits)
    return result


def _record_retrieval_failure(
    diagnostics: dict[str, object] | None,
    *,
    reason: str,
    exc: BaseException | None = None,
) -> None:
    """Record a safe backend failure without leaking connection or SQL details."""

    if diagnostics is None:
        return
    diagnostics["reason"] = reason
    if exc is not None:
        diagnostics["exception_type"] = type(exc).__name__
    record_event("rag.qdrant.degraded", {
        "rag.degraded_reason": reason,
        "rag.exception_type": type(exc).__name__ if exc is not None else None,
    })


def _finish_retrieval(result: RagSearchResult) -> RagSearchResult:
    """Emit safe runtime diagnostics after the result has been finalized."""

    diagnostics = result.diagnostics
    if diagnostics is not None and (
        diagnostics.degraded
        or diagnostics.context_incomplete
        or diagnostics.reranker_degraded
    ):
        record_event("rag.retrieve.degraded", {
            "rag.backend": result.backend,
            "rag.strategy": result.strategy,
            "rag.degraded": diagnostics.degraded,
            "rag.fallback_from": diagnostics.fallback_from,
            "rag.degraded_reason": diagnostics.degraded_reason,
            "rag.exception_type": diagnostics.exception_type,
            "rag.private_knowledge_status": diagnostics.tenant_private_knowledge_status,
            "rag.context_incomplete": diagnostics.context_incomplete,
            "rag.reranker_degraded": diagnostics.reranker_degraded,
            "rag.index_compatibility": diagnostics.index_compatibility,
            "rag.index_fingerprint_digest": diagnostics.index_fingerprint_digest,
        })
    return result


def _normalize_private_read(value: TenantPrivateKnowledgeRead | list[KnowledgeChunk]) -> TenantPrivateKnowledgeRead:
    """Accept the pre-P0 list shape from old integrations during migration."""

    if isinstance(value, TenantPrivateKnowledgeRead):
        return value
    chunks = list(value)
    return TenantPrivateKnowledgeRead(chunks, "ok" if chunks else "empty")


def _qdrant_routes(
    doc_types: set[KnowledgeDocType] | None,
    metric_name: str | None,
) -> list[tuple[set[KnowledgeDocType] | None, str | None]]:
    """Build explicit metric-specific and generic routes for Qdrant.

    Qdrant's simple exact-match filter cannot express the local rule
    ``metric_name == target OR metric_name IS NULL`` consistently through both
    LlamaIndex and the native client.  Separate routes make the policy
    auditable and prevent another metric's dedicated document from leaking
    into an analysis/SOP query.
    """

    if not metric_name:
        return [(doc_types, None)]
    if doc_types and "metric" in doc_types and len(doc_types) > 1:
        routes: list[tuple[set[KnowledgeDocType] | None, str | None]] = [
            ({"metric"}, metric_name),
        ]
        for doc_type in sorted(doc_types - {"metric"}):
            routes.append(({doc_type}, None))
        return routes
    return [(doc_types, metric_name)]


def _retrieve_qdrant_by_policy(
    question: str,
    *,
    backend: str,
    doc_types: set[KnowledgeDocType] | None,
    metric_name: str | None,
    top_k: int,
    candidate_k: int,
    collection_name: str | None,
    diagnostics: dict[str, object],
    tenant_id: str | None = None,
    authorization_context: AuthorizationContext | None = None,
) -> RagSearchResult | None:
    """Execute policy routes and fuse/deduplicate their already-filtered hits."""

    route_results: list[RagSearchResult] = []
    route_descriptions: list[dict[str, object]] = []
    for route_doc_types, route_metric_name in _qdrant_routes(doc_types, metric_name):
        route_diagnostics: dict[str, object] = {}
        route_result: RagSearchResult | None = None
        if backend == "qdrant_hybrid":
            route_result = _retrieve_from_qdrant_native(
                question,
                doc_types=route_doc_types,
                metric_name=route_metric_name,
                top_k=top_k,
                candidate_k=candidate_k,
                collection_name=collection_name,
                diagnostics=route_diagnostics,
                tenant_id=tenant_id,
                authorization_context=authorization_context,
            )
        if route_result is None:
            route_result = _retrieve_from_qdrant(
                question,
                doc_types=route_doc_types,
                metric_name=route_metric_name,
                top_k=top_k,
                candidate_k=candidate_k,
                collection_name=collection_name,
                diagnostics=route_diagnostics,
                tenant_id=tenant_id,
                authorization_context=authorization_context,
            )
        route_descriptions.append({
            "doc_types": sorted(route_doc_types) if route_doc_types else None,
            "metric_name": route_metric_name,
        })
        if route_result is not None:
            route_results.append(route_result)
        if route_diagnostics:
            diagnostics.update(route_diagnostics)

    if not route_results:
        return None
    if len(route_results) == 1:
        result = route_results[0]
        result.filters["retrieval_routes"] = route_descriptions
        return result

    merged: dict[str, RagSearchHit] = {}
    for route_result in route_results:
        for hit in route_result.hits:
            previous = merged.get(hit.chunk_id)
            if previous is None or hit.score > previous.score:
                merged[hit.chunk_id] = hit
    first = route_results[0]
    first.hits = sorted(merged.values(), key=lambda hit: hit.score, reverse=True)[:candidate_k]
    first.filters["retrieval_routes"] = route_descriptions
    first.filters["route_fusion"] = "dedupe_max_score"
    first.context = _build_context(first.hits)
    return first


def retrieve_knowledge(
    question: str,
    *,
    doc_types: set[KnowledgeDocType] | None = None,
    metric_name: str | None = None,
    top_k: int | None = None,
    tenant_context: TenantContext | None = None,
    reranker_strategy: str | None = None,
    qdrant_collection: str | None = None,
    knowledge_root: str | None = None,
    backend_override: str | None = None,
    authorization_context: AuthorizationContext | None = None,
    authorization_boundary: AuthorizationBoundary | dict[str, object] | None = None,
    current_scope_fingerprint: str | None = None,
) -> RagSearchResult:
    """检索知识库的统一入口。

    参数：
    - `question`：用户原始问题。
    - `doc_types`：可选知识类型过滤集合，例如 `{"metric"}`、`{"sop"}`。
    - `metric_name`：可选标准指标名。为空时会尝试从问题中自动推断。
    - `top_k`：可选返回条数。为空时使用 RAG 配置中的默认值。
    - `reranker_strategy`：P3.3 重排策略 `rule` / `cross_encoder` / `rrf`。
      None 时取配置（`rag_reranker_enabled` 关闭则 `rule`）；评测时可显式传入做 A/B。

    返回值：
    - `RagSearchResult`：结构化检索结果，包含命中 chunk、citation、context、
      backend、strategy、filters 和质量信号。

    具体操作：
    1. 读取 RAG 配置。
    2. 如果后端配置为 Qdrant，则先尝试 `_retrieve_from_qdrant`。
    3. 如果 Qdrant 不可用、模型未就绪或检索异常，则降级到 `_retrieve_local`。
    4. 对最终命中应用重排策略（rule/cross_encoder/rrf；CE 不可用则降级 rule）。

    使用场景：
    - Graph 的 RAG 节点在每次用户请求中调用。
    - Harness 评测 RAG 命中率时调用。
    """

    rag_settings = get_rag_settings()
    context = tenant_context or current_tenant_context()
    if authorization_boundary is not None:
        if context is None:
            raise PermissionError("RAG authorization context is required")
        authz = authorization_context or build_authorization_context(context)
        authorization_boundary = coerce_authorization_boundary(authorization_boundary)
        if not current_scope_fingerprint:
            raise StaleAuthorization("current query scope fingerprint is required")
        assert_authorization_boundary(
            authz,
            authorization_boundary,
            current_scope_fingerprint=current_scope_fingerprint,
        )
    limit = top_k or rag_settings.top_k
    candidate_limit = max(limit, rag_settings.candidate_k)
    resolved_metric_name, resolution_source, resolution_conflict = _resolve_metric_name(
        question, metric_name,
    )
    diagnostic_resolution_source = (
        "inferred" if resolution_source == "inferred_legacy" else resolution_source
    )
    private_read = _normalize_private_read(_tenant_private_chunks(context))
    base_diagnostics = RagRetrievalDiagnostics(
        tenant_private_knowledge_status=(
            private_read.status if private_read.status != "not_applicable" else None
        ),
        context_incomplete=private_read.status == "unavailable",
        requested_metric_name=metric_name,
        resolved_metric_name=resolved_metric_name,
        metric_resolution_source=diagnostic_resolution_source,
        metric_resolution_conflict=resolution_conflict,
    )
    backend = backend_override or rag_settings.backend
    if backend in {"qdrant", "qdrant_vector", "qdrant_hybrid"}:
        result: RagSearchResult | None = None
        backend_diagnostics: dict[str, object] = {}
        result = _retrieve_qdrant_by_policy(
            question,
            backend=backend,
            doc_types=doc_types,
            metric_name=None if resolution_conflict else resolved_metric_name,
            top_k=limit,
            candidate_k=candidate_limit,
            collection_name=qdrant_collection,
            diagnostics=backend_diagnostics,
            tenant_id=context.tenant_id if context else None,
            authorization_context=authorization_context,
        )
        if result is not None:
            result.diagnostics = base_diagnostics.model_copy(update={
                "degraded": bool(backend_diagnostics.get("reason")),
                "fallback_from": backend if backend_diagnostics.get("reason") else None,
                "degraded_reason": backend_diagnostics.get("reason"),
                "exception_type": backend_diagnostics.get("exception_type"),
                "index_compatibility": str(
                    backend_diagnostics.get("index_compatibility") or "verified"
                ),
                "index_fingerprint_digest": backend_diagnostics.get("index_fingerprint_digest"),
            })
            # Tenant-private chunks intentionally live in PostgreSQL so RLS can
            # enforce their visibility.  They must still participate when the
            # global corpus is served by Qdrant; returning Qdrant directly here
            # would silently omit private knowledge in production.
            private_chunks = private_read.chunks
            if private_chunks:
                private_result = _retrieve_local(
                    question, doc_types=doc_types, metric_name=resolved_metric_name,
                    top_k=limit, candidate_k=candidate_limit,
                    private_chunks=private_chunks, knowledge_root=knowledge_root,
                    authorization_context=authorization_context,
                )
                merged = {hit.chunk_id: hit for hit in result.hits}
                for hit in private_result.hits:
                    if hit.chunk_id.startswith(f"tenant:{context.tenant_id}:"):
                        merged[hit.chunk_id] = hit
                hits = sorted(merged.values(), key=lambda hit: hit.score, reverse=True)[:candidate_limit]
                result.hits = hits
                result.context = _build_context(hits)
                result.filters["tenant_private_chunks"] = len(private_chunks)
            return _finish_retrieval(_apply_rerank_strategy(result, question, reranker_strategy))
        fallback_reason = str(backend_diagnostics.get("reason") or "qdrant_unavailable")
        local_result = _retrieve_local(
            question,
            doc_types=doc_types,
            metric_name=None if resolution_conflict else resolved_metric_name,
            top_k=limit,
            candidate_k=candidate_limit,
            private_chunks=private_read.chunks,
            tenant_id=context.tenant_id if context else None,
            knowledge_root=knowledge_root,
            authorization_context=authorization_context,
        )
        local_result.diagnostics = base_diagnostics.model_copy(update={
            "degraded": True,
            "fallback_from": backend,
            "degraded_reason": fallback_reason,
            "exception_type": backend_diagnostics.get("exception_type"),
        })
        return _finish_retrieval(_apply_rerank_strategy(local_result, question, reranker_strategy))
    local_result = _retrieve_local(
        question,
        doc_types=doc_types,
        metric_name=None if resolution_conflict else resolved_metric_name,
        top_k=limit,
        candidate_k=candidate_limit,
        private_chunks=private_read.chunks,
        tenant_id=context.tenant_id if context else None,
        knowledge_root=knowledge_root,
        authorization_context=authorization_context,
    )
    local_result.diagnostics = base_diagnostics
    return _finish_retrieval(_apply_rerank_strategy(local_result, question, reranker_strategy))


def _retrieve_local(
    question: str,
    *,
    doc_types: set[KnowledgeDocType] | None,
    metric_name: str | None,
    top_k: int,
    candidate_k: int | None = None,
    private_chunks: list[KnowledgeChunk] | None = None,
    tenant_id: str | None = None,
    knowledge_root: str | None = None,
    authorization_context: AuthorizationContext | None = None,
) -> RagSearchResult:
    """执行本地 hybrid 检索。

    参数：
    - `question`：用户原始问题。
    - `doc_types`：知识类型过滤集合。
    - `metric_name`：标准指标名过滤或加权条件。
    - `top_k`：返回命中数量。

    返回值：
    - `RagSearchResult`：`backend=local`、`strategy=local_hybrid` 的检索结果。

    具体操作：
    1. 读取本地知识 chunk。
    2. 按 doc_type 和 metric_name 做候选过滤。
    3. 分别计算 sparse_score、semantic_score、metadata_score。
    4. 按配置权重融合为 rerank_score。
    5. 返回 top_k 命中，并记录 filters 和 signals。

    使用场景：
    - Qdrant 未启动、索引未构建、embedding 模型不可用时的稳定降级方案。
    - 单元测试也可在无外部服务时验证 RAG 主链路。
    """

    chunks = _filter_chunks(
        [*load_knowledge_chunks(knowledge_root), *(private_chunks or [])],
        doc_types,
        metric_name,
        tenant_id=tenant_id,
        authorization_context=authorization_context,
    )
    scored: list[tuple[float, KnowledgeChunk, RagRetrievalSignals]] = []
    for chunk in chunks:
        rerank_score, signals = _rerank_chunk(
            question,
            chunk,
            doc_types=doc_types,
            metric_name=metric_name,
        )
        if rerank_score > 0:
            scored.append((rerank_score, chunk, signals))

    rag_settings = get_rag_settings()
    scored.sort(key=lambda item: item[0], reverse=True)
    candidate_limit = max(top_k, candidate_k or top_k)
    hits = [_hit_from_chunk(chunk, score, signals) for score, chunk, signals in scored[:candidate_limit]]
    return RagSearchResult(
        query=question,
        backend="local",
        strategy="local_hybrid",
        top_k=top_k,
        filters={
            "doc_types": sorted(doc_types) if doc_types else None,
            "metric_name": metric_name,
            "candidate_k": candidate_limit,
            "weights": {
                "vector": rag_settings.vector_weight,
                "sparse": rag_settings.sparse_weight,
                "semantic": rag_settings.semantic_weight,
                "metadata": rag_settings.metadata_weight,
            },
        },
        index_version=rag_settings.index_version,
        hits=hits,
        context=_build_context(hits),
    )


# 与 Qdrant 交互的正式检索路径。
# 注意：这个函数只负责“在线检索”，不负责把 docs 写入 Qdrant；
# 写入动作在 `app.rag.indexing.build_qdrant_index` 中完成。
def _retrieve_from_qdrant(
    question: str,
    *,
    doc_types: set[KnowledgeDocType] | None,
    metric_name: str | None,
    top_k: int,
    candidate_k: int | None = None,
    collection_name: str | None = None,
    diagnostics: dict[str, object] | None = None,
    tenant_id: str | None = None,
    authorization_context: AuthorizationContext | None = None,
) -> RagSearchResult | None:
    """从 Qdrant 向量数据库检索知识 chunk。

    参数：
    - `question`：用户原始问题。
    - `doc_types`：知识类型过滤集合。当前仅在集合长度为 1 时下推为 Qdrant metadata filter。
    - `metric_name`：标准指标名过滤条件。
    - `top_k`：返回命中数量。

    返回值：
    - `RagSearchResult | None`：
      - 成功时返回 `backend=qdrant`、`strategy=qdrant_vector` 的结构化结果。
      - 依赖缺失、Qdrant 不可用、embedding 加载失败或检索异常时返回 `None`，
        上层会自动降级到本地 `local_hybrid`。

    具体操作：
    1. 调用 `configure_rag_runtime` 设置模型缓存和离线模式。
    2. 延迟导入 LlamaIndex / Qdrant 依赖，避免缺少向量库依赖时影响主服务启动。
    3. 读取 `.env` 中的 Qdrant 服务地址，例如 `http://localhost:6333`。
    4. 根据 `doc_types` 和 `metric_name` 构造 metadata filter。
    5. 使用 HuggingFace embedding 模型把用户问题转成 query vector。
    6. 使用 `QdrantClient` 连接 Qdrant 服务。
    7. 使用 `QdrantVectorStore` 指定要查询的 collection，例如 `bizinsight_knowledge`。
    8. 使用 `VectorStoreIndex` 和 retriever 执行向量相似度检索。
    9. 将 Qdrant 命中节点转换为项目内部的 `RagSearchHit`。
    10. 将所有命中包装成统一的 `RagSearchResult` 返回。

    使用场景：
    - `.env` 中 `RAG_BACKEND=qdrant_vector` 时优先调用。
    - 这是正式 RAG 数据库检索路径，也是后续升级 dense+sparse hybrid 的挂载点。
    """

    # 第 1 步：准备 RAG 运行环境。
    # 这里会读取 RAG 配置，并设置 LlamaIndex / HuggingFace 的缓存目录。
    # 如果 embedding 模型已经下载到本地缓存，还会开启离线优先模式，
    # 避免每次检索都访问 HuggingFace 网络。
    rag_settings = configure_rag_runtime()

    # 第 2 步：延迟导入向量检索相关依赖。
    # 这些依赖只在真正走 Qdrant 检索时需要。
    # 如果依赖不存在，函数返回 None，让上层 `retrieve_knowledge` 自动降级到 local_hybrid。
    try:
        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
    except ImportError as exc:
        _record_retrieval_failure(diagnostics, reason="qdrant_dependency_unavailable", exc=exc)
        return None

    # 第 3 步：读取应用配置。
    # `app_settings.qdrant_url` 来自 `.env` 的 QDRANT_URL，
    # 当前通常是 `http://localhost:6333`，也就是 Docker 中运行的 Qdrant 服务。
    app_settings = get_settings()

    # 第 4 步：构造 metadata filter。
    # Qdrant 不只支持向量相似度搜索，也支持按 payload/metadata 过滤。
    # 这里的 metadata 是入库时写入每个 chunk 的信息，例如：
    # - doc_type: metric / sop / template
    # - metric_name: gmv / order_count / ...
    #
    # 例子：
    # 用户问“GMV 是什么？”，任务类型是 metric_definition，
    # 上游会传入 doc_types={"metric"}，这里就会构造 `doc_type == metric` 的过滤条件。
    filters = []
    if doc_types and len(doc_types) == 1:
        # 当前 LlamaIndex 的 ExactMatchFilter 是单字段精确匹配。
        # 因此当 doc_types 只有一个值时，可以直接下推到 Qdrant。
        # 如果 doc_types 有多个值，当前版本暂不下推复杂 OR 条件，避免行为不确定。
        filters.append(ExactMatchFilter(key="doc_type", value=next(iter(doc_types))))
    if metric_name:
        # 如果上游明确传入标准指标名，则进一步缩小检索范围。
        # 例如 metric_name="gmv" 时，只召回 GMV 相关 chunk。
        filters.append(ExactMatchFilter(key="metric_name", value=metric_name))

    try:
        # 第 5 步：配置 LlamaIndex 使用的 embedding 模型。
        # 检索时，LlamaIndex 会用这个模型把用户问题 `question` 转成向量。
        # 该模型必须和入库时生成 chunk 向量的模型保持一致，否则相似度空间会不匹配。
        # Reuse the process-level model cache shared with the native Qdrant
        # path; constructing the transformer for every request adds a large
        # avoidable cold-start cost.
        Settings.embed_model = _load_dense_embedding(rag_settings.embedding_model)

        # 第 6 步：创建 Qdrant 官方 Python 客户端。
        # 这一步类似关系型数据库中的“创建数据库连接”。
        # 它只知道 Qdrant 服务在哪里，还没有指定具体 collection。
        owned_client = True
        client = QdrantClient(url=app_settings.qdrant_url)

        compatibility = check_collection_fingerprint(
            client,
            collection_name or rag_settings.active_alias or rag_settings.collection_name,
            index_mode="dense",
        )
        if diagnostics is not None:
            diagnostics["index_compatibility"] = compatibility["status"]
            diagnostics["index_fingerprint_digest"] = compatibility.get("digest")
        if compatibility["status"] == "mismatch":
            _record_retrieval_failure(diagnostics, reason="qdrant_index_incompatible")
            if diagnostics is not None:
                diagnostics["index_fingerprint_fields"] = compatibility.get("different_fields", [])
            if owned_client:
                client.close()
            return None

        # 第 7 步：创建 LlamaIndex 的 Qdrant 适配器。
        # `collection_name` 决定本次检索访问 Qdrant 中哪一个 collection。
        # 当前项目的 collection 通常是 `bizinsight_knowledge`。
        #
        # 可以类比：
        # - QdrantClient ≈ 数据库连接
        # - collection_name ≈ 表名
        # - QdrantVectorStore ≈ LlamaIndex 对这张“向量表”的访问适配器
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name or rag_settings.active_alias or rag_settings.collection_name,
        )

        # 第 8 步：把 QdrantVectorStore 包装成 LlamaIndex 的索引对象。
        # 这里不会重新入库，也不会重新切 chunk；
        # 它只是告诉 LlamaIndex：后续检索要从这个已有的 Qdrant collection 中读取。
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # 第 9 步：创建 retriever。
        # retriever 是真正执行“用户问题 -> 相似 chunk”的检索器。
        # - similarity_top_k 这里使用 candidate_k，而不是最终 top_k。
        #   原因是后面还会做 rerank：先多召回一些候选，再结合关键词/语义/metadata 重排。
        # - filters 会被下推为 Qdrant metadata filter。
        candidate_k = max(top_k, candidate_k or rag_settings.candidate_k)
        retriever = index.as_retriever(
            similarity_top_k=candidate_k,
            filters=MetadataFilters(filters=filters) if filters else None,
        )

        # 第 10 步：执行向量检索。
        # LlamaIndex 内部会：
        # 1. 使用 Settings.embed_model 将 question 转成 query vector；
        # 2. 调用 QdrantVectorStore；
        # 3. 通过 qdrant-client 向 Qdrant 发起查询；
        # 4. 从指定 collection 中返回 candidate_k 个相似 chunk。
        try:
            nodes = retriever.retrieve(question)
        finally:
            if owned_client:
                client.close()
    except Exception as exc:
        # 任何 Qdrant / embedding / 网络 / collection 异常，都不在这里直接抛出。
        # 返回 None 后，上层会走 `_retrieve_local` 兜底，保证 chat 主链路不中断。
        _record_retrieval_failure(diagnostics, reason="qdrant_query_failed", exc=exc)
        return None

    # 第 11 步：把 LlamaIndex 返回的节点转换为项目内部统一的命中结构。
    # LlamaIndex 返回的是 NodeWithScore，里面包含：
    # - node：命中的文本节点；
    # - score：相似度分数；
    # - metadata：入库时写入的 payload，例如 source_path、doc_type、section。
    resolved_metric_name = metric_name or _infer_query_metric(question)
    scored: list[tuple[float, KnowledgeChunk, RagRetrievalSignals]] = []
    for node_with_score in nodes:
        metadata = dict(node_with_score.node.metadata)
        visibility = str(metadata.get("visibility") or "global")
        if visibility == "tenant" and metadata.get("tenant_id") != tenant_id:
            continue
        if visibility not in {"global", "tenant"}:
            continue
        classification = str(metadata.get("classification") or "internal")
        allowed_classifications = set(getattr(authorization_context, "allowed_classifications", ()) or {"public", "internal"})
        if classification not in allowed_classifications:
            continue
        allowed_roles = {str(item) for item in (metadata.get("allowed_roles") or [])}
        if allowed_roles and (authorization_context is None or authorization_context.role not in allowed_roles):
            continue

        # chunk_id 是 citation 的关键字段。
        # 优先使用入库时写入 metadata 的 chunk_id；
        # 如果旧数据没有该字段，则回退到 LlamaIndex node_id。
        chunk_id = str(metadata.get("chunk_id", node_with_score.node.node_id))

        # 先把 Qdrant / LlamaIndex 节点还原成项目内部的 KnowledgeChunk。
        # 这样 Qdrant 路径和 local_hybrid 路径可以复用同一套 rerank 逻辑。
        chunk = KnowledgeChunk(
            chunk_id=chunk_id,
            doc_id=str(metadata.get("doc_id", chunk_id)),
            title=str(metadata.get("title", metadata.get("source_path", ""))),
            text=node_with_score.node.get_content().strip(),
            source_path=str(metadata.get("source_path", "")),
            doc_type=metadata.get("doc_type", "unknown"),
            metric_name=metadata.get("metric_name"),
            tags=list(metadata.get("tags") or []),
            section=metadata.get("section"),
            document_hash=metadata.get("document_hash"),
            index_version=metadata.get("index_version") or rag_settings.index_version,
            pipeline_version=metadata.get("pipeline_version"),
            content_hash=metadata.get("content_hash"),
            source_start=metadata.get("source_start"),
            source_end=metadata.get("source_end"),
            visibility=visibility,
            tenant_id=metadata.get("tenant_id"),
            classification=classification,
            allowed_roles=list(allowed_roles),
        )

        # 对 Qdrant 候选做二阶段重排。
        # vector_score 来自 Qdrant 的向量相似度；sparse/semantic/metadata 分由本地规则补充。
        # 这样比“直接使用向量相似度 top_k”更稳，尤其适合 GMV/AOV 等业务术语。
        rerank_score, signals = _rerank_chunk(
            question,
            chunk,
            doc_types=doc_types,
            metric_name=resolved_metric_name,
            vector_score=float(node_with_score.score or 0.0),
        )
        scored.append((rerank_score, chunk, signals))

    scored.sort(key=lambda item: item[0], reverse=True)
    hits = [_hit_from_chunk(chunk, score, signals) for score, chunk, signals in scored[:candidate_k]]

    strategy: Literal["qdrant_hybrid", "qdrant_vector"] = (
        "qdrant_hybrid" if rag_settings.backend == "qdrant_hybrid" else "qdrant_vector"
    )
    weights = {
        "vector": rag_settings.vector_weight,
        "sparse": rag_settings.sparse_weight,
        "semantic": rag_settings.semantic_weight,
        "metadata": rag_settings.metadata_weight,
    }

    # 第 12 步：返回统一的 RAG 检索结果。
    # 即使底层来自 Qdrant，返回类型也和本地 local_hybrid 保持一致。
    # workflow 只需要消费 RagSearchResult，不需要关心底层数据库细节。
    return RagSearchResult(
        query=question,
        backend="qdrant",
        strategy=strategy,
        top_k=top_k,
        filters={
            "doc_types": sorted(doc_types) if doc_types else None,
            "metric_name": metric_name,
            "candidate_k": candidate_k,
            "index_version": rag_settings.index_version,
            "weights": weights,
        },
        index_version=rag_settings.index_version,
        hits=hits,
        context=_build_context(hits),
    )


def _retrieve_from_qdrant_native(
    question: str,
    *,
    doc_types: set[KnowledgeDocType] | None,
    metric_name: str | None,
    top_k: int,
    candidate_k: int | None = None,
    collection_name: str | None = None,
    diagnostics: dict[str, object] | None = None,
    embed_fn=None,
    client=None,
    sparse_model_path: Path | str | None = None,
    tenant_id: str | None = None,
    authorization_context: AuthorizationContext | None = None,
) -> RagSearchResult | None:
    """Qdrant native dense+sparse 融合检索。

    该路径使用同一 collection 下的 dense 与 sparse 两个命名空间，通过 Qdrant 的
    prefetch + RRF fusion 做原生混合检索，而不是「dense 召回 + Python 侧关键词重排」。
    sparse 特征由 `app.rag.sparse` 生成，查询时必须加载索引构建时持久化的同一份词表。

    不可用时返回 None（sparse 模型缺失、查询词不在词表、Qdrant 不可达），上层自动降级。
    """
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        _record_retrieval_failure(diagnostics, reason="qdrant_dependency_unavailable", exc=exc)
        return None

    rag_settings = get_rag_settings()
    query_collection = collection_name or rag_settings.active_alias or rag_settings.collection_name
    owned_client = client is None
    try:
        client = client or QdrantClient(url=get_settings().qdrant_url)
        physical_collection = resolve_collection_alias(client, query_collection)
    except Exception as exc:
        _record_retrieval_failure(diagnostics, reason="qdrant_connection_failed", exc=exc)
        return None

    model = sparse_features.load_sparse_model(
        sparse_model_path
        or _sparse_model_path(physical_collection)
    )
    if model is None:
        _record_retrieval_failure(diagnostics, reason="sparse_model_unavailable")
        if owned_client:
            client.close()
        return None
    vocab, idf = model
    if embed_fn is None:
        embed_fn = make_dense_embed_fn()

    try:
        dense = embed_fn(question)
        sparse = sparse_features.sparse_embed(question, vocab, idf)
        if not sparse["indices"]:
            if owned_client:
                client.close()
            return None
    except Exception as exc:
        _record_retrieval_failure(diagnostics, reason="qdrant_embedding_failed", exc=exc)
        if owned_client:
            client.close()
        return None

    must = []
    if doc_types and len(doc_types) == 1:
        must.append(models.FieldCondition(key="doc_type", match=models.MatchValue(value=next(iter(doc_types)))))
    if metric_name:
        must.append(models.FieldCondition(key="metric_name", match=models.MatchValue(value=metric_name)))
    query_filter = models.Filter(must=must) if must else None
    candidate_k = max(top_k, candidate_k or rag_settings.candidate_k)

    try:
        compatibility = check_collection_fingerprint(
            client,
            query_collection,
            index_mode="dense_sparse",
        )
        if diagnostics is not None:
            diagnostics["index_compatibility"] = compatibility["status"]
            diagnostics["index_fingerprint_digest"] = compatibility.get("digest")
        if compatibility["status"] == "mismatch":
            _record_retrieval_failure(diagnostics, reason="qdrant_index_incompatible")
            if diagnostics is not None:
                diagnostics["index_fingerprint_fields"] = compatibility.get("different_fields", [])
            if owned_client:
                client.close()
            return None
        try:
            response = client.query_points(
                collection_name=query_collection,
                prefetch=[
                    models.Prefetch(query=dense, using="dense", limit=candidate_k, filter=query_filter),
                    models.Prefetch(
                        query=models.SparseVector(indices=list(sparse["indices"]), values=list(sparse["values"])),
                        using="sparse", limit=candidate_k, filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=candidate_k,
                with_payload=True,
            )
        finally:
            if owned_client:
                client.close()
    except Exception as exc:
        _record_retrieval_failure(diagnostics, reason="qdrant_hybrid_query_failed", exc=exc)
        return None

    query_indices = set(sparse["indices"])
    matched_terms = [term for term, index in vocab.items() if index in query_indices][:20]
    hits: list[RagSearchHit] = []
    for scored in response.points:
        payload = scored.payload or {}
        visibility = str(payload.get("visibility") or "global")
        if visibility == "tenant" and payload.get("tenant_id") != tenant_id:
            continue
        if visibility not in {"global", "tenant"}:
            continue
        classification = str(payload.get("classification") or "internal")
        allowed_classifications = set(getattr(authorization_context, "allowed_classifications", ()) or {"public", "internal"})
        if classification not in allowed_classifications:
            continue
        allowed_roles = {str(item) for item in (payload.get("allowed_roles") or [])}
        if allowed_roles and authorization_context is not None and authorization_context.role not in allowed_roles:
            continue
        chunk_id = str(payload.get("chunk_id", scored.id))
        hits.append(
            RagSearchHit(
                chunk_id=chunk_id,
                text=payload.get("text") or "",
                score=round(float(scored.score), 4),
                citation=RagCitation(
                    source_path=str(payload.get("source_path", "")),
                    title=str(payload.get("title", "")),
                    doc_type=payload.get("doc_type", "unknown"),
                    metric_name=payload.get("metric_name"),
                    chunk_id=chunk_id,
                    section=payload.get("section"),
                    index_version=payload.get("index_version") or rag_settings.index_version,
                    source_start=payload.get("source_start"),
                    source_end=payload.get("source_end"),
                ),
                signals=RagRetrievalSignals(matched_terms=matched_terms),
            )
        )
    return RagSearchResult(
        query=question,
        backend="qdrant",
        strategy="qdrant_hybrid",
        top_k=top_k,
        filters={
            "doc_types": sorted(doc_types) if doc_types else None,
            "metric_name": metric_name,
            "native_dense_sparse": True,
            "dense_model": rag_settings.embedding_model,
            "sparse_vocab_size": len(vocab),
            "fusion": "RRF",
            "candidate_k": candidate_k,
            "index_version": rag_settings.index_version,
        },
        index_version=rag_settings.index_version,
        hits=hits,
        context=_build_context(hits),
    )


def retrieve_for_release(
    question: str, release: dict, *, doc_types: set[str] | None = None,
    metric_name: str | None = None, top_k: int | None = None, tenant_context=None,
    authorization_context: AuthorizationContext | None = None,
    authorization_boundary: AuthorizationBoundary | dict[str, object] | None = None,
    current_scope_fingerprint: str | None = None,
) -> RagSearchResult:
    """Release-aware entry point used by Shadow and Canary execution.

    Candidate configuration is immutable in the release spec.  The existing
    retriever stays the enforcement point for tenant/private-chunk filtering.
    """
    spec = dict(release.get("spec") or {})
    backend = str(spec.get("backend") or "")
    if backend == "local_hybrid" and not spec.get("local_snapshot_ref"):
        raise ValueError("Local release is missing immutable local_snapshot_ref")
    if backend in {"qdrant", "qdrant_vector", "qdrant_hybrid"} and not spec.get("qdrant_collection"):
        raise ValueError("Qdrant release is missing immutable qdrant_collection")
    result = retrieve_knowledge(
        question, doc_types=doc_types, metric_name=metric_name, top_k=top_k,
        tenant_context=tenant_context, reranker_strategy=str(spec.get("reranker_strategy") or "rule"),
        qdrant_collection=spec.get("qdrant_collection"),
        knowledge_root=spec.get("local_snapshot_ref"), backend_override=spec.get("backend"),
        authorization_context=authorization_context,
        authorization_boundary=authorization_boundary,
        current_scope_fingerprint=current_scope_fingerprint,
    )
    result.filters["release_id"] = release.get("release_id")
    result.filters["release_spec_hash"] = release.get("spec_hash")
    return result


def retrieve_metric_notes(question: str) -> str:
    """兼容旧代码的指标知识检索函数。

    参数：
    - `question`：用户原始问题。

    返回值：
    - `str`：指标类知识检索后的 context 文本。

    使用场景：
    - 早期代码只消费字符串形式的 `metric_context`。
    - 新代码应优先使用 `retrieve_knowledge`，以获得 citation 和 signals。
    """

    return retrieve_knowledge(question, doc_types={"metric"}).context
