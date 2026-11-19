"""Metric Definition 节点 — RAG 知识检索。

V1 阶段内部复用现有 RAG 检索逻辑（retrieve_knowledge），
Target 阶段将升级为独立 LLM Agent（使用 RAG 工具）。

MECE 写入字段：metric_context, rag_result, rag_citations, current_agent, node_history
"""

from app.graph.state import AgentState
from app.agents.contracts import EvidenceBundle, stable_evidence_id
from app.context.orchestrator import ContextOrchestrator
from app.context.packer import estimate_tokens
from app.llm.routing import model_for
from app.security import ensure_safe_untrusted_content, is_trusted_knowledge_source
from app.rag.release_controller import retrieve_routed
from app.rag.retriever import retrieve_knowledge
from app.rag.retriever import _apply_rerank_strategy
from app.tenancy import TenantContext
from app.auth.context import build_authorization_context
from app.scope.canonical import CanonicalScope
from app.schemas.explorer import ControlledAnalysisScope
from app.core.config import get_settings
from app.rag.release import published_rag_version
from app.resilience import execute_with_resilience
from app.observability import record_event
from time import perf_counter


# 不同任务类型对应不同知识类型过滤策略。
# 指标定义 Agent 不会误召回报告模板；Root Cause Agent 则可以同时召回
# 指标定义和异常分析 SOP，避免异常分析只拿到指标口径却拿不到诊断方法。
DOC_TYPES_BY_TASK: dict[str, set[str]] = {
    "metric_definition": {"metric"},
    "anomaly_analysis": {"metric", "sop"},
    "report_generation": {"metric", "sop", "template"},
}


def _resolve_registry_metric(question: str) -> str | None:
    """Resolve the longest registered metric name or alias in the question."""
    folded = question.casefold()
    candidates: list[tuple[int, str]] = []
    try:
        from app.metrics.registry import get_registry

        for definition in get_registry().list():
            terms = (definition.metric_id, definition.display_name, *definition.aliases,
                     *definition.planning.phrases)
            for term in terms:
                value = str(term).strip()
                if value and value.casefold() in folded:
                    candidates.append((len(value), definition.metric_id))
    except Exception:
        return None
    return max(candidates)[1] if candidates else None


def _authoritative_definition(metric_id: str, scope: str) -> dict[str, object] | None:
    """Build a non-LLM authority record from the validated metric registry."""
    try:
        from app.metrics.registry import get_registry

        definition = get_registry().get(metric_id)
        implementation = definition.implementations.get(scope)
        if implementation is None:
            return None
        return {
            "metric_id": definition.metric_id,
            "display_name": definition.display_name,
            "aliases": list(definition.aliases),
            "unit": definition.unit,
            "display_format": definition.display_format,
            "base_grain": definition.base_grain,
            "aggregation": definition.aggregation,
            "direction": definition.direction,
            "version": definition.version,
            "scope": scope,
            "capabilities": sorted(implementation.capabilities),
            "allowed_dimensions": list(implementation.dimensions),
            "lineage_source": implementation.lineage_source,
            "lineage_description": implementation.lineage_description,
            "documentation": definition.documentation,
        }
    except Exception:
        return None


def _authority_context(definition: dict[str, object]) -> str:
    """Render registry fields as a locked, human-readable evidence block."""
    dimensions = ", ".join(str(item) for item in definition.get("allowed_dimensions", [])) or "无"
    capabilities = ", ".join(str(item) for item in definition.get("capabilities", [])) or "无"
    return (
        "【权威指标注册表（优先于 RAG 补充材料）】\n"
        f"指标 ID：{definition['metric_id']}\n"
        f"显示名称：{definition['display_name']}\n"
        f"单位：{definition['unit']}；基础粒度：{definition['base_grain']}；聚合方式：{definition['aggregation']}\n"
        f"数据范围：{definition['scope']}；数据来源：{definition['lineage_source']}\n"
        f"口径说明：{definition['lineage_description'] or '未提供'}\n"
        f"允许维度：{dimensions}\n"
        f"可执行能力：{capabilities}\n"
        f"注册版本：{definition['version']}"
    )


def metric_definition(state: AgentState) -> AgentState:
    """检索 RAG 知识库，获取指标定义、口径和分析 SOP 上下文。

    参数：
    - `state`：当前工作流状态，至少包含 `user_message` 和 `task_type`。

    返回值：
    - `AgentState`：写入 RAG 检索结果后的状态。
    """

    started = perf_counter()
    published_version = published_rag_version()
    retrieval_question = str(
        (state.get("context_compilation") or {}).get("standalone_question")
        or state["user_message"]
    )
    metric_id = _resolve_registry_metric(retrieval_question)
    scope = str(state.get("scope_type") or "platform")
    authoritative = _authoritative_definition(metric_id, scope) if metric_id else None
    retrieval_context = None
    retrieval_authz = None
    current_scope_fingerprint = None
    if state.get("tenant_id") and state.get("user_id"):
        retrieval_context = TenantContext(
            tenant_id=str(state["tenant_id"]),
            user_id=str(state["user_id"]),
            role=state.get("user_role") or "admin",
            scope_type=state.get("scope_type") or "platform",
            seller_id=state.get("seller_id"),
        )
        controlled_payload = state.get("controlled_analysis_scope")
        if controlled_payload and state.get("authorization_boundary"):
            controlled = ControlledAnalysisScope.model_validate(controlled_payload)
            retrieval_authz = build_authorization_context(
                retrieval_context,
                allowed_metric_ids=controlled.allowed_metrics,
                allowed_dimensions=controlled.allowed_dimensions,
                metric_scope_type="seller",
            )
            current_scope = CanonicalScope.resolve(
                retrieval_authz,
                seller_ids=[controlled.seller_id],
                start_date=controlled.default_start_date,
                end_date=controlled.default_end_date,
                metric_ids=controlled.allowed_metrics,
                dimensions=controlled.allowed_dimensions,
                granularity=controlled.granularity,
            )
            current_scope_fingerprint = current_scope.query_scope_fingerprint
    original_question = str(state.get("user_message") or "").strip()

    def _retrieve_one(query: str):
        if not get_settings().rag_release_controller_enabled:
            return retrieve_knowledge(
                query, doc_types=DOC_TYPES_BY_TASK.get(state.get("task_type")),
                metric_name=metric_id, tenant_context=retrieval_context,
                authorization_context=retrieval_authz,
                authorization_boundary=state.get("authorization_boundary") or None,
                current_scope_fingerprint=current_scope_fingerprint,
            )
        return retrieve_routed(
            query, context=retrieval_context,
            session_id=state.get("memory_session_id"), trace_id=state.get("trace_id"),
            doc_types=DOC_TYPES_BY_TASK.get(state.get("task_type")), metric_name=metric_id,
            authorization_context=retrieval_authz,
            authorization_boundary=state.get("authorization_boundary") or None,
            current_scope_fingerprint=current_scope_fingerprint,
        )

    def _retrieve_dual():
        queries = list(dict.fromkeys(q for q in (original_question, retrieval_question) if q))
        results = [_retrieve_one(query) for query in queries]
        if len(results) == 1:
            return results[0], {queries[0]: results[0]}
        # Reciprocal-rank fusion combines independently retrieved original and
        # rewritten queries without allowing the rewrite to erase exact terms.
        ranked: dict[str, tuple[object, float, list[str]]] = {}
        for query, result_item in zip(queries, results):
            for rank, hit in enumerate(result_item.hits, start=1):
                score = 1.0 / (60 + rank)
                prior = ranked.get(hit.chunk_id)
                ranked[hit.chunk_id] = (
                    hit, score + (prior[1] if prior else 0.0),
                    list(dict.fromkeys((prior[2] if prior else []) + [query])),
                )
        first = results[0]
        fused_hits = [
            row[0].model_copy(update={
                "score": row[1],
                "signals": row[0].signals.model_copy(update={
                    "rerank_score": row[1],
                }),
            })
            for row in sorted(ranked.values(), key=lambda row: row[1], reverse=True)
        ]
        merged = first.model_copy(update={
            "query": retrieval_question,
            "hits": fused_hits,
            "context": "",
            "filters": {**first.filters, "dual_query": True, "fusion": "rrf", "queries": queries},
        })
        merged = _apply_rerank_strategy(merged, retrieval_question, None)
        return merged, dict(zip(queries, results))

    result, query_results = execute_with_resilience(
        _retrieve_dual, "rag", timeout_seconds=get_settings().rag_timeout_seconds,
    )
    ensure_safe_untrusted_content(result.context, "rag_document")
    # 运行诊断与查询 filters 分离；filters 只描述约束，diagnostics 描述
    # 实际是否降级、私有知识是否可用以及 reranker 是否成功。
    diagnostics = getattr(result, "diagnostics", None)
    retrieval_diagnostics = diagnostics.model_dump() if diagnostics else {}
    record_event("rag.retrieve", {
        "trace_id": state.get("trace_id"), "thread_id": state.get("thread_id"),
        "rag.backend": result.strategy, "rag.index_version": result.index_version, "rag.release_version": published_version,
        "rag.degraded": retrieval_diagnostics.get("degraded", False),
        "rag.fallback_from": retrieval_diagnostics.get("fallback_from"),
        "rag.degraded_reason": retrieval_diagnostics.get("degraded_reason"),
        "rag.exception_type": retrieval_diagnostics.get("exception_type"),
        "rag.private_knowledge_status": retrieval_diagnostics.get("tenant_private_knowledge_status"),
        "rag.context_incomplete": retrieval_diagnostics.get("context_incomplete", False),
        "rag.reranker_degraded": retrieval_diagnostics.get("reranker_degraded", False),
        "rag.reranker_reason": retrieval_diagnostics.get("reranker_reason"),
        "rag.index_compatibility": retrieval_diagnostics.get("index_compatibility"),
        "rag.index_fingerprint_digest": retrieval_diagnostics.get("index_fingerprint_digest"),
        "metric_name_filter": metric_id,
        "rag.query_count": len(query_results),
        "rag.original_query_hit_ids": [hit.chunk_id for hit in query_results.get(original_question, result).hits],
        "rag.rewritten_query_hit_ids": [hit.chunk_id for hit in query_results.get(retrieval_question, result).hits],
        "rag.fused_hit_ids": [hit.chunk_id for hit in result.hits],
        "authoritative_registry": bool(authoritative),
        "duration_ms": round((perf_counter() - started) * 1000, 3), "status": "ok",
    })
    rag_context = result.context or "暂无 RAG 补充材料。"
    if retrieval_diagnostics.get("context_incomplete"):
        rag_context = (
            "【注意：租户私有知识库当前不可用，以下上下文可能不完整；请勿把未检索到私有知识当作不存在。】\n\n"
            + rag_context
        )
    state["metric_authoritative_definition"] = authoritative
    state["metric_context"] = (
        f"{_authority_context(authoritative)}\n\n【RAG 补充材料】\n{rag_context}"
        if authoritative else rag_context
    )
    state["rag_result"] = result.model_dump()
    # WP1 Critic source governance: the deterministic RAG review runs here,
    # in the RAG-owning node, and the result is handed to Critic as control
    # metadata.  Critic no longer reads state["rag_result"] directly.
    try:
        from app.agents.critic_agent import review_rag_result

        state["rag_review"] = review_rag_result(state["rag_result"])
    except Exception:
        state["rag_review"] = {"status": "skipped", "issues": [], "reason": "review_unavailable"}
    state["rag_query_trace"] = {
        "original_question": original_question,
        "standalone_question": retrieval_question,
        "queries": {
            query: {
                "hit_ids": [hit.chunk_id for hit in query_result.hits],
                "scores": [hit.score for hit in query_result.hits],
                "strategy": query_result.strategy,
            }
            for query, query_result in query_results.items()
        },
        "fused_hit_ids": [hit.chunk_id for hit in result.hits],
        "fusion": "rrf" if len(query_results) > 1 else "single_query",
    }
    state["rag_citations"] = [citation.model_dump() for citation in result.citations]
    bundles = state.setdefault("evidence_bundles", [])
    for citation in state["rag_citations"]:
        content = {
            "citation": citation,
            "query": retrieval_question,
            "original_query": state["user_message"],
            "release_version": published_version,
        }
        bundles.append(EvidenceBundle(
            evidence_id=stable_evidence_id("rag", citation.get("source_path", "unknown"), content),
            source_type="rag",
            source_ref=citation.get("source_path", "unknown"),
            content=content,
            trusted=is_trusted_knowledge_source(citation.get("source_path", "")),
        ).model_dump(mode="json"))
    # WP1: metric_definition receives only registry + RAG context.  The bundle
    # is also the budget boundary for its rendered metric_context; normal
    # outputs are unchanged, over-budget output is replaced by the packed
    # bundle rather than silently exceeding the agent's input budget.
    metric_model = model_for("extract")[0]
    metric_bundle = ContextOrchestrator().build_specialist_bundle(
        state, "metric_definition", instruction=retrieval_question, model=metric_model,
    )
    available_tokens = int((metric_bundle.budget or {}).get("available_context_tokens") or 0)
    if (
        metric_bundle.text
        and available_tokens > 0
        and estimate_tokens(str(state.get("metric_context") or ""), metric_model) > available_tokens
    ):
        state["metric_context"] = metric_bundle.text

    state["current_agent"] = "metric_definition"
    # revision 计数：记录该节点被执行的次数，Critic 用它限制返工次数，防止无限循环。
    state.setdefault("_revision_counts", {})
    state["_revision_counts"]["metric_definition"] = state["_revision_counts"].get("metric_definition", 0) + 1
    state.setdefault("node_history", []).append(
        {
            "node": "metric_definition",
            "summary": f"RAG retrieved {len(result.hits)} hits via {result.strategy}",
        }
    )
    return state
