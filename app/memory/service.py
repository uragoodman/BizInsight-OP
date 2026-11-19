"""记忆候选构建与召回。此处不调用 LLM，且不让记忆决定 MCP 参数。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import date, datetime, timezone
from uuid import uuid4

from app.graph.state import AgentState
from app.core.config import get_settings
from app.memory.contracts import AnalysisMemory, SessionMemory
from app.memory.embeddings import EMBEDDING_MODEL, EMBEDDING_VERSION, embed_texts
from app.memory.repository import MemoryRepository
from app.memory.source_versions import build_source_snapshot
from app.schemas.explorer import ControlledAnalysisScope
from app.tenancy import TenantContext
from app.observability import record_event

_MAX_TURNS = 3
_MAX_EVIDENCE_REFS = 20
_MAX_CLAIMS = 15
_MAX_SNAPSHOT_BYTES = 50_000

# P3.1 混合排序权重：语义 + 范围重叠 + 时间衰减 + 证据完整度；MMR 多样性惩罚。
_SEMANTIC_WEIGHT = 0.40
_SCOPE_WEIGHT = 0.25
_RECENCY_WEIGHT = 0.20
_EVIDENCE_WEIGHT = 0.15
_MMR_LAMBDA = 0.5
_MEMORY_TOP_K = 3


def _date_overlap_ratio(row: dict, scope_start: date, scope_end: date) -> float:
    """记忆日期范围与当前受控范围的重叠占比（范围贴合度）。"""
    overlap_start = max(row["start_date"], scope_start)
    overlap_end = min(row["end_date"], scope_end)
    overlap = max(0, (overlap_end - overlap_start).days + 1)
    total = max(1, (row["end_date"] - row["start_date"]).days + 1)
    return min(1.0, overlap / total)


def _recency_score(row: dict) -> float:
    """时间衰减：90 天半衰期，越新的结论越接近 1。"""
    created = row.get("created_at")
    if not created:
        return 0.0
    age_days = max(0, (datetime.now(timezone.utc) - created).days)
    return 0.5 ** (age_days / 90.0)


def _evidence_score(row: dict) -> float:
    """证据完整度：5 条证据封顶。"""
    return min(1.0, len(row.get("evidence_ids") or []) / 5.0)


def _metric_overlap(left: list[str], right: list[str]) -> float:
    """指标标签 Jaccard 重叠，用作 MMR 的多样性惩罚。"""
    if not left or not right:
        return 0.0
    a, b = set(left), set(right)
    return len(a & b) / max(1, len(a | b))


def _mmr_value(row: dict, selected: list[dict]) -> float:
    """MMR：relevance - λ·max_similarity(selected)；用指标重叠近似相似度。"""
    base = float(row.get("_hybrid_score", 0.0))
    diversity = max((_metric_overlap(row.get("metric_tags") or [], item.get("metric_tags") or []) for item in selected), default=0.0)
    return base - _MMR_LAMBDA * diversity


def _hybrid_rank(rows: list[dict], scope: dict, has_semantic: bool) -> list[dict]:
    """对召回候选做混合排序 + MMR 去重，取前 _MEMORY_TOP_K。

    语义分来自 SQL 余弦（normalized embedding → dot）；无向量时降级为
    范围重叠 + 时间衰减 + 证据完整度（确定性排序）。
    """
    scope_start, scope_end = _scope_dates(scope)
    scored: list[dict] = []
    for row in rows:
        semantic = max(0.0, min(1.0, float(row.get("semantic_score") or 0.0))) if has_semantic else 0.0
        score = (
            _SEMANTIC_WEIGHT * semantic
            + _SCOPE_WEIGHT * _date_overlap_ratio(row, scope_start, scope_end)
            + _RECENCY_WEIGHT * _recency_score(row)
            + _EVIDENCE_WEIGHT * _evidence_score(row)
        )
        row["_hybrid_score"] = round(score, 4)
        scored.append(row)
    selected: list[dict] = []
    remaining = list(scored)
    while remaining and len(selected) < _MEMORY_TOP_K:
        best = max(remaining, key=lambda item: _mmr_value(item, selected))
        selected.append(best)
        remaining.remove(best)
    return selected


def canonical_memory_scope(scope: dict | ControlledAnalysisScope) -> dict:
    """从受控范围内提取规范化键用于 fingerprint 和存储。

    字段名统一为 default_start_date / default_end_date（与 ControlledAnalysisScope 一致）。
    metric_name 不进入 fingerprint —— GMV→差评→物流的追问不因当前指标变化就视为 scope 变更。
    """
    if isinstance(scope, ControlledAnalysisScope):
        s = scope.model_dump(mode="json")
    else:
        s = dict(scope)
    return {
        "seller_id": s.get("seller_id", ""),
        "default_start_date": str(s.get("default_start_date") or s.get("start_date", "")),
        "default_end_date": str(s.get("default_end_date") or s.get("end_date", "")),
        "granularity": s.get("granularity", "day"),
        "allowed_metrics": tuple(sorted(s.get("allowed_metrics", []))),
        "allowed_dimensions": tuple(sorted(s.get("allowed_dimensions", []))),
    }


def scope_fingerprint(scope: dict | ControlledAnalysisScope) -> str:
    """从规范化 scope 生成 SHA256 fingerprint。"""
    canonical = canonical_memory_scope(scope)
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False,
                     default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def normalized_claim_hash(insight_text: str, task_type: str, fingerprint: str, metric_tags: list[str]) -> str:
    """Exact, conservative deduplication key; numbers and dates are preserved."""
    normalized = unicodedata.normalize("NFKC", insight_text).strip()
    normalized = re.sub(r"\s+", " ", normalized).casefold()
    payload = json.dumps({
        "insight": normalized, "task_type": task_type,
        "scope_fingerprint": fingerprint, "metric_tags": sorted(metric_tags),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scope_dates(scope: dict) -> tuple[date, date]:
    # 兼容 v1 (start_date) 和 v2 (default_start_date) 键
    s = scope.get("default_start_date") or scope.get("start_date")
    e = scope.get("default_end_date") or scope.get("end_date")
    return date.fromisoformat(str(s)), date.fromisoformat(str(e))


def _claim_candidates(state: AgentState) -> list[dict]:
    known = {item.get("evidence_id") for item in state.get("evidence_bundles", [])}
    return [claim for claim in state.get("claims", []) if claim.get("value") and (ids := claim.get("source_evidence_ids", [])) and all(item in known for item in ids)]


def _metric_tags(state: AgentState, scope: dict) -> list[str]:
    tags = [str(scope.get("metric_name", ""))]
    result = state.get("tool_result", {}).get("result", {})
    for key in ("metric_result", "trend_result", "compare_result", "breakdown_result"):
        name = result.get(key, {}).get("metric_name")
        if name:
            tags.append(str(name))
    return list(dict.fromkeys(item for item in tags if item))[:8]


# ====================================================================
# Phase 7: 多轮上下文快照构建
# ====================================================================


def _build_recent_turn(state: AgentState) -> dict:
    """从当前 final state 构建一轮的压缩摘要。

    不保存的内容:
      - review_comment_message 等原始评论正文
      - 用户的 user_message
      - LLM 的完整 answer
      - 完整 analysis_presentation（含所有卡片）
    """
    claims = state.get("claims", [])
    hypothesis = state.get("hypothesis_chains_by_metric", {})
    scope = state.get("controlled_analysis_scope", {})
    canonical = canonical_memory_scope(scope) if scope else {}

    evidence = state.get("evidence_bundles", [])
    evidence_refs = [
        {"evidence_id": ev.get("evidence_id"),
         "source_type": ev.get("source_type"),
         "source_ref": ev.get("source_ref")}
        for ev in evidence[:_MAX_EVIDENCE_REFS]
    ]
    approved_claims = []
    for c in claims:
        eids = c.get("source_evidence_ids", [])
        if eids and any(ev["evidence_id"] in eids for ev in evidence):
            approved_claims.append({
                "value": str(c.get("value", ""))[:200],
                "source_evidence_ids": eids[:5],
            })
            if len(approved_claims) >= _MAX_CLAIMS:
                break
    return {
        "trace_id": state.get("trace_id", ""),
        "task_type": state.get("task_type", ""),
        "canonical_scope": canonical,
        "scope_fingerprint": scope_fingerprint(canonical) if canonical else "",
        "approved_claims": approved_claims,
        "hypothesis_chain_summary": {
            mid: [{"hypothesis": s.get("hypothesis", "")[:120],
                    "verdict": s.get("verdict", ""),
                    "evidence_ids": s.get("evidence_ids", [])}
                  for s in chain[-5:]]
            for mid, chain in hypothesis.items()
        } if hypothesis else {},
        "evidence_refs": evidence_refs,
        "source_data_version": "olist_2018",
        "query_version": "1.0",
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_session_snapshot(state: AgentState, existing: dict | None = None) -> dict:
    """构建或追加一轮到滚动 snapshot。最多保留 _MAX_TURNS 轮。"""
    turn = _build_recent_turn(state)
    recent = list(existing.get("recent_turns", [])) if existing else []
    recent.insert(0, turn)
    return {
        "schema_version": "2.0",
        "recent_turns": recent[:_MAX_TURNS],
    }


def _guard_snapshot_size(snapshot: dict, max_bytes: int = _MAX_SNAPSHOT_BYTES) -> dict:
    """snapshot 超 max_bytes 时截断为仅 claims，防止 JSONB 膨胀。"""
    raw = json.dumps(snapshot, default=str)
    if len(raw.encode("utf-8")) <= max_bytes:
        return snapshot
    return {
        "schema_version": "2.0",
        "recent_turns": [
            {"approved_claims": t.get("approved_claims", [])[:5],
             "scope_fingerprint": t.get("scope_fingerprint", ""),
             "captured_at": t.get("captured_at", "")}
            for t in snapshot.get("recent_turns", [])
        ],
        "_truncated": True,
    }


def build_memories(
    state: AgentState,
    *,
    include_legacy_session: bool = True,
) -> tuple[SessionMemory | None, AnalysisMemory | None]:
    """仅从 Critic 已放行的、绑定证据的 claim 构建记忆；不从最终自然语言答案自由提取。"""
    scope = state.get("controlled_analysis_scope")
    if not scope:
        raise ValueError("Memory requires a trusted controlled_analysis_scope")
    claims = _claim_candidates(state)
    if not claims:
        raise ValueError("Memory requires an evidence-bound claim")
    start_date, end_date = _scope_dates(scope)
    fingerprint = scope_fingerprint(scope)
    evidence_ids = list(dict.fromkeys(evidence_id for claim in claims for evidence_id in claim["source_evidence_ids"]))
    first_claim = str(claims[0]["value"]).replace("\n", " ").strip()[:500]
    seller_id = str(scope["seller_id"])
    summary = f"已确认：店铺 {seller_id}，{start_date}至{end_date}，{first_claim}"[:1000]

    session: SessionMemory | None = None
    if include_legacy_session:
        memory_session_id = state.get("memory_session_id") or state.get("chat_session_id")
        if memory_session_id:
            # Legacy/non-durable callers still use memory_sessions. Durable chat
            # owns its rolling summary in chat_sessions and must not build a
            # second summary/snapshot from the same committed turns.
            existing_snapshot: dict | None = None
            try:
                repo = MemoryRepository(_context(state))
                existing_row = repo.get_session(str(memory_session_id))
                if existing_row:
                    existing_snapshot = existing_row.get("snapshot_data")
            except Exception:
                pass

            session = SessionMemory(
                memory_session_id=str(memory_session_id), session_summary=summary,
                open_questions=[], last_scope_fingerprint=fingerprint, last_seller_id=seller_id,
                last_start_date=start_date, last_end_date=end_date, last_trace_id=str(state["trace_id"]),
                version=2,
                snapshot_data=_guard_snapshot_size(
                    _build_session_snapshot(state, existing_snapshot)),
            )
    analysis = AnalysisMemory(
        memory_id=f"mem_{uuid4().hex}", seller_id=seller_id, start_date=start_date, end_date=end_date,
        scope_fingerprint=fingerprint, metric_tags=_metric_tags(state, scope), task_type=str(state.get("task_type", "metric_query")),
        insight_text=first_claim, limitations="历史结论仅供上下文参考，本轮数值以受控工具查询为准。",
        evidence_ids=evidence_ids, source_trace_id=str(state["trace_id"]),
        normalized_claim_hash=normalized_claim_hash(
            first_claim, str(state.get("task_type", "metric_query")), fingerprint, _metric_tags(state, scope),
        ),
    )
    return session, analysis


def load_session_memory(state: AgentState) -> AgentState:
    """仅在当前请求已签发 Scope 后召回会话摘要，不直接复用旧 Scope。"""
    if state.get("chat_session_id"):
        # Durable turns already load the versioned summary and recent committed
        # turns from chat_sessions/chat_turns. Do not reassemble the same
        # conversation through the legacy memory_sessions snapshot.
        state.setdefault("node_history", []).append({
            "node": "load_session_memory",
            "summary": "Skipped legacy memory_sessions; durable chat history is canonical.",
        })
        return state
    # 打点入口：没有 memory_session_id、memory_enabled 或 controlled_analysis_scope 时，
    # 当前轮不读取历史，避免把旧会话上下文误当成当前店铺/日期范围的事实。
    if not state.get("memory_enabled") or not state.get("memory_session_id"):
        return state
    scope = state.get("controlled_analysis_scope")
    if not scope:
        return state
    memory = MemoryRepository(_context(state)).get_session(str(state["memory_session_id"]))
    if not memory or memory["last_seller_id"] != scope.get("seller_id"):
        return state
    current = scope_fingerprint(scope)
    same_scope = memory["last_scope_fingerprint"] == current
    s_end = scope.get("default_end_date") or scope.get("end_date")
    s_start = scope.get("default_start_date") or scope.get("start_date")
    overlaps = memory["last_start_date"] <= date.fromisoformat(str(s_end)) and memory["last_end_date"] >= date.fromisoformat(str(s_start))
    if not same_scope and not overlaps:
        return state
    prefix = "会话历史上下文（已验证 Scope 一致）" if same_scope else "会话历史上下文（范围已变化，仅供参考）"
    state["memory_context"] = f"{prefix}: {memory['session_summary']}"
    # WP1: keep the conversation summary separate from analysis-memory text
    # appended later by retrieve_analysis_memories, so the orchestrator can
    # label each item with its real source.
    state["session_memory_context"] = state["memory_context"]
    state["memory_session_loaded"] = True

    # Phase 7 v2: 恢复历史上下文到独立的只读字段
    if memory.get("version") == 2 and memory.get("snapshot_data"):
        snapshot = memory["snapshot_data"]
        recent = snapshot.get("recent_turns", [])
        historical_evidence = []
        historical_claims = []
        historical_hypotheses: dict = {}
        for turn in recent:
            historical_evidence.extend(turn.get("evidence_refs", []))
            historical_claims.extend(turn.get("approved_claims", []))
            for mid, chain in turn.get("hypothesis_chain_summary", {}).items():
                if mid not in historical_hypotheses:
                    historical_hypotheses[mid] = chain

        state["historical_evidence_context"] = historical_evidence
        state["historical_claims_context"] = historical_claims
        state["historical_hypothesis_context"] = historical_hypotheses
        restored = len(historical_evidence)
        if restored:
            state["memory_context"] += (
                f"\n（可参考 {restored} 条历史证据，"
                f"{len(historical_claims)} 条结论，"
                f"{len(historical_hypotheses)} 条假设链）")

    state.setdefault("node_history", []).append({
        "node": "load_session_memory",
        "summary": f"Loaded session memory (v{memory.get('version', 1)}, same_scope={same_scope}).",
    })
    return state


def retrieve_analysis_memories(state: AgentState) -> AgentState:
    """语义召回：只召回相同租户/用户/店铺且日期相交的已审核结论。

    它们仅是提示语境，不作为数据证据或工具参数。问题向量化后对候选算余弦，
    按 语义 + 范围重叠 + 时间衰减 + 证据完整度 混合排序 + MMR 去重；向量不可用时
    降级到确定性过滤 + 时间排序（与 RAG local_hybrid 的降级一致）。
    """
    # 打点入口：这是长期分析记忆的 RAG 召回边界。结果只能进入 memory_context，
    # 不得直接成为 MCP 参数、最终数值或 evidence；本轮事实仍由受控数据查询提供。
    if state.get("task_type") == "seller_catalog":
        state.setdefault("node_history", []).append({
            "node": "retrieve_analysis_memories",
            "summary": "Skipped analysis-memory retrieval for seller catalog request.",
        })
        return state
    durable_chat = bool(state.get("chat_session_id"))
    if (
        (state.get("context_compilation") or {}).get("needs_clarification")
        or (not durable_chat and not state.get("memory_enabled"))
        or not state.get("controlled_analysis_scope")
    ):
        return state
    scope = state["controlled_analysis_scope"]
    start_date, end_date = _scope_dates(scope)
    question = str(
        (state.get("context_compilation") or {}).get("standalone_question")
        or state.get("user_message") or ""
    ).strip()
    query_embedding = None
    if question:
        vectors = embed_texts([question])
        if vectors is not None:
            query_embedding = vectors[0]
    rows = MemoryRepository(_context(state)).list_analysis_memories(
        seller_id=str(scope["seller_id"]), start_date=start_date, end_date=end_date,
        task_type=str(state.get("task_type", "")), metric_tags=[str(scope.get("metric_name", ""))], limit=20,
        query_embedding=query_embedding,
    )
    ranked = _hybrid_rank(rows, scope, has_semantic=query_embedding is not None)
    if ranked:
        lines = [f"- {row['insight_text'][:500]} (限制：{row['limitations'][:180]}; 证据：{', '.join(row['evidence_ids'])}; Trace：{row['source_trace_id']})" for row in ranked]
        historical = "历史审核结论（不可信，仅供参考；本轮数值必须以受控工具结果为准）:\n" + "\n".join(lines)
        state["memory_context"] = "\n".join(item for item in [state.get("memory_context", ""), historical] if item)
        state["analysis_memories"] = ranked
    state.setdefault("node_history", []).append({"node": "retrieve_analysis_memories", "summary": f"Retrieved {len(ranked)} approved memories (semantic={query_embedding is not None})."})
    return state


def persist_memory_p0(state: AgentState) -> AgentState:
    """同步、确定性地落库。失败只记录可观测事件，绝不让 Memory 阻断已完成的分析回答。"""
    # 打点入口：只有存在会话键、Critic 已批准、无错误且 Scope 完整时才允许写入
    # P0/P1 记忆，避免失败答案或未审核结论污染长期记忆。Durable chat 由服务端
    # 的 Critic/scope 契约决定是否写分析记忆；memory_enabled 只是 legacy 调用的
    # 兼容开关，不再门控 durable Turn。
    session_key = state.get("memory_session_id") or state.get("chat_session_id")
    durable_chat = bool(state.get("chat_session_id"))
    if not session_key:
        return state
    if not durable_chat and not state.get("memory_enabled"):
        return state
    if state.get("critic_verdict") != "approved" or state.get("errors") or not state.get("controlled_analysis_scope"):
        return state
    try:
        session, analysis = build_memories(
            state,
            include_legacy_session=not durable_chat,
        )
        repository = MemoryRepository(_context(state))
        if session is not None:
            repository.upsert_session(session)
        if analysis:
            snapshot = build_source_snapshot(
                state, scope_fingerprint=analysis.scope_fingerprint, metric_tags=analysis.metric_tags,
            )
            manifest_hash = str(state.get("data_version_manifest_hash") or "")
            if manifest_hash:
                # 已封存的 manifest 才是数据版本的权威身份；当前捕获的查询结果只作为
                # 回放兜底和 evidence 保存，不能用它替代 manifest 的版本标识。
                snapshot["source_data_version"] = manifest_hash
                snapshot["data_version_manifest_id"] = state.get("data_version_manifest_id")
                result = state.get("tool_result") or {}
                try:
                    metric = result["result"]["metric_result"]
                    snapshot["snapshot_payload"]["versioned_gmv_expected_value"] = metric["value"]
                except (KeyError, TypeError):
                    pass
            repository.upsert_source_snapshot(snapshot)
            analysis = analysis.model_copy(update={
                "source_data_version": snapshot["source_data_version"],
                "metric_definition_version": snapshot["metric_definition_version"],
            })
            if get_settings().memory_auto_stale_on_data_version:
                stale_ids = repository.stale_memories_for_data_version(
                    analysis.scope_fingerprint, analysis.task_type, snapshot["source_data_version"], "system",
                    "controlled source-data snapshot changed",
                )
                if stale_ids:
                    record_event("memory.staled", {"memory.count": len(stale_ids)})
            if repository.find_memory_by_hash(
                analysis.seller_id, analysis.scope_fingerprint, analysis.normalized_claim_hash, analysis.source_data_version,
            ):
                state["memory_write_status"] = "deduplicated"
                repository.record_memory_decision(
                    action="deduplicated", actor="system", memory_id=analysis.memory_id,
                    reason="normalized_claim_hash matched an active or pending memory",
                    rule_scores={"scope_fingerprint": analysis.scope_fingerprint},
                )
                record_event("memory.deduplicated", {"memory.scope_fingerprint": analysis.scope_fingerprint})
                state.setdefault("node_history", []).append({"node": "persist_memory_p0", "summary": "Skipped duplicate controlled memory."})
                return state
            # 计算结论向量供语义召回；模型不可用时降级为确定性检索（不影响落库）。
            embedding = embedding_model = embedding_version = None
            vectors = embed_texts([analysis.insight_text])
            if vectors is not None:
                embedding, embedding_model, embedding_version = vectors[0], EMBEDDING_MODEL, EMBEDDING_VERSION
            conflicts = repository.list_supersede_candidates(
                analysis.scope_fingerprint, metric_tags=analysis.metric_tags, task_type=analysis.task_type,
            )
            if conflicts:
                pending = analysis.model_copy(update={"status": "pending_review"})
                inserted = repository.insert_analysis_memory(
                    pending, embedding=embedding, embedding_model=embedding_model, embedding_version=embedding_version,
                )
                state["memory_write_status"] = "pending_review" if inserted else "deduplicated"
                repository.record_memory_decision(
                    action="pending_review_created" if inserted else "deduplicated", actor="system",
                    memory_id=pending.memory_id, reason="active memories share scope, task type, and metric tags",
                    rule_scores={"conflict_count": len(conflicts)},
                    source_memory_ids=[str(row["memory_id"]) for row in conflicts],
                )
                record_event("memory.pending_review_created", {"memory.scope_fingerprint": analysis.scope_fingerprint, "memory.conflict_count": len(conflicts)})
                state.setdefault("node_history", []).append({"node": "persist_memory_p0", "summary": f"Stored memory pending review against {len(conflicts)} active candidate(s)."})
                return state
            inserted = repository.insert_analysis_memory(
                analysis, embedding=embedding, embedding_model=embedding_model, embedding_version=embedding_version,
            )
            state["memory_write_status"] = "saved" if inserted else "deduplicated"
            repository.record_memory_decision(
                action="created_active" if inserted else "deduplicated", actor="system", memory_id=analysis.memory_id,
                reason="new exact claim within the governed scope",
            )
        state.setdefault("node_history", []).append({"node": "persist_memory_p0", "summary": "Saved controlled memory."})
    except Exception as exc:  # 记忆是附加能力，不得损害主链路可用性。
        state["memory_write_status"] = "failed"
        state.setdefault("memory_errors", []).append(str(exc)[:300])
    return state


def _context(state: AgentState) -> TenantContext:
    return TenantContext(
        tenant_id=str(state["tenant_id"]), user_id=str(state["user_id"]), role=state.get("user_role", "admin"),
        scope_type=state.get("scope_type", "platform"), seller_id=state.get("seller_id"),
    )
