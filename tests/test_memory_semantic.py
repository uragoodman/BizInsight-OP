"""P3.1/P3.2 记忆语义召回与生命周期测试：嵌入降级、混合排序、余弦检索、supersede。"""

from __future__ import annotations

from datetime import date, datetime, timezone

from app.memory.contracts import AnalysisMemory
from app.memory.embeddings import embed_texts
from app.memory.repository import MemoryRepository
from app.memory.service import _date_overlap_ratio, _hybrid_rank
from app.tenancy import SellerGrant, TenantContext

_TEST_TENANT = "test-mem-semantic"


def _ctx() -> TenantContext:
    return TenantContext(
        tenant_id=_TEST_TENANT, user_id="test-user", role="analyst",
        scope_type="seller", seller_id="s-test",
        seller_grants=(SellerGrant(seller_id="s-test"),),
    )


def _scope() -> dict:
    return {
        "seller_id": "s-test",
        "default_start_date": date(2018, 8, 1), "default_end_date": date(2018, 8, 7),
        "granularity": "day", "metric_name": "gmv",
    }


def test_embed_texts_falls_back_when_model_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("app.memory.embeddings._load_embedder", lambda: None)
    assert embed_texts(["x"]) is None
    assert embed_texts([]) == []


def test_date_overlap_ratio() -> None:
    row = {"start_date": date(2018, 8, 1), "end_date": date(2018, 8, 7)}
    assert _date_overlap_ratio(row, date(2018, 8, 1), date(2018, 8, 7)) == 1.0
    assert _date_overlap_ratio(row, date(2018, 8, 8), date(2018, 8, 14)) == 0.0


def _row(memory_id: str, semantic: float, tags: list[str], created: datetime) -> dict:
    return {
        "memory_id": memory_id, "start_date": date(2018, 8, 1), "end_date": date(2018, 8, 7),
        "created_at": created, "evidence_ids": ["ev_a", "ev_b", "ev_c", "ev_d", "ev_e"],
        "metric_tags": tags, "insight_text": "GMV 下降", "semantic_score": semantic,
        "limitations": "", "source_trace_id": f"t_{memory_id}",
    }


def test_hybrid_rank_prefers_semantic_top(monkeypatch) -> None:
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows = [
        _row("m1", 0.90, ["gmv"], now),
        _row("m2", 0.60, ["gmv"], now),
        _row("m3", 0.30, ["customer"], now),
    ]
    ranked = _hybrid_rank(rows, _scope(), has_semantic=True)
    assert ranked[0]["memory_id"] == "m1"  # 语义分最高排第一
    assert len(ranked) <= 3
    assert "_hybrid_score" in ranked[0]


def test_hybrid_rank_falls_back_without_semantic() -> None:
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows = [_row("m1", 0.9, ["gmv"], now), _row("m2", 0.5, ["gmv"], now)]
    ranked = _hybrid_rank(rows, _scope(), has_semantic=False)
    # 无向量时只按 范围/时间/证据 排序，仍然返回且不抛错
    assert len(ranked) == 2
    assert all("_hybrid_score" in r for r in ranked)


def test_repository_semantic_recall_and_supersede_lifecycle() -> None:
    from uuid import uuid4

    ctx = _ctx()
    repo = MemoryRepository(ctx)
    # 每次运行唯一 ID，避免重复运行时与上次遗留的 superseded 行冲突（UNIQUE + DO NOTHING）。
    mem_a = f"mem_sem_{uuid4().hex[:10]}"
    mem_b = f"mem_sem_{uuid4().hex[:10]}"
    a = AnalysisMemory(
        memory_id=mem_a, seller_id="s-test",
        start_date=date(2018, 8, 1), end_date=date(2018, 8, 7),
        scope_fingerprint="b" * 64, metric_tags=["gmv"], task_type="anomaly_analysis",
        insight_text="GMV 环比下降主要来自订单量下滑",
        evidence_ids=["ev_abc123"], source_trace_id=f"trace_sem_{uuid4().hex[:8]}",
    )
    repo.insert_analysis_memory(a, embedding=[1.0, 0.0, 0.0, 0.0], embedding_model="test", embedding_version="v1")
    rows = repo.list_analysis_memories(seller_id="s-test", metric_tags=["gmv"], query_embedding=[1.0, 0.0, 0.0, 0.0], limit=20)
    assert any(r["memory_id"] == mem_a for r in rows)
    assert rows[0]["semantic_score"] is not None

    # P3.2 生命周期：新结论取代旧结论，旧记忆默认不再召回，谱系可查。
    b = AnalysisMemory(
        memory_id=mem_b, seller_id="s-test",
        start_date=date(2018, 8, 1), end_date=date(2018, 8, 7),
        scope_fingerprint="b" * 64, metric_tags=["gmv"], task_type="anomaly_analysis",
        insight_text="GMV 下降主因订单量下滑（更新版）",
        evidence_ids=["ev_abc123", "ev_def456"], source_trace_id=f"trace_sem_{uuid4().hex[:8]}",
    )
    repo.insert_analysis_memory(b, embedding=[1.0, 0.0, 0.0, 0.0], embedding_model="test", embedding_version="v1")
    assert repo.supersede_analysis_memory(mem_a, mem_b, "reviewer")
    active = repo.list_analysis_memories(seller_id="s-test", metric_tags=["gmv"], limit=50)
    assert not any(r["memory_id"] == mem_a for r in active)
    old = repo.get_analysis_memory(mem_a)
    assert old["status"] == "superseded"
    assert old["superseded_by"] == mem_b


def test_review_pending_memory_activates_and_supersedes_atomically() -> None:
    """A reviewed conflict must never leave both the old and new claim active."""
    from uuid import uuid4

    repo = MemoryRepository(_ctx())
    old_id = f"mem_review_old_{uuid4().hex[:10]}"
    pending_id = f"mem_review_pending_{uuid4().hex[:10]}"
    common = {
        "seller_id": "s-test", "start_date": date(2018, 8, 1), "end_date": date(2018, 8, 7),
        "scope_fingerprint": "d" * 64, "metric_tags": ["gmv"], "task_type": "anomaly_analysis",
        "evidence_ids": ["ev_abc123"],
    }
    old = AnalysisMemory(
        memory_id=old_id, insight_text="GMV 下滑来自订单量下降", source_trace_id=f"trace_old_{uuid4().hex[:8]}",
        normalized_claim_hash=uuid4().hex * 2, **common,
    )
    pending = AnalysisMemory(
        memory_id=pending_id, insight_text="GMV 下滑来自客单价下降", source_trace_id=f"trace_pending_{uuid4().hex[:8]}",
        status="pending_review", normalized_claim_hash=uuid4().hex * 2, **common,
    )
    assert repo.insert_analysis_memory(old)
    assert repo.insert_analysis_memory(pending)
    assert not any(row["memory_id"] == pending_id for row in repo.list_analysis_memories(seller_id="s-test"))

    activated = repo.review_pending_memory(pending_id, "supersede", "reviewer", "evidence updated", old_id)

    assert activated["status"] == "active"
    assert repo.get_analysis_memory(old_id)["status"] == "superseded"
    assert repo.get_analysis_memory(old_id)["superseded_by"] == pending_id
    assert any(row["memory_id"] == pending_id for row in repo.list_analysis_memories(seller_id="s-test"))
    decisions = repo.list_memory_decisions(pending_id)
    assert any(row["action"] == "superseded" and old_id in row["source_memory_ids"] for row in decisions)


def test_source_snapshot_replay_and_version_stales_matching_memories() -> None:
    from uuid import uuid4

    repo = MemoryRepository(_ctx())
    scope = "9" * 64
    old_version = uuid4().hex * 2
    new_version = uuid4().hex * 2
    snapshot = {
        "snapshot_id": f"msnap_{old_version[:32]}", "source_data_version": old_version,
        "scope_fingerprint": scope, "source_trace_id": f"trace_snapshot_{uuid4().hex[:8]}",
        "metric_definition_version": uuid4().hex * 2,
        "snapshot_payload": {"schema_version": "v1", "result": {"gmv": 10}}, "payload_bytes": 48,
    }
    assert repo.upsert_source_snapshot(snapshot)
    assert repo.get_source_snapshot(old_version)["snapshot_payload"]["result"]["gmv"] == 10
    ids = [f"mem_stale_{uuid4().hex[:10]}", f"mem_stale_{uuid4().hex[:10]}"]
    for index, memory_id in enumerate(ids):
        assert repo.insert_analysis_memory(AnalysisMemory(
            memory_id=memory_id, seller_id="s-test", start_date=date(2018, 8, 1), end_date=date(2018, 8, 7),
            scope_fingerprint=scope, metric_tags=["gmv"], task_type="anomaly_analysis",
            insight_text=f"versioned claim {index}", evidence_ids=["ev_abc123"],
            source_trace_id=f"trace_stale_{uuid4().hex[:8]}", source_data_version=old_version,
            metric_definition_version=snapshot["metric_definition_version"], normalized_claim_hash=uuid4().hex * 2,
        ))
    assert set(repo.stale_memories_for_data_version(scope, "anomaly_analysis", new_version, "system", "ingestion updated")) == set(ids)
    assert all(repo.get_analysis_memory(memory_id)["status"] == "stale" for memory_id in ids)
    assert any(row["action"] == "staled" for row in repo.list_memory_decisions(ids[0]))
