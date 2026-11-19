from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from app.memory.service import build_memories, load_session_memory, normalized_claim_hash, persist_memory_p0, scope_fingerprint


def _scope(start: str = "2018-08-01", end: str = "2018-08-31") -> dict:
    return {
        "seller_id": "seller-a", "default_start_date": start, "default_end_date": end,
        "granularity": "day",
        "allowed_metrics": ("gmv",), "allowed_dimensions": ("product_category",),
        "metric_name": "gmv",
    }


def _state() -> dict:
    return {
        "tenant_id": "tenant-a", "user_id": "user-a", "user_role": "admin", "scope_type": "platform", "seller_id": None,
        "memory_enabled": True, "memory_session_id": "session-a", "trace_id": "trace-a",
        "controlled_analysis_scope": _scope(), "critic_verdict": "approved", "errors": [],
        "claims": [{"value": "GMV 下降 10%", "source_evidence_ids": ["ev_data"]}],
        "evidence_bundles": [{"evidence_id": "ev_data", "source_type": "mcp"}],
        "tool_result": {"result": {"metric_result": {"metric_name": "gmv"}}}, "node_history": [],
    }


def test_scope_fingerprint_is_stable_and_scope_sensitive() -> None:
    assert scope_fingerprint(_scope()) == scope_fingerprint(dict(reversed(list(_scope().items()))))
    assert scope_fingerprint(_scope()) != scope_fingerprint(_scope(end="2018-09-01"))


def test_build_memories_uses_only_evidence_bound_claims() -> None:
    session, analysis = build_memories(_state())
    assert session.last_scope_fingerprint == scope_fingerprint(_scope())
    assert analysis is not None
    assert analysis.evidence_ids == ["ev_data"]
    assert analysis.source_trace_id == "trace-a"
    assert analysis.normalized_claim_hash


def test_durable_chat_does_not_build_duplicate_legacy_session_summary(monkeypatch) -> None:
    from app.memory import service

    state = _state()
    state["chat_session_id"] = "session-a"

    class ForbiddenRepository:
        def __init__(self, _context):
            raise AssertionError("durable turns must not read memory_sessions")

    monkeypatch.setattr(service, "MemoryRepository", ForbiddenRepository)
    session, analysis = build_memories(state, include_legacy_session=False)

    assert session is None
    assert analysis is not None


def test_durable_chat_skips_legacy_session_read_and_write(monkeypatch) -> None:
    from app.memory import service

    inserted = []

    class AnalysisOnlyRepository:
        def __init__(self, _context): pass
        def get_session(self, _session_id):
            raise AssertionError("durable turns must not read memory_sessions")
        def upsert_session(self, _memory):
            raise AssertionError("durable turns must not write memory_sessions")
        def upsert_source_snapshot(self, _snapshot): pass
        def stale_memories_for_data_version(self, *_args): return []
        def find_memory_by_hash(self, *_args): return None
        def list_supersede_candidates(self, *_args, **_kwargs): return []
        def insert_analysis_memory(self, memory, **_kwargs):
            inserted.append(memory)
            return True
        def record_memory_decision(self, **_kwargs): pass

    state = _state()
    state["chat_session_id"] = "session-a"
    monkeypatch.setattr(service, "MemoryRepository", AnalysisOnlyRepository)
    monkeypatch.setattr(service, "embed_texts", lambda _texts: None)
    monkeypatch.setattr(service, "get_settings", lambda: SimpleNamespace(
        memory_auto_stale_on_data_version=False,
    ))

    result = persist_memory_p0(state)

    assert result["memory_write_status"] == "saved"
    assert len(inserted) == 1


def test_durable_chat_ignores_deprecated_memory_enabled_override(monkeypatch) -> None:
    from app.memory import service

    inserted = []

    class AnalysisOnlyRepository:
        def __init__(self, _context): pass
        def get_session(self, _session_id):
            raise AssertionError("durable turns must not read memory_sessions")
        def upsert_session(self, _memory):
            raise AssertionError("durable turns must not write memory_sessions")
        def upsert_source_snapshot(self, _snapshot): pass
        def stale_memories_for_data_version(self, *_args): return []
        def find_memory_by_hash(self, *_args): return None
        def list_supersede_candidates(self, *_args, **_kwargs): return []
        def insert_analysis_memory(self, memory, **_kwargs):
            inserted.append(memory)
            return True
        def record_memory_decision(self, **_kwargs): pass

    state = _state()
    state["chat_session_id"] = "session-a"
    state["memory_enabled"] = False
    monkeypatch.setattr(service, "MemoryRepository", AnalysisOnlyRepository)
    monkeypatch.setattr(service, "embed_texts", lambda _texts: None)
    monkeypatch.setattr(service, "get_settings", lambda: SimpleNamespace(
        memory_auto_stale_on_data_version=False,
    ))

    result = persist_memory_p0(state)

    # memory_enabled is a legacy-only compatibility switch; durable chat keeps
    # writing governed analysis memories regardless of the deprecated field.
    assert result["memory_write_status"] == "saved"
    assert len(inserted) == 1


def test_legacy_memory_disabled_still_skips_analysis_memory_write(monkeypatch) -> None:
    from app.memory import service

    class ForbiddenRepository:
        def __init__(self, _context):
            raise AssertionError("disabled legacy memory must not touch the repository")

    state = _state()
    state["memory_enabled"] = False
    monkeypatch.setattr(service, "MemoryRepository", ForbiddenRepository)

    result = persist_memory_p0(state)

    assert "memory_write_status" not in result


def test_durable_chat_ignores_deprecated_memory_enabled_for_analysis_recall(monkeypatch) -> None:
    from app.memory import service

    calls = []

    class AnalysisRepository:
        def __init__(self, _context): pass
        def list_analysis_memories(self, **kwargs):
            calls.append(kwargs)
            return []

    state = _state()
    state["chat_session_id"] = "durable-session"
    state["memory_enabled"] = False
    state["context_compilation"] = {"standalone_question": "继续分析 GMV"}
    monkeypatch.setattr(service, "MemoryRepository", AnalysisRepository)
    monkeypatch.setattr(service, "embed_texts", lambda _texts: None)

    result = service.retrieve_analysis_memories(state)

    assert len(calls) == 1
    assert calls[0]["seller_id"] == "seller-a"
    assert result is state


def test_legacy_memory_disabled_still_skips_analysis_recall(monkeypatch) -> None:
    from app.memory import service

    class ForbiddenRepository:
        def __init__(self, _context):
            raise AssertionError("disabled legacy memory must not query analysis memories")

    state = _state()
    state["memory_enabled"] = False
    monkeypatch.setattr(service, "MemoryRepository", ForbiddenRepository)

    result = service.retrieve_analysis_memories(state)

    assert result is state


def test_normalized_claim_hash_is_format_stable_but_preserves_numbers() -> None:
    fp = scope_fingerprint(_scope())
    assert normalized_claim_hash(" GMV  下降 10% ", "metric_query", fp, ["gmv"]) == normalized_claim_hash("gmv 下降 10%", "metric_query", fp, ["gmv"])
    assert normalized_claim_hash("GMV 下降 10%", "metric_query", fp, ["gmv"]) != normalized_claim_hash("GMV 下降 12%", "metric_query", fp, ["gmv"])


def test_source_snapshot_is_deterministic_and_contains_only_governed_inputs() -> None:
    from app.memory.source_versions import build_source_snapshot

    state = _state()
    fp = scope_fingerprint(_scope())
    first = build_source_snapshot(state, scope_fingerprint=fp, metric_tags=["gmv"])
    second = build_source_snapshot(state, scope_fingerprint=fp, metric_tags=["gmv"])
    assert first["source_data_version"] == second["source_data_version"]
    assert first["snapshot_payload"]["tool_result"] == state["tool_result"]
    assert "user_message" not in first["snapshot_payload"]


def test_persist_memory_deduplicates_exact_hash(monkeypatch) -> None:
    class FakeRepository:
        def __init__(self, context): pass
        def get_session(self, session_id): return None
        def upsert_session(self, memory): pass
        def upsert_source_snapshot(self, snapshot): pass
        def stale_memories_for_data_version(self, *args): return []
        def find_memory_by_hash(self, *args): return {"memory_id": "old"}
        def list_supersede_candidates(self, *args, **kwargs): return []
        def insert_analysis_memory(self, *args, **kwargs): raise AssertionError("duplicate must not insert")
        def record_memory_decision(self, **kwargs): pass
    monkeypatch.setattr("app.memory.service.MemoryRepository", FakeRepository)
    state = persist_memory_p0(_state())
    assert state["memory_write_status"] == "deduplicated"


def test_persist_memory_marks_same_scope_different_claim_pending_review(monkeypatch) -> None:
    inserted = []
    class FakeRepository:
        def __init__(self, context): pass
        def get_session(self, session_id): return None
        def upsert_session(self, memory): pass
        def upsert_source_snapshot(self, snapshot): pass
        def stale_memories_for_data_version(self, *args): return []
        def find_memory_by_hash(self, *args): return None
        def list_supersede_candidates(self, *args, **kwargs): return [{"memory_id": "old-active"}]
        def insert_analysis_memory(self, memory, **kwargs): inserted.append(memory); return True
        def record_memory_decision(self, **kwargs): pass
    monkeypatch.setattr("app.memory.service.MemoryRepository", FakeRepository)
    state = persist_memory_p0(_state())
    assert state["memory_write_status"] == "pending_review"
    assert inserted[0].status == "pending_review"


def test_build_memories_rejects_unbound_claim() -> None:
    state = _state()
    state["claims"] = [{"value": "未验证结论", "source_evidence_ids": ["ev_missing"]}]
    try:
        build_memories(state)
    except ValueError as exc:
        assert "evidence-bound" in str(exc)
    else:
        raise AssertionError("unbound claim must not be persisted")


def test_load_session_memory_only_keeps_overlapping_same_seller(monkeypatch) -> None:
    class FakeRepository:
        def __init__(self, context): pass
        def get_session(self, session_id):
            return {
                "last_seller_id": "seller-a", "last_scope_fingerprint": scope_fingerprint(_scope()),
                "last_start_date": __import__("datetime").date(2018, 8, 1), "last_end_date": __import__("datetime").date(2018, 8, 31),
                "session_summary": "已确认结论", "last_trace_id": "trace-old",
            }

    monkeypatch.setattr("app.memory.service.MemoryRepository", FakeRepository)
    state = _state()
    result = load_session_memory(state)
    assert result["memory_session_loaded"] is True
    assert "已确认结论" in result["memory_context"]


def test_durable_chat_does_not_load_legacy_session_memory(monkeypatch) -> None:
    from app.memory import service

    class ForbiddenRepository:
        def __init__(self, _context):
            raise AssertionError("durable turns must read chat_sessions, not memory_sessions")

    monkeypatch.setattr(service, "MemoryRepository", ForbiddenRepository)
    state = _state()
    state["chat_session_id"] = "session-a"

    result = load_session_memory(state)

    assert "memory_session_loaded" not in result
    assert "memory_sessions" in result["node_history"][-1]["summary"]


def test_persist_memory_failure_does_not_add_graph_error(monkeypatch) -> None:
    class BrokenRepository:
        def __init__(self, context): raise RuntimeError("database unavailable")

    monkeypatch.setattr("app.memory.service.MemoryRepository", BrokenRepository)
    state = persist_memory_p0(_state())
    assert state["memory_write_status"] == "failed"
    assert state["errors"] == []


def test_llm_consolidation_creates_pending_review_only(monkeypatch) -> None:
    from app.memory.consolidation import propose_memory_consolidation

    scope = "a" * 64
    sources = [
        {"memory_id": "m1", "seller_id": "seller-a", "start_date": date(2018, 8, 1), "end_date": date(2018, 8, 31),
         "scope_fingerprint": scope, "metric_tags": ["gmv"], "task_type": "anomaly_analysis",
         "insight_text": "订单量下滑", "limitations": "", "evidence_ids": ["ev_one"], "review_reason": "", "source_data_version": None, "metric_definition_version": None},
        {"memory_id": "m2", "seller_id": "seller-a", "start_date": date(2018, 8, 1), "end_date": date(2018, 8, 31),
         "scope_fingerprint": scope, "metric_tags": ["gmv"], "task_type": "anomaly_analysis",
         "insight_text": "客单价下滑", "limitations": "", "evidence_ids": ["ev_two"], "review_reason": "", "source_data_version": None, "metric_definition_version": None},
    ]
    inserted = []
    decisions = []

    class FakeRepository:
        def list_active_memories_for_scope(self, *_args, **_kwargs): return sources
        def insert_analysis_memory(self, memory, **_kwargs): inserted.append(memory); return True
        def record_memory_decision(self, **kwargs): decisions.append(kwargs)
        def get_analysis_memory(self, memory_id): return {"memory_id": memory_id, "status": inserted[0].status}

    monkeypatch.setattr("app.memory.consolidation.get_settings", lambda: SimpleNamespace(
        memory_llm_consolidation_enabled=True, memory_llm_consolidation_max_claims=5,
        memory_llm_consolidation_model="",
    ))
    monkeypatch.setattr("app.memory.consolidation.llm_extract", lambda **_kwargs: {
        "insight_text": "GMV 同时受订单量和客单价下滑影响", "limitations": "仅限该周期", "evidence_ids": ["ev_one", "ev_two"],
    })
    monkeypatch.setattr("app.memory.consolidation.embed_texts", lambda _texts: None)

    result = propose_memory_consolidation(FakeRepository(), scope, "reviewer")

    assert result["status"] == "pending_review"
    assert inserted[0].status == "pending_review"
    assert decisions[-1]["action"] == "proposal_created"


def test_consolidation_job_freezes_approved_sources(monkeypatch) -> None:
    from app.memory.consolidation_jobs import enqueue_consolidation_job

    class FakeRepository:
        def list_active_memories_for_scope(self, _scope, _limit):
            return [
                {"memory_id": "m1", "seller_id": "seller-a", "start_date": date(2018, 8, 1), "end_date": date(2018, 8, 31), "scope_fingerprint": "b" * 64, "metric_tags": ["gmv"], "task_type": "anomaly_analysis", "insight_text": "a", "limitations": "", "evidence_ids": ["ev_one"], "source_trace_id": "t1", "review_reason": "", "source_data_version": "1" * 64, "metric_definition_version": "2" * 64},
                {"memory_id": "m2", "seller_id": "seller-a", "start_date": date(2018, 8, 1), "end_date": date(2018, 8, 31), "scope_fingerprint": "b" * 64, "metric_tags": ["gmv"], "task_type": "anomaly_analysis", "insight_text": "b", "limitations": "", "evidence_ids": ["ev_two"], "source_trace_id": "t2", "review_reason": "", "source_data_version": "1" * 64, "metric_definition_version": "2" * 64},
            ]
        def create_consolidation_job(self, job_id, actor, scope, payload, input_hash):
            return {"job_id": job_id, "actor": actor, "scope_fingerprint": scope, "input_payload": payload, "input_hash": input_hash, "status": "queued"}

    monkeypatch.setattr("app.memory.consolidation_jobs.get_settings", lambda: SimpleNamespace(memory_llm_consolidation_enabled=True, memory_llm_consolidation_max_claims=5))
    job = enqueue_consolidation_job(FakeRepository(), "b" * 64, "reviewer")
    assert job["status"] == "queued"
    assert job["input_payload"]["source_memory_ids"] == ["m1", "m2"]
    assert "insight_text" in job["input_payload"]["sources"][0]


def test_consolidation_worker_completes_only_pending_proposal(monkeypatch) -> None:
    from app.memory import worker

    job = {"job_id": "j1", "tenant_id": "tenant-a", "user_id": "user-a", "actor": "reviewer", "scope_fingerprint": "c" * 64, "lease_token": "lease", "input_payload": {"source_memory_ids": ["m1", "m2"], "sources": [{"memory_id": "m1"}, {"memory_id": "m2"}]}}
    finished = {}
    decisions = []
    class FakeRepository:
        def __init__(self, _context): pass
        def record_memory_decision(self, **kwargs): decisions.append(kwargs)

    monkeypatch.setattr(worker, "_claim", lambda: job)
    monkeypatch.setattr(worker, "MemoryRepository", FakeRepository)
    monkeypatch.setattr(worker, "propose_memory_consolidation", lambda *_args, **_kwargs: {"memory_id": "pending-proposal", "status": "pending_review"})
    monkeypatch.setattr(worker, "_finish", lambda _job, **kwargs: finished.update(kwargs))
    assert worker.run_one() is True
    assert finished == {"status": "completed", "proposed_memory_id": "pending-proposal"}
    assert decisions[-1]["action"] == "proposal_completed"


# =========================================================================
# Phase 7: canonical_memory_scope
# =========================================================================


def test_canonical_memory_scope_normalizes_keys() -> None:
    """输入含 start_date 的 dict，输出使用 default_start_date。"""
    from app.memory.service import canonical_memory_scope

    raw = {"seller_id": "s1", "start_date": "2018-08-01", "end_date": "2018-08-31",
           "granularity": "month", "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"]}
    canonical = canonical_memory_scope(raw)
    assert "default_start_date" in canonical
    assert "default_end_date" in canonical
    assert canonical["default_start_date"] == "2018-08-01"
    assert "start_date" not in canonical
    assert "end_date" not in canonical


def test_canonical_memory_scope_excludes_metric_name() -> None:
    """不同 metric_name 但其他相同 → fingerprint 相同。"""
    from app.memory.service import canonical_memory_scope, scope_fingerprint

    a = {"seller_id": "s1", "default_start_date": "2018-08-01", "default_end_date": "2018-08-31",
         "granularity": "day", "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
         "metric_name": "gmv"}
    b = {"seller_id": "s1", "default_start_date": "2018-08-01", "default_end_date": "2018-08-31",
         "granularity": "day", "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"],
         "metric_name": "order_count"}
    assert canonical_memory_scope(a) == canonical_memory_scope(b)
    assert scope_fingerprint(a) == scope_fingerprint(b)


def test_canonical_memory_scope_differs_by_date() -> None:
    """不同 default_start_date → fingerprint 不同。"""
    from app.memory.service import scope_fingerprint

    a = {"seller_id": "s1", "default_start_date": "2018-08-01", "default_end_date": "2018-08-31",
         "granularity": "day", "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"]}
    b = {"seller_id": "s1", "default_start_date": "2018-09-01", "default_end_date": "2018-09-30",
         "granularity": "day", "allowed_metrics": ["gmv"], "allowed_dimensions": ["product_category"]}
    assert scope_fingerprint(a) != scope_fingerprint(b)


# =========================================================================
# Phase 7: _build_recent_turn 和 _build_session_snapshot
# =========================================================================


def test_build_recent_turn_excludes_review_text() -> None:
    """snapshot 中不包含 review_comment_message。"""
    from app.memory.service import _build_recent_turn

    state = _state()
    state["evidence_bundles"] = [
        {"evidence_id": "ev_review", "source_type": "explorer",
         "source_ref": "review_text",
         "content": {"review_comment_message": "敏感评论内容"}},
    ]
    turn = _build_recent_turn(state)
    for ev in turn.get("evidence_refs", []):
        assert "review_comment_message" not in str(ev)
    assert all("evidence_id" in ev for ev in turn.get("evidence_refs", []))


def test_session_snapshot_rolls_to_max_turns() -> None:
    """5 轮 → snapshot 只保留 3 轮。"""
    from app.memory.service import _build_session_snapshot

    snapshot = None
    for i in range(5):
        s = _scope(start=f"2018-08-{1+i:02d}", end=f"2018-08-{1+i:02d}")
        state = _state()
        state["controlled_analysis_scope"] = s
        state["trace_id"] = f"trace-{i}"
        snapshot = _build_session_snapshot(state, snapshot)
    assert len(snapshot["recent_turns"]) == 3
    assert snapshot["recent_turns"][0]["trace_id"] == "trace-4"


def test_session_snapshot_size_guard() -> None:
    """超 50KB → _truncated。"""
    from app.memory.service import _guard_snapshot_size

    big = {"recent_turns": [{"data": "x" * 60000}]}
    guarded = _guard_snapshot_size(big, max_bytes=1000)
    assert guarded.get("_truncated") is True
    assert len(guarded["recent_turns"]) <= len(big["recent_turns"])


# =========================================================================
# Phase 7: load_session_memory 历史字段恢复
# =========================================================================


def _v2_memory_row(same_seller: bool = True, version: int = 2) -> dict:
    fp = scope_fingerprint(_scope())
    return {
        "last_seller_id": "seller-a" if same_seller else "seller-b",
        "last_scope_fingerprint": fp,
        "last_start_date": __import__("datetime").date(2018, 8, 1),
        "last_end_date": __import__("datetime").date(2018, 8, 31),
        "session_summary": "v2 测试摘要",
        "last_trace_id": "trace-old",
        "version": version,
        "snapshot_data": {
            "schema_version": "2.0",
            "recent_turns": [{
                "trace_id": "trace-old",
                "evidence_refs": [{"evidence_id": "ev_old", "source_type": "explorer"}],
                "approved_claims": [{"value": "上轮结论", "source_evidence_ids": ["ev_old"]}],
                "hypothesis_chain_summary": {"gmv": [{"hypothesis": "品类假设", "verdict": "supported"}]},
            }],
        },
    }


def test_load_session_memory_historical_fields(monkeypatch) -> None:
    """v2 snapshot 被加载到 historical_* 字段。"""
    from app.memory.service import MemoryRepository

    class FakeV2Repo:
        def __init__(self, context): pass
        def get_session(self, session_id): return _v2_memory_row()

    monkeypatch.setattr("app.memory.service.MemoryRepository", FakeV2Repo)
    state = load_session_memory(_state())
    assert len(state.get("historical_evidence_context", [])) == 1
    assert len(state.get("historical_claims_context", [])) == 1
    assert "gmv" in state.get("historical_hypothesis_context", {})


def test_load_session_memory_does_not_append_bundles(monkeypatch) -> None:
    """evidence_bundles 不被跨轮数据污染。"""
    from app.memory.service import MemoryRepository

    class FakeV2Repo:
        def __init__(self, context): pass
        def get_session(self, session_id): return _v2_memory_row()

    monkeypatch.setattr("app.memory.service.MemoryRepository", FakeV2Repo)
    state = _state()
    original_len = len(state.get("evidence_bundles", []))
    result = load_session_memory(state)
    assert len(result.get("evidence_bundles", [])) == original_len


def test_load_session_memory_v1_fallback(monkeypatch) -> None:
    """version=1 → 不加载 snapshot。"""
    from app.memory.service import MemoryRepository

    class FakeV1Repo:
        def __init__(self, context): pass
        def get_session(self, session_id): return _v2_memory_row(version=1)

    monkeypatch.setattr("app.memory.service.MemoryRepository", FakeV1Repo)
    state = load_session_memory(_state())
    assert "historical_evidence_context" not in state
    assert state.get("memory_session_loaded") is True


def test_load_session_memory_different_seller(monkeypatch) -> None:
    """不同 seller → 不加载。"""
    from app.memory.service import MemoryRepository

    class FakeDiffSellerRepo:
        def __init__(self, context): pass
        def get_session(self, session_id): return _v2_memory_row(same_seller=False)

    monkeypatch.setattr("app.memory.service.MemoryRepository", FakeDiffSellerRepo)
    state = load_session_memory(_state())
    assert "historical_evidence_context" not in state
    assert "memory_session_loaded" not in state


def test_historical_evidence_not_in_claims(monkeypatch) -> None:
    """source_evidence_ids 中不包含历史 ev ID。"""
    from app.memory.service import MemoryRepository

    class FakeV2Repo:
        def __init__(self, context): pass
        def get_session(self, session_id): return _v2_memory_row()

    monkeypatch.setattr("app.memory.service.MemoryRepository", FakeV2Repo)
    state = load_session_memory(_state())
    for claim in state.get("claims", []):
        for eid in claim.get("source_evidence_ids", []):
            assert not eid.startswith("ev_old"), f"historical evidence {eid} found in claims"


def test_scope_change_clears_historical_fields(monkeypatch) -> None:
    """scope 变化且日期不重叠时 historical_* 字段不存在。"""
    from app.memory.service import MemoryRepository

    class FakeDiffScopeRepo:
        def __init__(self, context): pass
        def get_session(self, session_id):
            row = _v2_memory_row()
            row["last_scope_fingerprint"] = "0" * 64
            # 日期不重叠
            row["last_start_date"] = __import__("datetime").date(2020, 1, 1)
            row["last_end_date"] = __import__("datetime").date(2020, 1, 31)
            return row

    monkeypatch.setattr("app.memory.service.MemoryRepository", FakeDiffScopeRepo)
    state = load_session_memory(_state())
    assert "historical_evidence_context" not in state


def test_build_memories_v2_snapshot() -> None:
    """build_memories 构造的 SessionMemory 含 snapshot_data。"""
    session, analysis = build_memories(_state())
    assert session.version == 2
    assert session.snapshot_data is not None
    assert "recent_turns" in session.snapshot_data
    assert len(session.snapshot_data["recent_turns"]) >= 1


# =========================================================================
# Phase 7: data_query 历史上下文注入
# =========================================================================


def test_data_query_historical_context_short_question() -> None:
    """短追问（<15 字）且有历史 claims 时注入历史上下文。"""
    from app.graph.nodes.data_query import _historical_query_hint

    state = {
        "historical_claims_context": [
            {"value": "GMV 下降 10%", "source_evidence_ids": ["ev_old"]},
        ],
        "historical_hypothesis_context": {"gmv": []},
    }
    question = "按品类拆开"
    result = _historical_query_hint(state, question)
    assert "上轮确认" in result
    assert "GMV" in result
    assert question in result  # 原始问题必须保留


def test_data_query_historical_context_long_question() -> None:
    """长提问（>=15 字）不注入历史上下文。"""
    from app.graph.nodes.data_query import _historical_query_hint

    result = _historical_query_hint({}, "为什么当前范围内的 GMV 下降了")
    assert "[历史上下文" not in result


def test_data_query_historical_context_no_history() -> None:
    """无历史 claims 时不注入。"""
    from app.graph.nodes.data_query import _historical_query_hint

    result = _historical_query_hint({}, "按品类拆开")
    assert "[历史上下文" not in result
    assert result == "按品类拆开"


def test_data_query_historical_context_pending_query() -> None:
    """pending_query 时跳过注入。"""
    from app.graph.nodes.data_query import _historical_query_hint

    result = _historical_query_hint({}, "按品类拆开", pending_query={"question": "查询 GMV"})
    assert "[历史上下文" not in result
    assert result == "查询 GMV"


def test_data_query_historical_context_flows_to_planner(monkeypatch) -> None:
    """验证增强后的 query_question 进入 plan_confirmed_metric_queries，而非原始 question。"""
    from app.graph.nodes.data_query import data_query
    from app.agents.data_query_agent import plan_confirmed_metric_queries

    captured_questions = []

    def mock_planner(question, **kwargs):
        captured_questions.append(question)
        from app.mcp.schemas import MetricPlanningDecision, MetricQueryBatchPlan
        return MetricQueryBatchPlan(
            decision=MetricPlanningDecision(status="unsupported", reason="mock"),
        )

    monkeypatch.setattr("app.graph.nodes.data_query.plan_confirmed_metric_queries", mock_planner)

    state = {
        "user_message": "按品类拆开",  # short question triggers historical hint
        "task_type": "metric_query",
        "trace_id": "test-trace", "thread_id": "test-thread",
        "use_current_explorer_context": False,
        "controlled_analysis_scope": None,
        "tenant_id": "test", "user_id": "test", "user_role": "admin",
        "scope_type": "platform", "seller_id": None,
        "memory_session_id": "test-session",
        "historical_claims_context": [
            {"value": "GMV 下降 10%", "source_evidence_ids": ["ev_old"]},
        ],
        "historical_hypothesis_context": {"gmv": []},
        "tool_calls": [], "claims": [], "evidence_bundles": [],
        "errors": [], "node_history": [], "current_agent": None,
        "rag_result": {}, "rag_citations": [], "metric_context": "",
        "runtime_steps": 0, "runtime_budget": {},
        "runtime_started_at": 0, "llm_usage": {}, "node_timings": [],
    }
    result = data_query(state)
    # 断言 planner 收到的 question 包含历史上下文
    assert len(captured_questions) >= 1, "planner was never called"
    planner_q = captured_questions[0]
    assert "上轮确认" in planner_q, (
        f"planner received original question, not augmented: {planner_q!r}"
    )
    assert "按品类拆开" in planner_q, "original question content missing"
