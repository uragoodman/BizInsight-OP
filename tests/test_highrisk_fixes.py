"""Tests for the high/medium risk fixes: error model, sanitization, SSE pipeline, root_cause limit, AgentState guard."""

from app.agents.contracts import AgentError
from app.core.exceptions import sanitize_exception
from app.graph.state import AGENTSTATE_KEYS, AgentState
from app.graph.nodes.root_cause import MAX_ROOT_CAUSE_ITERATIONS, MAX_ROOT_CAUSE_DIMENSIONS, _finalize_root_cause
from app.schemas.events import SSE_EVENT_ADAPTER, SseBaseEvent, SseStartedEvent, SseNodeStartedEvent, SseCompletedEvent, SseNodeError


# ============================================================
# Step 1: Three-tier error model
# ============================================================

def test_agenterror_has_public_and_diagnostic_fields() -> None:
    """AgentError 新增字段必须存在且可选。"""
    err = AgentError(code="test", message="test")
    assert err.public_message is None
    assert err.diagnostic_message is None

    err2 = AgentError(code="test", message="test", public_message="安全描述", diagnostic_message="诊断摘要")
    assert err2.public_message == "安全描述"
    assert err2.diagnostic_message == "诊断摘要"


def test_sanitize_exception_known_type() -> None:
    """已知异常类型返回对应的 public_message。"""
    # 模拟 sqlalchemy.exc.ProgrammingError
    class FakeProgrammingError(Exception):
        pass
    FakeProgrammingError.__module__ = "sqlalchemy.exc"
    FakeProgrammingError.__qualname__ = "ProgrammingError"

    public, diagnostic = sanitize_exception(FakeProgrammingError("syntax error at or near"))
    assert public == "数据查询异常，已保留 Trace"
    assert "syntax error" in diagnostic


def test_sanitize_exception_default() -> None:
    """未知异常类型返回默认 public_message。"""
    class UnknownError(Exception):
        pass

    public, diagnostic = sanitize_exception(UnknownError("something broke"))
    assert public == "分析过程中出现内部异常，已保留 Trace"


def test_sanitize_exception_masks_sensitive_data() -> None:
    """sanitize_exception 掩盖 POSTGRES URI。"""
    raw_message = 'Connection to postgresql+psycopg://user:secret@localhost:5432/db failed'
    class DbError(Exception):
        pass
    DbError.__module__ = "sqlalchemy.exc"
    DbError.__qualname__ = "OperationalError"

    public, diagnostic = sanitize_exception(DbError(raw_message))
    assert "<REDACTED>" in diagnostic, f"expected redacted, got: {diagnostic}"
    assert "secret" not in diagnostic


# ============================================================
# Step 2: SSE event pipeline
# ============================================================

def test_sse_event_base_class_has_sequence_number() -> None:
    """所有 SSE 事件继承 SseBaseEvent，包含 sequence_number。"""
    event = SseStartedEvent(trace_id="t1")
    assert event.sequence_number == 0
    event2 = SseStartedEvent(trace_id="t1", sequence_number=42)
    assert event2.sequence_number == 42


def test_sse_discriminated_union_validates_correctly() -> None:
    """TypeAdapter 判别联合能正确识别各事件类型。"""
    raw = {"event": "started", "trace_id": "abc", "sequence_number": 0}
    validated = SSE_EVENT_ADAPTER.validate_python(raw)
    assert validated.event == "started"
    assert validated.trace_id == "abc"

    raw2 = {"event": "node_started", "node": "data_query", "sequence_number": 1}
    validated2 = SSE_EVENT_ADAPTER.validate_python(raw2)
    assert validated2.event == "node_started"
    assert validated2.node == "data_query"


def test_sse_discriminated_union_rejects_unknown() -> None:
    """未知 event 判别值抛出 ValidationError。"""
    import pydantic
    try:
        SSE_EVENT_ADAPTER.validate_python({"event": "unknown_type", "sequence_number": 0})
        assert False, "should have raised"
    except pydantic.ValidationError:
        pass


def test_sse_node_error_does_not_contain_diagnostic_message() -> None:
    """SseNodeError 不包含 diagnostic_message 敏感字段。"""
    err = SseNodeError(code="test", public_message="safe message", agent_name="data_query")
    d = err.model_dump()
    assert "diagnostic_message" not in d
    assert d["public_message"] == "safe message"


def test_sse_completed_event_carries_sequence_number() -> None:
    """completed 事件携带 sequence_number。"""
    event = SseCompletedEvent(task_type="metric_query", answer="ok", trace_id="t1", error=False, error_detail=None, sequence_number=10)
    assert event.sequence_number == 10
    d = event.model_dump(mode="json")
    assert d["sequence_number"] == 10


# ============================================================
# Step 6: AgentState keys
# ============================================================

def test_agentstate_keys_match_annotations() -> None:
    """AGENTSTATE_KEYS 必须与 AgentState 声明完全一致。"""
    assert AGENTSTATE_KEYS == frozenset(AgentState.__annotations__), (
        f"AGENTSTATE_KEYS missing: {frozenset(AgentState.__annotations__) - AGENTSTATE_KEYS}"
    )


# ============================================================
# Step 5: Root cause loop limit
# ============================================================

def test_root_cause_iteration_limit_is_sync_with_dimensions() -> None:
    """MAX_ROOT_CAUSE_ITERATIONS = 1 + MAX_ROOT_CAUSE_DIMENSIONS。"""
    assert MAX_ROOT_CAUSE_ITERATIONS == 1 + MAX_ROOT_CAUSE_DIMENSIONS


def test_finalize_root_cause_accepts_custom_stop_reason() -> None:
    """_finalize_root_cause 接受自定义 stop_reason 和 limitations。"""
    state: dict = {
        "primary_tool_result": {"result": {"metric_result": {"metric_name": "gmv"}}},
        "root_cause_evidence": [],
        "trace_id": "test",
    }
    result = _finalize_root_cause(state, stop_reason="root_cause_iteration_limit", limitations=["test limitation"])
    assert result["analysis_result"]["stop_reason"] == "root_cause_iteration_limit"
    # 因果限制说明会自动追加
    assert "test limitation" in result["analysis_result"]["limitations"]
    assert any("贡献分析不等同于因果证明" in lim for lim in result["analysis_result"]["limitations"])
    # Also verify the standard keys exist
    assert result["analysis_result"]["summary"]
