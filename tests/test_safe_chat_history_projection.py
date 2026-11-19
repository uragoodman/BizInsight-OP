from contextlib import contextmanager
from types import SimpleNamespace

from app.chat_history.safe_projection import (
    safe_analysis_presentation,
    safe_clarification,
    safe_suspended_turn_snapshot,
)


def test_suspended_snapshot_keeps_only_bounded_display_safe_clarification():
    snapshot = safe_suspended_turn_snapshot(
        {
            "task_type": "clarification",
            "clarification": {
                "status": "clarification_required",
                "reason_code": "ambiguous_dimension",
                "reason": "原问题隐私内容: SELECT secret",
                "missing_slots": ["dimension", {"raw": "private"}, "raw question SQL", "comparison_period"],
                "dimension_candidates": ["seller", "product_category", "raw_sql"],
                "original_question": "must not persist here",
                "sql": "SELECT secret",
                "token": "private",
                "state": {"full": "graph state"},
            },
        },
        "请明确一个分组维度。",
    )

    assert snapshot == {
        "task_type": "clarification",
        "answer": "请明确一个分组维度。",
        "clarification": {
            "status": "clarification_required",
            "reason_code": "ambiguous_dimension",
            "reason": "问题中有多个候选分组维度，需要明确选择。",
            "missing_slots": ["dimension", "comparison_period"],
            "dimension_candidates": ["seller", "product_category"],
        },
    }


def test_safe_clarification_rejects_unstructured_candidate_values():
    projected = safe_clarification({
        "status": "clarification_required",
        "reason_code": "ambiguous_dimension",
        "reason": "原始问题和 SQL 不得进入历史投影",
        "dimension_candidates": [None, 1, "SELECT * FROM raw.orders"],
        "ambiguous_candidates": [
            {"metric_id": "gmv", "display_name": "伪造指标名", "sql": "SELECT secret"},
            {"metric_id": "not_registered", "display_name": "不应出现"},
        ],
    })
    assert projected is not None
    assert projected["status"] == "clarification_required"
    assert projected["reason_code"] == "ambiguous_dimension"
    assert projected["reason"] == "问题中有多个候选分组维度，需要明确选择。"
    assert projected["dimension_candidates"] == []
    assert projected["ambiguous_candidates"] == [{
        "metric_id": "gmv",
        "display_name": "成交总额",
    }]


def test_safe_clarification_projects_only_registered_template_choices(monkeypatch):
    from app.analysis.templates import registry

    monkeypatch.setattr(registry, "list_chat_visible_templates", lambda: [
        {"name": "cohort_retention", "display_name": "Cohort 留存分析"},
        {"name": "customer_lifecycle", "display_name": "客户生命周期分析"},
    ])
    projected = safe_clarification({
        "status": "complex_template_required",
        "reason_code": "multiple_capabilities",
        "candidate_templates": [
            {"template_id": "cohort_retention", "display_name": "伪造名称"},
            {"template_id": "customer_lifecycle", "display_name": "伪造名称"},
            {"template_id": "unregistered", "display_name": "不应显示"},
        ],
        "original_question": "不得回显的任意原文",
    })

    assert projected is not None
    assert projected["candidate_templates"] == [
        {"template_id": "cohort_retention", "display_name": "Cohort 留存分析"},
        {"template_id": "customer_lifecycle", "display_name": "客户生命周期分析"},
    ]
    assert "original_question" not in projected


def test_premise_resume_reason_is_persisted_as_an_allowlisted_code():
    snapshot = safe_suspended_turn_snapshot({
        "interaction_reason_code": "premise_direction_mismatch",
        "premise_check": {"private": "must not persist"},
        "question": "must not persist",
    }, "请确认后继续。")

    assert snapshot["interaction_reason_code"] == "premise_direction_mismatch"
    assert "premise_check" not in snapshot
    assert "question" not in snapshot


def test_analysis_presentation_projection_keeps_cards_but_drops_unknown_fields():
    projected = safe_analysis_presentation({
        "status": "completed",
        "facts": [{"metric_name": "gmv", "value": 10, "query_results": ["not a card field"]}],
        "raw_sql": "SELECT private_data",
    })

    assert projected is not None
    assert projected["facts"][0]["value"] == 10
    assert "raw_sql" not in projected
    assert "query_results" not in projected["facts"][0]


def test_chat_history_reads_final_trace_and_restores_legacy_scope_prompt(monkeypatch):
    from app.chat_history import repository as repository_module

    row = {
        "id": "turn-1",
        "client_turn_id": None,
        "message": "按最近消费时间、购买频次和消费金额给客户做 RFM 分群。",
        "answer": "请先选择店铺、开始日期和结束日期，再确认复杂分析计划。",
        "task_type": "complex_analysis",
        "trace_id": "final-trace-1",
        "status": "completed",
        "interaction_state": "none",
        "error_code": None,
        "error_detail": {},
        "result_snapshot": {
            "presentation": {
                "status": "completed",
                "facts": [{"metric_name": "gmv", "value": 10, "date_or_range": "2018-08-01"}],
                "raw_state": {"private": True},
            },
        },
        "context_snapshot": {},
        "scope_fingerprint": None,
        "created_at": "2026-01-01T00:00:00Z",
        "resume_execution_status": None,
        "resume_execution_interaction_state": None,
        "resume_thread_id": None,
        "resume_scope_fingerprint": None,
        "session_scope_fingerprint": None,
    }

    class FakeResult:
        def mappings(self):
            return self

        def all(self):
            return [row]

    class FakeConnection:
        statement = ""

        def execute(self, statement, _params):
            self.statement = str(statement)
            return FakeResult()

    connection = FakeConnection()

    class FakeEngine:
        @contextmanager
        def begin(self):
            yield connection

    monkeypatch.setattr(repository_module, "engine", FakeEngine())
    repository = repository_module.ChatHistoryRepository.__new__(repository_module.ChatHistoryRepository)
    repository.context = SimpleNamespace(tenant_id="tenant-1", user_id="user-1")
    repository._scope = lambda _connection: None

    items, has_more = repository.list_turns("session-1", 0, 20)

    assert has_more is False
    assert "COALESCE(NULLIF(t.final_trace_id, ''), t.trace_id) AS trace_id" in connection.statement
    assert items[0]["trace_id"] == "final-trace-1"
    assert items[0]["clarification"] == {
        "status": "complex_scope_required",
        "reason_code": "scope_required",
    }
    assert items[0]["presentation"]["facts"][0]["value"] == 10
    assert "raw_state" not in items[0]["presentation"]
    assert "result_snapshot" not in items[0]
