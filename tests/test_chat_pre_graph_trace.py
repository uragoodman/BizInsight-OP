from app.tenancy.context import TenantContext


def test_pre_graph_clarification_trace_is_minimal_and_scope_bound(monkeypatch) -> None:
    from app.api.routes import chat

    captured: dict = {}

    def fake_write_trace(trace_id: str, payload: dict) -> None:
        captured["trace_id"] = trace_id
        captured["payload"] = payload

    monkeypatch.setattr("app.harness.storage.write_trace", fake_write_trace)
    context = TenantContext(
        tenant_id="tenant-a",
        user_id="owner-a",
        role="owner",
        scope_type="seller",
        seller_id="seller-a",
    )

    chat._persist_pre_graph_trace(
        context,
        trace_id="trace-a",
        task_type="clarification",
        answer="请确认这是继续当前分析任务，还是开始新的分析任务。",
        clarification={
            "status": "task_relationship_required",
            "reason_code": "task_relationship_ambiguous",
            "original_message": "换成另一个店铺再看",
            "task_candidates": [{"summary": "raw candidate"}],
        },
        scope_fingerprint="scope-a",
        task_id="task-a",
        relation_decision={
            "action": "clarify",
            "reason": "task_relationship_ambiguous",
            "resolved_message": "raw question must not enter the trace",
            "task_candidates": [{"summary": "raw candidate"}],
        },
    )

    assert captured["trace_id"] == "trace-a"
    payload = captured["payload"]
    assert payload["tenant_id"] == "tenant-a"
    assert payload["user_id"] == "owner-a"
    assert payload["seller_id"] == "seller-a"
    assert payload["status"] == "suspended"
    assert payload["interaction_state"] == "awaiting_human_input"
    assert payload["query_results"] == []
    assert payload["tool_calls"] == []
    assert payload["current_query_scope_fingerprint"] == "scope-a"
    assert payload["clarification"]["status"] == "task_relationship_required"
    assert "original_message" not in payload["clarification"]
    assert "task_candidates" not in payload["clarification"]
    assert payload["turn_relation_decision"] == {
        "action": "clarify",
        "reason": "task_relationship_ambiguous",
    }
    assert payload["node_history"][0]["reason_code"] == "task_relationship_required"


def test_pre_graph_trace_persistence_failure_does_not_raise(monkeypatch) -> None:
    from app.api.routes import chat

    monkeypatch.setattr(
        "app.harness.storage.write_trace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("trace backend down")),
    )
    context = TenantContext(
        tenant_id="tenant-a",
        user_id="owner-a",
        role="owner",
        scope_type="seller",
        seller_id="seller-a",
    )

    chat._persist_pre_graph_trace(
        context,
        trace_id="trace-a",
        task_type="clarification",
        answer="请确认范围。",
        clarification={"status": "complex_scope_required"},
        scope_fingerprint="scope-a",
    )


def test_clarification_interaction_state_is_explicit() -> None:
    from app.api.routes.chat import _lifecycle_result_status

    status, interaction, error_code, _ = _lifecycle_result_status({
        "event": "completed",
        "clarification": {"status": "task_relationship_required"},
    })
    assert (status, interaction, error_code) == (
        "suspended", "awaiting_human_input", None,
    )

    status, interaction, _, _ = _lifecycle_result_status({
        "event": "completed",
        "clarification": {"status": "complex_scope_required"},
    })
    assert (status, interaction) == ("suspended", "awaiting_scope")

    status, interaction, _, _ = _lifecycle_result_status({
        "event": "completed",
        "clarification": {"status": "invalid_llm_output"},
    })
    assert (status, interaction) == ("suspended", "awaiting_metric_confirmation")


def test_unsupported_capability_boundary_is_terminal_not_suspended() -> None:
    from app.api.routes.chat import _lifecycle_result_status, _task_status_from_result

    result = {
        "event": "completed",
        "clarification": {
            "status": "unsupported",
            "reason_code": "period_contribution_not_registered",
        },
        "presentation": {"status": "unsupported"},
    }

    assert _lifecycle_result_status(result) == ("abstained", "none", None, None)
    assert _task_status_from_result(result) == "abstained"


def test_seller_scope_change_is_refused_before_the_graph() -> None:
    from types import SimpleNamespace

    from app.api.routes.chat import _seller_scope_change_clarification

    payload = SimpleNamespace(
        message="换成另一个店铺再看同一日期的 GMV。",
        use_current_explorer_context=True,
        explorer_context=object(),
        seller_scope_payload=None,
    )
    clarification = _seller_scope_change_clarification(payload)

    assert clarification is not None
    assert clarification["status"] == "seller_scope_change_requires_new_scope"
    assert clarification["reason_code"] == "seller_scope_change_requires_new_scope"


def test_seller_scope_change_does_not_block_metric_or_explicit_seller_carrier() -> None:
    from types import SimpleNamespace

    from app.api.routes.chat import _seller_scope_change_clarification

    metric_change = SimpleNamespace(
        message="换成订单量再看。",
        use_current_explorer_context=True,
        explorer_context=object(),
        seller_scope_payload=None,
    )
    assert _seller_scope_change_clarification(metric_change) is None

    explicit_seller = SimpleNamespace(
        message="换成另一个店铺再看同一日期的 GMV。",
        use_current_explorer_context=True,
        explorer_context=object(),
        seller_scope_payload=object(),
    )
    assert _seller_scope_change_clarification(explicit_seller) is None

    unscoped = SimpleNamespace(
        message="换成另一个店铺再看同一日期的 GMV。",
        use_current_explorer_context=False,
        explorer_context=None,
        seller_scope_payload=None,
    )
    assert _seller_scope_change_clarification(unscoped) is None


def test_seller_scope_change_interaction_state_is_awaiting_scope() -> None:
    from app.api.routes.chat import _lifecycle_result_status

    status, interaction, error_code, _ = _lifecycle_result_status({
        "event": "completed",
        "clarification": {"status": "seller_scope_change_requires_new_scope"},
    })
    assert (status, interaction, error_code) == ("suspended", "awaiting_scope", None)


def test_sse_terminal_outcome_and_error_code_are_preserved_for_lifecycle() -> None:
    from app.api.routes.chat import _lifecycle_result_status, _task_status_from_result
    from app.schemas.events import SSE_EVENT_ADAPTER, SseCompletedEvent

    partial = SseCompletedEvent(
        error=True,
        error_code="critic_review_incomplete",
        error_detail="审核未完整完成。",
        terminal_status="abstained",
        presentation={"status": "partial"},
    ).model_dump(mode="json")
    partial = SSE_EVENT_ADAPTER.validate_python(partial).model_dump(mode="json")
    assert _task_status_from_result(partial) == "abstained"
    assert _lifecycle_result_status(partial) == (
        "abstained", "none", "critic_review_incomplete",
        {"code": "critic_review_incomplete", "message": "审核未完整完成。"},
    )

    rejected = SseCompletedEvent(
        error=True,
        error_code="critic_rejected",
        error_detail="结果未通过审核。",
        terminal_status="failed",
    ).model_dump(mode="json")
    assert _task_status_from_result(rejected) == "failed"
    assert _lifecycle_result_status(rejected)[:3] == (
        "failed", "none", "critic_rejected",
    )


def test_completed_terminal_with_error_is_not_reported_as_completed() -> None:
    from app.api.routes.chat import _lifecycle_result_status

    failed = {
        "event": "completed",
        "terminal_status": "completed",
        "error": True,
        "error_code": "request_contract_comparison_periods_drift",
        "error_detail": "comparison periods drifted",
        "presentation": {"status": "failed"},
    }
    assert _lifecycle_result_status(failed) == (
        "failed",
        "none",
        "request_contract_comparison_periods_drift",
        {"code": "request_contract_comparison_periods_drift", "message": "comparison periods drifted"},
    )

    partial = {
        "event": "completed",
        "terminal_status": "completed",
        "error": True,
        "error_code": "critic_review_incomplete",
        "error_detail": "review incomplete",
        "presentation": {"status": "partial"},
    }
    assert _lifecycle_result_status(partial)[0] == "abstained"
