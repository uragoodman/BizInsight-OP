from time import perf_counter

from app.graph.runtime import instrument_node
from app.graph.workflow import persist_trace


def test_persist_trace_runs_after_graph_step_budget(monkeypatch) -> None:
    """The forensic trace must survive the guard that stopped the workflow."""
    saved = []
    monkeypatch.setattr("app.graph.workflow.write_trace", lambda trace_id, state: saved.append((trace_id, state)))
    state = {
        "trace_id": "budget-trace",
        "node_history": [],
        "runtime_steps": 32,
        "runtime_started_at": perf_counter(),
        "runtime_budget": {"max_graph_steps": 32, "max_latency_ms": 0},
        "node_timings": [],
        "errors": [{"code": "graph_step_budget_exceeded"}],
    }

    result = instrument_node("persist_trace", persist_trace)(state)

    assert saved and saved[0][0] == "budget-trace"
    assert result["node_history"][-1]["node"] == "persist_trace"


def test_persist_trace_runs_after_graph_latency_budget(monkeypatch) -> None:
    saved = []
    monkeypatch.setattr("app.graph.workflow.write_trace", lambda trace_id, state: saved.append(trace_id))
    state = {
        "trace_id": "latency-trace",
        "node_history": [],
        "runtime_steps": 0,
        "runtime_started_at": perf_counter() - 1,
        "runtime_budget": {"max_graph_steps": 32, "max_latency_ms": 0},
        "node_timings": [],
        "errors": [{"code": "graph_latency_budget_exceeded"}],
    }

    instrument_node("persist_trace", persist_trace)(state)

    assert saved == ["latency-trace"]


def test_non_persistence_nodes_still_obey_graph_step_budget() -> None:
    state = {
        "runtime_steps": 32,
        "runtime_started_at": perf_counter(),
        "runtime_budget": {"max_graph_steps": 32, "max_latency_ms": 60_000},
        "node_timings": [],
        "errors": [],
    }

    result = instrument_node("test_node", lambda current: current)(state)

    assert result["errors"][-1]["code"] == "graph_step_budget_exceeded"


def test_data_query_database_exception_keeps_typed_context_before_tool_dispatch(monkeypatch, caplog) -> None:
    from app.graph.nodes import data_query as data_query_node
    from sqlalchemy.exc import OperationalError

    state = {
        "trace_id": "db-failure-trace",
        "runtime_steps": 0,
        "runtime_started_at": perf_counter(),
        "runtime_budget": {"max_graph_steps": 32, "max_latency_ms": 60_000},
        "node_timings": [],
        "node_history": [],
        "errors": [],
        "tool_calls": [],
        "user_message": "查询 2024-05-04 的 GMV",
        "task_type": "metric_query",
    }

    def fail_during_server_planning(*_args, **_kwargs):
        raise OperationalError("SELECT private_payload", {}, RuntimeError("secret detail"))

    monkeypatch.setattr(
        data_query_node, "plan_confirmed_metric_queries", fail_during_server_planning,
    )
    result = instrument_node("data_query", data_query_node.data_query)(state)

    error = result["errors"][-1]
    assert error["code"] == "database_error"
    assert error["request_id"] is None
    assert error["failure_context"] == {
        "category": "database_error",
        "service": "postgresql",
        "code": "database_execution_failed",
        "cause_type": "OperationalError",
        "request_id": None,
        "retryable": None,
    }
    assert error["retryable"] is False
    assert "secret detail" not in str(error)
    assert "private_payload" not in str(error)
    assert "secret detail" not in caplog.text
    assert "private_payload" not in caplog.text
    assert result["critic_verdict"] == "rejected"
    assert result["tool_calls"] == []


def test_diagnostic_plan_database_error_routes_directly_to_finalize() -> None:
    from app.graph.workflow import _route_after_diagnostic_plan, _route_after_observe

    assert _route_after_diagnostic_plan({"errors": [{"code": "database_error"}]}) == "diagnostic_finalize"
    assert _route_after_diagnostic_plan({"errors": [{"code": "token_budget_exceeded"}]}) == "diagnostic_finalize"
    assert _route_after_diagnostic_plan({"errors": []}) == "diagnostic_decide"
    assert _route_after_observe({"errors": [{"code": "investigation_execution_failed"}]}) == "diagnostic_finalize"
    assert _route_after_observe({"errors": [{"code": "token_budget_exceeded"}]}) == "diagnostic_finalize"
    assert _route_after_observe({"errors": []}) == "diagnostic_decide"
