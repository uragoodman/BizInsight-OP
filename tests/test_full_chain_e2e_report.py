import base64
import json
import os
import sys
from types import SimpleNamespace

from scripts.run_full_chain_e2e import (
    _chat_turn_disposition,
    _playwright_audit,
    _run_browser_process,
    _turn_status_by_test,
)


def test_browser_output_is_logged_as_utf8_and_escaped_for_legacy_console(tmp_path, capsys) -> None:
    output_log = tmp_path / "playwright-console.log"
    child_env = dict(os.environ)
    child_env["PYTHONIOENCODING"] = "utf-8"

    return_code = _run_browser_process(
        [sys.executable, "-c", "print('› 测试')"],
        cwd=tmp_path,
        env=child_env,
        output_log=output_log,
    )

    assert return_code == 0
    assert output_log.read_text(encoding="utf-8") == "› 测试\n"
    console_output = capsys.readouterr().out
    assert console_output == "\\u203a \\u6d4b\\u8bd5\n"
    assert console_output.isascii()


def _playwright_document(*, status: str, attachments: list[dict]) -> dict:
    return {
        "suites": [{
            "specs": [{
                "title": "expanded multi_turn case-1",
                "file": "tests/e2e/expanded-matrix.spec.ts",
                "tests": [{
                    "results": [{
                        "status": status,
                        "duration": 12,
                        "attachments": attachments,
                    }],
                }],
            }],
        }],
    }


def _json_attachment(name: str, payload: dict) -> dict:
    return {
        "name": name,
        "contentType": "application/json",
        "body": base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii"),
    }


def test_playwright_audit_maps_expanded_trace_even_when_case_fails(tmp_path) -> None:
    report = tmp_path / "playwright-report.json"
    report.write_text(json.dumps(_playwright_document(
        status="failed",
        attachments=[_json_attachment("expanded-case-evidence.json", {
            "test": "expanded multi_turn case-1",
            "status": "in_progress",
            "trace_ids": ["trace-turn-1", "trace-turn-2"],
            "client_turn_ids": ["turn-1", "turn-2"],
            "session_ids": ["session-1"],
        })],
    )), encoding="utf-8")

    audit = _playwright_audit(report)

    assert audit["trace_ids_by_test"]["expanded multi_turn case-1"] == [
        "trace-turn-1", "trace-turn-2",
    ]
    assert audit["client_turn_ids_by_test"]["expanded multi_turn case-1"] == [
        "turn-1", "turn-2",
    ]
    assert audit["session_ids_by_test"]["expanded multi_turn case-1"] == ["session-1"]
    assert audit["test_results"][0]["status"] == "failed"
    assert audit["test_results"][0]["evidence_status"] == "ok"
    assert audit["missing_trace_tests"] == []


def test_playwright_audit_preserves_turn_order_and_expected_terminal_statuses(tmp_path) -> None:
    report = tmp_path / "playwright-status-report.json"
    report.write_text(json.dumps(_playwright_document(
        status="passed",
        attachments=[
            _json_attachment("browser-diagnostics.json", {
                "test": "expanded multi_turn case-1",
                "clientTurnIds": ["turn-a", "turn-b"],
            }),
            _json_attachment("expanded-case-evidence.json", {
                "test": "expanded multi_turn case-1",
                "trace_ids": ["trace-1", "trace-2"],
                "client_turn_ids": ["turn-b", "turn-a", "turn-b"],
                "expected_browser_terminal_events": ["suspended", "completed"],
                "observed_browser_terminal_events": ["suspended", "completed"],
                "expected_terminal_statuses": ["failed", "completed"],
            }),
        ],
    )), encoding="utf-8")

    audit = _playwright_audit(report)

    title = "expanded multi_turn case-1"
    assert audit["client_turn_ids_by_test"][title] == ["turn-b", "turn-a"]
    assert audit["expected_terminal_statuses_by_test"][title] == ["failed", "completed"]
    assert audit["test_results"][0]["expected_terminal_statuses"] == ["failed", "completed"]
    assert audit["test_results"][0]["expected_browser_terminal_events"] == ["suspended", "completed"]
    assert audit["test_results"][0]["observed_browser_terminal_events"] == ["suspended", "completed"]


def test_playwright_audit_preserves_status_alternatives_and_non_durable_expectations(tmp_path) -> None:
    report = tmp_path / "playwright-status-alternatives-report.json"
    report.write_text(json.dumps(_playwright_document(
        status="passed",
        attachments=[_json_attachment("expanded-case-evidence.json", {
            "test": "expanded attribution case-1",
            "client_turn_ids": ["turn-a", "turn-b"],
            "expected_terminal_statuses": [["completed", "abstained"], None],
        })],
    )), encoding="utf-8")

    audit = _playwright_audit(report)

    assert audit["expected_terminal_statuses_by_test"]["expanded attribution case-1"] == [
        ["completed", "abstained"], None,
    ]
    assert audit["test_results"][0]["expected_terminal_statuses"] == [
        ["completed", "abstained"], None,
    ]


def test_playwright_audit_marks_interrupted_case_without_trace(tmp_path) -> None:
    report = tmp_path / "playwright-report.json"
    report.write_text(json.dumps(_playwright_document(status="interrupted", attachments=[])), encoding="utf-8")

    audit = _playwright_audit(report)

    assert audit["test_results"][0]["status"] == "interrupted"
    assert audit["test_results"][0]["trace_ids"] == []
    assert audit["test_results"][0]["evidence_status"] == "missing_trace"
    assert audit["missing_trace_tests"] == ["expanded multi_turn case-1"]


def test_playwright_audit_collects_history_trace_and_metric_turn_evidence(tmp_path) -> None:
    report = tmp_path / "playwright-history-report.json"
    report.write_text(json.dumps(_playwright_document(
        status="passed",
        attachments=[
            _json_attachment("browser-diagnostics.json", {
                "test": "ambiguous dimension history",
                "traceIds": ["trace-dimension"],
            }),
            _json_attachment("history-resume-evidence.json", {
                "test": "premise history resume",
                "session_id": "session-resume", "turn_id": "turn-resume",
                "client_turn_id": "client-resume", "thread_id": "thread-resume",
                "initial_trace_id": "trace-paused", "resumed_trace_id": "trace-resumed",
            }),
            _json_attachment("history-metric-evidence.json", {
                "test": "metric history confirmation",
                "session_id": "session-metric", "old_turn_id": "turn-old",
                "new_turn_id": "turn-new", "old_client_turn_id": "client-old",
                "new_client_turn_id": "client-new", "old_trace_id": "trace-old",
                "new_trace_id": "trace-new", "confirmed_metric_id": "gmv",
                "old_thread_id": "thread-old", "new_thread_id": "thread-new",
            }),
        ],
    )), encoding="utf-8")

    audit = _playwright_audit(report)

    assert audit["history_resume_by_test"]["premise history resume"]["turn_id"] == "turn-resume"
    metric = audit["history_metric_by_test"]["metric history confirmation"]
    assert metric["old_turn_id"] == "turn-old"
    assert metric["new_turn_id"] == "turn-new"
    assert audit["trace_ids_by_test"]["ambiguous dimension history"] == ["trace-dimension"]
    assert audit["trace_ids_by_test"]["premise history resume"] == ["trace-paused", "trace-resumed"]
    assert audit["trace_ids_by_test"]["metric history confirmation"] == ["trace-new", "trace-old"]
    assert audit["missing_trace_tests"] == []


def test_expected_human_input_is_not_reported_as_orphaned_nonterminal_turn() -> None:
    assert _chat_turn_disposition(
        status="interrupted",
        interaction_state="awaiting_human_input",
        task_type="anomaly_analysis",
        execution_status=None,
        error_code=None,
    ) == "expected_suspended_premise_confirmation"
    assert _chat_turn_disposition(
        status="interrupted",
        interaction_state="awaiting_human_input",
        task_type="metric_query",
        execution_status="suspended",
        error_code=None,
    ) == "expected_suspended_human_input"
    assert _chat_turn_disposition(
        status="interrupted",
        interaction_state="awaiting_scope",
        task_type="complex_analysis",
        execution_status=None,
        error_code=None,
    ) == "expected_suspended_scope_input"
    assert _chat_turn_disposition(
        status="running",
        interaction_state="awaiting_human_input",
        task_type="anomaly_analysis",
        execution_status="running",
        error_code=None,
    ) == "unresolved_nonterminal_turn"


def test_turn_status_report_associates_failed_and_abstained_outcomes_by_client_id() -> None:
    audit = {
        "client_turn_ids_by_test": {"case-09": ["bad-output", "premise", "resolved"]},
        "expected_terminal_statuses_by_test": {"case-09": ["failed", "abstained", "completed"]},
    }
    rows = [
        {
            "turn_id": "durable-1", "client_turn_id": "bad-output", "status": "failed",
            "interaction_state": "none", "task_type": "metric_query", "trace_id": "trace-1",
            "execution_status": "failed", "error_code": "INVALID_LLM_OUTPUT",
        },
        {
            "turn_id": "durable-premise", "client_turn_id": "premise", "status": "abstained",
            "interaction_state": "none", "task_type": "anomaly_analysis", "trace_id": "trace-premise",
            "execution_status": "abstained", "error_code": "INSUFFICIENT_EVIDENCE",
        },
        {
            "turn_id": "durable-2", "client_turn_id": "resolved", "status": "completed",
            "interaction_state": "none", "task_type": "metric_query", "trace_id": "trace-2",
            "execution_status": "completed", "error_code": None,
        },
    ]

    status = _turn_status_by_test(audit, rows)["case-09"]

    assert status["status_match"] is True
    assert [item["observed_statuses"] for item in status["status_checks"]] == [
        ["failed"], ["abstained"], ["completed"],
    ]
    assert status["unmatched_client_turn_ids"] == []


def test_turn_status_report_flags_missing_rows_and_unexpected_statuses() -> None:
    audit = {
        "client_turn_ids_by_test": {"case-08": ["premise"], "case-10": ["seller"]},
        "expected_terminal_statuses_by_test": {"case-08": ["abstained"], "case-10": ["completed"]},
    }
    status = _turn_status_by_test(audit, [{
        "turn_id": "durable-seller", "client_turn_id": "seller", "status": "failed",
        "interaction_state": "none", "task_type": "metric_query", "trace_id": "trace-seller",
        "execution_status": "failed", "error_code": "unexpected",
    }])

    assert status["case-08"]["status_match"] is False
    assert status["case-08"]["unmatched_client_turn_ids"] == ["premise"]
    assert status["case-10"]["status_match"] is False
    assert status["case-10"]["status_checks"][0]["observed_statuses"] == ["failed"]
    assert _chat_turn_disposition(
        status="interrupted",
        interaction_state="awaiting_human_input",
        task_type="anomaly_analysis",
        execution_status="suspended",
        error_code="agent_execution_failed",
    ) == "unresolved_nonterminal_turn"


def test_turn_status_report_accepts_any_allowed_terminal_status_and_no_durable_null() -> None:
    audit = {
        "client_turn_ids_by_test": {"case-accepted": ["completed-or-abstained", "non-durable"]},
        "expected_terminal_statuses_by_test": {
            "case-accepted": [["completed", "abstained"], None],
        },
    }
    status = _turn_status_by_test(audit, [{
        "turn_id": "turn-1", "client_turn_id": "completed-or-abstained", "status": "completed",
        "interaction_state": "none", "task_type": "anomaly_analysis", "trace_id": "trace-1",
        "execution_status": "completed", "error_code": None,
    }])["case-accepted"]

    assert status["status_match"] is True
    assert [item["matches_expected"] for item in status["status_checks"]] == [True, True]
    assert status["status_checks"][0]["expected_status"] == ["completed", "abstained"]


def test_turn_status_report_rejects_expected_case_without_client_turn_mapping() -> None:
    status = _turn_status_by_test({
        "client_turn_ids_by_test": {},
        "expected_terminal_statuses_by_test": {"case-without-ids": ["completed"]},
    }, [])

    assert status["case-without-ids"]["status_match"] is None
    assert status["case-without-ids"]["expectation_error"] == (
        "expected_status_count_must_match_unique_client_turn_ids"
    )


def test_stream_finalizer_persists_compiler_slots_for_same_scope_followups(monkeypatch) -> None:
    from app.api.routes import chat

    captured: dict = {}
    binding = SimpleNamespace(
        execution=SimpleNamespace(execution_id="execution-1"),
        turn=SimpleNamespace(turn_id="turn-1"),
        query_scope_fingerprint="scope-1",
        service=SimpleNamespace(commit_metadata=lambda _turn_id: {
            "turn_id": "turn-1", "final_execution_id": "execution-1", "commit_version": 1,
        }),
    )

    def fake_finalize(_binding, **kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(chat, "finalize_chat_lifecycle", fake_finalize)
    result = {
        "event": "completed", "answer": "GMV 为 100。", "trace_id": "trace-1",
        "task_type": "metric_query",
    }
    turn_resolution = {
        "known_slots": {
            "metric": {"value": "gmv", "source": "current_turn"},
            "date": {"value": "2018-08-31", "source": "current_turn"},
        },
        "scope_fingerprint": "scope-1",
    }

    assert chat._finalize_binding(binding, result, turn_resolution=turn_resolution)

    persisted = captured["result"]
    assert persisted["turn_context"]["context_slots"] == turn_resolution["known_slots"]
    assert persisted["turn_context"]["authorized_metrics"] == []
    assert persisted["turn_resolution"] == turn_resolution
