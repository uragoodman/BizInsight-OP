"""Presentation status follows the structured diagnostic terminal outcome."""

import importlib
import socket

import pytest

from app.graph.nodes import answer as answer_node
from app.graph.nodes.answer import generate_answer
from app.graph.nodes.critic import critic


@pytest.fixture(autouse=True)
def _block_network_and_public_llm(monkeypatch):
    """These renderer/status tests must never use developer Provider credentials."""
    blocked_targets = []
    llm_calls = []

    def deny_network(target, *_args, **_kwargs):
        blocked_targets.append(str(target))
        raise OSError("network blocked by test fixture")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)

    # Prove the guard blocks before any socket call reaches the OS; clear the
    # probe afterward so only unexpected test-path attempts fail the assertion.
    with pytest.raises(OSError, match="network blocked by test fixture"):
        socket.create_connection(("198.51.100.1", 443), timeout=0.01)
    assert blocked_targets == ["('198.51.100.1', 443)"]
    blocked_targets.clear()

    llm_client = importlib.import_module("app.llm.client")

    def fake_chat(*_args, **_kwargs):
        llm_calls.append("chat")
        return '{"recommendations": []}'

    def fake_extract(*_args, **_kwargs):
        llm_calls.append("extract")
        return None

    def fake_classify(*_args, **_kwargs):
        llm_calls.append("classify")
        return None

    def fake_action(*_args, **_kwargs):
        llm_calls.append("action")
        return None

    for name, replacement in (
        ("llm_chat", fake_chat), ("llm_extract", fake_extract),
        ("llm_classify", fake_classify), ("llm_agent_action", fake_action),
    ):
        monkeypatch.setattr(llm_client, name, replacement)

    critic_node = importlib.import_module("app.graph.nodes.critic")
    for name in (
        "llm_review_evidence_completeness", "llm_review_groundedness",
        "llm_review_logic_consistency",
    ):
        monkeypatch.setattr(critic_node, name, fake_extract)

    yield {"network_attempts": blocked_targets, "llm_calls": llm_calls}

    assert not blocked_targets, f"unexpected attempted network access: {blocked_targets}"


def _analysis(stop_code: str) -> dict:
    return {
        "task_type": "anomaly_analysis",
        "summary": "已保留冲突证据；当前不能给出数值分解。",
        "abstain": stop_code != "ENOUGH_EVIDENCE",
        "evidence_ids": ["evidence-conflict-a", "evidence-conflict-b"],
        "limitations": ["指标趋势与注册分子／分母无法对账。"],
        "investigation_summary": {
            "stop_reason": {"code": stop_code, "detail": "fixture stop detail"},
        },
    }


@pytest.mark.parametrize(
    ("stop_code", "errors", "expected_status"),
    [
        ("INVALID_DATA", [], "partial"),
        ("ENOUGH_EVIDENCE", [], "completed"),
        ("DATA_INSUFFICIENT", [], "partial"),
        ("EXECUTION_FAILED", [{"code": "diagnostic_query_failed"}], "failed"),
    ],
)
def test_answer_status_matches_diagnostic_terminal_code(
    stop_code, errors, expected_status, _block_network_and_public_llm, monkeypatch,
):
    analysis = _analysis(stop_code)
    if stop_code == "ENOUGH_EVIDENCE":
        analysis.update({
            "abstain": False,
            "summary": "GMV 数值下降 20，订单量分解贡献为 -12。",
            "contribution": [{
                "dimension": "metric_factor",
                "metric_name": "gmv",
                "unit": "BRL",
                "total_delta": -20,
                "items": [{
                    "dimension_value": "order_count",
                    "baseline_value": 10,
                    "current_value": 8,
                    "delta_value": -12,
                }],
            }],
        })
        monkeypatch.setattr(
            answer_node, "_dimension_contribution_text",
            lambda *_args, **_kwargs: pytest.fail("legacy metric_factor card is not a validated dimension card"),
        )
    result = generate_answer({
        "task_type": "anomaly_analysis",
        "analysis_result": analysis,
        "tool_result": {},
        "errors": errors,
        "evidence_bundles": [],
    })

    assert result["analysis_presentation"]["status"] == expected_status
    assert result["analysis_result"] == analysis
    assert "指标趋势与注册分子／分母无法对账。" in result["answer"]
    if stop_code == "ENOUGH_EVIDENCE":
        assert len(result["analysis_presentation"]["contributions"]) == 1
        assert "订单量分解贡献为 -12" in result["answer"]
        # The fixture has no refreshed, trusted evidence IDs. Recommendation
        # LLM calls are gated on those IDs; the validated contribution prose
        # remains deterministic and must not send this unverified legacy card.
        assert _block_network_and_public_llm["llm_calls"] == []


def test_critic_preserves_terminal_diagnostic_status_over_completed_draft(_block_network_and_public_llm):
    result = critic({
        "task_type": "anomaly_analysis",
        "analysis_result": _analysis("INVALID_DATA"),
        "analysis_presentation": {"status": "completed", "limitations": []},
        "answer": "已保留冲突证据。",
        "errors": [],
        "claims": [],
        "evidence_bundles": [],
    }, llm_review_mode="critic_llm_review_off")

    assert result["analysis_presentation"]["status"] == "partial"


def test_ordinary_compare_with_typed_missing_data_is_partial(_block_network_and_public_llm):
    result = generate_answer({
        "task_type": "metric_query",
        "terminal_code": "DATA_INSUFFICIENT",
        "tool_result": {
            "question": "比较送达时长",
            "result": {
                "status": "not_found",
                "metric_name": "avg_delivery_lead_time",
                "baseline_date": "2017-08-20",
                "baseline_value": 10.01,
                "current_date": "2018-10-18",
                "current_value": None,
            },
        },
        "decision_records": [{
            "limitation_code": "DATA_INSUFFICIENT",
            "metric_id": "avg_delivery_lead_time",
            "comparison": {
                "baseline_date": "2017-08-20", "current_date": "2018-10-18",
            },
            "sql_executed": True,
        }],
        "errors": [],
        "evidence_bundles": [],
    })

    assert result["terminal_code"] == "DATA_INSUFFICIENT"
    assert result["analysis_presentation"]["status"] == "partial"
    assert result["analysis_presentation"]["facts"] == []


def test_execution_error_still_outranks_data_insufficient_terminal(_block_network_and_public_llm):
    result = generate_answer({
        "task_type": "metric_query",
        "terminal_code": "DATA_INSUFFICIENT",
        "tool_result": {"result": {"status": "not_found", "metric_name": "avg_delivery_lead_time"}},
        "errors": [{"code": "database_execution_failed"}],
        "evidence_bundles": [],
    })

    assert result["analysis_presentation"]["status"] == "failed"
