"""Prompt projection must retain registered-decomposition review evidence."""

import json

from app.context.orchestrator import ContextOrchestrator


def test_registered_decomposition_projection_keeps_review_fields() -> None:
    context = {
        "operation_intent": "registered_numeric_decomposition",
        "task_type": "metric_query",
        "target_metric_id": "avg_order_value",
        "method_id": "registered_numeric_decomposition_v1",
        "scope": {"seller_id": "seller-a"},
        "baseline_date": "2024-05-03",
        "current_date": "2024-05-04",
        "decomposition_review_status": "passed",
        "verified_numeric_evidence": [
            {"evidence_id": "ev-1", "metric_id": "avg_order_value", "value": 10.0}
        ],
        "limitations": ["business_causality_not_established"],
        "stop_reason": {"code": "ENOUGH_EVIDENCE"},
        "business_causality": "not_established_by_numeric_decomposition",
    }

    items = ContextOrchestrator()._tool_evidence_items({"critic_investigation_context": context})

    assert len(items) == 1
    projection = json.loads(items[0].content)
    assert projection["operation_intent"] == "registered_numeric_decomposition"
    assert projection["target_metric_id"] == "avg_order_value"
    assert projection["verified_numeric_evidence"][0]["evidence_id"] == "ev-1"
    assert projection["limitations"] == ["business_causality_not_established"]
    assert projection["stop_reason"] == {"code": "ENOUGH_EVIDENCE"}
    assert "verified_numeric_evidence" not in projection["omitted_fields"]
    assert "operation_intent" not in projection["omitted_fields"]
