"""Deterministic claim/evidence and coverage projection for Critic context."""

import json

from app.context.critic_evidence import (
    CRITIC_EVIDENCE_SUMMARY_VERSION,
    build_critic_evidence_items,
)


def _evidence(evidence_id: str, *, metric: str = "gmv") -> dict:
    return {
        "evidence_id": evidence_id,
        "source_type": "mcp",
        "source_ref": f"registry:{metric}",
        "trusted": True,
        "content": {
            "request_id": "request-private",
            "question": "PRIVATE_ORIGINAL_QUESTION",
            "metric_id": metric,
            "query_type": "metric_trend",
            "result": {
                "agent": "DataQueryAgent",
                "status": "ok",
                "parsed": {
                    "query_type": "metric_trend",
                    "metric_name": metric,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                    "granularity": "day",
                    "filters": {},
                    "authorized_metric": {"metric_id": metric, "unused": "drop"},
                    "unused": None,
                },
                "trend_result": {
                    "metric_name": metric,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                    "granularity": "day",
                    "unit": "BRL",
                    "points": [
                        {"bucket": "2024-01-01", "value": 120.0},
                        {"bucket": "2024-01-02", "value": 90.0},
                    ],
                    "status": "ok",
                },
            },
        },
    }


def test_critic_summary_keeps_claim_links_exact_facts_and_coverage() -> None:
    state = {
        "current_query_scope_fingerprint": "scope-hash",
        "data_version_manifest_hash": "data-v4",
        "claims": [{
            "claim_type": "analysis",
            "value": "GMV 下降 30 BRL，来自 GMV 趋势证据。",
            "source_evidence_ids": ["ev-trend"],
        }],
        "evidence_bundles": [
            _evidence("ev-trend"),
            _evidence("ev-trend"),  # duplicate ids are de-duplicated deterministically
            {
                "evidence_id": "ev-contribution",
                "source_type": "deterministic_attribution",
                "source_ref": "verified_contribution",
                "trusted": True,
                "content": {
                    "metric_id": "gmv",
                    "contribution": [{
                        "dimension": "product_category",
                        "components": [{"dimension_value": "health", "delta_value": -30.0}],
                    }],
                },
            },
        ],
        "analysis_presentation": {
            "limitations": ["统计贡献不能证明业务因果。"],
            "evidence": [{"evidence_id": "ev-trend"}],
        },
        "result_completeness": {"status": "complete", "missing_dimensions": []},
    }
    investigation = {
        "target": {"metric_id": "gmv"},
        "verified_evidence_ids": ["ev-trend", "ev-contribution"],
        "verified_numeric_evidence": [{"evidence_id": "ev-trend", "values": {"duplicate": True}}],
        "investigation_limitations": [],
        "business_causality": {"causal_effect_estimated": False},
        "action_results": [{
            "action_id": "action-1", "action_type": "ANALYZE_TREND",
            "status": "completed", "verified_evidence_ids": ["ev-trend"],
            "failure_reason": "omit narrative details",
        }],
    }

    items, metadata = build_critic_evidence_items(state, investigation)
    by_id = {item.item_id: item for item in items}
    claim_ledger = json.loads(by_id["critic.claim_evidence"].content)
    coverage = json.loads(by_id["critic.coverage"].content)
    trend = json.loads(by_id["evidence.ev-trend"].content)
    investigation_projection = json.loads(by_id["evidence.investigation_context"].content)

    assert metadata["schema_version"] == CRITIC_EVIDENCE_SUMMARY_VERSION
    assert metadata["required_evidence_ids"] == ["ev-contribution", "ev-trend"]
    assert metadata["deduplicated_evidence_count"] == 1
    assert claim_ledger["claims"][0]["claim_id"] == "claim_001"
    assert claim_ledger["claims"][0]["evidence_ids"] == ["ev-trend"]
    assert claim_ledger["claims"][0]["binding_status"] == "bound"
    assert coverage["metrics"] == ["gmv"]
    assert coverage["dimensions"] == ["product_category"]
    assert coverage["periods"] == ["2024-01-01", "2024-01-02"]
    assert coverage["trends"][0]["point_count"] == 2
    assert coverage["missing"] == ["统计贡献不能证明业务因果。"]
    assert trend["facts"]["result"]["trend_result"]["points"] == [
        {"bucket": "2024-01-01", "value": 120.0},
        {"bucket": "2024-01-02", "value": 90.0},
    ]
    assert investigation_projection["numeric_evidence_refs"] == ["ev-trend"]
    all_content = "\n".join(item.content for item in items)
    assert "PRIVATE_ORIGINAL_QUESTION" not in all_content
    assert "request-private" not in all_content


def test_missing_claim_evidence_is_explicit_and_not_called_bound() -> None:
    state = {
        "claims": [{
            "claim_type": "metric", "value": "GMV 是 99。",
            "source_evidence_ids": ["ev-missing"],
        }],
        "evidence_bundles": [],
    }
    items, metadata = build_critic_evidence_items(state, None)
    claims = json.loads(next(item for item in items if item.item_id == "critic.claim_evidence").content)
    coverage = json.loads(next(item for item in items if item.item_id == "critic.coverage").content)

    assert claims["claims"][0]["binding_status"] == "evidence_unavailable"
    assert metadata["missing_evidence_ids"] == ["ev-missing"]
    assert coverage["missing_evidence_ids"] == ["ev-missing"]
