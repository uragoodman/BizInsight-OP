from __future__ import annotations

import json

import pytest

from app.security.abac import validate_policy_rule
from app.security.result_policy import apply_result_policy


def test_abac_rule_is_bounded_and_normalized() -> None:
    rule = validate_policy_rule({
        "role": "analyst", "agent": "data_query",
        "allow": ["query_gmv", "query_gmv"], "deny": ["query_secret"],
        "conditions": {"scope_types": ["seller"], "actions": ["execute"]},
    })
    assert rule["allow"] == ["query_gmv"]
    assert rule["conditions"]["scope_types"] == ["seller"]


def test_abac_rejects_unknown_conditions() -> None:
    with pytest.raises(ValueError):
        validate_policy_rule({"role": "analyst", "agent": "data_query", "conditions": {"sql": "1=1"}})


def test_reidentification_policy_suppresses_explicit_risk() -> None:
    decision = apply_result_policy({
        "sample_size": 100,
        "quasi_identifiers": ["city", "age_band", "device", "hour"],
        "unique_combination": True,
        "value": 42,
    })
    assert decision.reidentification_risk_score >= 3
    assert decision.payload["value"] is None
    assert decision.payload["suppression_reason"] == "reidentification_risk"


def test_compose_contains_operational_workers() -> None:
    compose = open("deploy/docker-compose.yml", encoding="utf-8").read()
    assert "permission-event-worker" in compose
    assert "lifecycle-reconciler" in compose


def test_masking_migration_declares_safe_views_and_outbox() -> None:
    migration = open("migrations/versions/20260923_52_governance_controls.py", encoding="utf-8").read()
    assert "governance.mask_identifier" in migration
    assert "v_delivered_orders_safe" in migration
    assert "abac_policy_rules" in migration
    assert "permission_event_outbox" in migration
