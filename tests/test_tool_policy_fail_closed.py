from __future__ import annotations

import json

from app.security.governance import ToolPermissionPolicy


def test_unknown_seller_tenant_does_not_inherit_public_allowlist(tmp_path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"public": {"analyst": {"data_query": ["query_gmv"]}}}), encoding="utf-8")
    policy = ToolPermissionPolicy(str(path))
    assert not policy.allows("tenant-unknown", "analyst", "data_query", "query_gmv", scope_type="seller")


def test_explicit_deny_wins_over_allow(tmp_path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({
        "tenant-a": {
            "analyst": {
                "data_query": {
                    "allow": ["query_gmv", "query_order_count"],
                    "deny": ["query_gmv"],
                }
            }
        }
    }), encoding="utf-8")
    policy = ToolPermissionPolicy(str(path))
    assert not policy.allows("tenant-a", "analyst", "data_query", "query_gmv", scope_type="seller")
    assert policy.allows("tenant-a", "analyst", "data_query", "query_order_count", scope_type="seller")


def test_versioned_policy_includes_avg_order_value_ratio_tool():
    policy = ToolPermissionPolicy("config/governance/tool_permissions.json")

    assert policy.allows(
        "platform", "admin", "data_query",
        "compare_avg_order_value_ratio_facts_by_dimension_period",
        scope_type="platform",
    )


def test_experiment_http_capability_is_explicit_and_not_public_for_seller_scope(tmp_path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"public": {"analyst": {"data_query": ["estimate_scoped_randomized_experiment"]}}}), encoding="utf-8")
    policy = ToolPermissionPolicy(str(path))
    assert not policy.allows("tenant-a", "analyst", "data_query", "estimate_scoped_randomized_experiment", scope_type="seller")
    assert policy.allows("platform", "admin", "data_query", "estimate_scoped_randomized_experiment", scope_type="platform")
