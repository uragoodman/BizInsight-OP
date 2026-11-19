import json

from app.security.governance import ToolPermissionPolicy


def test_tool_policy_conditions_are_default_deny(tmp_path):
    path = tmp_path / "tools.json"
    path.write_text(json.dumps({"tenant": {"analyst": {"data_query": {
        "allow": ["query_gmv"],
        "conditions": {"scope_types": ["seller"], "tenant_ids": ["tenant"]},
    }}}}), encoding="utf-8")
    policy = ToolPermissionPolicy(str(path))
    assert policy.allows("tenant", "analyst", "data_query", "query_gmv", scope_type="seller")
    assert not policy.allows("tenant", "analyst", "data_query", "query_gmv", scope_type="platform")
    assert not policy.allows("other", "analyst", "data_query", "query_gmv", scope_type="seller")
