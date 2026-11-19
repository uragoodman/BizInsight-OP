from __future__ import annotations

from app.agents.data_query_agent import validate_value_tool_mapping


def test_explicit_metric_tool_mapping_matches_registry_and_mcp() -> None:
    validate_value_tool_mapping()
