"""Build the independent 101-case non-root evaluation manifest.

The manifest deliberately points at existing executable tests.  It is a
coverage inventory, not a claim that all cases are Provider-accuracy tests.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation" / "datasets" / "non_root_101_v1.json"


def nodes(file: str, names: list[str]) -> list[dict[str, str]]:
    return [
        {
            "source": f"tests/{file}::{name}",
            "verification": "pytest_unit_or_contract",
        }
        for name in names
    ]


GROUPS = {
    "analysis_report": ("普通经营分析与报告", nodes("test_chat.py", [
        "test_chat_route_queries_gmv_metric",
        "test_chat_route_queries_metric_trend",
        "test_chat_route_queries_metric_compare",
        "test_chat_route_queries_metric_breakdown",
        "test_chat_route_queries_customer_count_metric",
        "test_chat_route_queries_customer_count_trend",
        "test_chat_route_queries_repeat_purchase_rate_compare",
        "test_chat_route_queries_gmv_by_seller_state_breakdown",
        "test_mcp_metric_value_tool_can_be_called_independently",
        "test_mcp_metric_trend_tool_can_be_called_independently",
        "test_mcp_metric_compare_tool_can_be_called_independently",
        "test_mcp_metric_breakdown_tool_can_be_called_independently",
        "test_mcp_customer_count_tool_can_be_called_independently",
        "test_mcp_new_metric_trend_tool_can_be_called_independently",
        "test_mcp_new_metric_compare_tool_can_be_called_independently",
        "test_mcp_breakdown_supports_seller_state_dimension",
        "test_mcp_breakdown_supports_product_category_dimension",
        "test_mcp_breakdown_supports_payment_type_dimension",
        "test_trace_records_mcp_invocation_details",
        "test_chat_route_returns_trace",
    ]) + nodes("test_report_insights.py", [
        "test_validate_insights_accepts_grounded_factual",
        "test_validate_insights_rejects_uncited_evidence",
        "test_validate_insights_requires_evidence_for_factual",
        "test_generate_insight_artifact_draft_when_valid",
        "test_contribution_angle_runs_attribution_and_drafts",
    ])),
    "routing_memory": ("Agent 路由、记忆和多轮对话", nodes("test_relation_resolver.py", [
        "test_continue_decision_reuses_whitelisted_evidence_only",
        "test_invalid_action_and_missing_continue_target_degrade_to_none",
        "test_new_and_clarify_decisions_are_mapped",
        "test_prior_evidence_ids_reads_persisted_trace_whitelist",
        "test_service_relation_maps_llm_decisions_without_breaking_api_contract",
        "test_service_relation_falls_back_to_keywords_when_llm_unavailable",
    ]) + nodes("test_memory.py", [
        "test_scope_fingerprint_is_stable_and_scope_sensitive",
        "test_load_session_memory_only_keeps_overlapping_same_seller",
        "test_session_snapshot_rolls_to_max_turns",
        "test_session_snapshot_size_guard",
        "test_load_session_memory_different_seller",
        "test_scope_change_clears_historical_fields",
    ]) + nodes("test_memory_semantic.py", [
        "test_hybrid_rank_prefers_semantic_top",
        "test_hybrid_rank_falls_back_without_semantic",
        "test_repository_semantic_recall_and_supersede_lifecycle",
        "test_review_pending_memory_activates_and_supersedes_atomically",
    ]) + nodes("test_context_packer.py", [
        "test_packer_keeps_required_input_and_records_optional_omission",
        "test_packer_marks_required_overflow_without_silent_loss",
    ])),
    "auth_security": ("权限、安全和审计", nodes("test_iam_account.py", [
        "test_scrypt_password_hash_is_salted_and_verifiable",
        "test_multi_seller_analyst_permission_and_grants",
        "test_code_analysis_roles_have_separate_share_and_audit_permissions",
        "test_session_token_does_not_embed_authorization_role",
    ]) + nodes("test_tenancy.py", [
        "test_tenant_context_is_immutable_and_scoped",
        "test_seller_scope_requires_a_real_seller_id",
        "test_trace_read_rejects_another_seller",
        "test_qdrant_search_merges_only_current_tenant_private_knowledge",
    ]) + nodes("test_scope_locking.py", [
        "test_resolve_explorer_context_disabled_records_trace",
        "test_resolve_explorer_context_enabled_missing_payload",
        "test_data_query_refuses_global_mcp_when_scope_missing",
        "test_data_query_calls_scoped_when_scope_present",
    ]) + nodes("test_delegation.py", [
        "test_delegation_token_preserves_tenant_scope_and_rejects_tampering",
    ]) + nodes("test_phase7_rls.py", [
        "test_rls_cannot_return_another_seller_tenant",
        "test_rls_without_context_returns_no_rows",
    ]) + nodes("test_rls_enforcement.py", [
        "TestRlsEnforcement::test_cross_tenant_rows_are_invisible",
        "TestRlsEnforcement::test_unset_context_is_fail_closed",
        "TestRlsEnforcement::test_platform_scope_sees_all_tenants",
    ])),
    "coding_agent": ("Coding Agent", nodes("test_code_analysis.py", [
        "test_candidate_policy_rejects_execution_and_network_primitives",
        "test_candidate_contract_and_requirements_are_strict",
        "test_upload_create_confirm_and_tenant_isolation",
        "test_confirm_rejects_tampered_plan_hash",
        "test_expired_confirmation_persists_expired_status",
    ]) + nodes("test_code_analysis_security.py", [
        "test_policy_allows_only_the_supplied_input_path",
        "test_policy_rejects_reexported_process_and_reader_bypasses",
        "test_policy_rejects_import_time_side_effects_and_output",
        "test_runner_captures_and_bounds_import_time_output",
        "test_runner_bounds_candidate_exception_rendering",
        "test_runner_rejects_svg_and_fake_png",
        "test_runner_enforces_result_contract_and_json_limit",
        "test_fake_provider_emits_the_complete_candidate_result_contract",
        "test_provider_context_is_server_scoped_and_not_a_user_token",
        "test_worker_bounds_and_verifies_source_before_execution",
        "test_worker_verifies_plan_file_and_contract_fingerprints",
    ]) + nodes("test_code_analysis_api_contract.py", [
        "test_code_analysis_read_routes_require_read_permission",
        "test_create_job_enforces_atomic_active_and_daily_quotas",
        "test_confirmation_rejects_database_plan_hash_drift",
        "test_revise_creates_new_immutable_version_and_supersedes_parent",
        "test_artifact_download_uses_chart_metadata",
    ]) + nodes("test_code_analysis_boundaries.py", [
        "test_model_proxy_rejects_tool_messages_and_oversized_prompts",
        "test_local_artifact_store_is_atomic_and_enforces_read_bound",
    ]) + nodes("test_code_analysis_dataset.py", [
        "test_prefetch_builds_governed_parquet_artifact",
    ]) + nodes("test_code_analysis_storage.py", [
        "test_local_store_rejects_traversal_and_bounds_reads",
    ])),
    "reliability": ("可靠性和并发", nodes("test_runtime.py", [
        "test_persist_trace_runs_after_graph_step_budget",
        "test_persist_trace_runs_after_graph_latency_budget",
        "test_non_persistence_nodes_still_obey_graph_step_budget",
    ]) + nodes("test_llm_fallback.py", [
        "test_llm_chat_falls_back_on_recoverable_error",
        "test_llm_chat_non_recoverable_error_does_not_fallback",
        "test_llm_extract_falls_back_on_recoverable_error",
        "test_fallback_source_index",
    ]) + nodes("test_llm_usage.py", [
        "test_provider_reported_usage_is_bound_to_current_user_and_trace",
    ]) + nodes("test_alert_triage.py", [
        "test_cooldown_blocks_repeated_auto_diagnosis",
        "test_review_rejects_unknown_action",
    ]) + nodes("test_open_diagnosis.py", [
        "test_llm_unavailable_degrades_to_deterministic_fallback_without_claims",
        "test_diagnosed_findings_emit_pending_review_event_only",
        "test_cooldown_skips_recently_diagnosed_metric",
    ]) + nodes("test_health.py", [
        "test_health",
    ]) + nodes("test_code_analysis_security.py", [
        "test_claim_caps_crash_retries_and_returns_lease_attempt",
    ])),
}


def main() -> None:
    cases = []
    for category, (label, entries) in GROUPS.items():
        for index, entry in enumerate(entries, start=1):
            cases.append({
                "id": f"nr-{category}-{index:02d}",
                "category": category,
                "category_label": label,
                "expected": "source_test_passes",
                **entry,
            })
    counts = {key: len(value[1]) for key, value in GROUPS.items()}
    assert counts == {
        "analysis_report": 25,
        "routing_memory": 18,
        "auth_security": 18,
        "coding_agent": 25,
        "reliability": 15,
    }, counts
    payload = {
        "dataset_version": "v1-non-root-101",
        "purpose": "Independent executable coverage manifest; not Provider business accuracy.",
        "total_cases": len(cases),
        "category_counts": counts,
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "total_cases": len(cases), "category_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
