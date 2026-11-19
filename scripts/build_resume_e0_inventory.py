"""Build a provenance-preserving E0 metric and E1 candidate inventory.

This joins the live metric registry to previously saved isolated-PostgreSQL
acceptance records. It does not run queries, contact a Provider, or turn old
test results into current-HEAD claims.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "testing" / "resume_data_collection_design_v1"

from app.metrics.registry import get_registry


def read_json(rel: str) -> Any:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def sha256(rel: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / rel).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> None:
    registry = get_registry()
    platform_value = read_json("docs/testing/platform_value_source_sync_final_20261011.json")
    platform_trend = read_json("docs/testing/platform_trend_acceptance_20261010.json")
    seller_value = read_json("docs/testing/seller_value_acceptance_46_20261010.json")
    seller_trend = read_json("docs/testing/seller_trend_acceptance_46_rerun_20261010.json")
    qualification = read_json("docs/testing/merge_acceptance_20261015/qualification_17.json")
    candidates = read_json("docs/testing/provider_candidate_protocol_17_20261013/candidate_protocol_17.json")

    platform_value_by_id = {row["metric_id"]: row for row in platform_value["workflow_cases"]}
    platform_trend_by_id = {row["metric_id"]: row for row in platform_trend["cases"]}
    seller_value_by_id = {row["metric_id"]: row for row in seller_value["cases"]}
    seller_trend_by_id = {row["metric_id"]: row for row in seller_trend["cases"]}
    candidate_by_id = {
        str(row.get("case_id") or row.get("id")): row
        for row in candidates.get("cases", candidates if isinstance(candidates, list) else [])
    }

    metric_rows: list[dict[str, Any]] = []
    for metric in registry.list():
        scope = "platform" if "platform" in metric.implementations else "seller"
        implementation = metric.implementations[scope]
        value_row = (platform_value_by_id if scope == "platform" else seller_value_by_id).get(metric.metric_id)
        trend_row = (platform_trend_by_id if scope == "platform" else seller_trend_by_id).get(metric.metric_id)
        if scope == "platform":
            value_result = {
                "evidence": "platform_value_source_sync_final_20261011.json",
                "historical_code_version": platform_value.get("git_head"),
                "query_value_correct": bool(value_row and value_row.get("actual_value") == value_row.get("expected_value")),
                "unit_correct": bool(value_row and value_row.get("actual_unit") == value_row.get("expected_unit")),
                "evidence_bound": bool(value_row and value_row.get("evidence_bound")),
                "critic_status": value_row.get("critic_status") if value_row else None,
                "presentation_status": value_row.get("presentation_status") if value_row else None,
                "full_task_pass": bool(value_row and value_row.get("full_task_pass")),
                "expected": value_row.get("expected_value") if value_row else None,
                "actual": value_row.get("actual_value") if value_row else None,
                "source_table_expected": value_row.get("source_table_expected") if value_row else None,
                "source_table_actual": value_row.get("source_table_actual") if value_row else None,
                "source_lineage_expected": value_row.get("source_lineage_expected") if value_row else None,
                "trace": value_row.get("trace_file") if value_row else None,
            }
            trend_result = {
                "evidence": "platform_trend_acceptance_20261010.json",
                "historical_code_version": platform_trend.get("code_version"),
                "numeric_pass": bool(trend_row and trend_row.get("numeric_pass")),
                "source_metadata_pass": bool(trend_row and trend_row.get("source_metadata_pass")),
                "unit": trend_row.get("unit") if trend_row else None,
                "expected": trend_row.get("expected") if trend_row else None,
                "actual": trend_row.get("actual") if trend_row else None,
                "sql_trace": trend_row.get("sql_trace") if trend_row else None,
            }
        else:
            value_result = {
                "evidence": "seller_value_acceptance_46_20261010.json",
                "historical_code_version": seller_value.get("repository_commit"),
                "query_value_correct": bool(value_row and value_row.get("value_pass")),
                "authorization_isolation_pass": bool(value_row and value_row.get("seller_filter_pass") and value_row.get("tenant_membership_grant_pass")),
                "test_case_pass": bool(value_row and value_row.get("test_case_pass")),
                "expected": value_row.get("expected_value") if value_row else None,
                "actual": value_row.get("actual_value") if value_row else None,
                "status": value_row.get("status") if value_row else None,
                "source_table_direct": value_row.get("source_table_direct_query") if value_row else None,
                "source_lineage_returned": value_row.get("source_lineage_returned") if value_row else None,
                "exception": value_row.get("exception") if value_row else None,
            }
            value_result["full_workflow_display_verified"] = False
            trend_result = {
                "evidence": "seller_trend_acceptance_46_rerun_20261010.json",
                "historical_code_version": seller_trend.get("git_head"),
                "numeric_pass": bool(trend_row and trend_row.get("numeric_pass")),
                "authorization_isolation_pass": bool(trend_row and trend_row.get("authorization_pass")),
                "source_metadata_pass": bool(trend_row and trend_row.get("source_metadata_pass")),
                "expected": trend_row.get("expected") if trend_row else None,
                "actual": trend_row.get("actual") if trend_row else None,
                "exception": trend_row.get("exception") if trend_row else None,
                "trace": trend_row.get("sql_trace") if trend_row else None,
            }
        metric_rows.append({
            "metric_id": metric.metric_id,
            "display_name": metric.display_name,
            "primary_scope_for_52_definition_inventory": scope,
            "unit": metric.unit,
            "aggregation": metric.aggregation,
            "registered_capabilities": sorted(implementation.capabilities),
            "registered_dimensions": list(implementation.dimensions),
            "direct_query_handler": implementation.handler_key,
            "registered_source_lineage": implementation.lineage_source,
            "registered_attribution_methods": list(metric.attribution.for_scope(scope).method_ids),
            "registered_attribution_dimensions": list(metric.attribution.for_scope(scope).dimensions),
            "value_acceptance": value_result,
            "daily_trend_acceptance": trend_result,
            "current_code_retested": False,
        })

    qualification_rows = []
    case_rows = candidates.get("cases", candidates if isinstance(candidates, list) else [])
    qualification_by_id = {str(row.get("case_id") or row.get("id")): row for row in qualification.get("cases", [])}
    for candidate in case_rows:
        case_id = str(candidate.get("case_id") or candidate.get("id") or "")
        qrow = qualification_by_id.get(case_id, {})
        qualification_rows.append({
            "case_id": case_id,
            "user_question": candidate.get("user_question") or candidate.get("query"),
            "target_metric": qrow.get("target_metric") or candidate.get("target_metric"),
            "operation": qrow.get("operation") or candidate.get("operation"),
            "scope": qrow.get("scope") or candidate.get("scope"),
            "method": qrow.get("method"),
            "dimension": qrow.get("dimension"),
            "comparison_period": qrow.get("comparison_period"),
            "qualification": qrow.get("qualification", "not_in_17_case_qualification"),
            "blockers": qrow.get("blockers", []),
            "qualification_uses_observed_score": qrow.get("qualification_uses_observed_score", False),
            "truth_source": candidate.get("truth_source") or candidate.get("source"),
            "scenario_family": candidate.get("scenario_family") or candidate.get("scenario_family_id"),
            "history_classification": "public-data development candidate; prior exposure is possible; not blind",
        })

    sources = [
        "config/metrics/*.yaml",
        "docs/testing/platform_value_source_sync_final_20261011.json",
        "docs/testing/platform_trend_acceptance_20261010.json",
        "docs/testing/seller_value_acceptance_46_20261010.json",
        "docs/testing/seller_trend_acceptance_46_rerun_20261010.json",
        "docs/testing/merge_acceptance_20261015/qualification_17.json",
        "docs/testing/provider_candidate_protocol_17_20261013/candidate_protocol_17.json",
    ]
    output = {
        "schema_version": "resume-e0-inventory-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_git_head": current_head(),
        "current_worktree_dirty": True,
        "registry_catalog_digest": getattr(registry, "catalog_digest", getattr(registry, "_catalog_digest", None)),
        "measurement_method": "current live registry capability inventory joined to saved historical isolated-PostgreSQL acceptance artifacts; no query was rerun by this builder",
        "scope_selection": "one primary implementation scope per metric definition: platform when present, otherwise seller; seller-only metrics are not double-counted",
        "counts": {
            "registered_metric_definitions": len(metric_rows),
            "value_operation_slots": len(metric_rows),
            "daily_trend_operation_slots": len(metric_rows),
            "platform_primary_scope": sum(row["primary_scope_for_52_definition_inventory"] == "platform" for row in metric_rows),
            "seller_primary_scope": sum(row["primary_scope_for_52_definition_inventory"] == "seller" for row in metric_rows),
            "historical_value_numeric_pass": sum(bool(row["value_acceptance"].get("query_value_correct")) for row in metric_rows),
            "historical_trend_numeric_pass": sum(bool(row["daily_trend_acceptance"].get("numeric_pass")) for row in metric_rows),
            "historical_platform_full_value_task_pass": sum(bool(row["value_acceptance"].get("full_task_pass")) for row in metric_rows if row["primary_scope_for_52_definition_inventory"] == "platform"),
            "seller_only_value_full_workflow_verified": 0,
            "seller_new_product_trend_failure_retained": [row["metric_id"] for row in metric_rows if row["daily_trend_acceptance"].get("numeric_pass") is False],
            "retest_on_current_worktree": 0,
        },
        "sources": {rel: sha256(rel) for rel in sources if (ROOT / rel).exists()},
        "historical_data_provenance": {
            "public_dataset": "Olist public CSV snapshot v2",
            "full_history_reconciliation": "docs/testing/provider_candidate_freeze_20261014_v3/database_loading_manifest_v3.json",
            "geolocation_loaded_for_new_work": False,
            "new_database_started": False,
            "note": "The joined value/trend acceptances used their own isolated fixtures/temporary PostgreSQL; this inventory does not imply one shared new Olist load or same-HEAD rerun.",
        },
        "metrics": metric_rows,
        "e1_existing_17_case_qualification": {
            "source": "docs/testing/merge_acceptance_20261015/qualification_17.json",
            "qualification_independent_of_scores": qualification.get("qualification_independent_of_historical_scores"),
            "summary": qualification.get("summary"),
            "cases": qualification_rows,
            "formal_60_case_design_ready": False,
            "blocker": "Only this 17-candidate subset has saved same-identity public authorization qualification; 12 eligible seeds do not constitute the designed 60-case set.",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "e0_metric_coverage_and_e1_qualification.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target.relative_to(ROOT)), "counts": output["counts"], "qualification": qualification.get("summary"), "current_head": output["current_git_head"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
