"""Qualify Y2GB fixed-answer challenge inputs from pinned persisted traces.

This is a local, deterministic projection and audit. It does not run the Agent,
database, Reviewer, or Provider. A candidate is included only when its target
claim can be checked against saved execution evidence or the current registered
metric contract. Blocked rows remain in the qualification ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from app.metrics.registry import get_registry
from evaluation.claim_verification.validate_y2gb_plan import _read_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "docs/testing/claim_verification_pilot_20261027/y1gb_y2gb_20261028/Y2GB_CASE_PLAN.jsonl"
DEFAULT_OUT = ROOT / "docs/testing/claim_verification_pilot_20261027/y1gb_y2gb_20261028/Y2GB_QUALIFIED_SOURCES.jsonl"
DEFAULT_LOCK = ROOT / "docs/testing/claim_verification_pilot_20261027/y1gb_y2gb_20261028/Y2GB_SOURCE_LOCK.json"
RAW_CASE_IDS = {"Y2-11", "Y2-12", "Y2-13", "Y2-14"}
RAW_R5_ASSERTIONS = {
    "Y2-11": "两日整体平均配送时长等于两个日均值的简单算术平均，即使每日订单数不同也一样。",
    "Y2-12": "缺失的配送时长桶可以按0小时计入期间平均值。",
    "Y2-13": "2018-10-18 的送达时长结果是 not_found，但可以按 0 天处理，再与 2017-08-20 的 10.01 天取平均，得到两日平均 5.005 天。",
    "Y2-14": "2017-01-06 的平均每单商品数为 1.00，说明当天至少有一笔已交付订单包含多件商品。",
}

# The four remaining candidates need source facts that are absent from their
# saved traces. They are not silently converted into generic method claims.
BLOCKERS = {
    "Y2-11": "Saved trend evidence has daily means but no daily eligible-order counts; cannot verify unequal populations or pooled-mean difference.",
    "Y2-12": "Saved trend contains two measured buckets and no explicit missing-bucket result; cannot verify missing-versus-zero semantics for the target bucket.",
    "Y2-13": "Saved compare has one not_found side; no two-period sufficient statistics to qualify the claimed period-mean aggregation error.",
    "Y2-14": "Saved aggregate is 1.0 items/order but no order-level distribution. Keep the empirical claim that every order differed blocked; do not infer distribution from a mean.",
}

TRUTH_BASIS = {
    "Y2-01": ("persisted_no_query_capability_stop", "The final trace records CAPABILITY_GAP, the exact metric/dimension request, authorization and registry checks, and sql_executed=false."),
    "Y2-02": ("persisted_no_query_capability_stop", "The final trace records CAPABILITY_GAP, the exact metric/dimension request, authorization and registry checks, and sql_executed=false."),
    "Y2-03": ("persisted_no_query_capability_stop", "The final trace records CAPABILITY_GAP, the exact metric/dimension request, authorization and registry checks, and sql_executed=false."),
    "Y2-04": ("registry_metric_identity_contract", "The same registry catalog defines on_time_delivery_rate over delivered comparable orders and order_final_delivery_rate over total orders with delivered status; equal point values do not make the metric identities equivalent."),
    "Y2-05": ("registry_lifecycle_stage_contract", "The registered approval event is an approval timestamp; shipment is carrier-handoff timestamp. They are distinct lifecycle events."),
    "Y2-06": ("registry_denominator_contract", "The registered approval-rate denominator is total orders, not only already-approved orders."),
    "Y2-07": ("registry_denominator_contract", "The registered shipment-rate denominator is total orders, not only shipped orders."),
    "Y2-08": ("registry_denominator_contract", "The registered final-delivery-rate denominator is total orders, not only shipped orders."),
    "Y2-09": ("registry_denominator_contract", "The registered on-time denominator is delivered comparable orders, not all created orders."),
    "Y2-10": ("registry_denominator_contract", "A point estimate computed over delivered comparable orders does not establish the rate for all placed orders."),
    "Y2-15": ("registry_and_query_grain_contract", "The metric is registered as an order-grain average; the saved query reads the delivered-order view. It is not a review-grain average."),
    "Y2-16": ("trusted_not_found_result", "The saved compare result has a valid baseline value and current status not_found/null; a missing value is not measured zero and cannot support a numeric decline."),
    "Y2-17": ("trusted_not_found_result", "The saved compare result has a valid baseline value and current status not_found/null; a missing value is not measured zero and cannot support a numeric decrease."),
    "Y2-18": ("registry_review_grain_and_scope", "The metric is an average over review records. The saved platform review query does not establish a mean for every customer or customers without linked reviews."),
    "Y2-19": ("registered_and_sql_eligibility_contract", "The registered denominator is delivered comparable orders; the current SQL counts only delivered orders with both actual and estimated delivery timestamps, so missing timestamp rows are not automatically counted as late."),
    "Y2-20": ("persisted_platform_scope_evidence", "The actual authorization and result are platform scoped. No seller-specific query/evidence exists in the source trace; a platform rate cannot be asserted as every seller's rate."),
    "Y2-21": ("persisted_platform_scope_evidence", "The actual authorization and result are platform scoped. No seller-specific query/evidence exists in the source trace; it does not establish a named seller's average."),
    "Y2-22": ("persisted_platform_scope_evidence", "The actual trend covers only the saved platform scope and two dates. It contains no seller-scoped query or seller evidence."),
    "Y2-23": ("persisted_no_query_capability_stop", "The final trace records a platform CAPABILITY_GAP, no SQL, and no seller-scoped query/evidence; it cannot establish a seller shipment rate."),
}

RAW_TRUTH_BASIS = {
    "Y2-11": ("independent_raw_olist_statistics", "2017-01-05 has 32 eligible delivered orders and mean 128.07 h; 2017-01-06 has 4 and mean 108.62 h. The pooled mean is 125.91 h, while the unweighted mean of daily means is 118.35 h."),
    "Y2-12": ("raw_source_plus_registered_missing_value_contract", "The source request covers 2017-01-05..2017-01-06, where both buckets have values. In the same raw dataset, 2017-01-01..2017-01-04 have no eligible rows; the registered trend path represents absent metric values as null/not_found, not measured zero. The supplemental empty buckets are explicitly outside the source request interval."),
    "Y2-13": ("independent_raw_olist_statistics_and_not_found_contract", "2017-08-20 has 103 eligible delivered orders and a 10.01-day query value; 2018-10-18 has no eligible delivered order and is not_found/null. Replacing that missing value with zero to claim a 5.005-day two-point mean is invalid."),
    "Y2-14": ("independent_raw_olist_order_distribution", "On 2017-01-06 there are four eligible delivered orders and four order items; the distribution is 1 item for each of the four orders. The planned R4 claim that every order had exactly one item is true for this date and cannot be used as a planted false claim. R5 instead asserts the contradictory distribution claim that at least one order had multiple items."),
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _period_from_observation(observation: dict[str, Any]) -> dict[str, Any] | None:
    raw = observation.get("period")
    return dict(raw) if isinstance(raw, dict) else None


def _tool_period_matches(parsed: dict[str, Any], period: dict[str, Any]) -> bool:
    kind = period.get("kind")
    if "baseline_date" in period and "current_date" in period:
        return parsed.get("baseline_date", parsed.get("baseline")) == period.get("baseline_date") and parsed.get("current_date", parsed.get("current")) == period.get("current_date")
    if "metric_date" in period:
        return parsed.get("metric_date", parsed.get("date")) == period.get("metric_date")
    if "start_date" in period and "end_date" in period:
        return parsed.get("start_date", parsed.get("start")) == period.get("start_date") and parsed.get("end_date", parsed.get("end")) == period.get("end_date")
    if kind == "point_pair":
        return parsed.get("baseline_date") == period.get("baseline") and parsed.get("current_date") == period.get("current")
    if kind == "point":
        return parsed.get("metric_date") == period.get("date")
    if kind == "range":
        return parsed.get("start_date") == period.get("start") and parsed.get("end_date") == period.get("end")
    return False


def _registry_projection(registry: Any, metric_ids: list[str]) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    for metric_id in metric_ids:
        definition = registry.get(metric_id)
        platform = definition.implementations.get("platform")
        metrics.append({
            "metric_id": definition.metric_id,
            "unit": definition.unit,
            "base_grain": definition.base_grain,
            "aggregation": definition.aggregation,
            "platform_capabilities": sorted(platform.capabilities) if platform else [],
            "platform_dimensions": sorted(platform.dimensions) if platform else [],
            "platform_lineage_source": platform.lineage_source if platform else None,
            "platform_lineage_description": platform.lineage_description if platform else None,
        })
    primary = metrics[0]
    return {
        "source": "authorized_registry_catalog",
        "role": "metric_definition_not_business_numeric_evidence",
        "metric_id": primary["metric_id"],
        "unit": primary["unit"],
        "catalog_digest": registry.catalog_digest,
        "metric_contracts": metrics,
    }


def _validate_trace(plan: dict[str, Any], saved: dict[str, Any], state: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    obs = saved.get("observation") or {}
    if saved.get("run_id") != plan.get("source_run_id") or saved.get("case_id") != plan.get("source_case_id"):
        return [], "source run/case identity mismatch"
    if obs.get("metric_id") != plan.get("metric_id"):
        return [], "observation metric does not match planned target metric"
    if obs.get("scope") != "platform":
        return [], "source observation is not platform scoped"
    if _period_from_observation(obs) is None or obs.get("period_match") is not True:
        return [], "source period is absent or did not match the observed task"

    bundles = [b for b in (state.get("evidence_bundles") or []) if isinstance(b, dict)]
    terminal = state.get("terminal_code")
    decisions = [d for d in (state.get("decision_records") or []) if isinstance(d, dict)]
    evidence_ids = {b.get("evidence_id") for b in bundles if b.get("evidence_id")}
    if terminal == "CAPABILITY_GAP":
        matches = [d for d in decisions if d.get("limitation_code") == "CAPABILITY_GAP"
                   and d.get("metric_id") == plan["metric_id"]
                   and d.get("scope") == "platform"
                   and d.get("dimension") == obs.get("dimension")
                   and d.get("operation") == "metric_contribution"
                   and d.get("authorization_checked") is True
                   and d.get("registry_capability_checked") is True
                   and d.get("sql_executed") is False]
        if len(matches) != 1 or evidence_ids:
            return [], "capability-stop trace lacks one matching no-SQL decision or unexpectedly contains evidence"
        return matches, None

    tool_calls = [x for x in (state.get("query_results") or []) if isinstance(x, dict)]
    matching = []
    for call in tool_calls:
        result = call.get("result") or {}
        parsed = result.get("parsed") or {}
        if parsed.get("metric_name") != plan["metric_id"]:
            continue
        request_id = call.get("request_id")
        auth = call.get("authorization_snapshot") or {}
        if (auth.get("decision") != "allowed" or auth.get("authorization_checked") is not True
                or auth.get("metric_id") != plan["metric_id"]
                or auth.get("scope") != "platform"
                or auth.get("response_request_id") != request_id
                or auth.get("tool_result_status") != result.get("status")):
            return [], "per-call authorization snapshot is missing or inconsistent with the tool result"
        if not _tool_period_matches(auth.get("period") or {}, _period_from_observation(obs) or {}):
            return [], "per-call authorized period does not match the observed period"
        linked = [b for b in bundles if (b.get("content") or {}).get("request_id") == request_id
                  and (b.get("content") or {}).get("result") == result
                  and b.get("evidence_id") == call.get("evidence_id")
                  and b.get("trusted") is True]
        if linked:
            matching.append((call, linked))
    if not matching:
        return [], "no saved tool result is linked to a source evidence ID"
    for call, linked in matching:
        result = call.get("result") or {}
        parsed = result.get("parsed") or {}
        auth = parsed.get("authorized_metric") or {}
        if auth.get("scope") not in (None, "platform"):
            return [], "source evidence authorization scope is not platform"
        if result.get("status") not in {"ok", "not_found"}:
            return [], "source tool result is neither an accepted value nor a controlled not_found"
        if not _tool_period_matches(parsed, _period_from_observation(obs) or {}):
            return [], "actual parsed tool period does not match the reviewed observation period"
        if result.get("status") == "not_found" and not any(
                (b.get("content") or {}).get("result", {}).get("status") == "not_found" for b in linked):
            return [], "not_found status is not preserved in its linked evidence bundle"
        if result.get("status") == "ok" and auth.get("catalog_digest") != get_registry().catalog_digest:
            return [], "source evidence catalog digest differs from current registry"
    return [b for _, links in matching for b in links], None


def _load_raw_statistics_audit(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    raw = path.read_bytes()
    audit = json.loads(raw.decode("utf-8"))
    if audit.get("audit_version") != "y2gb-raw-source-statistics-r1":
        raise ValueError("unsupported raw-statistics audit version")
    dataset = audit.get("dataset") or {}
    expected_files = dataset.get("source_files") or {}
    data_dir = ROOT / dataset.get("local_cache_path", "")
    for name, metadata in expected_files.items():
        source_path = data_dir / name
        if not source_path.is_file() or _sha(source_path.read_bytes()) != metadata.get("sha256"):
            raise ValueError(f"raw-statistics source hash mismatch or missing file: {name}")
    required_cases = {"Y2-11", "Y2-12", "Y2-13", "Y2-14"}
    if not required_cases.issubset(audit):
        raise ValueError("raw-statistics audit does not cover Y2-11..Y2-14")
    # These checks prevent a stale or malformed audit from silently qualifying a case.
    if (audit["Y2-11"]["date_buckets"]["2017-01-05"]["eligible_order_count"] != 32
            or audit["Y2-11"]["date_buckets"]["2017-01-06"]["eligible_order_count"] != 4
            or audit["Y2-11"]["pooled_two_day_query_value_hours_rounded_to_2dp"] != "125.91"
            or audit["Y2-11"]["simple_average_of_rounded_daily_means_hours"] != "118.345"):
        raise ValueError("Y2-11 raw sufficient-statistic invariants do not match")
    if audit["Y2-12"]["missing_days"] != ["2017-01-01", "2017-01-02", "2017-01-03", "2017-01-04"]:
        raise ValueError("Y2-12 missing-bucket audit does not match the locked dates")
    if (audit["Y2-13"]["dates"]["2017-08-20"]["eligible_delivered_order_count"] != 103
            or audit["Y2-13"]["dates"]["2018-10-18"]["eligible_delivered_order_count"] != 0):
        raise ValueError("Y2-13 raw date coverage invariants do not match")
    if (audit["Y2-14"]["eligible_delivered_order_count"] != 4
            or audit["Y2-14"]["total_order_items"] != 4
            or audit["Y2-14"]["per_order_item_count_distribution"] != {"1": 4}):
        raise ValueError("Y2-14 raw order-grain distribution invariants do not match")
    return audit, _sha(raw)


def _supplemental_raw_evidence(case_id: str, audit: dict[str, Any], audit_sha256: str) -> dict[str, Any]:
    """Create visibly separate source facts; never masquerade as runtime tool evidence."""
    body = audit[case_id]
    # The audit's conclusion is scoring/truth material. Keep only measured facts
    # in the Reviewer-facing evidence so the answer key cannot leak into input.
    facts = {key: value for key, value in body.items() if key != "conclusion"}
    return {
        "evidence_id": f"raw-audit-{case_id}-{audit_sha256[:12]}",
        "trusted": True,
        "evidence_type": "independent_raw_source_audit",
        "origin": "locally_cached_kagglehub_csv_independent_calculation",
        "role": "supplemental_source_facts_not_agent_execution_or_tool_evidence",
        "audit_sha256": audit_sha256,
        "content": {
            "case_id": case_id,
            "metric_id": {
                "Y2-11": "avg_delivery_transit_hours",
                "Y2-12": "avg_delivery_transit_hours",
                "Y2-13": "avg_delivery_lead_time",
                "Y2-14": "avg_items_per_order",
            }[case_id],
            "scope": "platform",
            "facts": facts,
            "dataset": audit["dataset"],
            "method_sources": audit["method_sources"],
            "caveat": "Supplemental independently calculated CSV facts; not produced by the historical Agent run and not an application evidence ID. Respect the case's stated period; out-of-period facts are labeled as such.",
        },
    }


def prepare(plan_path: Path, output_path: Path, lock_path: Path | None = None,
            raw_statistics_audit_path: Path | None = None) -> dict[str, Any]:
    if output_path.exists() or (lock_path is not None and lock_path.exists()):
        raise RuntimeError("Refusing to overwrite qualification evidence or source lock")
    plan = _read_jsonl(plan_path)
    if len(plan) != 23 or len({x["case_id"] for x in plan}) != 23:
        raise ValueError("Y2 plan must contain exactly 23 unique entries")
    registry = get_registry()
    raw_audit, raw_audit_sha256 = _load_raw_statistics_audit(raw_statistics_audit_path)
    if raw_audit:
        for item in plan:
            case_id = item.get("case_id")
            if case_id not in RAW_CASE_IDS:
                continue
            if item.get("candidate_revision") != "R5_raw_statistics_qualification":
                raise ValueError(f"{case_id}: raw facts may qualify only a versioned R5 candidate, never an unmodified historical plan")
            if item.get("planned_faulty_statement") != RAW_R5_ASSERTIONS[case_id]:
                raise ValueError(f"{case_id}: planned assertion differs from the independently audited R5 claim")
    rows: list[dict[str, Any]] = []
    locked_sources: dict[str, dict[str, Any]] = {}
    for item in plan:
        trace_path = ROOT / item["source_trace"]
        raw_bytes = trace_path.read_bytes()
        saved = json.loads(raw_bytes.decode("utf-8"))
        state = (saved.get("trace") or {}).get("state_snapshot") or {}
        observation = saved.get("observation") or {}
        original_answer = state.get("final_answer") or state.get("answer")
        user_query = state.get("user_message")
        if not isinstance(original_answer, str) or not original_answer.strip() or not isinstance(user_query, str) or not user_query.strip():
            raise ValueError(f"{item['case_id']}: exact persisted question and answer are required")
        source_refs, trace_error = _validate_trace(item, saved, state)
        locked_sources.setdefault(saved["run_id"], {
            "source_case_id": saved.get("case_id"), "source_run_id": saved.get("run_id"),
            "source_trace": item["source_trace"], "source_trace_sha256": _sha(raw_bytes),
            "source_question": state.get("user_message"), "original_answer": original_answer,
            "original_answer_sha256": _sha(original_answer.encode("utf-8")),
            "task_type": state.get("task_type"), "metric_planning_decision": state.get("metric_planning_decision"),
            "authorization_snapshots": state.get("authorized_metrics"),
            "decision_records": state.get("decision_records"), "terminal_code": state.get("terminal_code"),
            "query_results": state.get("query_results"), "evidence_bundles": state.get("evidence_bundles"),
            "graph_path": (saved.get("trace") or {}).get("graph_path"),
        })
        raw_case_blocked = item["case_id"] in BLOCKERS and not (raw_audit and item["case_id"] in RAW_CASE_IDS)
        status = "blocked" if raw_case_blocked or trace_error else "qualified"
        if status == "blocked":
            blocker = (BLOCKERS.get(item["case_id"], "") if raw_case_blocked else "") or trace_error or "source validation failed"
            rows.append({"case_id": item["case_id"], "qualification_status": "blocked", "blocker": blocker,
                         "source_trace_sha256": _sha(raw_bytes), "original_answer_sha256": _sha(original_answer.encode("utf-8"))})
            continue

        basis = RAW_TRUTH_BASIS.get(item["case_id"]) if raw_audit and item["case_id"] in RAW_CASE_IDS else TRUTH_BASIS.get(item["case_id"])
        if not basis:
            raise ValueError(f"{item['case_id']}: no independently reviewed target-truth basis")

        # The source case itself remains byte-for-byte in its original trace.
        # The reviewer corpus is a predeclared fixed-answer challenge: append
        # exactly one planned assertion and keep its key outside reviewer input.
        challenged_answer = original_answer.rstrip() + "\n\n" + item["planned_faulty_statement"]
        decision = state.get("metric_planning_decision") or {}
        contract = {
            "metric_id": item["metric_id"],
            "operation_intent": decision.get("operation_intent"),
            "dimension": decision.get("dimension"),
            "scope": "platform",
            "comparison": _period_from_observation(observation),
            "source": "persisted_trace_projection",
            "resolution_status": "reconstructed_from_saved_compiler_planner_and_observation_fields",
        }
        if contract["operation_intent"] is None:
            raise ValueError(f"{item['case_id']}: saved planner operation intent is missing")
        if contract["dimension"] != observation.get("dimension"):
            raise ValueError(f"{item['case_id']}: planner and observed dimension disagree")
        decision_rows = [d for d in (state.get("decision_records") or []) if isinstance(d, dict)]
        op_context = None
        provenance = "not_captured"
        if state.get("terminal_code") or decision_rows:
            op_context = {"terminal_code": state.get("terminal_code"), "decision_records": decision_rows,
                          "source_run_id": saved.get("run_id"), "source_trace_sha256": _sha(raw_bytes),
                          "projection_stage": "persisted_final_trace"}
            provenance = "persisted_final_trace_projection"
        evidence = [b for b in (state.get("evidence_bundles") or []) if isinstance(b, dict)]
        source_evidence = list(evidence)
        supplemental_evidence: list[dict[str, Any]] = []
        if raw_audit and item["case_id"] in RAW_CASE_IDS:
            supplemental_evidence.append(_supplemental_raw_evidence(item["case_id"], raw_audit, raw_audit_sha256 or ""))
            evidence.extend(supplemental_evidence)
        metric_ids = [item["metric_id"]]
        if item["case_id"] == "Y2-04":
            metric_ids.append("order_final_delivery_rate")
        registry_evidence = _registry_projection(registry, metric_ids)
        task_type = state.get("task_type") or "metric_query"
        reviewer_case = {
            "case_id": item["case_id"], "user_query": user_query,
            "task_requirements": [], "request_contract": contract,
            "evidence_bundles": evidence, "operational_context": op_context,
            "operational_context_provenance": provenance,
            "registry_evidence": registry_evidence,
            "answer": challenged_answer, "task_type": task_type,
        }
        basis_ref = f"Y2GB_TRUTH_AUDIT_R5.md#{item['case_id']}" if raw_audit and item["case_id"] in RAW_CASE_IDS else f"Y2GB_TRUTH_AUDIT.md#{item['case_id']}"
        rows.append({
            "case_id": item["case_id"], "qualification_status": "qualified",
            "review_case": reviewer_case,
            "truth_audit": {"verified": True, "target_issue_id": item["case_id"],
                            "basis_ref": basis_ref, "truth_scope": basis[0],
                            "basis_summary": basis[1], "source_trace_sha256": _sha(raw_bytes),
                            "original_answer_sha256": _sha(original_answer.encode("utf-8")),
                            "faulty_answer_sha256": _sha(challenged_answer.encode("utf-8")),
                            "source_evidence_ids": [b.get("evidence_id") for b in source_evidence if b.get("evidence_id")],
                            "supplemental_evidence_ids": [b.get("evidence_id") for b in supplemental_evidence],
                            "source_decision_ids": [d.get("decision_id") for d in source_refs if isinstance(d, dict) and d.get("decision_id")],
                            "source_result_statuses": [((b.get("content") or {}).get("result") or {}).get("status") for b in source_evidence],
                            "raw_statistics_audit_sha256": raw_audit_sha256 if raw_audit and item["case_id"] in RAW_CASE_IDS else None,
                            "registry_catalog_digest": registry.catalog_digest},
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(x, ensure_ascii=True, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    if lock_path is not None:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "schema_version": "y2gb-source-lock-v2" if raw_audit else "y2gb-source-lock-v1",
            "source_count": len(locked_sources),
            "candidate_count": len(rows),
            "registry_catalog_digest": registry.catalog_digest,
            "raw_statistics_audit_path": str(raw_statistics_audit_path.resolve().relative_to(ROOT)).replace("\\", "/") if raw_statistics_audit_path else None,
            "raw_statistics_audit_sha256": raw_audit_sha256,
            "registry_config_sha256": _sha((ROOT / "config/metrics/order_funnel.yaml").read_bytes() +
                                              (ROOT / "config/metrics/fulfillment.yaml").read_bytes() +
                                              (ROOT / "config/metrics/customer.yaml").read_bytes() +
                                              (ROOT / "config/metrics/quality.yaml").read_bytes()),
            "sources": list(locked_sources.values()),
        }, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    qualified = sum(row["qualification_status"] == "qualified" for row in rows)
    return {"plan_entries": len(rows), "qualified": qualified, "blocked": len(rows)-qualified,
            "registry_catalog_digest": registry.catalog_digest, "provider_calls": 0, "reviewer_calls": 0,
            "database_calls": 0, "path": str(output_path),
            "source_lock_path": str(lock_path) if lock_path else None,
            "unique_source_traces": len(locked_sources)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--raw-statistics-audit", type=Path,
                        help="Independently audited raw CSV facts; required to qualify Y2-11..14 in a new candidate revision")
    args = parser.parse_args()
    print(json.dumps(prepare(args.plan, args.output, args.source_lock, args.raw_statistics_audit), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
