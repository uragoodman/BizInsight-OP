"""Validate, dry-run, or explicitly execute the six-case B/G development batch.

The default behavior is not execution. ``--execute`` additionally requires an
explicit unit cap, matching estimate/authorization fields, a passing Phase A
plan, and a loopback isolated PostgreSQL admin URL. No Critic full/off runner
or 17-case candidate batch entrypoint is used here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs/testing/resume_bg_ab_dev_20261016"
RESULTS = ARTIFACTS / "results_v1"
sys.path.insert(0, str(ROOT))

from evaluation.provider_candidate_batch import (  # noqa: E402
    CostPolicy, EstimatedCostLedger, RunLedger,
    load_isolated_candidate_database,
)
from evaluation.resume_bg_executor import (  # noqa: E402
    MAX_ESTIMATE_LIMIT_CNY, run_bg_batch, select_never_started,
    validate_bg_schedule, validate_execute_budget,
)
from evaluation.resume_bg_protocol import SCORER_VERSION, score_bg_run  # noqa: E402


def _read_inputs() -> tuple[dict, list[dict], dict, dict, dict]:
    cases_document = json.loads((ARTIFACTS / "cases_6.json").read_text(encoding="utf-8"))
    cases = cases_document["cases"]
    manifest = json.loads((ARTIFACTS / "run_manifest_12.json").read_text(encoding="utf-8"))
    plan = json.loads((ARTIFACTS / "execution_plan.json").read_text(encoding="utf-8"))
    scorer = json.loads((ARTIFACTS / "score_version_v3.json").read_text(encoding="utf-8"))
    snapshot = json.loads((ROOT / "docs/testing/provider_critic_candidate_db_reconcile_20261013/coverage_manifest_olist_candidate_reconcile_070951232c48.json").read_text(encoding="utf-8"))
    return cases_document, cases, manifest, plan, {"scorer": scorer, "snapshot": snapshot}


def _validate_scorer_record(scorer: dict) -> None:
    if scorer.get("protocol_version") != SCORER_VERSION:
        raise RuntimeError("B/G execution requires the current versioned scorer protocol.")
    if scorer.get("arms") != ["basic_tool_loop", "governed_workflow"]:
        raise RuntimeError("Scoring arm set differs from the B/G execution contract.")
    if score_bg_run.__name__ != "score_bg_run":
        raise RuntimeError("B/G score function binding is invalid.")


def _snapshot_hash(snapshot: dict) -> str:
    path = ROOT / "docs/testing/provider_critic_candidate_db_reconcile_20261013/coverage_manifest_olist_candidate_reconcile_070951232c48.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(*, estimate_limit_cny: float = MAX_ESTIMATE_LIMIT_CNY) -> tuple[dict, list[dict], dict, dict, dict]:
    cases_document, cases, manifest, plan, extras = _read_inputs()
    _validate_scorer_record(extras["scorer"])
    status = validate_bg_schedule(cases, manifest, execution_plan=plan, cost_limit_cny=estimate_limit_cny)
    expected_snapshot_hash = (plan.get("data_snapshot") or {}).get("coverage_manifest_sha256")
    actual_snapshot_hash = _snapshot_hash(extras["snapshot"])
    if not expected_snapshot_hash or actual_snapshot_hash != expected_snapshot_hash:
        raise RuntimeError("Olist source coverage manifest hash mismatch.")
    raw_hashes = extras["snapshot"].get("raw_csv_sha256") or {}
    if len(raw_hashes) != 9 or not all(extras["snapshot"].get("truth_source_hashes_match", {}).values()):
        raise RuntimeError("Olist source manifest is incomplete or candidate truth source hashes differ.")
    extras["snapshot_hash"] = actual_snapshot_hash
    extras["expected_csv_hashes"] = {
        "olist_orders_dataset.csv": raw_hashes["orders"],
        "olist_order_items_dataset.csv": raw_hashes["order_items"],
        "olist_order_payments_dataset.csv": raw_hashes["order_payments"],
        "olist_customers_dataset.csv": raw_hashes["customers"],
        "olist_products_dataset.csv": raw_hashes["products"],
        "olist_sellers_dataset.csv": raw_hashes["sellers"],
        "olist_order_reviews_dataset.csv": raw_hashes["order_reviews"],
        "olist_geolocation_dataset.csv": raw_hashes["geolocation"],
        "product_category_name_translation.csv": raw_hashes["product_category_name_translation"],
    }
    status["olist_source_manifest_sha256"] = actual_snapshot_hash
    status["candidate_agent_runs"] = 0
    status["provider_sends"] = 0
    return cases_document, cases, manifest, plan, {**extras, "status": status}


def _summary_for_selection(manifest: dict, ledger_path: Path, cost_path: Path, *, max_units: int) -> dict:
    ledger = RunLedger(ledger_path)
    cost_ledger = EstimatedCostLedger(
        CostPolicy(cumulative_estimate_limit_cny=MAX_ESTIMATE_LIMIT_CNY), cost_path,
    )
    selected = select_never_started(manifest["entries"], ledger, max_units=max_units)
    budget = validate_execute_budget(
        selected_count=len(selected), policy=cost_ledger.policy, cost_ledger=cost_ledger,
        estimate_limit_cny=MAX_ESTIMATE_LIMIT_CNY,
        provider_authorization_cny=MAX_ESTIMATE_LIMIT_CNY,
    )
    return {
        "selected_units": [
            {"schedule_index": row["schedule_index"], "case_id": row["case_id"],
             "arm": row["arm"], "replicate": row["replicate"], "run_id": row["run_id"]}
            for row in selected
        ],
        "budget": budget,
        "existing_lifecycle": ledger.latest(),
        "provider_sends": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--estimate-limit-cny", type=float)
    parser.add_argument("--provider-authorization-cny", type=float)
    args = parser.parse_args()

    limit = args.estimate_limit_cny if args.estimate_limit_cny is not None else MAX_ESTIMATE_LIMIT_CNY
    cases_document, cases, manifest, plan, extras = prepare(estimate_limit_cny=float(limit))
    if args.validate_only:
        print(json.dumps(extras["status"], ensure_ascii=False, indent=2))
        return 0

    ledger_path = RESULTS / "events.jsonl"
    cost_path = RESULTS / "cost_reservations.jsonl"
    if args.dry_run:
        max_units = args.max_units if args.max_units is not None else len(manifest["entries"])
        output = {
            **extras["status"],
            **_summary_for_selection(manifest, ledger_path, cost_path, max_units=max_units),
            "mode": "dry_run_only",
            "database_started": False,
            "provider_sends": 0,
            "phase_b_allowed": not bool(plan.get("execution_blocked")),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
        return 0

    if args.max_units is None or args.estimate_limit_cny is None or args.provider_authorization_cny is None:
        parser.error("--execute requires --max-units, --estimate-limit-cny, and --provider-authorization-cny.")
    if plan.get("execution_blocked"):
        raise RuntimeError("Phase A remains blocked; no database, ledger reservation, or Provider request was started.")
    admin_url = os.environ.get("ISOLATED_TEST_POSTGRES_ADMIN_URL", "")
    if not admin_url:
        raise RuntimeError("Set ISOLATED_TEST_POSTGRES_ADMIN_URL to the isolated loopback maintenance database URL.")
    outputs = run_bg_batch(
        root=ROOT, cases=cases, manifest=manifest, execution_plan=plan,
        max_units=args.max_units, estimate_limit_cny=args.estimate_limit_cny,
        provider_authorization_cny=args.provider_authorization_cny,
        admin_url_text=admin_url, ledger_path=ledger_path, cost_ledger_path=cost_path,
        trace_dir=RESULTS / "traces", expected_csv_hashes=extras["expected_csv_hashes"],
    )
    print(json.dumps({"started_outputs": len(outputs), "results": outputs}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
