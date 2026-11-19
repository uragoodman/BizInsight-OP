"""Run the full 30-case fixed-answer review comparison (off / full / routed).

This is a development comparison, not an independent benchmark.  It runs the
*reviewer adapters* over frozen answers, so it measures review cost and
detection on identical inputs -- not end-to-end workflow quality.

Cost discipline:
  * every unit is reserved before it starts and the reservation is never
    released on unknown usage;
  * a hard cumulative cap is enforced by `CostPolicy`, so a bad estimate stops
    the run instead of overspending;
  * `--execute` is the only path that can send a provider request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.core.config import get_settings
from app.llm.routing import model_for
from evaluation.claim_verification.fixed_review import review
from evaluation.provider_candidate_batch import CostPolicy, EstimatedCostLedger

ARMS = ("off", "full", "routed")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _usage_for(record: dict[str, Any]) -> dict[str, Any]:
    """Normalise the adapter's usage record for the cost ledger."""

    summary = record.get("llm_token_summary") or {}
    if not summary.get("logical_call_count"):
        return {"status": "complete", "total_input_tokens": 0, "total_output_tokens": 0}
    if summary.get("request_status") != "requested":
        return {"status": "unknown"}
    return {
        "status": "complete" if summary.get("known_input_tokens_subtotal") is not None
        and summary.get("known_output_tokens_subtotal") is not None else "unknown",
        "total_input_tokens": int(summary.get("known_input_tokens_subtotal") or 0),
        "total_output_tokens": int(summary.get("known_output_tokens_subtotal") or 0),
    }


def _estimate(cases: list[dict[str, Any]], annotations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Estimate from the two observed smoke runs plus a conservative margin."""

    observed = {
        "full": {"input": 3_487, "output": 1_731, "calls": 3},
        "routed": {"input": 755, "output": 838, "calls": 1},
        "off": {"input": 0, "output": 0, "calls": 0},
    }
    per_arm = {}
    for arm, base in observed.items():
        tokens = {"input": 0, "output": 0, "calls": 0}
        for case in cases:
            # Answers vary 200-1800 chars; scale linearly from the smoke case.
            scale = max(0.5, min(3.0, len(case["answer"]) / 900))
            tokens["input"] += int(base["input"] * scale)
            tokens["output"] += int(base["output"] * scale)
            tokens["calls"] += base["calls"]
        per_arm[arm] = tokens
    total = {k: sum(v[k] for v in per_arm.values()) for k in ("input", "output", "calls")}
    cny = (total["input"] * 1.094 + total["output"] * 3.427) / 1_000_000
    return {
        "source": "linear scaling of fixed_semantic_smoke_r1 observed tokens",
        "per_arm": per_arm,
        "total": total,
        "estimated_cny": round(cny, 4),
        "units": len(cases) * len(ARMS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-cny", type=float, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise RuntimeError("Output directory already exists; refuse overwrite/retry")
    cases = [json.loads(x) for x in args.inputs.read_text(encoding="utf-8").splitlines() if x.strip()]
    annotation_rows = [json.loads(x) for x in args.annotations.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(cases) != 30 or len({c["case_id"] for c in cases}) != 30:
        raise ValueError("Expected exactly 30 unique frozen inputs")
    annotations = {row["case_id"]: row for row in annotation_rows}
    if set(annotations) != {c["case_id"] for c in cases}:
        raise ValueError("annotations must cover exactly the frozen inputs")

    settings = get_settings()
    host = urlparse(settings.deepseek_base_url or "").hostname
    model = model_for("extract")[0]
    if args.execute and (not settings.deepseek_api_key or host != "dashscope.aliyuncs.com" or model != "qwen3.8-flash"):
        raise RuntimeError("Configured provider/model does not match the frozen development policy")

    policy = CostPolicy(
        max_sdk_calls_per_workflow=3,      # Full issues three logical calls
        # The default 200k-token cap reserves 0.70 CNY per unit, which makes a
        # 90-unit run exceed a 40 CNY authorization while the measured usage is
        # ~5k tokens per run.  20k keeps ~5.7x headroom over the observed
        # largest input and makes the reservation a usable guard rather than a
        # number that always trips.
        input_token_estimate_cap_per_call=20_000,
        cumulative_estimate_limit_cny=args.limit_cny,
    )
    estimate = _estimate(cases, annotations)
    worst_case_cny = policy.max_per_run_cny() * len(cases) * len(ARMS)

    args.output_dir.mkdir(parents=True)
    manifest = {
        "purpose": "30-case fixed-answer development comparison; not an independent benchmark",
        "arms": list(ARMS),
        "cases": len(cases),
        "planned_units": len(cases) * len(ARMS),
        "input_sha256": _sha(args.inputs),
        "annotations_sha256": _sha(args.annotations),
        "provider_host": host,
        "extract_model": model,
        "usage_estimate": estimate,
        "worst_case_reservation_cny": round(worst_case_cny, 4),
        "authorized_limit_cny": args.limit_cny,
        "route_annotations_are_verified": all(
            row.get("human_verified") is True and row.get("oracle_claim_labels") is True
            for row in annotations.values()
        ),
        "caveat": (
            "Full is the isolated review adapter, not the path-aware production workflow. "
            "Routed currently falls back to whole-answer review because no row has verified "
            "claim labels; the Full/Routed delta therefore measures call merging, not Claim routing."
        ),
        "mode": "execute" if args.execute else "dry_run",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "mode": manifest["mode"],
        "planned_units": manifest["planned_units"],
        "usage_estimate_cny": estimate["estimated_cny"],
        "worst_case_reservation_cny": manifest["worst_case_reservation_cny"],
        "authorized_limit_cny": args.limit_cny,
    }, ensure_ascii=False))
    if not args.execute:
        return

    ledger = EstimatedCostLedger(policy, args.output_dir / "cost_reservations.jsonl")
    events = args.output_dir / "ledger.jsonl"
    results = args.output_dir / "results.jsonl"
    for case in sorted(cases, key=lambda c: c["case_id"]):
        annotation = annotations[case["case_id"]]
        case_input = dict(case)
        case_input["route_annotations"] = {
            "oracle_claim_labels": bool(annotation.get("oracle_claim_labels")),
            "coverage_verified": bool(annotation.get("human_verified")),
            "claims": annotation["claims"],
            "requirement_coverage": annotation["requirement_coverage"],
        }
        for arm in ARMS:
            run_id = f"{case['case_id']}:{arm}"
            started = time.perf_counter()
            try:
                reserved = ledger.reserve(run_id)
            except Exception as exc:  # cumulative cap reached
                _append(events, {"run_id": run_id, "status": "blocked_by_cost_cap",
                                 "error_type": type(exc).__name__})
                print(json.dumps({"stopped": "cost_cap", "run_id": run_id}, ensure_ascii=False))
                return
            _append(events, {"run_id": run_id, "status": "started", "reserved_cny": reserved})
            try:
                record = review(case_input, arm)
            except Exception as exc:
                _append(events, {"run_id": run_id, "status": "failed", "error_type": type(exc).__name__})
                continue
            usage = _usage_for(record)
            accounting = ledger.record_usage(run_id, usage)
            record["run_id"] = run_id
            record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            record["cost_accounting"] = accounting
            _append(results, record)
            _append(events, {"run_id": run_id, "status": "completed", "verdict": record["verdict"],
                             "usage_status": usage["status"], "actual_cny": accounting.get("actual_cny")})

    print(json.dumps({"completed_units": sum(1 for _ in results.open(encoding="utf-8")),
                      "reserved_cny": ledger.reserved_cny}, ensure_ascii=False))


if __name__ == "__main__":
    main()
