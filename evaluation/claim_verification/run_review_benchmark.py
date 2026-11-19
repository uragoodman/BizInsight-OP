"""Run off / full (optionally routed) over a frozen corpus.

Same adapter, same cost discipline as the 30-case run.  `--corpus` is either
the clean corpus or the hard-variant file; both expose the fields the adapter
expects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from uuid import uuid4
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.core.config import get_settings
from app.llm.routing import model_for
from app.llm.call_config import LLMCallConfig
from evaluation.claim_verification.fixed_review import review
from evaluation.provider_candidate_batch import CostPolicy, EstimatedCostLedger


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _usage(record: dict[str, Any]) -> dict[str, Any]:
    summary = record.get("llm_token_summary") or {}
    if not summary.get("logical_call_count"):
        return {"status": "complete", "total_input_tokens": 0, "total_output_tokens": 0}
    if (summary.get("request_status") != "requested"
            or summary.get("usage_complete") is not True
            or summary.get("unknown_usage_attempt_count", 0)
            or summary.get("known_input_tokens_subtotal") is None
            or summary.get("known_output_tokens_subtotal") is None):
        return {"status": "unknown"}
    return {
        "status": "complete",
        "total_input_tokens": int(summary.get("known_input_tokens_subtotal") or 0),
        "total_output_tokens": int(summary.get("known_output_tokens_subtotal") or 0),
    }


def _assert_no_unresolved_prior(prior_output_dir: Path) -> None:
    """Do not silently resend a unit that may already have called a provider."""
    prior_ledger = prior_output_dir / "ledger.jsonl"
    if not prior_ledger.exists():
        return
    seen_started: set[str] = set()
    for line in prior_ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rid = item.get("run_id")
        status = item.get("status") or item.get("event")
        if rid and status == "started":
            seen_started.add(rid)
    prior_results = prior_output_dir / "results.jsonl"
    if prior_results.exists():
        durable_results = {json.loads(line)["run_id"] for line in prior_results.read_text(encoding="utf-8").splitlines() if line.strip()}
    else:
        durable_results = set()
    # A ledger terminal event is not a substitute for a durable result. Failed
    # or interrupted units stay consumed and are never silently sent again.
    unresolved = seen_started - durable_results
    if unresolved:
        raise RuntimeError(
            "Prior output contains started units without terminal results; refusing automatic retry: "
            + ", ".join(sorted(unresolved))
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-cny", type=float, required=True)
    parser.add_argument("--arms", default="off,full")
    parser.add_argument("--prior-output-dir", type=Path, help="Reuse completed units from a matching run; refuse any started unit without a durable terminal result.")
    parser.add_argument("--scoring-freeze", type=Path, help="Record the pre-run scoring rules hash.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("Output directory already exists; refuse overwrite/retry")

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    corpus = [json.loads(x) for x in args.corpus.read_text(encoding="utf-8").splitlines() if x.strip()]
    prior_completed: set[str] = set()
    batch_id = uuid4().hex
    if args.prior_output_dir:
        prior_manifest = json.loads((args.prior_output_dir / "manifest.json").read_text(encoding="utf-8"))
        if prior_manifest.get("corpus_sha256") != _sha(args.corpus) or tuple(prior_manifest.get("arms", [])) != arms:
            raise ValueError("Prior run corpus or arms do not match")
        batch_id = str(prior_manifest.get("batch_id") or "")
        if not batch_id:
            raise ValueError("Prior run has no batch ID; refusing ambiguous resume")
        prior_results = args.prior_output_dir / "results.jsonl"
        if prior_results.exists():
            prior_completed = {json.loads(line)["run_id"] for line in prior_results.read_text(encoding="utf-8").splitlines() if line.strip()}
        _assert_no_unresolved_prior(args.prior_output_dir)
    settings = get_settings()
    host = urlparse(settings.deepseek_base_url or "").hostname
    model = model_for("extract")[0]
    if args.execute and (not settings.deepseek_api_key or host != "dashscope.aliyuncs.com" or model != "qwen3.8-flash"):
        raise RuntimeError("Configured provider/model does not match the frozen development policy")

    # Each of the three Full review stages may retry a TLS/TCP connection
    # establishment once. Reserve worst-case sends before execution.
    policy = CostPolicy(max_sdk_calls_per_workflow=6, input_token_estimate_cap_per_call=20_000,
                        cumulative_estimate_limit_cny=args.limit_cny)
    llm_call_config = LLMCallConfig.provider_ab_experiment(
        settings, provider_id="primary", max_completion_tokens=4096, timeout_seconds=300,
        connect_retry_attempts=1, connect_timeout_seconds=20,
    )
    args.output_dir.mkdir(parents=True)
    manifest = {
        "corpus_sha256": _sha(args.corpus),
        "cases": len(corpus),
        "arms": list(arms),
        "planned_units": len(corpus) * len(arms),
        "provider_host": host,
        "extract_model": model,
        "worst_case_reservation_cny": round(policy.max_per_run_cny() * len(corpus) * len(arms), 4),
        "authorized_limit_cny": args.limit_cny,
        "caveat": "Fixed-answer reviewer comparison on a public development corpus; not an independent benchmark.",
        "mode": "execute" if args.execute else "dry_run",
        "batch_id": batch_id,
        "llm_call_config": llm_call_config.to_state(),
        "transient_connection_policy": {
            "max_retries_per_logical_call": 1,
            "retry_only": ["httpx.ConnectError", "httpx.ConnectTimeout", "TLS handshake timeout"],
            "connect_timeout_seconds": 20,
            "sdk_retries": 0,
            "other_timeouts_retried": False,
        },
        "call_measurement_semantics": "application-level SDK invocations; not confirmed wire-level HTTP requests",
        "prior_output_dir": str(args.prior_output_dir) if args.prior_output_dir else None,
        "prior_completed_units": len(prior_completed),
        "planned_new_units": len(corpus) * len(arms) - len(prior_completed),
        "scoring_freeze_sha256": _sha(args.scoring_freeze) if args.scoring_freeze else None,
        "code_sha256": {str(p): _sha(p) for p in (
            Path(__file__), Path("evaluation/claim_verification/fixed_review.py"),
            Path("evaluation/claim_verification/deterministic_probe.py"),
            Path("evaluation/claim_verification/claim_routing.py"),
            Path("app/agents/critic_agent.py"), Path("app/llm/client.py"),
        )},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode": manifest["mode"], "units": manifest["planned_units"],
                      "worst_case_cny": manifest["worst_case_reservation_cny"]}, ensure_ascii=False))
    if not args.execute:
        return

    ledger = EstimatedCostLedger(policy, args.output_dir / "cost_reservations.jsonl")
    events = args.output_dir / "ledger.jsonl"
    call_events = args.output_dir / "call_events.jsonl"
    results = args.output_dir / "results.jsonl"
    for case in sorted(corpus, key=lambda c: c["case_id"]):
        for arm in arms:
            run_id = f"{batch_id}:{case['case_id']}:{arm}"
            if run_id in prior_completed:
                continue
            try:
                reserved = ledger.reserve(run_id)
            except Exception as exc:
                _append(events, {"run_id": run_id, "status": "blocked_by_cost_cap", "error_type": type(exc).__name__})
                print(json.dumps({"stopped": "cost_cap", "run_id": run_id}, ensure_ascii=False))
                return
            _append(events, {"run_id": run_id, "status": "started", "reserved_cny": reserved})
            print(json.dumps({"run_id": run_id, "event": "unit_started"}, ensure_ascii=False), flush=True)
            started = time.perf_counter()

            def persist_call_event(event: dict[str, Any]) -> None:
                _append(call_events, event)
                print(json.dumps({
                    "run_id": run_id, "event": event.get("event"),
                    "stage": event.get("stage"), "operation": event.get("operation"),
                    "call_id": event.get("call_id"), "attempt_id": event.get("attempt_id"),
                    "result": event.get("result"), "verdict": event.get("verdict"),
                }, ensure_ascii=False), flush=True)

            try:
                record = review(
                    case, arm, evaluation_run_id=run_id,
                    event_sink=persist_call_event,
                    llm_call_config=llm_call_config if args.execute else None,
                )
            except KeyboardInterrupt:
                persisted = [json.loads(line) for line in call_events.read_text(encoding="utf-8").splitlines()
                             if line.strip()] if call_events.exists() else []
                unit_events = [item for item in persisted if item.get("run_id") == run_id]
                started_attempts = {item.get("attempt_id") for item in unit_events
                                    if item.get("event") == "sdk_invocation_started" and item.get("attempt_id")}
                finished_attempts = {item.get("attempt_id") for item in unit_events
                                     if item.get("event") == "sdk_invocation_finished" and item.get("attempt_id")}
                unresolved_attempts = sorted(started_attempts - finished_attempts)
                _append(events, {"run_id": run_id, "status": "interrupted_result_unknown",
                                 "application_sdk_invocation_started_count": len(started_attempts),
                                 "application_sdk_invocation_unresolved_count": len(unresolved_attempts),
                                 "unresolved_attempt_ids": unresolved_attempts,
                                 "retry_forbidden": True})
                print(json.dumps({"run_id": run_id, "event": "interrupted_result_unknown",
                                  "unresolved_attempt_count": len(unresolved_attempts)}, ensure_ascii=False), flush=True)
                raise
            except Exception as exc:
                _append(events, {"run_id": run_id, "status": "failed", "error_type": type(exc).__name__})
                print(json.dumps({"run_id": run_id, "event": "unit_failed", "error_type": type(exc).__name__}, ensure_ascii=False), flush=True)
                continue
            usage = _usage(record)
            accounting = ledger.record_usage(run_id, usage)
            record.update({"run_id": run_id, "source_case_id": case.get("source_case_id", case["case_id"]),
                           "cohort": case.get("cohort"), "error_family": case.get("error_family"),
                           "expected": case.get("expected", "should_pass"),
                           "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                           "cost_accounting": accounting})
            _append(results, record)
            _append(events, {"run_id": run_id, "status": "completed", "verdict": record["verdict"],
                             "usage_status": usage["status"], "actual_cny": accounting.get("actual_cny")})
            print(json.dumps({"run_id": run_id, "event": "unit_completed", "verdict": record["verdict"],
                              "sdk_attempt_count": (record.get("llm_token_summary") or {}).get("sdk_attempt_count"),
                              "usage_status": usage["status"]}, ensure_ascii=False), flush=True)
    print(json.dumps({"completed_units": sum(1 for _ in results.open(encoding="utf-8")),
                      "reserved_cny": round(ledger.reserved_cny, 4)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
