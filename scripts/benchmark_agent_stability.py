"""Repeat the Phase 5 Agent evaluation and persist run-to-run variance."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_agent_vs_workflow import run_agent_arm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--provider", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    for run_number in range(max(1, args.runs)):
        result = run_agent_arm(provider_enabled=args.provider)
        results.append({"run": run_number + 1, "passed_gate": result.get("passed_gate"), "metrics": result.get("metrics", {}), "details_count": len(result.get("details", [])), "fallback_count": len(result.get("agent_fallback", {}).get("details", []))})
    metric_keys = ("completion_rate", "evidence_coverage", "correct_abstain_rate", "invalid_call_count", "duplicate_call_count", "trajectory_branch_rate", "p95_latency_ms", "total_tokens", "total_cost")
    aggregate = {}
    for key in metric_keys:
        values = [float(item["metrics"][key]) for item in results if key in item["metrics"]]
        aggregate[key] = {"mean": round(statistics.mean(values), 4), "min": round(min(values), 4), "max": round(max(values), 4), "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0} if values else None
    output = args.output or ROOT / "data" / "evaluations" / ("agent-stability-provider.json" if args.provider else "agent-stability-offline.json")
    payload = {"schema_version": "1.0", "captured_at": datetime.now(timezone.utc).isoformat(), "evaluation_mode": "provider_enabled" if args.provider else "offline_scripted_protocol", "runs": results, "aggregate": aggregate, "limitations": ["Provider runs measure the configured evaluation subset; filtered workflow-fallback cases are reported separately and not counted as Agent quality."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": len(results), "aggregate": aggregate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
