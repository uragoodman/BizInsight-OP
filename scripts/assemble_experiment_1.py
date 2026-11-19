"""Assemble experiment 1 from the two independently persisted Provider arms."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


agent = load(ROOT / "data/evaluations/agent-failed-cases-provider.json")
single = load(ROOT / "data/evaluations/experiment-1-single-call.json")
agent_rows = agent.get("details", []) + agent.get("longtail", {}).get("details", [])
single_rows = single.get("details", [])

agent_by_id = {row["id"]: row for row in agent_rows}
single_by_id = {row["id"]: row for row in single_rows}
ids = [row["id"] for row in single_rows if row["id"] in agent_by_id]

def arm(rows: dict, key: str) -> dict:
    selected = [rows[item] for item in ids]
    passed = sum(bool(row.get(key, row.get("passed"))) for row in selected)
    return {
        "cases": len(selected),
        "passed": passed,
        "pass_at_1": round(passed / len(selected), 4) if selected else 0.0,
    }


agent_pass = sum(bool(agent_by_id[item].get("passed")) for item in ids)
single_pass = sum(bool(single_by_id[item].get("single_call", {}).get("passed")) for item in ids)
agent_tokens = int(agent.get("metrics", {}).get("total_tokens", 0)) + int(agent.get("longtail", {}).get("metrics", {}).get("total_tokens", 0))
single_tokens = int(single.get("measurement", {}).get("total_tokens", 0))
payload = {
    "schema_version": "1.0",
    "experiment": "governed_agent_vs_single_call_llm",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "provider_enabled": True,
    "dataset": "golden.json / v4-rootcause-100, matched 12 cases",
    "arms": {
        "governed_agent": {
            "cases": len(ids), "passed": agent_pass,
            "pass_at_1": round(agent_pass / len(ids), 4) if ids else 0.0,
            "total_tokens": agent_tokens,
            "avg_tokens_per_case": round(agent_tokens / len(ids), 3) if ids else 0,
            "total_cost": 0.0,
            "source": "agent-failed-cases-provider.json (details + longtail.details)",
        },
        "single_call": {
            "cases": len(ids), "passed": single_pass,
            "pass_at_1": round(single_pass / len(ids), 4) if ids else 0.0,
            "total_tokens": single_tokens,
            "avg_tokens_per_case": round(single_tokens / len(ids), 3) if ids else 0,
            "total_cost": 0.0,
            "source": "experiment-1-single-call.json",
        },
    },
    "case_ids": ids,
    "limitations": [
        "Both arms use the real configured Provider but deterministic golden evidence, not live PostgreSQL business data.",
        "Cost is zero because no valid token price is configured.",
        "The two arms were run at different times; Provider latency variance is not a controlled variable.",
    ],
}
output = ROOT / "data/evaluations/experiment-1-cost.json"
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
