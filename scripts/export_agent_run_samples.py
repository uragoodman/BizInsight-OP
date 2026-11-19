"""Export representative, redacted AgentRun records from persisted traces."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    trace_dir = ROOT / "data" / "traces"
    by_domain: dict[str, list[dict[str, object]]] = {}
    scanned = 0
    for path in trace_dir.glob("*.json"):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        scanned += 1
        agent_run = (envelope.get("state_snapshot") or {}).get("agent_run")
        if not isinstance(agent_run, dict) or not agent_run.get("run_id"):
            continue
        domain = str(agent_run.get("domain") or "unknown")
        # Keep the protocol fields needed for an interview/audit sample and
        # omit the larger plan payload, which can contain business context.
        sample = {key: agent_run.get(key) for key in ("run_id", "domain", "decision_driver", "status", "rounds", "tool_calls", "input_tokens", "output_tokens", "estimated_cost", "termination", "trace")}
        sample["source_trace_file"] = path.name
        by_domain.setdefault(domain, []).append(sample)
    samples = [items[0] for _, items in sorted(by_domain.items())]
    payload = {"schema_version": "1.0", "captured_at": datetime.now(timezone.utc).isoformat(), "traces_scanned": scanned, "agent_run_count": sum(len(items) for items in by_domain.values()), "domains": dict(Counter(sample["domain"] for items in by_domain.values() for sample in items)), "samples": samples, "limitations": ["Samples are representative persisted traces, not a production traffic distribution.", "Business plan payloads are intentionally omitted from this export."]}
    output = ROOT / "data" / "quality" / "agent-run-samples-latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "traces_scanned": scanned, "agent_run_count": payload["agent_run_count"], "domains": payload["domains"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
