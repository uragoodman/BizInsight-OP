"""Run two fixed-input development cases through isolated reviewer adapters.

This smoke test checks integration and usage records, not accuracy or cost
advantage. It intentionally refuses retries and reuse of an output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import get_settings
from app.llm.routing import model_for
from evaluation.claim_verification.fixed_review import review


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append(path: Path, entry: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("Output directory already exists; refuse overwrite/retry")
    cases = [json.loads(line) for line in args.inputs.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) != 30 or len({x["case_id"] for x in cases}) != 30:
        raise ValueError("Expected exactly 30 unique blind inputs")
    cases = sorted(cases, key=lambda x: x["case_id"])[:2]
    settings = get_settings()
    model = model_for("extract")[0]
    host = urlparse(settings.deepseek_base_url or "").hostname
    if args.execute and (not settings.deepseek_api_key or host != "dashscope.aliyuncs.com" or model != "qwen3.8-flash"):
        raise RuntimeError("Expected development provider and frozen extract model unavailable")
    paths = [Path(__file__), Path("evaluation/claim_verification/fixed_review.py"), Path("evaluation/claim_verification/deterministic_probe.py"), Path("evaluation/claim_verification/claim_routing.py"), Path("evaluation/claim_verification/annotation_schema.py"), Path("evaluation/claim_verification/operational_context.py"), Path("evaluation/claim_verification/registry_evidence.py"), Path("app/agents/critic_agent.py"), Path("app/llm/critic_prompt.py"), Path("app/llm/client.py")]
    manifest = {
        "purpose": "two-case development integration preflight; no accuracy/cost conclusion",
        "input_sha256": _sha(args.inputs), "code_sha256": {str(p): _sha(p) for p in paths},
        "provider_host": host, "extract_model": model,
        "case_ids": [x["case_id"] for x in cases], "arms": ["off", "full", "routed"],
        "planned_units": 6, "planned_logical_calls_max": 8,
        "caveat": "Full is isolated review functions using the snapshot task type and normalized shared evidence, not the original path-aware workflow.",
        "mode": "execute" if args.execute else "dry_run",
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "units": 6, "provider_sends": 0}))
        return
    for case in cases:
        for arm in manifest["arms"]:
            event = {"case_id": case["case_id"], "arm": arm}
            _append(args.output_dir / "ledger.jsonl", {**event, "status": "started", "timestamp": time.time()})
            try:
                result = review(case, arm)
            except Exception as exc:
                _append(args.output_dir / "ledger.jsonl", {**event, "status": "failed", "error_type": type(exc).__name__, "usage": "unknown"})
                continue
            _append(args.output_dir / "results.jsonl", result)
            _append(args.output_dir / "ledger.jsonl", {**event, "status": "completed", "verdict": result["verdict"]})
    print(json.dumps({"mode": "execute", "planned_units": 6, "ledger": str(args.output_dir / "ledger.jsonl")}))


if __name__ == "__main__":
    main()
