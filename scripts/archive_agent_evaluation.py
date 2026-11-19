"""Archive an existing Agent evaluation artifact into the unified evidence store.

This is useful for historical Provider results whose original process has
already finished.  It does not rerun the model; it registers the JSON that is
already on disk and, by default, does not copy possibly unrelated live Trace
files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_agent_vs_workflow import _persist_unified_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--copy-traces", action="store_true")
    args = parser.parse_args()
    aggregate = args.aggregate if args.aggregate.is_absolute() else ROOT / args.aggregate
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    payload.setdefault("run_id", args.run_id or f"agent_eval_archive_{uuid4().hex}")
    bundle = _persist_unified_run(payload, ROOT / "evaluation" / "datasets" / "golden.json", aggregate, copy_traces=args.copy_traces)
    print(json.dumps({"run_id": payload["run_id"], "bundle": str(bundle)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
