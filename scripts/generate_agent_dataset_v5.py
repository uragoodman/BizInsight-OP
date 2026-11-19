"""Build the expanded Agent evaluation dataset without altering v4 artifacts."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.scenario_library_v5 import scenario_dicts


SOURCE = ROOT / "evaluation" / "datasets" / "golden.json"
OUTPUT = ROOT / "evaluation" / "datasets" / "golden_v5_agent_368.json"


def main() -> int:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    retained = list(data["cases"])
    retained.extend(scenario_dicts())
    data["cases"] = retained
    data["dataset_version"] = "v5-agent-tiered-368"
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = Counter(case.get("category") for case in data["cases"])
    print(json.dumps({
        "dataset_version": data["dataset_version"],
        "total_cases": len(data["cases"]),
        "by_category": dict(counts),
        "added_rootcause_cases": len(scenario_dicts()),
        "output": str(OUTPUT),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
