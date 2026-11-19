"""Regenerate the generated root-cause golden cases from the scenario library.

Single source of truth: ``evaluation/scenario_library.py`` defines the
patterns; this script syncs them into ``evaluation/datasets/golden.json``
(replacing previously generated cases, keeping hand-written ones).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.scenario_library import scenario_dicts  # noqa: E402


def main() -> int:
    path = ROOT / "evaluation" / "datasets" / "golden.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    kept = [
        case for case in data["cases"]
        if not str(case.get("id", "")).startswith(("gen-", "core-core_"))
    ]
    generated = scenario_dicts()
    data["cases"] = kept + generated
    data["dataset_version"] = "v4-rootcause-100"
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter
    patterns = Counter(case.get("pattern") for case in generated)
    print(json.dumps({
        "total_cases": len(data["cases"]),
        "generated": len(generated),
        "by_pattern": dict(patterns),
        "output": str(path),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
