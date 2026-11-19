"""Deterministic Phase 12 contract checks used by CI without a live model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.llm.routing import profile_for


ROOT = Path(__file__).resolve().parents[1]


def run_phase12_acceptance() -> list[dict[str, Any]]:
    """Verify route policy and keep reviewed failure scenarios visible in CI.

    Behaviour involving a real provider is deliberately not asserted here: it
    belongs to the opt-in live regression run, where its usage and spend are
    independently budgeted.
    """
    cases = json.loads((ROOT / "evaluation" / "datasets" / "phase12_failures.json").read_text(encoding="utf-8"))["cases"]
    checks = [
        {"id": "phase12-fast-routing", "passed": profile_for("classify") == "fast"},
        {"id": "phase12-standard-routing", "passed": profile_for("chat") == "standard"},
        {"id": "phase12-quality-routing", "passed": profile_for("report") == "quality"},
    ]
    checks.extend({"id": item["id"], "passed": bool(item["expected"]), "category": item["category"], "expected": item["expected"]} for item in cases)
    return checks
