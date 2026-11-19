"""Validate the prepared six-case B/G development manifest without execution."""
from __future__ import annotations

import argparse
import ipaddress
import json
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "testing" / "resume_bg_ab_dev_20261016"
sys.path.insert(0, str(ROOT))

from evaluation.resume_bg_protocol import ARMS, build_manifest, validate_cases  # noqa: E402


def block_external_network() -> None:
    """Fail closed if this preparation command unexpectedly attempts egress."""
    def deny(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = str(host).casefold() == "localhost"
        if not loopback:
            raise RuntimeError("external network is blocked during B/G preparation")
        raise RuntimeError("preparation validation must not open network sockets")

    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = lambda address, *args, **kwargs: deny(None, address)


def validate_saved_artifacts() -> dict:
    block_external_network()
    cases_doc = json.loads((ARTIFACTS / "cases_6.json").read_text(encoding="utf-8"))
    manifest = json.loads((ARTIFACTS / "run_manifest_12.json").read_text(encoding="utf-8"))
    plan = json.loads((ARTIFACTS / "execution_plan.json").read_text(encoding="utf-8"))
    cases = cases_doc["cases"]
    validate_cases(cases)
    expected = build_manifest(cases, seed=manifest["seed"])
    if expected != manifest:
        raise ValueError("Saved manifest differs from deterministic schedule for the recorded seed.")
    if plan.get("provider_sends") != 0 or plan.get("phase") != "A_only_no_candidate_execution":
        raise ValueError("Preparation artifacts must remain Provider-free and non-executing.")
    if any(row.get("status") != "not_started" for row in manifest["entries"]):
        raise ValueError("Preparation manifest contains a started unit.")
    if set(row.get("arm") for row in manifest["entries"]) != set(ARMS):
        raise ValueError("Saved manifest contains an unexpected experiment arm.")
    return {
        "experiment_id": manifest["experiment_id"],
        "planned_units": len(manifest["entries"]),
        "unique_run_ids": len({row["run_id"] for row in manifest["entries"]}),
        "arms": {arm: sum(row["arm"] == arm for row in manifest["entries"]) for arm in ARMS},
        "provider_sends": 0,
        "execution_blocked": bool(plan.get("execution_blocked")),
        "blocked_cases": plan.get("blocked_case_ids") or [case["case_id"] for case in cases if case.get("readiness") == "blocked"],
        "cost_estimate_cny": plan["cost_estimate"]["estimated_cny"],
        "provider_batch_authorized": bool(plan.get("provider_batch_authorized")),
        "provider_authorization_status": plan.get("provider_authorization_status"),
        "phase_b_allowed": not bool(plan.get("execution_blocked")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", required=True)
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_saved_artifacts(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
