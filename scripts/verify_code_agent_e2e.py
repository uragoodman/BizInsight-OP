"""Run the opt-in, real OpenCode + sandbox acceptance check.

This script intentionally does not start Docker services or read provider
credentials.  It talks only to a running API with an already-issued bearer
token, so it can prove the production-shaped boundary without putting secrets
in source code or CI logs.

Usage (after the code-agent Compose profile is healthy)::

    $env:CODE_AGENT_E2E_TOKEN = '<login access token>'
    python -m scripts.verify_code_agent_e2e --api-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time

import httpx


SAMPLE_CSV = b"month,gmv,orders\n2024-01,1200,12\n2024-02,1500,15\n2024-03,1350,13\n"
TERMINAL = {"completed", "failed", "cancelled", "expired", "superseded"}


def _request(client: httpx.Client, method: str, path: str, **kwargs) -> dict:
    response = client.request(method, path, **kwargs)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text[:500]
        raise RuntimeError(f"{method} {path} returned {response.status_code}: {detail}") from exc
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--csv", help="CSV file to upload; defaults to the tiny smoke fixture")
    args = parser.parse_args()
    token = os.getenv("CODE_AGENT_E2E_TOKEN", "").strip()
    if not token:
        print("CODE_AGENT_E2E_TOKEN is required; no request was sent.", file=sys.stderr)
        return 2

    base_url = args.api_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"}
    if args.csv:
        csv_path = os.path.abspath(args.csv)
        with open(csv_path, "rb") as handle:
            csv_payload = handle.read()
        csv_name = os.path.basename(csv_path)
    else:
        csv_payload = SAMPLE_CSV
        csv_name = "golden-sales.csv"
    with httpx.Client(base_url=base_url, headers=headers, timeout=30.0) as client:
        _request(client, "GET", "/health/live")
        uploaded = _request(
            client,
            "POST",
            "/code-analysis/files",
            files={"file": (csv_name, io.BytesIO(csv_payload), "text/csv")},
        )
        if uploaded.get("file_type") != "csv" or not uploaded.get("data_contract"):
            raise RuntimeError("Upload did not return a validated CSV data contract")
        planned = _request(
            client,
            "POST",
            "/code-analysis/jobs",
            json={
                "file_id": uploaded["file_id"],
                "question": "Summarize GMV and order changes by month; state limitations.",
            },
        )
        job = _request(
            client,
            "POST",
            f"/code-analysis/jobs/{planned['job_id']}/confirm",
            json={"plan_hash": planned["plan_hash"]},
        )
        job_id = job["job_id"]
        deadline = time.monotonic() + max(10, args.timeout_seconds)
        while job.get("status") not in TERMINAL and time.monotonic() < deadline:
            time.sleep(2)
            job = _request(client, "GET", f"/code-analysis/jobs/{job_id}")
        if job.get("status") != "completed":
            raise RuntimeError(
                f"Job {job_id} ended as {job.get('status')}; error={job.get('error_code')}"
            )
        audit = job.get("latest_audit") or {}
        if audit.get("provider") != "opencode":
            raise RuntimeError(
                "Acceptance requires the real OpenCode provider; got "
                f"{audit.get('provider')!r}."
            )
        if not audit.get("candidate_code_hash") or not audit.get("runner_image_digest"):
            raise RuntimeError("Completed job is missing candidate or runner audit fingerprints")
        result_artifact = next(
            (item for item in job.get("artifacts", []) if item.get("kind") == "result"),
            None,
        )
        if not result_artifact:
            raise RuntimeError("Completed job is missing its result artifact")
        result = _request(
            client,
            "GET",
            f"/code-analysis/jobs/{job_id}/artifacts/{result_artifact['artifact_id']}",
        )
        if not isinstance(result.get("conclusion"), str) or not result["conclusion"].strip():
            raise RuntimeError("Result artifact has no natural-language conclusion")
    print(
        "PASS real-code-agent-e2e "
        f"job={job_id} provider={audit['provider']} image={audit['runner_image_digest'][:16]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
