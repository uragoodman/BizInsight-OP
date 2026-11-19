"""Verify Code Agent ownership and lease behavior with concurrent users.

This test deliberately uses the deterministic provider.  OpenCode is verified
separately by ``verify_code_agent_e2e``; this script isolates queue, ownership,
artifact, trace, and lease correctness from model latency and model variance.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path

import httpx
import psycopg


CSV = b"month,gmv,orders\n2024-01,1200,12\n2024-02,1500,15\n2024-03,1350,13\n"
TERMINAL = {"completed", "failed", "cancelled", "expired", "superseded"}


async def request(client: httpx.AsyncClient, method: str, path: str, **kwargs) -> dict:
    response = await client.request(method, path, **kwargs)
    if response.is_error:
        raise RuntimeError(f"{method} {path} -> {response.status_code}: {response.text[:400]}")
    return response.json()


async def create_and_run(api_url: str, token: str, index: int, timeout: float) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(base_url=api_url, headers=headers, timeout=30.0) as client:
        uploaded = await request(
            client,
            "POST",
            "/code-analysis/files",
            files={"file": (f"concurrency-{index}.csv", io.BytesIO(CSV), "text/csv")},
        )
        planned = await request(
            client,
            "POST",
            "/code-analysis/jobs",
            json={
                "file_id": uploaded["file_id"],
                "question": f"Concurrency isolation check for user {index}: summarize the uploaded data.",
            },
        )
        job = await request(
            client,
            "POST",
            f"/code-analysis/jobs/{planned['job_id']}/confirm",
            json={"plan_hash": planned["plan_hash"]},
        )
        deadline = time.monotonic() + timeout
        while job.get("status") not in TERMINAL and time.monotonic() < deadline:
            await asyncio.sleep(0.25)
            job = await request(client, "GET", f"/code-analysis/jobs/{planned['job_id']}")
        if job.get("status") not in TERMINAL:
            raise RuntimeError(f"job {planned['job_id']} did not reach a terminal state")
        return {
            "index": index,
            "user_id": job.get("user_id"),
            "job_id": planned["job_id"],
            "file_id": uploaded["file_id"],
            "status": job.get("status"),
            "artifact_ids": [item["artifact_id"] for item in job.get("artifacts", [])],
            "audit": job.get("latest_audit"),
        }


def db_checks(database_url: str, job_ids: list[str]) -> dict:
    placeholders = ",".join(["%s"] * len(job_ids))
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT job_id,user_id,status,attempts,lease_token,lease_expires_at
                    FROM governance.code_analysis_jobs WHERE job_id IN ({placeholders})""",
                job_ids,
            )
            jobs = [dict(zip(("job_id", "user_id", "status", "attempts", "lease_token", "lease_expires_at"), row)) for row in cur.fetchall()]
            cur.execute(
                f"""SELECT job_id,COUNT(*) FROM governance.code_analysis_audits
                    WHERE job_id IN ({placeholders}) GROUP BY job_id""",
                job_ids,
            )
            audits = {row[0]: row[1] for row in cur.fetchall()}
            cur.execute(
                f"""SELECT job_id,COUNT(DISTINCT trace_id),COUNT(*)
                    FROM governance.code_analysis_audits WHERE job_id IN ({placeholders})
                    GROUP BY job_id""",
                job_ids,
            )
            traces = {row[0]: {"unique": row[1], "rows": row[2]} for row in cur.fetchall()}
            cur.execute(
                f"""SELECT j.job_id,COUNT(a.artifact_id),COUNT(DISTINCT a.user_id)
                    FROM governance.code_analysis_jobs j
                    LEFT JOIN governance.code_analysis_artifacts a ON a.job_id=j.job_id
                    WHERE j.job_id IN ({placeholders}) GROUP BY j.job_id""",
                job_ids,
            )
            artifacts = {row[0]: {"count": row[1], "distinct_users": row[2]} for row in cur.fetchall()}
    return {"jobs": jobs, "audit_counts": audits, "traces": traces, "artifacts": artifacts}


async def async_main(args: argparse.Namespace) -> dict:
    admin_email = os.environ.get("CODE_AGENT_CONCURRENCY_ADMIN_EMAIL", "").strip()
    admin_password = os.environ.get("CODE_AGENT_CONCURRENCY_ADMIN_PASSWORD", "")
    if not admin_email or not admin_password:
        raise RuntimeError("CODE_AGENT_CONCURRENCY_ADMIN_EMAIL/PASSWORD are required")
    async with httpx.AsyncClient(base_url=args.api_url, timeout=30.0) as client:
        login = await request(client, "POST", "/auth/login", json={"email": admin_email, "password": admin_password})
        admin_token = login["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        users = []
        for index in range(args.users):
            email = f"code-concurrency-{index}@example.local"
            password = f"Concurrency!{index:03d}x"
            created = await request(
                client,
                "POST",
                "/auth/members",
                headers=admin_headers,
                json={"email": email, "display_name": f"Concurrency {index}", "password": password, "role": "admin", "seller_grants": []},
            )
            member_login = await request(client, "POST", "/auth/login", json={"email": email, "password": password})
            users.append({"user_id": created["user_id"], "token": member_login["access_token"]})
    results = await asyncio.gather(*[create_and_run(args.api_url, item["token"], index, args.timeout) for index, item in enumerate(users)])
    for result, user in zip(results, users):
        result["expected_user_id"] = user["user_id"]
    checks = db_checks(args.database_url, [item["job_id"] for item in results])
    db_jobs = {row["job_id"]: row for row in checks["jobs"]}
    for result in results:
        result["user_id"] = db_jobs[result["job_id"]]["user_id"]
    passed = (
        len(results) == args.users
        and all(item["status"] == "completed" for item in results)
        and all(item["user_id"] == item["expected_user_id"] for item in results)
        and all(len(item["artifact_ids"]) >= 1 for item in results)
        and len(checks["jobs"]) == args.users
        and all(row["status"] == "completed" and row["attempts"] == 1 and row["lease_token"] is None for row in checks["jobs"])
        and all(checks["audit_counts"].get(item["job_id"]) == 1 for item in results)
        and all(checks["traces"].get(item["job_id"], {}).get("unique") == 1 for item in results)
        and all(checks["artifacts"].get(item["job_id"], {}).get("distinct_users") == 1 for item in results)
    )
    return {"passed": passed, "users": args.users, "results": results, "database_checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--evidence", default="docs/代码Agent-20用户并发验收-20260830.md")
    args = parser.parse_args()
    report = asyncio.run(async_main(args))
    Path(args.evidence).write_text(
        "# Code Agent 20 用户并发验收（2026-08-30）\n\n"
        "本测试使用确定性 Fake Provider，隔离模型延迟，专门验证并发任务的用户归属、artifact 隔离、audit/trace 唯一性和 lease 收敛。\n\n"
        f"- 用户数：{report['users']}\n- 结果：{'PASS' if report['passed'] else 'FAIL'}\n"
        f"- completed：{sum(item['status'] == 'completed' for item in report['results'])}/{report['users']}\n"
        f"- audit：每个任务 1 条；trace：每个任务 1 个；attempts：每个任务 1 次；终态 lease：全部释放\n\n"
        "任务明细（job/user/artifact 数量）：\n\n"
        + "\n".join(
            f"- `{item['job_id']}` / `{item['expected_user_id']}` / {report['database_checks']['artifacts'][item['job_id']]['count']}"
            for item in report['results']
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": report["passed"], "users": report["users"], "completed": sum(item["status"] == "completed" for item in report["results"])}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
