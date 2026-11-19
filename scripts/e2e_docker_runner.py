"""真实 Docker Runner E2E：预取 → 建 job → confirm → worker(run_one) → Docker 沙箱执行 → 结果。

验证 P2 完整闭环：RLS 受控数据工件 → Code Agent job → 隔离容器执行 → 报告证据。
"""

from __future__ import annotations

import sys

from sqlalchemy import text

from app.code_analysis.dataset import prefetch_business_dataset
from app.code_analysis.service import confirm_job, create_job, get_job
from app.code_analysis.worker import run_one
from app.data.database import engine
from app.tenancy import SellerGrant, TenantContext


def main() -> int:
    # 1) 真实 seller + 范围
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT seller_id, min(purchase_date) AS s, max(purchase_date) AS e "
            "FROM analytics.fact_order_items GROUP BY seller_id ORDER BY count(*) DESC LIMIT 1"
        )).mappings().first()
    ctx = TenantContext(
        tenant_id="e2e-tenant", user_id="e2e-user", role="analyst",
        scope_type="seller", seller_id=row["seller_id"],
        seller_grants=(SellerGrant(seller_id=row["seller_id"]),),
    )
    print(f"[1] seller={row['seller_id']} range={row['s']}~{row['e']}")

    # 2) 预取受控业务数据 → 工件
    result = prefetch_business_dataset(
        ctx, seller_id=row["seller_id"],
        start_date=row["s"], end_date=row["e"],
        metric_names=["gmv", "order_count"],
    )
    file_id = result["file_id"]
    print(f"[2] prefetched file_id={file_id} rows={result['rows']} fingerprint={result['scope_fingerprint']}")

    # 3) 创建 job + 确认
    job = create_job(ctx, file_id, "对预取的经营数据做描述性概览并说明治理来源")
    plan_hash = job["plan_hash"]
    print(f"[3] job={job['job_id']} status={job['status']}")
    confirmed = confirm_job(ctx, job["job_id"], plan_hash)
    print(f"[4] confirmed status={confirmed['status']}")

    # 5) worker 单次运行（含 Docker 沙箱执行）
    ran = run_one()
    print(f"[5] worker run_one returned={ran}")

    # 6) 结果
    final = get_job(ctx, job["job_id"])
    print(f"[6] final status={final['status']}")
    if final["status"] == "completed":
        conclusion = (final.get("result_data") or {}).get("conclusion", "")
        print("    conclusion:", conclusion[:120])
        print("    artifacts:", [(a["kind"], a["size_bytes"]) for a in final.get("artifacts", [])])
        audit = final.get("latest_audit") or {}
        print("    audit: provider=%s runner_digest=%s result_hash=%s" % (
            audit.get("provider"), audit.get("runner_image_digest") or "?", audit.get("result_hash") or "?"))
        if final.get("artifacts"):
            print("E2E PASS")
            return 0
    print(f"E2E FAILED (status={final['status']} error={final.get('error_code')})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
