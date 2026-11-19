"""Create and remove an isolated PostgreSQL seller identity for browser E2E."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sqlalchemy import text

from app.data.database import engine
from app.iam.service import hash_password
from app.code_analysis.storage import get_artifact_store


def seed() -> None:
    suffix = uuid4().hex[:12]
    tenant_id = f"e2e_{suffix}"
    user_id = f"e2e_usr_{suffix}"
    email = f"e2e-{suffix}@example.test"
    password = f"E2e-{uuid4().hex[:16]}!"
    seller_id = "6560211a19b47992c3666cc44a7e94c0"
    with engine.begin() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM analytics.fact_order_items WHERE seller_id=:seller LIMIT 1"),
            {"seller": seller_id},
        ).scalar()
        if not exists:
            raise RuntimeError(f"Configured E2E seller has no order data: {seller_id}")
        conn.execute(text("INSERT INTO iam.tenants(tenant_id,tenant_name,scope_type) VALUES(:tenant,'Playwright E2E','seller')"), {"tenant": tenant_id})
        conn.execute(text("INSERT INTO iam.users(user_id,display_name,email,password_hash) VALUES(:user,'Playwright E2E',:email,:password_hash)"), {
            "user": user_id, "email": email, "password_hash": hash_password(password),
        })
        conn.execute(text("INSERT INTO iam.tenant_memberships(tenant_id,user_id,role,granted_by) VALUES(:tenant,:user,'owner',:user)"), {"tenant": tenant_id, "user": user_id})
        conn.execute(text("INSERT INTO iam.membership_seller_grants(tenant_id,user_id,seller_id,granted_by) VALUES(:tenant,:user,:seller,:user)"), {
            "tenant": tenant_id, "user": user_id, "seller": seller_id,
        })
    print(json.dumps({"tenant_id": tenant_id, "user_id": user_id, "email": email, "password": password, "seller_id": seller_id}))


def cleanup(tenant_id: str, user_id: str) -> None:
    object_store = get_artifact_store()
    with engine.begin() as conn:
        # Foreign-key children and scoped test artifacts, all limited to this run's tenant.
        object_keys = conn.execute(text("""
          SELECT object_key FROM governance.code_analysis_files WHERE tenant_id=:tenant
          UNION ALL
          SELECT object_key FROM governance.code_analysis_artifacts WHERE tenant_id=:tenant
        """), {"tenant": tenant_id}).scalars().all()
        for object_key in object_keys:
            object_store.delete(str(object_key))
        conn.execute(text("DELETE FROM governance.tenant_report_insights WHERE tenant_id=:tenant"), {"tenant": tenant_id})
        conn.execute(text("DELETE FROM governance.tenant_delivery_audit WHERE tenant_id=:tenant"), {"tenant": tenant_id})
        conn.execute(text("DELETE FROM governance.tenant_traces WHERE tenant_id=:tenant"), {"tenant": tenant_id})
        conn.execute(text("DELETE FROM governance.tenant_delegated_tasks WHERE tenant_id=:tenant"), {"tenant": tenant_id})
        conn.execute(text("DELETE FROM governance.tenant_knowledge_chunks WHERE tenant_id=:tenant"), {"tenant": tenant_id})
        conn.execute(text("DELETE FROM governance.tenant_outbox WHERE tenant_id=:tenant"), {"tenant": tenant_id})
        conn.execute(text("DELETE FROM governance.tenant_reports WHERE tenant_id=:tenant"), {"tenant": tenant_id})
        conn.execute(text("DELETE FROM analytics.tenant_metric_targets WHERE tenant_id=:tenant"), {"tenant": tenant_id})
        conn.execute(text("DELETE FROM analytics.inventory_snapshots WHERE tenant_id=:tenant"), {"tenant": tenant_id})
        conn.execute(text("DELETE FROM governance.code_analysis_jobs WHERE tenant_id=:tenant"), {"tenant": tenant_id})
        conn.execute(text("DELETE FROM governance.code_analysis_files WHERE tenant_id=:tenant"), {"tenant": tenant_id})
        conn.execute(text("DELETE FROM governance.audit_events WHERE tenant_id=:tenant AND user_id=:user"), {"tenant": tenant_id, "user": user_id})
        conn.execute(text("DELETE FROM iam.auth_sessions WHERE tenant_id=:tenant AND user_id=:user"), {"tenant": tenant_id, "user": user_id})
        conn.execute(text("DELETE FROM iam.membership_seller_grants WHERE tenant_id=:tenant AND user_id=:user"), {"tenant": tenant_id, "user": user_id})
        conn.execute(text("DELETE FROM iam.tenant_memberships WHERE tenant_id=:tenant AND user_id=:user"), {"tenant": tenant_id, "user": user_id})
        conn.execute(text("DELETE FROM iam.users WHERE user_id=:user"), {"user": user_id})
        conn.execute(text("DELETE FROM iam.tenants WHERE tenant_id=:tenant"), {"tenant": tenant_id})


if __name__ == "__main__":
    if sys.argv[1] == "seed":
        seed()
    elif sys.argv[1] == "cleanup":
        cleanup(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "cleanup-state":
        state = json.loads(Path(sys.argv[2]).read_text(encoding="utf8"))
        cleanup(state["tenant_id"], state["user_id"])
    else:
        raise SystemExit("usage: e2e_accounts.py seed|cleanup [tenant_id user_id]")
