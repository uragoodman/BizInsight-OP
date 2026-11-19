"""Periodically reconcile lifecycle state under freshly loaded IAM contexts."""

from __future__ import annotations

import argparse
import json
import logging
import time

from sqlalchemy import text

from app.auth.context import build_authorization_context
from app.chat_lifecycle.reconciliation import run_lifecycle_reconciliation
from app.data.database import engine
from app.iam.service import load_active_tenant_context
from app.observability import record_event


logger = logging.getLogger("bizinsight.lifecycle_reconciler")


def active_member_keys() -> list[tuple[str, str]]:
    """Return active tenant/user keys; roles and grants are reloaded per key."""

    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT m.tenant_id, m.user_id
            FROM iam.tenant_memberships m
            JOIN iam.tenants t ON t.tenant_id=m.tenant_id
            JOIN iam.users u ON u.user_id=m.user_id
            WHERE t.status='active' AND u.status='active' AND m.revoked_at IS NULL
            ORDER BY m.tenant_id, m.user_id
        """)).all()
    return [(str(row[0]), str(row[1])) for row in rows]


def run_reconciliation_pass(
    *,
    tenant_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, int]:
    """Run a bounded pass for all active members or one IAM-verified member.

    Caller-supplied tenant/user values only select an identity.  They never
    supply a role or scope: those are always reloaded from the IAM registry.
    """

    if bool(tenant_id) != bool(user_id):
        raise ValueError("tenant_id and user_id must be supplied together")
    keys = [(tenant_id, user_id)] if tenant_id and user_id else active_member_keys()
    result = {
        "identities_seen": len(keys),
        "identities_processed": 0,
        "identities_skipped": 0,
        "identity_errors": 0,
        "executions_interrupted": 0,
        "orphan_tasks_interrupted": 0,
    }
    for current_tenant_id, current_user_id in keys:
        try:
            context = load_active_tenant_context(
                tenant_id=current_tenant_id,
                user_id=current_user_id,
            )
            if context is None:
                result["identities_skipped"] += 1
                continue
            authz = build_authorization_context(context)
            item = run_lifecycle_reconciliation(context, authz)
        except Exception as exc:  # keep one bad tenant from stopping the bounded sweep
            result["identity_errors"] += 1
            logger.exception("Lifecycle reconciliation failed for one active IAM member")
            record_event("chat.lifecycle.reconciliation.member_failed", {
                "tenant_id": current_tenant_id,
                "user_id": current_user_id,
                "error_type": type(exc).__name__,
            })
            continue
        result["identities_processed"] += 1
        result["executions_interrupted"] += int(item["interrupted"])
        result["orphan_tasks_interrupted"] += int(item["orphan_tasks_interrupted"])
    record_event("chat.lifecycle.reconciliation.sweep", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", help="Optionally limit the pass to an IAM-verified tenant")
    parser.add_argument("--user", help="Optionally limit the pass to an IAM-verified user")
    parser.add_argument("--interval-seconds", type=int, default=0,
                        help="Repeat with this interval; 0 runs exactly once.")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Maximum passes when interval is enabled.")
    parser.add_argument("--forever", action="store_true",
                        help="Keep running as an operations worker until terminated.")
    args = parser.parse_args()
    if bool(args.tenant) != bool(args.user):
        parser.error("--tenant and --user must be supplied together")
    if args.interval_seconds < 0:
        parser.error("--interval-seconds must be non-negative")
    passes = (10**9 if args.forever else max(1, args.iterations)) if args.interval_seconds > 0 else 1
    for index in range(passes):
        result = run_reconciliation_pass(tenant_id=args.tenant, user_id=args.user)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.interval_seconds > 0 and index + 1 < passes:
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
