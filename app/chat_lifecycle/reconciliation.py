"""Bounded lifecycle reconciliation entry point for cron/worker schedulers."""

from __future__ import annotations

from typing import Any

from app.auth.context import AuthorizationContext
from app.chat_lifecycle.service import LifecycleService
from app.core.config import get_settings
from app.tenancy import TenantContext
from app.observability import record_event


def run_lifecycle_reconciliation(
    context: TenantContext,
    authz: AuthorizationContext,
    *,
    stale_seconds: int | None = None,
) -> dict[str, Any]:
    """Run one bounded, tenant-scoped reconciliation pass.

    The function is intentionally one-shot so Docker, cron, Kubernetes Jobs,
    or an external scheduler can control cadence without embedding an
    unbounded loop in the API process.
    """

    service = LifecycleService(context, authz)
    stale = stale_seconds or get_settings().chat_execution_stale_seconds
    interrupted = service.reconcile_stale_executions(stale_seconds=stale)
    orphan_tasks = service.reconcile_stale_orphan_tasks(stale_seconds=max(stale, 86_400))
    integrity = service.reconcile_integrity()
    result = {
        "interrupted": interrupted,
        "orphan_tasks_interrupted": orphan_tasks,
        "integrity": integrity,
        "stale_seconds": stale,
        "orphan_task_stale_seconds": max(stale, 86_400),
    }
    record_event("chat.lifecycle.reconciliation.completed", {
        "tenant_id": context.tenant_id,
        "user_id": context.user_id,
        "interrupted": interrupted,
        "orphan_tasks_interrupted": orphan_tasks,
        "violations": sum(value for key, value in integrity.items() if key != "active_executions"),
    })
    return result
