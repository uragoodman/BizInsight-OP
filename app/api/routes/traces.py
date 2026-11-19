from fastapi import APIRouter, Depends, HTTPException

from app.harness.storage import list_traces, read_trace
from app.api.auth import require_tenant_context
from app.tenancy import TenantContext
from app.iam.service import require_permission

router = APIRouter(prefix="/traces", tags=["traces"])


def _redact_semantic_audit(trace: dict) -> dict:
    """Keep P0 audit spans out of the HTTP trace view.

    The trace file retains the audit block for offline diagnosis, but the API
    (including the admin full view) must not surface an internal, versioned
    observation structure as part of the public trace response.
    """

    snapshot = trace.get("state_snapshot")
    if not isinstance(snapshot, dict):
        return trace
    compilation = snapshot.get("context_compilation")
    if not isinstance(compilation, dict) or "p0_audit" not in compilation:
        return trace
    redacted = dict(trace)
    safe_snapshot = dict(snapshot)
    safe_compilation = dict(compilation)
    safe_compilation.pop("p0_audit", None)
    safe_snapshot["context_compilation"] = safe_compilation
    redacted["state_snapshot"] = safe_snapshot
    return redacted


def _trace_view(trace: dict, context: TenantContext) -> dict:
    """Return a role-appropriate trace view.

    Trace ownership is enforced by the storage layer.  This second boundary
    prevents ordinary analysts from reading prompts, answers, evidence, or
    tool payloads that are useful for debugging but not required for routine
    operational visibility.  Platform/owner/admin users retain the full
    trace for incident investigation.
    """

    require_permission(context, "read")
    if context.is_platform_admin or context.role in {"owner", "admin"}:
        return _redact_semantic_audit(trace)
    snapshot = trace.get("state_snapshot") if isinstance(trace, dict) else {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    safe_snapshot = {
        key: snapshot[key]
        for key in ("tenant_id", "trace_id", "status", "error_code", "task_type", "current_agent")
        if key in snapshot
    }
    return {
        "recorded_at": trace.get("recorded_at"),
        "trace_version": trace.get("trace_version", {}),
        "graph_path": trace.get("graph_path", []),
        "state_snapshot": safe_snapshot,
        "events": [
            {"node": event.get("node"), "status": event.get("status"), "duration_ms": event.get("duration_ms")}
            for event in (trace.get("events") or [])
            if isinstance(event, dict)
        ],
        "view": "operational_summary",
    }


@router.get("")
def get_traces(tenant_context: TenantContext = Depends(require_tenant_context)) -> dict[str, list[dict]]:
    return {"items": list_traces(tenant_context)}


@router.get("/{trace_id}")
def get_trace(trace_id: str, tenant_context: TenantContext = Depends(require_tenant_context)) -> dict:
    trace = read_trace(trace_id, tenant_context)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return _trace_view(trace, tenant_context)
