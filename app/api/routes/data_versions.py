"""Version-manifest impact and replay APIs for the GMV pilot."""
import hashlib
import json
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from pydantic import BaseModel

from app.api.auth import require_analysis_write, require_tenant_context
from app.analysis.contracts import PeriodicReport
from app.data.database import engine
from app.data_versions.service import get_manifest, impact_memories, query_gmv_manifest, resolve_gmv_manifest, versioned_growth_diagnostics
from app.memory.repository import MemoryRepository
from app.tenancy import TenantContext
from app.tenancy.report_store import TenantReportStore

router = APIRouter(tags=["data-versioning"])

class ReplayRequest(BaseModel):
    artifact_type: str
    artifact_id: str
    mode: str

_ARTIFACT_TYPES = {"analysis_memory", "complex_execution", "report", "insight"}
_MODES = {"exact", "compare_latest"}

@router.get("/data-versions/{version_id}/impact")
def impact(version_id: str, apply: bool = False, context: TenantContext = Depends(require_tenant_context)) -> dict:
    try:
        return {"version_id": version_id, "applied": apply, "affected_memories": impact_memories(context, version_id, apply=apply)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@router.post("/replays")
def replay(payload: ReplayRequest, _: str = Depends(require_analysis_write), context: TenantContext = Depends(require_tenant_context)) -> dict:
    if payload.artifact_type not in _ARTIFACT_TYPES or payload.mode not in _MODES:
        raise HTTPException(status_code=422, detail=f"artifact_type must be one of {sorted(_ARTIFACT_TYPES)}; mode must be exact or compare_latest")
    if payload.artifact_type == "complex_execution":
        return _replay_complex_execution(payload.artifact_id, payload.mode, context)
    if payload.artifact_type in {"report", "insight"}:
        return _replay_report_artifact(payload.artifact_type, payload.artifact_id, payload.mode, context)
    memory = MemoryRepository(context).get_analysis_memory(payload.artifact_id)
    if not memory or not memory.get("source_data_version"):
        raise HTTPException(status_code=404, detail="Replayable memory not found")
    old = get_manifest(context, str(memory["source_data_version"]))
    if not old:
        return {"status": "unavailable", "reason": "sealed manifest is expired or unavailable"}
    try:
        exact = query_gmv_manifest(context, old["manifest_id"], memory["seller_id"], memory["start_date"], memory["end_date"])
    except ValueError:
        return {"status": "unavailable", "reason": "underlying sealed data is unavailable"}
    if payload.mode == "exact":
        snapshot = MemoryRepository(context).get_source_snapshot(str(memory["source_data_version"])) or {}
        expected = dict(snapshot.get("snapshot_payload") or {}).get("versioned_gmv_expected_value")
        status = "matched" if expected is None or float(expected) == exact["value"] else "drifted"
        result = {"status": status, "manifest": old["manifest_hash"], "expected_value": expected, "result": exact}
        _record_replay(context, "analysis_memory", payload.artifact_id, payload.mode, status, old["manifest_id"], None, result)
        return result
    latest = resolve_gmv_manifest(context, {}, str(memory["scope_fingerprint"]))
    current = query_gmv_manifest(context, latest["manifest_id"], memory["seller_id"], memory["start_date"], memory["end_date"])
    status = "drifted" if exact["value"] != current["value"] else "matched"
    result = {"status": status, "old_manifest": old["manifest_hash"], "latest_manifest": latest["manifest_hash"], "old_result": exact, "latest_result": current, "delta_value": current["value"] - exact["value"], "stale_recommended": status == "drifted"}
    _record_replay(context, "analysis_memory", payload.artifact_id, payload.mode, status, old["manifest_id"], latest["manifest_id"], result)
    return result


def _scope(conn, context: TenantContext) -> None:
    conn.execute(text("SELECT set_config('app.tenant_id', :v, true)"), {"v": context.tenant_id})
    conn.execute(text("SELECT set_config('app.user_id', :v, true)"), {"v": context.user_id})


def _replay_complex_execution(execution_id: str, mode: str, context: TenantContext) -> dict:
    with engine.begin() as conn:
        _scope(conn, context)
        audit = conn.execute(text("SELECT * FROM governance.analysis_execution_audits WHERE execution_id=:id"), {"id": execution_id}).mappings().first()
    if not audit or not audit["data_version_manifest_id"] or not dict(audit["metadata"] or {}).get("versioned_gmv_execution"):
        raise HTTPException(status_code=404, detail="Execution is not a replayable versioned GMV artifact")
    metadata = dict(audit["metadata"] or {})
    scope, params = dict(metadata["scope"]), dict(metadata["params"])
    old = get_manifest(context, str(audit["data_version_manifest_hash"]))
    if not old:
        return {"status": "unavailable", "reason": "sealed manifest is expired or unavailable"}
    try:
        exact = versioned_growth_diagnostics(context, old["manifest_id"], scope["seller_id"], __import__("datetime").date.fromisoformat(params["start_date"]), __import__("datetime").date.fromisoformat(params["end_date"]))
    except ValueError:
        return {"status": "unavailable", "reason": "underlying sealed data is unavailable"}
    exact_hash = hashlib.sha256(json.dumps(exact, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()
    if mode == "exact":
        status = "matched" if exact_hash == audit["result_hash"] else "drifted"
        result = {"status": status, "manifest": old["manifest_hash"], "result_hash": exact_hash, "expected_result_hash": audit["result_hash"], "result": exact}
        _record_replay(context, "complex_execution", execution_id, mode, status, old["manifest_id"], None, result)
        return result
    latest = resolve_gmv_manifest(context, scope, str(old["scope_fingerprint"]))
    current = versioned_growth_diagnostics(context, latest["manifest_id"], scope["seller_id"], __import__("datetime").date.fromisoformat(params["start_date"]), __import__("datetime").date.fromisoformat(params["end_date"]))
    status = "drifted" if exact != current else "matched"
    result = {"status": status, "old_manifest": old["manifest_hash"], "latest_manifest": latest["manifest_hash"], "old_result": exact, "latest_result": current, "stale_recommended": status == "drifted"}
    _record_replay(context, "complex_execution", execution_id, mode, status, old["manifest_id"], latest["manifest_id"], result)
    return result


def _replay_report_artifact(artifact_type: str, report_id: str, mode: str, context: TenantContext) -> dict:
    """Replay a report (or the report behind an insight) against its pinned manifest.

    Reports generated by the legacy unversioned path carry no manifest and are
    honestly reported as `unavailable` — never replayed against latest data and
    presented as if the old result still held.
    """
    try:
        stored = TenantReportStore(context).get(report_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Report not found")
    manifest_hash = stored.get("data_version_manifest_hash")
    if not manifest_hash:
        return {"status": "unavailable", "reason": "report has no data-version manifest (legacy unversioned generation path)"}
    old = get_manifest(context, manifest_hash)
    if not old:
        return {"status": "unavailable", "reason": "sealed manifest is expired or unavailable"}
    report = PeriodicReport.model_validate(stored["report_json"])
    scope = report.scope
    if scope.scope_type != "seller" or not scope.seller_id:
        return {"status": "unavailable", "reason": "only seller-scoped GMV reports are version-replayable"}
    try:
        exact = versioned_growth_diagnostics(context, old["manifest_id"], scope.seller_id, report.period.start_date, report.period.end_date)
    except ValueError:
        return {"status": "unavailable", "reason": "underlying sealed data is unavailable"}
    if mode == "exact":
        status = "matched" if exact.get("status") == "ok" else "drifted"
        result = {"status": status, "manifest": old["manifest_hash"], "result": exact}
        _record_replay(context, artifact_type, report_id, mode, status, old["manifest_id"], None, result)
        return result
    latest = resolve_gmv_manifest(context, scope.model_dump(mode="json"), str(old["scope_fingerprint"]))
    current = versioned_growth_diagnostics(context, latest["manifest_id"], scope.seller_id, report.period.start_date, report.period.end_date)
    status = "drifted" if exact != current else "matched"
    result = {"status": status, "old_manifest": old["manifest_hash"], "latest_manifest": latest["manifest_hash"], "old_result": exact, "latest_result": current, "stale_recommended": status == "drifted"}
    _record_replay(context, artifact_type, report_id, mode, status, old["manifest_id"], latest["manifest_id"], result)
    return result


def _record_replay(context: TenantContext, artifact_type: str, artifact_id: str, mode: str, status: str, old_manifest_id: str, compared_manifest_id: str | None, result: dict) -> None:
    with engine.begin() as conn:
        _scope(conn, context)
        conn.execute(text("""INSERT INTO governance.data_replay_audits(replay_id,tenant_id,user_id,artifact_type,artifact_id,mode,status,old_manifest_id,compared_manifest_id,result)
          VALUES(:id,:tenant,:user,:type,:artifact,:mode,:status,:old,:compared,CAST(:result AS jsonb))"""), {"id": f"replay_{uuid4().hex}", "tenant": context.tenant_id, "user": context.user_id, "type": artifact_type, "artifact": artifact_id, "mode": mode, "status": status, "old": old_manifest_id, "compared": compared_manifest_id, "result": json.dumps(result, default=str)})
