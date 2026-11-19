from __future__ import annotations

import hashlib
import inspect
import json
import time
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

# Load decorator registrations before reading the whitelist registry.
import app.analysis.templates  # noqa: F401
from app.analysis.templates.registry import (
    get_template,
    list_chat_visible_templates,
    list_templates,
)
from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG
from app.context.semantic_contract_v2 import SealedRouteDecision
from app.data.database import engine
from app.data.repositories.explorer import controlled_scope, preview_explorer
from app.harness.storage import write_trace
from app.core.exceptions import sanitize_exception
from app.schemas.explorer import ExplorerQuery
from app.tenancy import TenantContext

PLAN_TTL_MINUTES = 30
EXECUTION_WALL_TIMEOUT_MS = 35_000

_FOCUS_OPTIONS: dict[str, list[dict[str, str]]] = {
    "cohort_retention": [{"key": "retention_curve", "label": "整体留存"}, {"key": "repeat_window", "label": "首购后复购窗口"}, {"key": "cohort_compare", "label": "群组差异"}],
    "rfm_segmentation": [{"key": "high_value", "label": "高价值维护"}, {"key": "at_risk", "label": "挽留风险"}, {"key": "growth", "label": "成长机会"}],
    "pareto_analysis": [{"key": "concentration", "label": "集中度风险"}, {"key": "head_categories", "label": "头部品类复核"}, {"key": "long_tail", "label": "长尾机会"}],
    "delay_review_correlation": [{"key": "fulfillment", "label": "履约环节"}, {"key": "regional", "label": "地区履约"}, {"key": "rating_link", "label": "评分关联验证"}],
}


def focus_options(template_name: str) -> list[dict[str, str]]:
    return _FOCUS_OPTIONS.get(template_name, [{"key": "overall", "label": "整体经营表现"}, {"key": "drivers", "label": "关键变化因素"}, {"key": "next_step", "label": "下一步验证"}])


def _preferences(template_name: str, values: dict | None = None) -> dict[str, str | None]:
    values = values or {}
    allowed = {item["key"] for item in focus_options(template_name)}
    focus_key = str(values.get("focus_key") or focus_options(template_name)[0]["key"])
    if focus_key not in allowed:
        raise ValueError("Selected focus is not available for this analysis template")
    note = str(values.get("note") or "").strip()
    if len(note) > 280:
        raise ValueError("Analysis note must be at most 280 characters")
    return {"focus_key": focus_key, "note": note or None, "source": str(values.get("source") or "user")}


def _fingerprint(
    template_name: str,
    scope: dict,
    params: dict,
    preferences: dict | None = None,
    *,
    template_code_hash: str | None = None,
    route_decision_hash: str | None = None,
) -> str:
    raw = json.dumps({
        "template_name": template_name,
        "scope": scope,
        "params": params,
        "preferences": preferences or {},
        "template_code_hash": template_code_hash,
        "route_decision_hash": route_decision_hash,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _legacy_fingerprint(template_name: str, scope: dict, params: dict) -> str:
    raw = json.dumps({"template_name": template_name, "scope": scope, "params": params}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _template_code_hash(fn) -> str:
    """Fingerprint the registered implementation, never user-provided code."""
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = f"{fn.__module__}:{fn.__qualname__}"
    return hashlib.sha256(source.encode()).hexdigest()


def _input_fingerprint(template_name: str, scope: dict, params: dict) -> str:
    """Identify the approved input contract; this is not a source-table snapshot."""
    return _digest({"template_name": template_name, "scope": scope, "params": params})


def _finish_audit(conn, execution_id: str, status: str, *, duration_ms: int, result_hash: str | None = None, trace_id: str | None = None, error_code: str | None = None) -> None:
    conn.execute(text("""UPDATE governance.analysis_execution_audits
      SET status=:status,finished_at=now(),duration_ms=:duration,result_hash=:result_hash,trace_id=:trace_id,error_code=:error_code
      WHERE execution_id=:id"""), {"id": execution_id, "status": status, "duration": duration_ms, "result_hash": result_hash, "trace_id": trace_id, "error_code": error_code})


def _scope(context: TenantContext, payload: dict) -> dict:
    # Preview is the authoritative scope validator; never trust browser rows.
    snapshot = preview_explorer(context, ExplorerQuery.model_validate(payload))
    return controlled_scope(snapshot).model_dump(mode="json")


def _revalidate_plan_scope(context: TenantContext, scope: dict, params: dict) -> None:
    seller_id = str(scope.get("seller_id") or "")
    start_date = str(params.get("start_date") or "")
    end_date = str(params.get("end_date") or "")
    if (
        not seller_id
        or str(params.get("seller_id") or "") != seller_id
        or str(scope.get("default_start_date") or "") != start_date
        or str(scope.get("default_end_date") or "") != end_date
    ):
        raise ValueError("Plan scope and execution parameters do not match")
    refreshed = _scope(context, {
        "seller_id": seller_id,
        "metric_name": str(scope.get("metric_name") or "gmv"),
        "start_date": start_date,
        "end_date": end_date,
        "granularity": str(scope.get("granularity") or "day"),
    })
    if (
        str(refreshed.get("seller_id") or "") != seller_id
        or str(refreshed.get("default_start_date") or "") != start_date
        or str(refreshed.get("default_end_date") or "") != end_date
    ):
        raise PermissionError("Plan seller/date scope is no longer authorized")


def _set_scope(conn, context: TenantContext) -> None:
    conn.execute(text("SELECT set_config('app.tenant_id', :v, true)"), {"v": context.tenant_id})
    conn.execute(text("SELECT set_config('app.user_id', :v, true)"), {"v": context.user_id})


def create_plan(context: TenantContext, question: str, explorer_context: dict, chat_session_id: str | None = None, chat_turn_id: str | None = None, template_name: str | None = None) -> dict:
    raise ValueError(
        "Direct natural-language plan creation is disabled; a server-sealed route_decision is required"
    )


def get_plan(context: TenantContext, plan_id: str) -> dict | None:
    with engine.begin() as conn:
        _set_scope(conn, context)
        row = conn.execute(text("SELECT * FROM governance.complex_analysis_plans WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user"), {
            "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
        }).mappings().one_or_none()
    return dict(row) if row else None


def _plan_document_v2(
    plan_id: str,
    name: str,
    question: str,
    scope: dict,
    params: dict,
    preferences: dict,
    *,
    chat_session_id: str | None,
    chat_turn_id: str | None,
    parent_plan_id: str | None = None,
    revision: int = 1,
    expires_at: datetime | None = None,
    route_decision_hash: str | None = None,
    contract_version: str | None = None,
    catalog_snapshot_hash: str | None = None,
    idempotency_key: str | None = None,
) -> dict:
    fn, display_name, required, _, description = get_template(name)
    plan = {"plan_id": plan_id, "template_name": name, "display_name": display_name, "purpose": description, "question": question,
            "chat_session_id": chat_session_id, "chat_turn_id": chat_turn_id, "required_params": required, "scope": scope, "params": params,
            "preferences": preferences, "focus_options": focus_options(name), "parent_plan_id": parent_plan_id, "revision": revision,
            "status": "pending_confirmation", "expires_at": (expires_at or datetime.now(timezone.utc) + timedelta(minutes=PLAN_TTL_MINUTES)).isoformat(),
            "estimated_cost": "受控白名单查询；最长 30 秒", "limitations": _limitations(name),
            "contract_version": contract_version,
            "catalog_snapshot_hash": catalog_snapshot_hash,
            "template_code_hash": _template_code_hash(fn), "route_decision_hash": route_decision_hash}
    if idempotency_key:
        plan["idempotency_key"] = idempotency_key
    return plan


def _insert_plan_v2(conn, context: TenantContext, plan: dict, parent_plan_id: str | None = None) -> None:
    conn.execute(text("""INSERT INTO governance.complex_analysis_plans
      (plan_id,tenant_id,user_id,chat_session_id,chat_turn_id,question,template_name,scope_data,params_data,preferences_data,plan_data,plan_hash,expires_at,parent_plan_id,revision)
      VALUES(:id,:tenant,:user,:session,:turn,:question,:template,CAST(:scope AS jsonb),CAST(:params AS jsonb),CAST(:preferences AS jsonb),CAST(:plan AS jsonb),:hash,:expires,:parent,:revision)"""),
      {"id": plan["plan_id"], "tenant": context.tenant_id, "user": context.user_id, "session": plan.get("chat_session_id"), "turn": plan.get("chat_turn_id"), "question": plan["question"], "template": plan["template_name"], "scope": json.dumps(plan["scope"]), "params": json.dumps(plan["params"]), "preferences": json.dumps(plan["preferences"]), "plan": json.dumps(plan, ensure_ascii=False), "hash": _fingerprint(plan["template_name"], plan["scope"], plan["params"], plan["preferences"], template_code_hash=plan.get("template_code_hash"), route_decision_hash=plan.get("route_decision_hash")), "expires": datetime.fromisoformat(plan["expires_at"]), "parent": parent_plan_id, "revision": plan["revision"]})


def _sealed_template_from_route_decision(route_decision: SealedRouteDecision) -> str:
    if not isinstance(route_decision, SealedRouteDecision):
        raise ValueError("A typed server-sealed route_decision is required")
    if not route_decision.verify_seal():
        raise ValueError("The server-sealed route_decision integrity check failed")
    if route_decision.route_decision_status != "SELECTED":
        raise ValueError("A sealed SELECTED route decision is required")
    if route_decision.catalog_snapshot_hash != DEFAULT_SEMANTIC_CATALOG.snapshot_hash():
        raise ValueError("The sealed route decision uses a stale semantic catalog")
    capability = next(
        (
            item
            for item in DEFAULT_SEMANTIC_CATALOG.capabilities
            if item.capability_id == route_decision.capability_id
        ),
        None,
    )
    if (
        capability is None
        or capability.route_id != route_decision.route_id
        or capability.template_id != route_decision.template_id
    ):
        raise ValueError("The sealed route decision does not match a registered capability")
    template_id = route_decision.template_id
    if template_id not in {str(item["name"]) for item in list_templates()}:
        raise ValueError("The selected template is not registered")
    if template_id not in {str(item["name"]) for item in list_chat_visible_templates()}:
        raise ValueError("The selected template is not visible to chat")
    return template_id


def create_plan_v2(
    context: TenantContext,
    question: str,
    explorer_context: dict,
    chat_session_id: str | None = None,
    chat_turn_id: str | None = None,
    template_name: str | None = None,
    preferences: dict | None = None,
    *,
    route_decision: SealedRouteDecision | None = None,
) -> dict:
    if route_decision is None:
        raise ValueError("A sealed route_decision is required; natural-language matching is disabled")
    if not isinstance(route_decision, SealedRouteDecision):
        raise ValueError("A typed server-sealed route_decision is required")
    # The service validates the sealed choice against the live whitelist and
    # never re-runs keyword routing on the question.
    name = _sealed_template_from_route_decision(route_decision)
    if template_name is not None and template_name != name:
        raise ValueError("Template name does not match the sealed route decision")
    scope = _scope(context, explorer_context)
    params = {"seller_id": scope["seller_id"], "start_date": scope["default_start_date"], "end_date": scope["default_end_date"]}
    route_decision_hash = _digest(route_decision.model_dump(mode="json"))
    idempotency_key = None
    idempotency_decision_hash = route_decision_hash
    if chat_turn_id:
        idempotency_key = _digest({
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "chat_session_id": chat_session_id,
            "chat_turn_id": chat_turn_id,
            "route_decision_hash": idempotency_decision_hash,
            "scope": scope,
        })
    plan = _plan_document_v2(
        str(uuid4()), name, question, scope, params, _preferences(name, preferences),
        chat_session_id=chat_session_id, chat_turn_id=chat_turn_id,
        route_decision_hash=route_decision_hash,
        contract_version=route_decision.contract_version,
        catalog_snapshot_hash=route_decision.catalog_snapshot_hash,
        idempotency_key=idempotency_key,
    )
    with engine.begin() as conn:
        _set_scope(conn, context)
        if idempotency_key:
            lock_key = f"complex-plan:{context.tenant_id}:{context.user_id}:{idempotency_key}"
            conn.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:lock_key, 0))"), {"lock_key": lock_key})
            pending = conn.execute(text("""SELECT plan_data FROM governance.complex_analysis_plans
              WHERE tenant_id=:tenant AND user_id=:user
                AND chat_session_id IS NOT DISTINCT FROM CAST(:session AS varchar)
                AND chat_turn_id IS NOT DISTINCT FROM CAST(:turn AS varchar)
                AND status='pending_confirmation' AND expires_at>now()
              ORDER BY created_at DESC FOR UPDATE"""), {
                "tenant": context.tenant_id, "user": context.user_id,
                "session": chat_session_id, "turn": chat_turn_id,
            }).mappings().all()
            for row in pending:
                existing = dict(row["plan_data"] or {})
                if existing.get("idempotency_key") == idempotency_key:
                    return existing
        if chat_session_id:
            conn.execute(text("""INSERT INTO governance.chat_sessions(tenant_id,user_id,chat_session_id,title,scope_data)
              VALUES(:tenant,:user,:session,:title,CAST(:scope AS jsonb)) ON CONFLICT(tenant_id,user_id,chat_session_id) DO UPDATE SET updated_at=now(),scope_data=EXCLUDED.scope_data"""), {"tenant": context.tenant_id, "user": context.user_id, "session": chat_session_id, "title": question[:80] or "新会话", "scope": json.dumps({"seller_id": scope["seller_id"], "start_date": scope["default_start_date"], "end_date": scope["default_end_date"], "metric_name": scope["metric_name"], "granularity": scope["granularity"]})})
        _insert_plan_v2(conn, context, plan)
    return plan


def get_plan_v2(context: TenantContext, plan_id: str) -> dict | None:
    with engine.begin() as conn:
        _set_scope(conn, context)
        row = conn.execute(text("SELECT * FROM governance.complex_analysis_plans WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user"), {
            "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
        }).mappings().one_or_none()
        if row:
            chain = []
            cursor = dict(row)
            while cursor:
                chain.append({"plan_id": cursor["plan_id"], "revision": cursor.get("revision") or 1, "status": cursor["status"], "created_at": cursor["created_at"].isoformat()})
                parent_id = cursor.get("parent_plan_id")
                if not parent_id:
                    break
                parent = conn.execute(text("SELECT plan_id,parent_plan_id,revision,status,created_at FROM governance.complex_analysis_plans WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user"), {
                    "id": parent_id, "tenant": context.tenant_id, "user": context.user_id,
                }).mappings().one_or_none()
                cursor = dict(parent) if parent else None
            latest_execution = conn.execute(text("""SELECT execution_id,status,started_at,finished_at,duration_ms,input_fingerprint,
              template_code_hash,result_hash,trace_id,error_code FROM governance.analysis_execution_audits
              WHERE complex_plan_id=:id AND tenant_id=:tenant AND user_id=:user
              ORDER BY started_at DESC LIMIT 1"""), {
                  "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
              }).mappings().one_or_none()
    if not row: return None
    stored, plan = dict(row), dict(row["plan_data"] or {})
    plan.update({"status": stored["status"], "expires_at": stored["expires_at"].isoformat(), "parent_plan_id": stored.get("parent_plan_id"), "revision": stored.get("revision", 1), "preferences": dict(stored.get("preferences_data") or plan.get("preferences") or {}), "focus_options": focus_options(str(stored["template_name"])), "revision_chain": list(reversed(chain)), "latest_execution": dict(latest_execution) if latest_execution else None, "trace_id": stored.get("trace_id") or plan.get("trace_id")})
    return plan


def revise_plan(context: TenantContext, plan_id: str, explorer_context: dict | None = None, preferences: dict | None = None) -> dict:
    now = datetime.now(timezone.utc)
    expired = False
    with engine.begin() as conn:
        _set_scope(conn, context)
        row = conn.execute(text("SELECT * FROM governance.complex_analysis_plans WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user FOR UPDATE"), {
            "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
        }).mappings().one_or_none()
        if not row: raise PermissionError("Complex analysis plan not found")
        if row["status"] != "pending_confirmation": raise ValueError("Only a pending plan can be revised")
        if row["expires_at"] <= now:
            conn.execute(text("UPDATE governance.complex_analysis_plans SET status='expired',updated_at=now() WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user"), {
                "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
            })
            expired = True
        else:
            old_scope = dict(row["scope_data"])
            source = explorer_context or {"seller_id": old_scope["seller_id"], "metric_name": old_scope["metric_name"], "start_date": old_scope["default_start_date"], "end_date": old_scope["default_end_date"], "granularity": old_scope["granularity"]}
            scope, name = _scope(context, source), str(row["template_name"])
            params = {"seller_id": scope["seller_id"], "start_date": scope["default_start_date"], "end_date": scope["default_end_date"]}
            safe_preferences = _preferences(name, preferences if preferences is not None else dict(row.get("preferences_data") or {}))
            prior_plan = dict(row.get("plan_data") or {})
            plan = _plan_document_v2(
                str(uuid4()), name, str(row["question"]), scope, params, safe_preferences,
                chat_session_id=row["chat_session_id"], chat_turn_id=row["chat_turn_id"],
                parent_plan_id=plan_id, revision=int(row.get("revision") or 1) + 1,
                expires_at=now + timedelta(minutes=PLAN_TTL_MINUTES),
                route_decision_hash=prior_plan.get("route_decision_hash"),
                contract_version=prior_plan.get("contract_version"),
                catalog_snapshot_hash=prior_plan.get("catalog_snapshot_hash"),
            )
            conn.execute(text("UPDATE governance.complex_analysis_plans SET status='superseded',updated_at=now() WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user"), {
                "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
            })
            _insert_plan_v2(conn, context, plan, plan_id)
    if expired:
        raise ValueError("Plan has expired; create a new analysis request")
    return plan


def cancel_plan(context: TenantContext, plan_id: str) -> bool:
    with engine.begin() as conn:
        _set_scope(conn, context)
        row = conn.execute(text("""UPDATE governance.complex_analysis_plans
          SET status=CASE WHEN status='running' THEN 'cancel_requested' ELSE 'cancelled' END,updated_at=now()
          WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user
            AND status IN ('pending_confirmation','running') RETURNING plan_id"""), {
                "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
            }).one_or_none()
    return row is not None


def execute_plan(context: TenantContext, plan_id: str) -> dict:
    execution_id = str(uuid4())
    started = time.monotonic()
    versioned_gmv_execution = False
    manifest_id = manifest_hash = None
    terminal_error: str | None = None
    try:
        with engine.begin() as conn:
            _set_scope(conn, context)
            row = conn.execute(text("SELECT * FROM governance.complex_analysis_plans WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user FOR UPDATE"), {
                "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
            }).mappings().one_or_none()
            if not row: raise PermissionError("Complex analysis plan not found")
            if str(row["tenant_id"]) != context.tenant_id or str(row["user_id"]) != context.user_id:
                raise PermissionError("Complex analysis plan not found")
            if row["status"] != "pending_confirmation": raise ValueError("Plan is no longer awaiting confirmation")
            if row["expires_at"] <= datetime.now(timezone.utc):
                conn.execute(text("UPDATE governance.complex_analysis_plans SET status='expired',updated_at=now() WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user"), {
                    "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
                })
                terminal_error = "Plan has expired"
            else:
                scope, params = dict(row["scope_data"]), dict(row["params_data"])
                preferences = dict(row.get("preferences_data") or {})
                plan_data = dict(row.get("plan_data") or {})
                template_name = str(row["template_name"])
                baseline_code_hash = str(plan_data.get("template_code_hash") or "")
                route_decision_hash = plan_data.get("route_decision_hash")
                if not baseline_code_hash:
                    # Legacy pending plans have no creation-time code baseline;
                    # fail closed and require the user to create or revise a plan.
                    raise ValueError("Template code baseline is missing; revise or create a new analysis plan")
                expected_hash = _fingerprint(
                    template_name, scope, params, preferences,
                    template_code_hash=baseline_code_hash,
                    route_decision_hash=str(route_decision_hash) if route_decision_hash else None,
                )
                if row["plan_hash"] != expected_hash:
                    conn.execute(text("UPDATE governance.complex_analysis_plans SET status='failed',updated_at=now() WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user"), {
                        "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
                    })
                    terminal_error = "Plan integrity validation failed"
                else:
                    if template_name not in {str(item["name"]) for item in list_templates()}:
                        raise ValueError("Template is no longer registered")
                    if template_name not in {str(item["name"]) for item in list_chat_visible_templates()}:
                        raise ValueError("Template is no longer visible")
                    try:
                        fn, display_name, _, _, _ = get_template(template_name)
                    except KeyError as exc:
                        raise ValueError("Template is no longer registered") from exc
                    code_hash = _template_code_hash(fn)
                    if code_hash != baseline_code_hash:
                        raise ValueError("Template code changed after plan creation")
                    _revalidate_plan_scope(context, scope, params)
                    input_hash = _input_fingerprint(template_name, scope, params)
                    # Complex templates still own their business computation; when the
                    # GMV pilot is enabled we nevertheless freeze the same source-version
                    # contract in their execution audit for traceability.
                    from app.core.config import get_settings
                    if get_settings().data_versioning_enabled:
                        from app.data_versions.service import resolve_gmv_manifest
                        from app.memory.service import scope_fingerprint
                        manifest = resolve_gmv_manifest(context, scope, scope_fingerprint(scope))
                        manifest_id, manifest_hash = manifest["manifest_id"], manifest["manifest_hash"]
                        versioned_gmv_execution = str(scope.get("metric_name")) == "gmv" and template_name == "growth_diagnostics"
                    conn.execute(text("UPDATE governance.complex_analysis_plans SET status='running',confirmed_at=now(),updated_at=now() WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user"), {
                        "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
                    })
                    conn.execute(text("""INSERT INTO governance.analysis_execution_audits
                      (execution_id,tenant_id,user_id,complex_plan_id,plan_revision,template_name,status,plan_hash,input_fingerprint,template_code_hash,data_version_manifest_id,data_version_manifest_hash,metadata)
                      VALUES(:execution,:tenant,:user,:plan,:revision,:template,'running',:plan_hash,:input_hash,:code_hash,:manifest_id,:manifest_hash,CAST(:metadata AS jsonb))"""),
                      {"execution": execution_id, "tenant": context.tenant_id, "user": context.user_id, "plan": plan_id, "revision": row.get("revision") or 1, "template": template_name, "plan_hash": row["plan_hash"], "input_hash": input_hash, "code_hash": code_hash, "manifest_id": manifest_id, "manifest_hash": manifest_hash, "metadata": json.dumps({"scope": scope, "params": params, "preferences_hash": _digest(preferences), "route_decision_hash": route_decision_hash, "template_code_hash_baseline": baseline_code_hash, "versioned_gmv_execution": versioned_gmv_execution}, ensure_ascii=False)})
        if terminal_error:
            # Let the transaction commit its terminal status before surfacing the
            # expected conflict to the API/lifecycle caller.
            raise ValueError(terminal_error)
    except IntegrityError as exc:
        raise ValueError("Another complex analysis is already running for this user") from exc
    try:
        if versioned_gmv_execution:
            from app.data_versions.service import versioned_growth_diagnostics
            result = versioned_growth_diagnostics(context, str(manifest_id), scope["seller_id"], date.fromisoformat(params["start_date"]), date.fromisoformat(params["end_date"]))
        else:
            result = fn(seller_id=scope["seller_id"], start_date=date.fromisoformat(params["start_date"]), end_date=date.fromisoformat(params["end_date"]))
    except Exception as exc:
        public_message, diagnostic = sanitize_exception(exc)
        with engine.begin() as conn:
            _set_scope(conn, context)
            conn.execute(text("""UPDATE governance.complex_analysis_plans
                SET status='failed',updated_at=now(),result_data=CAST(:result AS jsonb)
                WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user"""), {
                    "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
                    "result": json.dumps({"status": "failed", "message": public_message, "diagnostic": diagnostic}, ensure_ascii=False),
                })
            _finish_audit(conn, execution_id, "failed", duration_ms=round((time.monotonic() - started) * 1000), error_code=type(exc).__name__)
        raise ValueError(public_message) from exc
    duration_ms = round((time.monotonic() - started) * 1000)
    if duration_ms > EXECUTION_WALL_TIMEOUT_MS:
        with engine.begin() as conn:
            _set_scope(conn, context)
            conn.execute(text("UPDATE governance.complex_analysis_plans SET status='failed',updated_at=now() WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user"), {
                "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
            })
            _finish_audit(conn, execution_id, "timeout", duration_ms=duration_ms, error_code="wall_timeout")
        raise ValueError("Complex analysis exceeded its execution time limit")
    trace_id = str(uuid4())
    evidence_id = f"complex_{plan_id[:12]}"
    recommendations = _recommendations(str(row["template_name"]), result, evidence_id)
    limitations = _limitations(str(row["template_name"]))
    if preferences.get("note"):
        limitations.append("用户备注属于待验证假设；当前模板未直接验证的部分不会作为分析结论。")
    presentation = {"status": "completed", "scope": {"seller_id": scope["seller_id"], "start_date": params["start_date"], "end_date": params["end_date"], "granularity": "month"},
                    "template_analyses": [{"template_name": row["template_name"], "display_name": display_name, "params": params, "result": result, "evidence_ids": [evidence_id]}],
                    "plan_preferences": preferences,
                    "execution_governance": {"execution_id": execution_id, "duration_ms": duration_ms,
                                             "input_fingerprint": input_hash, "template_code_hash": code_hash},
                    "recommendations": [{"text": x, "evidence_ids": [evidence_id]} for x in recommendations],
                    "limitations": limitations, "evidence": [{"evidence_id": evidence_id, "source_type": "complex_template", "summary": display_name}], "trace_id": trace_id}
    answer = _answer(display_name, result, recommendations, str(row["template_name"]), preferences)
    action = _action_draft(str(row["template_name"]), display_name, result, recommendations, evidence_id, trace_id)
    if action:
        action["plan_preferences"] = preferences
    with engine.begin() as conn:
        _set_scope(conn, context)
        status = conn.execute(text("SELECT status FROM governance.complex_analysis_plans WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user FOR UPDATE"), {
            "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
        }).scalar_one_or_none()
        if status == "cancel_requested":
            conn.execute(text("UPDATE governance.complex_analysis_plans SET status='cancelled',updated_at=now() WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user"), {
                "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
            })
            _finish_audit(conn, execution_id, "cancelled", duration_ms=duration_ms, error_code="cancel_requested")
            return {"answer": "Complex analysis was cancelled before its result was saved.", "trace_id": "", "presentation": {"status": "failed", "limitations": ["The execution was cancelled; no conclusion or action draft was created."]}, "action_created": False, "task_type": "complex_analysis", "cancelled": True}
        conn.execute(text("""UPDATE governance.complex_analysis_plans
          SET status='completed',completed_at=now(),updated_at=now(),trace_id=:trace,result_data=CAST(:result AS jsonb)
          WHERE plan_id=:id AND tenant_id=:tenant AND user_id=:user"""), {
              "id": plan_id, "tenant": context.tenant_id, "user": context.user_id,
              "trace": trace_id,
              "result": json.dumps({"result": result, "presentation": presentation}, ensure_ascii=False),
          })
        _finish_audit(conn, execution_id, "completed", duration_ms=duration_ms, result_hash=_digest(result), trace_id=trace_id)
        if action:
            conn.execute(text("""INSERT INTO governance.action_plans(action_plan_id,tenant_id,user_id,complex_plan_id,title,content)
              VALUES(:id,:tenant,:user,:complex,:title,CAST(:content AS jsonb))"""), {"id": str(uuid4()), "tenant": context.tenant_id, "user": context.user_id, "complex": plan_id, "title": action["title"], "content": json.dumps(action, ensure_ascii=False)})
    write_trace(trace_id, {"trace_id": trace_id, "tenant_id": context.tenant_id, "user_id": context.user_id, "trace_kind": "complex_analysis", "node_history": [{"node": "complex_plan", "summary": "confirmed whitelist template"}, {"node": "human_clarification", "summary": preferences}, {"node": "execute_complex_analysis", "summary": display_name}], "complex_plan_id": plan_id, "plan_revision": row.get("revision", 1), "preferences": preferences, "result": result})
    return {"answer": answer, "trace_id": trace_id, "presentation": presentation, "action_created": bool(action), "task_type": "complex_analysis"}


def _limitations(name: str) -> list[str]:
    common = ["仅执行白名单模板；结果受锁定店铺和日期范围约束。"]
    if name == "delay_review_correlation": return common + ["统计关联不等于因果；仅样本≥50 且 p<0.05 才产生行动建议。"]
    if name == "pareto_analysis": return common + ["第一版仅支持品类维度，不代表商品级集中度。"]
    if name == "cohort_retention": return common + ["样本少于 30 名客户时不形成留存结论。"]
    if name in {"growth_diagnostics", "fulfillment_funnel", "customer_lifecycle", "category_portfolio", "logistics_experience", "payment_cancellation"}:
        return common + ["分析包输出的是描述性经营证据；需要通过后续对照或实验验证经营动作效果。"]
    return common + ["RFM 为描述性分群，不等同于客户流失因果预测。"]


def _recommendations(name: str, result: dict, evidence: str) -> list[str]:
    if result.get("status") != "ok": return []
    if name == "delay_review_correlation":
        chi = result.get("chi_square") or {}
        if result.get("sample_size", 0) >= 50 and chi.get("p_value") is not None and chi["p_value"] < 0.05:
            return ["优先排查延迟订单对应的履约环节，并以低评分率变化验证改善效果。"]
    elif name == "cohort_retention" and result.get("total_customers", 0) >= 30:
        return ["设计复购或召回实验，并在后续月份验证留存变化。"]
    elif name == "rfm_segmentation" and result.get("total_customers", 0) >= 30:
        return ["针对高价值与需挽留分群设计差异化触达实验，并比较复购率。"]
    elif name == "pareto_analysis" and len(result.get("items", [])) >= 2:
        return ["优先复核头部品类的供给、价格与投放配置，并监测集中度变化。"]
    elif name == "growth_diagnostics" and result.get("monthly"):
        return ["先复核波动月份的头部品类贡献，再为一个高影响品类制定价格、供给或投放对照方案。"]
    elif name == "fulfillment_funnel" and result.get("stages"):
        return ["优先处理订单量最大且未完成的履约阶段，并在下一个周期验证交付率与时长。"]
    elif name == "customer_lifecycle" and result.get("rfm", {}).get("status") == "ok":
        return ["围绕高价值与需挽留客群设计差异化复购实验，并按 Cohort 观察后续留存。"]
    elif name == "category_portfolio" and len(result.get("categories", [])) >= 2:
        return ["对头部品类分别复核运费占比、评分和客单价，避免只按 GMV 分配资源。"]
    elif name == "logistics_experience" and result.get("states"):
        return ["优先复核高延迟州的承运与发货环节，并以延迟率和低评分率作为验证指标。"]
    elif name == "payment_cancellation" and result.get("payment_segments"):
        return ["比较支付方式和分期分组的取消率，优先排查高取消且订单量足够的组合。"]
    return []


def _answer(display_name: str, result: dict, recommendations: list[str], template_name: str | None = None, preferences: dict | None = None) -> str:
    if result.get("status") != "ok": return f"{display_name}未形成可用结论：数据量或数据覆盖不足。建议扩大锁定范围后再次确认。"
    text = f"已完成{display_name}，结果已绑定到本次 Trace。"
    if preferences:
        focus = next((item["label"] for item in focus_options(template_name or "") if item["key"] == preferences.get("focus_key")), preferences.get("focus_key"))
        if focus:
            text += f"\n\n本次关注点：{focus}。"
        if preferences.get("note"):
            text += f"\n待验证假设：{preferences['note']}。未被当前证据支持的部分不会作为结论。"
    if result.get("summary"):
        text += f"\n\n核心发现：{result['summary']}"
    return text + ("\n\n行动建议：" + "；".join(recommendations) if recommendations else "\n\n当前结果未达到行动建议的证据阈值，建议先补充数据或进一步验证。")


def _action_draft(name: str, display: str, result: dict, recommendations: list[str], evidence: str, trace: str) -> dict | None:
    if not recommendations: return None
    return {"title": f"{display}行动草案", "finding": display, "evidence_ids": [evidence], "trace_id": trace, "actions": recommendations,
            "target_metric": "留存率/复购率" if name in {"cohort_retention", "rfm_segmentation", "customer_lifecycle"} else "低评分率" if name in {"delay_review_correlation", "logistics_experience"} else "取消率" if name == "payment_cancellation" else "GMV集中度",
            "validation_window": "下一个完整月", "success_criterion": "目标指标相对基线改善，并复核数据口径", "priority": "medium", "limitations": _limitations(name)}
