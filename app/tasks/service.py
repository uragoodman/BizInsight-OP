from __future__ import annotations
import json
from secrets import token_hex
from datetime import datetime, timezone
from sqlalchemy import text
from app.data.database import engine
from app.data.rls import set_local_context
from app.memory.repository import MemoryRepository
from app.chat_history.repository import ChatHistoryRepository
from app.tenancy import TenantContext
from app.agents.relation_resolver import resolve_relation_llm, prior_evidence_ids
from app.agents.runtime import make_agent_run

TERMINAL={"completed","abstained","failed","cancelled","interrupted","superseded"}
def _scope(conn,c):
    set_local_context(conn, c)
def create_task(context: TenantContext, session_id:str, objective:str, scope:dict|None=None, status:str="draft") -> dict:
    task_id=f"ctk_{token_hex(12)}"; title=objective.strip()[:160] or "分析任务"
    with engine.begin() as conn:
        _scope(conn,context)
        conn.execute(text("""INSERT INTO governance.chat_sessions(tenant_id,user_id,chat_session_id,title,scope_data)
          VALUES(:t,:u,:s,:title,CAST(:scope AS jsonb)) ON CONFLICT DO NOTHING"""),{"t":context.tenant_id,"u":context.user_id,"s":session_id,"title":title,"scope":json.dumps(scope) if scope else None})
        conn.execute(text("""INSERT INTO governance.conversation_tasks(task_id,tenant_id,user_id,chat_session_id,title,objective,status,scope_data)
          VALUES(:id,:t,:u,:s,:title,:objective,:status,CAST(:scope AS jsonb))"""),{"id":task_id,"t":context.tenant_id,"u":context.user_id,"s":session_id,"title":title,"objective":objective[:4000],"status":status,"scope":json.dumps(scope) if scope else None})
    return get_task(context,task_id) or {}
def get_task(context, task_id):
    with engine.begin() as conn:
        _scope(conn,context); row=conn.execute(text("SELECT * FROM governance.conversation_tasks WHERE task_id=:id"),{"id":task_id}).mappings().one_or_none()
    return dict(row) if row else None


def _bounded_task_state(
    row: dict,
    state_patch: dict,
    *,
    status: str,
    trace_id: str,
    turn_id: str,
    relation_decision: dict | None = None,
) -> dict:
    """Merge one completed execution into a bounded, versioned task snapshot."""
    old = dict(row.get("task_state") or {})
    old_turns = list(old.get("recent_turns") or [])
    if any(str(item.get("turn_id")) == turn_id for item in old_turns if isinstance(item, dict)):
        return old

    scope_fingerprint = str(state_patch.get("scope_fingerprint") or "")
    data_version = str(state_patch.get("data_version_manifest_hash") or "")
    compatible = (
        bool(scope_fingerprint)
        and bool(data_version)
        and old.get("scope_fingerprint") == scope_fingerprint
        and str(old.get("data_version_manifest_hash") or "") == data_version
    )
    claims = list(old.get("approved_claims") or []) if compatible else []
    seen_claims = {str(item.get("value")) for item in claims if isinstance(item, dict)}
    claims.extend(
        item for item in state_patch.get("approved_claims", [])
        if isinstance(item, dict) and item.get("value") and str(item["value"]) not in seen_claims
    )
    hypotheses = dict(old.get("hypothesis_chain_summary") or {}) if compatible else {}
    hypotheses.update(state_patch.get("hypothesis_chain_summary") or {})
    evidence = list(old.get("evidence_refs") or []) if compatible else []
    evidence_by_id = {
        str(item.get("evidence_id")): item for item in evidence
        if isinstance(item, dict) and item.get("evidence_id")
    }
    for item in state_patch.get("evidence_refs", []):
        if isinstance(item, dict) and item.get("evidence_id") and item.get("trusted", True):
            evidence_by_id[str(item["evidence_id"])] = item

    resolution = (relation_decision or {}).get("context_resolution") or {}
    prior_slots = dict(old.get("context_slots") or {}) if compatible else {}
    slot_source = state_patch.get("context_slots") or resolution.get("known_slots") or {}
    new_slots = {}
    for name, slot in slot_source.items():
        if not isinstance(slot, dict) or slot.get("value") in (None, ""):
            continue
        bounded = dict(slot)
        if (
            bounded.get("source") in {"current_turn", "current_turn_correction"}
            and not bounded.get("source_turn_id")
        ):
            bounded["source_turn_id"] = turn_id
        new_slots[str(name)] = bounded
    prior_slots.update(new_slots)
    recent_turns = (old_turns + [{
        "turn_id": turn_id,
        "trace_id": trace_id,
        "task_type": state_patch.get("task_type"),
        "status": status,
        "scope_fingerprint": scope_fingerprint,
        "data_version_manifest_hash": data_version,
        "relation_action": (relation_decision or {}).get("action"),
        "relation_reason": str((relation_decision or {}).get("reason") or "")[:160],
        "context_source_turn_id": resolution.get("source_turn_id"),
        "context_sources": list(resolution.get("context_sources") or [])[:5],
        "resolved_slot_names": sorted((resolution.get("known_slots") or {}).keys())[:10],
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }])[-20:]
    return {
        "schema_version": "task-state-v1",
        "version": 1,
        "state_version": int(row.get("task_state_version") or 0) + 1,
        "objective": str(row.get("objective") or "")[:1000],
        "status": status,
        "scope_fingerprint": scope_fingerprint,
        "data_version_manifest_hash": data_version,
        "latest_trace_id": trace_id or None,
        "latest_turn_id": turn_id,
        "completed_steps": list(state_patch.get("completed_steps") or [])[-40:],
        "approved_claims": claims[-20:],
        "hypothesis_chain_summary": {key: value[-5:] for key, value in list(hypotheses.items())[-12:]},
        "evidence_refs": list(evidence_by_id.values())[-40:],
        "pending_questions": list(state_patch.get("pending_questions") or [])[:10],
        "context_slots": prior_slots,
        "recent_turns": recent_turns,
        "continuity_reset_reason": None if compatible or not old_turns else "scope_or_data_version_changed",
    }


def _task_state_patch_from_trace(context: TenantContext, trace_id: str, scope_fingerprint: str) -> dict:
    trace = None
    if trace_id:
        try:
            from app.harness.storage import read_trace

            trace = read_trace(trace_id, context)
        except Exception:  # noqa: BLE001 - task status must remain writable without trace enrichment
            trace = None
    snapshot = (trace or {}).get("state_snapshot", trace or {})
    has_controlled_scope = bool(snapshot.get("controlled_analysis_scope"))
    verdict = str(snapshot.get("critic_verdict") or "")
    evidence_rows = snapshot.get("evidence_bundles") or []
    evidence_refs = [
        {
            "evidence_id": str(row.get("evidence_id")),
            "source_type": row.get("source_type"),
            "source_ref": str(row.get("source_ref") or "")[:200],
            "trusted": bool(row.get("trusted", True)),
        }
        for row in evidence_rows[:40]
        if isinstance(row, dict) and row.get("evidence_id")
    ] if has_controlled_scope else []
    claims = []
    if verdict == "approved":
        known_evidence = {row["evidence_id"] for row in evidence_refs if row["trusted"]}
        for claim in snapshot.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            source_ids = [str(value) for value in claim.get("source_evidence_ids", [])]
            if source_ids and any(value in known_evidence for value in source_ids):
                claims.append({"value": str(claim.get("value") or "")[:200], "source_evidence_ids": source_ids[:5]})
    hypotheses = snapshot.get("hypothesis_chains_by_metric") or {}
    turn_resolution = snapshot.get("turn_resolution") or {}
    context_slots = turn_resolution.get("known_slots") or (
        (snapshot.get("context_compilation") or {}).get("known_slots") or {}
    )
    bounded_slots = {
        str(name): {
            key: slot.get(key)
            for key in (
                "value", "source", "confidence", "source_turn_id",
                "source_task_state_version", "candidate_id",
            )
            if slot.get(key) is not None
        }
        for name, slot in context_slots.items()
        if isinstance(slot, dict) and slot.get("value") not in (None, "")
    }
    return {
        "task_type": snapshot.get("task_type"),
        "scope_fingerprint": str(snapshot.get("current_query_scope_fingerprint") or "") if has_controlled_scope else "",
        "data_version_manifest_hash": str(snapshot.get("data_version_manifest_hash") or ""),
        "context_slots": bounded_slots,
        "completed_steps": [
            str(row.get("node")) for row in (snapshot.get("node_history") or [])
            if isinstance(row, dict) and row.get("node")
        ],
        "approved_claims": claims,
        "hypothesis_chain_summary": {
            str(metric): [
                {"hypothesis": str(step.get("hypothesis") or "")[:120],
                 "verdict": str(step.get("verdict") or ""),
                 "evidence_ids": list(step.get("evidence_ids") or [])[:5]}
                for step in chain[-5:] if isinstance(step, dict)
            ]
            for metric, chain in hypotheses.items() if isinstance(chain, list)
        },
        "evidence_refs": evidence_refs,
        "pending_questions": list((snapshot.get("metric_planning_decision") or {}).get("missing_metrics") or [])[:10],
    }


def record_task_progress(
    context: TenantContext,
    task_id: str,
    *,
    status: str,
    trace_id: str,
    turn_id: str,
    scope_fingerprint: str,
    relation_decision: dict | None = None,
) -> dict | None:
    """Persist an idempotent, scope/version-bound task progress snapshot.

    The conversation turn/trace remain the audit source. This snapshot is a
    bounded read model used to resume understanding across ordinary turns.
    """
    patch = _task_state_patch_from_trace(context, trace_id, scope_fingerprint)
    with engine.begin() as conn:
        _scope(conn, context)
        row = conn.execute(text("""
            SELECT * FROM governance.conversation_tasks
             WHERE task_id=:id FOR UPDATE
        """), {"id": task_id}).mappings().one_or_none()
        if row is None:
            return None
        existing_state = dict(row.get("task_state") or {})
        if any(str(item.get("turn_id")) == turn_id for item in existing_state.get("recent_turns", []) if isinstance(item, dict)):
            return dict(row)
        task_state = _bounded_task_state(
            dict(row), patch, status=status, trace_id=trace_id, turn_id=turn_id,
            relation_decision=relation_decision,
        )
        updated = conn.execute(text("""
            UPDATE governance.conversation_tasks
               SET status=:status,
                   latest_trace_id=COALESCE(NULLIF(:trace,''), latest_trace_id),
                   task_state=CAST(:task_state AS jsonb),
                   task_state_version=task_state_version+1,
                   updated_at=now(),
                   completed_at=CASE WHEN :status IN ('completed','abstained','failed','cancelled','interrupted','superseded') THEN now() ELSE completed_at END
             WHERE task_id=:id AND task_state_version=:expected_version
               AND (status NOT IN ('completed','abstained','failed','cancelled','interrupted','superseded') OR status=:status)
         RETURNING *
        """), {
            "id": task_id, "status": status, "trace": trace_id,
            "task_state": json.dumps(task_state, ensure_ascii=False),
            "expected_version": int(row.get("task_state_version") or 0),
        }).mappings().one_or_none()
        return dict(updated) if updated else None
def update_task(context,task_id,*,status=None,trace_id=None,route=None,pending=None,summary=None,reopen=False):
    values={"id":task_id,"status":status,"trace":trace_id,"route":json.dumps(route) if route is not None else None,"pending":json.dumps(pending) if pending is not None else None,"summary":json.dumps(summary) if summary is not None else None,"reopen":bool(reopen)}
    with engine.begin() as conn:
        _scope(conn,context); row=conn.execute(text("""UPDATE governance.conversation_tasks SET
          status=COALESCE(:status,status),latest_trace_id=COALESCE(:trace,latest_trace_id),route_data=COALESCE(CAST(:route AS jsonb),route_data),pending_data=COALESCE(CAST(:pending AS jsonb),pending_data),summary_data=COALESCE(CAST(:summary AS jsonb),summary_data),updated_at=now(),completed_at=CASE WHEN :reopen THEN NULL WHEN :status IN ('completed','abstained','failed','cancelled','interrupted','superseded') THEN now() ELSE completed_at END WHERE task_id=:id AND (status NOT IN ('completed','abstained','failed','cancelled','interrupted','superseded') OR status=:status OR :reopen) RETURNING *"""),values).mappings().one_or_none()
    return dict(row) if row else None
def task_context(context:TenantContext,session_id:str)->dict:
    with engine.begin() as conn:
        _scope(conn,context)
        tasks=[dict(r) for r in conn.execute(text("""SELECT * FROM governance.conversation_tasks
            WHERE tenant_id=:tenant AND user_id=:user AND chat_session_id=:session
            ORDER BY updated_at DESC"""),{"tenant":context.tenant_id,"user":context.user_id,"session":session_id}).mappings().all()]
        session_state=conn.execute(text("""SELECT active_task_id,focus_owner_execution_id,
            last_committed_turn_id,session_state_version,turn_commit_version,
            conversation_summary,summary_covered_from_turn_id,summary_covered_to_turn_id,
            summary_policy_version FROM governance.chat_sessions
            WHERE tenant_id=:tenant AND user_id=:user AND chat_session_id=:session"""),
            {"tenant":context.tenant_id,"user":context.user_id,"session":session_id}).mappings().one_or_none()
    session=MemoryRepository(context).get_session(session_id)
    l1=(session or {}).get("snapshot_data") or {}; l2=MemoryRepository(context).list_analysis_memories(limit=5)
    focused_id=(session_state or {}).get("active_task_id")
    current=next((t for t in tasks if t["task_id"]==focused_id),None)
    if current is None and session_state is None:
        current=next((t for t in tasks if t["status"] not in TERMINAL),tasks[0] if tasks else None)
    return {"tasks":tasks,"current_task":current,"session_state":dict(session_state or {}),"memory":{"l0":{"kind":"working","task_id":current["task_id"] if current else None,"status":current["status"] if current else "idle"},"l1":{"kind":"session_summary","summary":(session_state or {}).get("conversation_summary") or (session or {}).get("session_summary"),"summary_coverage":{"from_turn_id":(session_state or {}).get("summary_covered_from_turn_id"),"to_turn_id":(session_state or {}).get("summary_covered_to_turn_id"),"policy_version":(session_state or {}).get("summary_policy_version")},"snapshot":l1,"reason":"same-session committed summary"},"l2":{"kind":"approved_evidence","items":l2,"reason":"approved evidence-bound conclusions only"}}}
def _relation_result(payload: dict, *, driver: str, reason: str) -> dict:
    payload["agent_run"] = make_agent_run(
        run_id=f"relation:{payload.get('task_id') or 'new'}",
        domain="relation",
        decision_driver=driver if driver in {"llm", "workflow_fallback"} else "workflow_fallback",
        status="finished" if payload.get("action") != "clarify" else "abstained",
        termination_initiator="agent" if driver == "llm" else "policy",
        termination_reason=reason,
        rounds=1,
        trace=[{"node": "relation_resolver", "summary": f"action={payload.get('action')}"}],
    ).model_dump(mode="json")
    return payload


def _resolve_message_context(
    context,
    session_id: str,
    message: str,
    scope_fingerprint: str | None,
    *,
    task_progress_context: dict | None = None,
    expected_data_version: str | None = None,
) -> dict:
    """Compile the message against one completed, same-scope predecessor.

    This intentionally reuses the graph's context compiler. It only receives
    the structured turn context returned by ChatHistoryRepository plus the
    selected Task State read model; it does not infer authorization or accept
    a client-supplied scope as trusted context.
    """
    previous = {}
    if scope_fingerprint:
        try:
            previous = ChatHistoryRepository(context).latest_completed_context(
                session_id, scope_fingerprint=scope_fingerprint,
            ) or {}
        except Exception:  # noqa: BLE001 - context failure degrades to the raw message
            previous = {}
    from app.context.compiler import compile_turn_context

    task_context = dict(task_progress_context or {})
    state = compile_turn_context({
        "user_message": message,
        "previous_turn_context": previous,
        "task_progress_context": task_context,
        "task_state_version": int(task_context.get("task_state_version") or 0),
        "current_query_scope_fingerprint": scope_fingerprint or "",
        "data_version_manifest_hash": expected_data_version or "",
        "node_history": [],
    })
    compilation = state["context_compilation"]
    # Keep the resolver's source-need/turn metadata on the API decision so the
    # graph can reuse the same decision instead of producing a second
    # interpretation.  Preference candidate values remain graph-runtime state
    # and are deliberately not copied into the API decision payload.
    compilation["turn_resolution"] = state.get("turn_resolution") or {}
    return compilation


def _active_tasks(tasks: list[dict]) -> list[dict]:
    return [
        task for task in tasks
        if isinstance(task, dict) and str(task.get("status") or "") not in TERMINAL
    ]


def _task_candidate_metadata(tasks: list[dict]) -> list[dict]:
    return [
        {
            "task_id": task.get("task_id"),
            "status": task.get("status"),
            "title": task.get("title"),
            "latest_trace_id": task.get("latest_trace_id"),
        }
        for task in tasks
        if isinstance(task, dict) and task.get("task_id")
    ]


def _explicit_task_from_message(tasks: list[dict], message: str) -> dict | None:
    """Select a task only when the message names exactly one candidate."""
    folded = message.casefold()
    matches: dict[str, dict] = {}
    for task in tasks:
        if not isinstance(task, dict) or not task.get("task_id"):
            continue
        labels = (
            str(task.get("title") or "").strip().casefold(),
            str(task.get("objective") or "").strip().casefold(),
        )
        if any(label and len(label) >= 2 and label in folded for label in labels):
            matches[str(task["task_id"])] = task
    return next(iter(matches.values())) if len(matches) == 1 else None


def _task_scope_conflict(task: dict | None, scope: dict | None) -> bool:
    if not task or not isinstance(scope, dict):
        return False
    old = task.get("scope_data") or {}
    if not isinstance(old, dict) or not old:
        return False
    for key in ("seller_id", "start_date", "end_date"):
        if old.get(key) not in (None, "") and scope.get(key) not in (None, ""):
            if str(old.get(key)) != str(scope.get(key)):
                return True
    return False


def _has_explicit_request_slot(compilation: dict) -> bool:
    slots = compilation.get("known_slots") or {}
    return any(
        isinstance(slot, dict)
        and slot.get("source") in {"current_turn", "current_turn_correction"}
        and slot.get("value") not in (None, "")
        for name, slot in slots.items()
        if name in {"metric", "date", "dimension", "operation"}
    )


def _compatible_task_progress(
    task: dict | None,
    *,
    scope_fingerprint: str | None,
    expected_data_version: str | None,
) -> dict:
    if not task:
        return {}
    progress = dict(task.get("task_state") or {})
    if not scope_fingerprint or not progress.get("scope_fingerprint") or str(progress.get("scope_fingerprint")) != str(scope_fingerprint):
        progress.pop("context_slots", None)
        progress.pop("approved_claims", None)
        progress.pop("evidence_refs", None)
        progress.pop("hypothesis_chain_summary", None)
    if not expected_data_version or not progress.get("data_version_manifest_hash") or str(progress.get("data_version_manifest_hash")) != str(expected_data_version):
        progress.pop("context_slots", None)
        progress.pop("approved_claims", None)
        progress.pop("evidence_refs", None)
        progress.pop("hypothesis_chain_summary", None)
    progress.setdefault("task_id", task.get("task_id"))
    progress.setdefault("task_state_version", int(task.get("task_state_version") or 0))
    return progress


def _attach_task_relation(
    context_resolution: dict,
    *,
    action: str,
    task_id: str | None,
    reason: str,
    task_candidates: list[dict],
    scope_conflict: bool = False,
    ambiguous: bool = False,
    driver: str = "workflow_fallback",
) -> dict:
    """Expose one task relation decision on the compiled turn context."""
    updated = context_resolution
    updated["task_relation"] = {
        "action": action,
        "task_id": task_id,
        "reason": reason,
        "candidate_task_ids": [
            str(item.get("task_id")) for item in task_candidates
            if isinstance(item, dict) and item.get("task_id")
        ],
        "task_candidates": _task_candidate_metadata(task_candidates),
        "scope_conflict": bool(scope_conflict),
        "ambiguous": bool(ambiguous),
        "decision_driver": driver,
    }
    if action == "clarify":
        updated["needs_clarification"] = True
        updated["clarification_reason"] = reason
    nested = updated.get("turn_resolution")
    if isinstance(nested, dict):
        nested["task_relation"] = dict(updated["task_relation"])
        nested["relation"] = (
            "clarify" if action == "clarify"
            else "parallel_new" if action == "parallel_new"
            else "continue" if action == "continue"
            else "new"
        )
        nested["needs_clarification"] = bool(updated.get("needs_clarification"))
        nested["clarification_reason"] = updated.get("clarification_reason")
    return updated


def relation(context,session_id,message,scope,requested_task_id=None,*,scope_fingerprint=None,expected_data_version=None):
    data=task_context(context,session_id); current=data["current_task"]
    tasks=list(data.get("tasks") or [])
    active_tasks=_active_tasks(tasks)
    # Explicit task selection must refer to a task in this exact tenant/user
    # session result; a globally valid task ID is not enough.
    requested = next((task for task in tasks if task.get("task_id") == requested_task_id), None)
    if requested_task_id and not requested:
        context_resolution = _resolve_message_context(
            context, session_id, message, scope_fingerprint,
            expected_data_version=expected_data_version,
        )
        context_resolution = _attach_task_relation(
            context_resolution, action="clarify", task_id=None,
            reason="task_not_in_session", task_candidates=tasks,
        )
        return _relation_result({"action":"clarify","task_id":None,"reason":"task_not_in_session","context_resolution":context_resolution,"resolved_message":message}, driver="workflow_fallback", reason="Selected task is not part of this session.")

    # Multiple active tasks are ambiguous unless the caller selected one by
    # ID or the message names exactly one task.  Never choose by updated_at.
    explicit_message_task = (
        _explicit_task_from_message(active_tasks or tasks, message)
        if (len(active_tasks) > 1 and not requested) else None
    )
    if len(active_tasks) > 1 and not requested and explicit_message_task is None:
        context_resolution = _resolve_message_context(
            context, session_id, message, scope_fingerprint,
            expected_data_version=expected_data_version,
        )
        context_resolution = _attach_task_relation(
            context_resolution, action="clarify", task_id=None,
            reason="multiple_active_tasks", task_candidates=active_tasks,
            ambiguous=True,
        )
        return _relation_result({
            "action":"clarify","task_id":None,"reason":"multiple_active_tasks",
            "context_resolution":context_resolution,"resolved_message":message,
        }, driver="workflow_fallback", reason="Multiple active tasks require an explicit selection.")

    selected_task = requested or explicit_message_task or current
    selected_progress = _compatible_task_progress(
        selected_task,
        scope_fingerprint=scope_fingerprint,
        expected_data_version=expected_data_version,
    )
    context_resolution = _resolve_message_context(
        context, session_id, message, scope_fingerprint,
        task_progress_context=selected_progress if selected_task else None,
        expected_data_version=expected_data_version,
    )
    scope_conflict = _task_scope_conflict(selected_task, scope)
    base_context = {
        "resolved_message": str(context_resolution.get("standalone_question") or message),
        "context_resolution": context_resolution,
    }
    if requested:
        if context_resolution.get("needs_clarification"):
            reason = str(context_resolution.get("clarification_reason") or "context_clarification_required")
            context_resolution = _attach_task_relation(
                context_resolution, action="clarify", task_id=requested_task_id,
                reason=reason, task_candidates=tasks, ambiguous=True,
            )
            return _relation_result({"action":"clarify","task_id":requested_task_id,"reason":reason,**base_context}, driver="workflow_fallback", reason="The turn resolver requires clarification.")
        if scope_conflict:
            context_resolution = _attach_task_relation(
                context_resolution, action="clarify", task_id=requested_task_id,
                reason="task_scope_conflict", task_candidates=tasks,
                scope_conflict=True,
            )
            return _relation_result({"action":"clarify","task_id":requested_task_id,"reason":"task_scope_conflict",**base_context}, driver="workflow_fallback", reason="Selected task scope differs from the current request scope.")
        context_resolution = _attach_task_relation(
            context_resolution, action="continue", task_id=requested_task_id,
            reason="user_selected", task_candidates=tasks, driver="explicit",
        )
        return _relation_result({"action":"continue","task_id":requested_task_id,"reason":"user_selected",**base_context}, driver="workflow_fallback", reason="User selected an existing task.")
    if explicit_message_task is not None and len(active_tasks) > 1:
        if context_resolution.get("needs_clarification"):
            reason = str(context_resolution.get("clarification_reason") or "context_clarification_required")
            context_resolution = _attach_task_relation(
                context_resolution, action="clarify",
                task_id=str(explicit_message_task["task_id"]), reason=reason,
                task_candidates=active_tasks, ambiguous=True,
            )
            return _relation_result({"action":"clarify","task_id":str(explicit_message_task["task_id"]),"reason":reason,**base_context}, driver="workflow_fallback", reason="The turn resolver requires clarification.")
        context_resolution = _attach_task_relation(
            context_resolution, action="continue", task_id=str(explicit_message_task["task_id"]),
            reason="explicit_task_reference", task_candidates=active_tasks, driver="explicit",
        )
        return _relation_result({
            "action":"continue","task_id":str(explicit_message_task["task_id"]),
            "reason":"explicit_task_reference",**base_context,
        }, driver="workflow_fallback", reason="The message named one active task.")
    if not current:
        context_resolution = _attach_task_relation(
            context_resolution, action="new", task_id=None,
            reason="no_active_task", task_candidates=tasks,
        )
        return _relation_result({"action":"new","reason":"no_active_task",**base_context}, driver="workflow_fallback", reason="No active task.")
    # AGENT_PLAN Phase 7: controlled LLM relation decision; the keyword
    # heuristic below stays as the deterministic degradation path.
    from app.core.config import get_settings
    if get_settings().relation_resolver_llm_enabled:
        prompt_task = dict(current)
        task_progress = dict(prompt_task.get("task_state") or {})
        if (
            not scope_fingerprint
            or not expected_data_version
            or task_progress.get("scope_fingerprint") != scope_fingerprint
            or str(task_progress.get("data_version_manifest_hash") or "") != expected_data_version
        ):
            # A task can remain semantically relevant after its evidence goes
            # stale. Keep the task identity for routing, but hide outdated
            # claims and progress from the relation model.
            prompt_task.pop("task_state", None)
        available_evidence_ids = (
            prior_evidence_ids(
                current.get("latest_trace_id"), 10, scope_fingerprint, expected_data_version,
            )
            if scope_fingerprint and expected_data_version else []
        )
        decision=resolve_relation_llm(
            prompt_task, message, scope,
            available_evidence_ids,
            context_resolution, data.get("tasks", []),
        )
        if decision:
            action=decision["action"]
            if action=="CONTINUE":
                if context_resolution.get("needs_clarification"):
                    reason = str(context_resolution.get("clarification_reason") or "context_clarification_required")
                    context_resolution = _attach_task_relation(
                        context_resolution, action="clarify", task_id=decision["task_id"],
                        reason=reason, task_candidates=tasks, ambiguous=True, driver="llm",
                    )
                    return _relation_result({"action":"clarify","task_id":decision["task_id"],"reason":reason,**base_context}, driver="llm", reason="The turn resolver requires clarification.")
                if scope_conflict:
                    context_resolution = _attach_task_relation(
                        context_resolution, action="clarify", task_id=decision["task_id"],
                        reason="task_scope_conflict", task_candidates=tasks,
                        scope_conflict=True, driver="llm",
                    )
                    return _relation_result({"action":"clarify","task_id":decision["task_id"],"reason":"task_scope_conflict",**base_context}, driver="llm", reason="Task scope differs from the current request scope.")
                context_resolution = _attach_task_relation(
                    context_resolution, action="continue", task_id=decision["task_id"],
                    reason=decision["reason"] or "llm_continue", task_candidates=tasks, driver="llm",
                )
                return _relation_result({"action":"continue","task_id":decision["task_id"],"reason":decision["reason"] or "llm_continue","decision":"CONTINUE","reusable_evidence_ids":decision["reusable_evidence_ids"],**base_context}, driver="llm", reason=decision["reason"] or "LLM continued the task.")
            if action=="CLARIFY":
                context_resolution = _attach_task_relation(
                    context_resolution, action="clarify", task_id=current["task_id"],
                    reason=decision["reason"] or "llm_clarify", task_candidates=tasks,
                    ambiguous=True, driver="llm",
                )
                return _relation_result({"action":"clarify","task_id":current["task_id"],"reason":decision["reason"] or "llm_clarify","decision":"CLARIFY",**base_context}, driver="llm", reason=decision["reason"] or "LLM requested clarification.")
            context_resolution = _attach_task_relation(
                context_resolution,
                action="new" if action == "NEW_TASK" else "parallel_new",
                task_id=None, reason=decision["reason"] or "llm_new_task",
                task_candidates=tasks, driver="llm",
            )
            return _relation_result({"action":"new","reason":"parallel_new_task" if action=="PARALLEL_NEW_TASK" else "llm_new_task","decision":action,"reusable_evidence_ids":decision["reusable_evidence_ids"],**base_context}, driver="llm", reason=decision["reason"] or "LLM started a new task.")
    old=current.get("scope_data") or {}; same=bool(scope and old and scope.get("seller_id")==old.get("seller_id") and scope.get("start_date")==old.get("start_date") and scope.get("end_date")==old.get("end_date"))
    relation_message = str(context_resolution.get("standalone_question") or message)
    follow=(context_resolution.get("relation") == "continue") or any(
        x in relation_message.lower() for x in ("继续","刚才","上述","这个","它","进一步","why","then")
    )
    if same and follow and not scope_conflict and not context_resolution.get("needs_clarification"):
        context_resolution = _attach_task_relation(
            context_resolution, action="continue", task_id=current["task_id"],
            reason="same_scope_follow_up", task_candidates=tasks,
        )
        return _relation_result({"action":"continue","task_id":current["task_id"],"reason":"same_scope_follow_up",**base_context}, driver="workflow_fallback", reason="Deterministic follow-up heuristic.")
    if (
        context_resolution.get("relation") == "independent"
        and _has_explicit_request_slot(context_resolution)
        and not context_resolution.get("needs_clarification")
    ):
        context_resolution = _attach_task_relation(
            context_resolution, action="new", task_id=None,
            reason="independent_topic", task_candidates=tasks,
        )
        return _relation_result({"action":"new","reason":"independent_topic",**base_context}, driver="workflow_fallback", reason="A new explicit question starts a new task.")
    clarify_reason = str(
        context_resolution.get("clarification_reason")
        or ("task_scope_conflict" if scope_conflict else "task_relationship_ambiguous")
    )
    context_resolution = _attach_task_relation(
        context_resolution, action="clarify", task_id=current["task_id"],
        reason=clarify_reason,
        task_candidates=tasks, scope_conflict=scope_conflict, ambiguous=True,
    )
    return _relation_result({"action":"clarify","task_id":current["task_id"],"reason":clarify_reason,**base_context}, driver="workflow_fallback", reason="Deterministic relation heuristic was ambiguous.")
