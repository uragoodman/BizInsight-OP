"""HTTP/Graph adapter for the durable chat lifecycle.

The graph remains unaware of database lifecycle writes.  This adapter creates
the durable Turn/Execution at the API boundary and exposes only immutable
metadata to the graph.  It is intentionally deterministic and never calls an
LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.auth.context import build_authorization_context
from app.chat_lifecycle.service import ExecutionHandle, LifecycleService, TurnHandle
from app.data.repositories.explorer import controlled_scope, preview_explorer
from app.scope.canonical import CanonicalScope
from app.scope.seller_scope import resolve_seller_scope, verify_seller_scope_payload
from app.schemas.explorer import ExplorerQuery
from app.tenancy import TenantContext


TERMINAL_TURN_STATES = {"completed", "abstained", "failed", "cancelled", "interrupted"}


def resolve_turn_task_type(primary: Any, fallback: Any = None) -> str:
    """Return a real result type, treating the creation sentinel as unset."""

    for value in (primary, fallback):
        candidate = str(value or "").strip()
        if candidate and candidate.lower() != "pending":
            return candidate
    return "unknown"


@dataclass(frozen=True, slots=True)
class ChatLifecycleBinding:
    service: LifecycleService
    turn: TurnHandle
    execution: ExecutionHandle | None
    persistence_turn_id: str
    authorization_boundary: dict[str, Any]
    query_scope_fingerprint: str
    seller_boundary: dict[str, Any] | None = None
    seller_scope_fingerprint: str | None = None
    active_reconnect: bool = False

    @property
    def is_replay(self) -> bool:
        return self.execution is None and self.turn.status in TERMINAL_TURN_STATES

    @property
    def is_suspended_resume(self) -> bool:
        return self.execution is not None and self.turn.status == "running"


def begin_chat_lifecycle(
    context: TenantContext,
    *,
    session_id: str | None,
    client_turn_id: str | None,
    inherited_from_turn_id: str | None = None,
    question: str,
    explorer_payload: dict[str, Any] | None,
    thread_id: str,
    trace_id: str,
    seller_scope_payload: dict[str, Any] | None = None,
) -> ChatLifecycleBinding | None:
    """Create or resume a durable scoped chat execution.

    Current chat requests without a session/client-turn/scope are kept on the
    legacy compatibility path until the caller supplies an explicit durable
    conversation identity.  Once all three are present, lifecycle setup is
    fail-closed: an unavailable migration or invalid server scope is raised
    instead of silently reverting to best-effort history.
    """

    if seller_scope_payload is not None:
        if explorer_payload is not None:
            raise ValueError("seller scope cannot be combined with an Explorer boundary")
        if not session_id or not client_turn_id:
            raise ValueError("seller scope requires a durable session and turn id")
        intent = verify_seller_scope_payload(seller_scope_payload, tenant=context)
        authz = build_authorization_context(
            context,
            allowed_metric_ids=intent.metric_ids,
            allowed_dimensions=intent.dimensions,
            metric_scope_type=context.scope_type,
        )
        canonical = resolve_seller_scope(intent, tenant=context, authz=authz)
        seller_boundary = {
            "permission_version": authz.permission_version,
            "policy_set_version": authz.policy_set_version,
            "authorization_fingerprint": authz.authz_digest,
            "query_scope_fingerprint": canonical.query_scope_fingerprint,
        }
        service = LifecycleService(context, authz)
        turn = service.create_turn(
            session_id=session_id,
            question=question,
            scope=canonical,
            client_turn_id=client_turn_id,
            inherited_from_turn_id=inherited_from_turn_id,
            context_snapshot={
                "scope": canonical.as_query_dict(),
                "seller_scope_intent": intent.model_dump(mode="json"),
                "seller_boundary": seller_boundary,
            },
        )
        execution = None
        active_reconnect = False
        if turn.status not in TERMINAL_TURN_STATES:
            if turn.status == "running":
                execution = service.load_suspended_execution(turn)
                if execution is None:
                    execution = service.load_active_execution(turn)
                    if execution is None:
                        raise RuntimeError("chat turn has no resumable active execution")
                    active_reconnect = True
                else:
                    thread_id = execution.thread_id or thread_id
                    trace_id = execution.trace_id or trace_id
                _assert_seller_execution_boundary(execution, authz, canonical)
            else:
                thread_id = f"turn:{turn.turn_id}"
                try:
                    execution = service.create_execution(
                        turn,
                        trace_id=trace_id,
                        thread_id=thread_id,
                        query_scope=canonical.narrow(),
                        execution_request_id=client_turn_id,
                    )
                except ValueError as exc:
                    if "active execution" not in str(exc):
                        raise
                    # A concurrent retry can race after the idempotency row
                    # check but before the first caller creates its Execution.
                    # Reuse that active execution as a reconnect; never mint a
                    # second attempt for the same client turn.
                    execution = service.load_active_execution(turn)
                    if execution is None:
                        raise
                    active_reconnect = True
                _assert_seller_execution_boundary(execution, authz, canonical)
        lifecycle_boundary = {
            **seller_boundary,
            "authz_ref": authz.authz_ref,
            "execution_id": execution.execution_id if execution else None,
            "turn_id": turn.turn_id,
        }
        return ChatLifecycleBinding(
            service=service,
            turn=turn,
            execution=execution,
            persistence_turn_id=turn.turn_id,
            authorization_boundary=lifecycle_boundary,
            query_scope_fingerprint=canonical.query_scope_fingerprint,
            seller_boundary=seller_boundary,
            seller_scope_fingerprint=canonical.query_scope_fingerprint,
            active_reconnect=active_reconnect,
        )

    if not session_id or not client_turn_id or not explorer_payload:
        return None

    explorer_query = ExplorerQuery.model_validate(explorer_payload)
    snapshot = preview_explorer(context, explorer_query)
    controlled = controlled_scope(snapshot)
    authz = build_authorization_context(
        context,
        allowed_metric_ids=controlled.allowed_metrics,
        allowed_dimensions=controlled.allowed_dimensions,
        metric_scope_type="seller",
    )
    canonical = CanonicalScope.resolve(
        authz,
        seller_ids=[controlled.seller_id],
        start_date=controlled.default_start_date,
        end_date=controlled.default_end_date,
        metric_ids=controlled.allowed_metrics,
        dimensions=controlled.allowed_dimensions,
        granularity=controlled.granularity,
    )
    service = LifecycleService(context, authz)
    turn = service.create_turn(
        session_id=session_id,
        question=question,
        scope=canonical,
        client_turn_id=client_turn_id,
        inherited_from_turn_id=inherited_from_turn_id,
        context_snapshot={
            "scope": canonical.as_query_dict(),
            "metric_name": controlled.metric_name,
        },
    )

    # An idempotent browser retry must not create a second execution.  A
    # suspended execution is resumed on the same LangGraph thread; an active
    # non-suspended execution is left for the original request/worker.
    #
    # A genuinely new Turn must get a fresh LangGraph thread.  The durable
    # conversation session and the structured predecessor context provide the
    # follow-up semantics; reusing an old thread here would silently mix a
    # normal follow-up with an earlier checkpoint.  Thread reuse is reserved
    # for the HITL/suspended branch below.
    execution = None
    active_reconnect = False
    if turn.status in TERMINAL_TURN_STATES:
        pass
    elif turn.status == "running":
        execution = service.load_suspended_execution(turn)
        if execution is None:
            execution = service.load_active_execution(turn)
            if execution is None:
                raise RuntimeError("chat turn has no resumable active execution")
            active_reconnect = True
        else:
            thread_id = execution.thread_id or thread_id
            trace_id = execution.trace_id or trace_id
    else:
        thread_id = f"turn:{turn.turn_id}"
        try:
            execution = service.create_execution(
                turn,
                trace_id=trace_id,
                thread_id=thread_id,
                query_scope=canonical.narrow(),
                execution_request_id=client_turn_id,
            )
        except ValueError as exc:
            if "active execution" not in str(exc):
                raise
            execution = service.load_active_execution(turn)
            if execution is None:
                raise
            active_reconnect = True

    boundary = {
        "permission_version": authz.permission_version,
        "policy_set_version": authz.policy_set_version,
        "authorization_fingerprint": authz.authz_digest,
        "query_scope_fingerprint": canonical.query_scope_fingerprint,
        "authz_ref": authz.authz_ref,
        "execution_id": execution.execution_id if execution else None,
        "turn_id": turn.turn_id,
    }
    return ChatLifecycleBinding(
        service=service,
        turn=turn,
        execution=execution,
        persistence_turn_id=turn.turn_id,
        authorization_boundary=boundary,
        query_scope_fingerprint=canonical.query_scope_fingerprint,
        active_reconnect=active_reconnect,
    )


def _assert_seller_execution_boundary(execution, authz, canonical: CanonicalScope) -> None:
    """Fail closed if a durable seller execution no longer matches this request."""

    if (
        execution.authorization_fingerprint != authz.authz_digest
        or execution.query_scope_fingerprint != canonical.query_scope_fingerprint
    ):
        raise ValueError("seller execution authorization or scope is stale")


def finalize_chat_lifecycle(
    binding: ChatLifecycleBinding,
    *,
    result: dict[str, Any] | None,
    status: str,
    interaction_state: str = "none",
    error_code: str | None = None,
    error_detail: dict[str, Any] | None = None,
    task_id: str | None = None,
    task_status: str | None = None,
    relation_decision: dict[str, Any] | None = None,
) -> bool:
    """Finalize exactly once; replay bindings are intentionally no-ops."""

    if binding.execution is None:
        return False
    answer = str((result or {}).get("answer") or "") or None
    committed = binding.service.finalize(
        binding.execution,
        status=status,
        answer=answer,
        result=result or {},
        error_code=error_code,
        error_detail=error_detail,
        interaction_state=interaction_state,
        final_node=str((result or {}).get("current_agent") or "") or None,
        task_id=task_id,
        task_status=task_status,
        relation_decision=relation_decision,
    )
    if not committed:
        raise RuntimeError("turn_commit_not_persisted")
    return True


def replay_response(binding: ChatLifecycleBinding) -> dict[str, Any]:
    """Build a safe response for an idempotent retry of a terminal Turn."""

    snapshot = dict(binding.turn.result_snapshot or {})
    metadata = binding.service.commit_metadata(binding.turn.turn_id)
    # Older durable Turns may still carry the creation sentinel even though
    # their result snapshot has the actual completed task type.
    task_type = resolve_turn_task_type(binding.turn.task_type, snapshot.get("task_type"))
    return {
        "task_type": task_type,
        "answer": binding.turn.answer or snapshot.get("answer") or "",
        "trace_id": binding.turn.trace_id or snapshot.get("trace_id") or "",
        "clarification": snapshot.get("clarification"),
        "presentation": snapshot.get("presentation"),
        "error": binding.turn.status in {"abstained", "failed", "interrupted"},
        "errors": snapshot.get("errors") or [],
        "durable_turn_id": metadata.get("turn_id") or binding.turn.turn_id,
        "execution_id": metadata.get("final_execution_id"),
        "commit_version": metadata.get("commit_version"),
    }
