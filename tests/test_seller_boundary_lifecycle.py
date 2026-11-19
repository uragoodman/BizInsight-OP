from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace

import pytest

from app.chat_lifecycle import integration
from app.scope.canonical import CanonicalScope
from app.scope.seller_scope import SellerScopeIntent


@dataclass
class _Turn:
    turn_id: str
    tenant_id: str
    user_id: str
    session_id: str
    client_turn_id: str
    authz_ref: str
    permission_version: str
    policy_set_version: str
    authorization_fingerprint: str
    scope_fingerprint: str
    status: str = "queued"
    task_type: str = "pending"
    answer: str | None = None
    trace_id: str = ""
    result_snapshot: dict | None = None


@dataclass
class _Execution:
    execution_id: str
    turn_id: str
    attempt_no: int
    trace_id: str
    thread_id: str
    authorization_fingerprint: str
    query_scope_fingerprint: str


class _MemoryLifecycleService:
    turns: dict[tuple[str, str, str, str], _Turn] = {}
    snapshots: dict[str, dict] = {}
    executions: dict[str, _Execution] = {}

    def __init__(self, context, authz):
        self.context, self.authz = context, authz

    def create_turn(self, *, session_id, question, scope, client_turn_id,
                    inherited_from_turn_id=None, context_snapshot=None):
        key = (self.context.tenant_id, self.context.user_id, session_id, client_turn_id)
        current = self.turns.get(key)
        if current:
            if (current.permission_version != self.authz.permission_version
                    or current.policy_set_version != self.authz.policy_set_version
                    or current.authorization_fingerprint != self.authz.authz_digest):
                raise ValueError("existing client turn has stale authorization or scope")
            if current.scope_fingerprint != scope.query_scope_fingerprint:
                raise ValueError("existing client turn has stale authorization or scope")
            return current
        if inherited_from_turn_id:
            parent = next((turn for turn in self.turns.values()
                           if turn.turn_id == inherited_from_turn_id
                           and turn.tenant_id == self.context.tenant_id
                           and turn.user_id == self.context.user_id
                           and turn.session_id == session_id
                           and turn.authorization_fingerprint == self.authz.authz_digest), None)
            if parent is None:
                raise ValueError("inherited turn does not belong to the current user and session")
        turn = _Turn(
            turn_id=f"turn-{len(self.turns) + 1}",
            tenant_id=self.context.tenant_id, user_id=self.context.user_id,
            session_id=session_id,
            client_turn_id=client_turn_id, authz_ref=self.authz.authz_ref,
            permission_version=self.authz.permission_version,
            policy_set_version=self.authz.policy_set_version,
            authorization_fingerprint=self.authz.authz_digest,
            scope_fingerprint=scope.query_scope_fingerprint,
        )
        self.turns[key] = turn
        self.snapshots[turn.turn_id] = context_snapshot or {}
        return turn

    def create_execution(self, turn, *, trace_id, thread_id, query_scope,
                         execution_request_id=None):
        execution = _Execution(
            execution_id=f"exec-{turn.turn_id}", turn_id=turn.turn_id,
            attempt_no=1, trace_id=trace_id, thread_id=thread_id,
            authorization_fingerprint=self.authz.authz_digest,
            query_scope_fingerprint=query_scope.query_scope_fingerprint,
        )
        self.executions[turn.turn_id] = execution
        return execution

    def load_suspended_execution(self, turn):
        return self.executions.get(turn.turn_id)

    def load_active_execution(self, turn):
        return self.executions.get(turn.turn_id)


@pytest.fixture(autouse=True)
def _reset_memory_service(monkeypatch):
    _MemoryLifecycleService.turns.clear()
    _MemoryLifecycleService.snapshots.clear()
    _MemoryLifecycleService.executions.clear()
    monkeypatch.setattr(integration, "LifecycleService", _MemoryLifecycleService)
    monkeypatch.setattr(
        integration, "verify_seller_scope_payload",
        lambda payload, *, tenant: SellerScopeIntent.model_validate(payload["intent"]),
    )
    monkeypatch.setattr(integration, "build_authorization_context", _authz_for)
    monkeypatch.setattr(integration, "resolve_seller_scope", _resolve_scope)


def _authz_for(context, *, allowed_metric_ids, allowed_dimensions, metric_scope_type):
    return SimpleNamespace(
        tenant_id=context.tenant_id, user_id=context.user_id,
        authz_ref=f"ref:{context.user_id}",
        authz_digest=context.authz_digest,
        permission_version=context.permission_version,
        policy_set_version=context.policy_set_version,
    )


def _resolve_scope(intent, *, tenant, authz):
    seller_ids = tuple(intent.seller_ids)
    return CanonicalScope(
        tenant_id=tenant.tenant_id, user_id=tenant.user_id,
        seller_ids=seller_ids,
        start_date=intent.start_date,
        end_date=intent.end_date,
        metric_ids=tuple(intent.metric_ids),
        dimensions=tuple(intent.dimensions),
        authorization_fingerprint=authz.authz_digest,
        permission_version=authz.permission_version,
        policy_set_version=authz.policy_set_version,
        metric_definition_versions=(("gmv", "1"),),
        actions=("analyze",),
        seller_scope_type=intent.seller_scope_type,
    )


def _context(*, tenant="tenant-1", user="user-1", permission="perm-1", policy="policy-1"):
    return SimpleNamespace(
        tenant_id=tenant, user_id=user, authz_digest=f"digest:{permission}:{policy}",
        permission_version=permission, policy_set_version=policy,
        scope_type="seller",
    )


def _payload(*, seller_ids=("seller-a",), start_date="2026-01-01"):
    return {"intent": {
        "seller_scope_type": "single", "seller_ids": seller_ids,
        "start_date": start_date, "end_date": "2026-01-31",
        "metric_ids": ("gmv",), "dimensions": ("seller",),
    }}


def _begin(context, payload, *, session="session-1", turn_id="client-turn-1", **kwargs):
    return integration.begin_chat_lifecycle(
        context, session_id=session, client_turn_id=turn_id,
        question="GMV by seller", explorer_payload=None,
        seller_scope_payload=payload, thread_id="thread-new", trace_id="trace-new",
        **kwargs,
    )


def test_seller_scope_is_durably_bound_and_resume_reuses_same_execution():
    context, payload = _context(), _payload()
    first = _begin(context, payload)
    assert first.seller_boundary == {
        "permission_version": "perm-1", "policy_set_version": "policy-1",
        "authorization_fingerprint": "digest:perm-1:policy-1",
        "query_scope_fingerprint": first.query_scope_fingerprint,
    }
    assert first.seller_scope_fingerprint == first.query_scope_fingerprint
    persisted = _MemoryLifecycleService.snapshots[first.turn.turn_id]
    assert persisted["seller_scope_intent"]["seller_ids"] == ["seller-a"]
    assert persisted["seller_boundary"] == first.seller_boundary

    first.turn.status = "running"
    resumed = _begin(context, payload)
    assert resumed.execution.execution_id == first.execution.execution_id
    assert resumed.query_scope_fingerprint == first.query_scope_fingerprint


@pytest.mark.parametrize("change", ["permission", "policy"])
def test_seller_resume_rejects_stale_permission_or_policy(change):
    payload = _payload()
    first = _begin(_context(), payload)
    if change == "permission":
        context = _context(permission="perm-2")
    else:
        context = _context(policy="policy-2")
    with pytest.raises(ValueError, match="stale authorization or scope"):
        _begin(context, payload)


def test_seller_resume_rejects_changed_scope():
    context = _context()
    _begin(context, _payload())
    with pytest.raises(ValueError, match="stale authorization or scope"):
        _begin(context, _payload(seller_ids=("seller-b",)))


def test_explorer_boundary_cannot_be_used_as_seller_boundary():
    with pytest.raises(ValueError, match="cannot be combined"):
        integration.begin_chat_lifecycle(
            _context(), session_id="session-1", client_turn_id="turn-1",
            question="question", explorer_payload={"forged": True},
            seller_scope_payload=_payload(), thread_id="thread", trace_id="trace",
        )


def test_inherited_seller_turn_is_bound_to_session_and_tenant():
    context = _context()
    parent = _begin(context, _payload(), session="session-1", turn_id="parent")
    parent.turn.status = "completed"
    with pytest.raises(ValueError, match="does not belong"):
        _begin(context, _payload(), session="session-2", turn_id="child",
               inherited_from_turn_id=parent.turn.turn_id)
    other_tenant = _context(tenant="tenant-2")
    with pytest.raises(ValueError, match="does not belong"):
        _begin(other_tenant, _payload(), session="session-1", turn_id="child",
               inherited_from_turn_id=parent.turn.turn_id)

