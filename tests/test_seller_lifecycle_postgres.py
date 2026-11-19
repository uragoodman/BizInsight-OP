from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest
from sqlalchemy import text

from test_seller_metric_value_acceptance import isolated_seller_db  # noqa: F401


@pytest.fixture
def seller_lifecycle_db(isolated_seller_db, monkeypatch):
    """Add lifecycle contract tables to the existing temporary seller DB."""
    isolated_engine, _statements = isolated_seller_db
    ddl = [
        "CREATE SCHEMA IF NOT EXISTS governance",
        """
        CREATE TABLE IF NOT EXISTS governance.chat_sessions (
          tenant_id varchar(128) NOT NULL, user_id varchar(128) NOT NULL,
          chat_session_id varchar(128) NOT NULL, title varchar(160) NOT NULL,
          scope_data jsonb, focus_owner_execution_id varchar(128),
          session_state_version integer NOT NULL DEFAULT 0,
          created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
          PRIMARY KEY (tenant_id, user_id, chat_session_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS governance.chat_turns (
          turn_id varchar(128) PRIMARY KEY, tenant_id varchar(128) NOT NULL,
          user_id varchar(128) NOT NULL, chat_session_id varchar(128) NOT NULL,
          question text NOT NULL, answer text, task_type varchar(64) NOT NULL,
          trace_id varchar(160) NOT NULL DEFAULT '', client_turn_id varchar(160),
          status varchar(32) NOT NULL DEFAULT 'queued', interaction_state varchar(64) NOT NULL DEFAULT 'none',
          inherited_from_turn_id varchar(128), context_snapshot jsonb, result_snapshot jsonb,
          permission_version varchar(128), policy_set_version varchar(128), authz_ref varchar(160),
          authz_digest varchar(128), scope_fingerprint varchar(128), final_execution_id varchar(128),
          final_trace_id varchar(160), warning_codes jsonb NOT NULL DEFAULT '[]'::jsonb,
          error_code varchar(64), error_detail jsonb, finalized_at timestamptz,
          created_at timestamptz NOT NULL DEFAULT now()
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_seller_lifecycle_client_turn ON governance.chat_turns(tenant_id,user_id,chat_session_id,client_turn_id) WHERE client_turn_id IS NOT NULL",
        """
        CREATE TABLE IF NOT EXISTS governance.conversation_executions (
          execution_id varchar(128) PRIMARY KEY, turn_id varchar(128) NOT NULL,
          tenant_id varchar(128) NOT NULL, user_id varchar(128) NOT NULL,
          attempt_no integer NOT NULL, status varchar(32) NOT NULL DEFAULT 'queued',
          interaction_state varchar(64) NOT NULL DEFAULT 'none',
          langgraph_thread_id varchar(160), trace_id varchar(160),
          authorized_permission_version varchar(128) NOT NULL,
          authorized_policy_set_version varchar(128) NOT NULL,
          authorization_fingerprint varchar(128) NOT NULL,
          query_scope_fingerprint varchar(128) NOT NULL,
          heartbeat_at timestamptz, result_snapshot jsonb, error_code varchar(64),
          error_detail jsonb, terminal_event varchar(128), final_node varchar(160),
          execution_request_id varchar(160), created_at timestamptz NOT NULL DEFAULT now(),
          started_at timestamptz, finished_at timestamptz, UNIQUE(turn_id,attempt_no)
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_seller_lifecycle_active_execution ON governance.conversation_executions(turn_id) WHERE status IN ('queued','running','suspended')",
        """
        CREATE TABLE IF NOT EXISTS governance.authorization_snapshots (
          authz_ref varchar(160) PRIMARY KEY, tenant_id varchar(128) NOT NULL,
          user_id varchar(128) NOT NULL, permission_version varchar(128) NOT NULL,
          policy_set_version varchar(128) NOT NULL, snapshot_digest varchar(128) NOT NULL,
          seller_scope_type varchar(32) NOT NULL DEFAULT 'explicit',
          metric_policy_ref varchar(160), dimension_policy_ref varchar(160), tool_policy_ref varchar(160),
          created_at timestamptz NOT NULL DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS governance.policy_versions (
          tenant_id varchar(128) NOT NULL, user_id varchar(128) NOT NULL,
          permission_version varchar(128) NOT NULL, policy_set_version varchar(128) NOT NULL,
          catalog_digest varchar(128), authz_digest varchar(128) NOT NULL,
          policy_json jsonb NOT NULL DEFAULT '{}'::jsonb, status varchar(32) NOT NULL DEFAULT 'active',
          created_at timestamptz NOT NULL DEFAULT now(),
          PRIMARY KEY (tenant_id,user_id,permission_version,policy_set_version)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS governance.lifecycle_audit_events (
          event_id bigserial PRIMARY KEY, event_name varchar(96) NOT NULL,
          tenant_id varchar(128) NOT NULL, user_id varchar(128) NOT NULL,
          authz_ref varchar(160), permission_version varchar(128), policy_set_version varchar(128),
          turn_id varchar(128), execution_id varchar(128), trace_id varchar(160),
          payload jsonb NOT NULL DEFAULT '{}'::jsonb, created_at timestamptz NOT NULL DEFAULT now()
        )
        """,
    ]
    with isolated_engine.begin() as connection:
        for statement in ddl:
            connection.execute(text(statement))
    monkeypatch.setattr("app.chat_lifecycle.service.engine", isolated_engine)
    from app.core.config import Settings

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: Settings(
            tenant_auth_secret="seller-lifecycle-pg-secret",
            seller_dimension_enabled=True,
            seller_multi_scope_enabled=True,
        ),
    )
    return isolated_engine


def _context(*, tenant_id: str, user_id: str, role: str = "analyst", grants=None):
    from app.tenancy import SellerGrant, TenantContext

    return TenantContext(
        tenant_id=tenant_id, user_id=user_id, role=role, scope_type="seller",
        seller_id="A",
        seller_grants=tuple(grants or (SellerGrant("A", "2024-01-01", "2024-01-31"),)),
    )


def _payload(context, *, seller_id="A", start="2024-01-10", end="2024-01-10"):
    from app.scope.seller_scope import SellerScopeIntent, issue_seller_scope_payload

    return issue_seller_scope_payload(SellerScopeIntent(
        seller_scope_type="single", seller_ids=(seller_id,),
        start_date=date.fromisoformat(start), end_date=date.fromisoformat(end),
        metric_ids=("gmv",), dimensions=("seller",),
    ), tenant=context)


def _begin(context, payload, *, session_id, client_turn_id, inherited_from_turn_id=None,
           explorer_payload=None):
    from app.chat_lifecycle.integration import begin_chat_lifecycle

    return begin_chat_lifecycle(
        context, session_id=session_id, client_turn_id=client_turn_id,
        question="按 seller 拆解 GMV", explorer_payload=explorer_payload,
        seller_scope_payload=payload, thread_id=f"thread-{uuid4().hex}",
        trace_id=f"trace-{uuid4().hex}", inherited_from_turn_id=inherited_from_turn_id,
    )


def test_real_postgres_seller_lifecycle_persists_boundary_and_reuses_execution(
    seller_lifecycle_db,
):
    from app.chat_lifecycle.integration import _assert_seller_execution_boundary
    from app.auth.context import build_authorization_context
    from app.scope.seller_scope import resolve_seller_scope, verify_seller_scope_payload

    context = _context(tenant_id=f"tenant-{uuid4().hex}", user_id=f"user-{uuid4().hex}")
    payload = _payload(context)
    session_id, client_turn_id = f"session-{uuid4().hex}", f"client-{uuid4().hex}"
    first = _begin(context, payload, session_id=session_id, client_turn_id=client_turn_id)
    assert first.turn.turn_id
    assert first.execution is not None
    assert first.seller_scope_fingerprint == first.query_scope_fingerprint

    with seller_lifecycle_db.begin() as connection:
        row = connection.execute(text("""
            SELECT t.status, t.scope_fingerprint, t.context_snapshot,
                   e.status AS execution_status, e.query_scope_fingerprint
            FROM governance.chat_turns t
            JOIN governance.conversation_executions e ON e.turn_id=t.turn_id
            WHERE t.turn_id=:turn
        """), {"turn": first.turn.turn_id}).mappings().one()
        stored = dict(row["context_snapshot"])
        assert row["status"] == "running"
        assert row["scope_fingerprint"] == first.seller_scope_fingerprint
        assert row["query_scope_fingerprint"] == first.seller_scope_fingerprint
        assert stored["seller_scope_intent"]["seller_ids"] == ["A"]
        assert stored["seller_boundary"] == first.seller_boundary

    resumed = _begin(context, payload, session_id=session_id, client_turn_id=client_turn_id)
    assert resumed.execution is not None
    assert resumed.execution.execution_id == first.execution.execution_id
    authz = build_authorization_context(
        context, allowed_metric_ids=("gmv",), allowed_dimensions=("seller",),
        metric_scope_type="seller",
    )
    intent = verify_seller_scope_payload(payload, tenant=context)
    canonical = resolve_seller_scope(intent, tenant=context, authz=authz)
    _assert_seller_execution_boundary(resumed.execution, authz, canonical)


@pytest.mark.parametrize("stale_kind", ["permission", "policy", "grant", "seller_set"])
def test_real_postgres_seller_lifecycle_rejects_stale_boundary(
    seller_lifecycle_db, monkeypatch, stale_kind,
):
    from app.tenancy import SellerGrant

    context = _context(
        tenant_id=f"tenant-{uuid4().hex}", user_id=f"user-{uuid4().hex}",
        grants=(SellerGrant("A", "2024-01-01", "2024-01-31"), SellerGrant("B", "2024-01-01", "2024-01-31")),
    )
    payload = _payload(context)
    session_id, client_turn_id = f"session-{uuid4().hex}", f"client-{uuid4().hex}"
    _begin(context, payload, session_id=session_id, client_turn_id=client_turn_id)

    changed_context = context
    changed_payload = payload
    if stale_kind == "permission":
        changed_context = _context(
            tenant_id=context.tenant_id, user_id=context.user_id, role="reviewer",
            grants=context.seller_grants,
        )
    elif stale_kind == "policy":
        from app.auth import context as auth_context

        current_policy_tools = auth_context._policy_tools
        monkeypatch.setattr(
            auth_context, "_policy_tools",
            lambda role, tenant_id, scope_type: tuple(sorted({
                *current_policy_tools(role, tenant_id, scope_type), "test_policy_version_change",
            })),
        )
    elif stale_kind == "grant":
        changed_context = _context(
            tenant_id=context.tenant_id, user_id=context.user_id, role=context.role,
            grants=(SellerGrant("A", "2024-01-01", "2024-01-30"), SellerGrant("B", "2024-01-01", "2024-01-31")),
        )
    elif stale_kind == "seller_set":
        changed_payload = _payload(context, seller_id="B")

    with pytest.raises(ValueError, match="stale authorization or scope"):
        _begin(
            changed_context, changed_payload,
            session_id=session_id, client_turn_id=client_turn_id,
        )


def test_real_postgres_seller_lifecycle_rejects_explorer_boundary_and_cross_identity_inheritance(
    seller_lifecycle_db,
):
    context = _context(tenant_id=f"tenant-{uuid4().hex}", user_id=f"user-{uuid4().hex}")
    payload = _payload(context)
    session_id, client_turn_id = f"session-{uuid4().hex}", f"client-{uuid4().hex}"
    first = _begin(context, payload, session_id=session_id, client_turn_id=client_turn_id)

    with pytest.raises(ValueError, match="cannot be combined"):
        _begin(
            context, payload, session_id=f"session-{uuid4().hex}",
            client_turn_id=f"client-{uuid4().hex}", explorer_payload={"legacy": True},
        )

    with seller_lifecycle_db.begin() as connection:
        connection.execute(text("UPDATE governance.chat_turns SET status='completed' WHERE turn_id=:turn"), {"turn": first.turn.turn_id})
        connection.execute(text("UPDATE governance.conversation_executions SET status='completed' WHERE turn_id=:turn"), {"turn": first.turn.turn_id})

    with pytest.raises(ValueError, match="does not belong"):
        _begin(
            context, payload, session_id=f"other-session-{uuid4().hex}",
            client_turn_id=f"client-{uuid4().hex}", inherited_from_turn_id=first.turn.turn_id,
        )
    other_tenant = _context(tenant_id=f"tenant-{uuid4().hex}", user_id=context.user_id)
    with pytest.raises(ValueError, match="does not belong"):
        _begin(
            other_tenant, _payload(other_tenant), session_id=session_id,
            client_turn_id=f"client-{uuid4().hex}", inherited_from_turn_id=first.turn.turn_id,
        )
