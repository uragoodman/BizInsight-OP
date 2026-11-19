# -*- coding: utf-8 -*-
"""Opt-in real-graph + real-LLM smoke for the WP5 durable chat chain.

Run against a cloned, migrated PostgreSQL database (POSTGRES_URL) with a real
LLM provider configured.  It verifies:

* two durable ``/chat`` turns on one session commit atomically (Turn,
  Execution, commit version, task association);
* the second turn inherits the first task and the captured model input
  contains a marker from the first turn (history really reaches the model);
* a durable ``/chat/stream`` turn emits ``completed`` only after the durable
  commit, and the event matches the committed row.

HITL is intentionally excluded: its trigger is a real-LLM verdict and is not
deterministic.  Deterministic LangGraph/PostgreSQL HITL tests remain the
authority for interrupt/replay semantics.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import text


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MARKER = "ZQ7"
MAX_LLM_CALLS = 60


def _session(client: TestClient, *, session: str, turn: str, message: str) -> dict:
    return client.post("/chat", json={
        "message": message,
        "use_current_explorer_context": True,
        "explorer_context": {
            "seller_id": "seller-1",
            "metric_name": "gmv",
            "start_date": "2018-08-01",
            "end_date": "2018-08-03",
            "granularity": "day",
        },
        "memory_enabled": False,
        "chat_session_id": session,
        "chat_turn_id": turn,
    })


def main() -> int:
    from app.data.database import engine
    from app.main import app
    from app.api.auth import require_tenant_context
    from app.tenancy import TenantContext
    from app.core.config import get_settings
    import app.llm.client as llm_client

    settings = get_settings()
    if not (settings.deepseek_api_key or "").strip():
        print("WP5_REAL_GRAPH_LLM_SKIPPED: no LLM API key configured")
        return 2

    context = TenantContext(
        tenant_id="platform", user_id=f"wp5-real-llm-{uuid.uuid4().hex[:8]}",
        role="admin", scope_type="platform",
    )
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO iam.tenants(tenant_id, tenant_name, scope_type, status)
            VALUES ('platform', 'wp5 real llm smoke', 'platform', 'active')
            ON CONFLICT (tenant_id) DO NOTHING
        """))
        connection.execute(text("""
            INSERT INTO iam.users(user_id, display_name, email, status)
            VALUES (:user, 'wp5 real llm smoke', :email, 'active')
            ON CONFLICT (user_id) DO NOTHING
        """), {"user": context.user_id, "email": f"{context.user_id}@example.invalid"})
        connection.execute(text("""
            INSERT INTO iam.tenant_memberships(tenant_id, user_id, role)
            VALUES ('platform', :user, 'admin') ON CONFLICT DO NOTHING
        """), {"user": context.user_id})

    calls: list[dict[str, str]] = []
    original_chat = llm_client.llm_chat
    original_classify = llm_client.llm_classify

    def recording_chat(system_prompt: str, user_message: str, **kwargs):
        calls.append({"kind": "chat", "system": str(system_prompt), "user": str(user_message)})
        if len(calls) > MAX_LLM_CALLS:
            raise RuntimeError(f"llm call cap exceeded ({MAX_LLM_CALLS})")
        return original_chat(system_prompt, user_message, **kwargs)

    def recording_classify(system_prompt: str, user_message: str, *args, **kwargs):
        calls.append({"kind": "classify", "system": str(system_prompt), "user": str(user_message)})
        if len(calls) > MAX_LLM_CALLS:
            raise RuntimeError(f"llm call cap exceeded ({MAX_LLM_CALLS})")
        return original_classify(system_prompt, user_message, *args, **kwargs)

    llm_client.llm_chat = recording_chat
    llm_client.llm_classify = recording_classify
    app.dependency_overrides[require_tenant_context] = lambda: context
    session_id = f"wp5-real-llm-{uuid.uuid4().hex}"
    client = TestClient(app)
    diagnostics: dict[str, object] = {"session": session_id}
    try:
        turn1 = f"turn-{uuid.uuid4().hex}"
        first = _session(
            client, session=session_id, turn=turn1,
            message=f"请帮我查询 2018-08-01 到 2018-08-03 的 GMV（编号 {MARKER}）",
        )
        body1 = first.json()
        assert first.status_code == 200, first.text
        assert body1.get("task_type") == "metric_query", body1
        assert body1.get("answer"), "real LLM returned an empty answer"
        durable_turn1 = body1.get("durable_turn_id")
        assert durable_turn1, body1

        with engine.connect() as connection:
            row1 = connection.execute(text("""
                SELECT turn_id, status, commit_version, task_id, final_execution_id
                  FROM governance.chat_turns
                 WHERE tenant_id=:tenant AND user_id=:user AND turn_id=:turn
            """), {"tenant": context.tenant_id, "user": context.user_id, "turn": durable_turn1}).mappings().one_or_none()
            execution1 = connection.execute(text("""
                SELECT status FROM governance.conversation_executions
                 WHERE execution_id=:execution
            """), {"execution": (row1 or {}).get("final_execution_id")}).scalar_one_or_none()
        assert row1 is not None and row1["status"] == "completed", row1
        assert int(row1["commit_version"]) == 1, row1
        assert row1["task_id"], row1
        assert execution1 == "completed", execution1
        task_id = row1["task_id"]

        turn2 = f"turn-{uuid.uuid4().hex}"
        marker_calls_before = len(calls)
        second = _session(
            client, session=session_id, turn=turn2,
            message="继续看一下 2018-08-02 的 GMV",
        )
        body2 = second.json()
        assert second.status_code == 200, second.text
        assert body2.get("answer"), "real LLM returned an empty answer on the follow-up"
        durable_turn2 = body2.get("durable_turn_id")
        assert durable_turn2, body2
        turn2_calls = calls[marker_calls_before:]
        prompt_text = "\n".join(item["system"] + "\n" + item["user"] for item in turn2_calls)
        diagnostics["turn2_prompt_chars"] = len(prompt_text)
        diagnostics["turn2_prompt_has_marker"] = MARKER in prompt_text
        assert turn2_calls, "the follow-up turn did not call the model"
        assert MARKER in prompt_text, (
            "the first turn's committed content did not reach the second turn's model input"
        )

        with engine.connect() as connection:
            row2 = connection.execute(text("""
                SELECT status, commit_version, task_id FROM governance.chat_turns
                 WHERE tenant_id=:tenant AND user_id=:user AND turn_id=:turn
            """), {"tenant": context.tenant_id, "user": context.user_id, "turn": durable_turn2}).mappings().one_or_none()
        assert row2 is not None and row2["status"] == "completed", row2
        assert int(row2["commit_version"]) == 2, row2
        assert row2["task_id"] == task_id, (row2, task_id)

        turn3 = f"turn-{uuid.uuid4().hex}"
        stream_calls_before = len(calls)
        completed_event = None
        committed_at_completed = None
        with client.stream("POST", "/chat/stream", json={
            "message": "再看 2018-08-03 的 GMV",
            "use_current_explorer_context": True,
            "explorer_context": {
                "seller_id": "seller-1", "metric_name": "gmv",
                "start_date": "2018-08-01", "end_date": "2018-08-03",
                "granularity": "day",
            },
            "memory_enabled": False,
            "chat_session_id": session_id,
            "chat_turn_id": turn3,
        }) as response:
            assert response.status_code == 200, response.read()
            for line in response.iter_lines():
                if not line or not str(line).startswith("data: "):
                    continue
                event = json.loads(str(line)[6:])
                if event.get("event") != "completed":
                    continue
                completed_event = event
                # The event must be emitted only after the durable commit:
                # query the row immediately, before the stream finishes.
                streamed_turn = event.get("durable_turn_id")
                assert streamed_turn, event
                with engine.connect() as connection:
                    committed_at_completed = connection.execute(text("""
                        SELECT status, commit_version FROM governance.chat_turns
                         WHERE tenant_id=:tenant AND user_id=:user AND turn_id=:turn
                    """), {
                        "tenant": context.tenant_id, "user": context.user_id, "turn": streamed_turn,
                    }).mappings().one_or_none()
                break
        assert completed_event is not None, "the stream never emitted completed"
        assert committed_at_completed is not None, "completed was emitted before the durable commit"
        assert committed_at_completed["status"] == "completed", committed_at_completed
        assert int(committed_at_completed["commit_version"]) == 3, committed_at_completed
        assert calls[stream_calls_before:], "the streamed turn did not call the model"
        diagnostics.update({
            "commits": [1, 2, int(committed_at_completed["commit_version"])],
            "task_id": task_id,
            "llm_calls": len(calls),
            "completed_event_keys": sorted(completed_event.keys()),
        })
        print("WP5_REAL_GRAPH_LLM_OK " + json.dumps(diagnostics, ensure_ascii=False))
        return 0
    except Exception:
        artifact_dir = Path(
            os.getenv("WP5_REAL_LLM_ARTIFACT_DIR") or tempfile.mkdtemp(prefix="wp5_real_llm_")
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "llm_calls.json").write_text(
            json.dumps(calls, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (artifact_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
        )
        print(f"WP5_REAL_GRAPH_LLM_ARTIFACTS={artifact_dir}")
        raise
    finally:
        llm_client.llm_chat = original_chat
        llm_client.llm_classify = original_classify
        app.dependency_overrides.pop(require_tenant_context, None)


if __name__ == "__main__":
    raise SystemExit(main())
