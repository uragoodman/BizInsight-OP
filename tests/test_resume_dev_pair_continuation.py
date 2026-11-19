"""Safe continuation checks for the one never-started governed dev unit."""
from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import shutil
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_resume_dev_pair as runner


SOURCE = runner.ORIGINAL_PAIR_DIR


def _copy_original_pair(tmp_path: Path) -> Path:
    destination = tmp_path / SOURCE.name
    shutil.copytree(SOURCE, destination)
    return destination


def test_original_governed_continuation_preflight_and_reservation_are_exact():
    before = {
        name: (SOURCE / name).read_bytes()
        for name in ("run_manifest.json", "run_events.jsonl", "cost_reservations.jsonl", "summary.json")
    }
    result = runner._validate_governed_continuation(SOURCE)

    assert result["manifest"]["run_ids"]["governed_workflow"] == runner.GOVERNED_RUN_ID
    assert result["existing_reserved_cny"] == pytest.approx(2.32871262)
    assert result["new_reservation_cny"] == pytest.approx(2.32871262)
    assert result["projected_reserved_cny"] == pytest.approx(4.65742524)
    assert result["projected_reserved_cny"] <= result["limit_cny"] == 5.0
    assert not result["target_dir"].exists()
    assert {name: (SOURCE / name).read_bytes() for name in before} == before


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("run_id", "continuation_manifest_run_ids_mismatch"),
        ("config", "continuation_manifest_model_or_config_mismatch"),
        ("governed_event", "continuation_governed_has_existing_ledger_events"),
        ("over_reservation", "continuation_cumulative_estimate_limit_exceeded"),
    ],
)
def test_continuation_preflight_fails_closed_on_history_or_config_drift(tmp_path, mutation, message):
    copied = _copy_original_pair(tmp_path)
    if mutation in {"run_id", "config"}:
        manifest_path = copied / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutation == "run_id":
            manifest["run_ids"]["governed_workflow"] = "different-run"
        else:
            manifest["model_config"]["max_graph_steps"] = 49
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "governed_event":
        with (copied / "run_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"run_id": runner.GOVERNED_RUN_ID, "event": "started"}) + "\n")
    else:
        with (copied / "cost_reservations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"run_id": "prior-run", "reserved_cny": 3.0}) + "\n")

    with pytest.raises(RuntimeError, match=message):
        runner._validate_governed_continuation(copied)


@pytest.mark.parametrize(
    "args",
    [
        SimpleNamespace(preflight_governed_continuation=False, execute_authorized_continuation=False, continue_governed_run_id=runner.GOVERNED_RUN_ID),
        SimpleNamespace(preflight_governed_continuation=False, execute_authorized_continuation=True, continue_governed_run_id="other-run"),
    ],
)
def test_cli_requires_explicit_authorization_and_fixed_governed_id(args, capsys):
    assert runner._main_governed_continuation(args) == 2
    assert capsys.readouterr().out.startswith("Refusing continuation")


def test_cli_preflight_reports_cumulative_reservation_without_starting(capsys):
    args = SimpleNamespace(
        preflight_governed_continuation=True,
        execute_authorized_continuation=False,
        continue_governed_run_id=runner.GOVERNED_RUN_ID,
    )
    assert runner._main_governed_continuation(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "preflight_passed_no_run_started"
    assert output["projected_reserved_cny"] == pytest.approx(4.65742524)
    assert output["estimate_limit_cny"] == 5.0
    assert output["database_started"] is False
    assert output["provider_sends"] == 0


def test_real_network_guard_allows_windows_event_loop_socketpair_but_blocks_public_network():
    originals = runner._network_guard("127.0.0.1")
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(asyncio.sleep(0))
        finally:
            loop.close()
        left, right = socket.socketpair()
        left.close()
        right.close()
        with pytest.raises(OSError, match="network policy blocked"):
            socket.create_connection(("203.0.113.10", 443), timeout=0.1)
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.socketpair = originals[:3]


@pytest.mark.parametrize(
    ("mode", "expected_attempts", "expected_record_count"),
    [("provider_ab_experiment", 1, 1), ("normal", 2, 0)],
)
def test_mcp_resilience_attempts_are_single_only_for_experiment(mode, expected_attempts, expected_record_count, monkeypatch):
    from app.core.execution_context import activate_state, reset_state
    from app.mcp import client as mcp_client
    from app.resilience import policy as resilience_policy

    resilience_policy._BREAKERS.clear()
    calls = []

    async def fail_transport(tool_name, arguments):
        calls.append(tool_name)
        raise OSError("synthetic transport failure")

    monkeypatch.setattr(mcp_client, "ensure_tool_allowed", lambda *args, **kwargs: None)
    monkeypatch.setattr(mcp_client, "_call_local_mcp_tool", fail_transport)
    state = {"trace_id": f"test-{mode}", "llm_call_config": {"mode": mode}}
    token = activate_state(state)
    try:
        with pytest.raises(Exception):
            mcp_client.call_local_mcp_tool(
                "query_metric_trend", {}, tenant_id="platform", role="admin",
                request_id="request-mcp-test",
            )
    finally:
        reset_state(token)
        resilience_policy._BREAKERS.clear()

    assert len(calls) == expected_attempts
    assert len(state.get("mcp_call_attempts", [])) == expected_record_count
    if expected_record_count:
        attempt = state["mcp_call_attempts"][0]
        assert attempt["request_id"] == "request-mcp-test"
        assert attempt["attempt_number"] == attempt["attempt_limit"] == 1
        assert attempt["outcome"] == "exception"
        assert attempt["exception_chain"]


def test_bare_tool_error_keeps_safe_nested_cause_and_mcp_attempt_count():
    from app.resilience import ExternalServiceError
    from evaluation.resume_bare_tool_loop import PublicToolExecutionFailure, _safe_tool_error

    try:
        raise OSError("do not persist this raw message")
    except OSError as socket_error:
        try:
            raise ExternalServiceError("mcp", "external_call_failed", "sanitized wrapper") from socket_error
        except ExternalServiceError as wrapped:
            wrapped.mcp_attempt_count = 1
            failure = PublicToolExecutionFailure(
                tool_name="compare_breakdown_by_dimension",
                request_id="safe-request-id",
                plan=SimpleNamespace(
                    query_type="metric_contribution", metric_name="order_count", dimension="product_category",
                    baseline_date="2024-05-03", current_date="2024-05-04", start_date=None, end_date=None,
                ),
                cause=wrapped,
            )
            try:
                raise failure from wrapped
            except PublicToolExecutionFailure as linked_failure:
                failure = linked_failure
    detail = _safe_tool_error(failure)
    assert detail["service_error_code"] == "external_call_failed"
    assert detail["cause_type"] == "ExternalServiceError"
    assert detail["mcp_attempt_count"] == 1
    assert [item["type"] for item in detail["safe_cause_chain"]] == ["ExternalServiceError", "OSError"]
    assert all("message" not in item for item in detail["safe_cause_chain"])


class _OfflineSDK:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        system = str((kwargs.get("messages") or [{}])[0].get("content") or "")
        if "可选分类" in system:
            content = "metric_query"
        elif "请只输出 JSON 对象" in system:
            content = "{}"
        else:
            content = "2024-05-03 至 2024-05-04，已交付订单量由 10 降至 6，减少 4 单；品类贡献是统计解释，不代表业务因果。"
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=60),
        )


class _OfflineClient:
    def __init__(self, fake_sdk, **kwargs):
        assert kwargs.get("max_retries") == 0
        assert kwargs.get("timeout") == 60.0
        self.chat = SimpleNamespace(completions=fake_sdk)

    def close(self):
        return None


class _BareLoopSDK:
    """Mechanical action sequence for exercising the real guarded tool loop."""

    def __init__(self, run_id: str):
        self.requests = []
        self.actions = [
            {
                "action": "TOOL_CALL", "tool": "query_trend",
                "arguments": {"metric_id": "order_count", "start_date": "2024-05-03", "end_date": "2024-05-04"},
            },
            {
                "action": "TOOL_CALL", "tool": "metric_contribution",
                "arguments": {"metric_id": "order_count", "baseline_date": "2024-05-03", "current_date": "2024-05-04", "dimension": "product_category"},
            },
            {
                "action": "FINAL",
                "answer": "2024-05-03 至 2024-05-04，订单量从10降至6，减少4单；品类贡献仅表示统计变化，不代表业务因果。",
                "claims": [{"text": "订单量变化及统计贡献", "evidence_ids": [f"{run_id}:evidence:1", f"{run_id}:evidence:2"]}],
                "presentation": {
                    "metric_id": "order_count", "baseline_date": "2024-05-03", "current_date": "2024-05-04",
                    "baseline_value": 10, "current_value": 6, "delta": -4, "unit": "orders",
                    "dimension": "product_category", "items": [
                        {"key": "A", "delta": -4, "evidence_ids": [f"{run_id}:evidence:2"]},
                        {"key": "B", "delta": 0, "evidence_ids": [f"{run_id}:evidence:2"]},
                    ],
                },
            },
        ]

    def create(self, **kwargs):
        self.requests.append(kwargs)
        action = self.actions[len(self.requests) - 1]
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(action, ensure_ascii=False)),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=40, completion_tokens=30),
        )


def test_bare_loop_executes_trend_contribution_and_final_under_real_guard_with_sdk_stub(
    monkeypatch, tmp_path,
):
    admin_url = os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL")
    if not admin_url:
        pytest.skip("Set ISOLATED_TEST_POSTGRES_ADMIN_URL to an isolated-test PostgreSQL maintenance URL")
    from app.core import config as core_config
    from app.llm import client as llm
    from app.llm.call_config import LLMCallConfig
    from app.tenancy import TenantContext
    from evaluation.provider_candidate_batch import RunLedger, bind_isolated_experiment_runtime
    from evaluation.resume_bare_tool_loop import run_bare_tool_loop

    admin_engine = engine = None
    db_name = None
    originals = None
    run_id = "offline-bare-guard-test"
    context = TenantContext(tenant_id="platform", user_id="resume-dev-pair-order-dimension", role="admin", scope_type="platform")
    fake_sdk = _BareLoopSDK(run_id)
    safe_settings = core_config.get_settings().model_copy(update={
        "deepseek_api_key": "offline-placeholder",
        "deepseek_base_url": "https://127.0.0.1/v1",
        "llm_providers_json": "[]",
        "llm_timeout_seconds": 60.0,
    })
    monkeypatch.setattr(core_config, "get_settings", lambda: safe_settings)
    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: _OfflineClient(fake_sdk, **kwargs))
    try:
        originals = runner._network_guard("127.0.0.1")
        admin_engine, engine, db_name = runner._create_seeded_db(admin_url)
        ledger = RunLedger(tmp_path / "bare-events.jsonl")
        ledger.append(run_id, "started", case_id="order_dimension", arm="basic_tool_loop")
        config, _policy = runner._continuation_config()
        call_config = LLMCallConfig.provider_ab_experiment(
            safe_settings, provider_id="primary", max_completion_tokens=4096,
        )
        with bind_isolated_experiment_runtime(
            engine=engine, trace_dir=tmp_path / "bare-traces", ledger=ledger, config=config,
        ):
            result = run_bare_tool_loop(
                runner.QUESTION, tenant_context=context, run_id=run_id,
                llm_call_config=call_config,
            )
        assert result["final"]["action"] == "FINAL"
        assert len(fake_sdk.requests) == 3
        assert [call.get("call_status") for call in result["tool_calls"]] == ["succeeded", "succeeded"]
        assert [call.get("tool") for call in result["tool_calls"]] == ["query_trend", "metric_contribution"]
        assert all(
            row.get("evidence_id") and
            (row.get("call") or {}).get("response", {}).get("result") is not None
            for row in result["tool_evidence"]
        )
        assert all(row["attempt_limit"] == 1 for row in result["mcp_call_attempts"])
        assert all(row["request_id"] for row in result["mcp_call_attempts"])
        assert all(row["outcome"] == "response_received" for row in result["mcp_call_attempts"])
        assert result["tool_calls"][0]["arguments"]["start_date"] == "2024-05-03"
        assert result["tool_calls"][0]["arguments"]["end_date"] == "2024-05-04"
        assert result["tool_calls"][1]["arguments"]["baseline_date"] == "2024-05-03"
        assert result["tool_calls"][1]["arguments"]["current_date"] == "2024-05-04"
        assert result["final"]["presentation"]["delta"] == -4
    finally:
        if originals:
            socket.socket.connect, socket.socket.connect_ex, socket.socketpair = originals[:3]
        if engine is not None:
            engine.dispose()
        if admin_engine is not None and db_name:
            from sqlalchemy import text
            with admin_engine.connect() as connection:
                connection.execute(text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
            admin_engine.dispose()


@pytest.fixture
def deny_external_network(monkeypatch):
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create = socket.create_connection

    def local(address):
        host = address[0] if isinstance(address, tuple) else address
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except (TypeError, ValueError):
            return False

    def connect(sock, address):
        if not local(address):
            raise AssertionError(f"external connection blocked in offline continuation test: {address!r}")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if not local(address):
            raise AssertionError(f"external connection blocked in offline continuation test: {address!r}")
        return original_connect_ex(sock, address)

    def create_connection(address, *args, **kwargs):
        if not local(address):
            raise AssertionError(f"external connection blocked in offline continuation test: {address!r}")
        return original_create(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    return socket.socket.connect, socket.socket.connect_ex


def test_offline_continuation_runs_only_governed_through_real_workflow_and_persists(
    monkeypatch, tmp_path,
):
    admin_url = os.getenv("ISOLATED_TEST_POSTGRES_ADMIN_URL")
    if not admin_url:
        pytest.skip("Set ISOLATED_TEST_POSTGRES_ADMIN_URL to an isolated-test PostgreSQL maintenance URL")
    copied = _copy_original_pair(tmp_path)
    manifest = json.loads((copied / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["provider_host"] = "127.0.0.1"
    (copied / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    from app.core import config as core_config
    from app.graph import workflow
    from app.llm import client as llm
    from app.tenancy import TenantContext

    fake_sdk = _OfflineSDK()
    safe_settings = core_config.get_settings().model_copy(update={
        "deepseek_api_key": "offline-placeholder",
        "deepseek_base_url": "https://127.0.0.1/v1",
        "llm_providers_json": "[]",
        "llm_timeout_seconds": 60.0,
    })
    monkeypatch.setattr(core_config, "get_settings", lambda: safe_settings)
    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: _OfflineClient(fake_sdk, **kwargs))
    workflow_calls = []
    actual_workflow = workflow.run_chat_workflow

    def capture_normal_entry(*args, **kwargs):
        workflow_calls.append({"message": args[0], "trace_id": kwargs.get("trace_id")})
        return actual_workflow(*args, **kwargs)

    monkeypatch.setattr(workflow, "run_chat_workflow", capture_normal_entry)
    models = {name: "qwen3.8-flash" for name in ("classify", "extract", "chat")}
    result = runner._run_governed_continuation(
        source_dir=copied,
        admin_url_text=admin_url,
        raw_settings=safe_settings,
        provider_host=manifest["provider_host"],
        models=models,
    )

    new_events = [row for row in runner.RunLedger(copied / "run_events.jsonl").events
                  if row.get("run_id") == runner.GOVERNED_RUN_ID]
    assert sum(row.get("event") == "started" for row in new_events) == 1
    assert not any(row.get("arm") == "basic_tool_loop" and row.get("event") == "started" for row in new_events)
    assert new_events[0]["event"] == "cost_reserved"
    assert new_events[-1]["event"] in {"completed", "failed"}
    assert len(workflow_calls) == 1
    assert workflow_calls[0]["message"] == runner.QUESTION
    assert workflow_calls[0]["trace_id"] == runner.GOVERNED_RUN_ID
    assert len(fake_sdk.requests) == sum(row.get("event") == "sdk_send_started" for row in new_events)
    assert manifest["provider_host"] == "127.0.0.1"
    assert result["run_id"] == runner.GOVERNED_RUN_ID
    assert (copied / "continuations" / runner.GOVERNED_RUN_ID / "continuation_summary.json").is_file()
    assert (copied / "continuations" / runner.GOVERNED_RUN_ID / "score.json").is_file()
    assert (copied / "continuations" / runner.GOVERNED_RUN_ID / "traces" / f"{runner.GOVERNED_RUN_ID}.json").is_file()
    assert (copied / "summary.json").read_bytes() == (SOURCE / "summary.json").read_bytes()
    assert (copied / "traces" / f"{runner.ORIGINAL_BASIC_RUN_ID}.json").read_bytes() == (
        SOURCE / "traces" / f"{runner.ORIGINAL_BASIC_RUN_ID}.json"
    ).read_bytes()
    reservation_rows = [json.loads(line) for line in (copied / "cost_reservations.jsonl").read_text().splitlines()]
    reserved = sum(float(row["reserved_cny"]) for row in reservation_rows if row.get("reserved_cny") is not None)
    assert reserved == pytest.approx(4.65742524)
