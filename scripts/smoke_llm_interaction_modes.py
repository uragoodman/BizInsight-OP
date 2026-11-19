#!/usr/bin/env python
"""Manual, opt-in smoke test for real LLM interaction modes.

This script sends a synthetic prompt to the configured primary provider. It is
not part of CI. Each mode runs in a child process so the parent can hard-stop a
single request after 120 seconds without waiting for a worker thread.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep
from urllib.parse import urlsplit
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SYSTEM = "你是一个谨慎的经营分析助手。只输出 JSON 对象，不要输出 Markdown。"
USER = (
    "两家店：A 的 GMV 下降 12% 但订单数上升 8%；"
    "B 的 GMV 下降 3% 但订单数下降 1%。"
    "哪家更可能需要检查客单价问题？"
    '只输出 {"choice":"A"|"B","reason":"不超过 60 字"}。'
)
MODES = (
    ("stream_fast", 256),
    ("stream_thinking", 1024),
    ("nonstream_fast", 256),
)
HARD_TIMEOUT_SECONDS = 120
RESULT_PREFIX = "SMOKE_RESULT_JSON="
STATUS_PATTERN = re.compile(r"\[ERROR/LLM_UPSTREAM\].*?status=(\d{3}|None)")
REQUEST_OPTION_KEYS = (
    "interaction_policy",
    "transport",
    "thinking",
    "connect_timeout_seconds",
    "idle_timeout_seconds",
    "total_timeout_seconds",
    "stream_options_include_usage",
    "stream_options_fallback",
    "transport_fallback",
)
ATTEMPT_KEYS = (
    "attempt_order",
    "provider",
    "model",
    "result",
    "processing_result",
    "exception_type",
    "duration_ms",
    "usage_status",
    "sdk_request_sent",
    "local_refusal_reason",
)


def _safe_request_options(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in REQUEST_OPTION_KEYS if key in value}


def _safe_attempts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        {key: attempt[key] for key in ATTEMPT_KEYS if key in attempt}
        for attempt in value
        if isinstance(attempt, dict)
    ]


def _status_from_captured_logs(value: str) -> int | None:
    statuses = STATUS_PATTERN.findall(value)
    for status in reversed(statuses):
        if status.isdigit():
            return int(status)
    return None


def _worker_result(mode: str, max_tokens: int) -> dict[str, Any]:
    from app.core.execution_context import activate_state, reset_state
    from app.core.config import get_settings
    from app.llm import client as llm
    from app.llm.interaction_policy import POLICIES

    settings = get_settings()
    if not settings.deepseek_api_key or not settings.deepseek_base_url:
        missing = []
        if not settings.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        if not settings.deepseek_base_url:
            missing.append("DEEPSEEK_BASE_URL")
        return {
            "mode": mode,
            "result_ok": False,
            "outcome": "configuration_error",
            "reason_code": "missing_required_provider_configuration",
            "missing_variables": missing,
            "status": None,
            "error_type": None,
            "attempts": [],
        }

    # The smoke must report a single real attempt and leave 429/timeouts for
    # the operator to inspect. Disable preset pre-token retry and provider
    # fallback in this process only; no application code or settings change.
    original_providers = llm._providers

    def primary_only_providers(*, call_config=None):
        return original_providers(call_config=call_config)[:1]

    llm._providers = primary_only_providers
    policy = replace(POLICIES[mode], retry_before_first_token=0)
    state = {
        "trace_id": f"smoke-{mode}",
        "current_agent": "manual_llm_interaction_smoke",
        "llm_calls": [],
        "llm_usage": {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0},
        "runtime_budget": {},
    }
    result = None
    error_type = None
    captured = io.StringIO()
    started = monotonic()
    token = activate_state(state)
    try:
        with redirect_stdout(captured), redirect_stderr(captured):
            result = llm.llm_extract(
                SYSTEM,
                USER,
                model="qwen3.8-flash",
                interaction_policy=policy,
                operation=f"smoke.{mode}",
                max_tokens=max_tokens,
            )
    except Exception as exc:  # report type only; never serialize provider text
        error_type = type(exc).__name__
    finally:
        reset_state(token)
    elapsed_ms = round((monotonic() - started) * 1000, 3)

    call = state.get("llm_calls", [])[-1] if state.get("llm_calls") else {}
    attempts = _safe_attempts(call.get("attempts"))
    outcome = str(call.get("outcome") or ("worker_error" if error_type else "missing_call_record"))
    choice = result.get("choice") if isinstance(result, dict) else None
    reason = result.get("reason") if isinstance(result, dict) else None
    result_ok = (
        isinstance(result, dict)
        and choice in {"A", "B"}
        and isinstance(reason, str)
        and len(reason) <= 60
    )
    failure = outcome != "success" or not result_ok or call.get("timeout_kind") is not None
    reason_code = call.get("not_sent_reason")
    if failure and not reason_code:
        if call.get("timeout_kind"):
            reason_code = call["timeout_kind"]
        elif outcome == "success" and not result_ok:
            reason_code = "invalid_result_schema"
        else:
            reason_code = outcome
    last_attempt_error = next(
        (attempt.get("exception_type") for attempt in reversed(attempts)
         if attempt.get("exception_type")),
        None,
    )
    return {
        "mode": mode,
        "max_tokens": max_tokens,
        "result_ok": bool(result_ok),
        "result_choice": choice,
        "result_reason_chars": len(reason) if isinstance(reason, str) else None,
        "outcome": outcome,
        "first_token_ms": call.get("first_token_ms"),
        "reasoning_chars": call.get("reasoning_chars", 0),
        "partial_output_chars": call.get("partial_output_chars", 0),
        "usage_status": call.get("usage_status", "unknown"),
        "timeout_kind": call.get("timeout_kind"),
        "elapsed_ms": call.get("duration_ms") or elapsed_ms,
        "request_options": _safe_request_options(call.get("request_options")),
        "status": _status_from_captured_logs(captured.getvalue()),
        "reason_code": reason_code,
        "attempts": attempts,
        "error_type": error_type or last_attempt_error,
        "pre_token_retries": 0,
        "provider_fallback": False,
    }


def _parse_worker_output(stdout: str, mode: str, max_tokens: int) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                value = json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def _git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _run_parent() -> int:
    from app.core.config import get_settings
    from app.llm.interaction_policy import POLICIES

    settings = get_settings()
    missing = []
    if not settings.deepseek_api_key:
        missing.append("DEEPSEEK_API_KEY")
    if not settings.deepseek_base_url:
        missing.append("DEEPSEEK_BASE_URL")
    if missing:
        print(json.dumps({
            "error": "required provider configuration is missing",
            "missing_variables": missing,
            "secret_values_printed": False,
        }, ensure_ascii=False))
        return 2

    host = urlsplit(settings.deepseek_base_url).hostname
    environment = {
        "provider_host": host,
        "model": "qwen3.8-flash",
        "head": _git_head(),
        "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
        "modes": [
            {
                "mode": mode,
                "max_tokens": max_tokens,
                "connect_timeout_seconds": POLICIES[mode].connect_timeout_seconds,
                "idle_timeout_seconds": POLICIES[mode].idle_timeout_seconds,
                "policy_total_timeout_seconds": POLICIES[mode].total_timeout_seconds,
            }
            for mode, max_tokens in MODES
        ],
        "pre_token_retries": 0,
        "provider_fallback": False,
    }
    print("Environment:")
    print(json.dumps(environment, ensure_ascii=False, indent=2))

    rows: list[dict[str, Any]] = []
    script_path = str(Path(__file__).resolve())
    for index, (mode, max_tokens) in enumerate(MODES):
        command = [
            sys.executable,
            script_path,
            "--allow-real-provider",
            "--_worker-mode",
            mode,
            "--_worker-max-tokens",
            str(max_tokens),
        ]
        started = monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=HARD_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            row = {
                "mode": mode,
                "max_tokens": max_tokens,
                "result_ok": False,
                "result_choice": None,
                "result_reason_chars": None,
                "outcome": "timeout",
                "first_token_ms": None,
                "reasoning_chars": 0,
                "partial_output_chars": 0,
                "usage_status": "unknown",
                "timeout_kind": "hard_guard_120s",
                "elapsed_ms": round((monotonic() - started) * 1000, 3),
                "request_options": {
                    "interaction_policy": mode,
                    "transport": POLICIES[mode].transport,
                    "thinking": POLICIES[mode].thinking,
                },
                "status": None,
                "reason_code": "smoke_hard_timeout_120s",
                "attempts": [],
                "error_type": "TimeoutExpired",
                "pre_token_retries": 0,
                "provider_fallback": False,
            }
        else:
            row = _parse_worker_output(completed.stdout, mode, max_tokens) or {
                "mode": mode,
                "max_tokens": max_tokens,
                "result_ok": False,
                "outcome": "worker_error",
                "first_token_ms": None,
                "reasoning_chars": 0,
                "usage_status": "unknown",
                "timeout_kind": None,
                "elapsed_ms": round((monotonic() - started) * 1000, 3),
                "request_options": {},
                "status": _status_from_captured_logs(
                    f"{completed.stdout}\n{completed.stderr}"
                ),
                "reason_code": "worker_result_missing",
                "attempts": [],
                "error_type": f"WorkerExit{completed.returncode}",
                "pre_token_retries": 0,
                "provider_fallback": False,
            }
            row["elapsed_ms"] = round((monotonic() - started) * 1000, 3)

        rows.append(row)
        print(f"Mode {mode}:")
        print(json.dumps(row, ensure_ascii=False, indent=2))
        mode_ok = (
            row.get("result_ok") is True
            and row.get("outcome") == "success"
            and row.get("timeout_kind") is None
        )
        if not mode_ok:
            break
        if index < len(MODES) - 1:
            sleep(1.0)

    print("SMOKE_SUMMARY_JSON=" + json.dumps({
        "environment": environment,
        "modes_run": rows,
        "all_passed": len(rows) == len(MODES) and all(
            row.get("result_ok") is True
            and row.get("outcome") == "success"
            and row.get("timeout_kind") is None
            for row in rows
        ),
    }, ensure_ascii=False))
    return 0 if len(rows) == len(MODES) and all(
        row.get("result_ok") is True
        and row.get("outcome") == "success"
        and row.get("timeout_kind") is None
        for row in rows
    ) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-real-provider",
        action="store_true",
        help="confirm that this manual smoke may send the synthetic prompt to the configured provider",
    )
    parser.add_argument("--_worker-mode", choices=[mode for mode, _ in MODES], help=argparse.SUPPRESS)
    parser.add_argument("--_worker-max-tokens", type=int, choices=[256, 1024], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.allow_real_provider:
        parser.error("real provider calls are disabled; pass --allow-real-provider explicitly")

    if args._worker_mode:
        try:
            result = _worker_result(args._worker_mode, args._worker_max_tokens or 256)
        except Exception as exc:  # do not emit raw exception text or provider content
            result = {
                "mode": args._worker_mode,
                "result_ok": False,
                "outcome": "worker_error",
                "first_token_ms": None,
                "reasoning_chars": 0,
                "usage_status": "unknown",
                "timeout_kind": None,
                "elapsed_ms": None,
                "request_options": {},
                "status": None,
                "reason_code": "worker_exception",
                "attempts": [],
                "error_type": type(exc).__name__,
            }
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
        return 0 if result.get("result_ok") and result.get("outcome") == "success" else 1
    return _run_parent()


if __name__ == "__main__":
    raise SystemExit(main())
