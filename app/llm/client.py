"""轻量 LLM 客户端封装。

项目使用 DeepSeek API（兼容 OpenAI SDK），依赖 `openai` 包。
配置来自 `.env`：`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`。

提供三个工具函数：
- `llm_classify()`：分类/路由，返回 allowed 中之一。
- `llm_extract()`：结构化 JSON 提取，返回 dict（可校验 Pydantic）。
- `llm_chat()`：通用对话/生成。
"""

from __future__ import annotations

import json
import ast
import hashlib
from math import ceil
import os
import re
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from time import monotonic, perf_counter
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlsplit
from uuid import uuid4

try:
    from openai import OpenAI
    from openai import (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
    import httpx
except ImportError:  # 允许无 LLM SDK 的本地/CI 环境走规则降级路径。
    OpenAI = None  # type: ignore[assignment,misc]
    APIConnectionError = APITimeoutError = InternalServerError = RateLimitError = Exception  # type: ignore[assignment,misc]
    httpx = None  # type: ignore[assignment]

from app.core.config import get_settings
from app.resilience import ExternalServiceError, execute_with_resilience
from app.observability import record_event
from app.core.execution_context import (
    TokenBudgetExceeded, active_state, current_graph_node, consume_llm_usage, record_token_budget_exceeded,
    token_budget_is_exhausted,
)
from app.usage.service import record_usage, quota_allows
from app.llm.routing import model_for, price_for, profile_for
from app.llm.call_config import LLMCallConfig
from app.llm.interaction_policy import InteractionPolicy, resolve_policy

# 可恢复的供应商故障：连接/超时/限流(429)/服务端 5xx → 允许切备用 provider。
# 业务/内容错误（4xx、schema 校验、预算超限、证据校验失败）不属于此类，绝不切换。
_RECOVERABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


class _StreamAttemptFailure(RuntimeError):
    """Safe wrapper carrying only whether a stream emitted any token."""

    def __init__(
        self, *, first_token_received: bool, total_deadline_exceeded: bool,
        idle_timeout: bool = False,
    ) -> None:
        self.first_token_received = first_token_received
        self.total_deadline_exceeded = total_deadline_exceeded
        self.idle_timeout = idle_timeout
        super().__init__("LLM streaming attempt failed.")


class _PolicyUnsupportedProvider(ValueError):
    def __init__(self, provider_id: str, reason: str) -> None:
        self.provider_id = provider_id
        super().__init__(reason)


class _AllPolicyProvidersUnsupported(RuntimeError):
    pass


class LocalSDKSendRefused(RuntimeError):
    """A local send gate rejected a request before it reached the SDK."""

    local_send_refused = True
    _SAFE_MESSAGES = {
        "missing_run_id": "SDK send refused: active workflow has no run ID.",
        "send_limit_reached": "SDK send refused before request: per-workflow send limit reached.",
        "model_mismatch": "SDK send refused: request model differs from frozen experiment model.",
        "completion_limit_mismatch": "SDK send refused: completion-token parameter differs from frozen config.",
        "input_estimate_limit": "SDK send refused: request exceeds frozen input-token estimate cap.",
        "critic_review_disabled": "SDK send refused: optional Critic LLM review is disabled in this arm.",
        "critic_thinking_mismatch": "SDK send refused: full-arm optional Critic review must disable thinking.",
        "token_budget_preflight": "SDK send refused: estimated request would exceed the remaining turn token budget.",
    }

    def __init__(self, reason_code: str) -> None:
        # Codes are identifiers only. Never attach prompts, credentials, or SDK
        # request bodies to this exception because callers persist the outcome.
        self.reason_code = str(reason_code)[:80]
        super().__init__(self._SAFE_MESSAGES.get(self.reason_code, "SDK send locally refused."))


def _local_send_refusal_reason(exc: BaseException) -> str | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "local_send_refused", False):
            return str(getattr(current, "reason_code", "local_send_refused"))[:80]
        current = getattr(current, "__cause__", None)
    return None


def _is_recoverable_provider_failure(exc: Exception) -> bool:
    """Recognize SDK transport failures after the resilience wrapper classifies them.

    ``execute_with_resilience`` wraps exhausted SDK errors in
    ``ExternalServiceError``. Provider failover still needs the original SDK
    classification; arbitrary application or response-processing errors must
    not trigger a backup provider.
    """
    if isinstance(exc, TokenBudgetExceeded):
        return False
    stream_failure = _find_stream_attempt_failure(exc)
    if stream_failure is not None:
        if stream_failure.total_deadline_exceeded:
            return False
        if stream_failure.idle_timeout:
            return True
        current: BaseException | None = stream_failure.__cause__
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, _RECOVERABLE_ERRORS):
                return True
            current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        return False
    if isinstance(exc, ExternalServiceError):
        if not exc.retryable:
            return False
        # The resilience layer emits its own timeout error when the worker
        # future exceeds the configured deadline; it has no SDK cause.
        if exc.service == "llm" and exc.code == "timeout":
            return True
        root: BaseException = exc
        while getattr(root, "__cause__", None) is not None:
            cause = getattr(root, "__cause__")
            if cause is None:
                break
            root = cause
        return root is not exc and isinstance(root, _RECOVERABLE_ERRORS)
    return isinstance(exc, _RECOVERABLE_ERRORS)


def _find_stream_attempt_failure(exc: BaseException) -> _StreamAttemptFailure | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _StreamAttemptFailure):
            return current
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return None


def _is_timeout_exception(exc: Exception) -> bool:
    """Recognize a timed verifier call without persisting provider details."""
    if isinstance(exc, ExternalServiceError) and exc.service == "llm" and exc.code == "timeout":
        return True
    stream_failure = _find_stream_attempt_failure(exc)
    if stream_failure is not None and (
        stream_failure.total_deadline_exceeded or stream_failure.idle_timeout
    ):
        return True
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError):
            return True
        if httpx is not None and isinstance(current, httpx.TimeoutException):
            return True
        if APITimeoutError is not Exception and isinstance(current, APITimeoutError):
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False


def _upstream_error_details(exc: Exception) -> dict[str, Any]:
    """Return a small, secret-safe description of an upstream failure.

    ``execute_with_resilience`` wraps provider exceptions in
    ``ExternalServiceError``.  Walk the cause chain so that HTTP status,
    provider request id, and the provider's structured error body are not
    lost, while deliberately excluding prompts, headers, and credentials.
    """
    root: BaseException = exc
    while getattr(root, "__cause__", None) is not None:
        cause = getattr(root, "__cause__")
        if cause is None:
            break
        root = cause

    response = getattr(root, "response", None)
    headers = getattr(response, "headers", None) or {}
    request_id = getattr(root, "request_id", None) or headers.get("x-request-id") or headers.get("request-id")
    status_code = getattr(root, "status_code", None) or getattr(response, "status_code", None)
    body = getattr(root, "body", None)
    if isinstance(body, dict):
        error = body.get("error") if isinstance(body.get("error"), dict) else body
        error_type = error.get("type") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        summary = f"{error_type}: {message}" if error_type and message else str(error)
    else:
        summary = str(root)
    summary = summary.replace("\\n", " ").strip()[:800]
    return {
        "exception_type": type(root).__name__,
        "status_code": int(status_code) if isinstance(status_code, int) else status_code,
        "request_id": str(request_id)[:200] if request_id else None,
        "summary": summary,
        "resilience_error": type(exc).__name__ if root is not exc else None,
        "resilience_code": getattr(exc, "code", None),
    }


def _record_upstream_error(operation: str, provider: _Provider, model: str, exc: Exception) -> None:
    details = _upstream_error_details(exc)
    attributes = {
        "llm.operation": operation,
        "llm.provider": provider.id,
        "gen_ai.request.model": model,
        "error.type": details["exception_type"],
        "error.status_code": details["status_code"],
        "error.request_id": details["request_id"],
        "error.summary": details["summary"],
        "llm.resilience_error": details["resilience_error"],
        "llm.resilience_code": details["resilience_code"],
    }
    record_event("llm.upstream_error", attributes)
    print(
        "[ERROR/LLM_UPSTREAM] "
        f"operation={operation} provider={provider.id} model={model} "
        f"status={details['status_code']!s} request_id={details['request_id']!s} "
        f"error={details['summary']}"
    )


@dataclass
class _Provider:
    """一个可调用的 LLM 提供商：id + OpenAI-compatible client + 按 operation 解析模型。"""
    id: str
    client: Any
    resolve: Callable[[str, str | None, bool], tuple[str, str]]  # (operation, explicit_model, quality) -> (model, profile)
    base_url: str | None = None


def _call_config(state: dict[str, Any] | None = None) -> LLMCallConfig:
    current = state if state is not None else active_state()
    stored = current.get("llm_call_config") if current is not None else None
    return LLMCallConfig.from_state(stored) if stored is not None else LLMCallConfig.normal(get_settings())


def _with_timeout_override(config: LLMCallConfig, timeout_seconds: float) -> LLMCallConfig:
    """Narrow one internal call deadline without changing request-wide policy."""
    timeout = float(timeout_seconds)
    if timeout <= 0:
        raise ValueError("Internal LLM timeout override must be positive.")
    effective_timeout = min(config.timeout_seconds, timeout)
    return replace(
        config,
        timeout_seconds=effective_timeout,
        sdk_max_retries=0,
        application_attempts=1,
        connect_retry_attempts=0,
        connect_timeout_seconds=None,
    )


def _sdk_timeout(config: LLMCallConfig) -> Any:
    """Build a split connect/total timeout only when a trusted experiment asks for it."""
    if config.connect_timeout_seconds is None:
        return config.timeout_seconds
    if httpx is None:
        raise RuntimeError("httpx is required for an explicit connection timeout policy")
    return httpx.Timeout(config.timeout_seconds, connect=config.connect_timeout_seconds)


def _is_transient_connection_establishment_error(exc: Exception) -> bool:
    """Retry only failures that occurred while establishing TLS/TCP, never read timeouts."""
    if httpx is None:
        return False
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (httpx.ConnectError, httpx.ConnectTimeout)):
            return True
        # A transport read/write/pool timeout is not a connection-establishment
        # failure, even if an upstream exception reuses handshake wording.
        if isinstance(current, httpx.TimeoutException):
            return False
        # OpenAI's timeout wrapper may preserve the TLS handshake timeout only
        # as a chained SSL/TimeoutError on some httpx versions.
        message = str(current).casefold()
        if "handshake operation timed out" in message:
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False


def _output_limit_kwargs(default_max_tokens: int, state: dict[str, Any] | None = None) -> dict[str, int]:
    """Use a total completion cap only for the explicitly isolated experiment mode."""
    config = _call_config(state)
    if config.mode == "provider_ab_experiment":
        # __post_init__ guarantees this is set for every experiment request.
        return {"max_completion_tokens": int(config.max_completion_tokens or 0)}
    return {"max_tokens": default_max_tokens}


def _primary_provider(settings, *, call_config: LLMCallConfig | None = None) -> _Provider | None:
    """主提供商：使用现有 deepseek 配置与 model_for 路由（fast/standard/quality）。"""
    if OpenAI is None or not settings.deepseek_api_key:
        return None
    effective = call_config or LLMCallConfig.normal(settings)
    return _Provider(
        id="primary",
        client=OpenAI(
            api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url,
            timeout=_sdk_timeout(effective), max_retries=effective.sdk_max_retries,
        ),
        resolve=lambda operation, explicit, quality: model_for(operation, explicit, quality=quality),
        base_url=str(settings.deepseek_base_url),
    )


def _backup_provider(settings, item: dict[str, Any], *, call_config: LLMCallConfig | None = None) -> _Provider | None:
    """从 LLM_PROVIDERS_JSON 构建一个备用提供商：key 从环境变量读，模型按 profile 映射。"""
    pid = str(item.get("id") or "")
    key = os.environ.get(str(item.get("api_key_env") or "")) or ""
    base_url = str(item.get("base_url") or "")
    if not pid or not key or not base_url or OpenAI is None:
        return None
    models = {str(k): str(v) for k, v in (item.get("models") or {}).items() if isinstance(v, str)}
    default_model = str(item.get("default_model") or settings.llm_model)

    def resolve(operation: str, explicit: str | None, quality: bool) -> tuple[str, str]:
        if explicit:
            return explicit, "explicit"
        profile = "quality" if quality else profile_for(operation)
        return models.get(profile) or default_model, profile

    effective = call_config or LLMCallConfig.normal(settings)
    return _Provider(
        id=pid,
        client=OpenAI(api_key=key, base_url=base_url, timeout=effective.timeout_seconds,
                      max_retries=effective.sdk_max_retries),
        resolve=resolve, base_url=base_url,
    )


def _providers(*, call_config: LLMCallConfig | None = None) -> list[_Provider]:
    """构建可调用 provider 列表：主 provider 在前，备用 provider（LLM_PROVIDERS_JSON）在后。

    备用列表为空时行为与旧版一致（只有主 provider，无回退）。
    """
    settings = get_settings()
    call_config = call_config or _call_config()
    result: list[_Provider] = []
    if call_config.provider_id in (None, "primary"):
        primary = _primary_provider(settings, call_config=call_config)
        if primary:
            result.append(primary)
    try:
        parsed = json.loads(settings.llm_providers_json or "[]")
        items = parsed if isinstance(parsed, list) else []
    except Exception:
        items = []
    for item in items:
        if isinstance(item, dict):
            item_id = str(item.get("id") or "")
            if call_config.provider_id is not None and item_id != call_config.provider_id:
                continue
            if call_config.provider_id is not None and not call_config.fallback_enabled:
                backup = _backup_provider(settings, item, call_config=call_config)
                return [backup] if backup else []
            backup = _backup_provider(settings, item, call_config=call_config)
            if backup:
                result.append(backup)
    if call_config.mode == "provider_ab_experiment" and call_config.provider_id == "primary":
        return result[:1]
    return result


def _get_client() -> Any | None:
    """返回主 provider 的 OpenAI client；无配置时返回 None（调用方降级）。

    保留该函数兼容旧调用；新逻辑用 `_providers()` 获取完整 provider 链。
    """
    providers = _providers()
    return providers[0].client if providers else None


def _usage_token(usage: Any, field: str) -> int | None:
    if usage is None:
        return None
    value = usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 and parsed == value else None


def _record_usage(response: Any, model: str, operation: str, profile: str = "standard", provider_id: str | None = None, fallback_from: str | None = None, *, call_record: dict[str, Any] | None = None) -> None:
    """从 LLM API 响应中提取 token 消耗、估算成本，并记录到可观测层。

    每次 LLM 调用完成后由 llm_chat/llm_classify/llm_extract 调用。
    执行三件事：
    1. 从响应的 usage 字段提取 prompt_tokens 和 completion_tokens
    2. 按配置的单价估算本次调用成本（美元）
    3. 调用 consume_llm_usage() 累加到节点级预算
    4. 调用 record_event("llm.call") 发送 OpenTelemetry 事件

    参数：
    - response：OpenAI-compatible API 的 ChatCompletion 响应对象
    - model：本次调用使用的模型名（如 "hy3"）
    """
    usage = getattr(response, "usage", None)
    actual_input_tokens = _usage_token(usage, "prompt_tokens")
    actual_output_tokens = _usage_token(usage, "completion_tokens")
    prompt_tokens = actual_input_tokens if actual_input_tokens is not None else 0
    completion_tokens = actual_output_tokens if actual_output_tokens is not None else 0
    if call_record is not None:
        for attempt in reversed(call_record.get("attempts", [])):
            if attempt.get("result") == "response_received":
                attempt["input_tokens"] = actual_input_tokens
                attempt["output_tokens"] = actual_output_tokens
                attempt["usage_status"] = (
                    "complete" if actual_input_tokens is not None and actual_output_tokens is not None
                    else "partial" if actual_input_tokens is not None or actual_output_tokens is not None
                    else "unknown"
                )
                if "usage_status" in call_record:
                    call_record["usage_status"] = attempt["usage_status"]
                break
        _refresh_llm_call_summary(active_state())
    settings = get_settings()
    input_price, output_price = price_for(profile)
    cost = (prompt_tokens / 1000 * input_price) + (completion_tokens / 1000 * output_price)
    if call_record is not None and actual_input_tokens is not None and actual_output_tokens is not None:
        call_record["estimated_cost"] = float(call_record.get("estimated_cost") or 0.0) + cost
    token_budget_error: TokenBudgetExceeded | None = None
    try:
        consume_llm_usage(prompt_tokens, completion_tokens, cost)
    except TokenBudgetExceeded as exc:
        # The response has already been received and its usage is billable.
        # Finish usage accounting and trace emission before stopping the caller.
        token_budget_error = exc
    state = active_state() or {}
    latest_attempt = next((item for item in reversed((call_record or {}).get("attempts", []))
                           if item.get("result") == "response_received"), None)
    _evaluation_journal_event(
        state, "usage_recorded",
        call_id=(call_record or {}).get("call_id"),
        attempt_id=(latest_attempt or {}).get("attempt_id"),
        operation=operation, provider=provider_id, model=model,
        input_tokens=actual_input_tokens, output_tokens=actual_output_tokens,
        usage_status=(latest_attempt or {}).get("usage_status", "unknown"),
    )
    if actual_input_tokens is not None and actual_output_tokens is not None:
        record_usage(
            tenant_id=str(state.get("tenant_id") or ""), user_id=str(state.get("user_id") or ""),
            trace_id=str(state.get("trace_id") or "") or None,
            chat_session_id=str(state.get("chat_session_id") or "") or None,
            task_id=str(state.get("task_id") or "") or None,
            evaluation_run_id=str(state.get("evaluation_run_id") or "") or None,
            route_profile=profile,
            operation=operation, model=model, input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens, estimated_cost=cost,
            provider_id=provider_id, fallback_from=fallback_from,
        )
    event_attributes = {
        "gen_ai.operation.name": operation, "gen_ai.request.model": model,
        "llm.estimated_cost": round(cost, 8),
        "gen_ai.response.finish_reasons": getattr(response.choices[0], "finish_reason", "unknown") if getattr(response, "choices", None) else "unknown",
    }
    if actual_input_tokens is not None:
        event_attributes["gen_ai.usage.input_tokens"] = actual_input_tokens
    if actual_output_tokens is not None:
        event_attributes["gen_ai.usage.output_tokens"] = actual_output_tokens
    if call_record is not None:
        event_attributes.update({"llm.call_id": call_record["call_id"], "llm.operation": operation})
    record_event("llm.call", event_attributes)
    if token_budget_error is not None:
        raise token_budget_error


def _initialize_interaction_observability(call: dict[str, Any]) -> None:
    """Attach safe streaming metrics to an opt-in interaction call only."""
    call.update({
        "first_token_ms": None,
        "reasoning_chars": 0,
        "partial_output_chars": 0,
        "usage_status": "unknown",
        "timeout_kind": None,
        "stream_options_fallback": False,
        "transport_fallback": False,
    })
    call["_interaction_started_monotonic"] = monotonic()


def _get_client() -> Any | None:
    """获取 OpenAI-compatible 客户端实例。

    如果未配置 API Key，返回 None，调用方自行降级。
    """

    settings = get_settings()
    if OpenAI is None or not settings.deepseek_api_key:
        return None
    call_config = _call_config()
    return OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout=call_config.timeout_seconds,
        max_retries=call_config.sdk_max_retries,
    )

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_reasoning_and_fences(text: str) -> str:
    """Drop reasoning blocks and markdown fences around a JSON payload."""
    text = _THINK_RE.sub("", text).strip()
    if text.startswith("<think>"):  # unclosed reasoning block
        text = text.rsplit("</think>", 1)[-1].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return text


def _loads_with_repair(text: str) -> Any | None:
    """Deterministic JSON repair before any schema validation.

    Handles the provider's degenerate modes in order: plain JSON, unescaped
    variant, and the escaped-string-key object (``{"{\"a\":1}": ""}``) whose
    single key carries the real payload.
    """
    candidates = [text]
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped.replace('\\"', '"'))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:  # noqa: BLE001 - try the next repair
            continue
    # Some reasoning providers emit a Python-literal-looking object with
    # single-quoted keys/values. literal_eval is deliberately used instead of
    # eval: it accepts only Python literals and cannot execute expressions.
    for candidate in (stripped, stripped.replace('"', "'")):
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (SyntaxError, ValueError):
            continue
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(parsed, dict) and len(parsed) == 1:
        (key, value), = parsed.items()
        if isinstance(key, str) and key.strip().startswith("{") and value in ("", None):
            for candidate in (key, key.replace('\\"', '"')):
                try:
                    inner = json.loads(candidate)
                    if isinstance(inner, dict):
                        return inner
                except Exception:  # noqa: BLE001
                    for literal_candidate in (candidate, candidate.replace('"', "'")):
                        try:
                            inner = ast.literal_eval(literal_candidate)
                            if isinstance(inner, dict):
                                return inner
                        except (SyntaxError, ValueError):
                            continue
    return None


def _quota_available() -> bool:
    state=active_state() or {}
    return not state.get("tenant_id") or quota_allows(str(state["tenant_id"]),str(state.get("user_id") or ""))


def _fallback_source(index: int) -> str | None:
    """索引 >0 表示该调用由备用 provider 完成，把回退源记录为 primary（供账本分桶）。"""
    return "primary" if index > 0 else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _refresh_llm_call_summary(state: dict[str, Any] | None) -> None:
    """Rebuild truthful task-level SDK/usage totals from request-local records."""
    if state is None:
        return
    calls = state.setdefault("llm_calls", [])
    attempts = [attempt for call in calls for attempt in call.get("attempts", [])]
    sent_attempts = [attempt for attempt in attempts if attempt.get("sdk_request_sent") is not False]
    requested = bool(sent_attempts)
    input_known = [attempt["input_tokens"] for attempt in sent_attempts if attempt.get("input_tokens") is not None]
    output_known = [attempt["output_tokens"] for attempt in sent_attempts if attempt.get("output_tokens") is not None]
    input_complete = all(attempt.get("input_tokens") is not None for attempt in sent_attempts)
    output_complete = all(attempt.get("output_tokens") is not None for attempt in sent_attempts)
    usage_complete = input_complete and output_complete
    input_total = sum(input_known) if input_complete else None
    output_total = sum(output_known) if output_complete else None
    state["llm_token_summary"] = {
        "request_status": "requested" if requested else ("not_sent" if calls else "no_model_request"),
        "logical_call_count": len(calls),
        "sdk_attempt_count": len(sent_attempts),
        "local_send_refusal_count": sum(attempt.get("result") == "locally_refused" for attempt in attempts),
        "successful_response_count": sum(attempt.get("result") == "response_received" for attempt in sent_attempts),
        "provider_exception_count": sum(attempt.get("result") == "provider_exception" for attempt in sent_attempts),
        "parse_failure_count": sum(attempt.get("processing_result") == "parse_failure" for attempt in sent_attempts),
        "known_input_tokens_subtotal": sum(input_known),
        "known_output_tokens_subtotal": sum(output_known),
        "known_tokens_subtotal": sum(input_known) + sum(output_known),
        "input_tokens": input_total if requested else 0,
        "output_tokens": output_total if requested else 0,
        "total_tokens": input_total + output_total if requested and usage_complete else 0 if not requested else None,
        "usage_complete": usage_complete,
        "unknown_usage_attempt_count": sum(attempt.get("usage_status") != "complete" for attempt in sent_attempts),
    }


def _evaluation_journal_event(
    state: dict[str, Any] | None, event: str, *, require_durable: bool = False, **fields: Any,
) -> bool:
    """Append a minimal evaluation event before/after the SDK boundary.

    An ``sdk_invocation_started`` event means the application entered the SDK
    call, not that a remote HTTP request definitely reached the provider.
    """
    sink = state.get("_evaluation_event_sink") if isinstance(state, dict) else None
    if not callable(sink):
        return True
    payload = {
        "event": event,
        "run_id": str(state.get("evaluation_run_id") or state.get("run_id") or ""),
        "trace_id": str(state.get("trace_id") or ""),
        "at": _utc_now(),
        **fields,
    }
    try:
        sink(payload)
        return True
    except Exception:
        if isinstance(state, dict):
            state["_evaluation_journal_failed"] = True
        if require_durable:
            raise LocalSDKSendRefused("evaluation_journal_unavailable")
        return False


def _begin_logical_call(
    operation: str,
    requested_model: str | None,
    *,
    request_options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    state = active_state()
    calls = state.setdefault("llm_calls", []) if state is not None else []
    call = {
        "call_id": f"llm_{uuid4().hex}",
        "call_sequence": len(calls) + 1,
        "trace_id": str(state.get("trace_id") or "") or None if state is not None else None,
        "node": (
            current_graph_node()
            or (str(state.get("current_agent") or "") or None if state is not None else None)
        ),
        "operation": operation,
        "requested_model": requested_model,
        # Only safe, explicitly supported request options are recorded. Never
        # store message bodies, credentials, or arbitrary SDK kwargs here.
        "request_options": dict(request_options or {}),
        "started_at": _utc_now(),
        "duration_ms": None,
        "outcome": "in_progress",
        "not_sent_reason": None,
        "attempts": [],
        "input_tokens_subtotal": 0,
        "output_tokens_subtotal": 0,
        "estimated_cost": None,
        "total_tokens": None,
        "usage_complete": True,
    }
    call["_started_perf"] = perf_counter()
    calls.append(call)
    if state is not None:
        _refresh_llm_call_summary(state)
    return call, state


def _begin_sdk_attempt(
    call: dict[str, Any], provider: _Provider, model: str,
    state: dict[str, Any] | None = None,
    call_config: LLMCallConfig | None = None,
) -> dict[str, Any]:
    attempt = {
        "attempt_id": f"llma_{uuid4().hex}",
        "call_id": call["call_id"],
        "attempt_order": len(call["attempts"]) + 1,
        "trace_id": call.get("trace_id"),
        "node": call.get("node"),
        "operation": call["operation"],
        "provider": provider.id,
        "model": model,
        "request_options": dict(call.get("request_options") or {}),
        "started_at": _utc_now(),
        "duration_ms": None,
        "result": "in_progress",
        "processing_result": "pending",
        "exception_type": None,
        "input_tokens": None,
        "output_tokens": None,
        "usage_status": "unknown",
        "_started_perf": perf_counter(),
    }
    call["attempts"].append(attempt)
    if state is not None:
        config = call_config or _call_config(state)
        models = [
            item.get("model")
            for logical_call in state.get("llm_calls", [])
            for item in logical_call.get("attempts", [])
            if item.get("model")
        ]
        state["llm_call_config_summary"] = config.trace_summary(models)
    return attempt


def _preflight_request_token_budget(
    call: dict[str, Any], state: dict[str, Any] | None, model: str,
    request_kwargs: dict[str, Any],
) -> None:
    """Refuse an LLM request before SDK entry when its bounded cost cannot fit.

    Only counts prompt metadata and token estimates in the call record. Prompt
    text itself is never persisted by this gate.
    """
    if not isinstance(state, dict):
        return
    budget = state.get("runtime_budget") or {}
    try:
        limit = int(budget.get("max_tokens", 0) or 0)
        usage = state.get("llm_usage") or {}
        consumed = int(usage.get("input_tokens", 0) or 0) + int(
            usage.get("output_tokens", 0) or 0
        ) + int(usage.get("preflight_reserved_tokens", 0) or 0)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return
    if limit <= 0:
        return
    if token_budget_is_exhausted(state):
        record_token_budget_exceeded(state)
        call["token_budget_preflight"] = {
            "allowed": False, "reason": "already_exhausted",
            "remaining_tokens": max(0, limit - consumed),
        }
        raise LocalSDKSendRefused("token_budget_preflight")

    messages = request_kwargs.get("messages")
    try:
        serialized_messages = json.dumps(
            messages, ensure_ascii=False, separators=(",", ":"), default=str,
        )
        from app.context.packer import estimate_tokens

        estimated_input = int(estimate_tokens(serialized_messages, model))
    except Exception:
        # The deterministic fallback still uses the same configured tokenizer
        # helper. If prompt material cannot be counted, fail closed for a
        # bounded request instead of sending an unbudgeted call.
        estimated_input = limit

    output_cap = request_kwargs.get("max_completion_tokens")
    if output_cap is None:
        output_cap = request_kwargs.get("max_tokens")
    if output_cap is None:
        settings = get_settings()
        output_cap = max(
            int(getattr(settings, "llm_extract_max_tokens", 0) or 0),
            int(getattr(settings, "llm_chat_max_tokens", 0) or 0),
        )
    try:
        output_reserve = max(0, int(output_cap or 0))
    except (TypeError, ValueError, OverflowError):
        output_reserve = limit
    settings = get_settings()
    try:
        margin_ratio = float(settings.llm_budget_preflight_margin_ratio)
    except (AttributeError, TypeError, ValueError, OverflowError):
        margin_ratio = 0.15
    margin_tokens = max(32, ceil(estimated_input * max(0.0, margin_ratio)))
    required = estimated_input + margin_tokens + output_reserve
    failed_attempt_reserve = max(0, int(call.get("unknown_usage_reserved_tokens", 0) or 0))
    remaining = max(0, limit - consumed - failed_attempt_reserve)
    preflight = {
        "allowed": required <= remaining,
        "estimated_input_tokens": estimated_input,
        "input_margin_tokens": margin_tokens,
        "reserved_output_tokens": output_reserve,
        "required_tokens": required,
        "remaining_tokens": remaining,
        "budget_limit_tokens": limit,
        "prior_failed_attempt_reserve_tokens": failed_attempt_reserve,
    }
    call["token_budget_preflight"] = preflight
    record_event("llm.token_budget_preflight", {
        "call_id": call.get("call_id"),
        "trace_id": call.get("trace_id"),
        "node": call.get("node"),
        "operation": call.get("operation"),
        **preflight,
    })
    if required > remaining:
        record_token_budget_exceeded(state)
        raise LocalSDKSendRefused("token_budget_preflight")


def _invoke_provider_sdk(call: dict[str, Any], state: dict[str, Any] | None,
                         provider: _Provider, model: str, operation: Callable[[], Any],
                         *, call_config: LLMCallConfig | None = None,
                         operation_manages_timeout: bool | None = None,
                         retry_if: Callable[[Exception], bool] | None = None,
                         request_kwargs: dict[str, Any] | None = None) -> Any:
    """Record one actual application-level SDK invocation (resilience may call this repeatedly)."""
    def invoke() -> Any:
        if isinstance(state, dict) and state.get("_evaluation_journal_failed"):
            raise LocalSDKSendRefused("evaluation_journal_unavailable")
        if request_kwargs is not None:
            _preflight_request_token_budget(call, state, model, request_kwargs)
        attempt = _begin_sdk_attempt(call, provider, model, state, call_config)
        preflight = call.get("token_budget_preflight") or {}
        if preflight.get("allowed") is True:
            attempt["preflight_required_tokens"] = int(preflight.get("required_tokens", 0) or 0)
        try:
            _evaluation_journal_event(
                state, "sdk_invocation_started", require_durable=True,
                call_id=call["call_id"], attempt_id=attempt["attempt_id"],
                node=call.get("node"), operation=call["operation"], provider=provider.id,
                model=model, attempt_order=attempt["attempt_order"],
                request_options=dict(call.get("request_options") or {}),
            )
            response = operation()
        except Exception as exc:
            local_refusal_reason = _local_send_refusal_reason(exc)
            is_local_refusal = local_refusal_reason is not None
            attempt["sdk_request_sent"] = not is_local_refusal
            attempt["result"] = "locally_refused" if is_local_refusal else "provider_exception"
            attempt["processing_result"] = "not_applicable"
            attempt["exception_type"] = type(exc).__name__
            if is_local_refusal:
                attempt["local_refusal_reason"] = local_refusal_reason
            elif attempt.get("preflight_required_tokens"):
                # No usage object is available for a provider exception. Keep
                # the conservative reservation across same-call retries and
                # later logical calls instead of treating the failed send as free.
                call["unknown_usage_reserved_tokens"] = int(
                    call.get("unknown_usage_reserved_tokens", 0) or 0
                ) + int(attempt["preflight_required_tokens"])
            attempt["duration_ms"] = round((perf_counter() - attempt.pop("_started_perf")) * 1000, 3)
            _refresh_llm_call_summary(state)
            record_event("llm.sdk_attempt", {
                "call_id": call["call_id"], "attempt_id": attempt["attempt_id"],
                "trace_id": call.get("trace_id"), "node": call.get("node"),
                "operation": call["operation"], "provider": provider.id, "model": model,
                "request_options": dict(call.get("request_options") or {}),
                "attempt_order": attempt["attempt_order"], "duration_ms": attempt["duration_ms"],
                "result": attempt["result"], "exception_type": attempt["exception_type"],
                "sdk_request_sent": attempt["sdk_request_sent"],
                "local_refusal_reason": local_refusal_reason,
            })
            _evaluation_journal_event(
                state, "sdk_invocation_finished", call_id=call["call_id"],
                attempt_id=attempt["attempt_id"], node=call.get("node"),
                operation=call["operation"], provider=provider.id, model=model,
                attempt_order=attempt["attempt_order"], result=attempt["result"],
                sdk_request_sent=attempt["sdk_request_sent"],
                local_refusal_reason=local_refusal_reason,
                exception_type=attempt["exception_type"], duration_ms=attempt["duration_ms"],
            )
            raise
        attempt["sdk_request_sent"] = True
        attempt["result"] = "response_received"
        attempt["duration_ms"] = round((perf_counter() - attempt.pop("_started_perf")) * 1000, 3)
        _refresh_llm_call_summary(state)
        record_event("llm.sdk_attempt", {
            "call_id": call["call_id"], "attempt_id": attempt["attempt_id"],
            "trace_id": call.get("trace_id"), "node": call.get("node"),
            "operation": call["operation"], "provider": provider.id, "model": model,
            "request_options": dict(call.get("request_options") or {}),
            "attempt_order": attempt["attempt_order"], "duration_ms": attempt["duration_ms"],
            "result": attempt["result"], "sdk_request_sent": True,
        })
        _evaluation_journal_event(
            state, "sdk_invocation_finished", call_id=call["call_id"],
            attempt_id=attempt["attempt_id"], node=call.get("node"),
            operation=call["operation"], provider=provider.id, model=model,
            attempt_order=attempt["attempt_order"], result="response_received",
            sdk_request_sent=True, duration_ms=attempt["duration_ms"],
        )
        return response
    config = call_config or _call_config(state)
    manages_timeout = (
        config.mode == "provider_ab_experiment"
        if operation_manages_timeout is None else operation_manages_timeout
    )
    if config.mode == "provider_ab_experiment":
        # The frozen experiment owns this boundary and cannot opt into a second
        # timeout owner through a normal interaction policy.
        manages_timeout = True
    effective_retry_if = retry_if
    if effective_retry_if is None and config.connect_retry_attempts:
        effective_retry_if = _is_transient_connection_establishment_error
    return execute_with_resilience(
        invoke, "llm", attempts=config.application_attempts + config.connect_retry_attempts,
        timeout_seconds=config.timeout_seconds,
        backoff_seconds=0.5 if config.connect_retry_attempts else 0.05,
        retry_if=effective_retry_if,
        **({"operation_manages_timeout": True} if manages_timeout else {}),
    )


def _interaction_call_config(
    state: dict[str, Any] | None, policy: InteractionPolicy,
) -> LLMCallConfig:
    """Build a bounded, explicit policy without mutating frozen experiments."""
    config = _call_config(state)
    if config.mode != "normal":
        raise ValueError(
            "LLM interaction policies cannot override a frozen provider experiment."
        )
    return replace(
        config,
        sdk_max_retries=0,
        application_attempts=1,
        timeout_seconds=policy.total_timeout_seconds,
        connect_retry_attempts=0,
        connect_timeout_seconds=None,
    )


def _interaction_timeout(
    policy: InteractionPolicy, remaining_seconds: float, *, streaming: bool,
) -> Any:
    """Create HTTPX connect/read timeouts; streaming read is an idle limit."""
    if httpx is None:
        raise RuntimeError("httpx is required for an explicit LLM interaction policy.")
    remaining = max(0.001, min(policy.total_timeout_seconds, remaining_seconds))
    connect = min(policy.connect_timeout_seconds, remaining)
    read = min(float(policy.idle_timeout_seconds), remaining) if streaming else remaining
    return httpx.Timeout(
        remaining,
        connect=connect,
        read=read,
        write=remaining,
        pool=connect,
    )


def _interaction_thinking_kwargs(
    policy: InteractionPolicy, model: str, provider: _Provider,
) -> dict[str, Any]:
    if policy.thinking == "provider_default":
        return {}
    # enable_thinking is a DashScope/Qwen request extension. Do not silently
    # claim that a different model/provider honors the same interaction mode.
    if model != "qwen3.8-flash":
        raise _PolicyUnsupportedProvider(
            provider.id,
            "Explicit thinking modes currently require the qwen3.8-flash model."
        )
    base_url = provider.base_url or str(getattr(provider.client, "base_url", ""))
    parsed_url = urlsplit(base_url)
    if (
        parsed_url.hostname != "dashscope.aliyuncs.com"
        or not parsed_url.path.rstrip("/").endswith("/compatible-mode/v1")
    ):
        raise _PolicyUnsupportedProvider(
            provider.id,
            "Explicit thinking modes currently require the DashScope compatible-mode endpoint."
        )
    return {"extra_body": {"enable_thinking": policy.thinking == "on"}}


def _close_stream(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _close_provider_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _call_provider_nonstream_with_deadline(
    provider: _Provider,
    request_kwargs: dict[str, Any],
    deadline: float,
    call_record: dict[str, Any],
) -> Any:
    """Close the per-call client at the total deadline, including body reads."""
    remaining = deadline - monotonic()
    if remaining <= 0:
        call_record["timeout_kind"] = "total"
        raise TimeoutError("LLM interaction total deadline exceeded.")
    timeout_event = threading.Event()

    def stop_at_deadline() -> None:
        timeout_event.set()
        call_record["timeout_kind"] = "total"
        _close_provider_client(provider.client)

    timer = threading.Timer(remaining, stop_at_deadline)
    timer.daemon = True
    timer.start()
    try:
        response = provider.client.chat.completions.create(**request_kwargs)
        if timeout_event.is_set() or monotonic() >= deadline:
            call_record["timeout_kind"] = "total"
            raise TimeoutError("LLM interaction total deadline exceeded.")
        return response
    except Exception as exc:
        if isinstance(exc, TimeoutError):
            if timeout_event.is_set() or monotonic() >= deadline:
                call_record["timeout_kind"] = "total"
            raise
        if timeout_event.is_set() or monotonic() >= deadline:
            call_record["timeout_kind"] = "total"
            raise TimeoutError("LLM interaction total deadline exceeded.") from exc
        raise
    finally:
        timer.cancel()


def _consume_provider_stream(
    provider: _Provider,
    request_kwargs: dict[str, Any],
    policy: InteractionPolicy,
    deadline: float,
    token_state: dict[str, bool],
    call_record: dict[str, Any],
    *,
    include_usage: bool,
) -> Any:
    """Consume one SSE response, retaining answer text and usage only."""
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise _StreamAttemptFailure(
            first_token_received=False, total_deadline_exceeded=True,
        )
    kwargs = dict(request_kwargs)
    kwargs.update({
        "stream": True,
        "timeout": _interaction_timeout(policy, remaining, streaming=True),
    })
    if include_usage:
        kwargs["stream_options"] = {"include_usage": True}
    stop_event = threading.Event()
    idle_event = threading.Event()
    watchdog_stop = threading.Event()
    stream_lock = threading.Lock()
    stream_holder: dict[str, Any] = {"stream": None}
    last_event_lock = threading.Lock()
    last_event_monotonic = [monotonic()]

    def close_active_stream() -> bool:
        with stream_lock:
            active_stream = stream_holder["stream"]
        if active_stream is not None:
            _close_stream(active_stream)
            return True
        return False

    poll_interval = min(
        0.25,
        max(0.005, min(float(policy.idle_timeout_seconds), policy.total_timeout_seconds) / 4),
    )

    def watchdog() -> None:
        while not watchdog_stop.wait(poll_interval):
            now = monotonic()
            if now >= deadline:
                call_record["timeout_kind"] = "total"
                stop_event.set()
                close_active_stream()
                _close_provider_client(provider.client)
                return
            with last_event_lock:
                idle_for = now - last_event_monotonic[0]
            if idle_for >= float(policy.idle_timeout_seconds):
                call_record["timeout_kind"] = "idle"
                idle_event.set()
                if not close_active_stream():
                    _close_provider_client(provider.client)
                return

    watchdog_thread = threading.Thread(
        target=watchdog, name="llm-stream-watchdog", daemon=True,
    )
    stream: Any | None = None
    content_parts: list[str] = []
    usage: Any | None = None
    finish_reason: str | None = None
    try:
        # One watchdog covers create(), idle gaps between complete SSE events,
        # and the total deadline. It closes the client while headers are still
        # being read and closes the stream after the handle becomes available.
        watchdog_thread.start()
        stream = provider.client.chat.completions.create(**kwargs)
        with stream_lock:
            stream_holder["stream"] = stream
        if stop_event.is_set() or idle_event.is_set() or monotonic() >= deadline:
            raise TimeoutError("LLM interaction total deadline exceeded.")

        for chunk in stream:
            if stop_event.is_set() or idle_event.is_set() or monotonic() >= deadline:
                raise TimeoutError("LLM interaction total deadline exceeded.")
            received_at = monotonic()
            with last_event_lock:
                last_event_monotonic[0] = received_at
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            for choice in getattr(chunk, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                reasoning = getattr(delta, "reasoning_content", None)
                content = getattr(delta, "content", None)
                if isinstance(reasoning, str) and reasoning:
                    token_state["first_token_received"] = True
                    if call_record.get("first_token_ms") is None:
                        call_record["first_token_ms"] = round(
                            (received_at - call_record["_interaction_started_monotonic"]) * 1000,
                            3,
                        )
                    call_record["reasoning_chars"] += len(reasoning)
                if isinstance(content, str) and content:
                    token_state["first_token_received"] = True
                    if call_record.get("first_token_ms") is None:
                        call_record["first_token_ms"] = round(
                            (received_at - call_record["_interaction_started_monotonic"]) * 1000,
                            3,
                        )
                    content_parts.append(content)
                    call_record["partial_output_chars"] += len(content)
                chunk_finish = getattr(choice, "finish_reason", None)
                if chunk_finish is not None:
                    finish_reason = str(chunk_finish)
        if stop_event.is_set() or idle_event.is_set() or monotonic() >= deadline:
            raise TimeoutError("LLM interaction total deadline exceeded.")
        call_record["partial_output_chars"] = 0
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="".join(content_parts)),
                finish_reason=finish_reason or "unknown",
            )],
            usage=usage,
        )
    except Exception as exc:
        if isinstance(exc, _StreamAttemptFailure):
            raise
        deadline_exceeded = stop_event.is_set() or monotonic() >= deadline
        if deadline_exceeded:
            call_record["timeout_kind"] = "total"
            stop_event.set()
            if stream is not None:
                _close_stream(stream)
            _close_provider_client(provider.client)
        elif idle_event.is_set() or (
            _is_timeout_exception(exc)
            and not _is_transient_connection_establishment_error(exc)
        ):
            call_record["timeout_kind"] = "idle"
        failure = _StreamAttemptFailure(
            first_token_received=token_state["first_token_received"],
            total_deadline_exceeded=deadline_exceeded,
            idle_timeout=call_record.get("timeout_kind") == "idle",
        )
        raise failure from exc
    finally:
        watchdog_stop.set()
        if watchdog_thread.ident is not None:
            watchdog_thread.join(timeout=1.0)
        if stream is not None:
            _close_stream(stream)


def _provider_error_status_and_text(exc: BaseException) -> tuple[int | None, str]:
    status: int | None = None
    fragments: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        candidate = getattr(current, "status_code", None)
        response = getattr(current, "response", None)
        if candidate is None:
            candidate = getattr(response, "status_code", None)
        if status is None and isinstance(candidate, int):
            status = candidate
        fragments.append(str(current))
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            fragments.append(json.dumps(body, ensure_ascii=False, default=str))
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return status, " ".join(fragments).casefold()


def _is_stream_options_unsupported(exc: BaseException) -> bool:
    status, message = _provider_error_status_and_text(exc)
    has_option = "stream_options" in message or "include_usage" in message
    unsupported = any(word in message for word in (
        "unsupported", "not supported", "does not support", "unknown parameter",
        "unrecognized", "invalid parameter", "not allowed", "unexpected",
    ))
    return status in {400, 422} and has_option and unsupported


def _is_stream_transport_unsupported(exc: BaseException) -> bool:
    status, message = _provider_error_status_and_text(exc)
    if status not in {400, 404, 405} or "stream_options" in message or "include_usage" in message:
        return False
    return "stream" in message and any(word in message for word in (
        "unsupported", "not supported", "does not support", "not available",
        "not allowed", "unexpected",
    ))


def _invoke_interaction_policy(
    call: dict[str, Any],
    state: dict[str, Any] | None,
    providers: list[_Provider],
    call_config: LLMCallConfig,
    policy: InteractionPolicy,
    operation: str,
    model: str | None,
    quality: bool,
    request_kwargs: dict[str, Any],
) -> tuple[Any, _Provider, str, str, int]:
    """Invoke one opt-in policy, retrying a stream only before its first token."""
    deadline = monotonic() + policy.total_timeout_seconds
    unsupported_provider_ids: list[str] = []
    attempted_provider = False
    last_provider_error: Exception | None = None
    provider_candidates = providers if policy.provider_fallback_enabled else providers[:1]
    for provider_index, provider in enumerate(provider_candidates):
        selected, profile = provider.resolve(operation, model, quality)
        call["profile"] = profile
        try:
            thinking_kwargs = _interaction_thinking_kwargs(policy, selected, provider)
        except _PolicyUnsupportedProvider:
            unsupported_provider_ids.append(provider.id)
            call["policy_unsupported_provider"] = provider.id
            call["policy_unsupported_providers"] = list(unsupported_provider_ids)
            call["request_options"]["policy_unsupported_provider"] = provider.id
            continue
        attempted_provider = True
        provider_kwargs = {**request_kwargs, "model": selected, **thinking_kwargs}
        retries = policy.retry_before_first_token if policy.transport == "stream" else 0
        retry_index = 0
        stream_options_enabled = policy.transport == "stream"
        stream_options_fallback_used = False
        transport_fallback_used = False
        while retry_index <= retries:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise ExternalServiceError(
                    "llm", "timeout", "LLM interaction total deadline exceeded.",
                )
            attempt_config = replace(call_config, timeout_seconds=remaining)
            token_state = {"first_token_received": False}
            try:
                if policy.transport == "stream":
                    response = _invoke_provider_sdk(
                        call, state, provider, selected,
                        lambda: _consume_provider_stream(
                            provider, provider_kwargs, policy, deadline, token_state,
                            call, include_usage=stream_options_enabled,
                        ),
                        call_config=attempt_config,
                        operation_manages_timeout=True,
                        request_kwargs=provider_kwargs,
                    )
                else:
                    nonstream_kwargs = {
                        **provider_kwargs,
                        "stream": False,
                        "timeout": _interaction_timeout(policy, remaining, streaming=False),
                    }
                    response = _invoke_provider_sdk(
                        call, state, provider, selected,
                        lambda: _call_provider_nonstream_with_deadline(
                            provider, nonstream_kwargs, deadline, call,
                        ),
                        call_config=attempt_config,
                        operation_manages_timeout=True,
                        request_kwargs=nonstream_kwargs,
                    )
                return response, provider, selected, profile, provider_index
            except Exception as exc:
                if _local_send_refusal_reason(exc) is not None:
                    raise

                # A 400 that specifically rejects stream_options/include_usage
                # gets one same-provider retry without that optional field.
                if (
                    policy.transport == "stream"
                    and stream_options_enabled
                    and not stream_options_fallback_used
                    and _is_stream_options_unsupported(exc)
                    and monotonic() < deadline
                ):
                    stream_options_fallback_used = True
                    stream_options_enabled = False
                    call["stream_options_fallback"] = True
                    call["request_options"]["stream_options_fallback"] = True
                    continue

                stream_failure = _find_stream_attempt_failure(exc)
                first_token_received = bool(
                    stream_failure and stream_failure.first_token_received
                )

                # A provider may reject SSE entirely. Fall back once on that
                # provider to the same request over its non-stream endpoint.
                if (
                    policy.transport == "stream"
                    and not transport_fallback_used
                    and not first_token_received
                    and _is_stream_transport_unsupported(exc)
                    and monotonic() < deadline
                ):
                    transport_fallback_used = True
                    call["transport_fallback"] = True
                    call["request_options"]["transport_fallback"] = "nonstream"
                    remaining = deadline - monotonic()
                    nonstream_kwargs = {
                        **provider_kwargs,
                        "stream": False,
                        "timeout": _interaction_timeout(policy, remaining, streaming=False),
                    }
                    try:
                        response = _invoke_provider_sdk(
                            call, state, provider, selected,
                            lambda: _call_provider_nonstream_with_deadline(
                                provider, nonstream_kwargs, deadline, call,
                            ),
                            call_config=replace(attempt_config, timeout_seconds=remaining),
                            operation_manages_timeout=True,
                            request_kwargs=nonstream_kwargs,
                        )
                    except Exception as fallback_exc:
                        if _local_send_refusal_reason(fallback_exc) is not None:
                            raise
                        exc = fallback_exc
                        stream_failure = None
                    else:
                        return response, provider, selected, profile, provider_index

                recoverable = _is_recoverable_provider_failure(exc)
                if (
                    policy.transport == "stream"
                    and recoverable
                    and not first_token_received
                    and stream_failure is not None
                    and not stream_failure.total_deadline_exceeded
                    and retry_index < retries
                    and monotonic() < deadline
                ):
                    retry_index += 1
                    continue
                _record_upstream_error(operation, provider, selected, exc)
                may_fallback = (
                    recoverable
                    and (policy.transport != "stream" or not first_token_received)
                    and provider_index + 1 < len(provider_candidates)
                    and monotonic() < deadline
                )
                if may_fallback:
                    last_provider_error = exc
                    break
                raise
    if unsupported_provider_ids and not attempted_provider:
        raise _AllPolicyProvidersUnsupported(
            "No configured provider supports this explicit thinking policy."
        )
    if last_provider_error is not None:
        raise last_provider_error
    raise ExternalServiceError("llm", "external_call_failed", "No interaction policy provider succeeded.")


def _llm_extract_with_interaction_policy(
    system_prompt: str,
    user_message: str,
    policy: InteractionPolicy,
    *,
    model: str | None,
    temperature: float,
    quality: bool,
    operation: str,
    max_tokens: int | None,
    response_format: dict[str, Any] | None,
) -> dict[str, Any] | None:
    state = active_state()
    call_config = _interaction_call_config(state, policy)
    request_options = {
        "interaction_policy": policy.name,
        "transport": policy.transport,
        "thinking": policy.thinking,
        "provider_fallback_enabled": policy.provider_fallback_enabled,
        "connect_timeout_seconds": policy.connect_timeout_seconds,
        "idle_timeout_seconds": policy.idle_timeout_seconds,
        "total_timeout_seconds": policy.total_timeout_seconds,
        "stream_options_include_usage": policy.transport == "stream",
    }
    if response_format is not None:
        request_options["response_format_type"] = str(response_format.get("type") or "unknown")
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict) and isinstance(json_schema.get("name"), str):
            request_options["response_schema_name"] = json_schema["name"][:100]
    call, state = _begin_logical_call(operation, model, request_options=request_options)
    _initialize_interaction_observability(call)
    if _stop_if_token_budget_exhausted(call, state):
        return None
    if not _quota_available():
        _finish_logical_call(call, state, "not_sent", "quota_blocked")
        return None
    providers = _providers(call_config=call_config)
    if not providers:
        _finish_logical_call(call, state, "not_sent", "no_provider_configured")
        return None

    full_prompt = (
        f"{system_prompt}\n\n"
        "请只输出 JSON 对象，不要包含 ```json 标记或额外说明。"
    )
    output_limits = _output_limit_kwargs(
        int(max_tokens if max_tokens is not None else get_settings().llm_extract_max_tokens), state,
    )
    if max_tokens is not None and "max_tokens" in output_limits:
        # Frozen provider experiments own max_completion_tokens; a normal
        # caller's max_tokens must not silently shrink the experiment budget.
        output_limits["max_tokens"] = min(output_limits["max_tokens"], int(max_tokens))
    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        **output_limits,
    }
    if response_format is not None:
        request_kwargs["response_format"] = response_format
    try:
        response, provider, selected, profile, provider_index = _invoke_interaction_policy(
            call, state, providers, call_config, policy, operation, model, quality,
            request_kwargs,
        )
        _record_usage(
            response, selected, operation, profile, provider_id=provider.id,
            fallback_from=_fallback_source(provider_index), call_record=call,
        )
        content = response.choices[0].message.content if response.choices else ""
        raw_output = content if isinstance(content, str) else ""
        text = raw_output.strip()
        if not text:
            _capture_semantic_eval_parse_failure_output(call, state, raw_output)
            _mark_processing_result(call, "empty_response")
            _finish_logical_call(call, state, "empty_response")
            return None
        repaired = _loads_with_repair(_strip_reasoning_and_fences(text))
        if repaired is None:
            _capture_semantic_eval_parse_failure_output(call, state, raw_output)
            _mark_processing_result(call, "parse_failure")
            _finish_logical_call(call, state, "parse_failure")
            return None
        _mark_processing_result(call, "success")
        _finish_logical_call(call, state, "success")
        return repaired
    except Exception as exc:
        if isinstance(exc, _AllPolicyProvidersUnsupported):
            _finish_logical_call(call, state, "configuration_error", "policy_unsupported_provider")
            return None
        if _is_timeout_exception(exc):
            if (
                isinstance(exc, ExternalServiceError)
                and exc.service == "llm" and exc.code == "timeout"
            ):
                call["timeout_kind"] = "total"
            _finish_logical_call(call, state, "timeout")
            return None
        local_refusal_reason = _local_send_refusal_reason(exc)
        if local_refusal_reason is not None:
            _finish_logical_call(call, state, "not_sent", local_refusal_reason)
            return None
        if isinstance(exc, ValueError) and not call.get("attempts"):
            _finish_logical_call(call, state, "configuration_error")
            raise
        if _latest_attempt_received_response(call):
            _mark_processing_result(call, "processing_error")
            _finish_logical_call(call, state, "processing_error")
        else:
            _finish_logical_call(call, state, "provider_error")
        print(f"[DEBUG/LLM] {operation} interaction-policy call FAILED: {type(exc).__name__}")
        return None


def _llm_chat_with_interaction_policy(
    system_prompt: str,
    user_message: str,
    policy: InteractionPolicy,
    *,
    model: str | None,
    temperature: float,
    max_tokens: int | None,
    operation: str,
    quality: bool,
    allowed: list[str] | None = None,
) -> str:
    state = active_state()
    call_config = _interaction_call_config(state, policy)
    request_options = {
        "interaction_policy": policy.name,
        "transport": policy.transport,
        "thinking": policy.thinking,
        "provider_fallback_enabled": policy.provider_fallback_enabled,
        "connect_timeout_seconds": policy.connect_timeout_seconds,
        "idle_timeout_seconds": policy.idle_timeout_seconds,
        "total_timeout_seconds": policy.total_timeout_seconds,
        "stream_options_include_usage": policy.transport == "stream",
    }
    call, state = _begin_logical_call(operation, model, request_options=request_options)
    _initialize_interaction_observability(call)
    if _stop_if_token_budget_exhausted(call, state):
        return ""
    if not _quota_available():
        _finish_logical_call(call, state, "not_sent", "quota_blocked")
        return ""
    providers = _providers(call_config=call_config)
    if not providers:
        _finish_logical_call(call, state, "not_sent", "no_provider_configured")
        return ""
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
    }
    request_kwargs.update(_output_limit_kwargs(
        max_tokens or get_settings().llm_chat_max_tokens, state,
    ))
    try:
        response, provider, selected, profile, provider_index = _invoke_interaction_policy(
            call, state, providers, call_config, policy, operation, model, quality,
            request_kwargs,
        )
        _record_usage(
            response, selected, operation, profile, provider_id=provider.id,
            fallback_from=_fallback_source(provider_index), call_record=call,
        )
        reply = response.choices[0].message.content.strip() if response.choices else ""
        input_tokens = call["attempts"][-1].get("input_tokens") if call.get("attempts") else None
        output_tokens = call["attempts"][-1].get("output_tokens") if call.get("attempts") else None
        print(
            f"[DEBUG/LLM] {operation} → len(reply)={len(reply)} token_in={input_tokens!s} "
            f"token_out={output_tokens!s} provider={provider.id}"
        )
        accepted = allowed is None or reply in allowed
        result = (
            "success" if accepted and reply else
            "parse_failure" if not accepted else
            "empty_response"
        )
        _mark_processing_result(call, result)
        _finish_logical_call(call, state, result)
        return reply if accepted else ""
    except Exception as exc:
        if isinstance(exc, _AllPolicyProvidersUnsupported):
            _finish_logical_call(call, state, "configuration_error", "policy_unsupported_provider")
            return ""
        if _is_timeout_exception(exc):
            if (
                isinstance(exc, ExternalServiceError)
                and exc.service == "llm" and exc.code == "timeout"
            ):
                call["timeout_kind"] = "total"
            _finish_logical_call(call, state, "timeout")
            return ""
        local_refusal_reason = _local_send_refusal_reason(exc)
        if local_refusal_reason is not None:
            _finish_logical_call(call, state, "not_sent", local_refusal_reason)
            return ""
        if isinstance(exc, ValueError) and not call.get("attempts"):
            _finish_logical_call(call, state, "configuration_error")
            raise
        if _latest_attempt_received_response(call):
            _mark_processing_result(call, "processing_error")
            _finish_logical_call(call, state, "processing_error")
        else:
            _finish_logical_call(call, state, "provider_error")
        print(f"[DEBUG/LLM] {operation} interaction-policy call FAILED: {type(exc).__name__}")
        return ""


def _finish_logical_call(call: dict[str, Any], state: dict[str, Any] | None,
                         outcome: str, not_sent_reason: str | None = None) -> None:
    if isinstance(state, dict) and not call.get("_preflight_reserved_accounted"):
        unknown_reserved = max(0, int(call.get("unknown_usage_reserved_tokens", 0) or 0))
        if unknown_reserved:
            usage = state.setdefault("llm_usage", {
                "input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0,
            })
            usage["preflight_reserved_tokens"] = int(
                usage.get("preflight_reserved_tokens", 0) or 0
            ) + unknown_reserved
            call["preflight_reserved_tokens_committed"] = unknown_reserved
            budget_limit = int((state.get("runtime_budget") or {}).get("max_tokens", 0) or 0)
            accounted_tokens = (
                int(usage.get("input_tokens", 0) or 0)
                + int(usage.get("output_tokens", 0) or 0)
                + int(usage.get("preflight_reserved_tokens", 0) or 0)
            )
            if budget_limit > 0 and accounted_tokens >= budget_limit:
                record_token_budget_exceeded(state)
        call["_preflight_reserved_accounted"] = True
    call["outcome"] = outcome
    call["not_sent_reason"] = not_sent_reason
    call["duration_ms"] = round((perf_counter() - call.pop("_started_perf")) * 1000, 3)
    call.pop("_interaction_started_monotonic", None)
    attempts = call.get("attempts", [])
    sent_attempts = [item for item in attempts if item.get("sdk_request_sent") is not False]
    known_input = [item["input_tokens"] for item in sent_attempts if item.get("input_tokens") is not None]
    known_output = [item["output_tokens"] for item in sent_attempts if item.get("output_tokens") is not None]
    call["input_tokens_subtotal"] = sum(known_input)
    call["output_tokens_subtotal"] = sum(known_output)
    call["usage_complete"] = all(item.get("usage_status") == "complete" for item in sent_attempts)
    if "usage_status" in call:
        statuses = [item.get("usage_status", "unknown") for item in sent_attempts]
        if statuses and all(status == "complete" for status in statuses):
            call["usage_status"] = "complete"
        elif any(status in {"complete", "partial"} for status in statuses):
            call["usage_status"] = "partial"
        else:
            call["usage_status"] = "unknown"
    call["total_tokens"] = (
        sum(known_input) + sum(known_output)
        if sent_attempts and call["usage_complete"] else 0 if not sent_attempts else None
    )
    _refresh_llm_call_summary(state)
    record_event("llm.logical_call", {
        "call_id": call["call_id"], "trace_id": call.get("trace_id"), "node": call.get("node"),
        "operation": call["operation"], "call_sequence": call["call_sequence"],
        "duration_ms": call["duration_ms"], "outcome": outcome,
        "sdk_attempt_count": len(attempts), "usage_complete": call["usage_complete"],
    })
    _evaluation_journal_event(
        state, "logical_call_finished", call_id=call["call_id"],
        node=call.get("node"), operation=call["operation"], outcome=outcome,
        not_sent_reason=not_sent_reason,
        attempt_ids=[item.get("attempt_id") for item in attempts],
        sdk_attempt_count=sum(item.get("sdk_request_sent") is not False for item in attempts),
        input_tokens_subtotal=call.get("input_tokens_subtotal"),
        output_tokens_subtotal=call.get("output_tokens_subtotal"),
        total_tokens=call.get("total_tokens"), usage_complete=call.get("usage_complete"),
    )
    call.pop("_preflight_reserved_accounted", None)


def _stop_if_token_budget_exhausted(
    call: dict[str, Any], state: dict[str, Any] | None,
) -> bool:
    """Close the shared SDK boundary before provider resolution or execution."""
    if not token_budget_is_exhausted(state):
        return False
    record_token_budget_exceeded(state)
    _finish_logical_call(call, state, "not_sent", "budget_exhausted")
    return True


def _mark_processing_result(call: dict[str, Any], result: str) -> None:
    for attempt in reversed(call.get("attempts", [])):
        if attempt.get("result") == "response_received":
            attempt["processing_result"] = result
            break


def _capture_semantic_eval_parse_failure_output(
    call: dict[str, Any], state: dict[str, Any] | None, raw_output: Any,
) -> None:
    """Keep a bounded parse diagnostic only for an explicitly marked eval case."""
    if (
        not isinstance(state, dict)
        or call.get("operation") != "extract.semantic_interpreter"
        or not isinstance(state.get("_semantic_interpreter_eval_raw_output_case_id"), str)
        or state.get("_semantic_interpreter_eval_raw_output_case_id") != call.get("trace_id")
    ):
        return
    text = raw_output if isinstance(raw_output, str) else ""
    call["semantic_interpreter_eval_raw_output_diagnostic"] = {
        "raw_output_chars": len(text),
        "raw_output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "truncated_text": text[:500],
        "truncated": len(text) > 500,
    }


def _latest_attempt_received_response(call: dict[str, Any]) -> bool:
    return any(attempt.get("result") == "response_received" for attempt in call.get("attempts", []))


def llm_classify(
    system_prompt: str,
    user_message: str,
    allowed: list[str],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    operation: str = "classify",
    interaction_policy: InteractionPolicy | str | None = None,
) -> str:
    """调用 LLM 从预定义分类中选一个结果。

    如果 LLM 不可用（无 API Key、网络异常、API 错误等），
    返回空字符串，由调用方负责降级。

    参数：
    - `system_prompt`：系统提示词，定义分类规则。
    - `user_message`：用户问题原文。
    - `allowed`：允许的分类值列表。
    - `model`：模型名，默认使用配置的 `hy3`。
    - `temperature`：采样温度，分类场景设为 0.0 保证确定性。
    - `operation`：调用方提供的逻辑操作名，默认保留历史值 `classify`。
    - `interaction_policy`：可选的显式交互 preset；省略时保留原调用路径。

    返回值：
    - `str`：LLM 从 `allowed` 中选择的分类值。
      失败时返回空字符串；调用方必须显式 fail closed，不得据用户文本猜测路由。

    可恢复供应商故障（连接/超时/429/5xx）会依次尝试备用 provider（见 `_providers`）。
    """

    policy = resolve_policy(interaction_policy)
    if policy is not None:
        allowed_text = "\n".join(f"- {value}" for value in allowed)
        full_prompt = (
            f"{system_prompt}\n\n可选分类：\n{allowed_text}\n\n"
            "请只输出分类名称，不要包含其他内容。"
        )
        return _llm_chat_with_interaction_policy(
            full_prompt,
            user_message,
            policy,
            model=model,
            temperature=temperature,
            max_tokens=512,
            operation=operation,
            quality=False,
            allowed=allowed,
        )

    call, state = _begin_logical_call(operation, model)
    if _stop_if_token_budget_exhausted(call, state):
        return ""
    if not _quota_available():
        _finish_logical_call(call, state, "not_sent", "quota_blocked")
        return ""
    providers = _providers()
    if not providers:
        _finish_logical_call(call, state, "not_sent", "no_provider_configured")
        return ""
    last_exc: Exception | None = None
    for index, provider in enumerate(providers):
        selected, profile = provider.resolve(operation, model, False)
        try:
            allowed_text = "\n".join(f"- {v}" for v in allowed)
            full_prompt = (
                f"{system_prompt}\n\n可选分类：\n{allowed_text}\n\n"
                "请只输出分类名称，不要包含其他内容。"
            )
            request_kwargs = {
                "model": selected,
                "messages": [
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": temperature,
                **_output_limit_kwargs(512, state),
            }
            response = _invoke_provider_sdk(
                call, state, provider, selected,
                lambda: provider.client.chat.completions.create(**request_kwargs),
                request_kwargs=request_kwargs,
            )
            _record_usage(response, selected, operation, profile,
                          provider_id=provider.id, fallback_from=_fallback_source(index), call_record=call)
            result = response.choices[0].message.content.strip() if response.choices else ""
            print(f"[DEBUG/LLM] {operation} → '{result}' (from {allowed}) len(prompt)={len(full_prompt)} provider={provider.id}")
            accepted = result in allowed
            _mark_processing_result(call, "success" if accepted else "parse_failure")
            _finish_logical_call(call, state, "success" if accepted else "parse_failure")
            return result if accepted else ""
        except Exception as exc:
            local_refusal_reason = _local_send_refusal_reason(exc)
            if local_refusal_reason is not None:
                _finish_logical_call(call, state, "not_sent", local_refusal_reason)
                return ""
            if _is_recoverable_provider_failure(exc):
                last_exc = exc
                _record_upstream_error(operation, provider, selected, exc)
                print(f"[DEBUG/LLM] provider {provider.id} recoverable failure; "
                      f"{'trying backup' if index + 1 < len(providers) else 'no fallback available'}: {type(exc).__name__}")
                continue
            if _latest_attempt_received_response(call):
                _mark_processing_result(call, "processing_error")
                _finish_logical_call(call, state, "processing_error")
            else:
                _record_upstream_error(operation, provider, selected, exc)
                _finish_logical_call(call, state, "provider_error")
            print(f"[DEBUG/LLM] {operation} FAILED: {type(exc).__name__}")
            return ""
    print(f"[DEBUG/LLM] {operation} FAILED: all providers unreachable: {last_exc}")
    _finish_logical_call(call, state, "provider_error")
    return ""


def llm_extract(
    system_prompt: str,
    user_message: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    quality: bool = False,
    operation: str = "extract",
    max_tokens: int | None = None,
    _enable_thinking: bool | None = None,
    _timeout_seconds: float | None = None,
    interaction_policy: InteractionPolicy | str | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """调用 LLM 输出结构化 JSON 数据。

    适合 Data Query Agent 提取 MetricQueryPlan、
    Critic Agent 提取审核裁决等场景。

    参数：
    - `system_prompt`：系统提示词，定义 JSON schema 和提取规则。
    - `user_message`：用户输入。
    - `model`：模型名。
    - `temperature`：采样温度。
    - `quality`：True 时走 quality profile（report/long_answer 类高质量路由）。
    - `operation`：usage/profile 记录中的逻辑操作名；默认为兼容旧行为的 `extract`。
    - `max_tokens`：可选的本次调用输出上限。
    - `response_format`：可选 provider-native structured output 参数；默认 None 保持旧行为。

    返回值：
    - `dict[str, Any] | None`：解析成功的 JSON dict；失败返回 None。
    """

    policy = resolve_policy(interaction_policy)
    if response_format is not None and not isinstance(response_format, dict):
        raise TypeError("response_format must be a JSON object")
    if policy is not None:
        if _enable_thinking is not None or _timeout_seconds is not None:
            raise ValueError(
                "interaction_policy cannot be combined with the internal thinking or timeout override."
            )
        return _llm_extract_with_interaction_policy(
            system_prompt,
            user_message,
            policy,
            model=model,
            temperature=temperature,
            quality=quality,
            operation=operation,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    if _enable_thinking not in (None, False):
        raise ValueError("Internal Critic request policy only supports the provider-default or non-thinking mode.")
    call_config_override: LLMCallConfig | None = None
    if _timeout_seconds is not None:
        call_config_override = _with_timeout_override(_call_config(), _timeout_seconds)
    request_options = (
        {"enable_thinking": False, "parameter_location": "extra_body"}
        if _enable_thinking is False
        else {"enable_thinking": None, "parameter_location": "omitted_provider_default"}
    )
    if response_format is not None:
        request_options["response_format_type"] = str(response_format.get("type") or "unknown")
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict) and isinstance(json_schema.get("name"), str):
            request_options["response_schema_name"] = json_schema["name"][:100]
    call, state = _begin_logical_call(operation, model, request_options=request_options)
    if _stop_if_token_budget_exhausted(call, state):
        return None
    if not _quota_available():
        _finish_logical_call(call, state, "not_sent", "quota_blocked")
        return None
    providers = (
        _providers(call_config=call_config_override)
        if call_config_override is not None else _providers()
    )
    if call_config_override is not None:
        # A dedicated verifier deadline bounds the logical call: do not spend
        # one full timeout on each configured application retry or fallback.
        providers = providers[:1]
    if not providers:
        _finish_logical_call(call, state, "not_sent", "no_provider_configured")
        return None
    last_exc: Exception | None = None
    for index, provider in enumerate(providers):
        selected, profile = provider.resolve(operation, model, quality)
        if _enable_thinking is False and (provider.id != "primary" or selected != "qwen3.8-flash"):
            _finish_logical_call(call, state, "configuration_error", "unsupported_critic_thinking_target")
            raise ValueError(
                "Non-thinking Critic experiment requires the configured primary OpenAI-compatible "
                "qwen3.8-flash endpoint; no request was sent."
            )
        try:
            full_prompt = (
                f"{system_prompt}\n\n"
                "请只输出 JSON 对象，不要包含 ```json 标记或额外说明。"
            )
            output_limits = _output_limit_kwargs(
                int(max_tokens if max_tokens is not None else get_settings().llm_extract_max_tokens),
                state,
            )
            if max_tokens is not None and "max_tokens" in output_limits:
                # Frozen provider experiments own max_completion_tokens; a
                # normal caller's max_tokens must not shrink that budget.
                output_limits["max_tokens"] = min(output_limits["max_tokens"], int(max_tokens))
            request_kwargs = {
                "model": selected,
                "messages": [
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": temperature,
                **output_limits,
                **({"extra_body": {"enable_thinking": False}} if _enable_thinking is False else {}),
            }
            if response_format is not None:
                request_kwargs["response_format"] = response_format

            def sdk_operation():
                return provider.client.chat.completions.create(**request_kwargs)
            if call_config_override is None:
                response = _invoke_provider_sdk(
                    call, state, provider, selected, sdk_operation,
                    request_kwargs=request_kwargs,
                )
            else:
                response = _invoke_provider_sdk(
                    call, state, provider, selected, sdk_operation,
                    call_config=call_config_override,
                    request_kwargs=request_kwargs,
                )
            _record_usage(response, selected, operation, profile,
                          provider_id=provider.id, fallback_from=_fallback_source(index), call_record=call)
            content = response.choices[0].message.content if response.choices else ""
            raw_output = content if isinstance(content, str) else ""
            text = raw_output.strip()
            if not text:
                _capture_semantic_eval_parse_failure_output(call, state, raw_output)
                print(f"[DEBUG/LLM] {operation} → None (empty response) provider={provider.id}")
                _mark_processing_result(call, "empty_response")
                _finish_logical_call(call, state, "empty_response")
                return None
            # 确定性 JSON 修复：剥离推理块/围栏 → 解包双重编码 → 再交给调用方做 schema 校验。
            repaired = _loads_with_repair(_strip_reasoning_and_fences(text))
            if repaired is None:
                _capture_semantic_eval_parse_failure_output(call, state, raw_output)
                print(f"[DEBUG/LLM] {operation} → None (unparseable after repair) provider={provider.id}")
                _mark_processing_result(call, "parse_failure")
                _finish_logical_call(call, state, "parse_failure")
                return None
            print(f"[DEBUG/LLM] {operation} → {list(repaired.keys()) if isinstance(repaired, dict) else type(repaired).__name__} provider={provider.id}")
            _mark_processing_result(call, "success")
            _finish_logical_call(call, state, "success")
            return repaired
        except Exception as exc:
            if call_config_override is not None and _is_timeout_exception(exc):
                _finish_logical_call(call, state, "timeout")
                return None
            local_refusal_reason = _local_send_refusal_reason(exc)
            if local_refusal_reason is not None:
                _finish_logical_call(call, state, "not_sent", local_refusal_reason)
                return None
            if _is_recoverable_provider_failure(exc):
                last_exc = exc
                _record_upstream_error(operation, provider, selected, exc)
                print(f"[DEBUG/LLM] provider {provider.id} recoverable failure; "
                      f"{'trying backup' if index + 1 < len(providers) else 'no fallback available'}: {type(exc).__name__}")
                continue
            if _latest_attempt_received_response(call):
                _mark_processing_result(call, "processing_error")
                _finish_logical_call(call, state, "processing_error")
            else:
                _record_upstream_error(operation, provider, selected, exc)
                _finish_logical_call(call, state, "provider_error")
            print(f"[DEBUG/LLM] {operation} FAILED: {type(exc).__name__}")
            return None
    print(f"[DEBUG/LLM] {operation} FAILED: all providers unreachable: {last_exc}")
    _finish_logical_call(call, state, "provider_error")
    return None


def llm_agent_action(
    system_prompt: str,
    user_message: str,
    *,
    model: str | None = None,
    max_tokens: int = 4096,
    interaction_policy: InteractionPolicy | str | None = None,
) -> dict[str, Any] | None:
    """Return one JSON action for a governed agent loop.

    The provider remains OpenAI-compatible and usage accounting/failover stay
    in ``llm_extract``.  The caller validates the returned object against its
    domain discriminated union before execution.
    """
    if interaction_policy is None:
        return llm_extract(
            system_prompt, user_message, model=model, quality=False,
            max_tokens=max_tokens,
        )
    return llm_extract(
        system_prompt,
        user_message,
        model=model,
        quality=False,
        max_tokens=max_tokens,
        interaction_policy=interaction_policy,
    )


def llm_chat(
    system_prompt: str,
    user_message: str,
    *,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    interaction_policy: InteractionPolicy | str | None = None,
    operation: str = "chat",
    quality: bool = False,
) -> str:
    """通用 LLM 对话调用。

    如果 LLM 不可用（无 API Key、网络异常等），返回空字符串。

    参数：
    - `system_prompt`：系统提示词。
    - `user_message`：用户输入。
    - `model`：模型名。
    - `temperature`：采样温度。
    - `max_tokens`：输出上限。

    返回值：
    - `str`：LLM 生成的回复文本。失败时返回空字符串。

    使用场景：
    - Data Query Agent 的自然语言→查询计划转换。
    - Critic Agent 的 groundedness 审核。
    - Report Agent 的报告草稿生成。
    """

    policy = resolve_policy(interaction_policy)
    if policy is not None:
        return _llm_chat_with_interaction_policy(
            system_prompt,
            user_message,
            policy,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            operation=operation,
            quality=quality,
        )

    call, state = _begin_logical_call(operation, model)
    if _stop_if_token_budget_exhausted(call, state):
        return ""
    if not _quota_available():
        _finish_logical_call(call, state, "not_sent", "quota_blocked")
        return ""
    providers = _providers()
    if not providers:
        _finish_logical_call(call, state, "not_sent", "no_provider_configured")
        return ""
    last_exc: Exception | None = None
    for index, provider in enumerate(providers):
        selected, profile = provider.resolve(operation, model, quality)
        try:
            kwargs: dict[str, Any] = {
                "model": selected,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": temperature,
            }
            kwargs.update(_output_limit_kwargs(max_tokens or get_settings().llm_chat_max_tokens, state))
            response = _invoke_provider_sdk(
                call, state, provider, selected,
                lambda: provider.client.chat.completions.create(**kwargs),
                request_kwargs=kwargs,
            )
            _record_usage(response, selected, operation, profile,
                          provider_id=provider.id, fallback_from=_fallback_source(index), call_record=call)
            reply = response.choices[0].message.content.strip() if response.choices else ""
            input_tokens = call["attempts"][-1].get("input_tokens")
            output_tokens = call["attempts"][-1].get("output_tokens")
            print(f"[DEBUG/LLM] {operation} → len(reply)={len(reply)} token_in={input_tokens!s} token_out={output_tokens!s} provider={provider.id}")
            result = "success" if reply else "empty_response"
            _mark_processing_result(call, result)
            _finish_logical_call(call, state, result)
            return reply
        except Exception as exc:
            local_refusal_reason = _local_send_refusal_reason(exc)
            if local_refusal_reason is not None:
                _finish_logical_call(call, state, "not_sent", local_refusal_reason)
                return ""
            if _is_recoverable_provider_failure(exc):
                last_exc = exc
                _record_upstream_error(operation, provider, selected, exc)
                print(f"[DEBUG/LLM] provider {provider.id} recoverable failure; "
                      f"{'trying backup' if index + 1 < len(providers) else 'no fallback available'}: {type(exc).__name__}")
                continue
            if _latest_attempt_received_response(call):
                _mark_processing_result(call, "processing_error")
                _finish_logical_call(call, state, "processing_error")
            else:
                _record_upstream_error(operation, provider, selected, exc)
                _finish_logical_call(call, state, "provider_error")
            print(f"[DEBUG/LLM] {operation} FAILED: {type(exc).__name__}")
            return ""
    print(f"[DEBUG/LLM] {operation} FAILED: all providers unreachable: {last_exc}")
    _finish_logical_call(call, state, "provider_error")
    return ""
