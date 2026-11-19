"""Narrow OpenAI-compatible egress proxy for the OpenCode generator network.

Run this as a separate service.  It owns model credentials; the generator only
receives its short-lived/internal token and cannot reach arbitrary hosts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from collections import defaultdict, deque

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from app.core.config import get_settings

app = FastAPI(title="BizInsight Code Analysis Model Proxy")
_WINDOWS: dict[str, deque[float]] = defaultdict(deque)
logger = logging.getLogger("bizinsight.code_model_proxy")
MAX_REQUEST_BYTES = 1_000_000
MAX_MESSAGES = 64
ALLOWED_ROLES = {"system", "user", "assistant"}
RETRYABLE_UPSTREAM_STATUS_CODES = {429, 500, 502, 503, 504}


class _ConcurrencyGate:
    """Process-local gate protecting the provider credential from bursts."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = 0

    def acquire(self, limit: int, timeout: float) -> bool:
        limit = max(1, int(limit))
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._active >= limit:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            self._active += 1
            return True

    def release(self) -> None:
        with self._condition:
            self._active = max(0, self._active - 1)
            self._condition.notify()


_UPSTREAM_GATE = _ConcurrencyGate()


class CompletionRequest(BaseModel):
    # Do not forward arbitrary OpenAI-compatible extension fields.  This proxy
    # is a narrow model boundary, not a transparent upstream relay.
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[dict]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


def _authorize(authorization: str | None) -> str:
    settings = get_settings()
    token = settings.code_analysis_model_proxy_token
    if not token or authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Model proxy authentication required")
    now = time.monotonic(); window = _WINDOWS[token]
    while window and now - window[0] >= 60: window.popleft()
    if len(window) >= settings.code_analysis_model_proxy_max_requests_per_minute:
        raise HTTPException(status_code=429, detail="Model proxy rate limit exceeded")
    window.append(now)
    return token


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)) -> dict:
    _authorize(authorization)
    settings = get_settings()
    model = settings.llm_model
    return {
        "object": "list",
        "data": [{"id": model, "object": "model", "owned_by": "bizinsight"}],
    }


def _upstream_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, min(float(value), 60.0))
    except ValueError:
        return None


def _backoff_seconds(response: httpx.Response, attempt: int, base: float) -> float:
    retry_after = _retry_after_seconds(response)
    if retry_after is not None:
        return retry_after
    exponential = min(max(0.1, base) * (2**attempt), 30.0)
    return exponential + random.uniform(0.0, max(0.05, exponential * 0.25))


def _upstream_error(response: httpx.Response) -> HTTPException:
    status = response.status_code
    if status in {401, 403}:
        logger.error("code-model-upstream-auth-failed status=%s", status)
        return HTTPException(status_code=502, detail="Model provider authentication failed")
    if status == 429:
        retry_after = _retry_after_seconds(response)
        return HTTPException(
            status_code=429,
            detail="Model provider is rate limited",
            headers={"Retry-After": str(max(1, int(retry_after or 1)))},
        )
    logger.error("code-model-upstream-failed status=%s body=%s", status, response.text[:500])
    return HTTPException(status_code=502, detail="Model provider request failed")


def _acquire_upstream(settings) -> None:
    if not _UPSTREAM_GATE.acquire(
        settings.code_analysis_model_proxy_max_concurrent_requests,
        settings.code_analysis_model_proxy_upstream_timeout_seconds,
    ):
        raise HTTPException(
            status_code=429,
            detail="Model provider concurrency limit reached",
            headers={"Retry-After": "2"},
        )


def _request_upstream(*, url: str, headers: dict, request_data: dict, settings) -> dict:
    max_retries = max(0, int(settings.code_analysis_model_proxy_upstream_max_retries))
    with httpx.Client(timeout=settings.code_analysis_model_proxy_upstream_timeout_seconds) as client:
        for attempt in range(max_retries + 1):
            response = client.post(url, headers=headers, json=request_data)
            if response.status_code < 400:
                return response.json()
            status = response.status_code
            if status not in RETRYABLE_UPSTREAM_STATUS_CODES or attempt >= max_retries:
                raise _upstream_error(response)
            delay = _backoff_seconds(
                response, attempt, settings.code_analysis_model_proxy_upstream_backoff_seconds
            )
            logger.warning(
                "code-model-upstream-retry status=%s attempt=%s/%s delay=%.2f",
                status, attempt + 1, max_retries, delay,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable upstream retry state")


def _open_upstream_stream(*, url: str, headers: dict, request_data: dict, settings):
    """Open a stream only after status/retry handling has completed."""
    client = httpx.Client(timeout=settings.code_analysis_model_proxy_upstream_timeout_seconds)
    max_retries = max(0, int(settings.code_analysis_model_proxy_upstream_max_retries))
    try:
        for attempt in range(max_retries + 1):
            stream_context = client.stream("POST", url, headers=headers, json=request_data)
            response = stream_context.__enter__()
            if response.status_code < 400:
                return client, stream_context, response
            status = response.status_code
            response.read()
            if status not in RETRYABLE_UPSTREAM_STATUS_CODES or attempt >= max_retries:
                stream_context.__exit__(None, None, None)
                raise _upstream_error(response)
            delay = _backoff_seconds(
                response, attempt, settings.code_analysis_model_proxy_upstream_backoff_seconds
            )
            stream_context.__exit__(None, None, None)
            logger.warning(
                "code-model-upstream-retry status=%s attempt=%s/%s delay=%.2f",
                status, attempt + 1, max_retries, delay,
            )
            time.sleep(delay)
    except Exception:
        client.close()
        raise
    raise RuntimeError("unreachable upstream retry state")


def _validated_messages(messages: list[dict]) -> list[dict]:
    if not messages or len(messages) > MAX_MESSAGES:
        raise HTTPException(status_code=422, detail="Invalid model message count")
    try:
        encoded = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Model messages must be JSON serializable") from exc
    if len(encoded) > MAX_REQUEST_BYTES:
        raise HTTPException(status_code=413, detail="Model request is too large")
    normalized: list[dict] = []
    for message in messages:
        role = str(message.get("role") or "")
        if role not in ALLOWED_ROLES:
            raise HTTPException(status_code=422, detail="Unsupported model message role")
        content = message.get("content")
        if not isinstance(content, (str, list)):
            raise HTTPException(status_code=422, detail="Unsupported model message content")
        normalized.append({"role": role, "content": content})
    return normalized


@app.post("/v1/chat/completions")
def chat(payload: CompletionRequest, authorization: str | None = Header(default=None)):
    _authorize(authorization)
    settings = get_settings()
    api_key = settings.code_analysis_upstream_api_key or settings.deepseek_api_key
    base_url = settings.code_analysis_upstream_base_url or settings.deepseek_base_url
    model = settings.llm_model
    if not api_key:
        raise HTTPException(status_code=503, detail="Model provider is not configured")
    messages = _validated_messages(payload.messages)
    request_data = {
        "model": model,
        "messages": messages,
        "temperature": payload.temperature if payload.temperature is not None else 0,
        "max_tokens": min(
            max(payload.max_tokens or 4000, 1),
            max(1, settings.code_analysis_model_proxy_max_output_tokens),
        ),
        "stream": payload.stream,
    }
    request_hash = hashlib.sha256(
        json.dumps(request_data, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()
    logger.info(
        "code-model-request hash=%s model=%s messages=%d stream=%s",
        request_hash[:16],
        model,
        len(messages),
        payload.stream,
    )
    upstream_url = _upstream_url(base_url or "https://opencode.ai/zen/go/v1")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    _acquire_upstream(settings)
    if payload.stream:
        try:
            client, stream_context, response = _open_upstream_stream(
                url=upstream_url, headers=headers, request_data=request_data, settings=settings
            )
        except Exception:
            _UPSTREAM_GATE.release()
            raise

        def event_stream():
            try:
                yield from response.iter_bytes()
            finally:
                try:
                    stream_context.__exit__(None, None, None)
                finally:
                    try:
                        client.close()
                    finally:
                        _UPSTREAM_GATE.release()

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    try:
        return _request_upstream(
            url=upstream_url, headers=headers, request_data=request_data, settings=settings
        )
    finally:
        _UPSTREAM_GATE.release()
