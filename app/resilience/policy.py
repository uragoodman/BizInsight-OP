"""LLM 调用的容错与韧性层。

提供统一的"重试 + 熔断 + 超时"保护机制，所有对外部服务
（LLM API、MCP 子进程）的调用都经过本模块包装。

核心组件：
- `CircuitBreaker`：故障熔断器（3 次连续失败 → 打开 → 15 秒后自动复位）
- `execute_with_resilience()`：指数退避重试 + 线程级超时 + 熔断保护

设计原则：
- 只保护幂等读操作（LLM chat/classify/extract、MCP 数据查询）
- 非幂等写操作不应经过此模块
- 熔断器按 service 名隔离（"llm"、"mcp"），互不影响
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextvars import copy_context
from time import sleep, monotonic
from typing import Callable, TypeVar

T = TypeVar("T")


class ExternalServiceError(RuntimeError):
    """统一的外部服务异常，携带服务名、错误码、是否可重试。

    与 AgentError 不同：AgentError 记录在 state.errors 中继续图流；
    而 ExternalServiceError 由 execute_with_resilience 向上抛出，
    由调用方（LLM client 或 MCP client）决定如何处理。
    """

    def __init__(self, service: str, code: str, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.service = service    # "llm" 或 "mcp"
        self.code = code           # "timeout" / "circuit_open" / "external_call_failed"
        self.retryable = retryable  # 是否可重试（不可重试的错误直接向上抛）


@dataclass
class CircuitBreaker:
    """基于连续失败计数的熔断器。

    状态机：
    - CLOSED（正常）：允许调用，连续失败达到 failure_threshold 时转为 OPEN
    - OPEN（熔断）：拒绝所有调用，reset_timeout_seconds 后转为 HALF-OPEN
    - HALF-OPEN（探活）：允许一次调用，成功则复位到 CLOSED，失败则回到 OPEN

    属性：
    - failure_threshold：连续失败多少次后打开熔断（默认 3）
    - reset_timeout_seconds：熔断打开后多久尝试复位（默认 15 秒）
    - failures：当前连续失败计数
    - opened_at：熔断打开的时间戳（monotonic 秒）
    """

    failure_threshold: int = 3
    reset_timeout_seconds: float = 15.0
    failures: int = 0
    opened_at: float | None = None

    def allow(self) -> bool:
        """判断当前是否允许发起调用。

        返回 False 表示熔断打开中，调用方应直接抛出 ExternalServiceError。
        返回 True 表示熔断关闭或已超时复位。
        """
        if self.opened_at is None:
            return True
        if monotonic() - self.opened_at >= self.reset_timeout_seconds:
            # 熔断超时，复位到正常状态
            self.failures, self.opened_at = 0, None
            return True
        return False

    def record_success(self) -> None:
        """调用成功时：清零失败计数，关闭熔断。"""
        self.failures, self.opened_at = 0, None

    def record_failure(self) -> None:
        """调用失败时：递增计数，达到阈值时打开熔断。"""
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = monotonic()


# 全局熔断器注册表，按 service 名隔离（"llm"、"mcp"）
_BREAKERS: dict[str, CircuitBreaker] = {}


def execute_with_resilience(
    operation: Callable[[], T],
    service: str,
    attempts: int = 2,
    backoff_seconds: float = 0.05,
    timeout_seconds: float | None = None,
    operation_manages_timeout: bool = False,
    retry_if: Callable[[Exception], bool] | None = None,
) -> T:
    """以重试+熔断+超时保护执行一次幂等的外部服务调用。

    执行流程：
    1. 检查熔断器状态，如果开路则立即拒绝
    2. 在线程池中执行操作（支持 timeout_seconds 作为硬超时）
    3. 成功 → 记录 success，返回结果
    4. 失败 → 判断是否可重试，是则指数退避后重试，否则记录 failure 并抛出
    5. 全部重试耗尽 → 记录 failure 并抛出 ExternalServiceError

    参数：
    - operation：要执行的幂等调用（无参 callable）
    - service：服务名（"llm"、"mcp"），用于隔离熔断器
    - attempts：最多尝试次数（含首次，默认 2 次）
    - backoff_seconds：退避基础间隔（第 n 次重试等待 backoff * 2^n 秒）
    - timeout_seconds：单次调用硬超时（秒），None 表示不设超时

    返回：
    - operation 的返回值

    抛出：
    - ExternalServiceError：熔断开路或全部重试耗尽
    """
    breaker = _BREAKERS.setdefault(service, CircuitBreaker())
    if not breaker.allow():
        raise ExternalServiceError(
            service, "circuit_open",
            f"{service} circuit is open.",
            retryable=True,
        )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            if timeout_seconds is None:
                # 无超时限制，直接在当前线程执行
                value = operation()
            else:
                # 有超时限制，在独立线程中执行，超时则取消
                executor = ThreadPoolExecutor(max_workers=1)
                # ContextVars (including the active workflow state used by the
                # LLM send gate) do not propagate through submit(). Copy a fresh
                # context for every attempt so retries and concurrent requests
                # cannot enter the same Context object or share request state.
                worker_context = copy_context()
                future = executor.submit(worker_context.run, operation)
                try:
                    # A timed SDK already owns its transport deadline. An
                    # equal outer timeout can return while the send worker is
                    # still writing its terminal event and usage. Keep context
                    # propagation, but join that one SDK attempt before return.
                    value = future.result(timeout=None if operation_manages_timeout else timeout_seconds)
                except FutureTimeout as exc:
                    future.cancel()
                    raise ExternalServiceError(
                        service, "timeout",
                        f"{service} call timed out.",
                    ) from exc
                finally:
                    # 超时后不等待无响应的外部客户端
                    executor.shutdown(wait=False, cancel_futures=True)
            breaker.record_success()
            return value
        except Exception as exc:
            last_error = exc
            # A local send guard can reject work before any remote call exists.
            # Do not retry it or count it as a dependency failure in the breaker.
            if getattr(exc, "local_send_refused", False):
                raise
            if retry_if is not None and not retry_if(exc):
                breaker.record_failure()
                raise ExternalServiceError(
                    service, "external_call_failed", str(exc), retryable=False,
                ) from exc
            # 如果是不可重试的错误（retryable=False），不等待直接抛
            retryable = not isinstance(exc, ExternalServiceError) or exc.retryable
            if not retryable or attempt == attempts - 1:
                breaker.record_failure()
                if isinstance(exc, ExternalServiceError):
                    raise
                raise ExternalServiceError(
                    service, "external_call_failed",
                    str(exc), retryable=True,
                ) from exc
            # 指数退避：第 n 次重试等待 backoff * 2^n 秒
            sleep(backoff_seconds * (2 ** attempt))
    raise ExternalServiceError(service, "external_call_failed", str(last_error))
