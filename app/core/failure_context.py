"""Stable, redacted failure metadata shared by query and investigation layers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ExecutionFailureContext(BaseModel):
    """Safe failure facts; intentionally excludes exception messages and SQL."""

    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "tool_timeout", "database_error", "capability_gap",
        "authorization_denied", "execution_error",
    ]
    service: Literal[
        "metric_registry", "mcp", "postgresql", "remote_mcp", "unknown",
    ]
    code: str
    cause_type: str
    request_id: str | None = None
    retryable: bool | None = None


_PUBLIC_MESSAGES = {
    "tool_timeout": "数据工具查询超时，未取得该项结果。",
    "database_error": "数据库查询执行失败，未取得该项结果。",
    "capability_gap": "当前指标或查询能力不支持该项操作。",
    "authorization_denied": "当前授权不允许执行该项查询。",
    "execution_error": "数据查询执行失败，未取得该项结果。",
}


def public_failure_message(context: ExecutionFailureContext | dict) -> str:
    """Return a user-safe summary based only on the structured category."""
    category = context.category if isinstance(context, ExecutionFailureContext) else context.get("category")
    return _PUBLIC_MESSAGES.get(str(category), _PUBLIC_MESSAGES["execution_error"])
