"""异常脱敏与三层错误模型。

将原始异常转换为用户可读的 public_message（SSE 前端使用）
和经脱敏的 diagnostic_message（仅 Trace/日志）。
"""

import re
from typing import Any

from app.core.config import get_settings


# 异常类型全限定名 → public_message 映射表
# 注意：映射键使用模块全路径，不以 type(exc).__name__ 匹配，避免类名冲突
_ERROR_PUBLIC_MAP: dict[str, str] = {
    "app.resilience.policy.ExternalServiceError": "外部服务调用异常，已保留 Trace",
    "sqlalchemy.exc.ProgrammingError": "数据查询异常，已保留 Trace",
    "sqlalchemy.exc.OperationalError": "数据库连接异常，请稍后重试",
    "builtins.ValueError": "输入参数异常，请检查问题表述",
    "builtins.PermissionError": "当前权限不足以执行此操作",
}

_DEFAULT_PUBLIC = "分析过程中出现内部异常，已保留 Trace"

# 需要从诊断信息中隐藏的敏感模式
_SENSITIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)(password|passwd|pwd|secret|token|api[_-]?key)\s*[:=]\s*\S+"),
    re.compile(r"postgresql\+psycopg://\S+?@"),           # 数据库连接 URI
    re.compile(r"bearer\s+\S+", re.IGNORECASE),
    re.compile(r"\b[a-f0-9]{32,}\b"),                     # 过长 hex（可能为 token）
    re.compile(r"VALUES?\s*\([^)]{50,}\)"),               # 带长参数的 SQL VALUES
]


def _qualified_name(exc: Exception) -> str:
    """返回异常的全限定名，如 sqlalchemy.exc.ProgrammingError"""
    module = type(exc).__module__
    name = type(exc).__qualname__
    return f"{module}.{name}"


def _mask_sensitive(text: str) -> str:
    """掩盖诊断消息中的敏感信息。"""
    result = text
    for pattern in _SENSITIVE_PATTERNS:
        result = pattern.sub("<REDACTED>", result)
    return result


def sanitize_exception(exc: Exception) -> tuple[str, str]:
    """将异常转换为 (public_message, diagnostic_message)。

    - public_message：用户可读的安全中文描述
    - diagnostic_message：经脱敏的诊断摘要（500 字符，无敏感信息）
    """
    qname = _qualified_name(exc)
    public_msg = _ERROR_PUBLIC_MAP.get(qname, _DEFAULT_PUBLIC)

    # 取原始消息的前 500 字符用于诊断
    raw_msg = str(exc)[:500]
    # 脱敏
    diag_msg = _mask_sensitive(raw_msg)
    # 去 SQL-like 多行
    diag_msg = re.sub(r"\s+", " ", diag_msg).strip()

    return public_msg, diag_msg
