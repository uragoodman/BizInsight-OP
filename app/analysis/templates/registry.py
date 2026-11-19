"""白名单分析模板注册表。

每个模板是一个纯函数，接收统一参数结构 (seller_id, start_date, end_date, **kwargs)，
返回结构化 dict。Agent 不能修改模板逻辑，只能选模板和传参数。
"""

from __future__ import annotations

from typing import Any, Callable

from app.core.config import get_settings

_TEMPLATE_REGISTRY: dict[str, tuple[Callable, str, list[str], list[str], str]] = {}


def register(
    name: str, display_name: str,
    required_params: list[str], keywords: list[str],
    description: str,
) -> Callable:
    """Register a whitelisted implementation and its descriptive metadata."""
    def decorator(fn: Callable) -> Callable:
        _TEMPLATE_REGISTRY[name] = (fn, display_name, required_params, keywords, description)
        return fn
    return decorator


def get_template(name: str) -> tuple[Callable, str, list[str], list[str], str]:
    """按模板名查找 (function, display_name, required_params, keywords, description)。"""
    if name not in _TEMPLATE_REGISTRY:
        raise KeyError(f"Unknown template: {name!r}")
    return _TEMPLATE_REGISTRY[name]


def list_templates() -> list[dict[str, Any]]:
    """列出所有注册模板的元信息（不含函数本身）。"""
    return [
        {"name": k, "display_name": v[1], "required_params": v[2],
         "keywords": v[3], "description": v[4]}
        for k, v in _TEMPLATE_REGISTRY.items()
    ]


def list_chat_visible_templates() -> list[dict[str, Any]]:
    """List templates participating in chat matching under the configured allowlist."""

    templates = list_templates()
    allowlist = {str(name) for name in get_settings().complex_chat_visible_templates if str(name)}
    if not allowlist:
        return templates
    return [template for template in templates if str(template["name"]) in allowlist]
