"""HTTP 请求可观测性中间件。

FastAPI 中间件，在每一个 HTTP 请求入站时创建 OpenTelemetry Span，
并将 request_id 注入响应头，使前端可以关联到后端的 trace。

工作流程：
1. 从请求头提取或生成 `x-request-id`（用于全链路追踪）
2. 在 `chat.request` Span 包裹下调用后续中间件和路由处理
3. 将 `x-request-id` 写回响应头，前端可用于 trace 查询
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import Request

from app.observability import span


async def trace_request(request: Request, call_next):
    """创建 HTTP 请求级别的 Span，注入并回传请求追踪 ID。

    这是整个请求生命周期的第一个 Span，后续所有 graph、LLM、MCP
    调用都在这个 Span 之下。request_id 贯穿前端 → 后端 → 图引擎
    → LLM → 存储的全链路。

    参数：
    - request：FastAPI Request 对象，从中读取 x-request-id 请求头
    - call_next：ASGI 中间件链的下一个处理函数

    返回：
    - Response：注入 x-request-id 响应头后的 HTTP 响应
    """
    request_id = request.headers.get("x-request-id", str(uuid4()))
    with span("chat.request", {
        "http.request.method": request.method,
        "url.path": request.url.path,
        "request_id": request_id,
    }):
        response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response
