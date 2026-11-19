"""Deterministic, tenant-authorized seller catalog responses."""

from __future__ import annotations


from app.data.repositories.explorer import search_sellers
from app.graph.state import AgentState
from app.tenancy import require_tenant_context


CATALOG_LIMIT = 50


def _markdown_cell(value: object) -> str:
    """Keep database labels on one Markdown table cell/line."""

    return (
        str(value or "-")
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("`", "\\`")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def seller_catalog(state: AgentState) -> AgentState:
    """List shops visible to the authenticated identity, without changing scope."""

    context = require_tenant_context()
    rows = search_sellers(context, query="", limit=CATALOG_LIMIT)
    items = [
        {
            "seller_id": str(row.get("seller_id") or ""),
            "label": str(row.get("label") or row.get("seller_id") or ""),
        }
        for row in rows
        if row.get("seller_id")
    ]

    if not items:
        answer = "当前账号没有可访问且有经营数据的店铺。"
    else:
        table = [
            "当前账号可访问的店铺（最多显示 50 家；这不会自动切换本会话的分析范围）：",
            "",
            "| 店铺 | 店铺 ID |",
            "| --- | --- |",
        ]
        table.extend(
            f"| {_markdown_cell(item['label'])} | `{_markdown_cell(item['seller_id'])}` |"
            for item in items
        )
        table.extend(("", "要开始分析某家店铺，请在“修改范围”中选择并应用它。"))
        answer = "\n".join(table)

    state["seller_catalog"] = {"items": items, "limit": CATALOG_LIMIT}
    state["answer"] = answer
    state["current_agent"] = "seller_catalog"
    state["next_agent"] = "persist_trace"
    state.setdefault("node_history", []).append(
        {
            "node": "seller_catalog",
            "summary": f"Returned {len(items)} authorized seller catalog entries; analysis scope unchanged.",
        }
    )
    return state
