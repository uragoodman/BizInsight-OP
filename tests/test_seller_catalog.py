"""Unit tests for the tenant-scoped seller catalog route and node."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agents.planner import build_execution_plan
from app.context.orchestrator import ContextOrchestrator
from app.graph.nodes import seller_catalog as seller_catalog_module
from app.graph.nodes import triage as triage_module
from app.graph.nodes.seller_catalog import _markdown_cell, seller_catalog
from app.llm.triage_prompt import ROUTING_TABLE
from app.tenancy import TenantContext


def _state(**overrides: object) -> dict[str, object]:
    return {"node_history": [], **overrides}


@pytest.mark.parametrize(
    ("question", "classified_task"),
    [
        ("列出我能访问的店铺", "seller_catalog"),
        # A catalog-like phrase cannot override the classifier's task decision.
        ("列出我能访问的店铺", "metric_query"),
        ("各店铺的 GMV 趋势如何", "metric_query"),
    ],
)
def test_seller_catalog_route_comes_from_registered_task_classification(
    monkeypatch: pytest.MonkeyPatch, question: str, classified_task: str,
) -> None:
    monkeypatch.setattr(
        "app.context.request_contract.validate_compiled_request_contract",
        lambda _compilation: (True, None),
    )
    monkeypatch.setattr(
        ContextOrchestrator,
        "build_triage_context",
        lambda *_args, **_kwargs: SimpleNamespace(bundle=SimpleNamespace(text="controlled triage context")),
    )
    captured: dict[str, object] = {}

    def classify(_system: str, user_message: str, allowed: list[str]) -> str:
        captured.update(question=user_message, allowed=allowed)
        return classified_task

    monkeypatch.setattr(triage_module, "llm_classify", classify)
    result = triage_module.triage({
        "user_message": question,
        "context_compilation": {
            "standalone_question": question,
            "needs_clarification": False,
            "request_contract": {"contract_hash": "test-contract"},
        },
        "node_history": [],
    })

    assert captured["question"]
    assert "seller_catalog" in captured["allowed"]
    assert result["task_type"] == classified_task
    assert result["next_agent"] == ROUTING_TABLE[classified_task]


def test_seller_catalog_is_pinned_in_routing_and_planner() -> None:
    assert ROUTING_TABLE["seller_catalog"] == "seller_catalog"
    plan = build_execution_plan("seller_catalog")
    assert [agent.value for agent in plan.agents] == ["seller_catalog"]


def test_seller_catalog_lists_authorized_rows_without_a_data_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = TenantContext(
        tenant_id="tenant-a",
        user_id="user-a",
        role="admin",
        scope_type="platform",
    )
    captured: dict[str, object] = {}

    def fake_search_sellers(
        active_context: TenantContext, query: str = "", limit: int = 20
    ) -> list[dict[str, object]]:
        captured.update(context=active_context, query=query, limit=limit)
        return [
            {"seller_id": "seller-1", "label": "旗舰店"},
            {"seller_id": "seller-2", "label": "A|B\n店"},
        ]

    monkeypatch.setattr(
        seller_catalog_module, "require_tenant_context", lambda: context
    )
    monkeypatch.setattr(seller_catalog_module, "search_sellers", fake_search_sellers)

    state = seller_catalog(_state())

    assert captured == {"context": context, "query": "", "limit": 50}
    assert state["seller_catalog"] == {
        "items": [
            {"seller_id": "seller-1", "label": "旗舰店"},
            {"seller_id": "seller-2", "label": "A|B\n店"},
        ],
        "limit": 50,
    }
    assert "旗舰店" in state["answer"]
    # Labels stay on one Markdown table row even when the database value has
    # a pipe or a newline.
    assert "| A\\|B 店 |" in state["answer"]
    assert state["current_agent"] == "seller_catalog"
    assert state["next_agent"] == "persist_trace"
    assert [item["node"] for item in state["node_history"]] == ["seller_catalog"]


def test_seller_catalog_is_empty_and_explicit_without_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = TenantContext(
        tenant_id="tenant-a",
        user_id="user-a",
        role="analyst",
        scope_type="seller",
        seller_id="seller-1",
    )
    monkeypatch.setattr(
        seller_catalog_module, "require_tenant_context", lambda: context
    )
    monkeypatch.setattr(
        seller_catalog_module, "search_sellers", lambda *args, **kwargs: []
    )

    state = seller_catalog(_state())

    assert state["seller_catalog"]["items"] == []
    assert state["answer"] == "当前账号没有可访问且有经营数据的店铺。"


def test_markdown_cell_keeps_labels_on_one_line() -> None:
    assert _markdown_cell("A|B\nC`D") == "A\\|B C\\`D"
