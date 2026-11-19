from __future__ import annotations

import ast
import inspect
from pathlib import Path

from app.complex_analysis import capability_router
from app.complex_analysis.capability_router_boundaries import capability_router_static_issues
from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG, RouteOutcome


def test_router_and_intent_boundary_have_no_raw_text_or_string_adjudication() -> None:
    assert capability_router_static_issues() == []


def test_router_signature_is_only_contract_and_catalog() -> None:
    assert list(inspect.signature(capability_router.route_semantic_contract).parameters) == [
        "contract", "catalog",
    ]


def test_router_ast_is_free_of_alias_template_comparisons_and_selector_mode() -> None:
    root = Path(__file__).resolve().parents[1]
    router_path = root / "app" / "complex_analysis" / "capability_router.py"
    intent_path = root / "app" / "graph" / "nodes" / "intent_router.py"
    trees = [ast.parse(path.read_text(encoding="utf-8")) for path in (router_path, intent_path)]
    registry_literals = {
        *(alias for concept in DEFAULT_SEMANTIC_CATALOG.concepts
          for alias in (*concept.exact_aliases, *concept.synonyms)),
        *(capability.template_id for capability in DEFAULT_SEMANTIC_CATALOG.capabilities
          if capability.template_id),
    }
    assert "complex_selector_mode" not in router_path.read_text(encoding="utf-8")
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            compared_strings = {
                child.value for operand in (node.left, *node.comparators) for child in ast.walk(operand)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            }
            assert not compared_strings & registry_literals
    assert "template_id" not in RouteOutcome.model_fields
    assert all(capability.template_id for capability in DEFAULT_SEMANTIC_CATALOG.capabilities)
