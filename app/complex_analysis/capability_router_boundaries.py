"""Static boundary checks shared by tests and the Phase 2 gate."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from app.complex_analysis.capability_router import route_semantic_contract
from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG

_RAW_TEXT_PARAMETERS = frozenset({
    "question", "user_message", "user_text", "raw_text", "raw_message",
    "raw_question", "raw_question_text", "question_text", "query_text",
    "utterance", "utterance_text", "source_text", "input_text", "text",
    "prompt", "message",
})
_FORBIDDEN_IMPORT_PARTS = (
    "embedding", "sentence_transformers", "flagembedding", "selector",
    "openai", "dashscope", "cohere", "voyageai",
)


def capability_router_static_issues(repository_root: Path | None = None) -> list[str]:
    """Return AST/inspection violations for the pure shadow-routing boundary."""
    root = repository_root or Path(__file__).resolve().parents[2]
    paths = (
        root / "app" / "complex_analysis" / "capability_router.py",
        root / "app" / "graph" / "nodes" / "intent_router.py",
    )
    registry_literals = {
        *(alias for concept in DEFAULT_SEMANTIC_CATALOG.concepts
          for alias in (*concept.exact_aliases, *concept.synonyms)),
        *(capability.template_id for capability in DEFAULT_SEMANTIC_CATALOG.capabilities
          if capability.template_id),
        *(capability.capability_id for capability in DEFAULT_SEMANTIC_CATALOG.capabilities),
        *(route.route_id for route in DEFAULT_SEMANTIC_CATALOG.routes),
    }
    issues: list[str] = []
    for path in paths:
        module_name = str(path.relative_to(root))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=module_name)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
                if node.args.vararg:
                    arguments.append(node.args.vararg)
                if node.args.kwarg:
                    arguments.append(node.args.kwarg)
                raw_names = sorted(
                    argument.arg for argument in arguments
                    if argument.arg.lower() in _RAW_TEXT_PARAMETERS
                )
                if raw_names:
                    issues.append(f"{module_name}:{node.name} accepts raw-text parameter(s): {raw_names}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name.lower()
                    if (
                        imported == "re"
                        or "explicit_template_matches" in imported
                        or any(part in imported for part in _FORBIDDEN_IMPORT_PARTS)
                    ):
                        issues.append(f"{module_name} imports forbidden module {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                imported_module = (node.module or "").lower()
                forbidden_names = [
                    alias.name for alias in node.names
                    if alias.name == "explicit_template_matches"
                ]
                if imported_module == "re" or any(
                    part in imported_module for part in _FORBIDDEN_IMPORT_PARTS
                ) or forbidden_names:
                    issues.append(f"{module_name} imports a forbidden adjudication dependency")
            elif isinstance(node, ast.Compare):
                compared_nodes = [node.left, *node.comparators]
                compared_strings = {
                    child.value for operand in compared_nodes for child in ast.walk(operand)
                    if isinstance(child, ast.Constant) and isinstance(child.value, str)
                }
                compared_attributes = {
                    child.attr for operand in compared_nodes for child in ast.walk(operand)
                    if isinstance(child, ast.Attribute)
                }
                if compared_strings & registry_literals:
                    issues.append(f"{module_name} compares text/catalog/template literals")
                if "template_id" in compared_attributes:
                    issues.append(f"{module_name} compares a template_id")

    parameter_names = list(inspect.signature(route_semantic_contract).parameters)
    if parameter_names != ["contract", "catalog"]:
        issues.append(f"route_semantic_contract has unexpected parameters: {parameter_names}")
    router_source = paths[0].read_text(encoding="utf-8")
    if "complex_selector_mode" in router_source:
        issues.append("capability router consults complex_selector_mode")
    return issues
