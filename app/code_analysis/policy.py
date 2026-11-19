"""Static policy gate for generated analysis.py artifacts.

This is deliberately conservative.  It complements, never replaces, the
container sandbox used by the worker.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

ALLOWED_IMPORT_ROOTS = {
    "pandas",
    "polars",
    "numpy",
    "scipy",
    "sklearn",
    "statsmodels",
    "matplotlib",
    "plotly",
    "json",
    "math",
    "statistics",
    "collections",
    "datetime",
}
ALLOWED_REQUIREMENTS = {
    "pandas",
    "polars",
    "numpy",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "matplotlib",
    "plotly",
    "openpyxl",
    "pyarrow",
}
FORBIDDEN_NAMES = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "__builtins__",
    "open",
    "input",
    "breakpoint",
    "exit",
    "quit",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}
FORBIDDEN_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "httpx",
    "urllib",
    "pathlib",
    "shutil",
    "ctypes",
    "importlib",
    "pickle",
    "marshal",
    "multiprocessing",
}

# Some allowed data-science packages re-export dangerous modules and helpers.
# Checking only the import root therefore is not a security boundary; for
# example, pandas exposes ``pandas.io.common.os``.
FORBIDDEN_ATTRIBUTE_SEGMENTS = FORBIDDEN_IMPORTS | {
    "ctypeslib",
    "environ",
    "getenv",
    "get_handle",
    "import_optional_dependency",
    "urlopen",
}
FORBIDDEN_CALL_NAMES = {
    "system",
    "popen",
    "fork",
    "forkpty",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
}

# Uploaded Phase 11 inputs are limited to CSV, XLSX, and Parquet.  Candidate
# code may read exactly the supplied input path through these entrypoints; it
# may not choose another path or invoke a deserializing/remote reader.
SAFE_INPUT_READERS = {
    "read_csv",
    "read_excel",
    "read_parquet",
    "scan_csv",
    "scan_parquet",
    "ExcelFile",
    "ParquetFile",
}
FORBIDDEN_READERS = {
    "read_clipboard",
    "read_feather",
    "read_fwf",
    "read_gbq",
    "read_hdf",
    "read_html",
    "read_json",
    "read_orc",
    "read_pickle",
    "read_sas",
    "read_spss",
    "read_sql",
    "read_sql_query",
    "read_sql_table",
    "read_stata",
    "read_table",
    "read_xml",
    "HDFStore",
    "fromfile",
    "genfromtxt",
    "load",
    "loadtxt",
    "memmap",
}
SENSITIVE_CALLABLE_NAMES = (
    SAFE_INPUT_READERS | FORBIDDEN_READERS | FORBIDDEN_CALL_NAMES
)


@dataclass(frozen=True)
class PolicyResult:
    valid: bool
    errors: list[str]


def _attribute_chain(node: ast.AST) -> list[str]:
    """Return a dotted name without evaluating it."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return list(reversed(parts))


def _is_literal(node: ast.AST | None) -> bool:
    if node is None:
        return True
    try:
        ast.literal_eval(node)
    except (ValueError, TypeError):
        return False
    return True


def _safe_annotation(node: ast.AST | None) -> bool:
    if node is None:
        return True
    return not any(
        isinstance(
            item,
            (
                ast.Call,
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.NamedExpr,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
            ),
        )
        for item in ast.walk(node)
    )


def _reader_path(call: ast.Call) -> ast.AST | None:
    if call.args:
        return call.args[0]
    for keyword in call.keywords:
        if keyword.arg in {
            "filepath_or_buffer",
            "io",
            "path",
            "path_or_buf",
            "source",
        }:
            return keyword.value
    return None


def _is_input_path(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Name) and node.id == "input_path"


def _is_allowed_top_level_call(node: ast.AST) -> bool:
    """Allow only the deterministic headless Matplotlib setup.

    The runner already sets ``MPLBACKEND=Agg``.  Some model-generated
    candidates still emit ``matplotlib.use("Agg")`` before importing pyplot.
    This exact literal-only call changes no files, processes, network, or
    input data; every other top-level call remains rejected.
    """
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    return (
        _attribute_chain(call.func) == ["matplotlib", "use"]
        and len(call.args) == 1
        and not call.keywords
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "Agg"
    )


def _validate_module_body(tree: ast.Module, errors: list[str]) -> None:
    """Reject code that executes candidate-controlled work during import."""
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if _is_allowed_top_level_call(node):
            continue
        if isinstance(node, ast.FunctionDef):
            if node.decorator_list:
                errors.append(
                    f"Top-level function decorators are not allowed: {node.name}"
                )
            if any(not _is_literal(value) for value in node.args.defaults):
                errors.append(
                    f"Top-level function defaults must be literals: {node.name}"
                )
            if any(
                value is not None and not _is_literal(value)
                for value in node.args.kw_defaults
            ):
                errors.append(
                    f"Top-level function defaults must be literals: {node.name}"
                )
            annotations = [
                node.returns,
                *(argument.annotation for argument in node.args.posonlyargs),
                *(argument.annotation for argument in node.args.args),
                *(argument.annotation for argument in node.args.kwonlyargs),
            ]
            if node.args.vararg:
                annotations.append(node.args.vararg.annotation)
            if node.args.kwarg:
                annotations.append(node.args.kwarg.annotation)
            if any(not _safe_annotation(annotation) for annotation in annotations):
                errors.append(
                    f"Executable function annotations are not allowed: {node.name}"
                )
            continue
        if isinstance(node, ast.Assign):
            if (
                all(isinstance(target, ast.Name) for target in node.targets)
                and _is_literal(node.value)
            ):
                continue
        if isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and _is_literal(node.value)
                and _safe_annotation(node.annotation)
            ):
                continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            chain = ".".join(_attribute_chain(node.value.func)) or "<unknown>"
            errors.append(f"Top-level function call is not allowed: {chain}")
        else:
            errors.append(
                f"Top-level executable statement is not allowed: {type(node).__name__}"
            )


def validate_candidate_code(code: str) -> PolicyResult:
    if len(code.encode("utf-8")) > 200_000:
        return PolicyResult(False, ["Candidate code exceeds 200 KB"])
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return PolicyResult(False, [f"Syntax error: {exc.msg}"])
    errors: list[str] = []
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    _validate_module_body(tree, errors)
    analyze_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "analyze"
    ]
    if len(analyze_functions) != 1:
        errors.append(
            "Candidate must define exactly one top-level analyze(input_path, output_dir)"
        )
    else:
        analyze = analyze_functions[0]
        positional = [*analyze.args.posonlyargs, *analyze.args.args]
        if (
            len(positional) != 2
            or [argument.arg for argument in positional] != [
                "input_path",
                "output_dir",
            ]
            or analyze.args.vararg is not None
            or analyze.args.kwarg is not None
            or analyze.args.kwonlyargs
        ):
            errors.append(
                "Candidate entrypoint must be analyze(input_path, output_dir)"
            )
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name.split(".")[0] for alias in node.names]
                if isinstance(node, ast.Import)
                else [(node.module or "").split(".")[0]]
            )
            for name in names:
                if name in FORBIDDEN_IMPORTS or name not in ALLOWED_IMPORT_ROOTS:
                    errors.append(f"Import is not allowed: {name}")
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".")[0]
                symbol = alias.name.split(".")[-1]
                if (
                    symbol in FORBIDDEN_ATTRIBUTE_SEGMENTS
                    or bound_name in FORBIDDEN_ATTRIBUTE_SEGMENTS
                    or symbol in SENSITIVE_CALLABLE_NAMES
                    or bound_name in SENSITIVE_CALLABLE_NAMES
                    or symbol.startswith("__")
                    or bound_name.startswith("__")
                ):
                    errors.append(f"Imported symbol is not allowed: {alias.name}")
        if isinstance(node, ast.Name) and (
            node.id in FORBIDDEN_NAMES or node.id.startswith("__")
        ):
            errors.append(f"Unsafe builtin is not allowed: {node.id}")
        if isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            parent = parents.get(node)
            if node.attr.startswith("__"):
                errors.append("Dunder attribute access is not allowed")
            if any(part in FORBIDDEN_ATTRIBUTE_SEGMENTS for part in chain):
                errors.append(
                    f"Unsafe module or file helper is not allowed: {'.'.join(chain)}"
                )
            if node.attr in SENSITIVE_CALLABLE_NAMES and not (
                isinstance(parent, ast.Call) and parent.func is node
            ):
                errors.append(
                    f"Aliasing sensitive callable is not allowed: {node.attr}"
                )
        if isinstance(node, ast.Call):
            chain = _attribute_chain(node.func)
            call_name = chain[-1] if chain else ""
            if call_name in FORBIDDEN_CALL_NAMES:
                errors.append(f"Process execution is not allowed: {call_name}")
            if call_name in SAFE_INPUT_READERS and not _is_input_path(
                _reader_path(node)
            ):
                errors.append(
                    f"{call_name} may only read the provided input_path"
                )
            if call_name in FORBIDDEN_READERS or (
                call_name.startswith("read_")
                and call_name not in SAFE_INPUT_READERS
            ):
                errors.append(f"Reader is not allowed: {call_name}")
        if isinstance(node, (ast.With, ast.AsyncWith, ast.Lambda, ast.ClassDef, ast.AsyncFunctionDef)):
            errors.append(f"Unsupported construct: {type(node).__name__}")
    return PolicyResult(not errors, sorted(set(errors)))


def validate_candidate_requirements(requirements: list[str]) -> PolicyResult:
    errors: list[str] = []
    for raw in requirements:
        # Generated jobs never install packages. Versions and extras are
        # rejected so the claimed dependency set stays deterministic.
        normalized = raw.strip().lower().replace("_", "-")
        if normalized not in ALLOWED_REQUIREMENTS:
            errors.append(f"Requirement is not preinstalled or allowed: {raw}")
    return PolicyResult(not errors, sorted(set(errors)))
