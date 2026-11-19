"""Entrypoint of the disposable, networkless analysis image."""
from __future__ import annotations

import base64
import contextlib
import io
import importlib.util
import json
import os
import signal
import sys
from pathlib import Path

MAX_OUTPUT_BYTES = int(os.getenv("CODE_ANALYSIS_MAX_OUTPUT_BYTES", str(20 * 1024 * 1024)))
MAX_RESULT_BYTES = int(
    os.getenv("CODE_ANALYSIS_MAX_RESULT_BYTES", str(2 * 1024 * 1024))
)
RESULT_PREFIX = "BIZINSIGHT_RESULT:"
EXECUTION_TIMEOUT_SECONDS = int(
    os.getenv("CODE_ANALYSIS_EXECUTION_TIMEOUT_SECONDS", "120")
)


class _BoundedText(io.TextIOBase):
    """Capture candidate diagnostics without allowing unbounded stdout."""

    def __init__(self, limit: int = 32_000) -> None:
        self.limit = limit
        self.parts: list[str] = []
        self.size = 0

    def write(self, value: str) -> int:
        remaining = max(0, self.limit - self.size)
        if remaining:
            fragment = str(value)[:remaining]
            self.parts.append(fragment)
            self.size += len(fragment)
        return len(value)

    def getvalue(self) -> str:
        return "".join(self.parts)


def _collect_outputs(result: dict, output_dir: Path) -> list[dict]:
    artifacts: list[dict] = []
    total = 0
    chart_paths = result.get("charts", [])
    if not isinstance(chart_paths, list):
        raise RuntimeError("charts must be a list")
    safe_root = output_dir.resolve()
    for index, raw_path in enumerate(chart_paths):
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = output_dir / path
        resolved = path.resolve()
        if safe_root not in resolved.parents:
            raise RuntimeError("chart output escaped output_dir")
        if not resolved.is_file():
            raise RuntimeError(f"chart output does not exist: {resolved.name}")
        if resolved.suffix.lower() != ".png":
            raise RuntimeError("only PNG chart outputs are allowed")
        payload = resolved.read_bytes()
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("chart output is not a valid PNG envelope")
        total += len(payload)
        if total > MAX_OUTPUT_BYTES:
            raise RuntimeError("generated artifacts exceed the output limit")
        artifacts.append(
            {
                "name": resolved.name[:255],
                "content_type": "image/png",
                "data_base64": base64.b64encode(payload).decode("ascii"),
                "index": index,
            }
        )
    result["charts"] = [{"name": artifact["name"], "index": artifact["index"]} for artifact in artifacts]
    return artifacts


def _execute_candidate(
    candidate_path: str,
    input_path: str,
    output_dir: Path,
) -> tuple[dict, str]:
    """Import and execute candidate code with bounded diagnostics throughout."""
    captured = _BoundedText()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        spec = importlib.util.spec_from_file_location("candidate", candidate_path)
        if not spec or not spec.loader:
            raise RuntimeError("Candidate module could not be loaded")
        module = importlib.util.module_from_spec(spec)
        # Import-time output and exceptions are candidate-controlled too.  Keep
        # the redirects active before exec_module so top-level output cannot
        # bypass the same bound applied to analyze().
        spec.loader.exec_module(module)
        result = module.analyze(input_path, str(output_dir))
    if not isinstance(result, dict):
        raise RuntimeError("analyze must return an object")
    return result, captured.getvalue()


def _validate_result_contract(result: dict) -> None:
    if (
        not isinstance(result.get("conclusion"), str)
        or not result["conclusion"].strip()
    ):
        raise RuntimeError("conclusion must be a non-empty string")
    method = result.get("method")
    if not isinstance(method, (str, dict)) or (
        isinstance(method, str) and not method.strip()
    ) or (isinstance(method, dict) and not method):
        raise RuntimeError("method must be a non-empty string or object")
    summary = result.get("summary")
    if not isinstance(summary, dict) or not summary:
        raise RuntimeError("summary must be a non-empty object")
    evidence = result.get("evidence")
    if not isinstance(evidence, list) or any(
        not isinstance(item, dict) for item in evidence
    ):
        raise RuntimeError("evidence must be a list of objects")
    charts = result.get("charts")
    if not isinstance(charts, list) or any(
        not isinstance(item, str) or not item.strip() for item in charts
    ):
        raise RuntimeError("charts must be a list of file-path strings")
    for key in ("limitations", "recommendations"):
        values = result.get(key)
        if not isinstance(values, list) or any(
            not isinstance(item, str) or not item.strip() for item in values
        ):
            raise RuntimeError(f"{key} must be a list of non-empty strings")
    if "reproducibility" in result:
        raise RuntimeError(
            "reproducibility is reserved for trusted worker enrichment"
        )


def _validate_result_size(result: dict) -> None:
    result_size = len(
        json.dumps(
            result,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if result_size > MAX_RESULT_BYTES:
        raise RuntimeError("runner result JSON exceeds the result limit")


def main() -> None:
    def _timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError("candidate execution exceeded its wall-clock limit")

    if hasattr(signal, "SIGALRM") and EXECUTION_TIMEOUT_SECONDS > 0:
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(EXECUTION_TIMEOUT_SECONDS)
    input_path = sys.argv[1]
    output_dir = Path("/work/output")
    candidate_path = os.getenv("CODE_ANALYSIS_CANDIDATE_PATH", "/work/candidate/analysis.py")
    try:
        result, candidate_log = _execute_candidate(
            candidate_path,
            input_path,
            output_dir,
        )
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
    # The API deliberately limits the contract to data; arbitrary HTML is not
    # accepted as an execution result.
    _validate_result_contract(result)
    _validate_result_size(result)
    artifacts = _collect_outputs(result, output_dir)
    envelope = {
        "result": result,
        "artifacts": artifacts,
        "candidate_log": candidate_log,
    }
    encoded = json.dumps(envelope, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_OUTPUT_BYTES * 2:
        raise RuntimeError("runner result envelope exceeds the output limit")
    print(RESULT_PREFIX + encoded)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # Never let Python's default traceback renderer print an unbounded,
        # candidate-controlled exception value after redirects are restored.
        try:
            detail = str(exc).replace("\r", " ").replace("\n", " ")[:2_000]
        except Exception:
            detail = "unprintable candidate error"
        sys.stderr.write(f"runner error ({type(exc).__name__}): {detail}\n")
        raise SystemExit(1) from None
