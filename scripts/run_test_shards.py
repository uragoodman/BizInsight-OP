"""Run declared test shards concurrently without pytest-xdist.

Manifest format is a JSON array. A pytest shard uses ``kind`` no-db/postgres,
``name``, ``paths`` and optional ``options``/``template``. A command shard uses
``kind: command`` plus ``argv`` and optional ``cwd``. Set ``serial: true`` for
outbox, template-mutating, or Resume artifact tests.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILED_RE = re.compile(r"^\s*FAILED\s+(.+?)(?:\s+-\s+|$)", re.MULTILINE)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SERIAL_HINTS = ("phase7_outbox", "outbox", "resume", "test_turn_resolver_wp2_db", "test_wp5_", "test_wp6_")


def _command(spec: dict[str, object]) -> tuple[list[str], Path, dict[str, str]]:
    kind = spec.get("kind")
    name = str(spec["name"])
    env = dict(os.environ)
    if kind in ("no-db", "postgres"):
        args = [sys.executable, str(ROOT / "scripts/run_test_shard.py"), "--shard", str(kind), "--name", name]
        if spec.get("template"):
            args += ["--template", str(spec["template"])]
        args += ["--", *[str(x) for x in spec.get("paths", [])], *[str(x) for x in spec.get("options", [])]]
        for key in spec.get("unset_env", []):
            env.pop(str(key), None)
        for key, value in dict(spec.get("env", {})).items():
            if value is None:
                env.pop(str(key), None)
            else:
                env[str(key)] = str(value)
        return args, ROOT, env
    if kind == "command":
        argv = [str(x) for x in spec["argv"]]
        if os.name == "nt" and argv and argv[0].lower() == "npm":
            argv[0] = shutil.which("npm.cmd") or shutil.which("npm") or argv[0]
        return argv, (ROOT / str(spec.get("cwd", "."))).resolve(), env
    raise ValueError(f"Unknown shard kind {kind!r} for {name}")


def _run(spec: dict[str, object]) -> dict[str, object]:
    name = str(spec["name"])
    started = time.perf_counter()
    command, cwd, env = _command(spec)
    try:
        completed = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    except OSError as exc:
        elapsed = time.perf_counter() - started
        return {"name": name, "returncode": 127, "elapsed": elapsed, "counts": [0, 0, 0, 0],
                "failure_ids": [], "summary": f"could not start command: {exc}", "output": str(exc)}
    elapsed = time.perf_counter() - started
    output = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
    summary_line = next((line for line in reversed(output.splitlines()) if re.search(r"\b(passed|failed|skipped|warnings?)\b", line)), "")
    counts = [int(re.search(rf"(\d+) {label}", summary_line).group(1)) if re.search(rf"(\d+) {label}", summary_line) else 0
              for label in ("passed", "failed", "skipped", "warnings?")]
    plain_output = ANSI_RE.sub("", output)
    failures = [item.strip() for item in FAILED_RE.findall(plain_output)]
    return {"name": name, "returncode": completed.returncode, "elapsed": elapsed, "counts": counts,
            "failure_ids": failures, "summary": summary_line, "output": output}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--report", type=Path, help="write exact shard results as JSON")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    specs = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(specs, list) or not specs:
        parser.error("manifest must be a non-empty JSON array")
    names = [str(item.get("name", "")) for item in specs]
    if not all(names) or len(set(names)) != len(names):
        parser.error("each shard needs a unique name")
    for spec in specs:
        joined = " ".join([str(spec.get("name", "")), *[str(x) for x in spec.get("paths", [])]]).lower()
        if any(hint in joined for hint in SERIAL_HINTS):
            spec["serial"] = True

    started = time.perf_counter()
    results: list[dict[str, object]] = []
    pending = list(specs)
    # Serial shards execute alone; ordinary shards fill the bounded pool.
    while pending:
        if pending[0].get("serial"):
            results.append(_run(pending.pop(0)))
            continue
        batch = []
        has_postgres_template_clone = False
        while pending and len(batch) < args.jobs and not pending[0].get("serial"):
            candidate = pending[0]
            if candidate.get("kind") == "postgres" and has_postgres_template_clone:
                break
            candidate = pending.pop(0)
            batch.append(candidate)
            if candidate.get("kind") == "postgres":
                # PostgreSQL permits only one CREATE DATABASE ... TEMPLATE at
                # a time for a given source. Keep DB clone shards serialized,
                # while allowing one to overlap a pure/mock or frontend shard.
                has_postgres_template_clone = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_run, spec) for spec in batch]
            results.extend(future.result() for future in futures)

    print("\nSHARD SUMMARY")
    for item in results:
        passed, failed, skipped, warnings = item["counts"]
        print(f"{item['name']}: rc={item['returncode']} passed={passed} failed={failed} skipped={skipped} warnings={warnings} elapsed={item['elapsed']:.2f}s")
        if item["summary"]:
            print(f"  pytest: {item['summary']}")
        if item["failure_ids"]:
            for test_id in item["failure_ids"]:
                print(f"  FAILED_ID: {test_id}")
        if item["returncode"]:
            print(f"  command output tail:\n{str(item['output'])[-5000:]}")
    total = time.perf_counter() - started
    print(f"PARALLEL_ELAPSED_SECONDS={total:.2f} JOBS={args.jobs}")
    if args.report:
        report_path = args.report if args.report.is_absolute() else ROOT / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({
            "manifest": str(manifest_path),
            "jobs": args.jobs,
            "elapsed_seconds": total,
            "shards": [{
                "name": item["name"],
                "returncode": item["returncode"],
                "elapsed_seconds": item["elapsed"],
                "passed": item["counts"][0],
                "failed": item["counts"][1],
                "skipped": item["counts"][2],
                "warnings": item["counts"][3],
                "failure_ids": item["failure_ids"],
                "summary": item["summary"],
            } for item in results],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"SHARD_REPORT={report_path}")
    return 1 if any(int(item["returncode"]) for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
