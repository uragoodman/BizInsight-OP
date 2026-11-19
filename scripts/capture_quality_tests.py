"""Run pytest and persist a versioned, machine-readable regression summary."""
# ============================================================
# 质量证据采集脚本：运行全量 pytest 并落盘回归摘要
# 作用：执行一次测试运行，把结果（通过/失败/跳过/耗时/来源 commit）持久化为
#       data/quality/test-latest.json，供质量仪表盘展示。
# 关键原则：失败运行也保留证据，但不会覆盖最近一次成功的报告。
# ============================================================
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


# 项目根目录（本脚本位于 scripts/，向上 1 级）。
ROOT = Path(__file__).resolve().parents[1]


def _count(pattern: str, output: str) -> int:
    """从 pytest 汇总行中提取数字统计。

    函数作用：用正则从 pytest 输出里抓取如 "3 passed" 中的 3。
    输入：pattern —— 匹配目标计数项的正则；output —— pytest 完整输出。
    输出：int —— 提取到的数字；未匹配时返回 0（避免 None 参与统计）。
    主要流程：
        1. 用 re.search 在输出中查找首个匹配；
        2. 命中则把捕获组转换为 int，否则返回 0。
    """
    match = re.search(pattern, output)
    return int(match.group(1)) if match else 0


def main() -> None:
    """运行全量 pytest 并持久化机器可读的回归摘要。

    函数作用：执行一次完整测试运行，输出并落盘回归统计，退出码透传 pytest 结果。
    输入：无（pytest 根目录固定为 tests/）。
    输出：无（成功写入 data/quality/test-latest.json，失败额外写 test-last-failure.json）。
    主要流程：
        1. 在独立临时目录运行 `pytest tests` 并禁用缓存插件，规避历史遗留的
           Windows ACL 与只读 .pytest_cache 问题；
        2. 解析输出的 passed/failed/errors/skipped 数量，统计耗时与来源 commit；
        3. 成功运行写入 test-latest.json；失败运行写入 test-last-failure.json，
           保留最近一次成功报告，避免仪表盘出现 "0/0"；
        4. 失败时把最后 4000 字符输出作为诊断证据随报告保存，并在 stderr 打印；
        5. 以 pytest 的退出码结束脚本，保证 CI 能感知测试失败。
    难理解点：
        - 测试根目录必须显式写 "tests"：仓库 data/ 下保留历史 pytest 工作目录，
          若让 pytest 从 ROOT 递归收集，会把这些 Windows 不透明目录当作可收集路径。
        - 不复用项目内 basetemp：之前被提升权限/中止的 pytest 进程可能留下 Windows ACL，
          导致每个 tmp_path 测试都失败。
        - 禁用 cacheprovider：只读的 .pytest_cache 不应让原本有效的测试变成采集失败。
    """
    started = perf_counter()
    # Always name the test root explicitly.  The repository keeps historical
    # pytest working directories under data/, and letting pytest recurse from
    # ROOT makes those opaque Windows directories look like collectable paths.
    # Do not reuse a project-local basetemp: prior elevated/aborted pytest
    # processes can leave a Windows ACL that makes every tmp_path test fail.
    # The cache plugin is also disabled because a read-only .pytest_cache must
    # not turn otherwise valid tests into a failed quality capture.
    quality_temp_root = ROOT / "data" / "quality"
    quality_temp_root.mkdir(parents=True, exist_ok=True)
    # A newly-created workspace-local root avoids the shared Windows temp
    # directory, which may have been ACL'd by an elevated historical run.
    with tempfile.TemporaryDirectory(prefix="pytest-capture-", dir=quality_temp_root, ignore_cleanup_errors=True) as base_temp:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "--basetemp", base_temp],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
    output = f"{result.stdout}\n{result.stderr}"
    # 记录来源 commit，便于关联「这份测试结果来自哪个提交」。
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        sha = "unknown"
    # 从 pytest 汇总输出解析各统计量，组装机器可读摘要。
    summary = {
        "captured_at": datetime.now(timezone.utc).isoformat(), "commit_sha": sha,
        "exit_code": result.returncode, "passed": _count(r"(\d+) passed", output),
        "failed": _count(r"(\d+) failed", output), "errors": _count(r"(\d+) errors?", output),
        "skipped": _count(r"(\d+) skipped", output), "executed": _count(r"(\d+) passed", output) + _count(r"(\d+) failed", output) + _count(r"(\d+) errors?", output) + _count(r"(\d+) skipped", output),
        "duration_seconds": round(perf_counter() - started, 3),
    }
    if result.returncode:
        # Keep a bounded diagnostic in the artifact: enough for a dashboard/CLI
        # investigation, without turning a test-result record into a raw log store.
        # 失败时附带最后 4000 字符输出作为诊断证据（有限体积，不整份存日志）。
        summary["failure_tail"] = output[-4000:]
    target = ROOT / "data" / "quality" / "test-latest.json"; target.parent.mkdir(parents=True, exist_ok=True)
    # A broken run is diagnostic evidence, not a regression result.  Preserve
    # the most recent successful report so the dashboard never claims "0/0".
    # 失败运行不覆盖最近成功报告：改写到 test-last-failure.json，避免仪表盘出现 "0/0"。
    if result.returncode:
        target = ROOT / "data" / "quality" / "test-last-failure.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if result.returncode:
        print("\npytest failed; last output follows:\n" + output[-4000:], file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
