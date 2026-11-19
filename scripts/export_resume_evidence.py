"""Create a concise, auditable evidence packet for a resume or interview.

This script reads already-captured evidence only.  It does not execute tests,
change business data, or turn unavailable measurements into claims.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "data" / "quality"
EVALUATIONS = ROOT / "data" / "evaluations"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def latest_usage() -> tuple[Path | None, dict[str, Any] | None]:
    candidates = sorted(QUALITY.glob("usage-*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in candidates:
        value = read_json(path)
        if value:
            return path, value
    return None, None


def evidence_commits(items: dict[str, dict[str, Any] | None]) -> dict[str, str]:
    return {name: str(value.get("commit_sha")) for name, value in items.items() if value and value.get("commit_sha")}


def workload(benchmark: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if not benchmark:
        return None
    return next((item for item in benchmark.get("results", []) if item.get("workload") == name), None)


def _json_safe(value: Any) -> Any:
    """Convert non-finite floats (e.g. limit_ms=inf) to null for strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _collect_implementation() -> tuple[dict[str, Any], list[str]]:
    """Count capability totals from live code so resume numbers track reality.

    agent_roles stays a fixed business口径 (6 specialist roles: triage,
    metric_definition, data_query, root_cause, report, critic) as defined in
    docs/project-evidence.md — the graph also contains helper nodes that are
    intentionally not counted, so there is no code enum to derive it from.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    counts: dict[str, Any] = {
        "agent_roles": 6,
        "agent_role_names": ["triage", "metric_definition", "data_query", "root_cause", "report", "critic"],
    }
    warnings: list[str] = []
    try:
        from app.metrics.registry import get_registry
        counts["registered_metrics"] = len(get_registry().list())
    except Exception as exc:  # pragma: no cover - environment dependent
        counts["registered_metrics"] = None
        warnings.append(f"指标注册数统计失败（{exc}）；registered_metrics 置为未知。")
    try:
        import app.analysis.templates  # noqa: F401  # import triggers @register side effects
        from app.analysis.templates.registry import list_templates
        counts["complex_analysis_templates"] = len(list_templates())
    except Exception as exc:  # pragma: no cover - environment dependent
        counts["complex_analysis_templates"] = None
        warnings.append(f"分析模板数统计失败（{exc}）；complex_analysis_templates 置为未知。")
    router_ts = ROOT / "web" / "src" / "router.ts"
    if router_ts.exists():
        text = router_ts.read_text(encoding="utf-8")
        counts["main_console_views"] = len(re.findall(r"component:\s*\(\s*\)\s*=>\s*import\(", text))
    else:
        counts["main_console_views"] = None
        warnings.append("未找到 web/src/router.ts；main_console_views 置为未知。")
    return counts, warnings


def markdown(packet: dict[str, Any]) -> str:
    quality = packet["quality"]
    lines = [
        "# BizInsight 简历证据包",
        "",
        f"- 生成时间：{packet['generated_at']}",
        f"- 当前代码提交：{packet['current_commit']}",
        f"- 证据状态：{'可用于简历' if packet['eligible_for_resume'] else '需先重跑或核对，不建议对外引用'}",
        "",
        "## 可引用数据",
        "",
    ]
    impl = packet["implementation"]
    impl_display = {
        key: ("未知" if impl.get(key) is None else impl.get(key))
        for key in ("registered_metrics", "complex_analysis_templates", "agent_roles")
    }
    lines.append(
        f"- 受控经营指标：{impl_display['registered_metrics']} 个；"
        f"复杂分析模板：{impl_display['complex_analysis_templates']} 类；"
        f"核心 Agent：{impl_display['agent_roles']} 类。"
    )
    golden_line = quality["golden"]
    lines.append(f"- Golden：{golden_line.get('passed') if golden_line else '未采集'}/{golden_line.get('cases') if golden_line else '未采集'}。")
    tests_line = quality["tests"]
    lines.append(
        f"- Python 回归：{tests_line.get('passed') if tests_line else '未采集'} passed / "
        f"{tests_line.get('failed') if tests_line else '未采集'} failed / "
        f"{tests_line.get('skipped') if tests_line else '未采集'} skipped。"
    )
    benchmark = quality["benchmark"]
    if benchmark:
        lines.append(f"- PostgreSQL 基准数据：{benchmark.get('fact_rows', '未采集')} facts，并发 {benchmark.get('environment', {}).get('concurrency', '未采集')}。")
        for item in benchmark.get("results", []):
            lines.append(f"  - {item.get('workload')}: p95 {item.get('p95_ms')} ms，{item.get('runs')} 次。")
    else:
        lines.append("- PostgreSQL 端到端基准：未采集，不能引用性能数字。")
    retrieval = quality.get("retrieval")
    if retrieval and retrieval.get("rule"):
        rule = retrieval["rule"]
        lines.append(f"- RAG 检索效果（{retrieval.get('cases', '?')} 案例，rule 基线）：doc_hit@5 {rule.get('doc_hit@5')}，MRR@5 {rule.get('mrr@5')}，p95 {rule.get('p95_ms')} ms。")
    else:
        lines.append("- RAG 检索效果：未采集，不能引用 hit@k。")
    triage_quality = quality.get("triage")
    if triage_quality:
        lines.append(f"- Triage 路由准确率：{triage_quality.get('passed')}/{triage_quality.get('cases')}（{triage_quality.get('accuracy')}，离线降级模式）。")
    else:
        lines.append("- Triage 路由准确率：未采集。")
    usage = quality["usage"]
    if usage and usage.get("status") == "available":
        total = usage.get("total_tokens", {})
        lines.append(f"- Provider 用量：{usage.get('traces')} 条 trace；单 trace total token p50/p95 = {total.get('p50')}/{total.get('p95')}。")
    else:
        lines.append("- Provider 用量：未采集或不可用，不能引用 token/成本。")
    lines += ["", "## 不能夸大的边界", ""]
    lines += [f"- {item}" for item in packet["boundaries"]]
    if packet["warnings"]:
        lines += ["", "## 需处理的告警", ""]
        lines += [f"- {item}" for item in packet["warnings"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=QUALITY / "resume-evidence-latest.json")
    parser.add_argument("--markdown-output", type=Path, default=QUALITY / "resume-evidence-latest.md")
    args = parser.parse_args()

    final = read_json(QUALITY / "final-evidence-latest.json")
    golden = read_json(EVALUATIONS / "latest.json")
    tests = read_json(QUALITY / "test-latest.json")
    benchmark = read_json(QUALITY / "benchmark-latest.json")
    retrieval = read_json(QUALITY / "rag-retrieval-latest.json")
    triage_routing = read_json(QUALITY / "triage-routing-latest.json")
    usage_path, usage = latest_usage()
    sources = {"final_evidence": final, "golden": golden, "tests": tests, "benchmark": benchmark, "retrieval": retrieval, "triage": triage_routing, "usage": usage}
    commits = evidence_commits(sources)
    expected = current_commit()
    warnings: list[str] = []
    if not final or not final.get("success"):
        warnings.append("最终证据采集缺失或失败；先运行 scripts.collect_final_evidence --full。")
    for source, sha in commits.items():
        if expected != "unknown" and not expected.startswith(sha) and not sha.startswith(expected[:7]):
            warnings.append(f"{source} 来自提交 {sha}，与当前提交 {expected[:12]} 不一致。")
    if not benchmark:
        warnings.append("未找到 PostgreSQL benchmark；不要引用端到端延迟或数据规模。")
    if not usage or usage.get("status") != "available":
        warnings.append("没有可用 provider 用量样本；不要引用 token、成本或调用量。")
    if not retrieval:
        warnings.append("未找到 RAG 检索效果证据；不要引用 hit@k。")
    if not triage_routing:
        warnings.append("未找到 Triage 路由准确率证据。")

    implementation, impl_warnings = _collect_implementation()
    warnings.extend(impl_warnings)

    packet = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_commit": expected,
        "eligible_for_resume": not warnings,
        "source_files": {
            "final_evidence": str(QUALITY / "final-evidence-latest.json"),
            "golden": str(EVALUATIONS / "latest.json"),
            "tests": str(QUALITY / "test-latest.json"),
            "benchmark": str(QUALITY / "benchmark-latest.json"),
            "retrieval": str(QUALITY / "rag-retrieval-latest.json"),
            "triage": str(QUALITY / "triage-routing-latest.json"),
            "usage": str(usage_path) if usage_path else None,
        },
        "source_commits": commits,
        "implementation": implementation,
        "quality": {
            "final_evidence_success": final.get("success") if final else None,
            "golden": None if not golden else {key: golden.get(key) for key in ("cases", "passed", "overall_rate", "dataset_version", "generated_at")},
            "tests": None if not tests else {key: tests.get(key) for key in ("passed", "failed", "errors", "skipped", "executed", "duration_seconds", "captured_at")},
            "benchmark": None if not benchmark else {"fact_rows": benchmark.get("fact_rows"), "environment": benchmark.get("environment"), "workloads": [workload(benchmark, name) for name in ("metric-period-90d-x15", "period-contribution-x5", "weekly-monthly-report")]},
            "retrieval": None if not retrieval else {"cases": retrieval.get("results", {}).get("rule", {}).get("cases"), "rule": {key: retrieval.get("results", {}).get("rule", {}).get(key) for key in ("doc_hit@5", "chunk_hit@5", "mrr@5", "hit@k", "p50_ms", "p95_ms")}, "dataset_ref": retrieval.get("dataset_ref")},
            "triage": None if not triage_routing else {key: triage_routing.get(key) for key in ("cases", "passed", "accuracy", "mode")},
            "usage": None if not usage else {key: usage.get(key) for key in ("status", "period_days", "traces", "total_tokens", "estimated_cost", "captured_at")},
        },
        "boundaries": [
            "Golden 与 fixture benchmark 是离线确定性证据，不代表真实用户正确率、线上吞吐或业务收益。",
            "RAG hit@k 与 Triage 路由准确率是离线确定性评估，不代表线上检索效果或真实模型路由准确率。",
            "真实 OpenCode、Qdrant 服务端端到端和 RAG Shadow/Canary 需要独立部署验收，未采集时不得表述为已上线。",
            "只有 provider 实际回报的 usage 才可引用 token 或成本；无样本应保持 unavailable。",
        ],
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(packet), ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown_output.write_text(markdown(packet), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown_output), "eligible_for_resume": packet["eligible_for_resume"], "warnings": warnings}, ensure_ascii=False))


if __name__ == "__main__":
    main()
