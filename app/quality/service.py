"""Collect only auditable project-quality evidence; never invent missing values."""
# ============================================================
# 质量证据聚合服务
# 作用：把可审计的项目质量证据（Golden 评测、PostgreSQL 性能基准、单元测试、
#       Trace 摘要、LLM 用量）聚合为统一快照，供质量仪表盘展示。
# 核心原则：只读真实产生/持久化的数据，绝不编造缺失值；缺失指标一律返回 None。
# 权限模型：快照按租户作用域区分——离线评测证据是全局的，运行时 Trace/用量按租户隔离。
# ============================================================
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from app.core.config import get_settings
from app.harness.storage import list_traces, read_trace
from app.tenancy import TenantContext
from app.usage.service import usage_overview


# 项目根目录（本文件位于 app/quality/，向上 2 级）。
ROOT = Path(__file__).resolve().parents[2]
# 质量证据落盘目录：data/quality/（存放 benchmark/test 证据与快照历史）。
_quality_data_dir = Path(get_settings().quality_data_dir)
QUALITY_DIR = _quality_data_dir if _quality_data_dir.is_absolute() else ROOT / _quality_data_dir


def _read_json(path: Path) -> dict[str, Any] | None:
    """安全读取 JSON 文件。

    函数作用：读取证据文件并保证返回 dict；文件不存在或内容损坏时返回 None。
    输入：path —— 目标 JSON 文件路径。
    输出：dict —— 解析后的对象；或 None —— 读取失败 / 内容不是对象。
    主要流程：
        1. 尝试以 UTF-8 读取并解析 JSON；
        2. 仅当顶层是 dict 时返回，否则按失败处理返回 None；
        3. 捕获 OSError（文件不存在）与 JSONDecodeError（内容损坏），统一返回 None。
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        # 仅当顶层是 dict 时才视为有效证据。
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        # 文件缺失或损坏都按「无证据」处理，不中断聚合流程。
        return None


def _git_sha() -> str:
    """返回当前提交的短 SHA（用于给证据标注来源版本）。

    函数作用：把快照关联到具体代码版本，便于回溯「这份证据来自哪个提交」。
    输入：无。
    输出：str —— 如 "d5132f8"；git 调用失败时返回 "unknown"。
    主要流程：
        1. 在项目根目录执行 git rev-parse --short HEAD；
        2. 取 stdout 并去除首尾空白；
        3. 任何异常（非 git 仓库、未安装 git）都回退为 "unknown"，不中断聚合流程。
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        # 非 git 仓库等场景回退为 unknown，保证聚合流程可用。
        return "unknown"


def _percentile(values: list[float], ratio: float) -> float | None:
    """计算已收集样本的分位数（用于 p50/p95 延迟）。

    函数作用：给出真实样本的分位点，避免用均值掩盖长尾延迟。
    输入：values —— 非空耗时样本列表；ratio —— 目标分位，如 0.5 / 0.95。
    输出：float —— 保留 3 位小数的分位值；空列表返回 None。
    难理解点：int(len * ratio + .999999) - 1 是把比例换算成「至少覆盖该比例」的
    下标（等价于向上取整后减一），再用 max/min 夹取到合法下标范围，防止越界。
    """
    if not values:
        return None
    ordered = sorted(values)
    # 下标换算见 docstring：按 ratio 取「至少覆盖该比例」的位置并夹取到合法范围。
    return round(ordered[min(len(ordered) - 1, max(0, int((len(ordered) * ratio + .999999)) - 1))], 3)


def _trace_bucket(trace: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """为一条 Trace 计算版本分桶键与展示维度。

    函数作用：质量指标只在「相同版本分代」内比较才有效。该函数把每条 Trace
    归入其所属的分代桶，旧版（无 trace_version）Trace 归入 legacy 桶。
    输入：trace —— 已读取的 Trace 顶层字典（含 trace_version 字段）。
    输出：tuple[str, dict[str,str]] —— (分桶键, 六个版本维度字典)。
    难理解点：
        - schema_version 不参与分桶键：它标记的是外层存储结构版本（迁移/兼容用），
          不代表业务语义差异，因此排除在 key 之外。
        - 维度值缺失时统一填 "unknown"，保证分桶字符串稳定可比较。
    """
    version = trace.get("trace_version")
    if not isinstance(version, dict):
        # 无版本信息的历史 Trace：归入 legacy 桶，单独展示不作对比。
        return "legacy", {"generation": "legacy", "schema_version": "unversioned"}
    dimensions = {
        key: str(version.get(key) or "unknown")
        for key in ("trace_kind", "generation", "schema_version", "workflow_version", "prompt_version", "rag_release_version", "permission_policy_version")
    }
    # 用业务版本维度拼成稳定分桶键（schema_version 属存储结构，不参与）。
    key = "|".join(f"{name}={dimensions[name]}" for name in ("trace_kind", "generation", "workflow_version", "prompt_version", "rag_release_version", "permission_policy_version"))
    return key, dimensions


def _duration_ms(state: dict[str, Any]) -> float | None:
    """累加单条 Trace 的节点耗时。

    函数作用：从 state 的 node_timings 中汇总整张图的执行耗时。
    输入：state —— Trace 的 state_snapshot 字典。
    输出：float —— 总耗时毫秒数；无有效样本时返回 None。
    难理解点：node_timings 兼容两种历史格式——list（新格式，元素为含 duration_ms
    的 dict）与 dict（旧格式，值直接是毫秒数），两种都先过滤出数值再求和。
    """
    timings = state.get("node_timings")
    values: list[float] = []
    if isinstance(timings, list):
        # 新格式：元素为含 duration_ms 的 dict，逐个抽取。
        for item in timings:
            if isinstance(item, dict) and isinstance(item.get("duration_ms"), (int, float)):
                values.append(float(item["duration_ms"]))
    elif isinstance(timings, dict):
        # 旧格式：值直接是毫秒数。
        values = [float(value) for value in timings.values() if isinstance(value, (int, float))]
    duration = sum(values)
    # 无有效样本时返回 None，避免把 0 误当成真实耗时。
    return duration if duration > 0 else None


def _decision_reasons(state: dict[str, Any]) -> dict[str, int]:
    """聚合一条 Trace 的结构化决策/错误码。

    函数作用：把 Critic 裁决里的非正常字段与运行时错误码统计成可量化的问题分布，
    用于仪表盘展示「最常见的打回/失败原因」。
    输入：state —— state_snapshot 字典。
    输出：dict[str, int] —— 形如 {"data_status:stale": 1, "error:budget": 2} 的计数。
    难理解点：只取 critic_decisions 的最后一条（最新一轮裁决），且只统计「非 ok 值」
    ——正常通过的字段不进分布，避免把健康状态误计为问题。
    """
    reasons: dict[str, int] = {}
    decisions = state.get("critic_decisions")
    if isinstance(decisions, list) and decisions and isinstance(decisions[-1], dict):
        # 只统计最新一轮裁决中的「非正常」字段值。
        decision = decisions[-1]
        for field in ("revision_target", "rag_status", "data_status", "claim_status"):
            value = decision.get(field)
            if value and value not in {"ok", "approved", "pass"}:
                label = f"{field}:{value}"
                reasons[label] = reasons.get(label, 0) + 1
        issues = decision.get("llm_issues")
        if isinstance(issues, list) and issues:
            reasons["critic:llm_issues"] = len(issues)
    errors = state.get("errors")
    if isinstance(errors, list):
        # 把运行时错误码也并入原因分布。
        for error in errors:
            code = error.get("code") if isinstance(error, dict) else None
            if code:
                label = f"error:{code}"
                reasons[label] = reasons.get(label, 0) + 1
    return reasons


def _summarize_bucket(key: str, dimensions: dict[str, str], entries: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总一个版本分桶内的全部 Trace 为质量指标。

    函数作用：对同一分代桶里的每条 Trace 统计 Critic 裁决分布、完成率、
    覆盖率、延迟分位与问题原因分布，构成仪表盘的一个数据块。
    输入：
        key        —— 分桶键（用于前端标识当前块）；
        dimensions —— 该桶的版本维度；
        entries    —— 桶内所有 Trace 的 state_snapshot 列表。
    输出：dict —— 含 traces/completion_rate/critic_verdicts/graph_latency_ms 等字段的汇总。
    难理解点：verdict 为 "missing" 表示该 Trace 没有 Critic 裁决（未走到审核），
    它不进入「已覆盖」统计，但计入总数，从而拉低 critic_coverage_rate。
    """
    verdicts: dict[str, int] = {}
    durations: list[float] = []
    reasons: dict[str, int] = {}
    completed = 0
    covered = 0
    for state in entries:
        # verdict 为 missing 表示未走到 Critic 审核，不计入覆盖率。
        verdict = str(state.get("critic_verdict") or "missing")
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        if verdict != "missing":
            covered += 1
        if verdict in {"approved", "pass", "human_review", "rejected", "fail"}:
            completed += 1
        duration = _duration_ms(state)
        if duration is not None:
            durations.append(duration)
        for reason, count in _decision_reasons(state).items():
            reasons[reason] = reasons.get(reason, 0) + count
    total = len(entries)
    rejected = verdicts.get("rejected", 0) + verdicts.get("fail", 0)
    accepted = verdicts.get("approved", 0) + verdicts.get("pass", 0) + verdicts.get("human_review", 0)
    return {
        "bucket": key,
        "dimensions": dimensions,
        "traces": total,
        "completed_traces": completed,
        "completion_rate": round(completed / total, 4) if total else None,
        "critic_coverage_rate": round(covered / total, 4) if total else None,
        "critic_verdicts": verdicts,
        "critic_accept_rate": round(accepted / covered, 4) if covered else None,
        "critic_reject_rate": round(rejected / covered, 4) if covered else None,
        "critic_revision_rate": round(verdicts.get("needs_revision", 0) / covered, 4) if covered else None,
        "graph_latency_ms": {"samples": len(durations), "p50": _percentile(durations, .5), "p95": _percentile(durations, .95)},
        "reason_counts": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[:12]),
    }


def trace_summary(context: TenantContext) -> dict[str, Any]:
    """汇总当前租户可见的全部 Trace，并选出「当前分代」作为默认展示块。

    函数作用：遍历租户 Trace，按版本分桶汇总，返回分桶列表 + 一个被选中的主块。
    输入：context —— 租户上下文（决定列出哪些 Trace）。
    输出：dict —— 含 buckets（全部分桶）、selected_bucket（默认选中块）及其统计字段。
    主要流程：
        1. 遍历 list_traces() 逐条读取，解析 state_snapshot 并跳过非 dict 的脏数据；
        2. 用 _trace_bucket 分桶，同桶 Trace 聚合到同一组；
        3. 对每个桶调用 _summarize_bucket 汇总；
        4. 排序时把 legacy 桶放在最后，其余按桶键排序；
        5. 取第一个桶（最新分代）作为选中块；无数据时返回一个空的 trace-v2 占位块。
    """
    grouped: dict[str, tuple[dict[str, str], list[dict[str, Any]]]] = {}
    for item in list_traces(context):
        trace = read_trace(str(item["trace_id"]), context) or {}
        state = trace.get("state_snapshot", {})
        # 跳过 state 非 dict 的脏 Trace，避免下游取值异常。
        if not isinstance(state, dict):
            continue
        key, dimensions = _trace_bucket(trace)
        grouped.setdefault(key, (dimensions, []))[1].append(state)
    buckets = [_summarize_bucket(key, dimensions, entries) for key, (dimensions, entries) in grouped.items()]
    buckets.sort(key=lambda item: (item["dimensions"].get("trace_kind") != "agent_chat", item["dimensions"].get("generation") == "legacy", item["bucket"]))
    # 不要把 buckets 挂到列表内的同一对象上：FastAPI 会把响应序列化为 JSON，
    # 自引用会导致 500 错误，因此先对选中块做一次浅拷贝再追加 buckets。
    selected = dict(buckets[0]) if buckets else _summarize_bucket("trace-v2", {"generation": "trace-v2"}, [])
    # 保留旧字段名以兼容已保存的历史快照，但语义已澄清：
    # terminal_rate 从来都不是真正的「终止率」，只是完成率的历史别名。
    selected["terminal_traces"] = selected["completed_traces"]
    selected["terminal_rate"] = selected["completion_rate"]
    selected["buckets"] = buckets
    selected["selected_bucket"] = selected["bucket"]
    return selected


def insight_summary() -> dict[str, Any]:
    """聚合洞察制品质量指标（平台 SQLite 路径）。

    返回值：dict，含 samples / success_rate / whitelist_reject_rate / critic_reject_rate /
    pending_review_count / latency_ms / tokens / cost。存储不可用时返回 {"samples": 0}。

    指标口径：
    - 成功率 = (draft + approved + published) / 终态总数（不含 generating）
    - 白名单拒绝率 = rejected 且含 uncited_evidence / 终态总数
    - Critic 驳回率 = rejected 且 critic_verdict=rejected / 终态总数
    - 延迟 p50/p95 来自 duration_ms；token/成本取每份洞察均值。
    """
    from app.analysis.publishing import ReportStore

    try:
        rows = ReportStore(get_settings().phase6_store_path).list_all_insights()
    except Exception:  # noqa: BLE001 - 存储不可用时返回空证据，不中断快照
        return {"samples": 0}
    terminal = [row for row in rows if row.get("status") != "generating"]
    total = len(terminal)
    if not total:
        return {"samples": 0}

    def count(pred: Any) -> int:
        return sum(1 for row in terminal if pred(row))

    success = count(lambda r: r.get("status") in {"draft", "approved", "published"})
    whitelist_reject = count(
        lambda r: r.get("status") == "rejected"
        and any("uncited_evidence" in str(item) for item in (r.get("validation_errors") or []))
    )
    critic_reject = count(
        lambda r: r.get("status") == "rejected" and r.get("critic_verdict") == "rejected"
    )
    durations = [float(r["duration_ms"]) for r in terminal if isinstance(r.get("duration_ms"), (int, float))]
    tokens = [(int(r.get("input_tokens") or 0), int(r.get("output_tokens") or 0)) for r in terminal if r.get("input_tokens") is not None]
    costs = [float(r.get("estimated_cost")) for r in terminal if isinstance(r.get("estimated_cost"), (int, float))]
    total_tokens = sum(input_tokens + output_tokens for input_tokens, output_tokens in tokens)
    return {
        "samples": total,
        "success_rate": round(success / total, 4),
        "whitelist_reject_rate": round(whitelist_reject / total, 4),
        "critic_reject_rate": round(critic_reject / total, 4),
        "pending_review_count": count(lambda r: r.get("status") == "pending_review"),
        "latency_ms": {"samples": len(durations), "p50": _percentile(durations, .5), "p95": _percentile(durations, .95)},
        "tokens": {"samples": len(tokens), "total": total_tokens, "avg_per_insight": round(total_tokens / len(tokens), 1) if tokens else None},
        "cost": {"samples": len(costs), "total": round(sum(costs), 6), "avg_per_insight": round(sum(costs) / len(costs), 6) if costs else None},
    }


def build_snapshot(context: TenantContext) -> dict[str, Any]:
    """组装一份完整的质量快照（不落盘）。

    函数作用：把当前时刻的离线评测证据 + 运行时数据合并成仪表盘可渲染的快照。
    输入：context —— 租户上下文，决定快照 scope（平台/租户）与 Trace/用量范围。
    输出：dict —— schema_version=2 的完整快照对象。
    主要流程：
        1. 读取 Golden（data/evaluations/latest.json）、性能基准与测试证据三个文件；
        2. 计算当前 git 短 SHA 作为来源版本；
        3. 按 context 是否为平台管理员决定 scope 是 "platform" 还是 "tenant"；
        4. 只取 Golden 中预先白名单化的字段（不含原始用例明细，避免泄露敏感数据）；
        5. 组装 trace_summary 与近 30 天用量摘要。
    """
    golden = _read_json(ROOT / "data" / "evaluations" / "latest.json")
    benchmark = _read_json(QUALITY_DIR / "benchmark-latest.json")
    tests = _read_json(QUALITY_DIR / "test-latest.json")
    return {
        "schema_version": 2,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": _git_sha(),
        "scope": "platform" if context.is_platform_admin else "tenant",
        "tenant_id": context.tenant_id,
        "user_id": context.user_id,
        # Golden 只暴露白名单字段，不含原始用例明细。
        "golden": None if not golden else {key: golden.get(key) for key in ("generated_at", "commit_sha", "dataset_version", "cases", "passed", "overall_rate")},
        "benchmark": benchmark,
        "tests": tests,
        "traces": trace_summary(context),
        "usage": usage_overview(tenant_id=context.tenant_id, user_id=context.user_id, days=30, limit=1)["summary"],
        # The insight store is currently global and has no tenant filter. Do
        # not expose its aggregate counts to a tenant dashboard until it can
        # be queried with an explicit tenant scope.
        "insights": insight_summary() if context.is_platform_admin else {"samples": 0},
    }


def _snapshot_path(*, create: bool = True) -> Path:
    """返回快照历史文件的路径，并在目录不存在时自动创建。

    函数作用：集中管理快照落盘位置。
    输入：无。
    输出：Path —— data/quality/snapshots.json。
    """
    # Reading the dashboard must remain read-only.  In the production image
    # the evidence directory may be mounted read-only (or may not exist yet),
    # while POST /api/quality/snapshots is the operation that needs to create
    # it.
    if create:
        QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    return QUALITY_DIR / "snapshots.json"


def capture_snapshot(context: TenantContext) -> dict[str, Any]:
    """捕获一次质量快照并持久化到历史文件。

    函数作用：调用 build_snapshot 生成当前快照，追加到 snapshots.json 的历史列表。
    输入：context —— 租户上下文。
    输出：dict —— 刚捕获的快照对象。
    主要流程：
        1. 生成当前快照；
        2. 读取已有历史（无文件时初始化为空 items 列表），过滤掉非 dict 脏数据；
        3. 追加新快照并只保留最近 60 条，防止文件无限膨胀；
        4. 用 default=str 序列化：SQLAlchemy 聚合成本可能是 Decimal，统一转字符串。
    """
    snapshot = build_snapshot(context)
    path = _snapshot_path()
    existing = _read_json(path) or {"items": []}
    items = [item for item in existing.get("items", []) if isinstance(item, dict)]
    items.append(snapshot)
    # 只保留最近 60 条快照，防止历史文件无限膨胀；default=str 兼容 Decimal 成本。
    path.write_text(json.dumps({"items": items[-60:]}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return snapshot


def dashboard(context: TenantContext) -> dict[str, Any]:
    """组装质量仪表盘接口的返回体。

    函数作用：提供「最新快照 + 最近历史」给 /api/quality 接口。
    输入：context —— 租户上下文。
    输出：dict —— {latest: 最新快照, history: 最近 20 条历史快照}。
    主要流程：
        1. 生成最新快照；
        2. 读取已保存的历史快照列表；
        3. 返回最新快照与最近 20 条历史（超出部分不返回，控制响应体积）。
    """
    latest = build_snapshot(context)
    stored = _read_json(_snapshot_path(create=False)) or {"items": []}
    expected_scope = "platform" if context.is_platform_admin else "tenant"
    history = [
        item for item in stored.get("items", [])
        if isinstance(item, dict)
        and item.get("scope") == expected_scope
        and item.get("tenant_id") == context.tenant_id
        and item.get("user_id") == context.user_id
    ]
    # 只返回最近 20 条历史，控制响应体积。
    return {"latest": latest, "history": history[-20:]}
