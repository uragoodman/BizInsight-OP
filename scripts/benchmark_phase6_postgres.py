"""Read-only Phase 6 benchmark against a populated PostgreSQL business dataset."""
# ============================================================
# Phase 6 PostgreSQL 只读性能基准
# 作用：在已灌入真实业务形态数据的 PostgreSQL 上，度量三类查询的 p50/p95 延迟：
#       (1) 15 个指标的 90 天周期查询；(2) GMV 的维度贡献分析；(3) 周报/月报生成。
# 用法：python -m scripts.benchmark_phase6_postgres
# 门槛：结果不满足环境变量限值时抛 RuntimeError；通过时把证据写入
#       data/quality/benchmark-latest.json 供质量仪表盘展示。
# ============================================================
from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
import json
from math import ceil
import os
import platform
from pathlib import Path
from statistics import median
from time import perf_counter
from datetime import datetime, timezone
from subprocess import check_output
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from app.analysis.contracts import PeriodSpec
from app.analysis.periodic_reports import build_periodic_report
from app.analysis.periods import previous_period
from app.analysis.semantic import METRIC_REGISTRY
from app.data.database import engine
from app.data.repositories.daily_metrics import (
    compare_breakdown_by_dimension_period,
    get_latest_metric_date,
    query_metric_period,
)


def percentile(values: list[float], ratio: float) -> float:
    """计算真实样本的分位数。

    函数作用：供 p50/p95 延迟统计使用，避免均值掩盖长尾延迟。
    输入：values —— 耗时样本列表；ratio —— 目标分位（如 0.5 / 0.95）。
    输出：float —— 排序后对应下标的分位值。
    难理解点：ceil(len*ratio)-1 把比例换算成「至少覆盖该比例」的下标，再用
    max/min 夹取到合法范围，防止越界。
    """
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, ceil(len(ordered) * ratio) - 1))]


def timed(operation) -> tuple[object, float]:
    """执行可调用对象并返回其结果与耗时（毫秒）。

    函数作用：统一测量某个查询/报告构建函数的耗时。
    输入：operation —— 零参可调用对象。
    输出：tuple —— (原始结果, 耗时毫秒)。
    主要流程：
        1. 记录 perf_counter 起点；
        2. 调用 operation() 得到结果；
        3. 返回结果与耗时差（转毫秒）。
    """
    started = perf_counter()
    result = operation()
    return result, (perf_counter() - started) * 1000


def natural_periods(latest: date) -> tuple[PeriodSpec, PeriodSpec]:
    """以数据最新日期为锚点，生成上一个自然周与上一个自然月区间。

    函数作用：基准测试需要一个稳定的参考区间；这里不依赖「今天」，而是锚定数据集
    里最新的业务日期，使离线/历史数据集也能得到确定可复现的结果。
    输入：latest —— 数据最新日期。
    输出：tuple —— (最近一个完整自然周, 最近一个完整自然月)。
    难理解点：
        - 周区间取「含最新日期在内的最近完整周」：回退到最近的周日作为周结束。
        - 月区间若最新日期不是当月最后一天，则回退到上月末，保证总是完整自然月。
    """
    week_end = latest - timedelta(days=(latest.weekday() + 1) % 7)
    week = PeriodSpec(start_date=week_end - timedelta(days=6), end_date=week_end, granularity="week")
    last_day = latest.replace(day=monthrange(latest.year, latest.month)[1])
    month_end = latest if latest == last_day else latest.replace(day=1) - timedelta(days=1)
    month = PeriodSpec(start_date=month_end.replace(day=1), end_date=month_end, granularity="month")
    return week, month


def assert_limit(name: str, values: list[float], variable: str) -> dict[str, object]:
    """校验一组样本的 p95 是否超过门限，并返回统计摘要。

    函数作用：把某类查询的耗时样本转成统计行，同时执行性能门禁——p95 超限即抛错。
    输入：name —— 负载名；values —— 耗时样本列表；variable —— 门限对应的环境变量名。
    输出：dict —— 含 workload/runs/p50_ms/p95_ms/limit_ms 的统计摘要。
    主要流程：
        1. 从环境变量读取门限（未设置时默认 inf，即不设限）；
        2. 计算 p50/p95 并组装摘要；
        3. p95 超过门限时抛 RuntimeError，让 CI/调用方能感知性能回退。
    """
    limit = float(os.getenv(variable, "inf"))
    result = {
        "workload": name, "runs": len(values), "p50_ms": round(median(values), 3),
        "p95_ms": round(percentile(values, 0.95), 3), "limit_ms": limit if limit != float("inf") else None,
    }
    if result["p95_ms"] > limit:
        raise RuntimeError(f"{name} p95 {result['p95_ms']}ms exceeds {limit}ms")
    return result


def persist_quality_evidence(output: dict[str, object]) -> None:
    """把真实的 PostgreSQL 基准结果持久化为质量仪表盘可读的证据文件。

    函数作用：把基准输出 + 环境信息 + 来源 commit 写为 data/quality/benchmark-latest.json。
    输入：output —— 基准统计结果字典。
    输出：无（写入文件）。
    主要流程：
        1. 读取当前 git 短 SHA（失败则回退为 "unknown"）；
        2. 组装证据：捕获时间、commit、环境信息（后端/最小行数/并发数/是否生成图表）；
        3. 覆盖写入 data/quality/benchmark-latest.json。
    """
    try:
        commit_sha = check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit_sha = "unknown"
    target = ROOT / "data" / "quality" / "benchmark-latest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": commit_sha,
        "environment": {
            "backend": "postgresql",
            "minimum_fact_rows": int(os.getenv("PHASE6_POSTGRES_MIN_FACT_ROWS", "90000")),
            "concurrency": 1,
            "charts_enabled": False,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        **output,
    }
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """执行 PostgreSQL 性能基准并输出/持久化结果。

    函数作用：依次度量指标查询、维度贡献、报告生成三类负载的耗时，做门禁校验后输出。
    输入：无（环境变量控制门限：*_P95_MS 与最小数据行数）。
    输出：无（打印结果并写入证据文件；任一负载不达标则抛异常退出）。
    主要流程：
        1. 数据前置校验：fact_orders 行数不低于最小值，daily_metrics 非空；
        2. 确定参考区间：以最新业务日期推导自然周/月，扫描起点为最近 90 天；
        3. 遍历全部指标执行 90 天周期查询，逐条记录耗时并校验结果状态；
        4. 按 GMV 的允许维度执行维度贡献对比，逐维记录耗时；
        5. 对周/月区间各构建 5 次报告，记录耗时并校验周期一致性；
        6. 汇总三类负载的 p95 门禁结果，持久化证据并打印。
    """
    minimum_rows = int(os.getenv("PHASE6_POSTGRES_MIN_FACT_ROWS", "90000"))
    # 前置校验：真实数据行数必须达到门槛，否则基准结果没有代表性。
    with engine.connect() as connection:
        fact_rows = int(connection.execute(text("SELECT COUNT(*) FROM analytics.fact_orders")).scalar_one())
        database_name = str(connection.execute(text("SELECT current_database()")).scalar_one())
        database_version = str(connection.execute(text("SELECT version()")).scalar_one())
    if fact_rows < minimum_rows:
        raise RuntimeError(f"PostgreSQL fact_orders has {fact_rows} rows; at least {minimum_rows} required")
    latest_raw = get_latest_metric_date()
    # 指标表为空则无法建立参考区间，直接终止。
    if latest_raw is None:
        raise RuntimeError("analytics.daily_metrics is empty")
    latest = date.fromisoformat(latest_raw)
    # 以最新业务日期推导自然周/月，扫描起点为最近 90 天。
    week, month = natural_periods(latest)
    scan_start = latest - timedelta(days=89)

    # 负载 1：遍历当前周期查询支持的指标执行 90 天周期查询。
    metric_times = []
    skipped_metrics: list[str] = []
    for metric_name in METRIC_REGISTRY:
        try:
            result, duration = timed(lambda name=metric_name: query_metric_period(name, scan_start, latest))
        except NotImplementedError:
            # Keep the benchmark on the supported period surface instead of
            # failing the whole gate for metrics that only expose daily/domain
            # handlers today.
            skipped_metrics.append(metric_name)
            continue
        if not isinstance(result, dict) or result.get("status") not in {"ok", "not_found"}:
            raise RuntimeError(f"invalid PostgreSQL result for {metric_name}")
        metric_times.append(duration)

    # 负载 2：按 GMV 允许的维度执行周期贡献对比。
    baseline = previous_period(month)
    contribution_times = []
    for dimension in METRIC_REGISTRY["gmv"].allowed_dimensions:
        result, duration = timed(lambda dim=dimension: compare_breakdown_by_dimension_period(
            "gmv", month.start_date, month.end_date,
            baseline.start_date, baseline.end_date, dim, 10,
        ))
        if not isinstance(result, dict) or result.get("status") not in {"ok", "not_found"}:
            raise RuntimeError(f"invalid PostgreSQL contribution result for {dimension}")
        contribution_times.append(duration)

    # 负载 3：周报/月报各构建 5 次（关闭图表与自动丰富）。
    report_times = []
    for period in (week, month) * 5:
        report, duration = timed(lambda p=period: build_periodic_report(
            p, ["gmv", "order_count", "avg_order_value"],
            include_charts=False, auto_enrich=False,
        ))
        if report.period != period:
            raise RuntimeError("PostgreSQL report period mismatch")
        report_times.append(duration)

    output = {
        "backend": "postgresql", "fact_rows": fact_rows, "latest_metric_date": latest.isoformat(),
        "database": {"name": database_name, "version": database_version},
        "skipped_period_metrics": skipped_metrics,
        "results": [
            assert_limit(
                f"metric-period-90d-x{len(metric_times)}",
                metric_times, "PHASE6_POSTGRES_METRIC_P95_MS",
            ),
            assert_limit("period-contribution-x5", contribution_times, "PHASE6_POSTGRES_CONTRIBUTION_P95_MS"),
            assert_limit("weekly-monthly-report", report_times, "PHASE6_POSTGRES_REPORT_P95_MS"),
        ],
    }
    persist_quality_evidence(output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
