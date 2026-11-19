"""受控业务数据集预取：按 RLS 授权范围把业务事实导出为 Parquet 工件，供 Code Agent 分析。

治理边界（对应 docs/improvement-plan.md P2）：
- 只接受白名单 metric + 受控 scope（seller/日期），不接收原始 SQL、表名或任意筛选表达式。
- 数据由确定性层按 RLS 授权范围查询（query_scoped_trend），LLM 生成的代码只能读取这份工件，永不直连数据库。
- 工件经 code_analysis_files 生命周期（配额 / 过期 / 删除审计）管理，随 job 证据可追溯。
"""

from __future__ import annotations

import io
import json
import csv
from datetime import date
from hashlib import sha256
from typing import Any

from app.code_analysis.service import upload_file
from app.data.repositories.explorer import query_scoped_trend
from app.metrics.registry import get_registry
from app.schemas.explorer import ControlledAnalysisScope
from app.tenancy import TenantContext

try:  # pyarrow 是声明依赖；缺失时降级为 CSV（pandas 同样可读，治理属性不变）。
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False

# 预取查询模板版本：随口径/查询逻辑变更递增，写入工件元数据与审计。
QUERY_VERSION = "dataset-v1"

# 单次预取行数上限，防止把上传配额打满。
MAX_DATASET_ROWS = 200_000


def _serialize_dataset(rows: list[dict[str, Any]], filename_base: str) -> tuple[bytes, str, str]:
    """把业务数据序列化为可分析工件：优先 Parquet，无 pyarrow 时降级 CSV。"""
    if _HAS_PYARROW:
        table = pa.Table.from_pylist(rows)
        buf = pa.BufferOutputStream()
        pq.write_table(table, buf)
        return buf.getvalue().to_pybytes(), f"{filename_base}.parquet", "application/vnd.apache.parquet"
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8"), f"{filename_base}.csv", "text/csv"


def _scope_fingerprint(seller_id: str, start_date: date, end_date: date, metrics: list[str]) -> str:
    """由范围构成稳定指纹：同一 seller/日期/指标 → 同一指纹（幂等与审计依据）。"""
    raw = json.dumps(
        {"seller": seller_id, "start": str(start_date), "end": str(end_date), "metrics": sorted(metrics)},
        sort_keys=True,
    )
    return sha256(raw.encode()).hexdigest()[:16]


def prefetch_business_dataset(
    context: TenantContext,
    *,
    seller_id: str,
    start_date: date,
    end_date: date,
    metric_names: list[str],
) -> dict[str, Any]:
    """按受控范围预取业务数据为 Parquet 工件，返回 code_analysis_files.file_id。

    参数：
    - `context`：租户上下文（RLS 隔离 + seller 授权依据）。
    - `seller_id`：目标卖家，必须在 context.allowed_seller_ids 内。
    - `start_date` / `end_date`：查询窗口（闭区间）。
    - `metric_names`：白名单指标（来自 metric registry，且支持 seller 趋势查询）。

    返回值：含 file_id / scope_fingerprint / query_version / rows / sha256 的工件描述，
    可直接用于 create_job。

    异常：指标不在白名单 → ValueError；seller 未授权 → PermissionError；范围内无数据 → ValueError。
    """
    registry = get_registry()
    for name in metric_names:
        if not registry.supports(name, "trend", "seller"):
            raise ValueError(f"Metric {name!r} is not available for business dataset pre-fetch")
    if str(seller_id) not in context.allowed_seller_ids:
        raise PermissionError("You cannot pre-fetch another seller's business data")
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")

    scope = ControlledAnalysisScope(
        seller_id=str(seller_id),
        default_start_date=start_date, default_end_date=end_date,
        metric_name=metric_names[0], granularity="day",
        allowed_metrics=tuple(metric_names),
    )
    rows: list[dict[str, Any]] = []
    sources: set[str] = set()
    for metric in metric_names:
        # 确定性层按 RLS 授权范围查询，不信任调用方传来的任意 SQL。
        trend = query_scoped_trend(scope, metric, start_date=start_date, end_date=end_date)
        sources.add(str(trend.get("source_table") or "unknown"))
        for point in trend.get("points", []):
            rows.append({"metric_date": str(point.get("bucket") or ""), "metric_name": metric, "value": point.get("value")})
    if not rows:
        raise ValueError("No business data available in the requested scope")
    if len(rows) > MAX_DATASET_ROWS:
        raise ValueError(f"Business dataset exceeds the row limit ({MAX_DATASET_ROWS})")

    # 元数据随工件落库：范围指纹 + 查询版本 + 行数 + 来源（进入 job 证据与最终报告）。
    fingerprint = _scope_fingerprint(str(seller_id), start_date, end_date, metric_names)
    raw, filename, content_type = _serialize_dataset(rows, f"biz_{fingerprint}")

    # 复用 upload_file 生命周期：配额 / 过期 / magic bytes 校验 / 删除审计。
    uploaded = upload_file(context, filename, content_type, raw)
    return {
        "file_id": uploaded["file_id"],
        "sha256": uploaded["sha256"],
        "size_bytes": uploaded["size_bytes"],
        "rows": len(rows),
        "metrics": metric_names,
        "scope_fingerprint": fingerprint,
        "query_version": QUERY_VERSION,
        "source_tables": sorted(sources),
        "expires_at": uploaded["expires_at"],
        "data_contract": uploaded["data_contract"],
    }
