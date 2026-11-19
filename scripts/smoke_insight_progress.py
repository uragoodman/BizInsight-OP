"""SSE 数据源冒烟：后台任务逐阶段写 progress（attribution → llm_generation → critic → 终态）。"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from app.analysis.contracts import InsightAngle, InsightBrief
from app.analysis.insights import submit_insight_task
from app.analysis.publishing import ReportStore
from app.analysis.reports import build_periodic_report_artifact


def main() -> int:
    """SSE 数据源冒烟：验证后台任务逐阶段写 progress（attribution → llm_generation → critic → 终态）。"""
    report = build_periodic_report_artifact("本周经营周报")
    if report is None:
        print("PROGRESS FAILED: 无法构建真实周报")
        return 1
    store = ReportStore(str(Path(tempfile.mkdtemp(prefix="progress-smoke-")) / "p.sqlite"))
    store.create_draft(report, "# 草稿", "smoke")
    brief = InsightBrief(angle=InsightAngle.CONTRIBUTION, max_insights=2, min_words=20, max_words=140, focus_metrics=["gmv"])
    placeholder = store.create_generating(report.report_id, 1, brief, "smoke")
    iv = int(placeholder["insight_version"])
    submit_insight_task(store, report.report_id, 1, iv, brief, "smoke", {"tenant_id": "", "user_id": ""})
    seen: list[str] = []
    deadline = time.time() + 90
    while time.time() < deadline:
        row = store.get_insight(report.report_id, 1, iv)
        progress = row.get("progress") or {}
        stage = progress.get("stage")
        if stage and (not seen or seen[-1] != stage):
            seen.append(stage)
            print(f"    stage: {stage}  ({progress.get('message', '')})")
        if row["status"] != "generating":
            print(f"final status: {row['status']}  stages_seen={seen}")
            break
        time.sleep(1)
    if row["status"] == "generating":
        print("PROGRESS FAILED: 超时")
        return 1
    if not seen or seen[-1] not in {"llm_generation", "critic"}:
        print("PROGRESS FAILED: 阶段进度未完整记录:", seen)
        return 1
    print("PROGRESS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
