"""P2 异步冒烟：真实报告 → generating 占位 → 后台线程 → finalize（含观测字段）。

用真实 DB + 真实 LLM 验证异步主链路不阻塞 POST、后台正确收尾。
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from app.analysis.contracts import InsightAngle, InsightBrief
from app.analysis.insights import submit_insight_task
from app.analysis.publishing import ReportStore
from app.analysis.reports import build_periodic_report_artifact


def main() -> int:
    """异步冒烟：真实报告 → generating 占位 → 后台线程 → finalize（含观测字段）；失败返回非零退出码。"""
    report = build_periodic_report_artifact("本周经营周报")
    if report is None:
        print("ASYNC FAILED: 无法构建真实周报")
        return 1
    tmp = Path(tempfile.mkdtemp(prefix="async-smoke-"))
    store = ReportStore(str(tmp / "phase6.sqlite"))
    store.create_draft(report, "# 草稿", "smoke")
    brief = InsightBrief(angle=InsightAngle.OVERVIEW, max_insights=2, min_words=20, max_words=140, focus_metrics=["gmv"])
    placeholder = store.create_generating(report.report_id, 1, brief, "smoke")
    iv = int(placeholder["insight_version"])
    print(f"[1] POST 秒回占位: insight_version={iv} status={placeholder['status']}")
    submit_insight_task(store, report.report_id, 1, iv, brief, "smoke", {"tenant_id": "", "user_id": ""})
    deadline = time.time() + 90
    row = placeholder
    while time.time() < deadline:
        time.sleep(2)
        row = store.get_insight(report.report_id, 1, iv)
        if row["status"] != "generating":
            break
    print(f"[2] 后台完成: status={row['status']} critic_verdict={row.get('critic_verdict')}")
    print(f"    duration_ms={row.get('duration_ms')} input_tokens={row.get('input_tokens')} "
          f"output_tokens={row.get('output_tokens')} cost={row.get('estimated_cost')}")
    print(f"    洞察条数={len(row.get('insights_json') or [])}")
    if row["status"] == "generating":
        print("ASYNC FAILED: 后台任务超时未完成")
        return 1
    print("ASYNC PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
