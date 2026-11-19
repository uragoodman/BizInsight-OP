"""Versioned quality-evidence snapshots for the engineering dashboard."""
# ============================================================
# app.quality 包：质量证据快照
# 作用：本包把可审计的项目质量证据（Golden 评测、性能基准、单元测试、
#       Trace 摘要、LLM 用量）聚合为统一快照并持久化，供「质量仪表盘」展示。
# 对外入口：service.py 的 dashboard() 与 capture_snapshot()，由
# app/api/routes/quality.py 调用，且要求 manage_usage 权限。
# ============================================================

